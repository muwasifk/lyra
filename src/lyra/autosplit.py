"""
Auto split: measure every device, then choose how many layers each one runs.

For one conversation, every token goes around the ring one stage at a time, so the time per token is a sum:

    time per token ≈ Σ (time of each layer on the device that runs it)  +  the hops around the ring

Because it's a sum, the fastest split isn't a balanced one: it puts as many layers as fit on the fastest device.
But every worker in the ring adds a hop, so a worker is only worth using if it saves more time than its hop costs.
So we try every set of workers (with a handful of devices that's at most a few dozen), find the fastest way to
split the layers between them, predict its time per token, and keep the best.

Layers aren't always the same size: hybrid models mix two kinds of layer, and some mixture-of-experts models
mix dense and expert layers. Each device runs one unbroken block of layers, in ring order, so which layers a
device gets matters, not just how many. `assign` works that out.

What we measure:
    free memory     how many bytes of layers fit, after a reserve for the KV cache and activations
    ms per layer    one layer of each kind in this model, with random weights, timed on a decode step
    round trip      a decode-sized hidden state sent to a worker and back

Each device's measurements end up in a "report", a plain dict (it goes over the wire as JSON):
    {"device": "cuda", "free_bytes": ..., "ms_per_layer": [one per layer], "offered_bytes": ..., "rtt_ms": ...,
     "is_driver": ...}
"""

import gc
import itertools
import statistics
import time
from typing import NamedTuple

import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

from .stage import build_stage, forward
from .wire import connect, recv_msg, send_msg

RESERVE = 1_000_000_000  # bytes kept free on every device for the KV cache, activations and the like
PROMPT_TOKENS = 128  # the layer benchmark first runs a prompt this long, so decode steps have something to attend to
WARMUP_STEPS = 3  # decode steps run before timing starts (the first few are always slow)
DECODE_STEPS = 20  # decode steps timed per layer benchmark
PINGS = 20  # round trips timed per worker
NOISE = 0.10  # timings wobble by several percent: each extra worker must look at least this much faster


class ModelSize(NamedTuple):
    layer_bytes: list  # the size of each decoder layer, in order
    layer_kinds: list  # for each layer, the index of the first layer built exactly like it (same kind, same speed)
    extra_bytes: int  # everything that isn't a layer: embedding, final norm, lm_head. Only the driver holds it.

    @property
    def num_layers(self):
        return len(self.layer_bytes)


# ----- measuring one device -----


def free_memory(device):
    """Bytes this device can still give us."""
    if device == "cuda":
        free_bytes = torch.cuda.mem_get_info()[0]  # mem_get_info() gives (free, total) in bytes
        return free_bytes
    if device == "mps":
        # Apple GPUs share RAM with the CPU. Use the GPU's own limit, not psutil's "available" RAM: on macOS that
        # leaves out memory the OS would happily compress or page out to make room, so it's far too low.
        return torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
    return psutil.virtual_memory().available


def cap_memory(device, offered_bytes):
    """Make PyTorch refuse to use more than offered_bytes here: an allocation past it fails with out-of-memory.
    PyTorch can only enforce this on GPUs; on the CPU the offer only limits what auto split plans for."""
    if device == "cuda":
        total_bytes = torch.cuda.mem_get_info()[1]  # mem_get_info() gives (free, total) in bytes
        torch.cuda.set_per_process_memory_fraction(min(1.0, offered_bytes / total_bytes))
    elif device == "mps":
        # On Apple GPUs the fraction is of the GPU's recommended limit, and may go up to 2.
        torch.mps.set_per_process_memory_fraction(min(2.0, offered_bytes / torch.mps.recommended_max_memory()))


def synchronize(device):
    """Wait for queued GPU work to finish. GPU calls return before the work is done, so timing needs this."""
    if device != "cpu":
        torch.accelerator.synchronize()


def model_sizes(name):
    """How big the model `name` is, layer by layer, worked out from its config alone: nothing is downloaded."""
    config = AutoConfig.from_pretrained(name)
    with torch.device("meta"):  # shapes only: nothing is allocated
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    base = getattr(model, model.base_model_prefix)

    layer_bytes = []
    layer_kinds = []
    first_of_kind = {}  # a layer's "recipe" (its class and every weight's name and shape) -> first layer like it
    for i, layer in enumerate(base.layers):
        size = 0
        recipe = [type(layer).__name__]
        for param_name, param in layer.named_parameters():
            size += param.nbytes
            recipe.append((param_name, tuple(param.shape)))
        layer_bytes.append(size)
        recipe = tuple(recipe)
        if recipe not in first_of_kind:
            first_of_kind[recipe] = i
        layer_kinds.append(first_of_kind[recipe])

    total_bytes = 0
    for param in model.parameters():  # a tied embedding/lm_head is one parameter, so it counts once
        total_bytes += param.nbytes
    return ModelSize(layer_bytes, layer_kinds, total_bytes - sum(layer_bytes))


def time_layer(name, device, index):
    """Milliseconds for layer `index` of `name` to process one token on `device`."""
    # Build just that layer on "meta", then give it real memory. That memory holds garbage, so fill it with random
    # weights: the speed doesn't depend on the values.
    model, base = build_stage(name, index, 1, first=False)
    base.to_empty(device=device)
    with torch.no_grad():
        for param in base.parameters():
            param.normal_(0, 0.02)
    for module in base.modules():
        if hasattr(module, "inv_freq"):
            model._init_weights(module)  # rotary frequencies, as in load_stage

    hidden_size = model.config.hidden_size
    cache = DynamicCache(config=base.config)
    prompt = torch.randn(1, PROMPT_TOKENS, hidden_size, dtype=torch.bfloat16, device=device)
    forward(base, prompt, 0, cache)

    times_ms = []
    for step in range(WARMUP_STEPS + DECODE_STEPS):
        token = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16, device=device)
        synchronize(device)
        start_time = time.perf_counter()
        forward(base, token, PROMPT_TOKENS + step, cache)
        synchronize(device)
        if step >= WARMUP_STEPS:
            times_ms.append((time.perf_counter() - start_time) * 1000)

    # Free the test layer before anyone loads real weights.
    del model, base, cache
    gc.collect()
    if device != "cpu":
        torch.accelerator.empty_cache()
    return statistics.median(times_ms)


def time_layers(name, device):
    """Milliseconds per token for every layer of `name` on `device`. Times one layer of each kind."""
    size = model_sizes(name)
    ms_of_kind = {}
    for kind in size.layer_kinds:
        if kind not in ms_of_kind:
            ms_of_kind[kind] = time_layer(name, device, kind)

    ms_per_layer = []
    for kind in size.layer_kinds:
        ms_per_layer.append(ms_of_kind[kind])
    return ms_per_layer


def measure(name, device, held_bytes=0, offered_bytes=None):
    """
    This device's report: free memory and ms per layer.

    held_bytes     layers we hold now; we'd free them before loading new ones, so they count as free
    offered_bytes  the most this device's owner lets us use (`lyra worker --memory`), or None for no limit
    """
    free_bytes = free_memory(device) + held_bytes
    if offered_bytes is not None:
        free_bytes = min(free_bytes, offered_bytes)
    report = {
        "device": device,
        "free_bytes": free_bytes,
        "ms_per_layer": time_layers(name, device),
        "offered_bytes": offered_bytes,
    }
    if device == "mps":
        report["ram_available_bytes"] = psutil.virtual_memory().available  # just for the log, to spot swapping
    return report


# ----- measuring a worker over the network -----


def answer_probe(conn, name, device, held_bytes, offered_bytes):
    """Worker side: send our report, then echo pings until the driver hangs up, so it can time the round trip."""
    send_msg(conn, measure(name, device, held_bytes, offered_bytes))
    try:
        while True:
            message, tensor = recv_msg(conn, device)
            send_msg(conn, message, tensor)
    except ConnectionError:
        pass  # the driver has what it needs


def probe(host, port, name, device, hidden_size):
    """Driver side: ask the worker at host:port for its report, and add the round-trip time to it."""
    sock = connect(host, port)
    send_msg(sock, {"type": "probe", "model": name})
    report, _ = recv_msg(sock, device)

    hidden = torch.zeros(1, 1, hidden_size, dtype=torch.bfloat16, device=device)  # what one decode hop carries
    times_ms = []
    for _ in range(PINGS):
        start_time = time.perf_counter()
        send_msg(sock, {"type": "ping"}, hidden)
        recv_msg(sock, device)
        times_ms.append((time.perf_counter() - start_time) * 1000)
    sock.close()

    report["rtt_ms"] = statistics.median(times_ms)
    return report


# ----- choosing the split -----


def room_bytes(report, size):
    """Bytes of layers this device can hold, after the reserve (and on the driver, the embedding and head)."""
    room = report["free_bytes"] - RESERVE
    if report["is_driver"]:
        room -= size.extra_bytes
    return max(0, room)


def assign(reports, chosen, size):
    """
    Split the layers between the chosen devices (indexes into reports) as fast as possible.

    Each device runs one unbroken block of layers, in ring order: the driver gets the first block, then each chosen
    worker in turn. A device may get an empty block. Returns a layer count for every device (0 for those not
    chosen), or None if the model doesn't fit on them.

    Working backwards from the last device, fastest[j] is the least time it takes
    the devices from here on to run layers j, j+1, ..., the last one. A device that starts its block at layer j
    and stops before layer k takes the time of layers j..k-1, plus fastest[k] for everyone after it.
    """
    num_layers = size.num_layers
    ring = sorted(chosen)  # ring order: the driver (index 0) first, then the workers in the order they were given
    never = float("inf")  # "this can't be done"

    # Before looking at any device: running no layers takes no time; running any layers is impossible.
    fastest = [never] * (num_layers + 1)
    fastest[num_layers] = 0.0
    block_end = []  # for each device in the ring: block_end[d][j] = where its block stops, if it starts at j

    for i in reversed(ring):
        report = reports[i]
        room = room_bytes(report, size)
        fastest_from_here = [never] * (num_layers + 1)
        best_end = [None] * (num_layers + 1)
        for start in range(num_layers + 1):
            block_bytes = 0
            block_ms = 0.0
            for end in range(start, num_layers + 1):  # this device runs layers start .. end - 1
                if end > start:
                    block_bytes += size.layer_bytes[end - 1]
                    block_ms += report["ms_per_layer"][end - 1]
                    if block_bytes > room:
                        break  # a longer block won't fit either
                total_ms = block_ms + fastest[end]
                if total_ms < fastest_from_here[start]:
                    fastest_from_here[start] = total_ms
                    best_end[start] = end
        fastest = fastest_from_here
        block_end.insert(0, best_end)

    if fastest[0] == never:
        return None

    # Follow the best choices forwards to turn them into layer counts.
    counts = [0] * len(reports)
    start = 0
    for d, i in enumerate(ring):
        end = block_end[d][start]
        counts[i] = end - start
        start = end
    return counts


def predict_ms(reports, counts):
    """(ms of layers, ms of network) per token, if each device runs the given number of layers, in ring order."""
    compute_ms = 0.0
    worker_rtts = []  # round trips of the workers that are in the ring
    start = 0
    for report, count in zip(reports, counts):
        for layer in range(start, start + count):
            compute_ms += report["ms_per_layer"][layer]
        start += count
        if not report["is_driver"] and count > 0:
            worker_rtts.append(report["rtt_ms"])

    # A token makes one hop per stage: driver -> worker 1 -> ... -> last worker -> back to the driver.
    # Each hop costs about half a round trip. We only measured driver <-> worker, so use their average.
    network_ms = 0.0
    if worker_rtts:
        hops = len(worker_rtts) + 1
        network_ms = hops * statistics.mean(worker_rtts) / 2
    return compute_ms, network_ms


def choose_counts(reports, size):
    """The layer count for every device (0 = left out) with the lowest predicted time per token."""
    driver = reports[0]
    if driver["free_bytes"] - RESERVE < size.extra_bytes:
        raise RuntimeError(f"the driver doesn't have room for the embedding and head ({size.extra_bytes / 1e9:.1f} GB)")

    best_counts = None
    best_score = None
    workers = list(range(1, len(reports)))

    # Try every set of workers: none, each one alone, every pair, ..., all of them. The driver is always in.
    for how_many in range(len(workers) + 1):
        for chosen_workers in itertools.combinations(workers, how_many):
            counts = assign(reports, [0] + list(chosen_workers), size)
            if counts is None:
                continue  # the model doesn't fit on just these

            compute_ms, network_ms = predict_ms(reports, counts)
            # Charge each worker a little extra, so one that only looks faster by measurement noise isn't used.
            score = (compute_ms + network_ms) * (1 + NOISE * how_many)
            if best_score is None or score < best_score:
                best_counts = counts
                best_score = score

    if best_counts is None:
        total_room = 0
        for report in reports:
            total_room += room_bytes(report, size)
        raise RuntimeError(
            f"the model doesn't fit: its layers need {sum(size.layer_bytes) / 1e9:.1f} GB, and all {len(reports)} "
            f"devices together have room for {total_room / 1e9:.1f} GB (see the measurements above)"
        )
    return best_counts


# ----- putting it together -----


def describe(report, size):
    """One line for the log: what we measured on this device."""
    fastest_layer = min(report["ms_per_layer"])
    slowest_layer = max(report["ms_per_layer"])
    if slowest_layer - fastest_layer < 0.01:
        speed = f"{fastest_layer:.2f} ms/layer"
    else:
        speed = f"{fastest_layer:.2f}–{slowest_layer:.2f} ms/layer, depending on the kind of layer"
    line = (
        f"  {report['device']}: {report['free_bytes'] / 1e9:.1f} GB free "
        f"({room_bytes(report, size) / 1e9:.1f} GB of it for layers), {speed}"
    )
    if report["offered_bytes"] is not None:
        line += f" (its owner offers {report['offered_bytes'] / 1e9:.1f} GB)"
    if "ram_available_bytes" in report:
        line += f", RAM available {report['ram_available_bytes'] / 1e9:.1f} GB"
    if not report["is_driver"]:
        line += f", round trip {report['rtt_ms']:.2f} ms"
    return line


def auto_split(name, addresses, device):
    """
    Measure the driver (this machine) and every worker, choose the layer counts, and print the reasoning.

    addresses  [(host, port), ...] with the driver first, in ring order
    Returns the layer count for each address, in the same order. 0 for a worker means: leave it out.
    """
    size = model_sizes(name)
    hidden_size = AutoConfig.from_pretrained(name).get_text_config().hidden_size
    kinds = len(set(size.layer_kinds))
    print(
        f"{name}: {size.num_layers} layers ({kinds} kind{'s' if kinds > 1 else ''}, "
        f"{sum(size.layer_bytes) / 1e9:.2f} GB), plus {size.extra_bytes / 1e9:.2f} GB of embedding and head"
    )

    reports = []
    for i, (host, port) in enumerate(addresses):
        is_driver = i == 0
        if is_driver:
            print(f"measuring {host} (this machine)…")
            report = measure(name, device)
            report["rtt_ms"] = 0.0
        else:
            print(f"measuring {host}:{port}…")
            report = probe(host, port, name, device, hidden_size)
        report["is_driver"] = is_driver
        reports.append(report)
        print(describe(report, size))  # now, so the numbers are in the log even if the model turns out not to fit

    counts = choose_counts(reports, size)

    split = []
    for (host, port), count in zip(addresses, counts):
        split.append(f"{host} {count}")
    print("split: " + ", ".join(split))

    compute_ms, network_ms = predict_ms(reports, counts)
    print(
        f"predicted ≈ {1000 / (compute_ms + network_ms):.1f} tok/s "
        f"({compute_ms:.0f} ms of layers + ~{network_ms:.1f} ms of network per token, not counting the head)"
    )

    for (host, port), count in zip(addresses[1:], counts[1:]):
        if count == 0:
            print(f"  leaving out {host}:{port}: it would cost more in network time than it saves")
    return counts
