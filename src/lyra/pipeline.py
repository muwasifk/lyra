"""The ring: the driver (stage 0) sends hidden states to worker 1, which sends to worker 2, ..., back to the driver.

The first message on every connection to a worker says what the connection is for:
    {"type": "probe", "model"}                   auto split is measuring us (see autosplit.py)
    {"type": "setup", "model", "plan", "rank"}   join a ring

Every stage has two connections: `upstream`, where hidden states arrive from the previous stage, and
`downstream`, where it sends them on to the next one. The driver's upstream is the last worker: that's what closes
the ring.

Setup: each worker reads its own entry plan[rank], connects to the next stage, forwards the setup with rank + 1,
then loads its layers. When the setup message gets back to the driver, the ring is up.
"""

import gc
import socket
import time
from typing import NamedTuple

import torch
from transformers import AutoTokenizer, DynamicCache

from .autosplit import answer_probe, cap_memory
from .stage import forward, load_stage
from .wire import accept, connect, recv_msg, send_msg

PORT = 7071  # default port every stage listens on


class PlanEntry(NamedTuple):
    """One entry of the plan: which machine runs which layers. Goes over the wire as a plain JSON list."""

    host: str
    port: int
    start: int
    count: int

    def __str__(self):
        if self.count == 0:
            return f"{self.host}[no layers]"
        return f"{self.host}[{self.start}..{self.start + self.count - 1}]"


def parse_address(text):
    """ "localhost:7072" -> ("localhost", 7072); "fedora" -> ("fedora", PORT)"""
    host, _, port = text.partition(":")
    return host, int(port if port else PORT)


def parse_plan(args):
    """["fedora=11", "localhost:7072=17"] -> [PlanEntry("fedora", 7071, 0, 11), PlanEntry("localhost", 7072, 11, 17)]"""
    plan, layer_idx = [], 0

    for a in args:
        addr, count = a.split("=")
        host, port = parse_address(addr)

        plan.append(PlanEntry(host, port, layer_idx, int(count)))
        layer_idx += int(count)

    return plan


def join_ring(upstream, setup):
    """Pass a setup message we received on `upstream` on to the next stage.
    Returns (downstream, model name, our PlanEntry)."""
    # JSON has no tuples, so each plan entry arrives as a plain list: [host, port, start, count]
    plan = []
    for host, port, start, count in setup["plan"]:
        plan.append(PlanEntry(host, port, start, count))

    rank = setup["rank"]  # our position in the ring; the driver is 0
    my_plan_entry = plan[rank]
    if rank == len(plan) - 1:
        next_stage = plan[0]  # we're the last worker: send back to the driver, closing the ring
    else:
        next_stage = plan[rank + 1]
    downstream = connect(next_stage.host, next_stage.port)

    # Pass the setup message on before loading our layers, so every stage loads at the same time.
    setup["rank"] = rank + 1
    send_msg(downstream, setup)
    return downstream, setup["model"], my_plan_entry


def run_stage(base, upstream, downstream, device):
    """Run our layers on every hidden state that arrives from upstream and send the result downstream. Never returns:
    raises ConnectionError when the ring closes."""
    cache = None
    while True:
        # message is {"pos": where these tokens start in the sequence}
        message, hidden = recv_msg(upstream, device)
        pos = message["pos"]
        if pos == 0:
            cache = DynamicCache(config=base.config)  # a new prompt: throw away the old one's keys and values

        hidden = forward(base, hidden, pos, cache)
        send_msg(downstream, message, hidden)


def serve(port, device, offered_bytes=None):
    """
    Worker: answer probes and join rings, forever. Our layers stay loaded until a ring wants different ones.

    offered_bytes  the most memory this worker may use, or None for whatever is free
    """
    if offered_bytes is not None:
        cap_memory(device, offered_bytes)
        print(f"offering at most {offered_bytes / 1e9:.1f} GB of {device} memory")
    srv = socket.create_server(("", port))
    base = None  # the layers we have loaded
    loaded_job = None  # which layers those are: (model, start, count)

    while True:
        print(f"listening on :{port}")
        upstream = accept(srv)
        downstream = None
        try:
            message, _ = recv_msg(upstream, device)

            if message["type"] == "probe":
                print(f"measuring this device for {upstream.getpeername()[0]}")
                held_bytes = 0  # the layers we'd free before loading new ones count as free memory
                if base is not None:
                    held_bytes = sum(param.nbytes for param in base.parameters())
                answer_probe(upstream, message["model"], device, held_bytes, offered_bytes)
                continue

            downstream, model, me = join_ring(upstream, message)
            job = (model, me.start, me.count)
            if job != loaded_job:  # auto split may have moved the cut since last time
                # Free the old layers first, or the old and new ones would both be in memory at once.
                base = None
                gc.collect()
                torch.accelerator.empty_cache()
                base, _ = load_stage(
                    model, me.start, me.count, first=False, device=device
                )  # workers don't need the model
                loaded_job = job

            print(f"running {me} on {device}")
            run_stage(base, upstream, downstream, device)
        except ConnectionError:
            if downstream is not None:
                print("ring closed")  # the driver left, so the ring falls apart one hop at a time
            else:
                print("connection dropped before it said what it wanted")
        finally:
            upstream.close()
            if downstream is not None:
                downstream.close()


class Driver:
    """Driver (stage 0): sets up the ring once, then generates as many replies as you like."""

    def __init__(self, name, plan, device):
        self.model_name = name  # the model's Hugging Face id, e.g. "Qwen/Qwen3-8B"
        self.plan = plan
        self.device = device
        my_plan_entry = plan[0]  # the driver is always the first entry

        if len(plan) == 1:
            # No workers (auto split found that everything fits here): no ring, just run the whole model.
            self.upstream = self.downstream = None
            self.base, self.model = load_stage(
                name, my_plan_entry.start, my_plan_entry.count, first=True, device=device
            )
            print(f"running all {my_plan_entry.count} layers on this machine")
        else:
            first_worker = plan[1]

            # Listen before anything else: the last worker will connect back to us here.
            srv = socket.create_server(("", my_plan_entry.port))

            # Send the setup message to the first worker. It travels around the ring while we load our own layers.
            self.downstream = connect(first_worker.host, first_worker.port)
            send_msg(self.downstream, {"type": "setup", "model": name, "plan": plan, "rank": 1})
            self.base, self.model = load_stage(
                name, my_plan_entry.start, my_plan_entry.count, first=True, device=device
            )

            # The last worker connects back to us and forwards the setup message one final time: the ring is up.
            self.upstream = accept(srv)
            srv.close()  # nobody else needs to connect
            recv_msg(self.upstream, device)
            print("ring up: " + " -> ".join(str(entry) for entry in plan))

        self.tok = AutoTokenizer.from_pretrained(name)
        self.gen = self.model.generation_config  # the model's own sampling settings (generation_config.json)
        if isinstance(self.gen.eos_token_id, int):  # configs give either one end token or a list of them
            self.eos = [self.gen.eos_token_id]
        else:
            self.eos = self.gen.eos_token_id
        self.stop_requested = False  # another thread (the TUI) can set this to end a reply early
        self.last_stats = None  # timings of the latest reply

    def close(self):
        """Leave the ring: the workers see it close and go back to listening."""
        if self.upstream is not None:
            self.upstream.close()
            self.downstream.close()

    def generate(self, messages, on_token, think=True):
        """Stream the reply to `messages` (a chat history) and return it as text.

        Each new piece of text goes to on_token(text), and the timings end up in self.last_stats.
        think=False asks models with a thinking mode (like Qwen3) to answer straight away.
        """
        self.stop_requested = False
        chat = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=think, return_tensors="pt", return_dict=True
        )
        ids = chat["input_ids"].to(self.device)  # the whole sequence so far: the prompt, then each new token
        prompt_len = ids.shape[1]
        # private, but it's exactly what generate() builds from generation_config: temperature, top-k/p, penalties...
        processors = self.model._get_logits_processor(self.gen, input_ids_seq_length=prompt_len, device=self.device)

        cache = DynamicCache(config=self.base.config)
        pos = 0  # where the next tokens start in the sequence; 0 also tells every worker to start a fresh cache
        x = ids  # what goes into our layers: the whole prompt first, then one new token at a time
        new_tokens = 0
        shown = ""  # the part of the reply already handed to on_token
        start_time = time.perf_counter()

        while True:  # until the model produces an end-of-sequence token
            # Once around the ring: our layers, then every worker's layers, then back to us.
            hidden = forward(self.base, x, pos, cache)
            if self.downstream is not None:
                send_msg(self.downstream, {"pos": pos}, hidden)
                _, hidden = recv_msg(self.upstream, self.device)
            pos += x.shape[1]

            # The head turns the last position's hidden state into a score for every token in the vocabulary.
            with torch.inference_mode():
                logits = self.model.lm_head(hidden[:, -1]).float()
                scores = processors(ids, logits)  # the processors see the whole history (e.g. repetition penalty)
            if self.gen.do_sample:
                x = torch.multinomial(scores.softmax(-1), 1)
            else:
                x = scores.argmax(-1, keepdim=True)
            token = x.item()

            new_tokens += 1
            if new_tokens == 1:
                first_token_time = time.perf_counter()  # the first step is the prefill: the whole prompt at once
            if token in self.eos:
                break
            ids = torch.cat([ids, x], dim=-1)

            # Decode the whole reply so far, not just this token: one character (an emoji, a curly quote...) can
            # be split over several tokens, and decoding part of it gives "�". If the text ends in the middle of
            # a character, hold it back until the rest arrives. on_token still gets called, so it can count tokens.
            text = self.tok.decode(ids[0, prompt_len:])
            if text.endswith("\ufffd"):
                on_token("")
            else:
                on_token(text[len(shown) :])
                shown = text
            if self.stop_requested:
                break
        end_time = time.perf_counter()

        # A stopwatch: everything counts, the network included. After the prefill, each step makes one token.
        decode_tokens = new_tokens - 1
        decode_time = end_time - first_token_time
        self.last_stats = {
            "prompt_tokens": prompt_len,
            "prefill_ms": (first_token_time - start_time) * 1000,
            "decode_tokens": decode_tokens,
            "decode_s": decode_time,
            "tok_per_s": decode_tokens / decode_time if decode_tokens > 0 else 0.0,
        }
        return self.tok.decode(ids[0, prompt_len:], skip_special_tokens=True)
