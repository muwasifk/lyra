<div align="center">
  <h1>LYRA</h1>
  <h2>Lets You Run Anything</h2>
</div>

https://github.com/user-attachments/assets/71649956-1f7e-4edd-bf66-b01e4ed2d13b

Run a Hugging Face language model that's too big for one machine by splitting its layers across several of them: a gaming PC, a MacBook, a friend's laptop, all over [Tailscale](https://tailscale.com). Each machine runs one block of the model's layers. A token travels around the machines like a relay baton, and the machine you're typing on samples the next token and streams the reply into a terminal chat app.

- **Pipeline parallelism over a normal network.** Layers 0–20 can live on an NVIDIA GPU and layers 21–39 on an Apple Silicon Mac. Per token, only one 8 KB hidden state crosses each network hop.
- **CUDA, Apple GPUs (MPS) and CPU,** mixed freely in one ring.
- **Each machine downloads only its own layers,** straight from the Hugging Face Hub, using byte ranges of the checkpoint files.
- **Auto split** measures every machine's free memory, its speed and its network round trip, then chooses the fastest split.
- **Stock Hugging Face models.** Lyra doesn't reimplement any model: every stage runs transformers' own code, with the parts it doesn't own swapped out. With sampling off, the output matches single-machine `generate()` token for token.

## Results

Two 8-billion-parameter models, each split across an **RTX 5070 (12 GB)** in a Linux desktop and a **MacBook Pro with an M3 Pro (18 GB)**, connected over Tailscale. Neither machine can hold either model in bf16 by itself. Auto split chose every split below.

| Model | Size (bf16) | Auto split: 5070 / Mac | Predicted | Measured | Prefill |
|---|---|---|---|---|---|
| [IBM Granite 4.2 8B](https://huggingface.co/ibm-granite/granite-4.2-8b) | 17.6 GB | 21 / 19 layers | 9.2 tok/s | **9.6 tok/s** | 0.26 s |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 16.4 GB | 19 / 17 layers | 10.0 tok/s | **9.6 tok/s** | 0.22 s |

**How these were measured.** The prompt was *"In one paragraph, compare matmul in PyTorch vs JAX."*, with thinking off and each model's own sampling settings. After one warm-up message, each model answered three times; the table shows the median. Granite's three runs measured 9.63–9.65 tok/s (190–270 tokens each). Qwen3-8B's measured 9.98, 9.56 and 7.35 tok/s (140–175 tokens). *Measured* is decode speed from Lyra's own stopwatch, network time included: tokens after the first, divided by the time after the first token. *Prefill* is the time to the first token, for a 27–30 token prompt.

Auto split's log for Granite shows how it chose. The 5070 is about 4.7× faster per layer, so it gets every layer it has room for:

```
ibm-granite/granite-4.2-8b: 40 layers (1 kind, 15.94 GB), plus 1.64 GB of embedding and head
measuring fedora (this machine)…
  cuda: 11.0 GB free (8.4 GB of it for layers), 0.84 ms/layer
measuring macbook:7071…
  mps: 12.9 GB free (11.9 GB of it for layers), 3.93 ms/layer, RAM available 5.5 GB, round trip 15.81 ms
split: fedora 21, macbook 19
predicted ≈ 9.2 tok/s (92 ms of layers + ~15.8 ms of network per token, not counting the head)
```

<details>
<summary>The answers (first run of each)</summary>

**Granite 4.2 8B**

> In PyTorch, matrix multiplication (matmul) is executed eagerly by default, meaning it runs immediately on the device (CPU or GPU) and produces immediate outputs, with automatic differentiation supported seamlessly through the torch.autograd system; gradients are tracked automatically during forward passes, and the operation integrates naturally with tensors that require gradients. In JAX, matmul (via `jax.numpy.dot` or `jax.scipy.linalg.multi_dot`) is executed in a lazy, functional style within a just-in-time (JIT) compiled context by default, enabling high performance through ahead-of-time compilation and fusion; automatic differentiation is built-in and composes naturally with the functional transformation model, but tensors are immutable and do not require gradients explicitly—instead, `jax.grad` or `jax.value_and_grad` are used to compute gradients analytically without tracking state during forward execution. Thus, PyTorch offers a more imperative, user-friendly interface with on-the-fly gradient tracking, while JAX emphasizes explicit transformations, JIT compilation, and functional purity, leading to often higher performance for batched or repeated computations but requiring a shift in mindset for gradient handling.
>
> (Word count: 150 words, concise, accurate, single paragraph as requested.)

**Qwen3-8B**

> In PyTorch and JAX, matrix multiplication (`matmul`) serves a similar purpose of performing linear algebra operations on tensors, but they differ in implementation and design philosophy. PyTorch, being primarily designed for deep learning, provides a straightforward and user-friendly API for `matmul` within its tensor operations, often used in conjunction with automatic differentiation for model training. JAX, on the other hand, emphasizes numerical computing and functional programming, offering a more flexible and composable approach to `matmul` through its JIT compilation and array manipulation capabilities, which also support automatic differentiation and vectorization. While both frameworks support efficient matrix operations, JAX tends to offer more control and performance optimizations, especially when leveraging hardware acceleration like GPUs or TPUs.

</details>

**Exactness.** With sampling off, these models give exactly the same output through a Lyra ring as `transformers`' `generate()` on one machine: Qwen3-0.6B and the hybrid convolution/attention model LFM2-350M (split two and three ways), plus Granite 4.2 3B and the hybrid linear-attention model Qwen3.5-4B (split two ways, on CPUs). The 8B models above are too big for that comparison, which is the point.

## Requirements

- **[uv](https://docs.astral.sh/uv/).** It installs the right Python (3.12) and the pinned dependencies (torch 2.14.1, transformers 5.20.2).
- **A GPU on each machine,** or a CPU, which is much slower:
  - **NVIDIA:** a recent driver that supports CUDA 13; the pinned PyTorch wheels are built for it.
  - **Apple Silicon:** macOS 14 or later, for bfloat16 on the GPU.
- **A network where every machine can reach every other one** on TCP port 7071 (or whichever you choose). Tailscale is the easy way. With Tailscale, add everyone's machines to the tailnet as members: a machine that is only *shared* into a tailnet can accept connections but can't open them, and the ring needs both.

Tested on Linux (Fedora, RTX 5070) and macOS (M3 Pro). Windows hasn't been tried.

## Quick start

On every machine:

```bash
git clone https://github.com/muwasifk/lyra
cd lyra
uv sync
```

On each machine that should **lend its GPU** (a worker):

```bash
uv run lyra worker
```

It prints `listening on :7071` and waits. Leave it running: it serves one session after another, and loads new layers by itself when the split changes.

On the machine you'll **chat from** (the driver):

```bash
uv run lyra
```

Then, in the terminal UI:

1. Type a model id in **model** (for example `ibm-granite/granite-4.2-8b`) and press Enter. Lyra reads its config and shows how many layers it has.
2. List the machines. The **driver** row is this machine. Add a **worker** row for each other machine, using its hostname or Tailscale name (suggestions come from `tailscale status`), with `:port` if it isn't 7071.
3. Press **auto split** to have Lyra measure the machines and fill in the layer counts, or type the counts yourself. They must add up to the model's layer count.
4. Press **connect.** Each machine downloads its layers the first time; after that they load from `~/.cache/lyra`.
5. Chat. The chat panel's border shows the prefill time and the tokens per second.

The **show thinking** switch controls models with a thinking mode, such as Qwen3, Qwen3.5 and Granite 4.2. Off, the model answers straight away, which is shorter and faster. On, Qwen3 shows its reasoning dimmed above the answer.

### Worker options

```bash
uv run lyra worker 7072              # listen on another port (then use host:7072 in the UI)
uv run lyra worker --memory 8        # offer at most 8 GB of GPU memory (unified memory on a Mac)
LYRA_DEVICE=cpu uv run lyra worker   # force a device: cuda, mps or cpu
```

`--memory` limits what auto split plans for this worker. On CUDA and Apple GPUs, PyTorch also refuses to allocate past it.

## How it works

```
 driver (rank 0)                         worker (rank 1)                  worker (rank 2)
 ┌──────────────────────────┐            ┌──────────────────┐            ┌──────────────────┐
 │ tokenizer → embed        │  hidden    │                  │  hidden    │                  │
 │ layers 0 – 11            │ ─────────▶ │ layers 12 – 23   │ ─────────▶ │ layers 24 – 35   │
 │ head: norm → lm_head     │            │ own KV cache     │            │ own KV cache     │
 │ sample the next token    │ ◀───────────────────────────────────────── │                  │
 └──────────────────────────┘                     hidden                 └──────────────────┘
```

- **The ring.** The driver sends a setup message carrying the plan (which host runs which layers) to the first worker. Each worker reads its own entry, connects to the next machine, passes the message on, and then loads its layers, so every machine loads at the same time. The last worker connects back to the driver, which closes the ring. Every token is one trip around it.
- **Stages without rewriting the model.** Every stage builds the stock transformers model on PyTorch's `meta` device (shapes, no memory), then replaces each part it doesn't own with a pass-through layer. Hugging Face's own forward pass then runs unchanged on what's left. The final norm is moved out to run once, on the driver, just before `lm_head`.
- **Partial downloads.** What's left of the model lists exactly the tensors this stage needs. Lyra reads the `.safetensors` headers, downloads just those byte ranges in chunks of at most 1 GB, and caches them per stage in `~/.cache/lyra`.
- **Caches stay local.** Each stage keeps the KV cache for its own layers, so after the first step only one token's hidden state crosses the network. Positions are sent with every message, because a stage that doesn't own layer 0 can't work them out itself.
- **Auto split.** For one conversation, time per token is the sum of every layer's time on the machine that runs it, plus one network hop per stage. So the fastest split isn't balanced: the fastest machine gets as many layers as fit. Lyra measures each machine (free memory, the speed of one layer of each kind, its round trip), tries every set of workers, and uses a small dynamic program to give each machine the best unbroken block of layers that fits. It also handles models whose layers differ in size.
- **Sampling** uses the model's own `generation_config.json` (temperature, top-k, top-p) and stops at its end-of-sequence tokens. Models that don't ship that file (Qwen3.5, for example) decode greedily.



## Security

> [!WARNING]
> **Run workers only on a network you trust, such as your own tailnet.**
>
> A worker listens on every network interface and has no authentication. Anyone who can reach its port can make it download a model from the Hub and run it on its GPU. They can't run their own code on it: Lyra never enables `trust_remote_code`. But they can use up its disk, memory and GPU time, and a malformed message can make it allocate a lot of memory.

## Development

```bash
uv sync                              # also installs the dev tools (ruff)
uv run ruff format src scripts       # format
uv run ruff check src scripts        # lint
uv run python scripts/tensor_ping.py serve          # on one machine
uv run python scripts/tensor_ping.py send <host>    # on another: checks tensors arrive intact, times round trips
```

To check that a ring is exact, turn sampling off on the driver (`driver.gen.do_sample = False`) and compare a reply with `model.generate(..., do_sample=False)` on a single machine. They should match token for token.

## License

[MIT](LICENSE). Models have their own licenses.
