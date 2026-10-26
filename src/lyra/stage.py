"""
One pipeline stage: a slice of an HF causal LM's layers, with only that slice's weights downloaded.

An HF causal LM (e.g. Qwen3ForCausalLM) is two parts:

    model                 the whole thing
      base                model.model: everything but the head
        embed_tokens      token ids -> hidden states
        layers[0..N-1]    the decoder layers: hidden states -> hidden states
        norm              final normalization
      lm_head             hidden states -> a score for every token in the vocabulary

A stage keeps the layers it owns and swaps everything else for PipelineIdentityLayer, which passes its input
straight through. HF's own forward then runs unchanged, and the skipped parts simply do nothing:

    first stage (driver)  embed_tokens + its layers, and the head (norm, lm_head) to finish each token
    every other stage     just its layers: hidden states in, hidden states out
"""

import math
import shutil
import struct
from pathlib import Path

import torch
from huggingface_hub import get_safetensors_metadata, get_session, hf_hub_url
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.distributed.pipeline_parallel import PipelineIdentityLayer

CACHE = Path.home() / ".cache" / "lyra"  # each stage's downloaded weights, keyed by model and layer range
CHUNK = 1 << 30  # download (and cache) in pieces of at most 1 GB, so RAM never holds a whole stage twice
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}  # safetensors dtype names


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_range(http, url, start, end):
    """Bytes [start, end) of the file at url. An HTTP range includes its last byte, hence end - 1."""
    headers = {"Range": f"bytes={start}-{end - 1}"}
    return http.get(url, headers=headers, follow_redirects=True).content


def checkpoint_name(key, weight_map):
    """
    The name the checkpoint uses for the tensor the model calls `key`. Usually it's the same name. But
    vision-language checkpoints (such as Qwen3.5) keep the text model one level down, under model.language_model,
    while the text-only model we build calls the same tensor model.layers... directly.
    """
    if key in weight_map:
        return key
    nested = key.replace("model.", "model.language_model.", 1)
    if nested in weight_map:
        return nested
    raise KeyError(f"the checkpoint has no tensor called {key} (or {nested})")


def fetch(name, keys):
    """Download just these tensors from the hub's .safetensors files with HTTP range requests, in ~1 GB chunks.

    A .safetensors file is [8-byte header length N][N bytes of JSON: name -> dtype, shape, data_offsets][raw bytes],
    and get_safetensors_metadata has already read every header for us. `keys` are the model's names for the
    tensors; each chunk is a dict from those names to tensors, whatever the checkpoint calls them.
    """
    meta = get_safetensors_metadata(name)
    http = get_session()
    source = {}  # the model's name for a tensor -> the checkpoint's name for it
    for key in keys:
        source[key] = checkpoint_name(key, meta.weight_map)

    # Big checkpoints are split across several files ("shards"). Go through the ones we need, one at a time.
    files = sorted({meta.weight_map[source[key]] for key in keys})
    for file in files:
        url = hf_hub_url(name, file)
        tensors = meta.files_metadata[file].tensors  # tensor name -> dtype, shape, data_offsets

        # The first 8 bytes say how long the JSON header is. The tensor data starts right after it,
        # and data_offsets count from there.
        header_len = struct.unpack("<Q", download_range(http, url, 0, 8))[0]
        data_start = 8 + header_len

        # Group tensors that sit back to back in the file, so a whole layer comes down in one request.
        wanted = [key for key in keys if meta.weight_map[source[key]] == file]
        wanted.sort(key=lambda key: tensors[source[key]].data_offsets[0])
        groups = []  # each group: {"start": byte, "end": byte, "keys": [tensor names]}
        for key in wanted:
            tensor_start, tensor_end = tensors[source[key]].data_offsets
            if groups and groups[-1]["end"] == tensor_start and tensor_end - groups[-1]["start"] <= CHUNK:
                groups[-1]["end"] = tensor_end
                groups[-1]["keys"].append(key)
            else:
                groups.append({"start": tensor_start, "end": tensor_end, "keys": [key]})

        for group in groups:
            data = bytearray(download_range(http, url, data_start + group["start"], data_start + group["end"]))
            chunk = {}
            for key in group["keys"]:
                info = tensors[source[key]]
                offset = info.data_offsets[0] - group["start"]  # where this tensor starts inside `data`
                flat = torch.frombuffer(data, dtype=DTYPES[info.dtype], offset=offset, count=math.prod(info.shape))
                chunk[key] = flat.reshape(info.shape)
            yield chunk


def final_norm_name(base):
    """
    The name of the final norm inside the base model. It's "norm" in most models, but not all: LFM2 calls it
    "embedding_norm". It's the one direct child of the base model that is a norm layer (e.g. Qwen3RMSNorm).
    """
    names = []
    for child_name, child in base.named_children():
        if "norm" in type(child).__name__.lower():
            names.append(child_name)
    if len(names) != 1:
        raise RuntimeError(f"can't tell which part of {type(base).__name__} is the final norm: {names}")
    return names[0]


def build_stage(name, start, count, first):
    """The model's architecture with no weights yet, cut down to this stage's part. Returns (model, base)."""
    config = AutoConfig.from_pretrained(name)
    with torch.device("meta"):  # "meta" tensors have a shape but no memory: nothing is allocated or downloaded
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    base = getattr(model, model.base_model_prefix)  # everything but lm_head: embed_tokens, layers, the final norm

    # mask out layers that a specific device is not handling using identity layers.
    for i in range(len(base.layers)):
        if i < start or i >= start + count:
            base.layers[i] = PipelineIdentityLayer()

    if not first:
        # only the driver does embedding, norm, and lm_head computation
        base.embed_tokens = PipelineIdentityLayer()
        setattr(base, final_norm_name(base), PipelineIdentityLayer())  # i.e. base.norm = ..., whatever it's called
        model.lm_head = PipelineIdentityLayer()

    return model, base


def load_weights(name, keys, folder, device):
    """This stage's weights, from the cache folder. If it isn't cached yet, download it there first."""
    if not folder.exists():
        partial = folder.with_suffix(".part")
        shutil.rmtree(partial, ignore_errors=True)  # leftovers from a download that didn't finish
        partial.mkdir(parents=True)
        for i, chunk in enumerate(fetch(name, keys)):
            save_file(chunk, partial / f"{i}.safetensors")  # straight to disk: RAM only ever holds one chunk
        partial.rename(folder)  # only a finished download counts as cached

    weights = {}
    for file in folder.glob("*.safetensors"):
        for key, tensor in load_file(file).items():
            weights[key] = tensor.to(device)
    total_gb = sum(tensor.nbytes for tensor in weights.values()) / 1e9
    print(f"{total_gb:.2f} GB of weights from {folder}")
    return weights


def load_stage(name, start, count, first, device):
    """
    Build this stage's part of the model `name` on `device`, with real weights, ready to run.

    The stage owns layers start .. start + count - 1. With first=True it's the driver's stage, so it also gets
    embed_tokens and the head. Only this stage's weights are downloaded, and only the first time: after that
    they come from the cache in ~/.cache/lyra.

    Returns (base, model):
        base    runs this stage's layers; pass it to forward()
        model   the whole HF model object. On the first stage, model.lm_head is the head (final norm, then
                lm_head) and model.generation_config holds the model's sampling settings. Other stages ignore it.
    """
    masked_model, base = build_stage(name, start, count, first)

    # Whatever is left in the state dict is exactly what this stage needs.
    keys = set(masked_model.state_dict())

    # Some models reuse the embedding matrix as lm_head ("tied" weights). Download it once and use it for both.
    tied = {}
    if first and masked_model.config.tie_word_embeddings:
        tied = masked_model._tied_weights_keys or {}  # e.g. {"lm_head.weight": "model.embed_tokens.weight"}
    for copy_key in tied:
        keys.discard(copy_key)

    folder_name = f"{start}-{count}"
    if start == 0 and not first:
        # Usually only the driver starts at layer 0. If auto split gave the driver no layers, a worker does too,
        # and its folder must not be mixed up with a driver's, which also holds the embedding and the head.
        folder_name += "-no-head"
    folder = CACHE / name.replace("/", "--") / folder_name
    weights = load_weights(name, keys, folder, device)
    for copy_key, source_key in tied.items():
        weights[copy_key] = weights[source_key]

    # assign=True: the parameters become these tensors, no second copy. It's strict, so a missing one fails loudly.
    masked_model.load_state_dict(weights, assign=True)

    # Rotary position frequencies aren't in the checkpoint: HF computes them when it builds the model.
    for module in masked_model.modules():
        if hasattr(module, "inv_freq"):
            module.to_empty(device=device)
            masked_model._init_weights(module)

    if first:  # (workers lost their final norm in build_stage already)
        # The model's own sampling settings: temperature, top-k, top-p, end tokens... from_config only made defaults.
        try:
            masked_model.generation_config = GenerationConfig.from_pretrained(name)
        except OSError:  # some models (e.g. Qwen3.5) don't ship generation_config.json
            print(f"{name} has no generation_config.json: using defaults from its config (greedy decoding)")
        # The head is the final norm, then lm_head. It runs on the driver, once every stage is done.
        norm_name = final_norm_name(base)
        masked_model.lm_head = nn.Sequential(getattr(base, norm_name), masked_model.lm_head)
        setattr(base, norm_name, PipelineIdentityLayer())

    return base.eval(), masked_model


@torch.inference_mode()
def forward(base, x, pos, cache):
    """
    Run this stage's layers on x and return the resulting hidden states, shape [1, tokens, hidden size].

    x      token ids [1, tokens] on the first stage; hidden states [1, tokens, hidden size] everywhere else
    pos    where x starts in the sequence: 0 for a new prompt, then the number of tokens processed so far
    cache  this stage's KV cache. It grows with every call, so each new token only attends to what's stored
    """
    # The positions of x's tokens, e.g. pos=18 with one token -> [[18]]. Rotary embeddings need them.
    # HF would work them out from layer 0's cache length, but later stages don't own layer 0, so that cache
    # stays empty and every token would look like position 0. So we always pass them explicitly.
    positions = torch.arange(pos, pos + x.shape[1], device=x.device).unsqueeze(0)  # shape [1, tokens]

    # Token ids are integers (long); hidden states are floats (bf16). HF takes them through different arguments.
    if x.dtype == torch.long:
        out = base(input_ids=x, past_key_values=cache, use_cache=True, position_ids=positions)
    else:
        out = base(inputs_embeds=x, past_key_values=cache, use_cache=True, position_ids=positions)
    return out.last_hidden_state
