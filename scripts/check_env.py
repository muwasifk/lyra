import platform

import torch

print(platform.node(), platform.system(), "torch", torch.__version__)

if torch.cuda.is_available():
    dev = "cuda"
    free, total = torch.cuda.mem_get_info()
    print(torch.cuda.get_device_name(), "sm", torch.cuda.get_device_capability())
    print(f"free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
elif torch.backends.mps.is_available():
    dev = "mps"
    print(f"MPS recommended max {torch.mps.recommended_max_memory() / 2**30:.1f} GiB")
else:
    dev = "cpu"

for dt in (torch.float16, torch.bfloat16):
    a = torch.randn(2048, 2048, device=dev, dtype=dt)
    ref = a.float() @ a.float()
    err = ((a @ a).float() - ref).abs().max() / ref.abs().max()
    print(f"{dt}: ok, rel err {err:.1e}")
