import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "mps"
name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(dev)

ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
out = model.generate(ids, max_new_tokens=20, do_sample=False)
print(tok.decode(out[0]))
