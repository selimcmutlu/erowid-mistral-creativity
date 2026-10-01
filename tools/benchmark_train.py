"""Measure LoRA training throughput for Mistral-7B on this machine (MPS) and project the full run.

Usage: .venv/bin/python tools/benchmark_train.py [model_id]
Writes a JSON summary to artifacts/logs/benchmark_train.json.
"""
import gc
import glob
import html
import json
import os
import re
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus used to fine-tune models in the original report", "erowid-main", "Experiences")
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "mistral-community/Mistral-7B-v0.2"
SEQ = 512
STEPS = 4
TOP5 = {"Cannabis", "LSD", "MDMA", "Mushrooms", "Salvia divinorum"}


def canonical(label):
    lab = html.unescape(label).strip()
    if "&" in lab:
        return None
    return re.split(r"\s+-\s+|\s*\(", lab, maxsplit=1)[0].strip()


# Corpus statistics needed for the projection
docs = []
for f in glob.glob(os.path.join(CORPUS, "*.json")):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    if canonical(j["drug"]) in TOP5:
        docs.append(" ".join(j["text"]) if isinstance(j["text"], list) else j["text"])
tok = AutoTokenizer.from_pretrained(MODEL_ID)
rng = np.random.default_rng(111)
sample = [docs[i] for i in rng.choice(len(docs), size=300, replace=False)]
lens = [min(len(tok(t).input_ids) + 25, SEQ) for t in sample]  # +25 ~ prompt tokens
n_train = int(len(docs) * 0.95)
tokens_total = n_train * float(np.mean(lens)) * 2  # 2 epochs
print(f"docs={len(docs)} train={n_train} mean_truncated_len={np.mean(lens):.0f} tokens(2 epochs)={tokens_total/1e6:.2f}M", flush=True)

ids = tok("\n\n".join(sample[:40])).input_ids
results = []


def bench(model, bs, gc_on, label):
    if gc_on:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    x = torch.tensor(ids[: bs * SEQ]).view(bs, SEQ).to("mps")
    times, losses = [], []
    torch.mps.empty_cache()
    for s in range(STEPS + 1):
        t = time.time()
        out = model(input_ids=x, labels=x)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        times.append(time.time() - t)
        losses.append(float(out.loss))
    step = float(np.mean(times[1:]))  # drop warm-up step
    tps = bs * SEQ / step
    mem = torch.mps.driver_allocated_memory() / 1e9
    r = dict(config=label, bs=bs, grad_ckpt=gc_on, sec_per_step=round(step, 2), tokens_per_sec=round(tps, 1),
             driver_mem_gb=round(mem, 1), losses=[round(l, 3) for l in losses],
             projected_hours=round(tokens_total / tps / 3600, 1))
    print(json.dumps(r), flush=True)
    results.append(r)
    del opt, x
    gc.collect()
    torch.mps.empty_cache()


lora = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                  bias="none", task_type="CAUSAL_LM")
for dtype, configs in [(torch.bfloat16, [(1, False), (1, True), (2, False), (2, True)]), (torch.float16, [(1, False)])]:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to("mps")
    model = get_peft_model(model, lora)
    for bs, gc_on in configs:
        try:
            bench(model, bs, gc_on, f"{str(dtype).split('.')[-1]} bs={bs} gc={gc_on}")
        except Exception as e:  # OOM or unsupported op
            print(json.dumps(dict(config=f"{dtype} bs={bs} gc={gc_on}", error=str(e)[:300])), flush=True)
            gc.collect()
            torch.mps.empty_cache()
    del model
    gc.collect()
    torch.mps.empty_cache()

os.makedirs(os.path.join(ROOT, "artifacts", "logs"), exist_ok=True)
json.dump(dict(model=MODEL_ID, docs=len(docs), n_train=n_train, tokens_total=tokens_total, results=results),
          open(os.path.join(ROOT, "artifacts", "logs", "benchmark_train.json"), "w"), indent=2)
print("BENCHMARK DONE", flush=True)
