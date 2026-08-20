"""Slim model-only copies (cfg + model state) of the paper checkpoints.

eval_loops.py and probe.py load only ck['cfg'] and ck['model']; optimizer,
RNG, and log states are cluster-side reproducibility extras. Writes to
slim_ckpts/<run>/ mirroring runs/.
"""
import os
import sys

import torch

DYN = [500, 1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 40000]

runs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
jobs = [(r, "ckpt_final.pt") for r in runs]
jobs += [("s5_logc2_curr32", f"ckpt_{s:06d}.pt") for s in DYN]

done = missing = 0
for r, name in jobs:
    src = os.path.join("runs", r, name)
    dst = os.path.join("slim_ckpts", r, name)
    if not os.path.exists(src):
        print(f"MISSING {src}", flush=True)
        missing += 1
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ck = torch.load(src, map_location="cpu")
    torch.save({"step": ck.get("step"), "cfg": ck["cfg"],
                "model": ck["model"]}, dst)
    done += 1
    if done % 10 == 0:
        print(f"{done}/{len(jobs)}", flush=True)
print(f"DONE slimmed={done} missing={missing}", flush=True)
