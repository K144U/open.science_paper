"""Project B - loops-vs-length measurement on a trained checkpoint (RQ-B1).

The headline Phase-1 quantity: for each sequence length n, the minimal loop
count T* at which last-position accuracy reaches --solve-acc.  Sequential scan
=> T* ~ n (linear); associative parallel-prefix => T* ~ log n (logarithmic).

Unlike the sweep baked into train.py, this uses a DENSE T-ladder so T*(n) can
be read off precisely, and evaluates a user-given length set (in- and
out-of-distribution).  Eval-only: loads ckpt, no training.

Usage:
  python code/projB/eval_loops.py --task s5 --ckpt runs/s5_curr32_5m/ckpt_final.pt \
      --lengths 4,8,12,16,24,32,40,48,64 --out runs/s5_curr32_5m/loops_vs_length.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB import tasks                                          # noqa: E402
from projB.model import LoopedConfig, LoopedTransformer          # noqa: E402


@torch.no_grad()
def last_acc(model, sampler, n, T, device, rng, split="iid",
             batch=256, n_batches=2):
    ok = tot = 0
    for _ in range(n_batches):
        tok, lab = sampler.sample(batch, n, rng, split=split)
        tok = torch.from_numpy(tok).to(device)
        lab = torch.from_numpy(lab).to(device)
        pred = model(tok, T).argmax(dim=-1)
        ok += (pred[:, -1] == lab[:, -1]).sum().item()
        tot += batch
    return ok / tot


def t_ladder(n, extra_factor=1.5):
    """Dense ladder: every small T, then coarser, up to ~extra_factor*n."""
    hi = int(extra_factor * n) + 1
    fine = list(range(1, min(hi, 33)))            # 1..32 dense
    coarse = list(range(36, hi, 4))               # then every 4
    return sorted({*fine, *coarse, n, max(1, n // 2), 2 * n})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=tasks.TASKS)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lengths", default="4,8,12,16,24,32,40,48,64")
    ap.add_argument("--solve-acc", type=float, default=0.9)
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sampler = tasks.TaskSampler(args.task, holdout_frac=args.holdout_frac)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = LoopedConfig(**ckpt["cfg"])
    assert cfg.vocab_size == sampler.vocab_size, \
        f"ckpt vocab {cfg.vocab_size} != task {args.task} vocab {sampler.vocab_size}"
    model = LoopedTransformer(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval_loops] task={args.task} K={sampler.K} "
          f"params={model.n_params()/1e6:.1f}M ckpt={args.ckpt} "
          f"solve_acc={args.solve_acc}", flush=True)

    lengths = [int(x) for x in args.lengths.split(",")]
    out = {"task": args.task, "ckpt": args.ckpt, "solve_acc": args.solve_acc,
           "curve": {}}
    print(f"{'n':>5} {'T*':>5} {'log2n':>6}   acc-by-T (iid)")
    for n in lengths:
        ladder = t_ladder(n)
        accs, t_star = {}, None
        for T in ladder:
            a = last_acc(model, sampler, n, T, args.device, rng, split="iid")
            accs[T] = round(a, 4)
            if t_star is None and a >= args.solve_acc:
                t_star = T
        ho = None
        if sampler.has_holdout and t_star is not None:
            ho = round(last_acc(model, sampler, n, t_star, args.device, rng,
                                split="heldout"), 4)
        out["curve"][n] = {"T_star": t_star, "heldout_at_Tstar": ho,
                           "acc_by_T": accs}
        import math
        peak = max(accs.values())
        shown = {k: v for k, v in accs.items() if k <= max(2 * n, 8)}
        print(f"{n:>5} {str(t_star):>5} {math.log2(n):>6.2f}   "
              f"peak={peak:.3f} ho@T*={ho}  {shown}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    # quick verdict
    solved = [(n, d["T_star"]) for n, d in out["curve"].items()
              if d["T_star"] is not None]
    print(f"\nsolved lengths (n, T*): {solved}", flush=True)
    if len(solved) >= 3:
        ns = np.array([n for n, _ in solved], float)
        ts = np.array([t for _, t in solved], float)
        # linear (T*~n) vs log (T*~log2 n) fit quality
        r_lin = np.corrcoef(ns, ts)[0, 1]
        r_log = np.corrcoef(np.log2(ns), ts)[0, 1]
        print(f"corr(T*, n)={r_lin:.3f}   corr(T*, log2 n)={r_log:.3f}  "
              f"slope T*/n ~ {np.polyfit(ns, ts, 1)[0]:.3f}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
