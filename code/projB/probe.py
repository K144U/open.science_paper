"""Project B - Phase-2 mechanistic probe (RQ-B2): prefix-state wavefront +
block-product ladder on a trained checkpoint.

We read the per-loop hidden states (model.forward(return_states=True) ->
[T+1, B, L, d]) and ask, with a linear probe at each (loop k, position i),
WHAT is linearly decodable:

  1. PREFIX-STATE WAVEFRONT.  Target = running product g_1..g_i at position i.
     For each position i, the earliest loop k where it becomes decodable
     (acc >= --solve-acc) is solve_loop[i].
       sequential scan  -> solve_loop[i] ~ i        (diagonal wavefront)
       associative      -> solve_loop[i] ~ log2(i)  (flat/log band)

  2. BLOCK-PRODUCT LADDER (at the final position).  Target = product of the
     2^k elements ending at the last position, probed across loops.
       associative parallel-prefix -> width-2^k block decodable at loop ~k
       sequential                  -> only the width-i prefix at loop ~i

The wavefront slope (corr/slope of solve_loop vs position, vs vs log2) is the
single-number "algorithm label"; the (loop x position) accuracy heatmap is the
centerpiece figure (baseline sequential = diagonal; c=2 log-budget = flat).

Usage:
  python code/projB/probe.py --ckpt runs/s5_logc2_curr32/ckpt_final.pt \
      --task s5 --n 16 --n-seqs 4000 --out runs/s5_logc2_curr32/probe_n16.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB import tasks                                          # noqa: E402
from projB.model import LoopedConfig, LoopedTransformer          # noqa: E402


@torch.no_grad()
def _standardize(x, mu, sd):
    return (x - mu) / sd


def linear_probe_acc(Xtr, ytr, Xte, yte, K, device, epochs=200, lr=0.05, wd=1e-3):
    """Multiclass linear probe test accuracy.  X: [N,d] tensor, y: [N] long."""
    d = Xtr.shape[1]
    W = nn.Linear(d, K).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(W(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (W(Xte).argmax(1) == yte).float().mean().item()


def block_products(table, seqs, width):
    """[B, n] product of the `width` elements ending at each position
    (positions < width-1 get the partial prefix)."""
    B, n = seqs.shape
    out = np.empty((B, n), dtype=np.int64)
    for i in range(n):
        lo = max(0, i - width + 1)
        acc = seqs[:, lo].copy()
        for j in range(lo + 1, i + 1):
            acc = table[acc, seqs[:, j]]
        out[:, i] = acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, choices=tasks.TASKS)
    ap.add_argument("--n", type=int, default=16, help="sequence length to probe")
    ap.add_argument("--T", type=int, default=0, help="loops to run (0 => n)")
    ap.add_argument("--n-seqs", type=int, default=4000)
    ap.add_argument("--fwd-chunk", type=int, default=1000,
                    help="sequences per return_states forward (caps GPU memory)")
    ap.add_argument("--solve-acc", type=float, default=0.9)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    n = args.n
    T = args.T or n

    rng = np.random.default_rng(args.seed)
    sampler = tasks.TaskSampler(args.task, holdout_frac=args.holdout_frac)
    K = sampler.K
    ck = torch.load(args.ckpt, map_location=dev)
    cfg = LoopedConfig(**ck["cfg"])
    model = LoopedTransformer(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[probe] {args.ckpt} task={args.task} K={K} n={n} T={T} "
          f"n_seqs={args.n_seqs}", flush=True)

    # tokens [N, n+1] (BOS at col 0); labels = prefix products, col 0 = IGNORE
    tok, lab = sampler.sample(args.n_seqs, n, rng, split="iid")
    seqs = tok[:, 1:]                                   # [N, n] group elements
    # chunk the return_states forward so peak GPU memory stays small (the shared
    # node is tight); collect [T+1, N, n+1, d] on CPU.
    chunks = []
    with torch.no_grad():
        for s in range(0, args.n_seqs, args.fwd_chunk):
            tk = torch.from_numpy(tok[s:s + args.fwd_chunk]).to(dev)
            _, st = model(tk, T, return_states=True)     # [T+1, b, n+1, d]
            chunks.append(st.float().cpu().numpy())
    states = np.concatenate(chunks, axis=1)             # [T+1, N, n+1, d]

    ntr = int(args.train_frac * args.n_seqs)
    tr, te = slice(0, ntr), slice(ntr, args.n_seqs)

    # ---- 1. prefix-state wavefront: acc[k, i] ----
    accs = np.full((T + 1, n + 1), np.nan)
    for k in range(T + 1):
        for i in range(1, n + 1):                       # skip BOS position 0
            X = states[k, :, i, :]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr = torch.from_numpy(_standardize(X[tr], mu, sd)).to(dev)
            Xte = torch.from_numpy(_standardize(X[te], mu, sd)).to(dev)
            y = lab[:, i]
            ytr = torch.from_numpy(y[tr]).long().to(dev)
            yte = torch.from_numpy(y[te]).long().to(dev)
            accs[k, i] = linear_probe_acc(Xtr, ytr, Xte, yte, K, dev)
        print(f"  loop {k}: max-pos-acc {np.nanmax(accs[k]):.3f}", flush=True)

    # earliest loop each position is solved
    solve_loop = {}
    for i in range(1, n + 1):
        col = accs[:, i]
        hits = np.where(col >= args.solve_acc)[0]
        solve_loop[i] = int(hits[0]) if len(hits) else None

    # wavefront metric: solve_loop vs position (linear) vs log2(position)
    solved = [(i, sl) for i, sl in solve_loop.items() if sl is not None]
    verdict = {}
    if len(solved) >= 3:
        pos = np.array([i for i, _ in solved], float)
        slp = np.array([s for _, s in solved], float)
        verdict = {
            "corr_linear": float(np.corrcoef(pos, slp)[0, 1]),
            "corr_log2": float(np.corrcoef(np.log2(pos), slp)[0, 1]),
            "slope_vs_pos": float(np.polyfit(pos, slp, 1)[0]),
        }

    # ---- 2. block-product ladder at the final position ----
    ladder = {}
    fin = n                                             # last token position
    for w in [2 ** j for j in range(0, int(math.log2(n)) + 1)]:
        bp = block_products(sampler.table, seqs, w)[:, -1]   # width-w block at end
        row = []
        for k in range(T + 1):
            X = states[k, :, fin, :]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr = torch.from_numpy(_standardize(X[tr], mu, sd)).to(dev)
            Xte = torch.from_numpy(_standardize(X[te], mu, sd)).to(dev)
            ytr = torch.from_numpy(bp[tr]).long().to(dev)
            yte = torch.from_numpy(bp[te]).long().to(dev)
            row.append(round(linear_probe_acc(Xtr, ytr, Xte, yte, K, dev), 4))
        first = next((k for k, a in enumerate(row) if a >= args.solve_acc), None)
        ladder[w] = {"acc_by_loop": row, "first_solved_loop": first}

    out = {"ckpt": args.ckpt, "task": args.task, "n": n, "T": T,
           "solve_acc": args.solve_acc,
           "wavefront_acc": accs.round(4).tolist(),
           "solve_loop": solve_loop, "verdict": verdict,
           "block_ladder": ladder}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== WAVEFRONT solve_loop[position] ===")
    print("  ", {i: solve_loop[i] for i in sorted(solve_loop)})
    print("  verdict:", verdict,
          "\n  (slope~1 & corr_linear>corr_log2 => SEQUENTIAL diagonal;"
          "\n   slope~0 flat / corr_log2>corr_linear => ASSOCIATIVE log-band)")
    print("=== BLOCK-PRODUCT ladder (width: first solved loop) ===")
    print("  ", {w: d["first_solved_loop"] for w, d in ladder.items()},
          "\n  (associative: width 2^k first solved at loop ~k)")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
