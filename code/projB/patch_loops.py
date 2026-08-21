"""Project B - causal damage cones by activation patching over (loop, position).

Why this exists.  Our behavioural measurement (eval_loops.py) shows that
budget-trained models have a frontier speed v(n) = n/T*(n) that GROWS with
sequence length, which a constant-speed frontier cannot do.  But behaviour
alone cannot say which law replaces it: an affine frontier

    T* = n/v + b        (fixed speed v, one-off start-up cost b < 0)

fits our lengths about as well as a genuinely sub-linear one.  The two make
different CAUSAL predictions, and this script tests them.

Patch the residual state at (position i, loop t) with the corresponding state
from a corrupted run and measure which output positions change.  The set of
affected positions is the *damage cone* of that site.  Its slope, how far the
cone advances per loop, is a direct read of the frontier speed, measured
causally rather than inferred from a threshold.

    constant-speed / affine  ->  cone slope is the same at every n
    growing speed            ->  cone slope shallows as n grows

That contrast is the whole point: it separates "a faster scan with a head
start" from "a scan that gets faster on longer inputs", which is the claim our
behavioural data cannot settle on its own.

Eval-only; no training.  Usage:

  python code/projB/patch_loops.py --task s5 \
      --ckpt runs/s5_logc2_s3/ckpt_final.pt --lengths 8,16,32 \
      --out runs/s5_logc2_s3/damage_cones.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB import tasks                                          # noqa: E402
from projB.model import LoopedConfig, LoopedTransformer          # noqa: E402
from projB.model import _rope_cache                              # noqa: E402


@torch.no_grad()
def run_loop(model, tokens, T, patch=None, record=False):
    """Mirror of LoopedTransformer.forward with an optional intervention.

    patch: (loop_k, pos_i, value[B, d]) - overwrite the state at position i
    immediately after loop k completes.  Kept as a straight reimplementation of
    the model's own loop so that patching cannot silently diverge from training
    semantics; assert_matches_forward() below pins that down.
    """
    cfg = model.cfg
    B, L = tokens.shape
    cos, sin = _rope_cache(cfg.d_model // cfg.n_heads, L, tokens.device)
    e = model.wte(tokens)
    for b in model.prelude:
        e = b(e, cos, sin)

    x = e
    states = [x.detach().clone()] if record else None
    for k in range(T):
        if cfg.injection and k > 0:
            x = x + e
        if cfg.loop_enc == "sin":
            x = x + model.loop_emb[k].to(x.dtype)
        k_emb = (model.loop_emb[k].to(x.dtype)
                 if model.core.film is not None else None)
        x = model.core(x, cos, sin, k_emb=k_emb)
        if patch is not None:
            pk, pi, pv = patch
            if k == pk:
                x = x.clone()
                x[:, pi, :] = pv.to(x.dtype)
        if record:
            states.append(x.detach().clone())

    for b in model.coda:
        x = b(x, cos, sin)
    logits = model.head(model.ln_f(x))
    return (logits, states) if record else logits


@torch.no_grad()
def assert_matches_forward(model, tokens, T):
    """Our reimplemented loop must equal the model's own forward, exactly."""
    a = run_loop(model, tokens, T)
    b = model(tokens, T)
    d = (a - b).abs().max().item()
    assert d < 1e-4, f"patched forward diverges from model.forward by {d:.2e}"


@torch.no_grad()
def damage_cone(model, tokens_clean, tokens_corrupt, T, device):
    """[T, L, L] fraction of sequences whose prediction at j changes when the
    state at (loop k, position i) is replaced by the corrupted run's state."""
    B, L = tokens_clean.shape
    base = run_loop(model, tokens_clean, T).argmax(-1)          # [B, L]
    _, corrupt_states = run_loop(model, tokens_corrupt, T, record=True)

    dmg = np.zeros((T, L, L), dtype=np.float32)
    for k in range(T):
        # states[k+1] is the state AFTER loop k
        st = corrupt_states[k + 1]
        for i in range(L):
            out = run_loop(model, tokens_clean, T,
                           patch=(k, i, st[:, i, :])).argmax(-1)
            dmg[k, i] = (out != base).float().mean(0).cpu().numpy()
    return dmg


def settle_loop(dmg, thresh=0.5):
    """settle(j) = the loop after which output j stops depending on its inputs.

    Causal twin of the probe's solve_loop(j): the loop by which position j's
    answer is committed, measured by intervention rather than decodability, so
    its slope is directly comparable to the probe wavefront slope (1.02
    baseline, 0.53 under budget).  Unlike a probe it cannot be dismissed as
    "represented but not linearly decodable".

    ONLY UPSTREAM SITES COUNT (i < j).  Patching position j's own state at the
    last loop trivially changes output j, because that state is what the readout
    head reads; including i = j makes settle(j) = T for every j and every model,
    which is exactly what the first canary run produced.  The meaningful
    question is when j stops depending on the positions it must compose:

        sequential scan  ->  output j still moves until loop ~ j, slope ~ 1
        faster frontier  ->  commits earlier, slope < 1

    Positions with no upstream (j = 0, the BOS slot) return -1 and are excluded.
    """
    T, L, _ = dmg.shape
    settle = []
    for j in range(L):
        if j == 0:
            settle.append(-1)
            continue
        last = -1
        for k in range(T):
            if dmg[k, :j, j].max() >= thresh:   # strictly upstream of j
                last = k
        settle.append(last + 1 if last >= 0 else -1)
    return settle


def fit_slope(settle):
    """Least-squares slope of settle(j) against position j.

    Positions pinned at the loop ceiling are censored: they may simply have run
    out of loops rather than settled, and a pile-up at T flattens the slope for
    the wrong reason.  We drop the top value only when SEVERAL positions share
    it, which is what a ceiling looks like; a single position at the maximum is
    an ordinary largest value and is kept.
    """
    pts = [(j, s) for j, s in enumerate(settle) if s > 0]
    if len(pts) >= 3:
        top = max(s for _, s in pts)
        if sum(1 for _, s in pts if s == top) > 1:
            interior = [(j, s) for j, s in pts if s < top]
            if len(interior) >= 3:
                pts = interior
    if len(pts) < 3:
        return None
    js = np.array([p[0] for p in pts], float)
    ss = np.array([p[1] for p in pts], float)
    return float(np.polyfit(js, ss, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=tasks.TASKS)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lengths", default="8,16,32")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--loops", type=int, default=0,
                    help="loops to run; 0 = use the model's T* if known, else n")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--holdout-frac", type=float, default=0.05)
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
    print(f"[patch_loops] task={args.task} params={model.n_params()/1e6:.1f}M "
          f"ckpt={args.ckpt}", flush=True)

    out = {"task": args.task, "ckpt": args.ckpt, "thresh": args.thresh,
           "cones": {}}
    print("%4s %4s %7s %9s   %s" % ('n','T','slope','v_causal','settle_loop(j)'))
    for n in [int(x) for x in args.lengths.split(",")]:
        T = args.loops if args.loops > 0 else n
        tok_c, _ = sampler.sample(args.batch, n, rng, split="iid")
        tok_x, _ = sampler.sample(args.batch, n, rng, split="iid")
        tok_c = torch.from_numpy(tok_c).to(args.device)
        tok_x = torch.from_numpy(tok_x).to(args.device)

        assert_matches_forward(model, tok_c, T)

        dmg = damage_cone(model, tok_c, tok_x, T, args.device)
        settle = settle_loop(dmg, args.thresh)
        slope = fit_slope(settle)
        # slope is loops per position; its reciprocal is positions per loop,
        # the causal analogue of the behavioural frontier speed v = n / T*.
        v_causal = (1.0 / slope) if slope and slope > 1e-6 else None
        out["cones"][n] = {
            "T": T,
            "settle_loop": settle,
            "wavefront_slope": slope,
            "v_causal": v_causal,
            "damage": dmg.tolist(),
        }
        print("%4d %4d %7s %9s   %s" %
              (n, T, ('%.3f' % slope) if slope else '-',
               ('%.2f' % v_causal) if v_causal else '-', settle), flush=True)

    # The decisive comparison: does the causal speed grow with n?
    sl = {n: c["v_causal"] for n, c in out["cones"].items() if c["v_causal"]}
    if len(sl) >= 2:
        lo, hi = min(sl), max(sl)
        growth = sl[hi] / sl[lo] if sl[lo] else float("nan")
        out["v_causal_growth"] = growth
        print(f"\ncausal speed at n={lo}: {sl[lo]:.2f}   "
              f"at n={hi}: {sl[hi]:.2f}   growth {growth:.2f}x", flush=True)
        print("constant-speed/affine predicts 1.00x; growing speed predicts >1",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
