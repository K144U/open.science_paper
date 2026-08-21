"""Project B — training + evaluation harness (plan §4.4, gate §4.8 weeks 1-2).

Trains a LoopedTransformer on prefix-product state tracking and measures the
paper's core quantity: the LOOPS-VS-LENGTH curve — for each eval length n,
the minimal loop count T* reaching >= --solve-acc last-position accuracy.
Sequential scan => T* ~ n (linear); associative => T* ~ log n.

Loop schedules (--loop-schedule, both Phase-1 training modes + intervention 1):
  fixed_n    T = n                      (enough loops for the sequential scan)
  uniform_n  T ~ U[ceil(n/2), 2n]       (stochastic, Huginn-flavored)
  log_n      T = ceil(c*log2 n), c=--log-c   (H-B2 budget; intervention 1)
  const      T = --T-const

Each batch uses a single length n ~ U[--n-min, --n-max] (no padding).  Loss =
CE over all positions (BOS ignored).  Eval per length: iid + heldout-pair
splits at the schedule's T, plus the T-sweep for the loops-vs-length curve.
Checkpoints every --ckpt-every steps (Phase-2 training movies).  All results
go to a JSON log next to the checkpoint dir.

Usage (GPU or CPU):
  python code/projB/train.py --task s5 --preset 5m --steps 3000 \
      --out runs/gate_s5_5m_seed0
  python code/projB/train.py --task parity --preset tiny --steps 300 \
      --device cpu --out runs/smoke            # local smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB import tasks                                   # noqa: E402
from projB.model import LoopedConfig, PRESETS, LoopedTransformer  # noqa: E402

EVAL_LENGTHS = (16, 64, 128, 256, 512, 1024)


def curr_n_max(step: int, args) -> int:
    """Length curriculum: ramp the sampled-length ceiling from
    --n-curr-start up to --n-max linearly over the first --curriculum-frac
    of training, then hold at --n-max.  0 disables (constant n-max).

    State tracking on hard groups (S5/A5) will not train from scratch on the
    full length range: at n=64 the T=n unrolled recurrence is 64 tied steps
    deep and gets no gradient traction.  Growing lengths gives the shallow-
    unroll gradient signal first (the empirical fix; FPRM-style tasks need it).
    """
    if args.curriculum_frac <= 0:
        return args.n_max
    frac = min(1.0, step / max(1.0, args.curriculum_frac * args.steps))

    # --curriculum-shape reshapes the ramp WITHOUT changing where it starts,
    # where it ends, or when it tops out, so the ablation isolates shape alone.
    # The paper's Limitations previously conceded shape was never varied.
    shape = getattr(args, "curriculum_shape", "linear")
    if shape == "exp":
        # slow at first, then fast: stays short for longer, so the scan has
        # more time to establish before the budget becomes binding.
        frac = frac ** 2
    elif shape == "log":
        # fast at first, then slow: reaches long lengths early.
        frac = math.sqrt(frac)
    elif shape == "step":
        # three discrete jumps rather than a smooth ramp.
        frac = math.floor(frac * 3.0 + 1e-9) / 3.0
    elif shape != "linear":
        raise ValueError("unknown curriculum shape %r" % shape)

    return int(round(args.n_curr_start + frac * (args.n_max - args.n_curr_start)))


def schedule_T(schedule: str, n: int, rng: np.random.Generator,
               log_c: float = 2.0, T_const: int = 32) -> int:
    if schedule == "fixed_n":
        return n
    if schedule == "uniform_n":
        return int(rng.integers(max(1, n // 2), 2 * n + 1))
    if schedule == "log_n":
        return max(1, math.ceil(log_c * math.log2(n)))
    if schedule == "const":
        return T_const
    if schedule == "const_jitter":
        # Depth sampled around a FIXED mean, independent of sequence length.
        # This is Huginn's contract: it has training-depth VARIANCE but no
        # scaling with n.  Together with const (no variance, no scaling) and
        # uniform_n (variance and scaling) it completes the 2x2 that separates
        # which of the two actually buys test-time loop tolerance.
        lo = max(1, T_const // 2)
        return int(rng.integers(lo, 2 * T_const + 1))
    raise ValueError(f"unknown schedule {schedule!r}")


@torch.no_grad()
def accuracy(model, sampler, n: int, T: int, device, rng,
             split: str = "iid", batch: int = 256, n_batches: int = 2):
    """(per-position acc, last-position acc, full-sequence acc)."""
    pos = last = full = tot = 0
    for _ in range(n_batches):
        tok, lab = sampler.sample(batch, n, rng, split=split)
        tok = torch.from_numpy(tok).to(device)
        lab = torch.from_numpy(lab).to(device)
        pred = model(tok, T).argmax(dim=-1)
        m = lab != tasks.IGNORE
        pos += (pred[m] == lab[m]).sum().item()
        tot += m.sum().item()
        last += (pred[:, -1] == lab[:, -1]).sum().item()
        full += ((pred == lab) | ~m).all(dim=1).sum().item()
    return pos / tot, last / (batch * n_batches), full / (batch * n_batches)


@torch.no_grad()
def loops_vs_length(model, sampler, device, rng, solve_acc: float = 0.9,
                    lengths=EVAL_LENGTHS, max_T_factor: float = 2.0):
    """For each n: last-pos acc over a T-ladder + minimal solving T.

    Ladder: powers of 2 up to max_T_factor*n plus n itself — dense enough to
    distinguish T* ~ log n from T* ~ n without an O(n)-point sweep.
    """
    out = {}
    for n in lengths:
        ladder = sorted({2 ** k for k in range(0, int(math.log2(n * max_T_factor)) + 1)
                         if 2 ** k <= max_T_factor * n} | {n, n // 2, 2 * n})
        accs, t_star = {}, None
        for T in ladder:
            _, last, _ = accuracy(model, sampler, n, T, device, rng,
                                  batch=128, n_batches=1)
            accs[str(T)] = round(last, 4)
            if t_star is None and last >= solve_acc:
                t_star = T
        out[str(n)] = {"acc_by_T": accs, "T_star": t_star}
    return out


def save_ckpt(path, step, cfg, model, opt, rng, log):
    """Full checkpoint: model + optimizer + RNG + running log, so a crashed run
    (the shared node throws intermittent CUDA driver faults) resumes exactly."""
    torch.save({"step": step, "cfg": cfg.__dict__, "model": model.state_dict(),
                "optim": opt.state_dict(),
                "np_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "log": log}, path)


def maybe_resume(args, model, opt, rng, log):
    """If --resume and a periodic checkpoint exists in --out, restore full state
    and return the step to continue from; else return 0 (fresh)."""
    if not args.resume:
        return 0
    ckpts = sorted(glob.glob(os.path.join(args.out, "ckpt_[0-9]*.pt")))
    if not ckpts:
        return 0
    ck = torch.load(ckpts[-1], map_location=args.device)
    model.load_state_dict(ck["model"])
    if "optim" in ck:
        opt.load_state_dict(ck["optim"])
    if "np_rng" in ck:
        rng.bit_generator.state = ck["np_rng"]
    if "torch_rng" in ck:
        torch.set_rng_state(ck["torch_rng"].cpu()
                            if hasattr(ck["torch_rng"], "cpu") else ck["torch_rng"])
    if ck.get("log"):
        log.clear(); log.update(ck["log"])
    print(f"[projB] RESUMED from {ckpts[-1]} at step {ck['step']}", flush=True)
    return int(ck["step"])


def train(args):
    device = args.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    sampler = tasks.TaskSampler(args.task, holdout_frac=args.holdout_frac)
    kw = dict(PRESETS[args.preset])
    cfg = LoopedConfig(vocab_size=sampler.vocab_size,
                       injection=not args.no_injection,
                       loop_enc=args.loop_enc, norm=args.norm, film=args.film,
                       prelude_layers=args.prelude, coda_layers=args.coda,
                       **kw)
    model = LoopedTransformer(cfg).to(device)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[projB] task={args.task} K={sampler.K} preset={args.preset} "
          f"params={model.n_params()/1e6:.1f}M inject={cfg.injection} "
          f"loop_enc={cfg.loop_enc} norm={cfg.norm} film={cfg.film} "
          f"prelude/coda={cfg.prelude_layers}/{cfg.coda_layers} "
          f"schedule={args.loop_schedule} seed={args.seed} bf16={use_bf16}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    warm = max(1, args.warmup)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        p = (step - warm) / max(1, args.steps - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    os.makedirs(args.out, exist_ok=True)
    log = {"config": vars(args), "n_params": model.n_params(), "train": [],
           "eval": {}, "loops_vs_length": None}
    start_step = maybe_resume(args, model, opt, rng, log)
    t0 = time.time()

    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        n = int(rng.integers(args.n_min, curr_n_max(step, args) + 1))
        T = schedule_T(args.loop_schedule, n, rng, args.log_c, args.T_const)
        tok, lab = sampler.sample(args.batch, n, rng, split="train")
        tok = torch.from_numpy(tok).to(device)
        lab = torch.from_numpy(lab).to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
            logits = model(tok, T)
            loss = F.cross_entropy(logits.flatten(0, 1).float(),
                                   lab.flatten(), ignore_index=tasks.IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                m = lab != tasks.IGNORE
                acc = (pred[m] == lab[m]).float().mean().item()
            rate = (step + 1 - start_step) / (time.time() - t0)
            print(f"  step {step + 1}/{args.steps}  n={n} T={T}  "
                  f"loss {loss.item():.4f}  acc {acc:.3f}  "
                  f"{rate:.1f} steps/s", flush=True)
            log["train"].append({"step": step + 1, "loss": round(loss.item(), 5),
                                 "acc": round(acc, 4), "n": n, "T": T})
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            save_ckpt(os.path.join(args.out, f"ckpt_{step + 1:06d}.pt"),
                      step + 1, cfg, model, opt, rng, log)

    # ---- final eval: iid + heldout at schedule-T, then the headline curve
    model.eval()
    for n in (args.n_max, 2 * args.n_max):
        T = schedule_T(args.loop_schedule, n, rng, args.log_c, args.T_const)
        for split in (("iid", "heldout") if sampler.has_holdout else ("iid",)):
            p, l, f_ = accuracy(model, sampler, n, T, device, rng, split=split)
            log["eval"][f"{split}_n{n}_T{T}"] = {
                "pos_acc": round(p, 4), "last_acc": round(l, 4),
                "full_acc": round(f_, 4)}
            print(f"  eval {split} n={n} T={T}: pos {p:.3f} last {l:.3f} "
                  f"full {f_:.3f}", flush=True)

    if args.skip_final_sweep:
        print("  (skipping loops-vs-length sweep: --skip-final-sweep)",
              flush=True)
    else:
        lengths = tuple(int(x) for x in args.eval_lengths.split(","))
        log["loops_vs_length"] = loops_vs_length(
            model, sampler, device, rng, solve_acc=args.solve_acc,
            lengths=lengths)
        for n, d in log["loops_vs_length"].items():
            print(f"  loops-vs-length n={n}: T*={d['T_star']}  {d['acc_by_T']}",
                  flush=True)

    save_ckpt(os.path.join(args.out, "ckpt_final.pt"),
              args.steps, cfg, model, opt, rng, log)
    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"wrote {args.out}/log.json", flush=True)
    return log


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", choices=tasks.TASKS, default="s5")
    p.add_argument("--preset", choices=list(PRESETS), default="5m")
    p.add_argument("--no-injection", action="store_true")
    p.add_argument("--loop-enc", choices=["none", "sin"], default="none")
    p.add_argument("--norm", choices=["pre", "sandwich"], default="pre")
    p.add_argument("--film", action="store_true")
    p.add_argument("--prelude", type=int, default=0)
    p.add_argument("--coda", type=int, default=0)
    p.add_argument("--loop-schedule", default="fixed_n",
                   choices=["fixed_n", "uniform_n", "log_n", "const", "const_jitter"])
    p.add_argument("--log-c", type=float, default=2.0)
    p.add_argument("--T-const", type=int, default=32)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--n-min", type=int, default=4)
    p.add_argument("--n-max", type=int, default=64)
    p.add_argument("--curriculum-frac", type=float, default=0.0,
                   help="ramp length ceiling over this fraction of steps "
                        "(0 = off); needed for S5/A5 from scratch")
    p.add_argument("--curriculum-shape", default="linear",
                   choices=["linear", "exp", "log", "step"],
                   help="ramp shape; start, end and top-out step are "
                        "unchanged, so this isolates shape alone")
    p.add_argument("--n-curr-start", type=int, default=8,
                   help="curriculum starting length ceiling")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--solve-acc", type=float, default=0.9)
    p.add_argument("--eval-lengths", default="16,64,128,256,512,1024")
    p.add_argument("--skip-final-sweep", action="store_true",
                   help="skip the loops-vs-length OOD sweep (recipe-debug runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--resume", action="store_true",
                   help="resume from the latest ckpt_NNNNNN.pt in --out if one "
                        "exists (model+optimizer+RNG+log); recovers a run killed "
                        "by the shared node's intermittent CUDA driver faults")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
