"""E1: does the frontier story hold for PRETRAINED looped LMs?

The paper's thesis is that what a depth-recurrent model does with its loops is
set by the resource regime it was trained under, not by its architecture.  Two
public pretrained looped LMs are a natural experiment on exactly that, because
they were trained under opposite loop contracts:

    Huginn-0125   samples recurrence depth around a mean of 32  -> SLACK
    Ouro-1.4B     trains at a small fixed depth (4)             -> TIGHT

Our thesis predicts they should differ in kind, not just in skill: the
slack-trained model should show a serial, one-step-per-item frontier, and the
tight-trained model should have been forced to compress.

METHOD.  For each model and each task length n, sweep the inference depth T and
record accuracy on text-rendered swap tracking.  T*(n) is the smallest depth
reaching the solve threshold.  The shape of T*(n) is the algorithm signature,
exactly as in the from-scratch experiments:

    constant-speed serial frontier  ->  T* = n/v with v fixed, so v(n) is flat
    something faster                ->  v(n) = n / T*(n) grows with n

CRITICAL: compare T*(n) shape WITHIN a model, never raw depths across models.
Huginn's 8-layer recurrent core and Ouro's 24-layer stack make loop counts
incommensurate; only the scaling is comparable.

THE PILOT COMES FIRST.  Both models can simply be at floor on this task, in
which case there is no T* to measure and the whole experiment is void.  Run
with --pilot to check small n against chance before committing to a full sweep.
That is the main risk and it is cheap to test.

Usage:
  python code/projB/bridge_depth.py --model huginn --pilot --out runs/e1/pilot_huginn.json
  python code/projB/bridge_depth.py --model ouro   --pilot --out runs/e1/pilot_ouro.json
  python code/projB/bridge_depth.py --model huginn --lengths 1,2,4,6,8,12,16 \
      --depths 1,2,4,8,16,24,32,48,64 --out runs/e1/sweep_huginn.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB import swap_task                                      # noqa: E402

HUGINN = "tomg-group-umd/huginn-0125"
OURO = "ByteDance/Ouro-1.4B"
OURO_MAX_STEPS = 4          # Ouro's trained recurrence depth


def load_model(which, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = HUGINN if which == "huginn" else OURO
    print(f"[e1] loading {name} (bf16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    return model, tok


def digit_token_ids(tok, k):
    """Token ids for the answers '1'..'k', preferring a leading-space variant.

    Scoring restricted to these ids turns the task into a k-way choice, so no
    answer-extraction regex is needed and a malformed generation cannot be
    silently scored as wrong for the wrong reason.
    """
    ids = {}
    for d in range(1, k + 1):
        cands = []
        for form in (f" {d}", f"{d}"):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                cands.append(enc[0])
        if not cands:
            raise RuntimeError(f"answer '{d}' is not a single token for this "
                               f"tokenizer; pick a different k or scoring")
        ids[d] = cands[0]
    return ids


@torch.no_grad()
def answer_logits(model, tok, which, prompts, depth, device):
    """Next-token logits at the final position, at the given recurrence depth."""
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    if which == "huginn":
        out = model(enc["input_ids"], num_steps=depth,
                    attention_mask=enc["attention_mask"])
    else:
        # Ouro exposes depth as an early-exit index (0-based).
        model.early_exit_step = max(0, min(depth, OURO_MAX_STEPS) - 1)
        out = model(enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out[0]
    # Last non-pad position per row: padding may be left or right.
    idx = enc["attention_mask"].sum(1) - 1
    return logits[torch.arange(logits.shape[0]), idx]


@torch.no_grad()
def accuracy(model, tok, which, rng, n_swaps, depth, device, k, batch,
             n_batches, n_shot):
    ids = digit_token_ids(tok, k)
    order = [ids[d] for d in range(1, k + 1)]
    ok = tot = 0
    for _ in range(n_batches):
        prompts, answers, _ = swap_task.make_batch(rng, n_swaps, batch, k)
        if n_shot:
            pre = swap_task.few_shot_prefix(rng, k, n_shot)
            prompts = [pre + p for p in prompts]
        lg = answer_logits(model, tok, which, prompts, depth, device)
        pred = lg[:, order].argmax(-1).cpu().numpy() + 1
        ok += int((pred == np.array(answers)).sum())
        tot += len(answers)
    return ok / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["huginn", "ouro"])
    ap.add_argument("--lengths", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--depths", default="")
    ap.add_argument("--cups", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--n-shot", type=int, default=2)
    ap.add_argument("--solve-acc", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot", action="store_true",
                    help="small-n floor check at full depth only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, tok = load_model(args.model, args.device)
    ch = swap_task.chance(args.cups)

    if args.depths:
        depths = [int(x) for x in args.depths.split(",")]
    elif args.model == "huginn":
        depths = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    else:
        depths = list(range(1, OURO_MAX_STEPS + 1))

    lengths = [int(x) for x in args.lengths.split(",")]
    out = {"model": args.model, "cups": args.cups, "chance": ch,
           "solve_acc": args.solve_acc, "n_shot": args.n_shot,
           "depths": depths, "curve": {}}

    # ---------------------------------------------------------------- pilot
    if args.pilot:
        full = max(depths)
        print("\n[pilot] %s at full depth %d, uniform chance=%.3f" %
              (args.model, full, ch))
        print("Baseline to beat is ECHO (answer the start cup, ignore the "
              "swaps), NOT uniform chance:")
        print("%4s %8s %8s %8s  %s" %
              ("n", "acc", "echo", "margin", "verdict"))
        n_tracking = 0
        for n in lengths[:5]:
            a = accuracy(model, tok, args.model, rng, n, full, args.device,
                         args.cups, args.batch, args.n_batches, args.n_shot)
            echo = swap_task.echo_baseline(n, args.cups)
            floor_n = max(echo, ch)
            tracking = a > floor_n + 0.10
            n_tracking += int(tracking)
            print("%4d %8.3f %8.3f %+8.3f  %s" %
                  (n, a, echo, a - floor_n,
                   "TRACKING" if tracking else "at echo/chance floor"),
                  flush=True)
            out["curve"][n] = {"pilot_acc_at_full_depth": round(a, 4),
                               "echo_baseline": round(echo, 4),
                               "beats_echo": bool(tracking)}
        # One length above the floor proves nothing: T*(n) is a CURVE, so at
        # least three usable rungs are needed before a sweep can measure
        # anything.
        usable = n_tracking >= 3
        out["n_lengths_tracking"] = n_tracking
        out["at_floor"] = not usable
        print("\nverdict: %s beats the echo baseline at %d of %d lengths -> %s"
              % (args.model, n_tracking, len(lengths[:5]),
                 "usable, proceed to the full sweep" if usable else
                 "NOT USABLE as specified; a T*(n) curve needs >=3 rungs"))
    # ----------------------------------------------------------- full sweep
    else:
        print(f"\n[sweep] {args.model}  chance={ch:.3f}  "
              f"solve>={args.solve_acc}")
        print(f"{'n':>4} {'T*':>4} {'v=n/T*':>7}   acc by depth")
        for n in lengths:
            accs, t_star = {}, None
            for T in depths:
                a = accuracy(model, tok, args.model, rng, n, T, args.device,
                             args.cups, args.batch, args.n_batches,
                             args.n_shot)
                accs[T] = round(a, 4)
                if t_star is None and a >= args.solve_acc:
                    t_star = T
            v = (n / t_star) if t_star else None
            out["curve"][n] = {"T_star": t_star, "acc_by_depth": accs,
                               "v": v}
            print(f"{n:>4} {str(t_star):>4} "
                  f"{('%.2f' % v) if v else '   -':>7}   {accs}", flush=True)

        # Within-model frontier speed: flat means a constant-speed serial
        # frontier, growth means something faster.
        vs = [(n, c["v"]) for n, c in out["curve"].items() if c.get("v")]
        if len(vs) >= 2:
            lo, hi = vs[0], vs[-1]
            out["v_growth"] = hi[1] / lo[1] if lo[1] else None
            print(f"\nv(n={lo[0]})={lo[1]:.2f}  v(n={hi[0]})={hi[1]:.2f}  "
                  f"growth {out['v_growth']:.2f}x")
            print("constant-speed frontier predicts 1.00x")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
