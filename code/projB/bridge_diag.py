"""Diagnose an at-floor E1 pilot: can the model not do it, or can we not read it?

The pilots put both pretrained models at or below the echo baseline.  Before
that is reported as a capability finding it has to be separated from a harness
failure, because the two look identical in an accuracy column.  This prints what
the model actually predicts at the answer position, so the distinction is
visible rather than assumed:

  * top tokens are cup numbers, just often wrong -> the model is engaging and
    the task is genuinely hard; the pilot result stands.
  * top tokens are punctuation, words, or a different answer format -> our
    scoring is reading the wrong thing and the pilot result is an artefact.

Also sweeps the easy knobs (cups, few-shot count, prompt style) at n=1, where a
competent model should be near ceiling, so a format problem shows up as a large
swing from a trivial change.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB import swap_task                                      # noqa: E402
from projB.bridge_depth import load_model, answer_logits, accuracy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["huginn", "ouro"])
    ap.add_argument("--depth", type=int, default=0, help="0 = model default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    model, tok = load_model(args.model, args.device)
    depth = args.depth or (64 if args.model == "huginn" else 4)

    # ---- 1. what does it actually predict at the answer position?
    rng = np.random.default_rng(0)
    prompts, answers, metas = swap_task.make_batch(rng, 1, 4, 5)
    pre = swap_task.few_shot_prefix(np.random.default_rng(1), 5, 2)
    shown = [pre + p for p in prompts]
    lg = answer_logits(model, tok, args.model, shown, depth, args.device)
    print("\n=== what the model predicts at the answer position (n=1) ===")
    for i in range(len(shown)):
        top = torch.topk(lg[i], 10)
        toks = [tok.decode([t]) for t in top.indices.tolist()]
        print("gold=cup %d  start=cup %d" % (answers[i], metas[i]["query"]))
        print("   top10: %s" % " | ".join(repr(t) for t in toks))
    print("\ntail of the prompt the model saw:")
    print(repr(shown[0][-160:]))

    # ---- 2. do easy knobs move it? at n=1 a competent model is near ceiling.
    print("\n=== n=1 accuracy under easier settings ===")
    print("(echo baseline is reported alongside; beating it is the bar)")
    print("%6s %8s %10s %8s %8s" % ("cups", "n_shot", "acc", "echo", "margin"))
    for k in (3, 5):
        for n_shot in (2, 5):
            a = accuracy(model, tok, args.model, np.random.default_rng(0),
                         1, depth, args.device, k, 32, 4, n_shot)
            e = swap_task.echo_baseline(1, k)
            print("%6d %8d %10.3f %8.3f %+8.3f" % (k, n_shot, a, e, a - e))


if __name__ == "__main__":
    main()
