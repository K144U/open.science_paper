"""Text-rendered swap tracking: the E1 bridge task.

The paper's biggest gap is that everything is from-scratch 5M-50M models on
synthetic token streams, so the bridge to pretrained looped LMs is argued rather
than demonstrated.  Concurrent work (arXiv:2607.20594) has the same gap and
explicitly leaves pretrained models to future work, which makes this the
clearest differentiator available to us.

The task is state tracking phrased in natural language, so a pretrained LM can
actually attempt it:

    There are 5 cups in a row, numbered 1 to 5.
    Swap the contents of cup 2 and cup 5.
    Swap the contents of cup 1 and cup 2.
    ...
    Question: which cup holds the ball that started in cup 3?

That is composition in the symmetric group, exactly the computation the paper
studies, but rendered as text.  BIG-bench `tracking_shuffled_objects` was
rejected as the primary instrument because its 3/5/7 objects give only three
rungs on the length axis; here the number of swaps n dials freely from 1 to 32,
which is what a loops-vs-length curve needs.

The measurement that matters is how required depth scales with n WITHIN a
model, never across models: Huginn's 8-layer recurrent core and Ouro's 24-layer
stack make raw loop counts incommensurate.

Design notes that keep this honest:

* The answer is a single integer, so scoring needs no generation heuristics and
  no answer-extraction regex, only the argmax over digit tokens.
* Only the FINAL position is queried, so a model cannot score by copying.
* Distractor swaps that do not touch the queried ball are included at the same
  rate at every n, so longer instances are not trivially easier or harder for
  reasons unrelated to composition depth.
* Chance is 1/k for k cups, and we report it, because a "solved" threshold that
  ignores chance is meaningless at small k.
"""
from __future__ import annotations

import numpy as np

CUPS_DEFAULT = 5


def _fmt_list(k):
    return ", ".join(str(i) for i in range(1, k))


def make_instance(rng, n_swaps, k=CUPS_DEFAULT, query=None):
    """One instance.  Returns (prompt, answer_int, meta).

    The ball starts in `query` and every swap is applied in order; the answer is
    the cup it ends in.  We track the permutation directly rather than
    re-deriving it at scoring time, so the label cannot disagree with the text.
    """
    if query is None:
        query = int(rng.integers(1, k + 1))

    pos = query                              # where the tracked ball is now
    lines = []
    for _ in range(n_swaps):
        a = int(rng.integers(1, k + 1))
        b = int(rng.integers(1, k + 1))
        while b == a:
            b = int(rng.integers(1, k + 1))
        lines.append(f"Swap the contents of cup {a} and cup {b}.")
        if pos == a:
            pos = b
        elif pos == b:
            pos = a

    header = (f"There are {k} cups in a row, numbered 1 to {k}. "
              f"A ball starts in cup {query}.")
    body = "\n".join(lines)
    question = (f"\nQuestion: after all the swaps, which cup holds the ball? "
                f"Answer with a single number.\nAnswer: cup")
    prompt = f"{header}\n{body}{question}"
    meta = {"n_swaps": n_swaps, "k": k, "query": query, "answer": pos}
    return prompt, pos, meta


def make_batch(rng, n_swaps, batch, k=CUPS_DEFAULT):
    prompts, answers, metas = [], [], []
    for _ in range(batch):
        p, a, m = make_instance(rng, n_swaps, k)
        prompts.append(p)
        answers.append(a)
        metas.append(m)
    return prompts, answers, metas


def few_shot_prefix(rng, k=CUPS_DEFAULT, n_shot=2, n_swaps=2):
    """Short worked examples so the model knows the answer format.

    Kept deliberately shallow (n_swaps=2) so the exemplars never demonstrate
    the depth we are trying to measure.
    """
    if n_shot <= 0:
        return ""
    out = []
    for _ in range(n_shot):
        p, a, _ = make_instance(rng, n_swaps, k)
        out.append(f"{p} {a}\n")
    return "\n".join(out) + "\n"


def chance(k=CUPS_DEFAULT):
    return 1.0 / k


def echo_baseline(n_swaps, k=CUPS_DEFAULT, n=20000, seed=12345):
    """Accuracy of a model that ignores the swaps and answers the START cup.

    This, not uniform chance, is the baseline that matters.  A random swap
    touches the tracked ball's cup only 2/k of the time, so for small n the ball
    usually has not moved and "echo the start" scores far above 1/k: about 0.61
    at n=1 and 0.40 at n=2 for k=5.  A model reported as "above chance" at short
    lengths may be doing nothing but echoing, which is exactly the failure this
    task exists to detect.  Beating THIS is the evidence of state tracking.
    """
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n):
        _, ans, meta = make_instance(rng, n_swaps, k)
        if ans == meta["query"]:
            hit += 1
    return hit / n


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n in (1, 3, 8):
        p, a, m = make_instance(rng, n, 5)
        print("=" * 60)
        print(p, a)
        print("meta:", m)
    # Label must equal an independent replay of the swaps.
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 12))
        p, a, m = make_instance(rng, n, 5)
        pos = m["query"]
        for line in p.split("\n"):
            if line.startswith("Swap"):
                nums = [int(t) for t in line.replace(".", "").split()
                        if t.isdigit()]
                x, y = nums[0], nums[1]
                if pos == x:
                    pos = y
                elif pos == y:
                    pos = x
        assert pos == a, f"label {a} != replay {pos}"
    print("\nlabel self-check passed on 200 instances")
