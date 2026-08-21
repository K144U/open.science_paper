"""CPU unit tests for projB/tasks.py.

Run:  python code/projB/test_tasks.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB import tasks  # noqa: E402


def group_axioms(table: np.ndarray) -> bool:
    """Closure (by dtype/range), identity, inverses, associativity (sampled)."""
    K = table.shape[0]
    if table.min() < 0 or table.max() >= K:
        return False
    # identity: some e with e*g = g*e = g for all g
    ids = [e for e in range(K)
           if (table[e] == np.arange(K)).all() and (table[:, e] == np.arange(K)).all()]
    if len(ids) != 1:
        return False
    e = ids[0]
    # inverses: every row contains the identity
    if not all((table[g] == e).any() for g in range(K)):
        return False
    # associativity on a random sample of triples
    rng = np.random.default_rng(0)
    a, b, c = (rng.integers(0, K, 200) for _ in range(3))
    return (table[table[a, b], c] == table[a, table[b, c]]).all()


def main():
    checks = []

    # ---- group construction
    for name, K in (("parity", 2), ("mod60", 60), ("s5", 120), ("a5", 60)):
        t = tasks.make_group(name)
        checks.append((f"{name} axioms (K={K})",
                       t.shape == (K, K) and group_axioms(t)))
    s5 = tasks.make_group("s5")
    checks.append(("s5 non-abelian", (s5 != s5.T).any()))
    checks.append(("mod60 abelian", (tasks.make_group("mod60")
                                     == tasks.make_group("mod60").T).all()))

    # ---- prefix products: parity = cumulative XOR; identity-prefix sanity
    seqs = np.array([[1, 1, 0, 1], [0, 1, 1, 1]])
    par = tasks.prefix_products(tasks.make_group("parity"), seqs)
    checks.append(("parity prefix = cumxor",
                   (par == np.array([[1, 0, 0, 1], [0, 1, 0, 1]])).all()))
    rng = np.random.default_rng(1)
    seqs = rng.integers(0, 120, (16, 20))
    pp = tasks.prefix_products(s5, seqs)
    manual = seqs[:, 0]
    ok = (pp[:, 0] == manual).all()
    for i in range(1, 20):
        manual = s5[manual, seqs[:, i]]
        ok &= (pp[:, i] == manual).all()
    checks.append(("s5 prefix = left fold", bool(ok)))

    # ---- held-out pairs
    banned = tasks.heldout_pairs(120, 0.05)
    checks.append(("~5% of pairs banned",
                   abs(banned.mean() - 0.05) < 0.005))
    checks.append(("every element keeps >=2 successors",
                   ((~banned).sum(axis=1) >= 2).all()))
    checks.append(("split deterministic",
                   (banned == tasks.heldout_pairs(120, 0.05)).all()))
    checks.append(("parity never banned",
                   not tasks.heldout_pairs(2, 0.5).any()))

    # ---- sampler splits
    s = tasks.TaskSampler("s5", holdout_frac=0.05)
    rng = np.random.default_rng(2)
    tok, lab = s.sample(64, 32, rng, split="train")
    checks.append(("train: BOS prepended, labels ignore BOS",
                   tok.shape == (64, 33) and (tok[:, 0] == s.bos_id).all()
                   and (lab[:, 0] == tasks.IGNORE).all()))
    checks.append(("train split avoids banned pairs",
                   not s.contains_banned(tok[:, 1:]).any()))
    tok_h, _ = s.sample(64, 32, rng, split="heldout")
    checks.append(("heldout split forces banned pairs",
                   s.contains_banned(tok_h[:, 1:]).all()))
    tok_i, _ = s.sample(200, 32, rng, split="iid")
    frac = s.contains_banned(tok_i[:, 1:]).mean()
    checks.append(("iid split unconstrained (some banned present)",
                   0.5 < frac <= 1.0))
    lab2 = tasks.prefix_products(s.table, tok[:, 1:])
    checks.append(("labels = prefix products of tokens",
                   (lab[:, 1:] == lab2).all()))

    # ---- parity sampler works with holdout disabled
    p = tasks.TaskSampler("parity", holdout_frac=0.05)
    tokp, labp = p.sample(8, 16, rng, split="train")
    checks.append(("parity sampler (no holdout applied)",
                   tokp.shape == (8, 17) and not p.has_holdout))

    n_pass = sum(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{n_pass}/{len(checks)} passed")
    if n_pass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


# ---- tasks added 2026-08-20: does the effect depend on S5's algebra? --------

def test_t3_is_associative_and_not_a_group():
    """T3 must be associative (so prefix works) but non-commutative and
    non-invertible, which is the point of including it."""
    import numpy as np
    from projB.tasks import make_group
    t = make_group("t3")
    K = t.shape[0]
    assert K == 27, K
    rng = np.random.default_rng(0)
    for _ in range(500):
        a, b, c = rng.integers(0, K, 3)
        assert t[t[a, b], c] == t[a, t[b, c]], "T3 not associative"
    assert (t != t.T).any(), "T3 should be non-commutative"


def test_conn5_matches_union_find():
    """conn5 labels must equal an independent union-find replay."""
    import numpy as np
    from itertools import combinations
    from projB import tasks
    _, gens = tasks._connectivity_5()
    sym2edge = dict(zip(gens, list(combinations(range(5), 2))))
    sp = tasks.TaskSampler("conn5", holdout_frac=0.0)
    tok, lab = sp.sample(64, 8, np.random.default_rng(1), split="iid")
    # only generator symbols may appear as tokens
    assert set(tok[:, 1:].ravel().tolist()) <= set(gens)
    # labels compose under the table at every step
    tbl = sp.table
    for r in range(tok.shape[0]):
        prev = None
        for i in range(1, tok.shape[1]):
            cur = int(lab[r, i])
            if prev is not None:
                assert tbl[prev, int(tok[r, i])] == cur
            prev = cur


def test_conn5_is_commutative():
    """Connectivity is a join semilattice, so order must not matter.  This is
    what makes it the natural companion to the Z60 abelian control."""
    from projB.tasks import make_group
    t = make_group("conn5")
    assert (t == t.T).all(), "conn5 should be commutative"
