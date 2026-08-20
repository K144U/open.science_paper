"""Project B — synthetic finite-group state-tracking tasks (plan §4.3).

A task instance is a sequence of group elements g_1..g_n; the label at every
position i is the running prefix product g_1·g_2·...·g_i.  A model that solves
the task at position i must track the group state — the canonical state-
tracking setup (Barrington; FPRM 2606.18206; "(How) Do LMs Track State").

Groups (all realized as Cayley tables over element indices 0..K-1):
  parity  Z2   (K=2)    minimal solvable anchor
  mod60   Z60  (K=60)   abelian control
  s5      S5   (K=120)  non-solvable, NC1-complete — the headline task
  a5      A5   (K=60)   non-solvable, Merrill et al. comparability

Data hygiene (§4.3, arXiv 2606.07254): HELD-OUT TRANSITION PAIRS — a fixed
fraction of ordered pairs (a, b) never appears adjacently in training
sequences, so local transition patterns cannot be memorized; the heldout eval
split forces at least one such pair per sequence.  The pair set depends only
on (task, holdout_frac, split_seed), NOT on the run seed — every run of a
study sees the same split.

Convention: element indices are also the token ids (vocab = K + 1, the last
id is BOS).  Targets align with input positions (BOS's target is ignored).
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

IGNORE = -100  # CE ignore_index for the BOS position


# ------------------------------------------------------------------- groups

def _perm_group_table(perms: list) -> np.ndarray:
    """Cayley table over an explicit, sorted list of permutation tuples.

    Composition convention: (p ∘ q)(x) = p[q[x]] — element (i, j) of the
    table is the index of p_i ∘ p_j; the prefix product of g_1..g_k is then
    g_1 ∘ g_2 ∘ ... ∘ g_k applied right-to-left, matching left-fold below.
    """
    index = {p: i for i, p in enumerate(perms)}
    K = len(perms)
    table = np.empty((K, K), dtype=np.int64)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            table[i, j] = index[tuple(p[q[x]] for x in range(len(p)))]
    return table


def _is_even(perm: tuple) -> bool:
    inv = sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm))
              if perm[i] > perm[j])
    return inv % 2 == 0


def make_group(name: str) -> np.ndarray:
    """Cayley table [K, K] -> product index for a named group."""
    if name == "parity":
        return np.array([[0, 1], [1, 0]], dtype=np.int64)
    if name == "mod60":
        k = np.arange(60)
        return (k[:, None] + k[None, :]) % 60
    if name == "s5":
        return _perm_group_table(sorted(permutations(range(5))))
    if name == "a5":
        perms = sorted(p for p in permutations(range(5)) if _is_even(p))
        return _perm_group_table(perms)
    raise ValueError(f"unknown task {name!r}")


TASKS = ("parity", "mod60", "s5", "a5")


def prefix_products(table: np.ndarray, seqs: np.ndarray) -> np.ndarray:
    """Left-fold prefix products.  seqs [B, n] -> labels [B, n]:
    labels[:, i] = seqs[:,0] ∘ seqs[:,1] ∘ ... ∘ seqs[:,i]."""
    out = np.empty_like(seqs)
    acc = seqs[:, 0].copy()
    out[:, 0] = acc
    for i in range(1, seqs.shape[1]):
        acc = table[acc, seqs[:, i]]
        out[:, i] = acc
    return out


# ------------------------------------------------- held-out transition pairs

def heldout_pairs(K: int, frac: float, split_seed: int = 777) -> np.ndarray:
    """Boolean mask [K, K]: True = ordered pair (a, b) is held out of training.

    Guarantees every element keeps >= 2 allowed successors (never applies to
    parity: K=2 gets an empty mask regardless of frac).
    """
    banned = np.zeros((K, K), dtype=bool)
    if K < 4 or frac <= 0:
        return banned
    rng = np.random.default_rng(split_seed)
    n_ban = int(round(frac * K * K))
    cand = rng.permutation(K * K)
    for flat in cand:
        if n_ban == 0:
            break
        a, b = divmod(int(flat), K)
        if (~banned[a]).sum() > 2:  # keep >=2 successors after banning
            banned[a, b] = True
            n_ban -= 1
    return banned


class TaskSampler:
    """Sequence sampler honoring the held-out transition-pair split.

    split="train"    — sequences containing NO held-out pair
    split="iid"      — unconstrained sequences (in-distribution eval)
    split="heldout"  — sequences containing >= 1 held-out pair
    """

    def __init__(self, task: str, holdout_frac: float = 0.05,
                 split_seed: int = 777):
        self.task = task
        self.table = make_group(task)
        self.K = self.table.shape[0]
        self.banned = heldout_pairs(self.K, holdout_frac, split_seed)
        self.has_holdout = bool(self.banned.any())
        # per-element allowed successor lists for train-split sampling
        self._allowed = [np.flatnonzero(~self.banned[a]) for a in range(self.K)]

    @property
    def vocab_size(self) -> int:  # elements + BOS
        return self.K + 1

    @property
    def bos_id(self) -> int:
        return self.K

    def sample(self, batch: int, n: int, rng: np.random.Generator,
               split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        """(tokens [B, n+1] with BOS prepended, labels [B, n+1], BOS=IGNORE)."""
        if split not in ("train", "iid", "heldout"):
            raise ValueError(f"unknown split {split!r}")
        if split == "train" and self.has_holdout:
            seqs = np.empty((batch, n), dtype=np.int64)
            seqs[:, 0] = rng.integers(0, self.K, batch)
            for i in range(1, n):
                # vectorized choice among allowed successors of seqs[:, i-1]
                counts = np.array([len(self._allowed[a]) for a in seqs[:, i - 1]])
                pick = rng.integers(0, counts)
                seqs[:, i] = np.array(
                    [self._allowed[a][p] for a, p in zip(seqs[:, i - 1], pick)])
        else:
            seqs = rng.integers(0, self.K, (batch, n))
            if split == "heldout" and self.has_holdout:
                # force one held-out transition at a random interior position
                ba, bb = np.nonzero(self.banned)
                which = rng.integers(0, len(ba), batch)
                pos = rng.integers(0, n - 1, batch)
                rows = np.arange(batch)
                seqs[rows, pos] = ba[which]
                seqs[rows, pos + 1] = bb[which]
        labels = prefix_products(self.table, seqs)
        tokens = np.concatenate(
            [np.full((batch, 1), self.bos_id, dtype=np.int64), seqs], axis=1)
        labels = np.concatenate(
            [np.full((batch, 1), IGNORE, dtype=np.int64), labels], axis=1)
        return tokens, labels

    def contains_banned(self, seqs: np.ndarray) -> np.ndarray:
        """[B] bool: does each (BOS-free) sequence contain a held-out pair?"""
        return self.banned[seqs[:, :-1], seqs[:, 1:]].any(axis=1)
