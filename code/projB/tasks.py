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
    if name == "s4":
        # Non-abelian but SOLVABLE, order 24.  Completes the complexity ladder
        # Z60 (abelian) -> S4 (non-abelian, solvable) -> A5/S5 (non-solvable),
        # which isolates whether NON-SOLVABILITY matters or merely
        # non-commutativity.  Being a group, every element is invertible, so the
        # prefix product depends on the whole history and the state cannot
        # collapse: that is exactly the property t3 and conn5 turned out to lack.
        return _perm_group_table(sorted(permutations(range(4))))
    if name == "t3":
        return _transformation_monoid_3()[0]
    if name == "conn5":
        return _connectivity_5()[0]
    raise ValueError(f"unknown task {name!r}")


def generators_for(name: str):
    """Symbols that may appear as TOKENS, when that is a strict subset of the
    state space.  None means every state is also a legal symbol (the group
    tasks).  conn5 is the case that needs this: its states are the 52 partitions
    of 5 nodes, but only the 10 single-edge partitions are ever fed as input."""
    if name == "conn5":
        return _connectivity_5()[1]
    return None


TASKS = ("parity", "mod60", "s4", "s5", "a5", "t3", "conn5")


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
        self.generators = generators_for(task)
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
        if split == "train" and self.has_holdout and self.generators is None:
            seqs = np.empty((batch, n), dtype=np.int64)
            seqs[:, 0] = rng.integers(0, self.K, batch)
            for i in range(1, n):
                # vectorized choice among allowed successors of seqs[:, i-1]
                counts = np.array([len(self._allowed[a]) for a in seqs[:, i - 1]])
                pick = rng.integers(0, counts)
                seqs[:, i] = np.array(
                    [self._allowed[a][p] for a, p in zip(seqs[:, i - 1], pick)])
        else:
            if self.generators is not None:
                gen = np.asarray(self.generators)
                seqs = gen[rng.integers(0, len(gen), (batch, n))]
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


# ---------------------------------------------------------------- new tasks
# Added 2026-08-20 to test whether the budget effect depends on the algebraic
# structure of S5, which is the obvious "is this just permutations?" objection.
#
# Both are still PREFIX computations over an associative operation, so the same
# loops-vs-length instrument applies unchanged, but they differ from S5 in the
# two properties that matter:
#
#   t3     the full transformation monoid on 3 points.  Non-invertible, so not
#          a group at all, but still non-commutative.  Isolates whether
#          invertibility matters.
#   conn5  undirected connectivity on 5 nodes, presented as a stream of edges.
#          The state is the connected-components partition and the operation is
#          partition join, which IS commutative.  This is the task Merrill and
#          Sabharwal prove needs log depth, so it ties our empirics to that
#          theory directly, and its commutativity makes it the natural companion
#          to the Z60 abelian control.

def _transformation_monoid_3():
    """All 27 functions [3] -> [3], with composition (f then g)."""
    from itertools import product as _product
    elems = sorted(_product(range(3), repeat=3))       # elems[i] = tuple image
    index = {e: i for i, e in enumerate(elems)}
    K = len(elems)
    table = np.empty((K, K), dtype=np.int64)
    for i, f in enumerate(elems):
        for j, g in enumerate(elems):
            # apply f first, then g: (g o f)(x) = g[f[x]]
            table[i, j] = index[tuple(g[f[x]] for x in range(3))]
    return table, list(range(K))                       # every element generates


def _partition_key(lab):
    """Canonical form of a partition given ANY labelling of its blocks.

    Blocks are renumbered by first appearance, so two labellings of the same
    partition map to the same key.  Takes a labelling, not parent pointers:
    conflating the two silently merges blocks.
    """
    seen = {}
    out = []
    for v in lab:
        if v not in seen:
            seen[v] = len(seen)
        out.append(seen[v])
    return tuple(out)


def _connectivity_5():
    """States = partitions of 5 nodes; symbols = the 10 undirected edges.

    The operation is partition join, which is associative and commutative, so
    prefix connectivity is exactly a prefix computation over a semilattice.
    Only the 10 single-edge partitions are ever fed as tokens; the remaining
    states are reachable but never emitted, which is what `generators` encodes.
    """
    from itertools import combinations as _combinations

    def canon(p):
        return _partition_key(list(p))

    # enumerate all reachable partitions from the discrete one
    start = canon(range(5))
    states = {start: 0}
    order = [start]
    edges = list(_combinations(range(5), 2))

    def join(state, edge):
        lab = list(state)
        a, b = lab[edge[0]], lab[edge[1]]
        if a == b:
            return state
        lo, hi = min(a, b), max(a, b)
        merged = [lo if v == hi else v for v in lab]
        return _partition_key(merged)

    i = 0
    while i < len(order):
        st = order[i]; i += 1
        for e in edges:
            nxt = join(st, e)
            if nxt not in states:
                states[nxt] = len(order)
                order.append(nxt)
    K = len(order)
    # symbol j is the partition produced by edge j alone
    gens = [states[join(start, e)] for e in edges]
    table = np.empty((K, K), dtype=np.int64)
    for a, sa in enumerate(order):
        for b, sb in enumerate(order):
            # join two partitions: merge classes implied by both
            lab = list(sa)
            for x in range(5):
                for y in range(x + 1, 5):
                    if sb[x] == sb[y] and lab[x] != lab[y]:
                        old, new = lab[y], lab[x]
                        lab = [new if v == old else v for v in lab]
            table[a, b] = states[_partition_key(lab)]
    return table, gens
