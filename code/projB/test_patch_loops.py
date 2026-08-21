"""Ground-truth tests for the causal patching instrument.

The first cluster run of patch_loops.py returned settle(j) = T for every
position of a sequential-scan baseline, i.e. no signal at all.  The cause was
that settle_loop counted patches at the position's OWN site, and overwriting
position j's state at the last loop trivially changes output j because that is
what the readout head reads.  These tests pin the corrected behaviour: feed in
damage arrays whose underlying algorithm is known and require the measured
slope to match it.

    python code/projB/test_patch_loops.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB.patch_loops import settle_loop, fit_slope     # noqa: E402


def make(T, L, commit):
    """dmg[k,i,j] = 1 iff i < j and k < commit(j).

    Reads as: output j still depends on its upstream inputs until loop
    commit(j), and is committed after that.
    """
    d = np.zeros((T, L, L), dtype=np.float32)
    for j in range(L):
        for k in range(T):
            if k < commit(j):
                d[k, :j, j] = 1.0
    return d


def test_sequential_scan():
    """One position per loop must read as slope 1.0, speed 1.0."""
    s = settle_loop(make(16, 17, lambda j: j))
    sl = fit_slope(s)
    assert abs(sl - 1.0) < 0.05, "scan should give slope 1.0, got %s" % sl


def test_double_speed():
    """Two positions per loop must read as slope ~0.5, speed ~2."""
    s = settle_loop(make(16, 17, lambda j: max(1, j // 2)))
    sl = fit_slope(s)
    assert 0.4 < sl < 0.6, "expected ~0.5, got %s" % sl


def test_self_patch_does_not_leak():
    """Damage at i == j must be ignored; it is a readout artefact, not compute.

    This is the regression test for the bug the first canary run exposed.
    """
    d = make(16, 17, lambda j: j)
    for j in range(17):
        d[:, j, j] = 1.0                      # maximal self-damage everywhere
    sl = fit_slope(settle_loop(d))
    assert abs(sl - 1.0) < 0.05, "self-patch leaked into the measure: %s" % sl


def test_flat_front_is_not_a_scan():
    """A model that commits everything at once must not look sequential."""
    s = settle_loop(make(16, 17, lambda j: 2))
    sl = fit_slope(s)
    assert abs(sl) < 0.1, "constant commit should give slope ~0, got %s" % sl


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except AssertionError as e:
                fails += 1
                print("FAIL %s: %s" % (name, e))
    print("\n%s" % ("all patching-instrument tests pass" if not fails
                    else "%d FAILURE(S)" % fails))
    sys.exit(1 if fails else 0)
