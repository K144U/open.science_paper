"""Test the covered-fraction bound against measured frontier speeds.

THE ARGUMENT.  A frontier that resolves at most v positions per loop resolves at
most v*T positions in T loops.  Training under the budget T(n) = ceil(c log2 n)
supervises every position, so a model that is correct at length n must satisfy

    v(n) * T(n)  >=  n        i.e.       v(n)  >=  n / ceil(c log2 n)

The right-hand side is not a constant.  It grows like n / log n, without bound.
So for any fixed speed v there is a length beyond which a constant-speed
frontier cannot be correct: the constant-speed solution is not merely unlikely
under a log budget, it is INFEASIBLE over a growing length range.

This turns the paper's "laziness" story into a claim with a bound behind it, and
it makes a quantitative prediction rather than a directional one: the measured
speed should track n / T(n), not merely increase.

    python paperB/theory_check.py
"""
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_test import load_curve                             # noqa: E402


def budget_c(name):
    """Training budget coefficient c, read from the run directory name."""
    m = re.search(r'logc(\d+)(p(\d+))?[_/]', name + '/')
    if m:
        return float(m.group(1) + ('.' + m.group(3) if m.group(3) else ''))
    if '2x2_budcur' in name or '2x2_budnocur' in name or 'dyn_c2' in name \
            or 'shape_' in name:
        return 2.0
    return None


def v_required(n, c):
    return n / math.ceil(c * math.log2(n))


def main():
    rows = []
    for d in sorted(glob.glob('runs/s5_*') + glob.glob('runs_new/s5_*')):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        c = budget_c(name)
        if c is None:                     # baselines and un-tying: no budget
            continue
        pts = load_curve(d)
        if not pts or len(pts) < 4:
            continue
        for n, t in pts:
            rows.append((name, c, n, n / t, v_required(n, c)))

    if not rows:
        print('no budget runs with >=4 solved lengths found')
        return 1

    # Aggregate by (c, n): does measured v track the bound?
    print('Measured frontier speed against the covered-fraction bound')
    print('A correct solution must satisfy v(n) >= n / ceil(c log2 n).')
    print('=' * 74)
    print('%5s %5s %7s %11s %11s %9s' %
          ('c', 'n', 'runs', 'v measured', 'v required', 'ratio'))
    print('-' * 74)
    cells = {}
    for name, c, n, vm, vr in rows:
        cells.setdefault((c, n), []).append((vm, vr))
    ok = tot = 0
    for (c, n) in sorted(cells):
        vals = cells[(c, n)]
        vm = sum(v for v, _ in vals) / len(vals)
        vr = vals[0][1]
        tot += 1
        if vm >= vr - 0.15:               # allow for T* threshold granularity
            ok += 1
        print('%5g %5d %7d %11.2f %11.2f %9.2f' %
              (c, n, len(vals), vm, vr, vm / vr if vr else float('nan')))
    print('-' * 74)
    print('cells where measured speed meets or exceeds the bound: %d/%d'
          % (ok, tot))

    # The sharper test: the bound predicts HOW MUCH the speed must grow.
    print()
    print('Predicted vs measured growth across the trained range, per budget:')
    for c in sorted({c for _, c, _, _, _ in rows}):
        ns = sorted({n for _, cc, n, _, _ in rows if cc == c})
        if len(ns) < 2:
            continue
        lo, hi = ns[0], ns[-1]
        pred = v_required(hi, c) / v_required(lo, c)
        mlo = sum(v for _, cc, n, v, _ in rows if cc == c and n == lo)
        nlo = sum(1 for _, cc, n, _, _ in rows if cc == c and n == lo)
        mhi = sum(v for _, cc, n, v, _ in rows if cc == c and n == hi)
        nhi = sum(1 for _, cc, n, _, _ in rows if cc == c and n == hi)
        meas = (mhi / nhi) / (mlo / nlo)
        print('  c=%-4g n=%d to n=%-3d  predicted %.2fx   measured %.2fx'
              % (c, lo, hi, pred, meas))
    return 0


if __name__ == '__main__':
    sys.exit(main())
