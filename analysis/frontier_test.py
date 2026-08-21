"""Test our budget-trained models against Zhang et al.'s linear-frontier law.

Zhang et al. (arXiv:2607.20594) report that weight-tied looped transformers
always install a *linear computation frontier* solving v positions per loop,
with v priced by the training contract (v ~ n_tr/T_tr, exactly 1.00 under T=n).
Under that law a model is characterised by a single constant speed:

    M1  T* = n / v            constant-speed frontier   (Zhang's law, 1 param)

Two alternatives depart from it:

    M2  T* = n / v + b        affine frontier           (2 params)
    M3  T* = a * log2(n) + b  sub-linear                (2 params)

The linear *slope* we report in the paper cannot separate these, because over
n in [4, 32] all three are nearly straight.  Two things can:

1. Model comparison by AIC, which charges M2/M3 for their extra parameter.
2. The local frontier speed v(n) = n / T*(n).  This is the physically
   meaningful quantity and is fit-free.  Under M1 it is CONSTANT in n.  Any
   growth in v(n) within a single model falsifies the constant-speed law for
   that model, whatever curve one prefers to fit.

Usage:
    python frontier_test.py                 # all S5 runs under runs/
    python frontier_test.py runs/s5_logc2_s3 ...
"""
import glob
import json
import math
import os
import sys

MIN_RUNGS = 4          # fewer than this cannot distinguish curve families
GROWTH_FLAT = 1.10     # v(n) within +-10% counts as constant


def load_curve(run_dir):
    path = os.path.join(run_dir, 'loops_vs_length.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        j = json.load(fh)
    pts = []
    for n_str, cell in sorted(j.get('curve', {}).items(), key=lambda kv: int(kv[0])):
        ts = cell.get('T_star')
        if ts is not None and float(ts) > 0:
            pts.append((int(n_str), float(ts)))
    return pts


def sse_through_origin(xs, ys):
    """y = m*x, least squares. Returns (m, sse)."""
    sxx = sum(x * x for x in xs)
    if sxx == 0:
        return 0.0, sum(y * y for y in ys)
    m = sum(x * y for x, y in zip(xs, ys)) / sxx
    return m, sum((y - m * x) ** 2 for x, y in zip(xs, ys))


def sse_affine(xs, ys):
    """y = m*x + c, least squares. Returns (m, c, sse)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my, sum((y - my) ** 2 for y in ys)
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    c = my - m * mx
    return m, c, sum((y - (m * x + c)) ** 2 for x, y in zip(xs, ys))


def aic(sse, k, n):
    """Gaussian AIC with small-sample correction (AICc)."""
    if sse <= 0:
        sse = 1e-12
    val = n * math.log(sse / n) + 2 * k
    denom = n - k - 1
    return val + (2 * k * (k + 1) / denom if denom > 0 else 0.0)


def analyse(pts):
    ns = [float(n) for n, _ in pts]
    ts = [float(t) for _, t in pts]
    k = len(pts)

    m1_slope, m1_sse = sse_through_origin(ns, ts)
    m2_slope, m2_int, m2_sse = sse_affine(ns, ts)
    m3_a, m3_b, m3_sse = sse_affine([math.log(n, 2) for n in ns], ts)

    cands = {
        'M1 const-frontier': aic(m1_sse, 1, k),
        'M2 affine-frontier': aic(m2_sse, 2, k),
        'M3 sub-linear': aic(m3_sse, 2, k),
    }
    best = min(cands, key=cands.get)

    v_of_n = [(n, n / t) for n, t in pts]
    vs = [v for _, v in v_of_n]
    growth = max(vs) / min(vs) if min(vs) > 0 else float('inf')

    return {
        'k': k,
        'v_const': (1.0 / m1_slope) if m1_slope > 1e-9 else float('inf'),
        'm2_intercept': m2_int,
        'best': best,
        'aic': cands,
        'v_of_n': v_of_n,
        'growth': growth,
        'flat': growth <= GROWTH_FLAT,
    }


def arm_of(name):
    if 'base' in name:
        return 'baseline (T=n)'
    if 'film' in name or '_le_' in name:
        return 'un-tying (H-B1)'
    if 'logc' in name:
        return 'budget (log)'
    return 'other'


def main():
    # runs/ holds the original sweep; runs_new/ holds everything added on
    # 2026-08-19/20. Both are searched so a pooled estimate cannot silently
    # omit an arm.
    runs = sys.argv[1:] or sorted(
        d for d in glob.glob('runs/s5_*') + glob.glob('runs_new/s5_*')
        if os.path.isdir(d) and 'curr32' not in d)

    rows = []
    for run in runs:
        pts = load_curve(run)
        if not pts or len(pts) < MIN_RUNGS:
            continue
        r = analyse(pts)
        r['name'] = os.path.basename(run)
        r['arm'] = arm_of(r['name'])
        rows.append(r)

    print('Does the constant-speed frontier law hold?  v(n) = n / T*(n)')
    print('Zhang et al.: v is a constant of the model, set by the training contract.')
    print('=' * 96)
    print('%-24s %-16s %6s %8s %9s  %s' %
          ('run', 'arm', 'v(fit)', 'v growth', 'flat?', 'best model by AICc'))
    print('-' * 96)
    for r in sorted(rows, key=lambda x: (x['arm'], x['name'])):
        print('%-24s %-16s %6.2f %8.2fx %9s  %s' %
              (r['name'], r['arm'], r['v_const'], r['growth'],
               'FLAT' if r['flat'] else 'grows', r['best']))

    print('=' * 96)
    print('%-18s %5s %14s %14s %s' %
          ('arm', 'runs', 'median growth', 'n flat', 'best-model counts'))
    print('-' * 96)
    for arm in ('baseline (T=n)', 'un-tying (H-B1)', 'budget (log)'):
        sub = [r for r in rows if r['arm'] == arm]
        if not sub:
            continue
        gs = sorted(r['growth'] for r in sub)
        med = gs[len(gs) // 2] if len(gs) % 2 else 0.5 * (gs[len(gs)//2 - 1] + gs[len(gs)//2])
        nflat = sum(1 for r in sub if r['flat'])
        counts = {}
        for r in sub:
            counts[r['best']] = counts.get(r['best'], 0) + 1
        cs = ', '.join('%s:%d' % (k.split()[0], v) for k, v in sorted(counts.items()))
        print('%-18s %5d %13.2fx %8d/%-4d %s' % (arm, len(sub), med, nflat, len(sub), cs))

    print()
    print('Reading: a constant-speed frontier predicts growth = 1.00x and M1.')
    print('Growth well above 1 means the model solves MORE positions per loop at')
    print('longer lengths, which no single-speed frontier can do.')


if __name__ == '__main__':
    main()
