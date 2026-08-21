"""Check the paper's headline numbers against the artifacts that produced them.

Tables in the paper were pasted from generator scripts, so they are safe by
construction. The risk is the PROSE aggregates, which were typed by hand from
analysis output: "29 of 29", "87 of 95", "median 1.93x" and so on. A number that
drifts from its data is invisible to verify.py (which checks structure) and to
dashcheck.py (which checks style), so this closes the gap.

Each check regenerates the value from runs/ and runs_new/ and compares it to
what the .tex actually says. Run before any submission.

    python paperB/verify_numbers.py
"""
import glob
import json
import io
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_test import load_curve, analyse, MIN_RUNGS   # noqa: E402

TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.tex')

fails = []
checks = 0


def check(label, claimed, actual, tol=0.0):
    """Compare a claim against a regenerated value."""
    global checks
    checks += 1
    if isinstance(actual, float) and isinstance(claimed, float):
        ok = abs(actual - claimed) <= tol
    else:
        ok = actual == claimed
    status = 'ok  ' if ok else 'FAIL'
    print('%s %-46s paper=%-10s data=%s' % (status, label, claimed, actual))
    if not ok:
        fails.append(label)


def cone_rows():
    """(task, arm, growth) for every damage_cones.json."""
    out = []
    for p in glob.glob('runs/*/damage_cones.json') + \
            glob.glob('runs_new/*/damage_cones.json'):
        name = os.path.basename(os.path.dirname(p))
        try:
            j = json.load(open(p))
        except Exception:
            continue
        vs = {}
        for n, c in j.get('cones', {}).items():
            v = c.get('v_causal')
            if v:
                vs[int(n)] = v
        if len(vs) < 2:
            continue
        ks = sorted(vs)
        growth = vs[ks[-1]] / vs[ks[0]]
        task = ('S5' if name.startswith('s5_') else
                'A5' if name.startswith('a5_') else 'Z60')
        if 'base' in name or '2x2_nobud' in name:
            arm = 'baseline'
        elif 'film' in name or '_le_' in name:
            arm = 'untying'
        else:
            arm = 'budget'
        out.append((task, arm, growth, name))
    return out


def main():
    tex = io.open(TEX, encoding='utf-8').read()
    rows = cone_rows()
    if not rows:
        print('no damage_cones.json found; run from the repo root')
        return 1

    # --- causal cones: no unbudgeted checkpoint may GROW ------------------
    # "Flat" is <=1.10x, not |g-1|<0.02: several un-tying models sit slightly
    # BELOW 1.0, which is ladder resolution, not a departure from constant speed.
    # Z60 is discussed separately as the abelian control (its causal wavefront
    # is flat at slope ~0, i.e. no frontier), so it is not part of the 29.
    ctrl = [r for r in rows
            if r[1] in ('baseline', 'untying') and r[0] in ('S5', 'A5')]
    check('unbudgeted checkpoints that GROW (must be 0)', 0,
          sum(1 for r in ctrl if r[2] > 1.10))
    check('unbudgeted checkpoints counted', 29, len(ctrl))

    s5b = [r for r in rows if r[0] == 'S5' and r[1] == 'budget']
    grow = [r for r in s5b if r[2] > 1.10]
    m = re.search(r'(\d+) of (\d+) \$.Sfive\$ checkpoints show a causal', tex)
    if m:
        check('S5 budget grow, as written in the tex',
              (int(m.group(1)), int(m.group(2))), (len(grow), len(s5b)))
    if s5b:
        m = re.search(r'median growth \$([\d.]+)\times\$', tex)
        if m:
            check('S5 budget median causal growth', float(m.group(1)),
                  round(statistics.median([r[2] for r in s5b]), 2), 0.02)

    a5b = [r for r in rows if r[0] == 'A5' and r[1] == 'budget']
    a5c = [r for r in rows if r[0] == 'A5' and r[1] == 'baseline']
    check('A5 budget grow / A5 baselines flat', '3/3 and 0/3',
          '%d/%d and %d/%d' % (sum(1 for r in a5b if r[2] > 1.10), len(a5b),
                               sum(1 for r in a5c if r[2] > 1.10), len(a5c)))

    check('total checkpoints patched', 130, len(
        glob.glob('runs/*/damage_cones.json') +
        glob.glob('runs_new/*/damage_cones.json')))

    # --- pooled c=2 row: STANDARD linear curriculum only ------------------
    # Curriculum-shape variants (exp/log/step) are a different condition and
    # must not be pooled here; doing so once inflated this row from 27 to 31.
    c2 = []
    for pat in ('runs*/s5_logc2_s[0-9]', 'runs*/s5_2x2_budcur_s*',
                'runs*/s5_dyn_c2_s*', 'runs*/s5_shape_linear_s*'):
        for d in glob.glob(pat):
            pts = load_curve(d)
            if pts and len(pts) >= MIN_RUNGS:
                c2.append(analyse(pts))
    if c2:
        m = re.search(r'budget \$c\{=\}2\$ & (\d+) &', tex)
        if m:
            check('c=2 pooled seeds in tab:frontier', int(m.group(1)), len(c2))
        pct = round(100 * sum(1 for r in c2 if r['growth'] > 1.25) / len(c2))
        m = re.search(r'give a (\d+)\% rate of growing speed', tex)
        if m:
            check('c=2 pooled percent growing', int(m.group(1)), pct, 3)

    # --- Table 1 must reconcile with the prose that cites it --------------
    # The check that would have caught the reviewer's finding: the abstract
    # said "9 of 9" and "33 of 38" while the table's own rows summed to 19
    # unbudgeted and 56 budget runs. A reviewer adding up the table got
    # different numbers from the abstract, on the central quantitative claim.
    rows = re.findall(r'^(baseline|un-tying|budget)[^&]*& (\d+) &', tex, re.M)
    if rows:
        unb = sum(int(n) for a, n in rows if a in ('baseline', 'un-tying'))
        bud = sum(int(n) for a, n in rows if a == 'budget')
        m = re.search(r'over the (\d+) unbudgeted and (\d+)\s+budget-trained', tex)
        if m:
            check('tab:frontier rows vs 4.3 prose',
                  (int(m.group(1)), int(m.group(2))), (unb, bud))
        m = re.search(r'none of the (\d+)\s+unbudgeted ones', tex)
        if m:
            check('abstract unbudgeted count vs table', int(m.group(1)), unb)
        m = re.search(r'on (\d+) of (\d+) budget-trained runs', tex)
        if m:
            check('abstract budget count vs table', int(m.group(2)), bud)

    # --- no macro mangled into a control character by a bad edit ----------
    raw = io.open(TEX, 'rb').read()
    check('formfeed bytes in .tex (macro corruption)', 0, raw.count(b'\x0c'))

    print()
    if fails:
        print('%d of %d checks FAILED: %s' % (len(fails), checks, ', '.join(fails)))
        return 1
    print('all %d checks pass: the paper\'s prose aggregates match the data' % checks)
    return 0


if __name__ == '__main__':
    sys.exit(main())
