"""Define, count and cross-check the two populations the paper reports on.

The reviewer found that Table 1 does not reconcile with the abstract. The cause
is that the paper reports two DIFFERENT populations with the same words:

  BEHAVIOURAL   every run with a loops_vs_length.json curve having >= 4 solved
                lengths, since fewer cannot distinguish M1/M2/M3. One row per
                RUN.
  CAUSAL        every checkpoint with a damage_cones.json, which needs only 2
                solved lengths to give a growth ratio. One row per CHECKPOINT,
                and a run can contribute several checkpoints.

They are not interchangeable and neither is a subset of the other. This script
prints both with their inclusion rules, so the numbers in the paper can be
regenerated rather than remembered.

    python paperB/populations.py
"""
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_test import load_curve, analyse, MIN_RUNGS   # noqa: E402

GROWS = 1.10        # growth above this counts as "speed grows"


def arm_of(name):
    """Classify a run directory name into a reporting arm.

    The reported population is 5M, standard linear curriculum, S5. Scale
    variants (15M, 50M), longer curricula and the n=64 extension are EXCLUDED:
    E3 showed capacity changes the growth factor, so pooling capacities into one
    row is a confound. Pooling them once inflated the c=2 row from 27 to 61.
    """
    if 'jit' in name or 'fixT' in name or 'unif' in name:
        return None                      # contract study, reported separately
    if 'shape' in name and 'linear' not in name:
        return None      # exp/log/step ramps are the ablation; linear IS the
                         # standard recipe and belongs in the pooled rows
    if '2x2_budnocur' in name:
        return None      # budget WITHOUT curriculum: a 2x2 cell, not the recipe
    if '15m' in name or '50m' in name:
        return None                      # scale arms, reported in app:breadth
    if 'long' in name or 'n64' in name or 'curr32' in name:
        return None                      # longer-curriculum arms, separate
    if 'film' in name or '_le_' in name or name.endswith('_le'):
        return 'un-tying'
    if '2x2_nobud' in name or 'base' in name:
        return 'baseline'
    m = re.search(r'logc(\d)(?:p(\d))?', name)
    if m:
        c = m.group(1) + ('.' + m.group(2) if m.group(2) else '')
        return 'budget c=' + c
    if '2x2_bud' in name or 'dyn_c2' in name or 'shape_linear' in name:
        return 'budget c=2'   # all three are the c=2 recipe under another name
    return None


def behavioural():
    rows = {}
    for d in sorted(glob.glob('runs/s5_*') + glob.glob('runs_new/s5_*')):
        if not os.path.isdir(d) or 'curr32' in d:
            continue
        arm = arm_of(os.path.basename(d))
        if arm is None:
            continue
        pts = load_curve(d)
        if not pts or len(pts) < MIN_RUNGS:
            continue
        rows.setdefault(arm, []).append(analyse(pts))
    return rows


def causal():
    rows = {}
    for p in (glob.glob('runs/*/damage_cones.json') +
              glob.glob('runs_new/*/damage_cones.json')):
        name = os.path.basename(os.path.dirname(p))
        if not name.startswith('s5_'):
            continue
        arm = arm_of(name)
        if arm is None:
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        vs = {int(n): c['v_causal'] for n, c in j.get('cones', {}).items()
              if c.get('v_causal')}
        if len(vs) < 2:
            continue
        ks = sorted(vs)
        rows.setdefault(arm, []).append(vs[ks[-1]] / vs[ks[0]])
    return rows


ORDER = ['baseline', 'un-tying', 'budget c=1', 'budget c=1.5',
         'budget c=2', 'budget c=2.5', 'budget c=3']


def main():
    b = behavioural()
    c = causal()

    print("POPULATION A: behavioural fit")
    print("  one row per RUN, >= %d solved lengths (fewer cannot separate "
          "M1/M2/M3)" % MIN_RUNGS)
    print("-" * 78)
    print("%-16s %5s %9s %9s %8s  %s" %
          ("arm", "runs", "median v", "med growth", "flat", "selected model"))
    tot_unb = tot_bud = grow_bud = m1_bud = 0
    for arm in ORDER:
        rs = b.get(arm, [])
        if not rs:
            continue
        gs = [r['growth'] for r in rs]
        vs = [r['v_const'] for r in rs]
        flat = sum(1 for g in gs if g <= GROWS)
        cnt = {}
        for r in rs:
            cnt[r['best'].split()[0]] = cnt.get(r['best'].split()[0], 0) + 1
        print("%-16s %5d %9.2f %8.2fx %5d/%-3d %s" % (
            arm, len(rs), statistics.median(vs), statistics.median(gs),
            flat, len(rs),
            ', '.join('%s:%d' % kv for kv in sorted(cnt.items()))))
        if arm in ('baseline', 'un-tying'):
            tot_unb += len(rs)
        else:
            tot_bud += len(rs)
            grow_bud += sum(1 for g in gs if g > GROWS)
            m1_bud += cnt.get('M1', 0)
    print("-" * 78)
    print("  unbudgeted runs: %d      budget runs: %d" % (tot_unb, tot_bud))
    print("  budget runs whose speed GROWS: %d of %d" % (grow_bud, tot_bud))
    print("  budget runs selecting M1     : %d of %d" % (m1_bud, tot_bud))

    print()
    print("POPULATION B: causal patching")
    print("  one row per CHECKPOINT, >= 2 solved lengths (a growth ratio needs 2)")
    print("-" * 78)
    unb_n = unb_g = bud_n = bud_g = 0
    for arm in ORDER:
        gs = c.get(arm, [])
        if not gs:
            continue
        g = sum(1 for x in gs if x > GROWS)
        print("%-16s %5d checkpoints  median %5.2fx  grow %d/%d" % (
            arm, len(gs), statistics.median(gs), g, len(gs)))
        if arm in ('baseline', 'un-tying'):
            unb_n += len(gs); unb_g += g
        else:
            bud_n += len(gs); bud_g += g
    print("-" * 78)
    print("  unbudgeted checkpoints: %d, of which grow: %d" % (unb_n, unb_g))
    print("  budget checkpoints    : %d, of which grow: %d" % (bud_n, bud_g))

    print()
    print("The two populations differ because the causal instrument needs only")
    print("two solved lengths while a curve fit needs four, and because patching")
    print("is run per checkpoint rather than per run. Report them separately.")


if __name__ == '__main__':
    main()
