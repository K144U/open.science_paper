#!/bin/bash
# E2, causal damage cones (ICLR strengthening, 2026-08-19).
#
# This is the experiment that decides the paper's central claim.  Behaviourally
# we can show the constant-speed frontier law of arXiv:2607.20594 fails under a
# hard budget, because v(n) = n/T*(n) grows with n.  What behaviour CANNOT do is
# say which law replaces it: an affine frontier (fixed speed, one-off start-up
# cost) fits our length range about as well as a sub-linear one.
#
# patch_loops.py measures settle(j), the loop after which no intervention can
# still change output j.  It is the causal twin of the probe's solve_loop(j),
# and its slope is directly comparable to the probe numbers already in the
# paper (1.02 baseline, 0.53 under budget).
#
#   affine / constant speed  ->  1/slope is the SAME at every n
#   growing speed            ->  1/slope RISES with n
#
# Run on baseline and budget checkpoints alike: the baselines are the control
# that must reproduce v = 1.00, which is also how we know the instrument works.
#
# Eval-only, no training.  Cost is O(T * n) forward passes per length, so a few
# minutes per checkpoint at these sizes.
set -euo pipefail
cd ${REPO}
L=${REPO}/pbs_logs
mkdir -p "$L"

# Controls first (must give v ~ 1.00), then the budget models under test.
# Chosen to span the reach range so the n-sweep is meaningful: budget seeds
# that solve n=32 are the only ones where growth can be measured at all.
CTRL="s5_base_s1 s5_base_s2 s5_15m_base_s1"
BUDG="s5_logc2_s3 s5_logc2_s4 s5_logc2_s7 s5_logc2_s8 s5_logc2_s9 \
      s5_logc1p5_s3 s5_logc1p5_s7 s5_logc2p5_s2 s5_15m_logc2_s1"

prev=""
k=0
for d in $CTRL $BUDG; do
  if [ ! -f "runs/$d/ckpt_final.pt" ]; then
    echo "SKIP $d (no ckpt_final.pt)" >&2
    continue
  fi
  k=$((k+1))
  dep=""
  [ -n "$prev" ] && dep="-W depend=afterany:$prev"
  # Lengths span the trained range; T is left at n so the full trajectory is
  # patchable (a budget model settles well before T=n, which is the point).
  prev=$(qsub -N patch$k -o "$L/patch_$d.out" -l walltime=3:00:00 \
      -l select=1:ncpus=2 $dep \
      -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--batch:64:--out:runs/$d/damage_cones.json",LENGTHS="8@16@24@32",MIN_FREE_MIB=8000 \
      scripts/run_projB_patch.pbs)
  echo "  patch $d : $prev"
done

echo "=== queued $k patching jobs ==="
qstat -u CLUSTER_USER | tail -n +6 | wc -l
