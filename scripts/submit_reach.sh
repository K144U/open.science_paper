#!/bin/bash
# E3 (ICLR strengthening, 2026-08-18) — is the REACH lottery undertraining or
# undercapacity?
#
# Paper v2 reports: budget tightness selects the algorithm class but only about
# half the seeds solve the full trained range (n=32 reached by 5/10 seeds at
# c=2, 5M, 40k steps).  Reviewers read that as "the intervention induces an
# efficient algorithm that often cannot do the task".  This batch asks whether
# reach is fixed by more compute, which would let us say: budget sets the
# ALGORITHM, capacity/training sets the COMPETENCE.
#
# Everything is held at the headline budget c=2 so all three arms compare
# against the existing 10-seed 5M/40k curve rather than splitting seeds across
# budgets.  10 seeds per arm keeps the Wilson intervals comparable.
#
#   Arm A  more training   5M,  80k steps, seeds 0-9   -> s5_logc2_long_s*
#   Arm B  more capacity   15M, 40k steps, seeds 3-9   -> s5_15m_logc2_s*
#                          (adds to existing s0,s1,s2 for a 10-seed arm)
#   Arm C  both            15M, 80k steps, seeds 0-9   -> s5_15m_logc2_long_s*
#
# CURRICULUM CONFOUND, handled: train.py ramps the length ceiling over
# curriculum_frac*steps.  At 40k steps frac=0.6 gives a 24k-step ramp.  The 80k
# arms therefore use frac=0.3 so the ramp stays 24k steps in ABSOLUTE terms and
# the only thing that changes is how long training continues after the
# curriculum tops out.  Using 0.6 at 80k would stretch the curriculum too and
# confound the two.
#
# Cost: ~4h (A) + ~3h (B) + ~9h (C) of GPU time, three staggered chains.
set -euo pipefail
cd ${REPO}
L=${REPO}/pbs_logs
mkdir -p "$L"

# 40k recipe, matching the existing c-sweep runs exactly
COMMON40="--n-min:4:--n-max:32:--curriculum-frac:0.6:--n-curr-start:4:--batch:256:--steps:40000:--lr:1e-3:--skip-final-sweep:--resume"
# 80k recipe: same 24k-step curriculum ramp, twice the total training
COMMON80="--n-min:4:--n-max:32:--curriculum-frac:0.3:--n-curr-start:4:--batch:256:--steps:80000:--lr:1e-3:--skip-final-sweep:--resume"

mkrun() {  # $1=preset $2=common $3=seed $4=outdir
  echo "--task:s5:--preset:$1:--loop-schedule:log_n:--log-c:2:$2:--seed:$3:--out:runs/$4"
}

join_runs() {  # join args with '@'
  local out=""
  for r in "$@"; do out="${out:+$out@}$r"; done
  echo "$out"
}

# queue one eval per run dir, chained after $1; echoes the last job id
chain_evals() {  # $1=prev_jobid $2=tag $3..=run dirs
  local prev=$1 tag=$2; shift 2
  local k=1
  for d in "$@"; do
    prev=$(qsub -N ${tag}_ev$k -o "$L/${tag}_ev_$d.out" -l walltime=4:00:00 \
        -W depend=afterany:$prev \
        -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=8000 \
        scripts/run_projB_eval.pbs)
    echo "  eval $d : $prev" >&2
    k=$((k+1))
  done
  echo "$prev"
}

# ===================== ARM A: 5M, 80k steps, seeds 0-9 =====================
A1=(); A2=(); dirsA=()
for s in 0 1 2 3 4; do A1+=("$(mkrun 5m "$COMMON80" $s s5_logc2_long_s$s)"); done
for s in 5 6 7 8 9; do A2+=("$(mkrun 5m "$COMMON80" $s s5_logc2_long_s$s)"); done
for s in 0 1 2 3 4 5 6 7 8 9; do dirsA+=("s5_logc2_long_s$s"); done

jA1=$(qsub -N reachA_t1 -o "$L/reachA_t1.out" -l walltime=20:00:00 \
      -v RUNS="$(join_runs "${A1[@]}")",MIN_FREE_MIB=10000 scripts/run_projB_train.pbs)
echo "armA train 1/2 (5m 80k s0-4): $jA1"
jA2=$(qsub -N reachA_t2 -o "$L/reachA_t2.out" -l walltime=20:00:00 -W depend=afterany:$jA1 \
      -v RUNS="$(join_runs "${A2[@]}")",MIN_FREE_MIB=10000 scripts/run_projB_train.pbs)
echo "armA train 2/2 (5m 80k s5-9): $jA2"
chain_evals "$jA2" reachA "${dirsA[@]}" >/dev/null

sleep 120

# ============= ARM B: 15M, 40k steps, seeds 3-9 (tops up to 10) =============
B=(); dirsB=()
for s in 3 4 5 6 7 8 9; do
  B+=("$(mkrun 15m "$COMMON40" $s s5_15m_logc2_s$s)")
  dirsB+=("s5_15m_logc2_s$s")
done
jB=$(qsub -N reachB_t1 -o "$L/reachB_t1.out" -l walltime=20:00:00 \
     -v RUNS="$(join_runs "${B[@]}")",MIN_FREE_MIB=15000 scripts/run_projB_train.pbs)
echo "armB train (15m 40k s3-9): $jB"
chain_evals "$jB" reachB "${dirsB[@]}" >/dev/null

sleep 120

# ===================== ARM C: 15M, 80k steps, seeds 0-9 =====================
C1=(); C2=(); dirsC=()
for s in 0 1 2 3 4; do C1+=("$(mkrun 15m "$COMMON80" $s s5_15m_logc2_long_s$s)"); done
for s in 5 6 7 8 9; do C2+=("$(mkrun 15m "$COMMON80" $s s5_15m_logc2_long_s$s)"); done
for s in 0 1 2 3 4 5 6 7 8 9; do dirsC+=("s5_15m_logc2_long_s$s"); done

jC1=$(qsub -N reachC_t1 -o "$L/reachC_t1.out" -l walltime=24:00:00 \
      -v RUNS="$(join_runs "${C1[@]}")",MIN_FREE_MIB=15000 scripts/run_projB_train.pbs)
echo "armC train 1/2 (15m 80k s0-4): $jC1"
jC2=$(qsub -N reachC_t2 -o "$L/reachC_t2.out" -l walltime=24:00:00 -W depend=afterany:$jC1 \
      -v RUNS="$(join_runs "${C2[@]}")",MIN_FREE_MIB=15000 scripts/run_projB_train.pbs)
echo "armC train 2/2 (15m 80k s5-9): $jC2"
chain_evals "$jC2" reachC "${dirsC[@]}" >/dev/null

echo "=== queued ==="
qstat -u CLUSTER_USER | tail -n +6 | wc -l
