#!/bin/bash
# E4 + E5 (ICLR strengthening, 2026-08-19), written after arXiv:2607.20594
# (Zhang et al.) appeared and reframed the paper around the boundary of their
# constant-speed frontier law.
#
# E4  THE 2x2 DE-CONFOUND.  Our budget intervention couples two ingredients, a
#     hard loop cap and a growing length curriculum, and the paper admits they
#     were never separated.  Reviewers will ask for this by name.  Four cells,
#     5 seeds each, everything else identical to the c=2 headline:
#
#                       curriculum ON            curriculum OFF (fixed n_max)
#       budget ON   s5_2x2_bud_cur_s*        s5_2x2_bud_nocur_s*
#       budget OFF  s5_2x2_nobud_cur_s*      s5_2x2_nobud_nocur_s*
#
#     "budget OFF" is the standard T=n contract; "curriculum OFF" trains at the
#     full length range from step 0 (--curriculum-frac 0).  The prediction our
#     dynamics story makes: the budget alone is not sufficient, because without
#     a growing curriculum the scan is never established and then broken.  If
#     bud_nocur matches bud_cur, our mechanism claim is wrong and we need to
#     know that before submission, not after.
#
# E5  MULTI-SEED TRAINING DYNAMICS.  The paper's most mechanistically
#     interesting result, that the scan is laid down first and compressed
#     later, currently rests on ONE run.  This trains 8 fresh c=2 seeds with
#     dense checkpoints so the slope trajectory can be reported as a band
#     rather than a line.  Reuses the existing 500-step checkpoint cadence.
#
# Cost: E4 is 20 x 5M/40k ~ 13 min each on an idle GPU; E5 is 8 more.  Both are
# cheap.  Queue position is the real cost, which is why they are chained behind
# nothing and submitted as independent arms.
#
# NOTE on resources: ncpus=2 and realistic walltimes.  train.py has no
# DataLoader workers and runs at ~105 steps/s for 5M, so a 40k run is ~7 min.
# Over-requesting either resource just delays scheduling (see
# paperB/E3_LAUNCH_2026-08-19.md).
set -euo pipefail
cd ${REPO}
L=${REPO}/pbs_logs
mkdir -p "$L"

# Shared with the c=2 headline so the 2x2 is comparable to it.
BASE="--task:s5:--preset:5m:--n-min:4:--n-max:32:--batch:256:--steps:40000:--lr:1e-3:--skip-final-sweep:--resume"
CUR="--curriculum-frac:0.6:--n-curr-start:4"     # ramp to n_max over 24k steps
NOCUR="--curriculum-frac:0"                      # full length range from step 0
BUD="--loop-schedule:log_n:--log-c:2"            # hard budget T = ceil(2 log2 n)
NOBUD="--loop-schedule:fixed_n"                  # standard contract T = n

mkrun() {  # $1=sched $2=curr $3=seed $4=outdir
  echo "$BASE:$1:$2:--seed:$3:--out:runs/$4"
}
join_runs() { local out=""; for r in "$@"; do out="${out:+$out@}$r"; done; echo "$out"; }

chain_evals() {  # $1=prev_jobid $2=tag $3..=run dirs
  local prev=$1 tag=$2; shift 2
  local k=1
  for d in "$@"; do
    prev=$(qsub -N ${tag}_ev$k -o "$L/${tag}_ev_$d.out" -l walltime=2:00:00 \
        -l select=1:ncpus=2 -W depend=afterany:$prev \
        -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=8000 \
        scripts/run_projB_eval.pbs)
    k=$((k+1))
  done
  echo "$prev"
}

echo "===== E4: the 2x2 de-confound (cap x curriculum), 5 seeds per cell ====="
for cell in "bud_cur:$BUD:$CUR" "bud_nocur:$BUD:$NOCUR" \
            "nobud_cur:$NOBUD:$CUR" "nobud_nocur:$NOBUD:$NOCUR"; do
  name="${cell%%:*}"; rest="${cell#*:}"; sched="${rest%%:*}"; curr="${rest#*:}"
  runs=(); dirs=()
  for s in 0 1 2 3 4; do
    runs+=("$(mkrun "$sched" "$curr" $s s5_2x2_${name}_s$s)")
    dirs+=("s5_2x2_${name}_s$s")
  done
  j=$(qsub -N e4_$name -o "$L/e4_$name.out" -l walltime=4:00:00 \
      -l select=1:ncpus=2 \
      -v RUNS="$(join_runs "${runs[@]}")",MIN_FREE_MIB=10000 \
      scripts/run_projB_train.pbs)
  echo "  $name (5 seeds): $j"
  chain_evals "$j" e4_$name "${dirs[@]}" >/dev/null
done

echo
echo "===== E5: training dynamics, 8 seeds with dense checkpoints ====="
# Seeds 10-17 so they cannot collide with the existing c=2 sweep (s0-s9).
runs=(); dirs=()
for s in 10 11 12 13 14 15 16 17; do
  runs+=("$(mkrun "$BUD" "$CUR" $s s5_dyn_c2_s$s)")
  dirs+=("s5_dyn_c2_s$s")
done
j=$(qsub -N e5_dyn -o "$L/e5_dyn.out" -l walltime=6:00:00 \
    -l select=1:ncpus=2 \
    -v RUNS="$(join_runs "${runs[@]}")",MIN_FREE_MIB=10000 \
    scripts/run_projB_train.pbs)
echo "  dynamics (8 seeds): $j"
chain_evals "$j" e5_dyn "${dirs[@]}" >/dev/null

echo
echo "=== queued ==="
qstat -u CLUSTER_USER | tail -n +6 | wc -l
