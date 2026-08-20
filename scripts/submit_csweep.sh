#!/bin/bash
# Finer c-sweep: c=1.5 and c=2.5, seeds 0/1/2, cloned from the c=2 recipe
# (s5, 5m, log_n schedule, curriculum 4->32 frac .6, batch 256, 40k steps,
# lr 1e-3, --skip-final-sweep, --resume).  Two train jobs chained afterany
# (no pick_gpu race), then 6 eval jobs chained serially.
set -euo pipefail
cd ${REPO}
L=${REPO}/pbs_logs

mk() {  # $1=tag $2=c $3=seed
  echo "--task:s5:--preset:5m:--loop-schedule:log_n:--log-c:$2:--n-min:4:--n-max:32:--curriculum-frac:0.6:--n-curr-start:4:--batch:256:--steps:40000:--lr:1e-3:--skip-final-sweep:--resume:--seed:$3:--out:runs/s5_logc$1_s$3"
}
RA="$(mk 1p5 1.5 0)@$(mk 1p5 1.5 1)@$(mk 1p5 1.5 2)"
RB="$(mk 2p5 2.5 0)@$(mk 2p5 2.5 1)@$(mk 2p5 2.5 2)"

jA=$(qsub -N c1p5_train -o "$L/c1p5_train.out" -l walltime=20:00:00 \
     -v RUNS="$RA",MIN_FREE_MIB=5000 scripts/run_projB_train.pbs)
echo "train c1p5 : $jA"
jB=$(qsub -N c2p5_train -o "$L/c2p5_train.out" -l walltime=20:00:00 \
     -W depend=afterany:$jA \
     -v RUNS="$RB",MIN_FREE_MIB=5000 scripts/run_projB_train.pbs)
echo "train c2p5 : $jB"

prev=$jB
for c in 1p5 2p5; do
  for s in 0 1 2; do
    d=s5_logc${c}_s${s}
    j=$(qsub -N ev_${c}_${s} -o "$L/ev_${d}.out" -l walltime=4:00:00 \
        -W depend=afterany:$prev \
        -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=4000 \
        scripts/run_projB_eval.pbs)
    echo "eval  $d : $j"
    prev=$j
  done
done
echo "=== queue ==="
qstat -u CLUSTER_USER
