#!/bin/bash
# Referee-proofing batch (2026-07-11): multi-seed the single-seed results +
# A5 cross-group replication + extra wavefront probes.
#   T1 s5 dose-response endpoints: c=1 s1,s2 + c=3 s1,s2
#   T2 a5 log-budget c=2: s0,s1,s2          (H-B2 on a second group)
#   T3 a5 baseline (fixed_n): s0,s1,s2      (T*=n on a second group)
#   T4 s5 baseline s1,s2 + loop_enc s1,s2   (multi-seed the 1-seed results)
# then 14 dense evals, then 3 probes (logc2_s1, logc1p5_s0, s5_base_s1).
# All serial afterany; every train has --resume; recipe = the locked one.
set -euo pipefail
cd ${REPO}
L=${REPO}/pbs_logs

COMMON="--preset:5m:--n-min:4:--n-max:32:--curriculum-frac:0.6:--n-curr-start:4:--batch:256:--steps:40000:--lr:1e-3:--skip-final-sweep:--resume"
mk() {  # $1=task $2=extra-args(colon,may be empty) $3=seed $4=outdir
  echo "--task:$1${2:+:$2}:$COMMON:--seed:$3:--out:runs/$4"
}

R1="$(mk s5 --loop-schedule:log_n:--log-c:1 1 s5_logc1_s1)@$(mk s5 --loop-schedule:log_n:--log-c:1 2 s5_logc1_s2)@$(mk s5 --loop-schedule:log_n:--log-c:3 1 s5_logc3_s1)@$(mk s5 --loop-schedule:log_n:--log-c:3 2 s5_logc3_s2)"
R2="$(mk a5 --loop-schedule:log_n:--log-c:2 0 a5_logc2_s0)@$(mk a5 --loop-schedule:log_n:--log-c:2 1 a5_logc2_s1)@$(mk a5 --loop-schedule:log_n:--log-c:2 2 a5_logc2_s2)"
R3="$(mk a5 "" 0 a5_base_s0)@$(mk a5 "" 1 a5_base_s1)@$(mk a5 "" 2 a5_base_s2)"
R4="$(mk s5 "" 1 s5_base_s1)@$(mk s5 "" 2 s5_base_s2)@$(mk s5 --loop-enc:sin 1 s5_le_s1)@$(mk s5 --loop-enc:sin 2 s5_le_s2)"

prev=""
i=1
for R in "$R1" "$R2" "$R3" "$R4"; do
  dep=""; [ -n "$prev" ] && dep="-W depend=afterany:$prev"
  j=$(qsub -N rp_train$i -o "$L/rp_train$i.out" -l walltime=20:00:00 $dep \
      -v RUNS="$R",MIN_FREE_MIB=5000 scripts/run_projB_train.pbs)
  echo "train$i: $j"
  prev=$j; i=$((i+1))
done

EVJOBS="s5:s5_logc1_s1 s5:s5_logc1_s2 s5:s5_logc3_s1 s5:s5_logc3_s2 a5:a5_logc2_s0 a5:a5_logc2_s1 a5:a5_logc2_s2 a5:a5_base_s0 a5:a5_base_s1 a5:a5_base_s2 s5:s5_base_s1 s5:s5_base_s2 s5:s5_le_s1 s5:s5_le_s2"
k=1
for td in $EVJOBS; do
  t="${td%%:*}"; d="${td##*:}"
  j=$(qsub -N rp_ev$k -o "$L/rp_ev_$d.out" -l walltime=4:00:00 \
      -W depend=afterany:$prev \
      -v ARGS="--task:$t:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=4000 \
      scripts/run_projB_eval.pbs)
  echo "eval $d: $j"
  prev=$j; k=$((k+1))
done

k=1
for d in s5_logc2_s1 s5_logc1p5_s0 s5_base_s1; do
  j=$(qsub -N rp_pr$k -o "$L/rp_pr_$d.out" -l walltime=4:00:00 \
      -W depend=afterany:$prev \
      -v ARGS="--ckpt:runs/$d/ckpt_final.pt:--task:s5:--n:16:--out:runs/$d/probe_n16.json",MIN_FREE_MIB=5000 \
      scripts/run_projB_probe.pbs)
  echo "probe $d: $j"
  prev=$j; k=$((k+1))
done
echo "=== queue ==="
qstat -u CLUSTER_USER | tail -n +6
