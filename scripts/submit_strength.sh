#!/bin/bash
# Strengthening batch (2026-07-11 night): A reliability curve (seeds 3-9 at
# c=1.5/2/2.5/3 -> P(efficient|c)), C mechanism depth (c=3 bimodal probe pair +
# 10-ckpt training-dynamics movie on c=2 seed0), D footnote hardening
# (15m_base s1,s2 + loop_enc s3,s4,s5). Two parallel chains, staggered 120s.
set -euo pipefail
cd ${REPO}
L=${REPO}/pbs_logs

COMMON="--n-min:4:--n-max:32:--curriculum-frac:0.6:--n-curr-start:4:--batch:256:--steps:40000:--lr:1e-3:--skip-final-sweep:--resume"
mkrun() {  # $1=preset $2=extra $3=seed $4=outdir
  echo "--task:s5:--preset:$1${2:+:$2}:$COMMON:--seed:$3:--out:runs/$4"
}
runs_for() {  # $1=ctag $2=cval $3..=seeds
  local ctag=$1 cval=$2; shift 2; local out=""
  for s in "$@"; do
    r=$(mkrun 5m "--loop-schedule:log_n:--log-c:$cval" "$s" "s5_logc${ctag}_s${s}")
    out="${out:+$out@}$r"
  done
  echo "$out"
}

# ============ CHAIN 1: c=1.5 + c=2 seeds -> evals -> dynamics movie ============
j=$(qsub -N rc1_t1 -o "$L/rc1_t1.out" -l walltime=20:00:00 \
    -v RUNS="$(runs_for 1p5 1.5 3 4 5 6 7 8 9)",MIN_FREE_MIB=10000 scripts/run_projB_train.pbs)
echo "rc1_t1 (c1.5 s3-9): $j"; prev=$j
j=$(qsub -N rc1_t2 -o "$L/rc1_t2.out" -l walltime=20:00:00 -W depend=afterany:$prev \
    -v RUNS="$(runs_for 2 2 3 4 5 6 7 8 9)",MIN_FREE_MIB=10000 scripts/run_projB_train.pbs)
echo "rc1_t2 (c2 s3-9): $j"; prev=$j
k=1
for c in 1p5 2; do for s in 3 4 5 6 7 8 9; do
  d=s5_logc${c}_s${s}
  j=$(qsub -N rc1_ev$k -o "$L/rc1_ev_$d.out" -l walltime=4:00:00 -W depend=afterany:$prev \
      -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=8000 \
      scripts/run_projB_eval.pbs)
  echo "rc1 eval $d: $j"; prev=$j; k=$((k+1))
done; done
k=1
for st in 000500 001000 002000 004000 008000 012000 016000 024000 032000 040000; do
  j=$(qsub -N rc1_pr$k -o "$L/rc1_pr_$st.out" -l walltime=6:00:00 -W depend=afterany:$prev \
      -v ARGS="--ckpt:runs/s5_logc2_curr32/ckpt_${st}.pt:--task:s5:--n:16:--out:runs/s5_logc2_curr32/probe_dyn_${st}.json",MIN_FREE_MIB=8000 \
      scripts/run_projB_probe.pbs)
  echo "rc1 dynprobe $st: $j"; prev=$j; k=$((k+1))
done

sleep 120

# ============ CHAIN 2: c=2.5 + c=3 seeds -> evals -> D -> bimodal probes ============
j=$(qsub -N rc2_t1 -o "$L/rc2_t1.out" -l walltime=20:00:00 \
    -v RUNS="$(runs_for 2p5 2.5 3 4 5 6 7 8 9)",MIN_FREE_MIB=10000 scripts/run_projB_train.pbs)
echo "rc2_t1 (c2.5 s3-9): $j"; prev=$j
j=$(qsub -N rc2_t2 -o "$L/rc2_t2.out" -l walltime=20:00:00 -W depend=afterany:$prev \
    -v RUNS="$(runs_for 3 3 3 4 5 6 7 8 9)",MIN_FREE_MIB=10000 scripts/run_projB_train.pbs)
echo "rc2_t2 (c3 s3-9): $j"; prev=$j
k=1
for c in 2p5 3; do for s in 3 4 5 6 7 8 9; do
  d=s5_logc${c}_s${s}
  j=$(qsub -N rc2_ev$k -o "$L/rc2_ev_$d.out" -l walltime=4:00:00 -W depend=afterany:$prev \
      -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=8000 \
      scripts/run_projB_eval.pbs)
  echo "rc2 eval $d: $j"; prev=$j; k=$((k+1))
done; done
RD="$(mkrun 15m "" 1 s5_15m_base_s1)@$(mkrun 15m "" 2 s5_15m_base_s2)@$(mkrun 5m --loop-enc:sin 3 s5_le_s3)@$(mkrun 5m --loop-enc:sin 4 s5_le_s4)@$(mkrun 5m --loop-enc:sin 5 s5_le_s5)"
j=$(qsub -N rc2_t3 -o "$L/rc2_t3.out" -l walltime=20:00:00 -W depend=afterany:$prev \
    -v RUNS="$RD",MIN_FREE_MIB=15000 scripts/run_projB_train.pbs)
echo "rc2_t3 (D: 15m_base s1,s2 + le s3-5): $j"; prev=$j
k=1
for d in s5_15m_base_s1 s5_15m_base_s2 s5_le_s3 s5_le_s4 s5_le_s5; do
  j=$(qsub -N rc2_evd$k -o "$L/rc2_ev_$d.out" -l walltime=4:00:00 -W depend=afterany:$prev \
      -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=8000 \
      scripts/run_projB_eval.pbs)
  echo "rc2 eval $d: $j"; prev=$j; k=$((k+1))
done
k=1
for d in s5_logc3_s1 s5_logc3_curr32; do
  j=$(qsub -N rc2_pr$k -o "$L/rc2_pr_$d.out" -l walltime=6:00:00 -W depend=afterany:$prev \
      -v ARGS="--ckpt:runs/$d/ckpt_final.pt:--task:s5:--n:16:--out:runs/$d/probe_n16.json",MIN_FREE_MIB=8000 \
      scripts/run_projB_probe.pbs)
  echo "rc2 bimodal probe $d: $j"; prev=$j; k=$((k+1))
done
echo "=== queue count ==="
qstat -u CLUSTER_USER | tail -n +6 | wc -l
