#!/bin/bash
# Extras batch (2026-07-11, cluster freed up): three parallel chains, staggered
# 120s so util-aware pick_gpu spreads them across idle GPUs.
#   X (film):    batch-MATCHED (256) FiLM 3-seed — de-confounds the batch-128
#                s0 run AND multi-seeds the H-B1 film falsification
#   Y (scale):   15m + 50m at c=2 (3 seeds) + baseline s0 anchors — does scale
#                fix the reach ceiling / keep the sequential default?
#   Z (breadth): mod60 abelian control (base + c=2), s5 c=2 n_max=64 OOD-range
#                run, then a5 wavefront probes (cross-group mechanism)
set -euo pipefail
cd ${REPO}
L=${REPO}/pbs_logs

COMMON="--n-min:4:--curriculum-frac:0.6:--n-curr-start:4:--batch:256:--steps:40000:--lr:1e-3:--skip-final-sweep:--resume"
mk() {  # $1=task $2=preset $3=extra(colon,may be empty) $4=nmax $5=seed $6=outdir
  echo "--task:$1:--preset:$2${3:+:$3}:--n-max:$4:$COMMON:--seed:$5:--out:runs/$6"
}
LOG2="--loop-schedule:log_n:--log-c:2"

# ---- chain X: film batch-256 3-seed ----
RX="$(mk s5 5m --film 32 0 s5_film_s0)@$(mk s5 5m --film 32 1 s5_film_s1)@$(mk s5 5m --film 32 2 s5_film_s2)"
jx=$(qsub -N xf_train -o "$L/xf_train.out" -l walltime=20:00:00 \
     -v RUNS="$RX",MIN_FREE_MIB=20000 scripts/run_projB_train.pbs)
echo "X train: $jx"
prev=$jx
for d in s5_film_s0 s5_film_s1 s5_film_s2; do
  j=$(qsub -N xf_ev_${d##*_} -o "$L/xf_ev_$d.out" -l walltime=6:00:00 -W depend=afterany:$prev \
      -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=10000 \
      scripts/run_projB_eval.pbs)
  echo "X eval $d: $j"; prev=$j
done

sleep 120

# ---- chain Y: scale (15m then 50m) ----
RY1="$(mk s5 15m "$LOG2" 32 0 s5_15m_logc2_s0)@$(mk s5 15m "$LOG2" 32 1 s5_15m_logc2_s1)@$(mk s5 15m "$LOG2" 32 2 s5_15m_logc2_s2)@$(mk s5 15m "" 32 0 s5_15m_base_s0)"
RY2="$(mk s5 50m "$LOG2" 32 0 s5_50m_logc2_s0)@$(mk s5 50m "$LOG2" 32 1 s5_50m_logc2_s1)@$(mk s5 50m "$LOG2" 32 2 s5_50m_logc2_s2)@$(mk s5 50m "" 32 0 s5_50m_base_s0)"
jy=$(qsub -N xs_train1 -o "$L/xs_train1.out" -l walltime=20:00:00 \
     -v RUNS="$RY1",MIN_FREE_MIB=20000 scripts/run_projB_train.pbs)
echo "Y train1 (15m): $jy"
jy2=$(qsub -N xs_train2 -o "$L/xs_train2.out" -l walltime=20:00:00 -W depend=afterany:$jy \
     -v RUNS="$RY2",MIN_FREE_MIB=20000 scripts/run_projB_train.pbs)
echo "Y train2 (50m): $jy2"
prev=$jy2
k=1
for d in s5_15m_logc2_s0 s5_15m_logc2_s1 s5_15m_logc2_s2 s5_15m_base_s0 s5_50m_logc2_s0 s5_50m_logc2_s1 s5_50m_logc2_s2 s5_50m_base_s0; do
  j=$(qsub -N xs_ev$k -o "$L/xs_ev_$d.out" -l walltime=6:00:00 -W depend=afterany:$prev \
      -v ARGS="--task:s5:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=10000 \
      scripts/run_projB_eval.pbs)
  echo "Y eval $d: $j"; prev=$j; k=$((k+1))
done

sleep 120

# ---- chain Z: breadth + a5 probes ----
RZ="$(mk mod60 5m "" 32 0 mod60_base_s0)@$(mk mod60 5m "$LOG2" 32 0 mod60_logc2_s0)@$(mk s5 5m "$LOG2" 64 0 s5_logc2_n64_s0)"
jz=$(qsub -N xb_train -o "$L/xb_train.out" -l walltime=20:00:00 \
     -v RUNS="$RZ",MIN_FREE_MIB=20000 scripts/run_projB_train.pbs)
echo "Z train: $jz"
prev=$jz
declare -A ZT=( [mod60_base_s0]=mod60 [mod60_logc2_s0]=mod60 [s5_logc2_n64_s0]=s5 )
k=1
for d in mod60_base_s0 mod60_logc2_s0 s5_logc2_n64_s0; do
  j=$(qsub -N xb_ev$k -o "$L/xb_ev_$d.out" -l walltime=6:00:00 -W depend=afterany:$prev \
      -v ARGS="--task:${ZT[$d]}:--ckpt:runs/$d/ckpt_final.pt:--out:runs/$d/loops_vs_length.json",MIN_FREE_MIB=10000 \
      scripts/run_projB_eval.pbs)
  echo "Z eval $d: $j"; prev=$j; k=$((k+1))
done
k=1
for d in a5_logc2_s0 a5_base_s0; do
  j=$(qsub -N xb_pr$k -o "$L/xb_pr_$d.out" -l walltime=6:00:00 -W depend=afterany:$prev \
      -v ARGS="--ckpt:runs/$d/ckpt_final.pt:--task:a5:--n:16:--out:runs/$d/probe_n16.json",MIN_FREE_MIB=10000 \
      scripts/run_projB_probe.pbs)
  echo "Z probe $d: $j"; prev=$j; k=$((k+1))
done
echo "=== full queue ==="
qstat -u CLUSTER_USER | tail -n +6
