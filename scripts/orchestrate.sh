#!/bin/bash
# Project B orchestrator: fan the whole ICLR experiment slate across every GPU.
#
# WHY THIS SHAPE.  Two things we measured on 2026-08-19 dictate the design:
#
#   1. The GPU node has 8 GPUs and 96 cpus.  GPUs 1-7 sat idle at 0-8% while
#      93/96 cpus were held by other users' CPU-bound chemistry jobs.  GPUs are
#      NOT the contended resource here; cpus are.  Measured on the node,
#      each python process sits at 97.8% CPU, i.e. under ONE core, so every job
#      asks for ncpus=1 and 1 GPU.  Asking for more is pure queue delay: 8 cpus
#      is what kept E3 queued, and 2 halved our concurrency again.
#
#   2. Short jobs backfill; long chains do not.  A 20h request has to wait for a
#      20h hole even when the work takes 10 minutes.  5M/40k trains at ~105
#      steps/s, about 7 minutes.  So every unit here is ONE training run with a
#      realistic walltime, submitted independently, rather than a 5-run chain.
#      That is what lets many units land on different GPUs at once.
#
# Usage:
#   scripts/orchestrate.sh --dry-run            # print the plan, submit nothing
#   scripts/orchestrate.sh all                  # everything
#   scripts/orchestrate.sh patch 2x2            # selected stages
#   scripts/orchestrate.sh --max N all          # cap concurrent submissions
#
# Stages:
#   patch    E2  causal damage cones on existing checkpoints (eval-only)
#   2x2      E4  cap x curriculum de-confound, 5 seeds x 4 cells
#   dyn      E5  training dynamics, 8 fresh seeds with dense checkpoints
#   extrap   E6  test-time loop scaling on existing checkpoints (eval-only)
#   shape    E9  curriculum-shape ablation: linear / exp / log / step
#   tasks    E7  task breadth: T3 monoid (non-group) and conn5 (connectivity)
#   stoch    E11 stochastic-depth training, the Huginn regime
#   fixdep   E12 FIXED-depth training, the contract real looped LMs use
#   huginn   E13 fixed MEAN with jitter: Huginn's actual contract
#
# Re-running is safe: every unit is skipped if its output already exists, and
# training units carry --resume.
set -uo pipefail
cd ${REPO}
L=pbs_logs; mkdir -p "$L"

DRY=0
MAX=0            # 0 = unlimited; otherwise stop after MAX submissions
STAGES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --max) MAX="$2"; shift ;;
    all) STAGES+=(patch 2x2 dyn extrap shape tasks stoch fixdep huginn) ;;
    patch|2x2|dyn|extrap|shape|tasks|stoch|fixdep|huginn) STAGES+=("$1") ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[ ${#STAGES[@]} -eq 0 ] && STAGES=(patch 2x2 dyn extrap shape tasks stoch fixdep huginn)

SUBMITTED=0
SKIPPED=0

want() {  # is stage $1 selected?
  local s
  for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

cap_reached() { [ "$MAX" -gt 0 ] && [ "$SUBMITTED" -ge "$MAX" ]; }

# submit_train <name> <walltime> <outdir> <colon-args>
# One training run, then its own eval chained behind it.  Independent of every
# other unit, which is the whole point.
submit_train() {
  local name=$1 wt=$2 out=$3 args=$4 task=${5:-s5}
  if [ -f "runs/$out/loops_vs_length.json" ]; then
    SKIPPED=$((SKIPPED+1)); return 0
  fi
  cap_reached && return 0
  if [ "$DRY" = 1 ]; then
    echo "  [train] $name  wt=$wt  -> runs/$out"
    SUBMITTED=$((SUBMITTED+1)); return 0
  fi
  local j
  j=$(qsub -N "$name" -o "$L/$name.out" -l walltime="$wt" -l select=1:ncpus=1 \
      -v RUNS="$args:--out:runs/$out",MIN_FREE_MIB=10000 \
      scripts/run_projB_train.pbs) || return 1
  qsub -N "${name}_ev" -o "$L/${name}_ev.out" -l walltime=2:00:00 \
      -l select=1:ncpus=1 -W depend=afterany:"$j" \
      -v ARGS="--task:$task:--ckpt:runs/$out/ckpt_final.pt:--out:runs/$out/loops_vs_length.json",MIN_FREE_MIB=8000 \
      scripts/run_projB_eval.pbs >/dev/null
  echo "  [train] $name : $j -> runs/$out"
  SUBMITTED=$((SUBMITTED+1))
}

# submit_eval <name> <walltime> <pbs> <colon-args> <outfile> [extra -v pairs]
submit_eval() {
  local name=$1 wt=$2 pbs=$3 args=$4 outfile=$5 extra=${6:-}
  if [ -f "$outfile" ]; then SKIPPED=$((SKIPPED+1)); return 0; fi
  cap_reached && return 0
  if [ "$DRY" = 1 ]; then
    echo "  [eval ] $name -> $outfile"
    SUBMITTED=$((SUBMITTED+1)); return 0
  fi
  local j
  j=$(qsub -N "$name" -o "$L/$name.out" -l walltime="$wt" -l select=1:ncpus=1 \
      -v ARGS="$args",MIN_FREE_MIB=8000${extra:+,$extra} "$pbs") || return 1
  echo "  [eval ] $name : $j -> $outfile"
  SUBMITTED=$((SUBMITTED+1))
}

BASE="--task:s5:--preset:5m:--n-min:4:--n-max:32:--batch:256:--steps:40000:--lr:1e-3:--skip-final-sweep:--resume"
CUR="--curriculum-frac:0.6:--n-curr-start:4"
NOCUR="--curriculum-frac:0"
BUD="--loop-schedule:log_n:--log-c:2"
NOBUD="--loop-schedule:fixed_n"

# ---------------------------------------------------------------- E2 patching
if want patch; then
  echo "=== E2: causal damage cones, EVERY checkpoint (eval-only) ==="
  # Each job is 1 to 3 minutes, so there is no reason to subsample: a large
  # causal sample is strictly better than a hand-picked dozen, and choosing
  # which checkpoints to patch is exactly the kind of decision a reviewer would
  # ask us to justify.  Task is inferred from the run-directory prefix, so the
  # A5 replication and the Z60 abelian control get causal evidence too.
  for ck in runs/*/ckpt_final.pt; do
    d=$(basename "$(dirname "$ck")")
    case "$d" in
      s5_*)    task=s5 ;;
      a5_*)    task=a5 ;;
      mod60_*) task=mod60 ;;
      *)       continue ;;
    esac
    submit_eval "pc_$d" 3:00:00 scripts/run_projB_patch.pbs \
      "--task:$task:--ckpt:runs/$d/ckpt_final.pt:--batch:64:--out:runs/$d/damage_cones.json" \
      "runs/$d/damage_cones.json" 'LENGTHS=8@16@24@32'
  done
fi

# ------------------------------------------------------------------- E4 2x2
if want 2x2; then
  echo "=== E4: 2x2 de-confound (cap x curriculum), 5 seeds per cell ==="
  for cell in "budcur:$BUD:$CUR" "budnocur:$BUD:$NOCUR" \
              "nobudcur:$NOBUD:$CUR" "nobudnocur:$NOBUD:$NOCUR"; do
    nm="${cell%%:*}"; rest="${cell#*:}"; sched="${rest%%:*}"; curr="${rest#*:}"
    for s in 0 1 2 3 4; do
      submit_train "e4_${nm}_s$s" 2:00:00 "s5_2x2_${nm}_s$s" \
        "$BASE:$sched:$curr:--seed:$s"
    done
  done
fi

# ------------------------------------------------------------- E5 dynamics
if want dyn; then
  echo "=== E5: training dynamics, 8 fresh seeds (dense checkpoints) ==="
  # Seeds 10-17 cannot collide with the existing c=2 sweep (s0-s9).
  for s in 10 11 12 13 14 15 16 17; do
    submit_train "e5_dyn_s$s" 2:00:00 "s5_dyn_c2_s$s" "$BASE:$BUD:$CUR:--seed:$s"
  done
fi

# --------------------------------------------------- E6 test-time loop scaling
if want extrap; then
  echo "=== E6: test-time loop scaling on existing checkpoints (eval-only) ==="
  # The pre-registered criterion was accuracy >=95% at 4x the training length,
  # and it FAILED.  Existing evals already cover 40/48/64, so the honest test of
  # the criterion as written is n=128 against n_max=32.  eval_loops.py sweeps a
  # dense T ladder to 2n, so this also grants far more loops than were ever
  # trained, which is the per-position rescue Zhang et al. report.
  # One length per job: qsub -v eats commas.
  for d in s5_logc2_s3 s5_logc2_s4 s5_logc2_s7 s5_logc2_s9 \
           s5_logc1p5_s3 s5_logc2_n64_s0 s5_base_s1; do
    [ -f "runs/$d/ckpt_final.pt" ] || { echo "  skip $d (no ckpt)"; continue; }
    submit_eval "ex128_$d" 4:00:00 scripts/run_projB_eval.pbs \
      "--task:s5:--ckpt:runs/$d/ckpt_final.pt:--lengths:128:--out:runs/$d/extrap_n128.json" \
      "runs/$d/extrap_n128.json"
  done
fi

# ------------------------------------------------- E9 curriculum-shape ablation
if want shape; then
  echo "=== E9: curriculum-shape ablation, 5 seeds per shape ==="
  # The only confound Limitations still concedes.  All four shapes start at the
  # same length, top out at the same step, and end at the same n_max, so this
  # isolates the SHAPE of the ramp and nothing else.  Budget held at c=2.
  for shp in linear exp log step; do
    for s in 0 1 2 3 4; do
      submit_train "e9_${shp}_s$s" 2:00:00 "s5_shape_${shp}_s$s"         "$BASE:$BUD:$CUR:--curriculum-shape:$shp:--seed:$s"
    done
  done
fi

# ------------------------------------------------------------ E7 task breadth
if want tasks; then
  echo "=== E7: task breadth, budget vs baseline on S4 ==="
  # NOTE 2026-08-20.  This stage originally used t3 (full transformation monoid)
  # and conn5 (graph connectivity).  Both were DISCARDED after measurement:
  #   t3     prefix products collapse to constant maps, 94% by n=8 and 100% by
  #          n=32, so the effective state space falls from 27 to 3.
  #   conn5  the graph is fully connected by the halfway point in 95% of n=16
  #          sequences, so the label freezes and the back half of the input is
  #          irrelevant.
  # Both therefore look "parallel" for algebraic reasons that say nothing about
  # the model, and t3 duly produced a spurious 12x speed growth.  The property
  # they lack is invertibility: in a GROUP the prefix product depends on the
  # whole history and cannot collapse.
  #
  # S4 is the right breadth axis instead: non-abelian but SOLVABLE, order 24.
  # With Z60 (abelian) and A5/S5 (non-solvable) it completes a complexity ladder
  # that isolates whether non-solvability matters or merely non-commutativity,
  # and it is the group Zhang et al. also use, so the comparison is direct.
  TBASE="--task:s4:--preset:5m:--n-min:4:--n-max:32:--batch:256:--steps:40000:--lr:1e-3:--skip-final-sweep:--resume"
  for s in 0 1 2 3 4; do
    submit_train "e7_s4_bud_s$s" 2:00:00 "s4_logc2_s$s"       "$TBASE:$BUD:$CUR:--seed:$s" s4
    submit_train "e7_s4_base_s$s" 2:00:00 "s4_base_s$s"       "$TBASE:$NOBUD:$CUR:--seed:$s" s4
  done
fi

# ------------------------------------------------- E11 stochastic-depth regime
if want stoch; then
  echo "=== E11: stochastic depth (the Huginn regime), 10 seeds ==="
  # E1 showed the pretrained bridge cannot be measured directly, so this tests
  # the same prediction on models we control.  Huginn samples recurrence depth
  # around a generous mean; our thesis says a SLACK schedule leaves loop use
  # lazy.  --loop-schedule uniform_n draws T ~ U[n/2, 2n], which is exactly that
  # regime, so the prediction is: frontier speed stays near 1.00 and flat, like
  # the T=n baseline, rather than growing as under a hard cap.  No new code.
  for s in 0 1 2 3 4 5 6 7 8 9; do
    submit_train "e11_unif_s$s" 2:00:00 "s5_unif_s$s"       "$BASE:--loop-schedule:uniform_n:$CUR:--seed:$s"
  done
fi

# ----------------------------------------------- E12 fixed-depth contract
if want fixdep; then
  echo "=== E12: FIXED-depth training, the contract real looped LMs use ==="
  # Huginn samples depth around a fixed mean, Ouro trains at fixed depth 4, and
  # Mixture-of-Recursions sets depth per token.  For all of them the loop budget
  # does not scale with input length: T(n) = T0.
  #
  # Our bound v(n) >= n/T(n) then reads v(n) >= n/T0, which grows LINEARLY in n,
  # more steeply than the n/log n of our own budget and unboundedly more than
  # the constant 1 of the standard T=n contract.  So the regime the field treats
  # as generous is in fact the harshest compression demand of the three.
  #
  # Prediction under test: frontier speed grows MORE steeply here than under
  # log budgets, approaching linear in n.
  # Five contracts, ten seeds each.  Varying T0 is the real test: the bound
  # predicts the binding-end speed v(n_max) = 32/T0, which spans a 4x range
  # across these settings, so this is a dose-response across contracts rather
  # than a single point.  Ten seeds because the 2x2 taught us that five can be
  # unrepresentative (its best cell read 5/5 where the pooled rate was 73%).
  #
  #   T0=6  -> predicted v(32) = 5.33
  #   T0=8  -> 4.00      T0=12 -> 2.67
  #   T0=16 -> 2.00      T0=24 -> 1.33
  for tc in 6 8 12 16 24; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      submit_train "e12_T${tc}_s$s" 2:00:00 "s5_fixT${tc}_s$s"         "$BASE:--loop-schedule:const:--T-const:$tc:$CUR:--seed:$s"
    done
  done
fi

# ------------------------------------------- E13 fixed mean with jitter (Huginn)
if want huginn; then
  echo "=== E13: fixed MEAN depth with jitter, Huginn's actual contract ==="
  # E11 and E12 differ in TWO ways at once (whether the mean scales with n, and
  # whether there is training-depth variance), so they cannot say which buys
  # tolerance of extra test-time loops.  This is the missing cell, and it is the
  # one that describes the real model:
  #
  #   contract                    mean scales   variance
  #   T = n                       yes           no
  #   T ~ U[n/2, 2n]   (E11)      yes           yes
  #   T = T0           (E12)      no            no
  #   T ~ U[T0/2, 2T0] (E13)      NO            YES   <- Huginn
  #
  # Measured so far: the two "mean scales" rows tolerate 2x loops at full
  # accuracy; fixed T collapses to chance by 4x.  If E13 tolerates, variance is
  # what buys test-time scaling and this explains why Huginn randomises depth.
  # If E13 collapses, scaling is what matters and test-time loop scaling should
  # degrade at long context for any fixed-mean model.
  for tc in 8 12; do
    for s in 0 1 2 3 4; do
      submit_train "e13_J${tc}_s$s" 2:00:00 "s5_jitT${tc}_s$s"         "$BASE:--loop-schedule:const_jitter:--T-const:$tc:$CUR:--seed:$s"
    done
  done
fi

echo
echo "=== orchestrator summary ==="
echo "submitted : $SUBMITTED"
echo "skipped   : $SKIPPED (output already present)"
if [ "$DRY" = 0 ]; then
  echo "queue now : $(qstat -u CLUSTER_USER 2>/dev/null | tail -n +6 | wc -l) jobs"
  echo "running   : $(qstat -u CLUSTER_USER 2>/dev/null | awk '$10=="R"' | wc -l)"
fi
