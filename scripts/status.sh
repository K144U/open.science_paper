#!/bin/bash
# Project B experiment status: what is done, running, queued, and missing.
#
# Run on the cluster:   bash scripts/status.sh
# Or from a laptop:     ssh jiit 'cd ${REPO} && bash scripts/status.sh'
set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || cd ${REPO}

count_done() {  # $1=glob of run dirs, $2=required output file
  local n=0 t=0 d
  for d in $1; do
    [ -d "$d" ] || continue
    t=$((t+1))
    [ -f "$d/$2" ] && n=$((n+1))
  done
  echo "$n/$t"
}

echo "=============================== QUEUE ==============================="
if command -v qstat >/dev/null 2>&1; then
  R=$(qstat -u CLUSTER_USER 2>/dev/null | awk '$10=="R"' | wc -l)
  Q=$(qstat -u CLUSTER_USER 2>/dev/null | awk '$10=="Q"' | wc -l)
  H=$(qstat -u CLUSTER_USER 2>/dev/null | awk '$10=="H"' | wc -l)
  echo "running $R   queued $Q   held $H"
  if [ "$R" -gt 0 ]; then
    echo
    qstat -u CLUSTER_USER 2>/dev/null | awk 'NR<=5 || $10=="R"'
  fi
  echo
  echo "GPUs in use by us: $(qstat -u CLUSTER_USER 2>/dev/null | awk '$10=="R"' | wc -l) of 8"
else
  echo "(qstat unavailable, not on the cluster)"
fi

echo
echo "============================ EXPERIMENTS ============================"
printf '%-34s %10s  %s\n' 'experiment' 'complete' 'output'
printf '%-34s %10s  %s\n' '----------' '--------' '------'
printf '%-34s %10s  %s\n' 'E3-A reach 5M/80k'  "$(count_done 'runs/s5_logc2_long_s*' loops_vs_length.json)" 'loops_vs_length.json'
printf '%-34s %10s  %s\n' 'E3-B reach 15M/40k' "$(count_done 'runs/s5_15m_logc2_s*' loops_vs_length.json)" 'loops_vs_length.json'
printf '%-34s %10s  %s\n' 'E3-C reach 15M/80k' "$(count_done 'runs/s5_15m_logc2_long_s*' loops_vs_length.json)" 'loops_vs_length.json'
printf '%-34s %10s  %s\n' 'E4 2x2 bud+cur'     "$(count_done 'runs/s5_2x2_budcur_s*' loops_vs_length.json)" 'loops_vs_length.json'
printf '%-34s %10s  %s\n' 'E4 2x2 bud+nocur'   "$(count_done 'runs/s5_2x2_budnocur_s*' loops_vs_length.json)" 'loops_vs_length.json'
printf '%-34s %10s  %s\n' 'E4 2x2 nobud+cur'   "$(count_done 'runs/s5_2x2_nobudcur_s*' loops_vs_length.json)" 'loops_vs_length.json'
printf '%-34s %10s  %s\n' 'E4 2x2 nobud+nocur' "$(count_done 'runs/s5_2x2_nobudnocur_s*' loops_vs_length.json)" 'loops_vs_length.json'
printf '%-34s %10s  %s\n' 'E5 dynamics 8 seeds' "$(count_done 'runs/s5_dyn_c2_s*' loops_vs_length.json)" 'loops_vs_length.json'
printf '%-34s %10s  %s\n' 'E2 damage cones'    "$(count_done 'runs/s5_*' damage_cones.json)" 'damage_cones.json'
printf '%-34s %10s  %s\n' 'E6 extrapolation n=128' "$(count_done 'runs/s5_*' extrap_n128.json)" 'extrap_n128.json'

echo
echo "============================= LIVE LOGS ============================="
for f in $(ls -t pbs_logs/runs_*.log 2>/dev/null | head -6); do
  printf '%-42s %s\n' "$(basename "$f")" "$(tail -1 "$f" 2>/dev/null | sed 's/^ *//' | cut -c1-70)"
done

echo
echo "============================== NODE ================================="
if command -v pbsnodes >/dev/null 2>&1; then
  pbsnodes -aSj 2>/dev/null | grep -E 'vnode|gpu01' | head -3
fi
