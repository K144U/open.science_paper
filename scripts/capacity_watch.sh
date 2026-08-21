#!/bin/bash
# Keep our PBS backlog non-empty so freed cores never sit idle.
#
# WHAT THIS IS *NOT*.  It is not a race to grab cores the moment they free.
# PBS already does that: our units request ncpus=1, and the scheduler starts a
# queued job as soon as a core is available. Polling faster than the scheduler
# cycle (scheduler_iteration = 600s) would win nothing.
#
# THE ACTUAL GAP.  PBS can only start work we have ALREADY SUBMITTED. The core
# slate is finite, so when the backlog drains, freed cores go to other users
# while we have nothing queued. This watcher keeps a reserve of queued units
# topped up, so there is always something for the scheduler to start.
#
# It draws from `orchestrate.sh topup`: extra SEEDS on arms that already exist,
# ordered by scientific value, with the mean-matched jitT19/fixT24 pair
# interleaved so a partial top-up grows both arms equally. It never invents a
# new experimental condition, so nothing it submits needs pre-registration.
#
# SAFETY. Every submission goes through orchestrate.sh, which skips a unit that
# already has output AND one that is already in flight (the qstat guard added
# 2026-08-20). So this cannot double-submit, and re-running it is always safe.
#
# Usage:
#   bash scripts/capacity_watch.sh                 # background it
#   tail -f /tmp/capacity_watch.log
#   grep -E 'SUBMIT|DRAINED|DOWN' /tmp/capacity_watch.log
#
# Env: LOW_WATER (refill below this many queued trainers), HIGH_WATER (refill
# up to this many), INTERVAL seconds, STAGES to draw from.
set -uo pipefail

LOG="${CAP_WATCH_LOG:-/tmp/capacity_watch.log}"
INTERVAL="${CAP_WATCH_INTERVAL:-180}"
HOST="${CAP_WATCH_HOST:-jiit}"
REPO="${CAP_WATCH_REPO:-${REPO}}"
LOW_WATER="${LOW_WATER:-8}"
HIGH_WATER="${HIGH_WATER:-14}"
STAGES="${CAP_WATCH_STAGES:-stoch huginn topup}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" >> "$LOG"; }

say "capacity watch started: refill to $HIGH_WATER when queued trainers < $LOW_WATER (every ${INTERVAL}s, stages: $STAGES)"

state="unknown"
last_report=""

while true; do
    # One ssh round-trip gets everything: our running/queued trainer counts
    # (excluding the _ev eval jobs, which are dependency-held and are not work
    # the scheduler can start early) and the node's free cores.
    out=$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" "
        cd $REPO 2>/dev/null || exit 9
        run=\$(qstat -u \$(whoami) -w 2>/dev/null | awk 'NR>5 && \$10==\"R\" {print \$4}' | grep -vc '_ev\$')
        que=\$(qstat -u \$(whoami) -w 2>/dev/null | awk 'NR>5 && \$10==\"Q\" {print \$4}' | grep -vc '_ev\$')
        asg=\$(pbsnodes jiit-gpu01 2>/dev/null | awk '/resources_assigned.ncpus/{print \$3}')
        echo \"\$run \$que \$asg\"
    " 2>/dev/null)

    if [ -z "$out" ]; then
        [ "$state" != "down" ] && { say "DOWN  (vpn or cluster unreachable)"; state=down; }
        sleep "$INTERVAL"; continue
    fi
    [ "$state" != "up" ] && { say "UP"; state=up; }

    running=$(echo "$out" | awk '{print $1}')
    queued=$(echo  "$out" | awk '{print $2}')
    assigned=$(echo "$out" | awk '{print $3}')
    : "${running:=0}" "${queued:=0}" "${assigned:=96}"
    free=$((96 - assigned))

    # Report state only when it changes, so the log stays readable.
    report="r=$running q=$queued free=$free"
    if [ "$report" != "$last_report" ]; then
        say "state $report"
        last_report="$report"
    fi

    # Refill when the backlog is thin. Note we top up on QUEUED depth, not on
    # free cores: if cores are free and we have queue depth, PBS is already
    # starting them and there is nothing for us to do.
    if [ "$queued" -lt "$LOW_WATER" ]; then
        need=$((HIGH_WATER - queued))
        say "SUBMIT backlog thin (q=$queued < $LOW_WATER), topping up by $need"
        res=$(ssh -o ConnectTimeout=30 -o BatchMode=yes "$HOST" \
              "cd $REPO && bash scripts/orchestrate.sh --max $need $STAGES 2>&1 | tail -4" 2>/dev/null)
        if [ -n "$res" ]; then
            echo "$res" | sed "s/^/[$(ts)]   /" >> "$LOG"
            # When every stage reports nothing left to give, say so once.
            if echo "$res" | grep -q "submitted : 0"; then
                say "DRAINED nothing left to submit in: $STAGES"
            fi
        fi
    fi

    sleep "$INTERVAL"
done
