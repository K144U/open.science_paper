#!/bin/bash
# Keep the tunnel warm, and watch node capacity at the same time.
#
# Two jobs, one probe, so we are not opening two ssh sessions every minute:
#
#   1. TUNNEL KEEPALIVE.  The cluster is on a private address reachable only
#      through the VPN, and the client is user-side, so this cannot dial it back
#      up.  What it can do is generate traffic every INTERVAL seconds, which
#      prevents the idle-timeout disconnects we kept hitting during long
#      stretches of local analysis, and timestamp every up/down transition so
#      the pattern is diagnosable.
#
#   2. CAPACITY WATCH.  Our jobs request ncpus=1 and there are always many
#      queued, so PBS starts them automatically as cores free: nothing needs
#      doing when capacity appears.  What is worth knowing is (a) when
#      contention changes, and (b) whether we are FAILING to pick up free cores,
#      which would mean something is wrong with our own submissions rather than
#      with the cluster being busy.  The STALL line below is that alarm.
#
# Usage:
#   bash scripts/vpn_watch.sh          # background it
#   tail -f /tmp/vpn_watch.log
#   grep -E 'DOWN|STALL|FREED' /tmp/vpn_watch.log    # just the events
set -uo pipefail

LOG="${VPN_WATCH_LOG:-/tmp/vpn_watch.log}"
INTERVAL="${VPN_WATCH_INTERVAL:-45}"
HOST="${VPN_WATCH_HOST:-jiit}"
NODE="${VPN_WATCH_NODE:-jiit-gpu01}"
TOTAL_CORES="${VPN_WATCH_CORES:-96}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

state="unknown"
down_since=""
last_free=-1
stall_count=0

echo "[$(ts)] watch started: keepalive + capacity on $NODE, every ${INTERVAL}s" >> "$LOG"

while true; do
    # One ssh call gets liveness, node reservation and our running count.
    out=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$HOST" \
        "assigned=\$(pbsnodes $NODE 2>/dev/null | awk '/resources_assigned.ncpus/{print \$3}');
         run=\$(qstat -u CLUSTER_USER 2>/dev/null | awk '\$10==\"R\"' | wc -l);
         q=\$(qstat -u CLUSTER_USER 2>/dev/null | awk '\$10==\"Q\"' | wc -l);
         echo \"\$assigned \$run \$q\"" 2>/dev/null)

    if [ -n "$out" ]; then
        if [ "$state" != "up" ]; then
            if [ -n "$down_since" ]; then
                echo "[$(ts)] UP    (was down since $down_since)" >> "$LOG"
            else
                echo "[$(ts)] UP" >> "$LOG"
            fi
            state=up; down_since=""
        fi

        assigned=$(echo "$out" | awk '{print $1}')
        running=$(echo "$out" | awk '{print $2}')
        queued=$(echo "$out"  | awk '{print $3}')
        [ -z "$assigned" ] && assigned=$TOTAL_CORES
        free=$((TOTAL_CORES - assigned))

        # Report only meaningful capacity changes, so the log stays readable.
        if [ "$last_free" -lt 0 ] || [ $((free - last_free)) -ge 4 ] \
                                  || [ $((last_free - free)) -ge 4 ]; then
            echo "[$(ts)] FREED cores ${last_free} -> ${free} of ${TOTAL_CORES}   ours: ${running} running, ${queued} queued" >> "$LOG"
            last_free=$free
        fi

        # Alarm: cores are free, we have work queued, and we are not taking it.
        # PBS schedules on a cycle, so only complain if it persists.
        if [ "$free" -ge 2 ] && [ "$queued" -gt 0 ]; then
            stall_count=$((stall_count+1))
            if [ "$stall_count" -eq 20 ]; then
                echo "[$(ts)] STALL ${free} cores free and ${queued} of ours queued for ~15 min: check job resource requests" >> "$LOG"
            fi
        else
            stall_count=0
        fi
    else
        if [ "$state" != "down" ]; then
            down_since="$(ts)"
            echo "[$(ts)] DOWN" >> "$LOG"
            state=down
        fi
    fi
    sleep "$INTERVAL"
done
