#!/bin/bash
# Pick the N "freest" GPU(s) on the shared jiit-gpu01 node and echo their
# indices for CUDA_VISIBLE_DEVICES.  PBS does NOT isolate GPUs here and the node
# is usually busy, so we choose live: among GPUs with >= MIN_FREE_MIB free, we
# prefer the LEAST-UTILIZED (idle co-resident), tie-broken by MOST free memory.
# This avoids landing on a roomy-but-contended GPU (a least-loaded-by-memory GPU
# at 60%+ util throttles small jobs badly).
#
#   export CUDA_VISIBLE_DEVICES=$(bash scripts/pick_gpu.sh 1)   # 1 freest GPU
#   MIN_FREE_MIB=30000 bash scripts/pick_gpu.sh 2               # 2 GPUs, >=30GB free each
#
# Requires each chosen GPU to have >= MIN_FREE_MIB free (default 18000 ~ Huginn bf16 + headroom).
set -euo pipefail

N="${1:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"

# rows as "util,free,idx"; keep only free>=MIN; sort by util ASC then free DESC.
mapfile -t rows < <(
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
               --format=csv,noheader,nounits \
    | awk -F',' -v m="$MIN_FREE_MIB" '{gsub(/ /,"",$0); if ($2+0 >= m) print $3","$2","$1}' \
    | sort -t, -k1,1n -k2,2nr
)

picked=()
for row in "${rows[@]}"; do
    idx="${row##*,}"                       # 3rd field = GPU index
    picked+=("$idx")
    [ "${#picked[@]}" -ge "$N" ] && break
done

if [ "${#picked[@]}" -lt "$N" ]; then
    echo "pick_gpu: only ${#picked[@]} GPU(s) with >=${MIN_FREE_MIB}MiB free, need $N" >&2
    exit 1
fi

( IFS=,; echo "${picked[*]}" )
