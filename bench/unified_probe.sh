#!/usr/bin/env bash
# Decode speed with and without the unified KV cache, at the batch sizes E1 uses.
# Same model and flags as run_microbench.sh; only -kvu differs.
set -euo pipefail
: "${LLAMA_BIN:?}" "${MODEL:?}"
OUT=results/microbench/unified_probe
mkdir -p "$OUT"
for mode in split unified; do
  flag=""; [ "$mode" = unified ] && flag="-kvu"
  "$LLAMA_BIN/llama-batched-bench" -m "$MODEL" -fa on $flag -c $((32 * 2112)) -b 2048 -ub 512 \
    -npp 2048 -ntg 64 -npl 1,4,8,16,32 2>&1 | tee "$OUT/batched_$mode.txt"
done
