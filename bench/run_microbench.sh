#!/usr/bin/env bash
# Microbenchmarks that calibrate costmodel/cost_model.py. Same script on M1 (Metal)
# and A10G (CUDA); only NPL changes.
#
# Usage:
#   LLAMA_BIN=~/llama.cpp/build/bin MODEL=~/models/qwen2.5-7b-instruct-q4_k_m.gguf \
#     bench/run_microbench.sh m1pro
#   NPL=1,2,4,8,16,32,64,128 LLAMA_BIN=... MODEL=... bench/run_microbench.sh a10g
#
# Flag spellings drift between llama.cpp builds: check `llama-bench --help` and
# `llama-batched-bench --help` on b11960 before the first run.
set -euo pipefail
: "${LLAMA_BIN:?set LLAMA_BIN to the llama.cpp build/bin directory}"
: "${MODEL:?set MODEL to the GGUF path}"

TAG=${1:-$(hostname -s)}
NPL=${NPL:-1,2,4,8,16,32}
NPP=${NPP:-2048}  # context per sequence in the batched decode test
NTG=${NTG:-64}
OUT="results/microbench/${TAG}-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

{
  echo "model: $MODEL ($(wc -c <"$MODEL") bytes)"
  "$LLAMA_BIN/llama-server" --version 2>&1 | head -5 || true
  uname -a
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
} >"$OUT/meta.txt"

for kv in f16 q8_0; do
  # 1. Prefill time vs prompt length: the re-prefill (drop) cost curve.
  "$LLAMA_BIN/llama-bench" -m "$MODEL" -fa on -ctk "$kv" -ctv "$kv" -r 3 -o json \
    -p 64,256,512,1024,2048,4096,8192 -n 0 >"$OUT/prefill_${kv}.json"

  # 2. Single-stream decode vs context depth: weight vs KV bandwidth.
  "$LLAMA_BIN/llama-bench" -m "$MODEL" -fa on -ctk "$kv" -ctv "$kv" -r 3 -o json \
    -p 0 -n 64 -d 0,1024,2048,4096,8000 >"$OUT/decode_depth_${kv}.json"

  # 3. Batched decode vs batch size: where decode stops scaling (the ridge).
  max_npl=$(tr ',' '\n' <<<"$NPL" | sort -n | tail -1)
  "$LLAMA_BIN/llama-batched-bench" -m "$MODEL" -fa on -ctk "$kv" -ctv "$kv" \
    -c $((max_npl * (NPP + NTG))) -b 2048 -ub 512 \
    -npp "$NPP" -ntg "$NTG" -npl "$NPL" 2>&1 | tee "$OUT/batched_${kv}.txt"
done

echo "results in $OUT"
