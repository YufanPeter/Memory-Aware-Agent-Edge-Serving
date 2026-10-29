#!/usr/bin/env bash
# Start llama-server for experiments. Override with environment variables:
#   MODEL     GGUF path         (default: models/qwen2.5-1.5b-instruct-q4_k_m.gguf)
#   CTX       total KV budget in tokens, shared by all slots (-c)   (default: 32768)
#   PARALLEL  number of slots; must be >= the runner's --k          (default: 8)
#   PORT      (default: 8080)
#   EXTRA     extra flags appended verbatim
#   LLAMA_SERVER  llama-server binary (default: llama-server on PATH; on Windows Git Bash,
#                 e.g. /d/llama.cpp-b11960/llama-server.exe)
#
# --cache-ram 0 turns off the host-RAM prompt cache so KV memory is only the GPU pool.
# --kv-unified makes all slots share one KV pool of CTX tokens (the memory budget).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
CTX=${CTX:-32768}
PARALLEL=${PARALLEL:-8}
PORT=${PORT:-8080}
mkdir -p results/slots
exec "${LLAMA_SERVER:-llama-server}" -m "$MODEL" -c "$CTX" -np "$PARALLEL" --port "$PORT" \
  -ngl 99 --kv-unified --cache-ram 0 --metrics --slot-save-path results/slots --no-webui \
  ${EXTRA:-}
