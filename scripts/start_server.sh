#!/usr/bin/env bash
# Start llama-server for experiments. Override with environment variables:
#   MODEL     GGUF path         (default: models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf)
#   CTX       total KV budget in tokens, shared by all slots (-c)   (default: 32768)
#   PARALLEL  number of slots; must be >= the runner's --k          (default: 8)
#   PORT      (default: 8080)
#   CACHE_RAM host-RAM prompt cache in MiB; 0 keeps KV only in the GPU pool (default: 0)
#   SLOT_PATH where slot save/restore files go (default: /dev/shm/slots if /dev/shm exists,
#             else results/slots). /dev/shm is host RAM, i.e. a real offload tier.
#   THREADS   CPU threads for llama-server (default: 2; the GPU does the work, the rest of
#             the CPUs are left to the client and its tools)
#   CMD_OUT   if set, the full command line is written to this file (meta.json picks it up)
#   EXTRA     extra flags appended verbatim
#   LLAMA_SERVER  llama-server binary (default: llama-server on PATH; on Windows Git Bash,
#                 e.g. /d/llama.cpp-b11960/llama-server.exe)
#
# --kv-unified makes all slots share one KV pool of CTX tokens (the memory budget).
# --threads-http is raised with PARALLEL so idle keep-alive connections from agents waiting
# on tools cannot starve slot-control requests.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}
CTX=${CTX:-32768}
PARALLEL=${PARALLEL:-8}
PORT=${PORT:-8080}
CACHE_RAM=${CACHE_RAM:-0}
THREADS=${THREADS:-2}
if [ -z "${SLOT_PATH:-}" ]; then
  if [ -d /dev/shm ]; then SLOT_PATH=/dev/shm/slots; else SLOT_PATH=results/slots; fi
fi
mkdir -p "$SLOT_PATH"
cmd=("${LLAMA_SERVER:-llama-server}" -m "$MODEL" -c "$CTX" -np "$PARALLEL" --port "$PORT"
     -ngl 99 -fa on -b 2048 -ub 512 -t "$THREADS" --threads-http $((2 * PARALLEL + 8))
     --kv-unified --cache-ram "$CACHE_RAM" --metrics --slot-save-path "$SLOT_PATH" --no-webui)
# shellcheck disable=SC2206
cmd+=(${EXTRA:-})
if [ -n "${CMD_OUT:-}" ]; then echo "${cmd[*]}" > "$CMD_OUT"; fi
exec "${cmd[@]}"
