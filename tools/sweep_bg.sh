#!/usr/bin/env bash
# Run on the GPU host: start a sweep in the background, detached from the ssh session.
#   ssh mem-aws 'bash ~/mem-aware/tools/sweep_bg.sh configs/sweeps/<name>.yaml [sweep args]'
# Output: results/<config basename>.log (appended). Stop everything with
#   ssh mem-aws 'bash ~/mem-aware/tools/sweep_bg.sh --stop'
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--stop" ]; then
  pkill -f "scripts[.]sweep" || true
  pkill -f "edgeserve[.]runner" || true
  pkill -f "build/bin/llama[-]server" || true
  exit 0
fi
export PATH=$HOME/.local/bin:$PATH
export LLAMA_SERVER=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
export MODEL=${MODEL:-$HOME/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}
cfg=$1; shift
log=results/$(basename "$cfg" .yaml).log
mkdir -p results
nohup uv run python -m scripts.sweep "$cfg" "$@" >> "$log" 2>&1 < /dev/null &
echo "started $cfg (pid $!), log $log"
