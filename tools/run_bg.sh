#!/usr/bin/env bash
# Run on the GPU host: run a command in the background, detached from the ssh session, with
# the llama.cpp/model environment the experiments use. Output goes to results/<name>.log.
#   ssh mem-aws 'bash ~/mem-aware/tools/run_bg.sh e0c uv run python bench/e0c_runtime.py'
# Several commands in sequence: pass one quoted string with "&&" or ";" and use bash -c.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export LLAMA_SERVER=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
export LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
export MODEL=${MODEL:-$HOME/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}
name=$1; shift
mkdir -p results
nohup "$@" >> "results/$name.log" 2>&1 < /dev/null &
echo "started $name (pid $!), log results/$name.log"
