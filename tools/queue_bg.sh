#!/usr/bin/env bash
# Run on the GPU host: work through a queue of sweep specs one at a time, detached from ssh.
# The queue is results/queue.txt, one "configs/sweeps/<name>.yaml [sweep args]" per line, or
# "! <shell command>" (run from the repo root, e.g. a script that writes and appends the next
# sweep); append lines while it runs. When the queue is empty the runner exits and, with --shutdown, stops the
# instance (sudo shutdown -h now; EBS-backed, so this is a stop, not a terminate).
#   ssh mem-aws 'bash ~/mem-aware/tools/queue_bg.sh --shutdown'
#   ssh mem-aws 'echo configs/sweeps/x.yaml >> ~/mem-aware/results/queue.txt'
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" != "--worker" ]; then
  mkdir -p results
  nohup bash "$0" --worker "${1:-}" >> results/queue.log 2>&1 < /dev/null &
  echo "queue runner started (pid $!), log results/queue.log"
  exit 0
fi
shutdown_at_end=${2:-}
export PATH=$HOME/.local/bin:$PATH
export LLAMA_SERVER=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
export MODEL=${MODEL:-$HOME/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}
q=results/queue.txt
touch "$q"
while :; do
  line=$(head -n 1 "$q")
  [ -z "$line" ] && break
  tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"
  echo "$(date -u +%FT%TZ) start $line"
  if [ "${line:0:1}" = "!" ]; then
    bash -c "${line:1}" >> results/queue_cmds.log 2>&1 || echo "$(date -u +%FT%TZ) FAILED $line"
    echo "$(date -u +%FT%TZ) done $line"
    continue
  fi
  spec=${line%% *}
  # shellcheck disable=SC2086
  uv run python -m scripts.sweep $line >> "results/$(basename "$spec" .yaml).log" 2>&1 || echo "$(date -u +%FT%TZ) FAILED $line"
  echo "$(date -u +%FT%TZ) done $line"
done
echo "$(date -u +%FT%TZ) queue empty"
if [ "$shutdown_at_end" = "--shutdown" ]; then sudo shutdown -h now; fi
