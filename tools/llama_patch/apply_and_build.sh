#!/usr/bin/env bash
# Apply the edgeserve fork-slot patch to a llama.cpp b11960 checkout and rebuild llama-server.
#
#   tools/llama_patch/apply_and_build.sh [~/llama.cpp]
#
# The patch only adds POST /slots/{id}?action=fork {"src": <slot>, "n_tokens": <n>}: slot `id`
# then shares the first n tokens of slot `src` (KV cells and cached-token list) without copying
# or recomputing them. No existing code path changes, so baselines that never call fork behave
# exactly as on the unpatched build. Only the server target is rebuilt (minutes, not the
# CUDA kernels).
set -euo pipefail
LLAMA=${1:-$HOME/llama.cpp}
PATCH="$(cd "$(dirname "$0")" && pwd)/fork_slot.diff"
cd "$LLAMA"
[ "$(git rev-parse --short=7 HEAD)" = "8642541" ] || { echo "expected llama.cpp b11960 (8642541)"; exit 1; }
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied"
else
  git apply "$PATCH"
fi
cmake --build build -j"$(nproc)" --target llama-server
grep -q 'action == "fork"' tools/server/server-context.cpp && echo "llama-server rebuilt with fork"
