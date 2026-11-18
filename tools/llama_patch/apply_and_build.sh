#!/usr/bin/env bash
# Apply the edgeserve llama-server patches to a llama.cpp b11960 checkout and rebuild llama-server.
#
#   tools/llama_patch/apply_and_build.sh [~/llama.cpp]
#
# Patches, applied in this order (the second is a diff against the first):
#   fork_slot.diff  POST /slots/{id}?action=fork {"src": <slot>, "n_tokens": <n>}: slot `id` shares
#                   the first n tokens of slot `src` (KV cells and cached-token list), no copy.
#   trim_slot.diff  POST /slots/{id}?action=trim {"ranges": [[p0, p1], ...]}: delete those token
#                   ranges from the slot's KV and cached tokens and shift later positions down
#                   (the context-shift operations, for arbitrary ranges), no recompute.
# No existing code path changes, so baselines that never call fork or trim behave exactly as on the
# unpatched build. Only the server target is rebuilt (minutes, not the CUDA kernels).
#
# The web UI is not embedded (LLAMA_USE_PREBUILT_UI=OFF): the UI assets for b11960 are no longer
# downloadable, and our servers run with --no-webui anyway.
set -euo pipefail
LLAMA=${1:-$HOME/llama.cpp}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$LLAMA"
git merge-base --is-ancestor 8642541 HEAD || { echo "expected llama.cpp b11960 (8642541)"; exit 1; }
for p in fork trim; do
  if grep -q "action == \"$p\"" tools/server/server-context.cpp; then
    echo "${p}_slot.diff already applied"
  else
    git apply "$HERE/${p}_slot.diff"
  fi
done
cmake -B build -DLLAMA_USE_PREBUILT_UI=OFF > /dev/null
cmake --build build -j"$(nproc)" --target llama-server
grep -q 'action == "trim"' tools/server/server-context.cpp && echo "llama-server rebuilt with fork + trim"
