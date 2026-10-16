#!/usr/bin/env bash
# Install llama.cpp. macOS: Homebrew (Metal). Linux: build from source with CUDA at a pinned commit.
set -euo pipefail
LLAMA_COMMIT=${LLAMA_COMMIT:-master}
if [[ "$(uname)" == "Darwin" ]]; then
  brew install llama.cpp
else
  cd "${LLAMA_DIR:-$HOME}"
  [[ -d llama.cpp ]] || git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp && git fetch && git checkout "$LLAMA_COMMIT"
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(nproc)" --target llama-server
  echo "Add $(pwd)/build/bin to PATH"
fi
llama-server --version
