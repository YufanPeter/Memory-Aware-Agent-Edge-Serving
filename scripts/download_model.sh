#!/usr/bin/env bash
# Download a GGUF model into models/. Usage: scripts/download_model.sh [repo] [file]
set -euo pipefail
REPO=${1:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}
FILE=${2:-qwen2.5-1.5b-instruct-q4_k_m.gguf}
cd "$(dirname "$0")/.."
mkdir -p models
hf download "$REPO" "$FILE" --local-dir models
shasum -a 256 "models/$FILE"
