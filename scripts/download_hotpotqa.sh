#!/usr/bin/env bash
# Download HotpotQA distractor validation split (parquet) into data/hotpotqa/.
set -euo pipefail
cd "$(dirname "$0")/.."
hf download hotpotqa/hotpot_qa --repo-type dataset \
  --include "distractor/validation-*" --local-dir data/hotpotqa
find data/hotpotqa -name "*.parquet"
