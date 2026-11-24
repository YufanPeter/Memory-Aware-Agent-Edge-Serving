#!/usr/bin/env bash
# Selective KV step 3, 16k main experiment, unattended (docs/phase2_ideas.md 4.6-4.7):
# 1. capacity on the final code (the base's sets the load levels; selective KV's, added after the first
# full run, gives the peak-goodput comparison), 2. load levels, 3. the 21 main runs, 4. the reports.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m scripts.sweep configs/sweeps/p3_cap16k.yaml
uv run python -m analysis.p3_tune results/p3_cap16k --json results/p3_cap16k/tune.json
uv run python -m scripts.make_p3_main_workloads --from-runs results/p3_cap16k/base_r1.75
# The load levels follow the plan only if the capacity is in the range seen so far (tuning 48.5, smoke 56.4);
# anything else means something broke, so stop before the main runs.
cap=$(cat results/p3_main16k_capacity.txt)
python3 -c "import sys; c=float('$cap'); sys.exit(0 if 40 <= c <= 75 else 1)" \
    || { echo "capacity $cap tasks/min outside 40-75: stopping before the main runs" >&2; exit 1; }
uv run python -m scripts.sweep configs/sweeps/p3_main16k.yaml
uv run python -m analysis.p3_main results/p3_main16k --capacity "$(cat results/p3_main16k_capacity.txt)" \
    --json results/p3_main16k/main.json
