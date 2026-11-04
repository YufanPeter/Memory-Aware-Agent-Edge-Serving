# Memory-Aware Agent Edge Serving

CSE 291A course project. HotpotQA ReAct agents served by llama.cpp under a bounded KV budget.
Design: [docs/project_design.md](docs/project_design.md).

This repo currently holds the **Phase 1 agent scenario**: an instrumented, reproducible
HotpotQA agent workload replayed by a FIFO, fixed-concurrency runner.

## Setup

```bash
uv sync
scripts/download_hotpotqa.sh                 # data/hotpotqa/ (26 MB)
uv run python -m scripts.build_corpus        # data/corpus/: 66,581 unique paragraphs + BM25 index
uv run python -m scripts.build_splits        # data/splits/dev.jsonl (200), test.jsonl (500)
uv run pytest                                # no model needed
```

Models: Qwen2.5-7B-Instruct Q4_K_M for experiments (target: one 8 GB GPU, e.g. RTX 4060),
Qwen2.5-1.5B-Instruct Q4_K_M for quick local checks.

```bash
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q4_k_m*.gguf" --local-dir models
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
```

The 7B GGUF is split in two files; pass the first one (`-00001-of-00002.gguf`) as `MODEL`.

## Run

```bash
# 1. freeze a workload (questions x profiles x arrival times)
uv run python -m scripts.make_workload configs/workloads/smoke.yaml

# 2. start the server: CTX = total KV budget (tokens, shared by all slots), PARALLEL >= K
PARALLEL=4 CTX=32768 scripts/start_server.sh

# 3. replay with FIFO admission and K concurrent agents (worker k uses slot k)
uv run python -m edgeserve.runner --workload workloads/smoke.json --k 4 --out results/smoke_k4
```

Runner options: `--delay-scale` (multiply tool latency), `--no-cache` (cache_prompt=false),
`--no-pin` (server picks slots), `--server-pid` (sample its RSS), `--mock` (no server).

## Scenario

| profile | what | context | output |
|---|---|---|---|
| `react_short` | ReAct; search returns 3 sentences | short | short |
| `react_long` | ReAct; search returns the whole paragraph + 4 BM25-related ones | long | short |
| `react_explain` | ReAct; a 150-250 word explanation before finish | short | long |
| `oneshot_long` | the question's 10 paragraphs, answered in one call | long | short |
| `closedbook` | no retrieval; quality baseline only, not used in serving workloads | short | short |

Parameters, including tool-latency distributions, are in `configs/profiles.yaml`. Workload specs
(mix, arrival process, seed) are in `configs/workloads/`. Frozen workloads are in `workloads/`.

The prompt is **append-only**: one ChatML assistant turn that grows by Thought / Action /
Observation, sent raw to `/completion`. Each step's prompt starts with the previous prompt plus
its output, so prefix reuse depends only on the server's KV cache.

## Outputs (`results/<run>/`)

- `events.jsonl`: task arrive/admit/done; per request start/end with `prompt_n`, `cache_n`,
  `predicted_n`, `prompt_ms`, `predicted_ms`, TTFT, slot, stop type; tool start/end with delay;
  periodic samples (GPU memory, server RSS, `/metrics`).
- `tasks.jsonl`: per task status (`ok`, `forced`, `parse_fail`, `ctx_overflow`, `llm_error`,
  `timeout`), prediction, gold, EM/F1, supporting-fact recall, latency, queue wait, and the full
  trace. The trace includes the **segment ledger**: the context split into `shared_prefix`,
  `question`, `thought_action`, `obs_sf` (supporting facts), `obs_other`, `obs_meta` and `control`.
- `meta.json`: server build and model, workload hash, git rev, host. `summary.json`: headline numbers.

## Notes

- On macOS, `uv` venv `.pth` files get the macOS `hidden` flag and Python skips them, so the
  editable install is invisible. That is why scripts run as `python -m scripts.<name>` from the
  repo root.
