# Memory-Aware Agent Edge Serving

A layered, pluggable framework for evaluating how admission and scheduling of
concurrent **agent** workloads behave under a bounded memory budget.
First instantiation: **HotpotQA ReAct agent** → **FCFS scheduler** → **llama.cpp `llama-server`** with **Qwen2.5-1.5B-Instruct Q4_K_M**.

## Architecture

```
 experiments/run.py  ── assembly only: config → registry.build(...) per layer → runner.run_point
 L5 Evaluation   evaluators/ (task quality)   metrics/ (system metrics, from events.jsonl)   analysis/
 L4 Workload     workloads/  arrival process × task mix  → TaskSpec stream
 L3b Agent       agents/     orchestration × tools × memory   (three independent axes)
 L3a Task        tasks/      dataset, initial context, concrete tools, gold answer
 L2 Scheduler    schedulers/ admit_task / pick_next / on_complete hooks; exposes LLMClient to agents
 L1 Backend      backends/   inference runtime: start/stop/generate/probe/capabilities
 L0 Model        configs/models/*.yaml  (pure data)
 cross-cutting   core/ (types, registry, events, config)   monitors/ (memory, GPU, queue, runtime metrics)
```

Rules: layers talk only through `agentserve/core/types.py` dataclasses and the
abstract base of the layer below. Concrete classes register with
`@register(kind, name)` and are selected by `type:` in YAML. Nothing outside
`runner.py` knows which concrete implementation is in use.

| Agent axis | Interface | Built-in | Decides |
|---|---|---|---|
| Orchestration | `agents/orchestration/base.py: Orchestrator` | `react`, `single_shot` | loop structure, #steps, when to stop |
| Tools | `agents/tools/base.py: Tool`, `executor.py: ToolExecutor`, `delay.py` | per-task tools; delays `none/fixed/exponential/lognormal/uniform` | idle time between LLM calls |
| Memory | `agents/memory/base.py: Memory` | `full_history`, `sliding_window` | context sent per step → KV growth / prefix reuse |

## Setup (macOS / Metal)

```bash
uv sync
scripts/setup_llamacpp.sh        # brew install llama.cpp (Linux: CUDA build)
scripts/download_model.sh        # -> models/qwen2.5-1.5b-instruct-q4_k_m.gguf
scripts/download_hotpotqa.sh     # -> data/hotpotqa/distractor/validation-*.parquet
uv run pytest                    # layer contract tests (fake backend, no model needed)
```

## Running

```bash
uv run python experiments/run.py configs/experiments/smoke_fake.yaml   # whole pipeline, simulated backend
uv run python experiments/run.py configs/experiments/smoke.yaml        # real model, K ∈ {1,4}, 8 tasks
uv run python experiments/run.py configs/experiments/quality_check.yaml  # agent EM/F1, no scheduling effects
uv run python experiments/run.py configs/experiments/phase1_sweep.yaml   # K × arrival rate × seeds
uv run python experiments/run.py configs/experiments/phase1_context.yaml # KV budget × short/long context
uv run python experiments/run.py configs/experiments/phase1_prefix_cache.yaml  # cache_prompt on/off

uv run python analysis/summarize.py results/phase1_sweep      # table -> summary.csv
uv run python analysis/plots.py results/phase1_sweep          # figures/ (sweep curves, timelines, context)
```

Override any key: `--set workload.n_tasks=20 --set backend.ctx_size=8192`; `--limit N` runs the first N points; finished points are skipped (resume) unless `--force`.

### Outputs (`results/<experiment>/<point>/`)
- `events.jsonl`: every request enqueue/dispatch/done (prompt/cached/completion tokens, TTFT, prefill/decode ms), tool start/end, task arrive/start/done, monitor samples
- `tasks.jsonl`: per task status, prediction, gold, EM/F1, full step trace
- `meta.json`: llama-server command + build, capabilities, host, git rev; `config.yaml`: fully resolved config
- `summary.json`: completed/failed tasks, task latency p50/p95/p99, queueing, throughput, tok/s, prefix-cache hit ratio, peak memory, EM/F1, per-mix breakdown (fairness / slowdown)

## Measurement notes (llama.cpp build 11429 / d81235049)
- **The memory budget is `backend.ctx_size`.** With `kv_unified: true`, all `parallel` slots share one KV pool that is allocated at startup (≈ 28 KiB/token for this model, so 16k tokens ≈ 448 MiB). Process memory therefore stays roughly flat, and memory pressure shows up as context overflows, deferred requests and lost prefix-cache reuse, not as growing RSS.
- `cache_ram` (the host-RAM prompt cache, 8 GiB by default in this build) is set to 0, so memory can be attributed to the KV pool.
- On macOS, `phys_footprint` (includes Metal buffers) is the memory metric to use. RSS also counts the mmap'd model file.
- `/metrics` in this build has no KV-usage gauge. We use `requests_processing`, `requests_deferred` and token counters, plus the per-request `timings.cache_n` (cached prefix tokens).

## Scheduling policies (L2)

All policies share `schedulers/base.py`, which provides the queue, chunked dispatch, slot pinning, KV accounting, online estimators and the overhead counter. A policy is a priority function plus a quantum.

| # | `scheduler.type` | Preemptive? | Priority (smaller first) |
|---|---|---|---|
| baseline | `fcfs` | no | arrival order |
| 4 | `rr` | every `quantum` tokens | time it (re)entered the queue |
| 5 | `srpt` | yes | est. remaining time = uncached prefill + predicted remaining decode (learned per mix) |
| 6 | `mlfq` | yes | level = log2(1 + attained/quantum); chunk = quantum·2^level; skip-join by prefill cost |
| 7 | `las` | yes | **agent-level** attained service (whole task's backend time), then task arrival |
| 8 | `mem_aware` | yes | any of the above via `order:`; plus admission by projected KV ≤ (1−headroom)·`ctx_size`, with strict reservation |

**How preemption works.** llama.cpp cannot preempt internally. With a `quantum`, each dispatch generates at most `quantum` tokens and then returns to the queue. If another request is picked, the first one has been preempted. It resumes by continuing its partial output (`/completion` on the rendered template + prefix). Preemptions, the wait while paused, and *recomputed* prompt tokens (KV lost while paused) are recorded per request.

**KV retention (#10): `scheduler.retention`, combinable with any policy.**

| `type` | Meaning |
|---|---|
| `server` (default) | llama.cpp picks slots by prompt similarity and purges idle slots itself. This is the fixed cache policy baseline. |
| `lru` / `largest` / `tool_aware` | Each agent is pinned to a slot (`id_slot`). When a slot or pool space is needed, the scheduler evicts an idle agent's KV: the least recently used one, the largest one, or the one expected to stay in its tool call longest (learned per-tool mean). |
| `mode: erase` / `swap` | erase = recompute on return; swap = `/slots?action=save` → erase → `restore` on return |

Measured on llama.cpp build 11429:
- Without pinning, resumed chunks land on a different slot and re-prefill (~315 tokens per resume in a K=1 RR test).
- With pinning, chunked output is byte-identical to unchunked and recomputes 0 tokens.
- An overflow of *active* KV fails every in-flight request (HTTP 500), so `mem_aware` turns a failure cliff into queueing.

First real-model result (`phase2_memory`: 30 agents, 70% long-context, KV pool of 6144 tokens, 8 slots, 1 seed):

| scheduler | finished/30 | throughput (tasks/min) | prefix hit | prefill tokens | F1 |
|---|---|---|---|---|---|
| fcfs K=4 | 14 (16 overflow) | 8.4 | 0.39 | 34k | 0.18 |
| fcfs K=8 | 1 (29 overflow) | 1.0 | 0.24 | 6k | 0.03 |
| mem_aware + lru/erase | 30 | 7.0 | 0.11 | 118k (thrashing) | 0.39 |
| mem_aware + lru/swap | 30 | **13.1** | 0.70 | 43k | 0.39 |

Experiments: `phase2_policies.yaml` (#4–#8 × arrival rate), `phase2_retention.yaml` (#10), `phase2_memory.yaml` (the overflow cliff), `phase2_smoke.yaml` (8 tasks per policy).
Additional summary metrics: `preemptions_total`, `preempt_wait_total_s`, `recompute_tokens_total`, `reuse_miss_tokens_total` (KV lost between agent steps), `prefill_tokens_total`, `evictions_{erase,swap,stale}`, `slot_ops_ms_total`, `ctx_overflow_failures`, `policy_overhead_us_per_decision`, `max_kv_used_est`, and long/short `slowdown_p95` (fairness).

## Extending
Each extension is one file with `@register(...)` plus one YAML. External modules can be loaded with `plugins: [my.module]` in an experiment.

| Add a… | Subclass | Register as | Config |
|---|---|---|---|
| model | (none, data only) | (none) | `configs/models/x.yaml` |
| inference runtime | `backends.base.Backend` | `"backend"` | `configs/backends/x.yaml` |
| scheduling policy | `schedulers.base.Scheduler` (`priority`, `quantum_for`, `pick_next`) | `"scheduler"` | `configs/schedulers/x.yaml` |
| KV retention policy | `schedulers.retention.RetentionPolicy` (`choose_victim`) | `"retention"` | `scheduler.retention` |
| task / dataset | `tasks.base.Task` | `"task"` | `configs/tasks/x.yaml` |
| orchestration | `agents.orchestration.base.Orchestrator` | `"orchestrator"` | `agent.orchestration` |
| memory policy | `agents.memory.base.Memory` | `"memory"` | `agent.memory` |
| tool-latency model | `agents.tools.delay.DelayModel` | `"delay"` | `agent.tools.delay` |
| arrival process | `workloads.arrival.ArrivalProcess` | `"arrival"` | `workload.arrival` |
| quality metric | `evaluators.base.Evaluator` | `"evaluator"` | `tasks.<name>.evaluators` |
| monitor | `monitors.base.Monitor` | `"monitor"` | `monitors:` |

A new built-in module must also be added to `BUILTIN_MODULES` in `agentserve/core/registry.py`.

## AWS GPU
See `docs/aws_gpu.md`.
