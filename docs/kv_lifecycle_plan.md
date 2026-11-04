# Plan: agent-level KV lifecycle management under a memory budget

Working plan for the rest of the project (Phase 1 due 11/4, final ~12/9). It refines
`docs/project_design.md` §6–7 against the course requirements and what the code and
llama.cpp actually support. Hardware: `mem-aws`, an AWS g5.xlarge (A10G 23,028 MiB,
4 vCPU, 15 GiB RAM, `/dev/shm` 7.8 GiB), llama.cpp b11960 (commit 8642541, CUDA sm_86),
Qwen2.5-7B-Instruct Q4_K_M (4.36 GiB weights, 56 KiB/token f16 KV).

Revised 10/29 after two adversarial reviews (rubric/method and feasibility).

## 1. Question and claim

Course question: *how can admission and scheduling decisions improve completed agent
work while keeping memory use bounded?*

Our focus is one mechanism, studied in depth:

> While an agent waits on a tool, its KV sits idle in the GPU pool. At each tool-call
> boundary we decide whether that KV stays (retain), moves to a bounded host tier
> (offload), or is freed and recomputed later (drop). New agents are admitted only when
> their projected peak KV fits. We show **when** this beats tuned simple baselines, **by how
> much**, and **which part** (admission or KV lifecycle) the gain comes from.

- **Q1 (Phase 1).** In which regimes is the KV pool, not compute, the binding resource?
  First-principles cost model, calibrated by microbenchmarks (M0), then checked by E1–E5.
- **Q2 (Phase 2).** Inside those regimes, how much capacity does idle-KV management add,
  and how much of it is admission alone vs the KV lifecycle (2×2 decomposition)?
- **Q3 (Phase 2).** What does it cost (policy overhead, slot-op stalls), and are long
  tasks treated fairly?

## 2. Scope decisions

| Decision | Choice |
|---|---|
| Our policy | **One**: pressure-gated TTL at tool boundaries + peak-KV admission (§4). Everything else is a baseline, an ablation, or an analysis reference. |
| Main baselines | Six configurations (§4.3). Others go to one appendix operating point. |
| Precision | Fixed: f16 KV. q8_0 only as an appendix sensitivity check. |
| Runtime controls | Only llama-server HTTP: `id_slot`, `cache_prompt`, `/slots/{id}?action=save|restore|erase`, `-c`, `-np`, `--kv-unified`, `--cache-ram`. No C++ changes. |
| Target device | A **discrete small GPU plus host RAM** (e.g. the team's RTX 4060 8 GB laptop). The A10G stands in for it by capping `-c`, which the course text allows. Unified-memory devices have no separate host tier; for them only admission + drop apply, and we say so. |
| Memory budget | GPU KV pool = `-c` tokens. Host offload tier = bytes in `/dev/shm/slots`, capped at a multiple of the pool: {0, 0.5×, 1×, 2×} of `-c` bytes. Every comparison reports total memory (pool + host tier). |
| Memory metric | The pool is preallocated, so nvidia-smi peak is identical across policies. We report **occupied KV tokens** (peak, p95, mean, token·s integral) from KV-manager accounting, cross-checked with `/slots`; host-tier peak bytes; server RSS. |
| Phase 3 (selective KV) | Optional, only after Phase 2 results. Text-level only. |

## 3. System design

### 3.1 Architecture: keep slot pinning, add a KV manager

Under `--kv-unified` the memory budget is `-c`, not `-np`, so agents stay pinned: worker *k*
uses slot *k*. `-np` is held fixed (64) for all pinned and managed runs, once E0a shows that
-np 4 vs 64 changes throughput by <3% at K=4. A **KV manager** decides what each slot holds.

```
arrivals ─▶ admission (FIFO) ─▶ worker k (agent loop, slot k) ──▶ llama-server
                 ▲                    │ tool start / before generate      ▲
                 │                    ▼                                   │ /completion
                 └────────────── KV manager ── policy ──────────────────┘ /slots save|restore|erase
                     states per agent: active · idle-in-tool · offloaded · dropped · stale
```

### 3.2 Invariants (no deadlock, no thrash, fair admission)

1. **Admission reserves projected peak KV of every in-flight agent**, including offloaded
   and dropped ones. The projection: current tokens + expected remaining steps × mean
   tokens/step for the profile, using a per-profile p95 from Phase 1 traces.
2. **Ready (returning) agents have strict priority** over new admissions.
3. If a ready agent cannot fit, the manager immediately gives up idle KV, ignoring TTLs:
   stale slots first, then idle-in-tool agents in order of cheapest give-up cost *B*.
   If nothing is evictable, the lowest-progress waiting agent is dropped and re-queued
   (counted as an event; admission's p95 projection makes this rare).
4. **Admission is strict FIFO with head-of-line blocking.** A blocked long task is not
   bypassed, so no starvation; we report per-profile slowdown to show the cost.
5. At most one give-up per agent per tool call (no offload/restore ping-pong).
6. On task timeout, timers are cancelled and the slot is released in a `finally`, with
   erase retried with backoff if the slot is still busy.

**Release does not erase.** A finished task's slot keeps the shared system/few-shot prefix
(~1.4k tokens), which the next task on that worker reuses (95.9% cache hit in Phase 1).
The manager tracks it as `stale` and erases it only when it is the chosen victim, so the
managed system pays no cost the always-retain baseline avoids.

### 3.3 Hook points (yufan branch, as of 3bc14d2)

| Where | Hook | Action |
|---|---|---|
| `agent.py:80`, inside `_generate` after the `req_start` emit | `await kvm.ensure_resident(task, need_tokens)` | Restore if offloaded (dropped contexts re-prefill via the full-prompt re-send). Emits `kv_wait`, so restore and waiting show up in step latency. |
| after `agent.py:82` | `kvm.after_generate(task, res)` | S = `prompt_n + cache_n + predicted_n`. |
| after `agent.py:166` (`tool_start` emitted, delay known) | `await kvm.on_tool_start(task, step, tool, S, true_delay)` | Policy decision. `true_delay` reaches only the clairvoyant reference. |
| after `agent.py:171` (`tool_end`) | `kvm.on_tool_end(task)` | Prefetch: start the restore now instead of waiting for the next generate. |
| `runner.py:158`, `try/finally` | `await kvm.release(task)` | Mark the slot stale; delete its offload file. |
| `runner.py` before admit | `await kvm.admit(task)` | Invariants 1, 2, 4. |

Token counts before a request are estimated as last known tokens + new chars × chars/token
for the segment type (fitted from Phase 1 traces). Validation runs measure the estimate's
error with `POST /tokenize` and reconcile with `/slots` once per step.

### 3.4 Components to build

- `edgeserve/kvm.py`: KV manager (accounting, timers, admission, events).
- `edgeserve/policies.py`: policy interface and the configurations in §4.3.
- `edgeserve/llm.py`: `slot_save/restore/erase`, `slots()`, reading server
  `timings.save_ms/restore_ms`, on a **separate httpx client without keep-alive** so slot
  ops never queue behind idle streaming connections. Reference: `origin/tianyu`
  `backends/llamacpp.py:243-265`, rewritten to edgeserve style.
- Offload files: one fixed name per slot (`slot_<k>.bin`, overwritten on save). The
  manager runs on the same host and unlinks the file after a successful restore or on
  release, the only client-side file operation. Host-tier cap is enforced in accounting:
  if offloaded bytes + S × 56 KiB would exceed it, the choice becomes drop.
- A failed restore counts as a drop (re-prefill) and emits `slot_op ok=false`; never silent.
- `MockLLM`: pool capacity, overflow behaviour as measured in E0a, save/restore/erase,
  optional "slot op blocks all decode" mode. Unit tests: every ready agent eventually runs;
  ≤1 give-up per agent per tool call.
- Events: `kv_state` (task, slot, from→to, tokens), `slot_op` (op, bytes, server ms, wall
  ms, ok), `kv_wait`, `admit_wait`.
- **Experiment tooling** (built on the Mac against MockLLM before GPU runs):
  `scripts/sweep.py` (YAML grid → server restart → wait `/health` → runner →
  `results/<grid>/<point>/rep<i>`, resumable); steady-state throughput and goodput in
  `summarize()` (completions inside the 10%–90% window); `analysis/aggregate.py`
  (all summaries → CSV + CIs); `analysis/plots.py`; `analysis/kv_timeline.py`;
  `tools/mem_aws.sh pull` (rsync results back).

### 3.5 Fixes before high-concurrency runs

1. Per-task `try/finally` in the runner (today a non-timeout exception aborts the run).
2. Tool compute (`tools.call`) via `asyncio.to_thread`; log its duration; sleep
   `max(0, delay − compute)` so the idle gap equals the seeded delay.
3. `nvidia-smi` sampling off the event loop.
4. `start_server.sh`: `SLOT_PATH` (default `/dev/shm/slots`), `--threads-http
   $((2*PARALLEL+8))`, `-fa on`, `-b 2048 -ub 512`, `-t 2`, `--cache-ram 0` unless a config
   says otherwise; write the full command line to `results/<run>/server_cmd.txt` and copy
   it into `meta.json`.
5. `react_long` title-miss cap frozen at 5 paragraphs (what the code does,
   `tools.py:83-84`, and what produced the measured 3.4k p50); fix the yaml comment.
6. A `perf` split (~1,500 validation items disjoint from dev/test) for performance runs, so
   K=32–64 runs have n ≥ 10·K tasks. Quality comparisons stay on dev/test.
7. If the client exceeds ~60% of a core at K=32 (logged per run), experiment runs switch to
   `stream=false` with `timings.prompt_ms` as the TTFT proxy.

## 4. Policy

### 4.1 Costs

All costs are in **pool token·seconds** (memory held × time). For an agent holding *S*
tokens that enters a tool call, with *S_active* tokens of other active agents:

- retain: *S* token·s per second of idle time **while the pool is under pressure**, 0 otherwise;
- offload: *B_off = S · (t_save + t_restore) + S_active · stall(S)*, where `stall` is the
  decode time others lose if slot ops block the server (measured in E0a);
- drop: *B_drop = (S_active + S) · t_prefill(S)*;
- give-up cost *B = min(B_off, B_drop)*, frozen at tool start; B_off is unavailable when the
  host tier is full.

*Pressure* = pool occupancy above a threshold θ, or a nonempty admission queue.

### 4.2 Our rule: pressure-gated TTL

Accumulate rent *R(t) = ∫ S · 1[pressure] dt* from tool start. When *R ≥ B* and the tool has
not returned, give the memory up via the cheaper option. Without pressure the rule never
evicts; with constant pressure it waits *T\* = B / S*. `t_save`, `t_restore`, `t_prefill(S)`
and `stall` are fitted online from server timings.

Claim, stated narrowly: *per decision, in the model's cost units and with B fixed at tool
start, the rule pays at most 2× what the clairvoyant decision pays* (deterministic ski
rental). This is a property of the decision rule, not a bound on goodput; P5 logs the
per-decision ratio as a sanity check and reports the goodput gap separately.

Knobs (θ, admission safety margin) are tuned on dev with the same grid budget used to tune
K for the fixed-K baseline, then frozen.

### 4.3 Configurations

Main table (six):

| Config | What | Why |
|---|---|---|
| **FIFO + best fixed K** | Phase 1 runner, always retain, K tuned on dev per budget | Course baseline |
| **llama.cpp default** | `--no-pin`, `-np = K`, `--cache-ram 0`; server picks slots | Runtime-default cache behaviour |
| **Fixed-TTL offload** | Retain 1 s, then offload; no cost model | The course's "simple fixed cache policy" |
| **Always offload** | Offload at every tool start (prefetch at tool end) | Strongest naive mechanism if offload is cheap |
| **Ours** | Pressure-gated TTL + peak-KV admission + prefetch | The method |
| **Clairvoyant-delay TTL** | Same rule, but knows the true delay: gives up at tool start iff the rent it would pay exceeds B | Reference for how much delay knowledge is worth. **Not an upper bound.** |

Analytic upper bound: the cost model's ideal-offload headroom (every resident slot
decoding). Captured fraction = (ours − retain) / (clairvoyant − retain), reported with a CI,
or "not estimable" when the denominator's CI includes 0.

Required decomposition (2×2) at the main operating point:

|  | always retain | TTL lifecycle |
|---|---|---|
| **fixed K** | FIFO + best K | TTL, fixed K |
| **peak-KV admission** | admission + retain | ours |

Ablations: TTL without offload (drop only); no pressure gating; **min-waste on E[t] per tool**
(InferCept-style expected-cost rule). Appendix, one operating point: always drop, LRU,
`--cache-ram` with `--no-pin`, q8_0 KV.

Delay misspecification (one experiment): the policies estimate from median 0.8 s / σ 0.6,
while true delays use σ 1.2 or a shifted per-tool median. Expected: min-waste ≥ TTL when the
distribution is right; TTL better when it is wrong. This is what TTL adds over InferCept-
style min-waste; Continuum also uses KV TTLs, and we position against both.

### 4.4 Pre-registered branch after M0/E0a

The decision is only interesting if *T\** is comparable to tool delays. After M0/E0a we
compute *r = T\* / median delay* per profile and budget (with the measured stall):

- **0.1 < r < 10**: TTL stays the headline.
- **r < 0.1** (offload cheap, non-blocking): the headline becomes *which idle KV goes to the
  bounded host tier, plus restore prefetch, plus admission*; always-offload is the bar, and
  TTL is reported as a no-benefit case.
- **r > 10** (offload blocks the server, re-prefill dear): the headline becomes *admission +
  drop timing*.

E3 and P5 delay scales are chosen so the median delay spans 0.3×–3× of *T\**.

## 5. Experiments

### 5.1 Method

- **Workloads.** Tuning on dev (200). Performance runs on the `perf` split, n ≥ 10·K. Final
  P1 figure on test (500). Every policy sees the same frozen workload file: same questions,
  arrivals and seeded delays (paired design).
- **Primary capacity metric.** Maximum sustainable arrival rate at which p95 completion
  latency ≤ SLO (3× isolated latency per profile), found by a rate sweep per config and
  budget. Secondary: steady-state throughput and goodput in the 10%–90% completion window;
  queue time reported separately.
- **Tails.** p50 and p95 are primary; p99 only when n ≥ 500.
- **Repeats and CIs.** ≥3 runs where F1 or the main P1 figure is compared; other throughput
  points get 1 run, with run-to-run CV measured at 2 points. Metrics are computed per run;
  CIs bootstrap runs, then tasks within runs.
- **Quality.** Paired per-question ΔF1 vs the baseline, equivalence-tested (TOST,
  |ΔF1| ≤ 0.03); completed tasks, failures/timeouts and mean output tokens next to every
  throughput number.
- **Memory.** Occupied KV tokens, host-tier peak bytes, server RSS, total; plus the
  memory-saving view: the smallest `-c` at which each policy matches best-fixed-K capacity.
  An iso-total-memory reference (always retain with `-c` raised by the host-tier token
  equivalent) shows whether offload beats simply having more GPU memory.
- **Every run records** server command line, `-np`, `-c`, `--cache-ram`, slot path, build,
  workload sha, git rev.

### 5.2 Phase 1 (by 11/4, ~6–7 GPU-hours)

| ID | What | Configuration |
|---|---|---|
| M0 | Microbenchmarks → calibrate the cost model | f16: prefill vs L, decode vs depth, batched decode vs batch; `slot_offload_bench` (save/restore vs size) |
| E0a | Runtime behaviour that Phase 1 depends on | (a) pinned slots exceeding `-c`: fail, queue, or evict idle? (b) does stale KV stay resident; does the server purge idle slots? (c) is the shared prefix stored once per slot? (d) per-slot context limit under `--kv-unified`, `--kv-unified-per-slot`; (e) `-np` 4 vs 64 overhead; (f) decode stall on 8 busy slots while slot 0 saves 3k tokens to `/dev/shm`; (g) `/health` and a slot save at K=np=64 with all agents mid-tool (HTTP thread starvation) |
| E1 | Concurrency | closed loop, K ∈ {1, 4, 8, 16, 32}, n=200 (K=1: 50); 1 run, +2 repeats at K=8 and 16; Poisson at 3 rates for K=16 |
| E2 | Context and budget | `-c` ∈ {8k, 16k, 32k} × react_long-heavy mix at K=16 |
| E3 | Idle periods | delay scale {0, 1, 4} |
| E4 | Prefix reuse | `cache_prompt` on/off × pin/no-pin, few-shot 6 |
| E5 | KV composition | offline from E1/E2 events: occupancy timeline (active / idle-in-tool / stale / queued) |

Each E0 item is a script under `bench/e0_*.py` that saves raw `/slots` JSON and nvidia-smi,
so every E0 finding has evidence attached. Phase 1 deliverable: the dominant bottleneck per
regime, backed by M0 + E1–E5, plus how well the cost model's predictions matched.

### 5.3 Phase 2

| ID | What | Configuration |
|---|---|---|
| E0b | Runtime behaviour Phase 2 depends on | slot ops on a busy slot; restore into another slot; restore into a fragmented pool (16k pool, 8 interleaved slots, erase every other, restore 3k); `--cache-ram` semantics |
| P1 | Main comparison | six configs × 3 budgets (where E2 shows pressure) × host tier 1×; capacity metric; test split, 3 runs |
| P2 | Decomposition and ablations | 2×2 + ablations at one budget, perf split, 1–2 runs |
| P3 | Load, mix, host tier | rate sweep; short/long mixes; bursty (MMPP); host tier {0, 0.5×, 2×} |
| P4 | No-benefit regimes | zero tool delay; ample budget; short contexts only |
| P5 | Delay knowledge | ours vs clairvoyant vs min-waste across delay scales; misspecification experiment |
| P6 | Overhead and fairness | decision time; slot-op time, stall, restore failure rate; per-profile slowdown CDF |
| P7 | Model check | cost-model predictions vs P1 |

Main figures: (1) capacity vs GPU budget, one line per config, analytic bound marked;
(2) the 2×2 decomposition; (3) KV-occupancy timelines, always retain vs ours;
(4) slowdown CDF by profile.

### 5.4 Phase 3 (optional)

Text-level compaction of observations at tool boundaries. Scorers: oracle (supporting-fact
flags), recency, observation masking. F1 and SF recall vs mean KV, including the re-prefill
cost of breaking the append-only prefix. One Pareto figure.

## 6. Timeline and compute budget

| Dates | Work | Done when |
|---|---|---|
| 10/29–10/30 | M0 (running); E0a scripts and runs; cost model updated with measured contexts and fitted efficiencies; runner fixes §3.5 1–5 | Calibrated predictions; E0a table |
| 10/30–10/31 | Sweep driver, steady-state metrics, aggregate/plot scripts (Mac, MockLLM) | One unattended sweep works end to end |
| 11/1–11/2 | E1–E5 | Curves |
| 11/3–11/4 | Phase 1 report | Submitted |
| 11/5–11/11 | E0b; KV manager, slot ops, MockLLM, events; baselines | All six configs run end to end on dev |
| 11/12–11/18 | §4.4 branch decision; tuning on dev; P1 | Main figure |
| 11/19–12/2 | P2–P7; Phase 3 only if on track | All figures |
| 12/3–12/9 | Report; reproducibility scripts; raw measurements packaged | Final submission |

| Block | Runs (approx.) | GPU-hours |
|---|---|---|
| M0 + E0a | — | ~3 |
| E1–E4 | ~25 | ~4 |
| P1 (test, 3 runs) | 6 configs × 3 budgets × 3 | ~13 |
| P2–P6 (perf, 1–2 runs) | ~60 | ~20–25 |
| Slack, reruns | — | ~10 |
| **Total** | | **~50 h ≈ $50** |

The instance is stopped whenever no run is in progress (`tools/mem_aws.sh stop`).

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| KV never binds at realistic K | Cap `-c` (allowed) and state the device class it models. |
| The TTL decision is trivial (§4.4) | Pre-registered headline switch; TTL becomes a no-benefit case. |
| Slot save/restore blocks the server | Report it as a measured cost; drop is preferred; motivates async offload. |
| Restore needs contiguous free cells | Failed restore = drop, counted; feasibility uses a headroom margin. |
| Gain comes from admission alone | The 2×2 shows it; we report it honestly and focus the analysis on when the lifecycle adds to admission. |
| CUDA non-determinism | Paired workloads, repeats where it matters, run-level CIs, TOST for quality. |
| Temporary AWS credentials expire mid-run | Runs execute on the instance and need no AWS credentials; `ssh mem-aws 'sudo shutdown -h now'` stops it without them. |
