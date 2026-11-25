# CLAUDE.md

Context for Claude Code sessions on this repo. The user writes in Chinese; reply in Chinese
unless asked otherwise. Course deliverables (reports) are written in English.

## Project

CSE 291A, "Memory-Aware Agent Edge Serving". Many HotpotQA ReAct agents share one GPU running
a local model through llama.cpp `llama-server`, under a bounded KV-cache budget. Team: Jiawei
Liu, Kaiqi Han, Tianyu Xu, You Chen, Yufan Shi.

- **Phase 1 (submitted 11/4, done):** reproducible agent workload, FIFO fixed-concurrency
  baseline, measurements E1–E5 of what limits throughput.
- **Phase 2 (current, final submission ~12/9), plan in `docs/phase2_ideas.md`, in this order:**
  1. *Base*: KV-budget admission + pre-request gate (never overflow the pool) + one shared copy of
     the system prompt (fork patch). Infrastructure, not the contribution.
  2. *Selective KV* (main contribution): keep only the useful segments of an agent's KV and delete
     the rest in place (llama-server patch: `seq_rm` + `seq_add`, like context shift), no
     re-prefill, triggered by memory pressure. Text-only deletion + re-prefill is the baseline.
  3. *Idea B*: grouped KV pool between `--kv-unified` and per-slot KV (llama.cpp patch), agents
     bin-packed by projected peak, migrated with save/restore.
- The instructor said scheduling alone earns only a regular grade; the KV work is the
  contribution. Every cache action must use controls llama.cpp exposes; our own llama-server
  patches are allowed (user, 2026-11-16) and must be stated openly in the report.

Evaluation: goodput (correct tasks/min) and p95/p99 latency at a fixed KV budget are the main
metrics, with request-level TTFT and TPOT p50/p95 to explain latency. Completed count, failures,
EM/F1 and output length are constraints that every performance table reports (quality is a
guardrail, not the objective; ±0.03 F1 counts as unchanged). KV occupancy, idle-KV share,
recomputed tokens and prefix-cache hits explain the results.

### Documents (read these before planning work)

| File | What |
|---|---|
| `docs/phase2_ideas.md` | **Phase 2 plan, the source of truth for current work** (Chinese): goal and metric table, the three directions, ideas explored and dropped, stage-1 results |
| `docs/phase2_notes.md` | llama.cpp b11960 internals Phase 2 relies on (attention span, placement, `seq_cp`, observability) and the X1 / X3 / base smoke measurements |
| `tools/observability/README.md` | Prometheus + Grafana dashboards for runs (live and replay) |
| `docs/phase1/report.tex` (`docs/phase1_report.pdf`) | Submitted Phase 1 report |
| `docs/phase1_details.md` | Phase 1 full results, limitations, reproduction commands (Appendix A) |
| `costmodel/README.md` | Cost model and microbenchmarks, frozen predictions vs Phase 1 calibration |

## Current status (2026-11-24)

- Phase 1 is done and merged to `main` (PR #1 from `jiawei`). The user works on Phase 2 alone.
- **Base** (`edgeserve/kvm.py`, `policies.py`, fork patch): KV-budget admission with overcommit rho, a
  pre-request gate, and golden slots that hold the shared system prompt as evictable caches. 0 overflows in
  every managed run. KV-manager fixes and their regression tests are in `tests/test_kvm.py` (§4.6).
- **Stage 1** (§3): trimming `react_long` observations to the paragraph halves peak KV and doubles goodput at
  equal F1, so observation text is the main KV waste. Idea A (pressure-adaptive trimming) was dropped.
- **Selective KV steps 1–2** (§4.1–4.2): in-place deletion works (~1.5 ms, full cache reuse, memory freed);
  BM25-30% "trim after use" cuts peak-KV p95 by 26% at unchanged F1 and beats text compaction + re-prefill
  (11–14% higher goodput). Config: `--selective kv --sel-scorer bm25 --sel-keep 0.3 --sel-keep-last 0`.
- **Step 3, finding the scenario** (§4.4–4.6): selective KV helps only when memory is the bottleneck AND
  enough KV is trimmable (5k-token `oneshot_long` tasks leave nothing to trim). Its edge over a *tuned* base
  equals the re-prefill cost the base pays: at 8k + 70% long the GPU is mostly idle, so the base's
  overbooking + re-prefill is nearly free and the two tie at their own best rho (the regime map's gains were
  against a base held at rho 1.5). rho sets concurrency only indirectly, so each configuration is compared at
  its own best rho. Single runs vary 10–15%: claims need repeats and the rule fixed in advance (clear = gap >
  2 x the larger sd). A longer-retrieval scenario was rejected by offline replay.
- **Step 3 results, 16k + 70% long, all on the same final code** (§4.7):
  - Capacity (dev 200, t=0, 2 runs, `configs/sweeps/p3_cap16k.yaml`): base rho 1.75 goodput 27.9 +- 0.4
    (53.5 tasks/min), kv rho 1.5 30.7 +- 0.6 (59.6): **+10.2% goodput, clear**, +11.5% throughput, paired
    dF1 -0.002. The earlier +26% (and +11% under the old pressure rule) came from code before the smoke-check
    fixes, which made the base faster; they are no longer the claim. Added kv rho 1.75: 33.7 +- 0.5 (64.2), i.e.
    +20.8% over the base at the same rho 1.75 (clear). The base re-checked at rho 2.0 on the final code: 25.3 +- 1.7,
    below 1.75, so 1.75 stays its best and +20.8% is the peak-goodput claim. No more runs for the report.
  - Text deletion + re-prefill control (same file, rho 1.5 / 1.75 for both mechanisms): best vs best kv 33.7 vs text
    33.2, **goodput tie** (+1.4%, bar 1.1). kv is faster per request in all 4 run pairs (TPOT p50 -12-18%, TTFT p95
    -22-26%, prefill tokens -14-16%) although it deletes 22% fewer chars (spans off token boundaries are skipped).
    So the gain over the base comes from deleting used search content; in-place deletion's edge is no re-prefill
    (per-request latency), not goodput, at 16k on the A10G. Both bests sit at rho 1.75 (grid edge): a limitation.
  - Main experiment (test 500, bursty, 50/80/110% of 53.5, 3 rounds, `results/p3_main16k/main.json`): at 80%
    latency p95 -22% / p99 -26% (clear); at 110% goodput +7.6% and p95/p99 -32%/-31%, better in all 3 rounds
    but not clear (the base varies across arrival sequences; open-loop goodput is capped by the arrival rate,
    and 110% of the base is about kv's capacity); 50% ties although deletion runs as often (~860 trims per
    run), so the gain comes from the freed memory. F1 unchanged (+0.001 to +0.016). vs fixed K3 at 110%:
    goodput 1.64x, p95 41 s vs 330 s. 0 overflows in 10,500 tasks.
- **Next**: the final report (by 12/9). Deferred: deleting whole used observations (`--sel-scorer recency
  --sel-keep 0 --sel-keep-last 0`, offline estimate: mean KV -38% vs -21%); idea B suggested for future work
  (the user decides).
- Development is on the AWS instance (see AWS below); the Mac is for MockLLM work.

## Phase 1 findings (A10G, 7B, 5,890 tasks)

- **Workload** (600 tasks, K=8): context first / peak p50 / peak p95: `react_short` 1.5k/1.8k/2.5k,
  `react_long` 1.5k/3.5k/6.2k, `react_explain` 1.7k/2.3k/3.1k (443 output tokens),
  `oneshot_long` 3.3k/3.3k/3.9k. F1 0.50–0.60 by profile; closed-book F1 0.25.
- **E1, ample pool (131k):** tasks/min K=1 14.5, K=4 36.8, **K=8 45.2 (peak)**, K=16 43.0,
  K=32 31.0 (p95 service 22.6 s → 141.7 s). Cause: in the unified pool attention cost scales
  with all slots' tokens (time per token 15 ms at K=4 → 254 ms at K=32, vs 28 ms with separate
  KV). Per-slot KV regions: 43.0 / 54.8 / 64.5 at K=8/16/32 (+27% at K=16).
- **E2, tight budgets:** active KV above the pool fails **all** in-flight requests (HTTP 500);
  completion drops from 89–100% to ≤24% when K is doubled past the budget. Contexts grow 2.3×
  (median) to 4.2× (p95) during a `react_long` task, so fixed K is sized for peaks: largest
  working K ≈ budget / 4k tokens (2, 4, 8, 16 for 8k–64k). 8k and 16k reach 48% / 80% of the
  throughput of larger budgets.
- **E3–E5:** under tight budgets 36–44% of held KV belongs to agents waiting on tools (pool
  ~58% occupied at 16k, K=4). With an ample pool idle KV barely matters (delay ×4: 44.1 → 41.0
  tasks/min). No prefix reuse: 4.9× prefill, 36% throughput. Unpinned slots: 31% of steps land
  on another slot, −17%. The 1.4k-token prefix is duplicated per slot (73% of a `react_short`
  context). Only 4.5% of retrieved `react_long` tokens are supporting facts.
- **Runtime costs (b11960):** re-prefill 0.45 s (2k) / 1.97 s (8k); slot save / restore to
  `/dev/shm` 1.35 / 4.74 GB/s and it **blocks all slots** (119 ms for 3k tokens); idle slots
  over the pool are purged lowest index first; `-np` 4 vs 64 at equal `-c` changes nothing.
  Break-even idle time for releasing KV is 1.7–3.0 s at K=16 vs a 0.8 s median tool delay, so
  the policy retains by default and releases only under pressure.
- GPU memory is fixed by the preallocated pool (5.0 GiB at 8k to 11.6 GiB at 131k); occupied
  KV tokens are the varying metric. `/metrics` has no KV gauge: compute occupancy from per-slot
  token counts or `/slots`. 7B weights 4.36 GiB; KV 56 KiB/token (f16).
- Failures: 971 context overflows (all E2, by design) and 9 from a client keep-alive race
  (`RemoteProtocolError`), to be fixed at the start of Phase 2 (slot ops on a separate client
  without keep-alive).
- **CUDA outputs are not deterministic with concurrent slots** (temperature 0, fixed seed).
  Compare quality with paired workloads, repeats or large n, never from one small run.

## Phase 2 next steps (details in `docs/phase2_ideas.md`)

Work one step at a time: define the scenarios (workload mix, pool sizes, question set, baselines)
before running experiments, report, and wait for the user's go-ahead before the next step.

1. Selective KV: steps 1–3 are done (`docs/phase2_ideas.md` §4.1–4.7).
2. Base: tuned and measured with repeats as part of step 3 (§4.5–4.7).
3. Idea B: suggested for future work; ~1 day feasibility check of the grouped-stream patch if the user wants it.
4. Final report and reproducibility package by 12/9.

## Decisions already made (do not reopen without the user)

- Written from scratch. Do not restore the reverted commit `ef37620` (the teammate branch
  `origin/tianyu` holds that older framework).
- `search` runs over a **global offline corpus**: every HotpotQA validation paragraph,
  deduplicated by title (66,581 pages). Easier than fullwiki, so accuracy is only compared
  across our own configurations. Agent v3 is frozen for all experiments.
- The prompt is **append-only**: ChatML rendered once on the client, one assistant turn that
  grows by Thought / Action / Observation, sent raw to `/completion`. Each step's prompt is a
  byte-prefix of the next.
- **Slot pinning**: worker k sends every request with `id_slot=k`, `cache_prompt=true` ("always
  retain" baseline). `--no-pin` keeps llama.cpp's own slot policy as a baseline.
- Model: **Qwen2.5-7B-Instruct Q4_K_M**, f16 KV, temperature 0; 1.5B only for quick checks.
- The KV budget is `-c` with `--kv-unified` (one pool shared by all slots, preallocated at
  startup); memory pressure is created by capping `-c`. `--cache-ram 0` keeps KV only in that
  pool. Never hard-code a host's limits: derive `-c` and K from the host.
- Numbers come from AWS g5.xlarge (A10G) with llama.cpp **b11960**; Mac and Windows runs are
  smoke checks only.

## Code map

- `edgeserve/`: `data.py` (HotpotItem, splits) · `corpus.py` (title match → BM25 fallback) ·
  `tools.py` (search/lookup, every sentence tagged supporting-fact or not) · `prompt.py` (rules,
  few-shot, segment ledger: shared_prefix, question, thought_action, obs_sf, obs_other, obs_meta,
  control) · `agent.py` (ReAct loop, forced finish, one-shot / closed-book) · `profiles.py` +
  `configs/profiles.yaml` · `delays.py` (seeded tool latency) · `workload.py` · `llm.py`
  (streaming client, MockLLM) · `runner.py` (FIFO, K workers, events/tasks/meta/summary,
  steady-state throughput and goodput) · `eval.py` (official EM/F1).
- Profiles: `react_short`, `react_long` (+ `react_long_l1..l3`, stage-1 observation levels), `react_explain`, `oneshot_long`; `closedbook` is a
  quality baseline only. Splits: dev 200 (tuning), test 500 (final), perf 1,500 (large sweeps).
- `scripts/`: run as modules from the repo root (`uv run python -m scripts.build_corpus`);
  `scripts/start_server.sh` (env: MODEL, CTX, PARALLEL, PORT, SLOT_PATH, LLAMA_SERVER, THREADS,
  CMD_OUT, EXTRA); `scripts/sweep.py` (YAML grid in `configs/sweeps/` → server restart → runner,
  resumable); `scripts/run_p3_main16k.sh` reproduces the 16k capacity + main experiment end to end
  (`scripts/make_p3_main_workloads.py` builds its workloads from the measured capacity).
- Phase 2: `kvm.py` (KV manager: accounting, admission, gate, give-up/restore, fork, `kv_state`
  events) · `policies.py` (configs: baselines `fixed_k`/`per_slot`/`llamacpp`/`always_offload`,
  `base`, `base_fork`, decomposition cells) · `kvtrim.py` (selective KV: text spans -> token ranges,
  `SelectiveKV` run by the agent during tool calls, runner `--selective kv|text`) · `selection.py`
  (which observation sentences to delete: oracle/bm25/entity/recency). `llm.py` has slot
  save/restore/erase/fork/trim and tokenize, and a MockLLM with a finite pool, overflow and slot ops.
- `tools/llama_patch/` (fork + trim patches and build script; patched binaries in
  `~/llama.cpp/build/bin`, local branch `edgeserve` in `~/llama.cpp`, unpatched in `build/bin-orig`) · `tools/observability/` (exporter + Prometheus + Grafana).
- `bench/` (E0 runtime probes, microbenchmarks, slot offload bench, `x1_speed_tax.py`,
  `x3_fork_check.py`, `s1_trim_check.py`), `analysis/` (aggregate, composition, kv_timeline,
  phase1_figures, `check_kvm.py`, `obs_levels.py`, `selective_quality.py`, step 3 reports `p3_regime.py`,
  `p3_tune.py`, `p3_main.py`), `costmodel/`, `tools/mem_aws.sh`.
- Tests: `uv run pytest` (mock LLM, no model). Mock end-to-end:
  `uv run python -m edgeserve.runner --workload workloads/smoke.json --k 2 --mock --out results/mock`.
- `data/`, `models/`, `results/`, `.env`, `*.pem` are gitignored and rebuilt per machine.
  `data/splits/dev.jsonl` sha256 starts `7c816039` on every machine.

## Machines

**AWS (experiments and main development):** the `CSE291` g5.xlarge in us-west-2 (A10G 23 GB,
4 vCPU, 15 GiB RAM, 96 GB disk), Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04), CUDA
13.2 at `/usr/local/cuda`. SSH alias `mem-aws` (user `ubuntu`); the public DNS changes on
every start. Layout:
- repo: `~/mem-aware` (git clone, branch `yufan`)
- llama.cpp: `~/llama.cpp`, tag `b11960`, built with
  `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j4`;
  run with `LLAMA_SERVER=~/llama.cpp/build/bin/llama-server scripts/start_server.sh`
- slot offload files: `/dev/shm/slots` (host RAM; shares the 15 GiB with server, runner, BM25)
- run long jobs (builds, servers, sweeps) inside `tmux` so SSH/VS Code disconnects don't kill them
- `tools/mem_aws.sh` hard-codes the teammate's Phase 1 instance and AWS profile, not this one.

**Mac (M1 Pro, 16 GB):** development and MockLLM tests. On macOS use `uv run --no-sync` (the
venv `.pth` file gets a hidden flag).

**Windows laptop (RTX 4060 8 GB, native):** local smoke runs only. Repo at
`D:\Memory-Aware-Agent-Edge-Serving`; llama.cpp b11960 `win-cuda-12.4-x64` at
`D:\llama.cpp-b11960`; run bash scripts from Git Bash. All text I/O passes `encoding="utf-8"`
and writes `\n`; `.gitattributes` forces LF. The GPU drives the display and the driver spills
past 8 GB into system RAM ("Prefer No Sysmem Fallback" only partly helps), so use `-c ≤ 32k`
there and treat a sudden prefill slowdown as spill.

## Working rules

- Never print, paste or commit private keys or API keys. The UCSD model API key lives in
  `.env` (gitignored) and is only used for offline analysis such as an LLM judge, never as
  the served model. Never copy `.pem` files onto the instance.
- Commit or push only when the user asks. Work on the `yufan` branch; changes reach `main`
  through PRs.
- Stop (do not terminate) the EC2 instance when idle.
