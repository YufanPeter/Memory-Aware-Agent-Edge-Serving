# CLAUDE.md

Context for Claude Code sessions on this repo. The user writes in Chinese; reply in Chinese
unless asked otherwise. Course deliverables (reports) are written in English.

## Project

CSE 291A, "Memory-Aware Agent Edge Serving". Many HotpotQA ReAct agents share one GPU running
a local model through llama.cpp `llama-server`, under a bounded KV-cache budget. Team: Jiawei
Liu, Kaiqi Han, Tianyu Xu, You Chen, Yufan Shi.

- **Phase 1 (submitted 11/4, done):** reproducible agent workload, FIFO fixed-concurrency
  baseline, measurements E1–E5 of what limits throughput.
- **Phase 2 (current, final submission ~12/9):** agent-level KV lifecycle manager at tool-call
  boundaries (retain / offload via slot save-restore / drop and re-prefill) plus admission that
  reserves each agent's projected peak KV.
- **Phase 3 (optional):** selective KV, i.e. trimming non-useful observation text, with HotpotQA
  supporting facts as the oracle.
- The instructor said scheduling alone earns only a regular grade; the KV work is the
  contribution. Every cache action must use controls llama.cpp actually exposes.

Evaluation: goodput (correct tasks/min) and p95/p99 latency at a fixed KV budget are the main
metrics. Completed count, failures, EM/F1 and output length are constraints that every
performance table reports. KV occupancy, idle-KV share, recomputed tokens and prefix-cache
hits explain the results.

### Documents (read these before planning work)

| File | What |
|---|---|
| `docs/kv_lifecycle_plan.md` | **Phase 2 plan, the source of truth for current work**: architecture, invariants, hook points, components, policy, configs, experiments P1–P7, timeline |
| `docs/phase1/report.tex` (`docs/phase1_report.pdf`) | Submitted Phase 1 report |
| `docs/phase1_details.md` | Phase 1 full results, limitations, reproduction commands (Appendix A) |
| `docs/project_design.md` | Original design (Chinese) |
| `docs/cost_model_and_microbench_plan.md`, `costmodel/` | Cost model, calibrated with Phase 1 measurements |

## Current status (2026-11-15)

- Phase 1 is done and merged to `main` (PR #1 from `jiawei`); `yufan` = `main` at `64091ce`.
- **Phase 2 code has not started**: there is no `edgeserve/kvm.py` or `edgeserve/policies.py`
  yet. The plan expected all baselines running end to end by 11/11, so we are about a week
  behind. Check with the user which parts teammates already own before starting a component.
- Development moves to the AWS instance (see AWS below); the Mac is for MockLLM work.

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

## Phase 2 next steps (details in `docs/kv_lifecycle_plan.md`)

1. `llm.py`: `slot_save/restore/erase`, `slots()`, server `timings`, on a separate httpx client
   without keep-alive. `MockLLM`: pool capacity, overflow as measured, slot ops, optional
   "slot op blocks all decode" mode.
2. `edgeserve/kvm.py` (accounting, admission, timers, events `kv_state`, `slot_op`, `kv_wait`,
   `admit_wait`) wired into the hook points in plan §3.3; `edgeserve/policies.py` with the six
   configs of §4.3 (FIFO + best K, llama.cpp default, fixed-TTL offload, always offload, ours =
   pressure-gated TTL + peak-KV admission + prefetch, clairvoyant-delay TTL). Unit tests on
   the Mac against MockLLM.
3. E0b on AWS (slot ops on busy slots, restore into another slot / a fragmented pool).
4. §4.4 branch decision, tuning on dev, then P1 (main comparison), P2 (2×2 decomposition and
   ablations), P3–P7.
5. Final report and reproducibility package by 12/9.

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
- Profiles: `react_short`, `react_long`, `react_explain`, `oneshot_long`; `closedbook` is a
  quality baseline only. Splits: dev 200 (tuning), test 500 (final), perf 1,500 (large sweeps).
- `scripts/`: run as modules from the repo root (`uv run python -m scripts.build_corpus`);
  `scripts/start_server.sh` (env: MODEL, CTX, PARALLEL, PORT, SLOT_PATH, LLAMA_SERVER, THREADS,
  CMD_OUT, EXTRA); `scripts/sweep.py` (YAML grid in `configs/sweeps/` → server restart → runner,
  resumable).
- `bench/` (E0 runtime probes, microbenchmarks, slot offload bench), `analysis/` (aggregate,
  composition, kv_timeline, phase1_figures), `costmodel/`, `tools/mem_aws.sh`.
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
