# CLAUDE.md

Context for Claude Code sessions on this repo. The user writes in Chinese; reply in Chinese
unless asked otherwise. Course deliverables (reports) are written in English.

## Project

CSE 291A, "Memory-Aware Agent Edge Serving". Many HotpotQA ReAct agents share one small GPU
running a local model through llama.cpp `llama-server`, under a bounded KV-cache budget.
Full design (in Chinese): `docs/project_design.md`. Phase 1 report draft: `docs/phase1_report.md`.

- **Phase 1 (due 11/4):** reproducible agent workload, FIFO fixed-concurrency baseline,
  measurements (E1–E5) that answer what limits throughput: active KV capacity, KV held by
  agents idle during tool calls, or re-prefill after eviction.
- **Phase 2:** KV lifecycle manager at tool-call boundaries (retain / offload via slot
  save-restore / drop and re-prefill), cost model with online estimates, KV-aware admission.
- **Phase 3:** selective KV (trim non-useful observation text), using HotpotQA supporting
  facts as the oracle. Text-level compaction vs. a small llama-server KV range-drop patch.
- The instructor said scheduling alone earns only a regular grade; the KV work is the
  contribution. Every cache action must use controls llama.cpp actually exposes.

Evaluation: goodput (correct tasks/min) and p95/p99 latency at a fixed KV budget are the main
metrics. Completed count, failures, EM/F1 and output length are constraints that every
performance table reports. KV occupancy, idle-KV share, recomputed tokens and prefix-cache
hits explain the results.

## Decisions already made (do not reopen without the user)

- Written from scratch. Do not restore the reverted commit `ef37620` (the teammate branch
  `origin/tianyu` holds that older framework).
- `search` runs over a **global offline corpus**: every HotpotQA validation paragraph,
  deduplicated by title (66,581 pages). Known caveat: easier than fullwiki (~5M pages), so
  accuracy is only compared across our own configurations.
- The prompt is **append-only**: ChatML rendered once on the client, one assistant turn that
  grows by Thought / Action / Observation, sent raw to `/completion`. Each step's prompt is a
  byte-prefix of the next.
- **Slot pinning**: worker k sends every request with `id_slot=k`, `cache_prompt=true`, which
  is the "always retain" baseline. `--no-pin` keeps llama.cpp's own slot policy as the
  fixed cache-policy baseline.
- Model: **Qwen2.5-7B-Instruct Q4_K_M** for experiments; 1.5B only for quick checks.
- Hardware, in two stages: **(1)** get low-load real serving working and observable on the
  local Windows laptop (RTX 4060 Laptop, 8 GB, native Windows, no WSL); **(2)** then run the
  real experiments on AWS, using that GPU's full memory. The Mac (M1 Pro, 16 GB unified
  memory) is for development and mock tests. Local numbers are smoke checks only, so nothing
  may hard-code the laptop's 8 GB / 32k limits: derive `-c` and K from the host.
- The KV budget is `-c` with `--kv-unified` (one pool shared by all slots, preallocated at
  startup). `--cache-ram 0` keeps KV only in that pool.

## Code map

- `edgeserve/`: `data.py` (HotpotItem, splits) · `corpus.py` (title match: exact → normalized →
  qualifier removed → fuzzy; BM25 suggestions on a miss; BM25 neighbors) · `tools.py`
  (search/lookup, every sentence tagged supporting-fact or not) · `prompt.py` (rules, 6
  hand-written few-shot examples, segment ledger) · `agent.py` (ReAct loop, forced finish,
  one-shot and closed-book modes) · `profiles.py` + `configs/profiles.yaml` · `delays.py`
  (seeded by seed, task, step, tool) · `workload.py` · `llm.py` (streaming client, MockLLM) ·
  `runner.py` (FIFO, K workers, events/tasks/meta/summary) · `eval.py` (official EM/F1).
- Profiles: `react_short`, `react_long`, `react_explain`, `oneshot_long`, plus `closedbook`
  (quality baseline only).
- `scripts/`: run as modules from the repo root, e.g. `uv run python -m scripts.build_corpus`.
  On macOS use `uv run --no-sync` (the venv `.pth` file gets a hidden flag).
- Tests: `uv run pytest` (mock LLM, no model needed). Mock end-to-end:
  `uv run python -m edgeserve.runner --workload workloads/smoke.json --k 2 --mock --out results/mock`.
- `data/`, `models/`, `results/`, `.env`, `*.pem` are gitignored and rebuilt per machine.

## Results so far (dev 100, react_short, K=2, delays off, run on the Mac)

| Setting | EM | F1 | Forced finish |
|---|---|---|---|
| Closed-book | 0.21 | 0.25 | — |
| ReAct v1 (fuzzy-title suggestions) | 0.40 | 0.47 | 45% |
| ReAct v2 (BM25 suggestions) | 0.48 | 0.57 | 33% |
| ReAct v2, rerun on Windows (CUDA, K=4) | 0.49 | 0.58 | 29% |
| ReAct v3 (title miss returns the top BM25 paragraph's text, current; Windows, 2 runs) | 0.53–0.56 | 0.64–0.65 | 16–19% |

- v3 vs v2 on Windows: SF recall 0.61 → 0.69–0.70, bridge F1 0.51 → 0.58–0.60, comparison
  F1 0.90 → 0.85 (one task of 20), LLM calls/task 4.57 → 4.0, tasks with a repeated action
  13% → 7–9%. ~16 tasks better, ~8 worse (the model sometimes trusts a wrong closest result).
- Profiles v3 (`profiles_check`, 20 dev questions each, 7B, K=4, delays off):

| Profile | F1 | Forced | LLM calls | Context p50 / max | Output tokens/task |
|---|---|---|---|---|---|
| `react_short` | 0.61 | 20% | 4.3 | 1.9k / 3.0k | 177 |
| `react_long` (10 paragraphs per search) | 0.61 | 10% | 3.5 | 3.4k / 5.3k | 129 |
| `react_explain` (+300-token Explanation, `ignore_eos`) | 0.75 | 5% | 4.2 | 2.3k / 2.9k | 429 |
| `oneshot_long` (10 own + 15 BM25 distractors) | 0.63 | — | 1 | 3.6k / 4.7k | 48 |

  30 distractors gave `oneshot_long` 5.3k context but F1 0.38, so 15 was chosen.
- **Outputs are not deterministic on CUDA with concurrent slots** (temperature 0, fixed seed):
  two runs of the same 20 `react_short` tasks had identical traces for only 3 tasks and the
  same prediction for 15; F1 moved by up to ~0.07 on 20 tasks. Batched kernels are not
  batch-invariant and this build has no deterministic mode. Compare quality across
  configurations with repeated runs or large n, never from a single 20-task run.
- This build has `--kv-unified-per-slot N` (per-slot context limit in the unified pool).

- Quality gate: F1 well above closed-book (pass), 0% parse errors (pass), forced finish
  33% vs. target 15% (fail). Main cause: the model repeats an identical action (13/100 tasks).
- F1 is 0.87 when all supporting facts were retrieved, 0.21 when none were: retrieval, not
  reasoning, limits quality. 52% of searches still miss on the title, then recover via suggestions.
- Workload: prompt p50 1,684 / max 2,289 tokens; 95.9% of prompt tokens hit the prefix cache;
  ~60 new tokens prefilled and ~37 decoded per step. The shared prefix is ~76% of each context.
  Supporting facts are ~24% of observation text.
- 1.5B smoke run with tool delays: K=1→4 raised throughput from 10.3 to 21.3 tasks/min. At K=1,
  tool calls take 36% of the time an agent holds its slot.

Measured on llama.cpp build b11960 (Mac, Metal):
- 7B weights take 4.36 GiB; KV takes exactly 56 KiB per token (f16); compute buffers ~0.2–0.3 GiB.
- An 8 GB budget therefore leaves ≈56k KV tokens at most (less on Windows if the GPU also
  drives the display: check idle VRAM with `nvidia-smi` first).
- `/metrics` in this build has **no KV-usage gauge**, and process RSS does not reflect Metal
  KV. KV occupancy has to be computed from per-slot token counts (or by polling `/slots`).

## Next steps

1. Decide whether to add a repeated-action hint in `agent.py`, rerun the gate, then freeze
   the agent version for all experiments.
2. Local Windows stage (see below): low-load real runs (1.5B smoke, then 7B) and check that
   the metrics we need are observable. Then move to AWS.
3. **E0 pressure test:** fill the KV pool and record llama-server's behavior (errors, queueing,
   or eviction of idle slots). Check whether the shared prefix is stored once per slot. If it is,
   16 slots × ~1.4k tokens ≈ 40% of a 56k pool.
4. Run the other profiles on 7B (~20 questions each) to measure real context sizes and
   quality, then set the workload mix. Lower the `react_explain` share if decoding is slow.
5. Code: a KV-occupancy monitor, `start_server.sh` defaulting to 7B with `-c` derived from the
   8 GB budget, and the server's memory breakdown saved into `meta.json`.
6. E1–E5 sweeps (K; `-c` ∈ {16k, 32k, ~56k}; delay scale; cache on/off × few-shot ×
   pin/no-pin; offline composition), then plots and the final Phase 1 report.

## Windows (native) setup notes

- Repo lives at `C:\Users\shiyu\dev\Memory-Aware-Agent-Edge-Serving`, outside OneDrive (models
  and results must not sync). `.gitattributes` forces LF everywhere; keep it.
- Python via `uv` (winget `astral-sh.uv`); `hf` via `uv tool install "huggingface_hub[cli]"`.
  Run the bash scripts from Git Bash. All text I/O passes `encoding="utf-8"` and writes `\n`
  (the Windows default is cp936 + CRLF); keep that in new code. Server RSS uses `psutil`.
- llama.cpp: the official **b11960** `win-cuda-12.4-x64` release plus its cudart zip, unpacked
  to `C:\Users\shiyu\dev\llama.cpp-b11960` (driver supports CUDA ≤ 13.3). Point
  `start_server.sh` at it with `LLAMA_SERVER=/c/Users/shiyu/dev/llama.cpp-b11960/llama-server.exe`.
- The GPU also drives the display: 1.1–2.5 GB is used before the server starts, depending on
  what else is open. That leaves at most ~32k usable KV tokens for 7B, not 56k (see below).
- In the NVIDIA Control Panel, set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem
  Fallback". Otherwise the Windows driver silently spills past 8 GB into system RAM and the
  budget stops being real. Set on 2026-10-28, and it only partly helps (7B, `-np 4`, probed
  with three 3k-token prefills): `-c 65536` fails at once with `cudaMalloc failed: out of
  memory`; 16k and 32k load and prefill in ~2 s; **40k loads but prefill is 5–15× slower**
  (WDDM pages memory out instead of failing); 48k never becomes healthy. So use `-c ≤ 32k`
  locally, close games and other GPU apps first (a running game held 1.2 GB), and treat a
  sudden prefill slowdown as a sign of spill.
- Data rebuilt on Windows: 66,581 pages; the four frozen `workloads/*.json` regenerate
  byte-identical, so the splits match the Mac's. `dev.jsonl` sha256 starts `7c816039`.
- First local runs (smoke, 12 tasks, tool delays on, `-c 16384 -np 4`):
  1.5B K=1 11.4 → K=4 37.1 tasks/min; 7B K=4 20.5 tasks/min, EM 0.33 / F1 0.46, GPU 6.6 GB
  total. 7B: ~1.3k prefill tok/s, ~37 decode tok/s per request, ~81% of prompt tokens cached;
  tool waits are 31% of slot-hold time (67–75% on 1.5B).

## Working rules

- Never print, paste or commit private keys or API keys. The UCSD model API key lives in
  `.env` (gitignored) and is only used for offline analysis such as an LLM judge, never as
  the served model.
- Commit or push only when the user asks. Work on the `yufan` branch.
- Stop (do not terminate) any EC2 instance when idle.
