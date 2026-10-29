# Phase 1 Report: Memory-Aware Agent Edge Serving

CSE 291A · Team: _[names]_ · October 28, 2026

## 1. Problem and scenario

We study one edge GPU serving many tool-using agents with a local 7B model. Each request is a HotpotQA multi-hop question. A ReAct agent solves it by alternating LLM calls with `search`/`lookup` tool calls. While a tool runs, the agent's KV cache sits idle on the GPU but will be needed again on the next step.

Phase 1 delivers two things. The first is a reproducible, instrumented version of this workload. The second is a FIFO, fixed-concurrency baseline that we use to measure where the KV budget goes.

The question Phase 1 answers (RQ1): under this workload, what limits throughput? We consider three candidates:

- (a) KV capacity for active requests
- (b) KV held by agents that are idle during tool calls
- (c) re-prefill after a context is evicted

We measure how each one scales with concurrency, context size and tool latency. Phase 2 acts on the answer by deciding, at each tool-call boundary, whether to retain, offload, drop or trim an agent's KV.

We chose HotpotQA because it annotates supporting facts. That lets us label every observation token in an agent's context as evidence or not. As a result, we can measure how much KV is useful, not only how much is used.

## 2. Workload

- **Data.** We use the HotpotQA distractor validation set: 7,405 questions, all at the "hard" level, 80% bridge and 20% comparison. From it we draw fixed, disjoint dev (200) and test (500) splits, stratified by question type with seed 0. All tuning happens on dev.
- **Tools.** Search runs over an offline global corpus: every validation paragraph, deduplicated by title (66,581 pages).
  - `search[entity]` matches a title by trying, in order: exact, normalized, trailing qualifier removed, fuzzy. On a miss it returns the top-5 BM25 full-text titles as suggestions, like the Wikipedia tool in the ReAct paper.
  - `lookup[keyword]` returns the next sentence of the current page that contains the keyword.
  - Every returned sentence carries its `(title, sentence index)` and a supporting-fact flag.
- **Prompt.** The system prompt holds the rules and 6 hand-written few-shot examples, none of them from the validation set. It is rendered as ChatML once, on the client. The agent then extends a single assistant turn append-only (Thought / Action / Observation). Each step's prompt is therefore a byte-prefix of the next, so prefix reuse depends only on what the server keeps in KV.
- **Task profiles.** These mix short and long contexts, short and long outputs, and single- and multi-step tasks.

| Profile | What it does | Steps | Context | Output |
|---|---|---|---|---|
| `react_short` | ReAct; `search` returns the first 3 sentences | 2–7 | ~2k tokens | ~40 tokens/step |
| `react_long` | ReAct; `search` returns the whole paragraph plus 4 BM25-related paragraphs | 2–7 | 4–8k | short |
| `react_explain` | Like `react_short`, plus a 150–250-word explanation before `finish` | 2–7 | ~2k | up to 512 tokens |
| `oneshot_long` | No tools; the question's own 10 paragraphs are given and answered in one call | 1 | ~2–3k | short |

- **Tool delays.** By default the delay is lognormal (median 0.8 s, σ 0.6, capped at 5 s), and `lookup` takes a fixed 0.1 s. Delays are seeded by (seed, task, step, tool), so they do not depend on scheduling order. A global scale factor sweeps them.
- **Arrivals.** Batch, uniform, Poisson or bursty (MMPP). Each workload is generated once and frozen to a JSON file, which the runner replays.

## 3. Baseline and measurement

- **Runtime.** llama.cpp `llama-server` (build b11960) runs Qwen2.5-7B-Instruct Q4_K_M at temperature 0 with a fixed seed, using `--kv-unified --cache-ram 0 --metrics`. The KV budget is set with `-c`.
- **Baseline.** There is one FIFO queue and K workers, and a task holds its worker until it finishes. Worker *k* sends every request with `id_slot=k` and `cache_prompt=true`. An agent's KV therefore stays resident through its tool calls: the "always retain" policy. An unpinned mode keeps llama.cpp's own slot selection as the fixed cache-policy baseline.
- **Logging.**
  - `events.jsonl`: task arrive, admit and done. For each request: start, first token and end, with prompt, cached and generated token counts and prefill/decode time. For each tool call: start and end.
  - `tasks.jsonl`: answer, EM/F1, supporting-fact recall, and a ledger of the prompt split into typed segments.
  - Periodic samples: GPU memory, server RSS and `/metrics`.
  - `meta.json`: build, model, workload hash and git revision.
- **Metrics.**
  - Throughput and goodput (tasks per minute that are answered correctly).
  - Latency at p50/p95/p99, queue wait and TTFT.
  - Prefix-cache hit rate.
  - KV composition by segment type.
  - Every performance table also reports the completed count and F1, so a speedup cannot come from skipping work or lowering quality.
- **Checks passed.**
  - The ledger's token count equals the server's `prompt_n`.
  - At step *i*, the cached token count matches the previous step's prompt length.
  - Predictions are identical across values of K.
  - 15 unit tests pass against a mock LLM.

## 4. Preliminary results

**Quality gate.** Settings: the 100 dev questions, `react_short`, K = 2, tool delays off. These runs used the same GGUF model on an M1 Pro (Metal), so the quality numbers carry over to the GPU, but the speed numbers do not.

| Setting | EM | F1 | Bridge F1 | Comparison F1 | Forced finish |
|---|---|---|---|---|---|
| Closed-book (no retrieval) | 0.21 | 0.25 | 0.15 | 0.67 | — |
| ReAct v1 (fuzzy-title suggestions) | 0.40 | 0.47 | 0.36 | 0.90 | 45% |
| ReAct v2 (BM25 suggestions) | 0.48 | 0.57 | 0.48 | 0.90 | 33% |

Against the gate criteria:

- F1 is well above closed-book (0.57 vs 0.25). Pass.
- The parse-error rate is 0%, against a target of under 5%. Pass.
- 33% of tasks reached the step limit and were forced to answer, against a target of under 15%. Fail.

The main cause of forced finishes is that the model repeats an identical action (13 of 100 tasks). Retrieval, not reasoning, limits quality: F1 is 0.87 when all supporting facts were retrieved and 0.21 when none were.

**Workload characterization** (ReAct v2, 462 LLM requests):

- **Steps.** 43% of tasks take 3 LLM calls. 33% run to the cap of 7 (6 steps plus a forced finish). This gives a long tail of task lengths.
- **Context size.** The prompt is 1,684 tokens at the median and at most 2,289. At about 56 KiB per token (fp16 KV for Qwen2.5-7B), an agent ends with roughly 100 MB of KV.
- **Prefix caching.** Append-only prompting works as intended: 95.9% of prompt tokens are served from the prefix cache. A step prefills only 60 new tokens at the median and decodes 37.
- **KV composition** (by characters). The shared prefix is 76% of each context, which makes it the first target for cross-agent sharing. Of the agent-private part:

| Segment | Share of private context |
|---|---|
| Model thoughts and actions | 37% |
| Observation framing ("could not find", suggestions) | 19% |
| Retrieved non-supporting text | 18% |
| Retrieved supporting facts | 12% |
| Template / control tokens | 9% |
| Question | 6% |

So supporting facts are only 24% of observation text. `react_long` is expected to dilute them further, which supports selective KV in Phase 3.

**Concurrency smoke test.** Settings: 1.5B model, 12 tasks, tool delays on. Going from K = 1 to K = 4:

- Throughput rose from 10.3 to 21.3 tasks per minute.
- p50 latency fell from 41 s to 21 s.

At K = 1, tool calls take 36% of the time an agent holds its slot. For about a third of that time the agent's pinned KV is unused.

## 5. Remaining Phase 1 experiments (on an RTX 4060 with 8 GB, by 11/4)

| Experiment | Sweep | Question it answers |
|---|---|---|
| E1 Concurrency | K ∈ {1, 2, 4, 8} × Poisson rate | Throughput–latency curve; best fixed K on dev |
| E2 Context growth | Profile mix × `-c` budget (8k–32k) | Where context overflow and failures begin |
| E3 Tool latency | Delay scale 0 → 4× (median 0 → 3.2 s) | Share of KV that is idle during tools |
| E4 Prefix reuse | `cache_prompt` on/off × few-shot ∈ {0, 2, 6} × pinned/unpinned | Effect of prefix caching on TTFT and throughput |
| E5 KV composition | Offline, HF tokenizer | Token-level timeline: active / idle-in-tool / non-supporting KV |

These experiments test three hypotheses:

- **H1.** At high concurrency, KV held by idle agents, not compute, is the bottleneck, and it grows with tool latency.
- **H2.** Observations dominate per-agent KV, and only a small part of them is supporting evidence.
- **H3.** Re-prefill cost grows linearly with context length, so which agent gets evicted matters.

## Appendix A: Timeline and milestones

| Dates | Work | Milestone | Status |
|---|---|---|---|
| 10/21–10/28 | Project design; HotpotQA data, corpus and splits; tools, ReAct agent, profiles, workload generator; FIFO runner and instrumentation; tests; 7B quality gate | End-to-end scenario on Qwen2.5-7B; F1 0.57 vs 0.25 closed-book | Done |
| 10/29 | Fix repeated actions; rerun the gate; lock the agent version; build llama.cpp with CUDA on the RTX 4060 machine (WSL2) | Agent and model frozen for all experiments | Next |
| 10/30–11/1 | E1–E4 sweeps on the RTX 4060 | Throughput–latency curves; best fixed K | Planned |
| 11/2 | E5 composition analysis; KV timeline plot | Stacked KV-occupancy figure answering RQ1 | Planned |
| 11/3–11/4 | Figures and Phase 1 writeup | **Phase 1 due 11/4** | Planned |
| 11/5–11/18 | Phase 2: KV lifecycle manager at tool boundaries (retain / offload via slot save–restore / drop and re-prefill), cost model with online estimates, KV-aware admission; baselines: best fixed K, llama.cpp default, always-retain / always-drop / always-offload, LRU | Phase 2 comparison and ablations | Planned |
| 11/19–12/2 | Phase 3: selective KV. Scorers: oracle, recency, relevance, agent-referenced. Text-level compaction vs a KV range-drop patch to `llama-server` (`llama_memory_seq_rm` / `seq_add`) | Quality–memory Pareto curve | Planned |
| 12/3–12/9 | No-benefit cases (zero tool delay, ample KV), fairness, policy overhead; final report and reproducible scripts | Final report | Planned |

Phase 2 and 3 dates are our own plan. They will move to match the course deadlines once those are announced.
