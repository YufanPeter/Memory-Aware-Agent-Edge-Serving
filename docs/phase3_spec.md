# Phase 3: selective KV at tool-call boundaries (TRIM)

Draft, 2026-11-17. Phase 2 gave the manager three ways to give up an idle agent's KV: offload it,
drop it to the shared family prefix, or erase it. All three treat the agent's private context as
one block. Phase 3 adds the missing option from the project question ("keep it, move it, throw it
away, or keep only part of it"): keep only the useful part. HotpotQA's supporting-fact labels make
"useful" measurable, so we can bound how much KV an agent needs and test realistic selectors
against that bound.

## 1. Mechanism (text level, only controls llama-server exposes)

The ReAct prompt is append-only: `... Action n: x[y]\nObservation n: <O_n>\nThought n+1: ...`.
Observation n is read by the model exactly once at full length, in the request that produces
Thought/Action n+1. After that, only its useful sentences matter to later steps; the Thought the
model wrote right after reading it already restates what it took from it.

**Eager TRIM (default, used in every experiment).** Right after Thought/Action n+1, before the
tool call starts, send one prefill-only request (`n_predict: 1`; b11960 treats 0 as 1, and the
sampled token is never decoded) with the compacted prompt. The slot then holds the compacted
context during the tool wait; the next request is append-only again. Same trick as Phase 2's
drop-to-prefix, with a different target. Costs one short request per step (the compacted
observation plus one Thought/Action, about 100-300 tokens).

**Lazy TRIM (implemented, not used).** Rewrite the consumed observation only when building the
next request, so no extra request. Two reasons not to use it: (a) the observation just read stays
full during the tool wait, so it cuts tool-wait KV by only 6-10% against 16-26% for eager
(offline trace replay); (b) under the unchanged Phase 2 manager it can break the restore bound:
if the slot is offloaded during the wait, the restore brings back the uncompacted image, which is
larger than the gate's reservation for the shorter next prompt (the manager reserves f, while
append-only prompts guaranteed f >= image). Both variants send identical prompts at every
generation step, so their answer quality is the same.

Both run in the agent, so the Phase 2 manager is unchanged: a compacting request is an ordinary
request whose prompt is shorter than the slot's current content, which the full gate already
charges at max(s, f) (the slot keeps its old tail until the server batches the new prompt).

## 2. Selectors

Sentence level. Observation framing (`obs_meta`: page titles, "Could not find ...", lookup
counters) is always kept. A run of dropped sentences becomes one " …" marker.

| Selector | Keeps | Role |
|---|---|---|
| `none` | everything | Phase 2 baseline |
| `oracle` | HotpotQA supporting facts (`obs_sf`) | upper bound on what selection can keep for free |
| `drop` | no sentences (titles only) | lower bound: rely on the Thoughts alone |
| `referenced` | sentences sharing a salient token with the Thought/Action written after reading it | realistic, uses the model's own reasoning as the relevance signal; near-zero cost |
| `relevance:m` | the m sentences per observation most similar to the question (IDF-weighted overlap) | realistic, question-only |

Salient tokens: lowercase word tokens, stopwords (including the ReAct format's own words)
removed, length ≥ 3 or containing a digit. `referenced` also removes the tokens of the sentence's
own page title, which appear in almost every sentence of the page; `relevance:m` keeps them,
because the question usually names the page's entity (on the perf split this raised
relevance:2's supporting-fact recall from 0.42 to 0.64 at the same kept size).

## 3. Questions and measurements

- **Q1: how much KV is useful?** Per task: peak context and mean context per step with each
  selector, share of observation tokens removed, supporting-fact recall of the kept text.
- **Q2: what does trimming cost in quality?** EM/F1 vs `none` (paired TOST, margin 0.05) on
  answered questions, forced-finish rate, steps per task.
- **Q3: what does it buy under a tight budget?** Overall goodput and completion at 16k ×4 and
  8k ×1, for B1 (fixed K) and ours, with and without TRIM. Extra re-prefilled tokens per task.

## 4. Protocol (same rules as Phase 2)

All TRIM runs are eager.

1. **Selector choice on perf** (`p1_long_200`, ample memory: 64k ×1, B1 K = 8, 2 reps;
   `configs/sweeps/p3_select.yaml`): none, oracle, drop, referenced, relevance:2, relevance:3.
   Rule, fixed before any run: the chosen realistic selector is the one among referenced,
   relevance:2 and relevance:3 with the smallest mean peak context whose mean F1 is within 0.03
   of none's; if none qualifies, the realistic selector with the highest F1, reported as such.
   **Outcome (2026-11-18):** every realistic selector qualified (none's mean F1 0.588; referenced
   0.610, relevance:2 0.619, relevance:3 0.626). Mean peak context: none 3,140, referenced 2,664,
   relevance:2 2,512, relevance:3 2,519 (oracle 2,471, drop 2,461). **relevance:2 is chosen**
   (smallest peak; relevance:3 is within 7 tokens and is reported alongside).
2. **Priors and knobs.** KV admission plans with peak priors; untrimmed priors would overestimate
   trimmed peaks. Refit them per trim mode from step 1's runs (same fitting script, same filter):
   `configs/kv_priors_trim_{oracle,relevance2}.json` (react_long q0.9 peak 4,904 untrimmed →
   3,254 oracle, 3,276 relevance:2). B1's fixed K is retuned on perf per trim mode at the two
   tight points (`configs/sweeps/p3_tune.yaml`; 16k ×4: K 4, 6, 8, 10; 8k ×1: K 2, 3, 4, 5; the
   Phase 2 rule, applied by `scripts/pick_b1.py`). Each grid has one K more than Phase 2's, added
   before any tuning run, because TRIM lowers per-agent KV. Ours keeps its Phase 2 knobs (omega 1
   at both points; k = K_max: 16 at 16k ×4, 8 at 8k ×1), so the comparison isolates TRIM.
3. **Dev evaluation** (`dev_long_200`, 3 reps): {B1 at its retuned K, ours} × {oracle, chosen}
   at 16k ×4 and 8k ×1, against the Phase 2 untrimmed cells; quality at ample memory on dev for
   none, oracle, chosen and drop (2 reps).
4. Report as in Phase 2: overall goodput, completion, Welch p plus steady-CI rule, F1 TOST
   (margin 0.05) on answered questions; TRIM's own costs (trim requests per task, their
   prefill tokens, extra time).

## 5. Not in scope / stretch

- KV-level range drop (a llama-server patch using `llama_memory_seq_rm` + `seq_add` to remove a
  token range without recompute) is approximate (later tokens were computed attending to the
  dropped span) and needs a runtime change. Stretch goal only; the text-level version answers the
  question with the controls llama-server already has.
- Trimming older observations again, or summarising them with the model, is out of scope.

## 6. Outcome (2026-11-18)

Tables: `docs/phase3/tables.md` (`analysis/phase3_report.py`); figures: `docs/figures/phase3/`
(`analysis/phase3_figures.py`); write-up: `docs/final/report.tex` §IV-D, IV-E.

- **Q1/Q2, ample memory (P1, P2).** No selector lowered F1. Dev: oracle +0.053, relevance:2 +0.039,
  drop +0.033 (all 90% CIs above 0; "not equivalent" in the TOST sense because higher). KV held when
  a tool call starts −25 to −28% on dev (−21 to −31% on perf), per-task peak −15 to −22%. Forced
  finishes 15–18% → 8–12%. Cost: 1.1–1.4 compaction requests and 124–178 prefilled tokens per task
  (401 for referenced), median 70–100 ms each.
- **B1 retune (P3).** 16k ×4: K 4 → 6 for both modes; 8k ×1: K stays 2 (K = 3 completes 86.5–91%).
- **Q3, tight budgets (P4, P5; dev, 3 reps).** B1+TRIM vs B1★: +52.5% (oracle) and +45.8%
  (relevance:2) at 16k ×4, both S; +12.5% and +8.0% at 8k ×1 (n.e.). Ours+TRIM vs untrimmed ours:
  naive −3.5% / −9.3% at 16k ×4 (n.e.), −12.6% / −18.5% at 8k ×1 (S); fixed +6.3% / +2.7% and
  −3.4% / −2.9% (all n.e.).
- **Deviation: the shrink-fast fix is post hoc.** In the first ours+TRIM runs (points without
  `_sf`), compaction requests queued in FIFO order at the full gate, and 19–28% of give-up
  operations hit a slot whose owner had a compaction waiting (51–112 restore-bound violations per
  run). `KVConfig.shrink_fast` (default true) dispatches a request that only shrinks its owner's
  resident slot at once and never takes such a slot as a victim; the `_sf` points rerun §4 step 3
  with it. Both are reported; the naive runs are reproducible with `shrink_fast: false`.
- **Why ours gains little from TRIM.** The refit priors let admission run ~50% more agents (time-mean
  10.0 vs 6.8 at 16k ×4, 4.8 vs 3.2 at 8k ×1); give-ups roughly double and re-prefilled tokens per
  task rise 3.4–3.7×. With TRIM, B1 beats ours at 16k ×4 (−6.8% / −5.9%, n.e.) and ours stays ahead
  at 8k ×1 (+8.3% n.e., +13.5% S).
- **Not run.** Ours' ω was not retuned under TRIM (this spec fixed its Phase 2 knobs), no TRIM run
  used the test split, and the KV-level range drop (§5) was not attempted.
