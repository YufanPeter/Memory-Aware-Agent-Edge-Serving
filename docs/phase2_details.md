# Phase 2 Technical Notes: Agent-Level KV Lifecycle at Tool-Call Boundaries

Design, runtime probes, implementation and evaluation behind the Phase 2 report. The design and
the pre-registered experiment plan are in `phase2_spec.md`; the generated tables T1–T6 are in
`phase2/tables.md` (from `analysis/phase2_report.py`); the figures are in `figures/phase2/`
(from `analysis/phase2_figures.py`, §10); raw data is in `results/`. All experiments are
finished. Numbers not printed in `tables.md` (B1 K = 5 at 16k ×1, and the extra comparisons
marked "computed from `runs.csv`") come from the same run directories through the same
functions (`analysis.phase2_report.delta`, `steady_ci`, `f1_tost`; seeded bootstrap, 2,000
replicates).

## 1. Question and contribution

Phase 1 found that under a bounded KV budget, the fixed-concurrency baseline must size K for the
peak contexts of all K agents at once (contexts grow 2–4× during a task), because over-committed
active requests all fail together. Agents in tool calls hold 36–44% of the KV at those small K.
Phase 2 asks:

**RQ2: can an agent-level KV manager complete more correct tasks per minute than tuned fixed-K
baselines at the same GPU KV budget, without losing quality, using only the controls llama.cpp
exposes?** At each tool-call boundary it decides whose idle KV leaves the pool, how, and when.

The mechanism has three layers. Each one can be switched off on its own (`configs/kvm.yaml`):

1. **Exact KV ledger and dispatch gate.** The client keeps an upper bound on the cells the server
   holds for every slot and sends a request only if the pool cannot overflow. With the full gate,
   an overflow ("Context size has been exceeded") or a server purge is excluded by construction,
   given that `/tokenize` is exact and charges fall only after responses (I1–I3, §3.8), and none
   happened in any of the 53 full-gate runs (§4.6).
2. **Peak-projection admission.** A new task is admitted if the sum of every admitted agent's
   projected peak fits an over-committed budget (1 + ω)(c − m). This replaces a fixed K that must
   cover everyone's simultaneous peak.
3. **Lifecycle at tool boundaries.** When the gate needs cells, it takes them from idle agents one
   victim at a time. The agent expected to return last goes first. Each victim is offloaded to host
   RAM, dropped to its shared prompt prefix, or erased, whichever the cost model says is cheaper.
   A returning agent may reclaim memory at once. A new task may take an idle agent's KV only
   after a pressure-gated TTL has expired.

Every cache action uses llama-server b11960's HTTP interface: `/completion` with `id_slot` and
`cache_prompt` (and `n_predict` 0 for drop-to-prefix), `/slots/{id}?action=save|restore|erase`,
`/tokenize`, `GET /slots` and `/props`. llama.cpp is not modified, and the agent code
(`edgeserve/agent.py`) is unchanged from Phase 1.

## 2. Runtime facts that shaped the design (E0b, E0c)

Each probe starts its own server (7B, f16 KV, A10G) and talks to it over HTTP only. Raw responses
and server logs are in `results/e0b/` and `results/e0c/`, and one-line outcomes are in
`results/e0b.log` and `results/e0c.log`. Source references are to llama.cpp b11960:
`server-context.cpp` (sc), `llama-kv-cache.cpp` (kv), `server-task.cpp`, `common.h`.

| Probe | Question | Finding | Consequence |
|---|---|---|---|
| idlecost (E0b) | Do idle slots slow other slots in the unified pool? | 4 decoding slots: 293 → 238.5 / 202.6 tok/s with 8 / 16 idle slots × 3k tokens; a 2k prefill 550 → 1,014 / 1,472 ms. Idle cells above the active ones cost the same (289.9 → 202.6 tok/s). | Idle KV costs every active request, not only memory. The re-prefill cost gets a span term (9.7 µs/token per 1k occupied cells). |
| highwater (E0b) | Is the cost set by occupied cells or by the highest occupied cell? | 16 idle × 3k below 4 active slots: 202.3 tok/s; idle erased while active cells stay high: 207.9; active slots re-placed low: 295.3. | The cost follows the top cell (below). Evicting idle KV saves no compute unless the top cells are freed, and the client cannot control placement. No proactive eviction. |
| truncate (E0b) | Can a slot be cut back to its shared prefix? | `/completion` with the 1,400-token prefix and `n_predict` 0: HTTP 200, prompt_n 1, cache_n 1,399, 20 ms; a 5.5k prompt on another slot then fits; the revisit reuses 1,400. | Drop to prefix is a real action: it frees the private tail and keeps the ≈1.4k ReAct prefix. |
| busy (E0b) | Save or erase on a generating slot? | Save is deferred until the stream ends (3,531 ms wall, 50 ms server); erase takes 1.4 ms. | Ops only on idle slots, 5 s op timeout, a release erases. |
| fragment (E0b) | Restore into a fragmented pool? | 3k tokens restored into holes left by erasing every other slot: 46.8 ms, cache_n 2,999, 0 purges. | A restore is feasible iff free cells ≥ its size; no extra margin. |
| cacheram (E0b) | Does `--cache-ram` keep evicted contexts? | Pinned revisits under pressure: cache_n [0, 0, 0, 0]. Unpinned A, B, C, A: [0, 2,999, 0, 2,999]. | B2 (llama.cpp's host cache) runs unpinned. `--cache-ram` is refused in pinned and managed runs (below). |
| perslot (E0b) | `--no-kv-unified` limits | 4,096 cells per slot; cross-slot restore in 38.1 ms (cache_n 2,999); a 5k prompt gets HTTP 400 "exceeds the available context size". | B3 regions are static; tasks above the region fail individually. |
| fragsave (E0c) | Is saving a fragmented slot slower? | Contiguous 3,800 tokens: save 33.8, restore 12.1 µs/token. Built under 8 concurrent decoders (3,489 tokens): save 59.0, restore 30.0 µs/token (1.7× and 2.5×). Re-prefill of the same tokens: 281.1 µs/token. | κ_s + κ_r = 88.9 < 281.1 µs/token, so offload still beats re-prefill for long tails. Offload is kept alongside drop. |
| tokenize (E0c) | Is `/tokenize` exact? | 550/550 real dev prompts (all four profiles): tok(P) = prompt_n + cache_n, max difference 0. | Prompt sizes come from `/tokenize`; margin m = 128 cells. |
| dropids (E0c) | Drop to prefix with token ids | Family prefixes of 1,421 (`react6`) and 1,633 (`react6x`) tokens. Drop: prompt_n 1, cache_n 1,420, 19.6 ms; the slot holds 1,421; the revisit reuses 1,421. 100/100 dev first prompts start with the prefix ids. | Drop sends the family's token ids; no 1-token loss. |
| opafter (E0c) | Op right after a stream ends | Save 63–68 ms wall vs 61–62 ms server; erase 1.7 ms. | Not deferred; ops follow the response with no `/slots` check. |
| disconnect (E0c) | Client closes a stream | The slot stopped and was idle after 53.7 ms holding 1,051 tokens; erase freed 1,051. | `release()` of a cancelled task erases as specified. |
| kvdebug (E0c) | Cost of `LLAMA_KV_CACHE_DEBUG=1 -lv 4` | Decode 293.5 → 292.5 tok/s, 2k prefill 569 → 569 ms; log 96 → 422 lines, but **0 `find_slot` lines**. | Cheap, but it does not print the per-ubatch used-cell lines the validation plan relied on (§4.6). |
| b2pressure (E0c) | Do B2 revisits hit the host cache under pressure? | Inconclusive: six agents' first prompts (6 × 1,450 tokens) together exceed the 8k pool, so all 6 first requests failed with HTTP 500 in both arms (`--cache-ram` 448 and 0) and no revisit happened. | B2 under pressure is measured only by the tuning and main runs (§5, §6). |
| reconrace (E0c) | Can a `/slots` snapshot predate an op already sent? | 20/20 polls sent 2–3 ms after an erase, while 8 other slots decoded, still showed the slot's 2,000 tokens; 0/20 arrived after the erase's response. Erase wall time under decode: median 17.5 ms. | Reconcile needs the per-slot version rule (§3.1). |

**Attention cost follows the high-water mark.** `llama_kv_cache::get_n_kv` returns
`GGML_PAD(cells.used_max_p1(), 256)` (kv:1255), so attention spans every cell up to the highest
occupied one, not just the occupied cells. `find_slot` (kv:≈997) restarts its search at cell 0
when `head > used + 2·n_tokens`, so new allocations fill low holes first, and the span shrinks only
when the top cells are freed. This explains both probes. In idlecost, each round sends new active
prompts, which are placed anew, so erasing the idle slots restores the full rate (290.7 tok/s). In
highwater, the active cells stay in place above the erased idle ones, and the rate stays at 207.9
tok/s until the active slots are re-placed (295.3). Phase 1 asked whether idle and stale KV slow
other agents in the unified pool. The answer is yes, by up to 31% in decode at 48k idle cells,
and an eviction removes this cost only if it lowers the top occupied cell. In the tight pools of
Phase 2 the span is at most c, so we report this tax as bounded and do not claim a compute gain
from eviction.

**`--cache-idle-slots` and the pinned NaN bug.** With `--cache-ram N > 0`, `--cache-idle-slots`
is on by default (common.h:640). Every task launch then saves all idle slots to the host-RAM prompt
cache and, with `--kv-unified`, clears them from the GPU pool (sc:3071–3086). A later request
restores the best cached entry by common prefix (server-task.cpp:1838, needs f_keep ≥ 0.25). This
is llama.cpp's built-in "offload every idle slot at every request start" policy, so we use it as
baseline B2. It does not work with pinned slots. A pinned request to a cleared slot computes
f_keep = 0/0 = NaN (computed at sc:1853), and the comparison `f_keep < 0.5` (sc:1861) is false
for NaN, so it never reloads. The pinned revisits
above (cache_n 0, 0, 0, 0) are consistent with this reading. Hence B2 runs unpinned with
`-np K`, and the runner refuses `--cache-ram ≠ 0` for any pinned or managed run.

**Two capacity limits.** The source also explains the Phase 1 collapse. When a ubatch finds no
cells, the server first purges one idle slot per retry. In unified mode it takes the first idle
slot that holds tokens, in slot-id order (sc:1911–1935). If nothing can be purged, it halves
`n_batch`. At `n_batch` 1 it fails every processing slot with "Context size has been exceeded"
(sc:4699–4745). So active requests are a hard limit, and idle or stale KV is a soft limit that
the server reclaims blindly. The b2pressure server log shows the halving sequence
(n_batch 1024 → … → 1) ending in that error.

**Ruled out.** Sharing the ReAct prefix across slots through `n_cmpl`: `server_slot::release()`
clears every child slot's prompt (sc:666–669), and no other HTTP route reaches `seq_cp`, so the
≈1.4k prefix stays duplicated per slot.

**Client fix.** The Phase 1 `RemoteProtocolError` failures were a keep-alive race: the server
closed an idle connection while the client reused it. The client now sets `keepalive_expiry` to
2 s, below the server's 5 s, and retries once if no byte was received (`edgeserve/llm.py`).
Three early P0 tuning runs used the committed Phase 1 client (revision `0667209`, no local
changes) and lost one task each to this race (`p0_baselines` c16k_k5, `p0x4_tune` c16k_k8 and
c32k_cr_k12). No run with the fixed client lost a task to it.

## 3. Design

### 3.1 Ledger and charges

For slot k the manager keeps a charge u_k, an upper bound on the cells the server holds for it.
U = Σ u_k. Here c is `-c` (read from `/props` and asserted equal to the sweep's budget) and
m = 128.

| Slot state | Charge u_k |
|---|---|
| ACTIVE (request in flight) | max(s_k, f), f = tok(P) + n_predict_eff |
| IDLE (owner in a tool), READY (owner waiting at the gate), STALE (task done), SAVING, DROPPED | s_k, the exact resident count |
| RESTORING | f of the request it serves (≥ the image size) |
| UNKNOWN (an op timed out) | max(s_k, image, f) until resolved |
| OFFLOADED, FREE | 0 |

- s_k is exact after every response: **s_k = prompt_n + cache_n + predicted_n − 1**, because the
  last sampled token is never decoded. After an op, s_k comes from the op's response
  (`n_erased`, `n_restored`, the drop's cache_n).
- Charges fall only when the server's response arrives. The max(s_k, f) term covers the time
  before the server removes a slot's old tail (sc:4357).
- tok(P) comes from the server's own `/tokenize`. On every response the manager checks that
  tok(P) = prompt_n + cache_n, and any difference is a `kv_violation`.
- **Reconcile.** `GET /slots` is polled at 1 Hz in every configuration. A snapshot can predate an
  op or request the client already sent (E0c reconrace), so each slot carries a version counter
  that is bumped on dispatch, response, op send and op completion. A slot is compared only if its
  version did not change while the poll was in flight, nothing was outstanding on it, and the
  server reports it idle. With the full gate, a server count below the ledger is a violation. With
  the active gate or no gate, it is a timestamped purge. A count above the ledger raises the
  charge and becomes an `undercount` violation if it repeats.
- **Baselines run the same manager in `observe` mode** (no gate, no ops), so B1 and B3 get
  exact KV token·s by state and purge timestamps. B2's breakdown is approximate, because the
  server clears idle slots at every launch and the client learns a request's slot only from the
  stream. Its throughput, goodput and failures are exact.

### 3.2 Gate

All waiting requests sit in one queue, ordered by priority, then by ready time. Priority 0 is a
returning agent (step ≥ 2, including a pending restore). Priority 1 is the first request of a newly
admitted task. Only the head may dispatch. One `asyncio.Lock` serializes planning, victim ops and
dispatch.

| Gate | Used by | Head h dispatches on slot k iff |
|---|---|---|
| `full` | every managed configuration (ours, ours_h0, managed_k, ablations) | U − u_k + max(s_k, f_h) ≤ c − m |
| `active` | retain configurations (gated_k, adm_retain): llama.cpp's purge reclaims idle KV | Σ_active u + max(s_k, f_h) ≤ c − m |
| `none` | B1, B2, B3 (Phase 1 behaviour) | always |

**Retain plus the full gate deadlocks.** Under retain, nothing ever evicts idle agents, so their
KV stays charged while their own returns wait at the gate. Once idle KV fills the pool, no head
can dispatch, and no idle agent can return to free anything. The retain configurations therefore
use the `active` gate and leave idle KV to the server's purge, which can always reclaim it.
`policies.py` enforces one gate per lifecycle (observe → none, server → active, managed → full).

**When the full gate fails** (deficit D > 0), the head evicts one victim at a time. It recomputes
the eligible set from the current ledger, takes the top-ranked victim, re-validates it, awaits its
op, and recomputes D. It dispatches as soon as D ≤ 0. A precomputed victim list would go stale
during 100–300 ms ops. A head that has evicted is *committed*: a newly arrived priority-0 waiter
queues behind it, so the freed cells are not taken from it. The commitment lapses when the head's
eligible set can no longer cover its deficit.

**Infeasible requests.** Let n_predict_eff = min(n_predict, c − m − tok(P)). A request is
`kv_infeasible` only if tok(P) ≥ c − m, and the task then fails. This matches what the server
itself cannot run: it rejects prompts of ≥ n_ctx tokens (sc:4081) and stops generation at
n_ctx (sc:2188). A request sent with n_predict_eff < n_predict is logged `kv_truncated`. One
request in all the main runs was truncated (8k ×1 ours_h0, rep 0).

### 3.3 Admission

Admission hands out FIFO tickets with head-of-line blocking (`edgeserve/admission.py`). Task j is
admitted iff

```
(a) n_admitted < K                  (fixed K, or the compute cap K_max for KV admission)
(b) the gate queue is empty and every admitted task has sent its first request
(c) Σ_i R̂_i + R̂_j ≤ (1 + ω)(c − m),   R̂_i = max(s_i + Ĝ, P̂0_p + Ĝ),  Ĝ = Ĝ_0.9(p, n_i)
```

- Ĝ_q(p, n) is the q-quantile of (final peak − size after request n) for profile p. It was
  fitted by `scripts/fit_kvm_priors.py` from Phase 1 runs with no overflow failure (E1 at 131k,
  E2 c64k, E2 c32k K = 8), using tasks with status ok or forced (`configs/kv_priors.json`). For
  `react_long`, P̂0 = 1,449 and Ĝ_0.9 = 3,460 at step 0, from 810 tasks. In-sample coverage at
  q = 0.9 is 0.90 for every profile.
- Offloaded and dropped agents keep their reservation, because they will return.
- (b) keeps a burst of admissions from slipping through the `/tokenize` round trip between
  admission and the first request, and adds no new agents while others wait for memory.
- ω is an efficiency knob, independent of the host tier, because safety comes from the gate. It
  is tuned per point over {0.5, 1, 2}.
- K_max is the best K with ample memory: 8 at ×1 (Phase 1 E1) and 16 at ×4 (131k ×4 on perf:
  K = 16 gave 17.22 correct/min vs 15.39 at K = 8, both with 200/200 completed; T1).

### 3.4 Lifecycle actions

| Action | Control | Frees | Cost on return |
|---|---|---|---|
| retain | none | 0 | 0 |
| offload | `save` to `/dev/shm/slots`, then `erase` | s | a restore, (κ_s + κ_r)·s; save and restore run on the server's main loop and stall every slot |
| drop to prefix | `/completion` with the family's prefix token ids, `n_predict` 0, same slot | s − L_f | re-prefill of s − L_f |
| erase | `erase` | s | re-prefill of s |

- **Family prefixes.** At start, the manager tokenizes two dummy first prompts per family and
  takes their longest common prefix of token ids. `react6` (short and long) gives L_f = 1,421 and
  `react6x` (explain) gives 1,633. `oneshot_long` has no family and is erased.
- **Host tier.** The host tier holds at most h·c·57,360 B (a slot file stores 57,360 B per
  token). With h = 1 that is 448 / 896 / 1,792 MiB at 8k / 16k / 32k. Offload is not used when it
  would exceed the tier. `ours_h0` sets h = 0, so it never offloads and uses no memory beyond the
  GPU pool.
- **Offload race.** If the victim's tool ends during the save, the erase is skipped and the file
  is deleted (`wasted_save`). The slot is intact.
- **Restore** happens inside the owner's gated dispatch. The gate reserves the request's f first.
  A failed restore becomes a re-prefill (none failed in any run). There is no prefetch: the
  agent's next request follows `tool_end` with no await in between.

### 3.5 Victim order

The eligible set is built in class order (`policies.rank_victims`):

1. the head's own slot, if it still holds a finished task's longer context;
2. STALE slots;
3. agents in `search`, ranked by expected residual tool time (largest first), ties broken by
   larger s;
4. agents in `lookup` (residual ≈ 0.1 s × scale, effectively last);
5. READY agents behind the head, latest-ready first. Only priority-0 heads use this class; the
   no-deadlock argument needs it.

The residual estimate is online and per tool, from the tool durations observed in the same run:
the mean of (d − elapsed) over observed durations d > elapsed. Below 30 observations of a tool, it
is blended with a weak prior that does not depend on the delay scale (lognormal, median 1 s,
σ 1.0, weight 10 observations). The manager never reads the true delay, except in the
`clairvoyant` ablation.

### 3.6 Action per victim

The cost model (`CostModel` in `kvm.py`) prices each action in server-seconds:
c_off = (κ_s + κ_r)·s and c_drop = (t_pf + t_span · span/1k)·(s − L_f). The priors are
κ_s = 35 µs, κ_r = 15 µs and t_pf = 220 µs per token, with t_span = 9.7 µs per token per 1k
occupied cells (E0a, E0b). They are refit online by an EWMA (α = 0.2) from the server's own
timings (`save_ms`, `restore_ms`, `prompt_ms`, minus the span term). `choose_action` then picks:

- offload iff the host tier has room and c_off < c_drop; otherwise drop to the prefix;
- erase only for a slot without a family, a slot already at its prefix, or, for priority-0 heads
  only, when every eligible victim is already at its prefix and D is still positive;
- STALE KV is never offloaded, since nobody returns for it. It is dropped to its prefix (a new
  task of the same family can reuse that), and erased only if D needs those cells too.

With the priors and an empty pool (span term 0), a `react_long` slot of s = 3.5k tokens is
offloaded (175 vs ≈ 460 ms), and a `react_short` slot of s = 1.8k is dropped (90 vs 83 ms, with
L_f = 1,421). In real runs the online fit moved. In the three G0 runs (§4.6, one run each), the
fitted κ_s + κ_r was larger at the larger budgets (83 / 146 / 245 µs/token at 8k ×1 / 16k ×4 /
32k ×4), which narrows offload's advantage over re-prefill (t_pf ≈ 248–286 µs/token). The runs
also differ in delay scale, and the cause was not isolated.

### 3.7 Timing: pressure-gated TTL and the two-tier rule

At `tool_start` the manager freezes a ski-rental break-even for the agent:
B_i = (A + S_i)·min(c_off, c_drop) and T\*_i = B_i / S_i. Here A is the active tokens at that
moment, since an op stalls them too, and S_i is the agent's resident tokens. Rent accrues as
ρ_i = ∫ S_i · p(t) dt, where p(t) = 1 only while a priority-1 head is blocked at the gate (50 ms
tick). The agent *expires* when ρ_i ≥ B_i.

- **Priority-0 heads (returning agents) ignore TTLs** and reclaim on demand. A returning agent is
  work in progress, and blocking it only delays a finish that would free memory.
- **Priority-1 heads (new tasks)** may take only free cells, STALE slots and *expired* idle agents.
  They never erase an agent already dropped to its prefix. This is the ski-rental choice: admit
  new work now at cost B, or wait for a natural release.
- Without pressure nothing is evicted, and nothing is ever evicted proactively. E0b shows that
  eviction lowers attention cost only if it frees the top occupied cells, which the client cannot
  control.

The ablations change exactly one of these rules: `lazy` (T\* = 0), `nosteal` (T\* = ∞),
`ttl_off` (B4: offload proactively after 1.25 × the online median `search` time), `always_off`
(B5: offload at every tool start) and `clairvoyant` (evict iff the true remaining delay > T\*).

### 3.8 Invariants and progress

- **I1** U ≤ c − m at all times (full gate). **I2** the server's cells for slot k ≤ u_k, because
  `/tokenize` is exact, generation is bounded by `n_predict_eff`, and charges fall only after
  responses. **I3** I1 and I2 give used ≤ c − m. `find_slot` is non-contiguous (kv:1012–1060),
  so it never fails, and there is no purge and no HTTP 500. With the full gate, a purge or an
  overflow is a `kv_violation`.
- **I4** Requests and ops on a slot are mutually exclusive. Ops are manager-owned tasks, awaited
  through `asyncio.shield`, so a cancelled head abandons only its wait while the victim's op
  completes and its charge stays exact. An op that exceeds its 5 s timeout leaves the slot UNKNOWN
  with an upper-bound charge until a `/slots` poll and an erase resolve it.
- **I5** Victims are never active and never the head. An agent is evicted at most twice per idle
  period (a drop, then an erase of the prefix), and the second only by a priority-0 head. Both
  count as one give-up.

**No deadlock, in brief.** A priority-0 head's eligible set contains every non-active resident
cell except its own. Active requests cannot fail (I3) and are bounded by `n_predict`, so they
finish and join that set. Evicting it all leaves U = u_k, so the head either dispatches or is
infeasible. A priority-1 head runs only when no priority-0 waiter exists. While it waits, every
idle agent's rent grows, so each one either expires (finite B) or returns and finishes within
`max_steps` + 2 requests. FIFO order within each class, with no bypass, rules out starvation. The
full argument, including livelock, is in `phase2_spec.md` §2.4.

### 3.9 Implementation and tests

| File | Lines | Content |
|---|---|---|
| `edgeserve/kvm.py` | 1,685 | `KVManager` (ledger, gate, victims, ops, TTL, reconcile, observe mode incl. B2), `CostModel`, `ManagedLLM` wrapper |
| `edgeserve/policies.py` | 201 | presets, online residual model, `rank_victims`, `choose_action`, `break_even` (pure functions) |
| `edgeserve/admission.py` | 165 | ticketed FIFO admission (fixed K, KV, duty), slot pool |
| `edgeserve/mockpool.py` | 655 | `MockPoolLLM`: cell-level llama-server mock (id-order purge, fail-all on overflow, deferred and stalling ops, stale `/slots`, fault injection) |
| `edgeserve/llm.py`, `runner.py`, `events.py` | edits | `SlotControl`, keep-alive fix, `--policy`/`--kvm`, event listeners, server-log counters |
| `scripts/fit_kvm_priors.py`, `analysis/{phase2_report,stats,server_log}.py` | 225 + 1,163 | admission priors, T1–T6, statistics |
| `analysis/phase2_figures.py` | 519 | F1–F4 (F2 also as the two report panels), KV state and timeline figures (§10) |
| `bench/e0b_runtime.py`, `bench/e0c_runtime.py` | 293 + 606 | runtime probes (§2) |

The manager attaches through a wrapper around the LLM client and an `EventLog` listener for the
`tool_start`/`tool_end` events the agent already emits, so agent traces and prompts are unchanged.

`uv run --no-sync pytest --collect-only -q` collects **167 tests**: 18 from Phase 1 and 149 new
ones (`test_kvm` 54, `test_analysis` 42, `test_mockpool` 21, `test_phase2_report` 11,
`test_policies` 9, `test_admission` 7, `test_llm` 5). All pass against the mock (52 s on the
Mac). Among them: the gate never overflows (20 seeds, 4k pool, 12 agents growing 700 tokens per
step); the ledger equals the mock's cells after every response and op; no deadlock when all memory
is held by READY agents; a priority-1 head never evicts an unexpired agent; cancellation in the
middle of a gate wait, a request or a victim's op leaves no queue entry and no orphan file; the
reconcile race; op timeouts; and `observe` + fixed K reproduces the Phase 1 request sequence
byte for byte.

## 4. Method

### 4.1 Setup

Same runtime as Phase 1: llama.cpp `llama-server` b11960 (commit 8642541, CUDA) on an AWS
g5.xlarge (A10G, 23,028 MiB), Qwen2.5-7B-Instruct Q4_K_M, f16 KV (56 KiB per token), temperature
0. Flags: `--kv-unified -c <budget> -np 64 -fa on -b 2048 -ub 512 --cache-ram 0
--slot-save-path /dev/shm/slots`, except B2 (unpinned, `-np K`, `--cache-ram` h·c) and B3
(`--no-kv-unified`). The workload is Phase 1's long-context mix: 70% `react_long`, 30%
`react_short`, 200 tasks, all released at t = 0 (closed loop). Tool delays are lognormal with
median 0.8 s, σ 0.6, cap 5 s (`lookup` 0.1 s), scaled by ×1 or ×4 (median 3.2 s), and seeded per
(seed, task, step, tool). The sweep driver restarts the server for every run.

### 4.2 Configurations

| ID | Preset | Admission | Gate | Lifecycle | Knob (tuned on perf) |
|---|---|---|---|---|---|
| B1 | `observe` | fixed K | none | llama.cpp purge (pinned, retain) | K |
| B2 | `observe` | fixed K | none | llama.cpp host prompt cache: saves and clears every idle slot at each launch; unpinned, `-np K`, `--cache-ram` h·c | K |
| B3 | `observe` | fixed K = `-np` = c/8192 | none | static per-slot regions | — |
| ours | `ours` | KV (ω), K_max | full | pressure-TTL + on-demand, cost-chosen action, online delay model, h = 1 | ω |
| ours_h0 | `ours_h0` | KV (ω), K_max | full | as ours, h = 0 (drop and erase only) | ours' ω |
| gated_k | `gated_k` | fixed K | active | llama.cpp purge | K |
| adm_retain | `adm_retain` | KV (ω) | active | llama.cpp purge | ω |
| managed_k | `managed_k` | fixed K | full | as ours | K |
| ablations | `lazy`, `nosteal`, `act_erase`, `act_offload`, `victim_lowid`, `always_off` (B5), `ttl_off` (B4), `clairvoyant` (R1), `oracle_prior` | KV | full | ours with one rule changed (§3.7) | ours' ω |

### 4.3 Splits and tuning protocol

- **Tuning uses the perf split** (`workloads/p1_long_200.json`), 1 run per knob: K for B1 and B2
  (`p0_baselines`, `p0x4_tune`, B2 at 8k ×1 in `p2_tune`, plus the Phase 1 E2 runs on the same
  workload), ω for ours
  (`p2_tune`), and the 2×2 knobs (`p2_tune`, and `p2_tune_v2`, which writes into the same
  `results/p2_tune/` directory). Tuning runs are never reported as results, and no delta is
  computed from them.
- **Tuning rule.** Take the knob with the highest overall goodput among those completing ≥ 95% of
  tasks. If none does, take the one with the highest completion (★⁻ in the tables). For B1 the
  main table also shows the highest-goodput K regardless of completion (†), since that is what an
  operator tuning only for speed would pick.
- **Evaluation uses the dev split** (`workloads/dev_long_200.json`, held out from tuning): fresh
  runs after the code freeze, 2 reps per cell (`p2_main`). A third rep was added at 16k ×4 and
  32k ×4 for every configuration except B2 at 32k ×4 (`p2_main_rep3`, same sweep name, so it
  writes `rep2` next to `p2_main`'s runs). B2 at 8k ×1 and B1 K = 5 at 16k ×1 (2 reps each) ran
  with the ablations (`p2_ablate`). The 2×2 cells ran 3 reps each (`p2_2x2`), and the ablations
  1 rep each, with 3 more reps for `act_offload` and `always_off` (`p2_2x2`).
- **Points:** 16k ×4 and 8k ×1 (headline), 16k ×1 and 32k ×4 (pre-registered as small or no
  benefit).

### 4.4 Metrics

- **Primary: overall goodput and completion.** Overall goodput is EM-correct tasks per minute over
  the whole run, with failed and `kv_infeasible` tasks counted as wrong. Completion is answered
  tasks / 200. Phase 1's steady rates (10th–90th percentile completion window) are secondary,
  because a run that fails tasks frees KV and its steady window looks faster than a run that
  finishes everything.
- **Mechanism columns.** Failures by status; **agents** (time-mean agents in flight);
  **give-ups** (give-up events, one per task and idle period, so a drop then an erase counts once,
  with the victims by action off/drop/erase summed from the per-task counters, so the action
  counts can exceed the events); **recompute** (re-prefilled tokens per task: for each pair of
  consecutive requests of a task, the cells held after the first minus the next one's cache_n,
  one estimator for every run); **op time** (slot-op server time / makespan); and server purges
  and overflow events from `server.log`.
- **No latency claim.** Closed-loop runs release every task at t = 0, so latency is completion
  time.

### 4.5 Statistics

- **Welch t-test on per-run overall goodput.** With n = 2–4 runs per side, p is indicative only.
- **Hierarchical bootstrap 95% CI** of the steady-goodput Δ (resampling runs across reps, then
  window completions within each run; `analysis/stats.py`), secondary.
- **Quality: paired per-question F1 TOST at margin 0.05**, averaged across reps, over the questions
  answered in every run on both sides (90% CI), so a completion gap is not read as a quality gap.
  The version that scores failed questions 0 is shown too, without a verdict. The test is
  calibrated with a B1-vs-B1 null (rep 1 vs rep 0) at every B1 cell.
- CUDA outputs are not batch-invariant (Phase 1: ±0.012 F1 over three identical 200-task runs), so
  F1 differences below ~0.03 are noise.
- **Decision rule for a goodput difference.** With n = 2–4 per side, a difference is called
  *supported* only if Welch p < 0.05 on overall goodput **and** the steady-goodput bootstrap CI
  excludes 0. Otherwise it is reported as *not established*, whatever the direction of the means
  or the per-run ranges. The same rule is applied to every comparison in §6 and §7: the main
  table, the 2×2 effects and the ablations with ≥ 2 runs. A preset with 1 run gets no test and
  is compared only with ours' rep range.

**As-run vs the spec.** The spec (§8.2) pre-registered ≥ 3 dev reps at the headline cells, Welch
on steady goodput, a test-500 confirmation, and further blocks (delay sweep, default mix, Poisson
latency, the iso-GPU-memory 24k test, the misspecified-prior test). The as-run plan has 3 reps at
16k ×4 and 32k ×4 and 2 at 8k ×1 and 16k ×1 (so the second headline point, 8k ×1, has fewer than
pre-registered), uses overall goodput as the primary metric (above), and covers the four points
of §6, the 2×2 at 16k ×4, the ablations and the 131k ×1 check of §7. The other blocks were not
run. The knob selection on perf already ranked by overall goodput and completion, and that rule is written in `p2_main.yaml`,
which fixed the knobs before the dev runs. The sources do not record whether overall goodput
became the primary *test* metric before or after the dev results were seen. For that reason the
pre-registered steady Δ (with its bootstrap CI) is reported next to it in every comparison row
of §6, and the decision rule above requires both.

### 4.6 Validation (G0)

Before the ω and 2×2 tuning (after the B1/B2 baseline tuning), one run per budget of ours (ω = 1, K_max = 8 / 16 / 16) on dev at 8k ×1, 16k ×4
and 32k ×4, with `LLAMA_KV_CACHE_DEBUG=1 -lv 4` (`g0_shakedown`; validation, not a result):

| Run | completed | overflows | server purges | kv_violation | reconcile errors | max U / (c − m) | ops (save / restore / drop / erase) |
|---|---|---|---|---|---|---|---|
| 8k ×1 | 199/200 (1 kv_infeasible) | 0 | 0 | 0 | 0 | 8,063 / 8,064 | 131 / 94 / 149 / 177 |
| 16k ×4 | 200/200 | 0 | 0 | 0 | 0 | 16,256 / 16,256 | 96 / 84 / 171 / 173 |
| 32k ×4 | 200/200 | 0 | 0 | 0 | 0 | 32,639 / 32,640 | 116 / 105 / 189 / 208 |

No save, restore, drop or erase failed or timed out, no restore failed, and no `/tokenize` call
fell back. The pool ran at its cap, without overflow. One planned check could not be done: this
build printed no `find_slot` lines with the debug flags (E0c kvdebug; 0 lines in all three G0 logs),
so "server used cells ≤ U" is supported only indirectly, by zero purges and overflows and by
clean reconciles.

The same safety counters hold across all 53 full-gate runs (3 G0; 16 tuning runs: 12 ours,
4 managed_k; 15 ours and ours_h0 main runs; 10 in `p2_ablate`: 9 ablations and ours at 131k ×1;
9 in `p2_2x2`: managed_k, act_offload and always_off): 0 overflows, 0 server purges,
0 `kv_violation` events, 0 failed restores, and max U ≤ c − m in every run. The 13 active-gate
runs on the fixed code (§7.1: gated_k K = 4 / 6 / 8 / 10, adm_retain ω = 0.5 / 1 / 2 on perf, and
the 6 dev reps) logged 0 overflows; only the two pre-fix adm_retain runs overflowed.

## 5. Baseline tuning: what fixed K and B2 can do

Summary of T1 (perf split, 1 run per knob; selection runs, not results):

| Budget × delay | B1 rule pick | B1 highest goodput | Larger K | B2 pick (host tier) | B3 |
|---|---|---|---|---|---|
| 8k ×1 | K=2: 89.0%, 10.24 ★⁻ | K=3: 62.0%, 10.89 | K=4: 24.0%; K=16: 0% | K=2 (448 MiB): 89.0%, 9.81 ★⁻ | — |
| 16k ×1 | K=3: 100%, 16.35 | K=5: 80.0%, 18.74 | K=6: 61.5%; K=8: 5.5% | K=4 (4096 MiB): 94.0%, 14.56 ★⁻ | K=2: 100%, 11.56 |
| 32k ×1 | K=8: 100%, 21.68 | same | K=10: 90.0%; K=12: 76.0%; K=16: 4.0% | K=8 (4096 MiB): 100%, 15.01 | K=4: 99.0%, 17.35 |
| 8k ×4 | K=2: 94.0%, 4.70 ★⁻ | K=4: 61.5%, 6.75 | K=6: 1.0% | K=3 (448 MiB): 78.0%, 5.87 ★⁻ | — |
| 16k ×4 | K=4: 98.0%, 8.73 | K=6: 81.0%, 10.51 | K=8: 46.0%; K=10: 15.0% | K=6 (896 MiB): 74.5%, 9.55 ★⁻ | — |
| 32k ×4 | K=8: 100%, 15.13 | same | K=12: 77.0%; K=16: 18.5% | K=8 (1792 MiB): 100%, 13.55 | — |
| 131k ×4 | K=16: 100%, 17.22 | same | (K=8: 100%, 15.39) | — | — |

Entries are completion and overall goodput (correct/min).

- **The safe fixed K is about budget / 4k tokens, at both delay scales**: K = 2 at 8k, 3–4 at 16k,
  8 at 32k. Longer tool delays did not raise it. The spec's active-request model predicted
  K ≈ 4 / 8 / 12 at ×4, on the grounds that llama.cpp purges idle pinned slots on demand. But at
  16k ×4, K = 8 completed 46%, and 32k ×4 K = 12 completed 77%. Purging idle slots does not stop
  the overflows.
- **Completion falls steeply past the safe K.** B1's highest-goodput K drops 19–39% of tasks
  (16k ×1 K=5: 80%; 16k ×4 K=6: 81%; 8k ×1 K=3: 62%; 8k ×4 K=4: 61.5%), and doubling K from the
  safe point leaves 4–62%.
- **B2 is slower than B1 at the same K, except in collapsed runs.** At equal K, B2's goodput was
  below B1's in 12 of the 14 tuning pairs. The two exceptions are runs where both collapsed
  (16k ×1 K=8: 1.97 vs 1.72, 5.5% completed each; 32k ×4 K=16: 5.23 vs 4.86, 18.0% vs 18.5%).
  Where both complete everything, B2 is slower: 32k ×1 15.01 vs 21.68 correct/min (−31%),
  32k ×4 13.55 vs 15.13 (−10%). Under pressure it fails like B1 (16k ×4: 74.5% at K=6 vs B1's
  81.0%), though at two equal-K pairs it completed more (8k ×4 K=3: 78.0% vs 74.0%; 32k ×4 K=12:
  88.5% vs 77.0%). Comparing the picks rather than equal K, B2's pick has higher goodput than
  B1's rule pick at both tight ×4 points (8k ×4: 5.87 at 78.0% vs 4.70 at 94.0%; 16k ×4: 9.55 at
  74.5% vs 8.73 at 98.0%), by running a larger K that drops more tasks; on dev at 16k ×1 it is
  also above B1★ (17.45 vs 16.91, §6). Its re-prefill is higher at the picks (286–625 tokens per task vs
  1–56 for B1; T2). This is consistent with llama.cpp saving and clearing every idle slot at each
  launch (§2). The cause of B2's misses was not traced further (the E0c probe was inconclusive).
- **Per-slot regions (B3) lose at tight budgets.** Only c/8192 regions fit: 2 at 16k (−29%
  goodput vs B1's pick) and 4 at 32k (−20%). B3 is therefore not carried into the main runs.
  Phase 1's per-slot advantage needed a 131k pool.

**ω tuning (T2).** The picks are ω = 1 at 8k ×1 and 16k ×4, and ω = 0.5 at 16k ×1 and 32k ×4.
A larger ω admits more agents and pays for them in give-ups, re-prefill and op time. At 16k ×4,
ω = 0.5 / 1 / 2 gave 5.0 / 6.8 / 9.5 agents, 33 / 240 / 484 give-ups, 54 / 562 / 1,572 recompute
tokens per task and 2.5 / 12.4 / 19.2% op time, while goodput stayed within 10.96–11.36 (all
200/200 completed). At 16k ×1 and 32k ×4, the larger ω (ω = 1) coincided with 20.2% and 24.3%
op time and 29% and 26% lower goodput than ω = 0.5. At 16k ×4, goodput was flat across ω
although op time rose to 19.2%. The best ω differs between points. Each knob had one run.

## 6. Main results (T3: dev split, fresh runs, 2–3 reps)

Means over reps, with min–max in parentheses. Reps: 3 at 16k ×4 and 32k ×4 (B2 at 32k ×4: 2), 2
at 8k ×1 and 16k ×1. Every run has 200 tasks. Failures are the mean per run by status. "–" means
no lifecycle action (observe mode: no give-ups, no slot ops). Throughput is answered tasks per
minute over the whole run. B1 ★ is the perf tuning pick and † its highest-goodput K. At 16k ×1,
B1 K = 4 is neither (it completed 92.0% on perf). B1† K = 5 at 16k ×1 ran in `p2_ablate` and enters T3 and F1
through `MAIN_EXTRA` in `analysis/phase2_report.py`.

| Point | Config | Knob | Reps | Completion % | Failures / run | Goodput (correct/min) | Throughput | F1 | Agents | Give-ups (off/drop/erase) | Recompute tok/task | Op time % | Purges / overflows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16k ×4 | B1★ | K=4 | 3 | 98.8 | ctx_overflow 2.3 | 10.97 (10.86–11.11) | 21.5 | 0.621 | 3.9 | – | 31 | – | 3 / 1 |
| | B1† | K=6 | 3 | 88.5 | ctx_overflow 23 | 13.95 (13.19–15.22) | 26.6 | 0.569 | 5.9 | – | 187 | – | 25 / 4 |
| | B2 | K=6, 896 MiB | 3 | 85.3 | ctx_overflow 29.3 | 13.23 (12.47–14.62) | 24.6 | 0.557 | 5.9 | – | 626 | – | 0 / 5 |
| | **ours** | ω=1 | 3 | **100.0** | 0 | **14.66** (13.98–15.11) | 28.2 | 0.635 | 6.8 | 208 (89/118/45) | 472 | 10.5 | 0 / 0 |
| | ours_h0 | ω=1 | 3 | 100.0 | 0 | 13.42 (12.55–13.85) | 26.5 | 0.618 | 6.8 | 223 (0/223/89) | 1,606 | 3.1 | 0 / 0 |
| 8k ×1 | B1★⁻ | K=2 | 2 | 91.5 | ctx_overflow 17 | 12.71 (12.65–12.77) | 24.7 | 0.586 | 2.0 | – | 5 | – | 2 / 8 |
| | B1† | K=3 | 2 | 61.3 | ctx_overflow 77.5 | 13.51 (13.30–13.72) | 24.0 | 0.420 | 3.0 | – | 84 | – | 30 / 26 |
| | B2 | K=2, 448 MiB | 2 | 91.0 | ctx_overflow 18 | 12.25 (11.99–12.51) | 22.6 | 0.598 | 2.0 | – | 233 | – | 0 / 9 |
| | **ours** | ω=1 | 2 | **98.8** | kv_infeasible 2.5 | **16.42** (16.40–16.43) | 30.2 | 0.654 | 3.2 | 185 (98/86/62) | 558 | 9.1 | 0 / 0 |
| | ours_h0 | ω=1 | 2 | 99.2 | kv_infeasible 1.5 | 14.74 (14.64–14.85) | 28.4 | 0.630 | 3.2 | 188 (0/188/89) | 1,692 | 1.2 | 0 / 0 |
| 16k ×1 | B1★ | K=3 | 2 | 100.0 | 0 | 16.91 (16.63–17.20) | 33.0 | 0.620 | 3.0 | – | 9 | – | 2 / 0 |
| | B1 | K=4 | 2 | 98.0 | ctx_overflow 4 | 21.83 (21.81–21.85) | 42.4 | 0.619 | 4.0 | – | 9 | – | 2 / 1 |
| | B1† | K=5 | 2 | 95.0 | ctx_overflow 10 | 24.03 (23.79–24.27) | 46.1 | 0.601 | 4.9 | – | 30 | – | 5.5 / 2 |
| | B2 | K=4, 896 MiB | 2 | 98.0 | ctx_overflow 4 | 17.45 (17.22–17.69) | 34.2 | 0.618 | 4.0 | – | 311 | – | 0 / 1 |
| | **ours** | ω=0.5 | 2 | **100.0** | 0 | **22.21** (22.03–22.39) | 42.5 | 0.634 | 5.1 | 42 (24/18/11) | 92 | 5.5 | 0 / 0 |
| 32k ×4 | B1★† | K=8 | 3 | 100.0 | 0 | 18.54 (18.07–19.13) | 35.5 | 0.632 | 7.8 | – | 10 | – | 1 / 0 |
| | B2 | K=8, 1792 MiB | 2 | 100.0 | 0 | 16.10 (15.71–16.49) | 31.4 | 0.632 | 7.8 | – | 668 | – | 0 / 0 |
| | **ours** | ω=0.5 | 3 | **100.0** | 0 | **20.16** (19.49–20.76) | 39.0 | 0.637 | 10.4 | 15 (11/4/3) | 21 | 2.7 | 0 / 0 |

Ours against each baseline (Δ of means; Welch p with n runs per side; steady Δ is in correct/min;
ΔF1 is over the questions answered in every run on both sides; the verdict applies the decision
rule of §4.5):

| Point | Ours vs | Δ goodput | Δ completion | Welch p (n) | Steady Δ goodput [95% CI] | ΔF1 answered [90% CI] (n q) | TOST ±0.05 | Verdict |
|---|---|---|---|---|---|---|---|---|
| 16k ×4 | B1★ K=4 | +33.6% | +1.2 pp | 0.01 (3/3) | +4.08 [+2.35, +5.75] | +0.009 [−0.009, +0.027] (194) | eq. | supported |
| | B1† K=6 | +5.0% | +11.5 pp | 0.40 (3/3) | +1.10 [−0.93, +3.06] | +0.008 [−0.008, +0.023] (151) | eq. | not established |
| | B2 K=6 | +10.8% | +14.7 pp | 0.17 (3/3) | +1.74 [−0.34, +3.89] | – | – | not established |
| 8k ×1 | B1★⁻ K=2 | +29.1% | +7.3 pp | 0.01 (2/2) | +4.01 [+1.80, +6.28] | +0.036 [+0.011, +0.060] (176) | not eq. (ours higher) | supported |
| | B1† K=3 | +21.5% | +37.5 pp | 0.04 (2/2) | +3.49 [+1.21, +5.74] | +0.027 [+0.005, +0.050] (97) | eq. | supported |
| | B2 K=2 | +34.0% | +7.8 pp | 0.04 (2/2) | +4.19 [+2.07, +6.28] | – | – | supported |
| 16k ×1 | B1★ K=3 | +31.3% | +0.0 pp | 0.01 (2/2) | +5.14 [+2.04, +8.18] | +0.014 [−0.009, +0.037] (200) | eq. | supported |
| | B1 K=4 | +1.7% | +2.0 pp | 0.28 (2/2) | +0.03 [−3.25, +3.53] | +0.004 [−0.019, +0.027] (193) | eq. | not established |
| | B1† K=5 | −7.6% | +5.0 pp | 0.03 (2/2) | −2.96 [−6.79, +0.74] | +0.012 [−0.009, +0.033] (185) | eq. | not established |
| | B2 K=4 | +27.3% | +2.0 pp | 0.01 (2/2) | +4.48 [+1.41, +7.63] | – | – | supported |
| 32k ×4 | B1★† K=8 | +8.7% | +0.0 pp | 0.03 (3/3) | +1.11 [−1.32, +3.59] | +0.005 [−0.017, +0.027] (200) | eq. | not established |
| | B2 K=8 | +25.2% | +0.0 pp | 0.01 (3/2) | +3.84 [+1.24, +6.47] | – | – | supported |

ours_h0 against the same baselines (T3):

| Point | ours_h0 vs | Δ goodput | Δ completion | Welch p (n) | Steady Δ goodput [95% CI] | ΔF1 answered [90% CI] (n q) | Verdict |
|---|---|---|---|---|---|---|---|
| 16k ×4 | B1★ K=4 | +22.3% | +1.2 pp | 0.03 (3/3) | +2.82 [+0.97, +4.67] | −0.009 [−0.029, +0.012] (194) eq. | supported |
| | B1† K=6 | −3.8% | +11.5 pp | 0.53 (3/3) | −0.16 [−2.36, +2.02] | −0.013 [−0.036, +0.010] (151) eq. | not established |
| | B2 K=6 | +1.5% | +14.7 pp | 0.83 (3/3) | +0.48 [−1.82, +2.62] | – | not established |
| 8k ×1 | B1★⁻ K=2 | +16.0% | +7.7 pp | 0.01 (2/2) | +2.28 [+0.20, +4.49] | +0.009 [−0.010, +0.028] (176) eq. | supported |
| | B1† K=3 | +9.1% | +38.0 pp | 0.06 (2/2) | +1.77 [−0.50, +4.12] | +0.016 [−0.005, +0.037] (97) eq. | not established |
| | B2 K=2 | +20.3% | +8.2 pp | 0.04 (2/2) | +2.46 [+0.29, +4.67] | – | supported |

### 6.1 Findings

All verdicts below use the decision rule of §4.5: *supported* needs Welch p < 0.05 on overall
goodput and a steady-goodput CI that excludes 0; anything else is *not established*.

**Against the tuned-rule B1 (★).** Ours completes as many or more tasks at every point (100% at
three points, 98.8% at 8k ×1). The goodput difference is supported at 16k ×4 (+33.6%; 100% vs
98.8% completed; p 0.01, CI [+2.35, +5.75]; 3 reps), at 8k ×1 (+29.1%; 98.8% vs 91.5%; p 0.01,
CI [+1.80, +6.28]) and at 16k ×1 against the rule's pick K = 3 (+31.3%; 100% vs 100%; p 0.01, CI
[+2.04, +8.18]). At 32k ×4 it is +8.7% (both 100%) with p 0.03 but a steady CI of [−1.32, +3.59],
so it is not established; the spec pre-registered ≈ 0 there. At 16k ×4 the third rep was added
because the first two did not separate ours from B1★ under the rule (Welch on rep0–rep1, computed
from `runs.csv`: p 0.09 at 16k ×4, 0.19 at 32k ×4; the `p2_main_rep3` header records the reason). Adding reps after seeing the result is a form of optional
stopping, so this p value is optimistic (§8).

**16k ×1: no goodput gain over a well-chosen fixed K.** The +31.3% is against the rule's pick
K = 3, chosen from one perf run per K. K = 4 completed 92.0% on perf, just under the 95%
threshold, but 98.0% on dev, where the difference is not established (+1.7%, p 0.28, CI [−3.25, +3.53]).
B1's highest-goodput K on perf, K = 5 (80.0% completed on perf), completed 95.0% on dev (10
`ctx_overflow` failures per run) and had a higher mean goodput (24.03 vs 22.21; ours −7.6%) and
throughput (46.1 vs 42.5 tasks/min) than ours. That difference is not established either (p 0.03,
steady CI [−6.79, +0.74]). At 16k ×1, ours' advantage over B1 is therefore completion (100% vs
95.0–100%), not goodput, and the +31.3% is mostly a product of the noisy single-run tuning rule.

**Against B1's highest-goodput K (†), which drops tasks.** On dev, B1† loses 11.5% of tasks at
16k ×4, 38.7% at 8k ×1 and 5.0% at 16k ×1 (at 32k ×4 † = ★). Ours completes more tasks at each of
these points. Its goodput is higher at 8k ×1 (+21.5%, p 0.04, CI [+1.21, +5.74]; supported), not
established at 16k ×4 (+5.0%, p 0.40), and lower in the mean at 16k ×1 (−7.6%, not established, above).
At 8k ×1, at similar concurrency (3.2 vs 3.0 agents), B1 K = 3 loses 77.5 tasks per run to
overflow, while ours loses none to overflow and pays 185 give-ups per run. How much of this is
due to the gate and how much to the lifecycle is not measured at 8k ×1 (the 2×2 ran only at
16k ×4, §7.2).

**Against B2 (llama.cpp's host cache, the same host RAM as ours).** Ours is supported at 8k ×1
(+34.0%, p 0.04, CI [+2.07, +6.28]), 16k ×1 (+27.3%, p 0.01, CI [+1.41, +7.63]) and 32k ×4
(+25.2%, both 100%; p 0.01, CI [+1.24, +6.47]). At 16k ×4 it is +10.8% with B2 completing 85.3%
(p 0.17, CI [−0.34, +3.89]): not established. B2 re-prefills 233–668 tokens per task across the
four points, against 21–558 for ours. At 32k ×4, where both complete everything, B2 is also
−13.2% against B1 (p 0.03, n 3/2; T6).

**ours_h0 under the same rule.** It is supported against B1★ at 16k ×4 (+22.3%) and 8k ×1
(+16.0%), and against B2 at 8k ×1 (+20.3%). Against B1† (16k ×4: −3.8%; 8k ×1: +9.1%, p 0.06) and
against B2 at 16k ×4 (+1.5%) it is not established.

**Quality.** Over the questions answered in every run on both sides, ours' per-question F1 is
equivalent to B1's within ±0.05 in 6 of the 7 comparisons of T3, and in the B1 K = 5 comparison
(+0.012, 90% CI [−0.009, +0.033], 185 questions). The exception is 8k ×1 vs K = 2, where ours is
higher (+0.036, [+0.011, +0.060]); the upper bound exceeds +0.05. We do not read this as a quality
gain, because the same cell's B1-vs-B1 null also fails equivalence. Across all ours and ours_h0
comparisons with B1, no CI's lower bound is below −0.036. The test is underpowered at this n: the
B1-vs-B1 null (rep 1 vs rep 0) fails at 2 of the 7 T3 cells (16k ×4 K=4: −0.026 [−0.056, +0.003];
8k ×1 K=2: −0.017 [−0.050, +0.015]), so "equivalent" here means "no loss larger than the
run-to-run noise was seen", not a strong equivalence result. Mean F1 (answered and failed
questions together) is 0.634–0.654 for ours against 0.586–0.632 for B1★.

**Failures.** Ours never overflowed. Its only failures are `kv_infeasible` at 8k ×1 (3 and 2 tasks
in the two runs): tasks whose context reached c − m = 8,064 tokens on that run's trajectory, which
no request in an 8k pool can serve. The set depends on the non-deterministic trajectory: t0019,
t0022 and t0036 failed in ours' rep 0, t0036 completed (forced) in rep 1. All three also failed
with `ctx_overflow` in all four B1 runs at 8k ×1, and t0019 and t0022 in both B2 runs.

### 6.2 Mechanism activity per point

These columns show what each configuration did. The split of the goodput differences between the
gate, admission and the lifecycle is measured only at 16k ×4 (§7.2).

- **16k ×4.** Ours keeps 6.8 agents in flight against B1★'s 3.9, without overflows. It makes 208
  give-up events per run; by action, 89 offloads, 118 drops and 45 erases (a drop then an erase of
  the prefix counts in both). It re-prefills 472 tokens per task
  (B1★: 31) and spends 10.5% of the makespan in slot ops that stall the server. B1† runs 5.9
  agents with no ops, re-prefills 187 tokens per task and loses 23 tasks per run (25 server purges
  and 4 overflow events per run).
- **8k ×1.** Ours runs 3.2 agents against 2.0 (B1★) and 3.0 (B1†), with 185 give-ups, 558
  recompute tokens per task and 9.1% op time. B1† at similar concurrency loses 77.5 tasks per run
  to overflow; ours loses none.
- **16k ×1.** Ours runs 5.1 agents at ω = 0.5, against 3.0 (B1★), 4.0 (B1 K=4) and 4.9 (B1 K=5),
  with 42 give-ups, 92 recompute tokens per task and 5.5% op time. B1 K = 5 reaches nearly the
  same concurrency without slot ops and loses 10 tasks per run to overflow.
- **32k ×4.** Ours runs 10.4 agents against 7.8, with 15 give-ups per run, 21 recompute tokens per
  task and 2.7% op time, so the lifecycle hardly acts here. The goodput difference is not
  established (§6.1).

**ours_h0 (no host memory) vs ours.** Without the host tier, every give-up is a drop or an erase.
Recompute rises 3.4× at 16k ×4 (1,606 vs 472 tokens per task) and 3.0× at 8k ×1 (1,692 vs 558),
and op time falls (3.1% vs 10.5%; 1.2% vs 9.1%), since a drop costs one short request. ours_h0's
goodput is 8.4% below ours at 16k ×4 (13.42 vs 14.66; p 0.09, steady CI [−3.25, +0.71]; T5) and
10.2% below at 8k ×1 (14.74 vs 16.42; p 0.04, steady CI [−3.84, +0.66]; computed from
`runs.csv`), with similar completion. Neither difference is established under the rule. Of
ours' goodput gain over B1★, ours_h0 retains 66% at 16k ×4 and 55% at 8k ×1. The spec expected
the opposite ordering (drop-only close to ours at ×1, about half of ours' gain at ×4). Ours uses
up to h·c of host RAM: a peak of 928 MB of the 940 MB tier in one 16k ×4 run.

### 6.3 Pre-registered predictions vs measured

The spec's predictions (§1) were steady throughput from Phase 1 calibrations. The measured values
are dev steady throughput and the overall-goodput Δ vs the B1 the tuning rule chose.

| Point | Predicted ours (tasks/min) | Measured ours steady thr. | Predicted Δ goodput vs B1 | Measured Δ vs B1★ |
|---|---|---|---|---|
| 8k ×1 | 28–31 | 30.6 | +25–35% (vs K=2) | +29.1% (K=2) |
| 16k ×1 | 40–42 | 42.9 | +5–10% (vs K=4) | +31.3% (K=3); +1.7% vs K=4; −7.6% vs K=5 |
| 16k ×4 | 30–36 | 29.1 | ≈ 0 to +25% (vs model K≈8) | +33.6% (K=4); +5.0% vs K=6 |
| 32k ×4 | 44–47 | 41.0 | ≈ 0 (vs model K≈12) | +8.7% (K=8), not established |

The throughput predictions fall within or just above the predicted range at ×1 (30.6 in 28–31;
42.9 vs 40–42). At ×4 the low ends of the predicted ranges are 3–7% above the measured values.
At 16k ×1 the Δ prediction against K = 4 (+5–10%) was not met: +1.7%, p 0.28. The Δ predictions
at ×4 assumed a B1 that could run K ≈ 8 / 12, which tuning showed it cannot (§5). Against the
B1 that the tuning rule did choose (K = 3 at 16k ×1, K = 4 at 16k ×4), the measured Δ is larger
than predicted, but those predictions were made against a different baseline.

## 7. Decomposition (2×2) and ablations

### 7.1 A bug in the active gate, found during tuning

The active gate is used only by the scheduling-only cells `gated_k` and `adm_retain`. During
tuning on perf, two of the three `adm_retain` runs (ω = 1 and 2) logged 3 overflow events each and
lost 17 and 21 tasks to "Context size has been exceeded". The full gate makes this impossible, so
the active gate had to be undercounting. The cause: a new task can be admitted into a STALE slot
that still holds a long context from the previous task. llama.cpp removes that old tail only when
the slot's prompt is added to a batch (sc:4357). If earlier slots fill `n_batch` first, the slot
is already processing and keeps its old tail, so the server cannot purge it. The active gate
counted only the request's f, not max(s, f). The fix counts a processing slot at max(s, f)
(`active_charge` and `_deficit` in `kvm.py`).

**Run provenance.** Each run's `meta.json` records `git_rev`: the commit plus a hash of the
uncommitted diff, which `tools/mem_aws.sh push` writes to `REVISION` on the GPU host. The runner
reads `REVISION` when it writes `meta.json` at the end of a run, not at its start. The P0 runs
record `0667209` (clean). G0, the ours ω tuning and `adm_retain` ω = 0.5 and 1 record
`0667209…+dirty-5d73d6b82d5e`. `adm_retain` ω = 2 and every later run record
`0667209…+dirty-7d4f061745bb`. The fix was pushed while `adm_retain` ω = 2 was running, so that
excluded run records the post-fix hash but ran the pre-fix code, consistent with its 3 overflows.
Every later run started a new runner process after the push and records the same hash as all
runs after it: `gated_k` K = 4 / 6 / 8, `managed_k` K = 4 / 6, the B2 8k ×1 tuning, and every
`p2_main`, `p2_tune_v2`, `p2_ablate` and `p2_2x2` run. These ran the fixed code. This is inferred
from the recorded hash and the start order, since no run records the hash at its start.

- The three `adm_retain` tuning runs in `p2_tune` (ω = 0.5 / 1 / 2) are excluded by rule (T4
  data notes), including ω = 0.5, which had 43 purges and no overflow. `adm_retain` was re-tuned
  on the fixed code (`p2_tune_v2`): ω = 0.5 / 1 / 2 completed 100% each, with goodput 10.62 /
  10.22 / 7.28 and 65 / 249 / 661 purges. The pick is ω = 0.5.
- The `gated_k` tuning runs (K = 4 / 6 / 8 in `p2_tune`, K = 10 in `p2_tune_v2`) all ran on the
  fixed code, so its pick (K = 6) needs no rerun. Spec §2.4 I3 requires every rep of an affected
  configuration to be rerun, so that reruns do not select for good episodes. Both scheduling-only
  cells meet it: no run of either that the tables use predates the fix. The 13 active-gate runs
  on the fixed code logged 0 overflows (§4.6).
- The full-gate configurations (ours, ours_h0, managed_k, ablations) always charged max(s_k, f)
  and never evaluate the changed branch; the baselines use no gate. G0 and the ours ω tuning ran
  on the earlier revision (`5d73…`). The fix itself changes only the active-gate branch, but the
  recorded hashes do not show whether anything else changed between the two revisions. Every
  dev result (§6, §7) ran on `7d4f…`.

### 7.2 2×2 decomposition at 16k ×4

|  | llama.cpp lifecycle (active gate, server purge) | managed lifecycle (full gate) |
|---|---|---|
| fixed K | `gated_k` | `managed_k` |
| KV admission | `adm_retain` | **ours** |

**Knobs (perf tuning, 1 run per knob; T4 selection runs, tuning data, no effects computed):**

| Cell | Knob | Completion % | Goodput | Agents | Purges | Recompute tok/task | Op time % |
|---|---|---|---|---|---|---|---|
| B1 | K=4 / 6 / 8 / 10 | 98.0 / 81.0 / 46.0 / 15.0 | 8.73 / 10.51 / 8.37 / 3.54 | 4.0 / 5.9 / 7.8 / 9.7 | 2 / 33 / 88 / 107 | 49 / 227 / 458 / 288 | – |
| gated_k | K=4 / 6 / 8 / 10 | 100 each | 9.05 / **11.54** / 9.48 / 7.79 | 4.0 / 5.9 / 7.7 / 9.2 | 2 / 86 / 362 / 575 | 14 / 960 / 3,674 / 5,171 | 0 |
| managed_k | K=4 / 6 / 8 / 10 | 100 each | 9.23 / 11.21 / **11.68** / 11.69 | 4.0 / 5.9 / 7.7 / 9.1 | 0 each | 1 / 223 / 711 / 1,232 | 0.2 / 8.1 / 16.6 / 18.0 |
| adm_retain | ω=0.5 / 1 / 2 | 100 each | **10.62** / 10.22 / 7.28 | 5.0 / 6.8 / 9.6 | 65 / 249 / 661 | 491 / 2,430 / 5,877 | 0 |
| ours | ω=0.5 / 1 / 2 | 100 each | 10.96 / **11.36** / 11.21 | 5.0 / 6.8 / 9.5 | 0 each | 54 / 562 / 1,572 | 2.5 / 12.4 / 19.2 |

Bold = the knob used on dev. The tuning rule picks `gated_k` K = 6, `adm_retain` ω = 0.5 and ours
ω = 1. For `managed_k` the rule's pick is K = 10 (11.69), but K = 8 and K = 10 tie on goodput
(11.68 vs 11.69) and K = 8 had the higher throughput (26.3 vs 25.1 tasks/min), so K = 8 was fixed
before the dev runs. This tie-break is outside the written rule.

On these single runs, llama.cpp's purge thrashes as concurrency grows: `gated_k` goes from 2 to
575 purges and from 14 to 5,171 recompute tokens per task between K = 4 and K = 10, and its
goodput peaks at K = 6 and falls to 7.79 at K = 10. `adm_retain` shows the same pattern over ω (65
→ 661 purges, 10.62 → 7.28). Under the managed lifecycle, goodput stays within 11.21–11.69 from
K = 6 to K = 10 with 0 purges, while recompute grows to 1,232 tokens per task (F2a). These are
tuning data and are not compared quantitatively.

**Dev reps (T4: `p2_2x2` at the perf-chosen knobs, 3 reps per cell; B1 and ours from `p2_main`):**

| Cell | Knob | Completion % | Goodput | Throughput | Agents | Give-ups (off/drop/erase) | Purges / overflows | Recompute tok/task | Op time % | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | K=4 | 98.8 | 10.97 (10.86–11.11) | 21.5 | 3.9 | – | 3 / 1 | 31 | – | 0.621 |
| B1 | K=6 | 88.5 | 13.95 (13.19–15.22) | 26.6 | 5.9 | – | 25 / 4 | 187 | – | 0.569 |
| gated_k | K=6 | 100.0 | 13.69 (13.32–14.10) | 26.9 | 5.9 | – | 58 / 0 | 706 | 0.0 | 0.624 |
| managed_k | K=8 | 100.0 | 15.18 (14.86–15.54) | 28.8 | 7.6 | 257 (115/143/78) | 0 / 0 | 721 | 13.6 | 0.640 |
| adm_retain | ω=0.5 | 100.0 | 12.85 (12.04–13.40) | 24.6 | 4.9 | – | 51 / 0 | 423 | 0.0 | 0.634 |
| ours | ω=1 | 100.0 | 14.66 (13.98–15.11) | 28.2 | 6.8 | 208 (89/118/45) | 0 / 0 | 472 | 10.5 | 0.635 |

**Effects.** The first four rows are T4's effects table; the steady CIs (*) and the last three
rows are computed from `runs.csv` with the same functions.

| Effect | Cells | Δ goodput | Δ % | Δ completion pp | Welch p (n) | Steady Δ [95% CI]* | Verdict |
|---|---|---|---|---|---|---|---|
| Lifecycle at fixed K | managed_k (K=8) − gated_k (K=6) | +1.48 | +10.8 | +0.0 | 0.01 (3/3) | +1.28 [−0.58, +3.25] | not established |
| Lifecycle under KV admission | ours (ω=1) − adm_retain (ω=0.5) | +1.80 | +14.0 | +0.0 | 0.03 (3/3) | +1.98 [+0.20, +3.89] | supported |
| Admission, llama.cpp lifecycle | adm_retain (ω=0.5) − gated_k (K=6) | −0.84 | −6.2 | +0.0 | 0.17 (3/3) | −1.03 [−2.96, +0.81] | not established |
| Admission, managed lifecycle | ours (ω=1) − managed_k (K=8) | −0.52 | −3.4 | +0.0 | 0.28 (3/3) | −0.32 [−2.34, +1.65] | not established |
| Gate only, same K* | gated_k (K=6) − B1 (K=6) | −0.26 | −1.9 | +11.5 | 0.73 (3/3) | +0.15 [−1.82, +2.04] | not established |
| Gate, K re-tuned* | gated_k (K=6) − B1★ (K=4) | +2.72 | +24.8 | +1.2 | 0.004 (3/3) | +3.13 [+1.52, +4.73] | supported |
| Lifecycle claim* | ours − gated_k (K=6), the better scheduling-only cell on perf and dev | +0.96 | +7.0 | +0.0 | 0.09 (3/3) | +0.95 [−0.86, +2.78] | not established |

All findings below are at 16k ×4 only, with 3 reps per cell at the perf-chosen knobs.

- **The gate (scheduling only).** At K = 6 the active gate completes all tasks where B1 loses 23
  per run (100% vs 88.5%), with no significant goodput change (−1.9%, p 0.73, not established).
  Because it completes everything at K = 6, the 95% tuning rule can choose K = 6 instead of 4,
  which gives +24.8% over B1★ (p 0.004, supported). This gain comes from running a higher K
  safely, not from the lifecycle: the +24.8% combines the gate with a K re-tune that the
  completion rule did not allow B1.
- **The lifecycle.** Replacing llama.cpp's purge with the managed lifecycle raises goodput by
  10.8% at fixed K (p 0.01, per-run ranges 13.32–14.10 vs 14.86–15.54, but a steady CI that
  includes 0, so not established under the rule) and by 14.0% under KV admission (supported). In
  both pairs the llama.cpp cell reclaims idle KV by purging (58 and 51 purges per run), while the
  managed cell has none and runs more agents (7.6 vs 5.9; 6.8 vs 4.9). `managed_k` re-prefills
  about as much as `gated_k` (721 vs 706 tokens per task) at 1.7 more agents in flight.
- **Admission.** Peak-projection admission did not raise goodput over a tuned fixed K under
  either lifecycle (−6.2% and −3.4%, neither established). The highest mean at this point is
  `managed_k` K = 8 (15.18), not ours (14.66).
- **Decomposition at 16k ×4.** Most of ours' gain over B1★ (+33.6%) is already reached by the
  scheduling-only cell (`gated_k`, +24.8%). Replacing llama.cpp's purge with the managed lifecycle
  adds 14.0% under KV admission (supported) and 10.8% at fixed K (not established).
  Peak-projection admission showed no gain (−6.2%, −3.4%, not established). The pre-registered
  comparison of ours with the better scheduling-only cell is +7.0% (p 0.09, not established).
  Which lifecycle components matter is not established (§7.3). All of this is one budget point
  with 3 reps per cell.

F2 (`figures/phase2/f2_decomposition.png`) shows both parts: (a) the perf selection runs per knob,
(b) the dev reps per cell with the two lifecycle effects and their verdicts. The report prints the
two panels as separate figures (`f2b_dev.png` in the body, `f2a_tuning.png` in the appendix).

### 7.3 Ablations at 16k ×4

Each ablation changes one rule of ours (ω = 1, K_max = 16) on dev (`p2_ablate`, 1 rep each;
`act_offload` and `always_off` have 3 more reps from `p2_2x2`). They are compared with ours' dev
mean (14.66, 3 reps, range 13.98–15.11) and, for reference, B1★'s (10.97). ours_h0 (`p2_main`)
is listed as the host-tier ablation. Every preset completed 100% of tasks with 0 violations (T5).

| Ablation | Rule changed | Question | n | Goodput | Δ vs ours | Welch p | Give-ups (off/drop/erase) | Recompute tok/task | Op time % |
|---|---|---|---|---|---|---|---|---|---|
| ours | — | — | 3 | 14.66 (13.98–15.11) | | | 208 (89/118/45) | 472 | 10.5 |
| `act_erase` | always erase | Value of keeping the prefix | 1 | 13.38 | −8.7% | – | 159 (0/0/159) | 1,908 | 0.0 |
| `ours_h0` | h = 0 (drop and erase only) | Value of the host tier | 3 | 13.42 (12.55–13.85) | −8.4% | 0.09 | 223 (0/223/89) | 1,606 | 3.1 |
| `lazy` | T\* = 0 | Does the TTL protect agents that are about to return? | 1 | 14.17 | −3.3% | – | 210 (93/117/54) | 568 | 12.4 |
| `clairvoyant` (R1) | evict iff the true remaining delay > T\* | Value of knowing delays | 1 | 14.31 | −2.4% | – | 183 (80/103/53) | 516 | 9.6 |
| `ttl_off` (B4) | offload after 1.25 × online median search time | Pressure-gated vs fixed TTL | 1 | 14.96 | +2.1% | – | 193 (192/1/0) | 52 | 15.1 |
| `always_off` (B5) | offload at every tool start | Eager offload | 4 | 15.00 (14.65–15.36) | +2.4% | 0.44 | 554 (534/19/2) | 169 | 21.2 |
| `oracle_prior` | residuals from the generator's distribution | Online estimate vs true prior | 1 | 15.24 | +4.0% | – | 181 (77/104/44) | 395 | 9.8 |
| `victim_lowid` | victims in slot-id order (llama.cpp's) | Value of residual ranking | 1 | 15.27 | +4.2% | – | 213 (82/131/50) | 496 | 10.6 |
| `nosteal` | T\* = ∞ | Is letting new tasks take idle KV worth it at all? | 1 | 15.28 | +4.3% | – | 205 (89/116/41) | 406 | 10.0 |
| `act_offload` | always offload when the tier has room | Value of the cost-based choice | 4 | 15.28 (14.25–16.10) | +4.3% | 0.31 | 154 (152/2/0) | 81 | 13.3 |

- **Keeping the prefix and the host tier.** The two largest losses come from removing them:
  `act_erase` −8.7% (1 run, below ours' lowest rep; 1,908 recompute tokens per task against 472)
  and `ours_h0` −8.4% (3 reps; p 0.09, steady CI [−3.25, +0.71], computed from `runs.csv`). At
  8k ×1, ours_h0 is −10.2% (p 0.04, steady CI [−3.84, +0.66]). These results are consistent with
  both mattering, but neither passes the decision rule.
- **Timing, victim order and delay knowledge.** These are single runs with |Δ| ≤ 4.3%, while ours'
  own three reps span 13.98–15.11 (7.6% of its mean). `lazy`, `clairvoyant` and `ttl_off` fall
  inside that range; `oracle_prior`, `victim_lowid` and `nosteal` lie 0.1–0.2 correct/min above
  its top.
  So there is no evidence at this point that the pressure-gated TTL, the residual ranking or
  letting new tasks steal idle KV helps, or that knowing the true delays would help
  (`clairvoyant` −2.4%). These rules are not shown to matter.
- **Offload choice and eager offload.** Always offloading instead of the cost-based choice
  (`act_offload`, same pressure-driven timing: +4.3%, p 0.31, steady CI [−1.57, +2.49]) and
  offloading at every tool start (`always_off`: +2.4%, p 0.44, steady CI [−1.52, +2.10]; 4 reps
  each, CIs computed from `runs.csv`) are not significantly different from ours, so the cost
  model's offload/drop choice is not shown to help. The 3 extra reps were run because their single `p2_ablate`
  runs had higher steady throughput than ours' first two reps (31.6 vs 28.6–29.2 tasks/min,
  `p2_2x2` header).
  They trade re-prefill (81 and 169 vs 472 tokens per task) for op time (13.3% and 21.2% vs
  10.5%), and `always_off` fills the host tier (peaks of 931–939 MB of 940 MB). Their mean F1 is
  0.623 against 0.635 for ours (not tested).

F4 (`figures/phase2/f4_ablations.png`) plots each preset against ours' mean and rep range.

### 7.4 No-benefit regimes (T6)

- **32k ×4** (pre-registered ≈ 0): ours +8.7% vs B1 K = 8 (p 0.03, steady CI [−1.32, +3.59]): not
  established. Ours runs 10.4 agents against 7.8, with 15 give-ups per run, 21 recompute tokens per
  task and 2.7% op time. B2 is −13.2% vs B1 (p 0.03, n 3/2).
- **131k ×1** (ample memory, 1 run each): B1 K = 8 27.50 vs ours (K_max 8, ω = 0.5) 27.54 correct/min
  (+0.1%), both 100% completed, 7.7 vs 7.8 agents. Ours made 0 give-ups and no slot ops, and its
  ledger peaked at 34,054 of 130,944 cells. With one run each, no difference is detectable,
  in either direction, from the manager's bookkeeping (gate, `/tokenize`, 1 Hz `/slots` polling).

## 8. Limitations

- **Few reps.** Welch p with n = 2–4 per side is indicative only, and 8k ×1 and 16k ×1 have 2 reps
  per cell (the spec pre-registered ≥ 3 at the headline cells, so 8k ×1 falls short). The two
  tests disagree in four comparisons, all with p < 0.05 and a steady CI that includes 0: ours vs
  B1 at 32k ×4 (p 0.03, [−1.32, +3.59]), ours vs B1 K = 5 at 16k ×1 (p 0.03, [−6.79, +0.74]),
  the lifecycle effect at fixed K (p 0.01, [−0.58, +3.25]) and ours_h0 vs ours at 8k ×1 (p 0.04,
  [−3.84, +0.66]). None disagrees the other way. Neither test is treated as decisive alone.
- **Reps added after seeing results.** The third rep at 16k ×4 and 32k ×4 was added because n = 2
  did not separate ours from B1, and the extra `act_offload` / `always_off` reps because their
  single runs looked better. This is optional stopping; the affected p values are optimistic.
- **B1 knobs come from one perf run per K.** Near the 95% threshold this choice is noisy: at
  16k ×1, K = 4 completed 92.0% on perf and 98.0% on dev, and K = 5 80.0% and 95.0%. Against
  K = 4 ours' goodput gain at 16k ×1 is +1.7% (p 0.28), and against K = 5 ours is −7.6% (not
  established), instead of +31.3% against the rule's pick. The `managed_k` knob (K = 8) is a
  tie-break outside the tuning rule (§7.2).
- **The 2×2 ran only at 16k ×4.** The attribution to gate and lifecycle (§7.2) is not measured at
  the other points, in particular not at 8k ×1.
- **Dev split only.** The test-500 confirmation and the delay, mix, open-loop (latency) and
  iso-GPU-memory blocks of the spec were not run. No latency claim is made.
- **The quality test is underpowered.** The B1-vs-B1 null fails the ±0.05 TOST at 2 of 7 cells, so
  the equivalence verdicts say no loss beyond run-to-run noise was seen.
- **Not iso-total-memory.** Ours uses up to h·c of host RAM (448–1,792 MiB). ours_h0, which uses
  none, is the iso-memory comparison. It retains 66% (16k ×4) and 55% (8k ×1) of ours' goodput
  gain over B1★; ours − ours_h0 is not established at either point (§6.2). B2 gets the same host
  tier as ours.
- **A post-freeze fix** in the active gate (§7.1). The runs that the tables use for the active-gate
  cells ran on the fixed code, but this is inferred from a revision hash written at the end of
  each run and from the start order. G0 and the ours ω tuning ran on an earlier revision.
- **Validation is partly indirect.** The KV debug flags print no `find_slot` lines in this build,
  so "server used ≤ U" rests on zero purges and overflows and on clean reconciles across 53
  full-gate runs.
- **Tables and figures.** B2's host cache at 16k ×1 was tuned on perf with 4,096 MiB but run on dev with the matched
  896 MiB. The cause of B2's low cache reuse was not traced (the E0c B2 probe overflowed at its
  first requests).
- **Op stalls.** Saves and restores block the server's main loop, so ours spends 10.5% of the
  makespan in ops on average (up to 11.1% in one run) at its tuned ω on dev at 16k ×4, up to 29.7% at
  ω = 2 (T2), and `always_off` 21.2%.
  Under load the online κ_s + κ_r reached 245 µs/token (G0, 32k ×4), against 89 µs/token in the
  fragmented-slot probe, which narrows offload's advantage over drop.
- **Scope.** One model (7B, f16 KV), one GPU (A10G, with pressure created by capping `-c`),
  synthetic tool delays, an offline corpus easier than full Wikipedia, and closed-loop batch
  arrivals of one mix (70/30 long/short). The `kv_infeasible` tasks at 8k depend on the
  non-deterministic trajectory and differ between runs (3 and 2).
- The high-water explanation (§2) comes from the source and two probes; we did not trace the
  kernels.

## 9. Reproducing

Code: branch `jiawei`, commit _[to be filled at submission]_. Set up the GPU host, data and
corpus as in Phase 1 (`phase1_details.md`, Appendix A), with `LLAMA_SERVER` and `MODEL` set, then:

```
uv sync
uv run python bench/e0b_runtime.py                              # E0b -> results/e0b/, results/e0b.log
uv run python bench/e0c_runtime.py                              # E0c -> results/e0c/, results/e0c.log
uv run python -m scripts.fit_kvm_priors                         # configs/kv_priors.json from Phase 1 runs
uv run python -m scripts.make_workload configs/workloads/dev_long_200.yaml   # frozen, already in workloads/
for s in p0_baselines p0x4_tune g0_shakedown p2_tune p2_main p2_tune_v2 p2_ablate p2_main_rep3 p2_2x2; do
  uv run python -m scripts.sweep configs/sweeps/$s.yaml          # on mem-aws: tools/sweep_bg.sh <yaml>
done
uv run python -m scripts.fit_kvm_priors --check 'results/g0_shakedown/*/rep*'
uv run python -m analysis.aggregate                             # results/runs.csv
uv run --no-sync python -m analysis.phase2_report               # docs/phase2/tables.md (T1-T6)
uv run --no-sync --with matplotlib python -m analysis.phase2_figures   # docs/figures/phase2/
uv run --no-sync pytest                                         # 167 tests, mock LLM, no GPU
```

The loop lists the sweeps in the order they ran. `p2_tune_v2` writes into `results/p2_tune/` and
`p2_main_rep3` into `results/p2_main/` (`rep2`), since they share those sweep names; the sweep
driver skips reps that already exist. `analysis.phase2_report` excludes the three pre-fix
`adm_retain` runs by name (`p2_tune/*_admret_w*`, §7.1) and is deterministic (seeded bootstrap).
The B1 tuning at ×1 also uses the Phase 1 E2 runs (`e2_budget`, `e2_budget_k`) on the same perf
workload. Presets are in `configs/kvm.yaml` and sweep definitions in `configs/sweeps/`. Every run
directory records the server command line, build, workload hash, code revision (§7.1 for when it
is read) and the manager's configuration (`meta.json`), the manager's summary
(`summary.json["kv"]`) and the server-log counters (`summary.json["server_log"]`).

## 10. Figures

All in `figures/phase2/`, from `analysis/phase2_figures.py`. The F2–F4 captions printed in
`phase2/tables.md` (from `analysis/phase2_report.py`) describe an earlier version of the figures;
the descriptions below match the current files.

- **F1** `f1_completion_goodput.png`: completion vs overall goodput at the four dev points
  (`p2_main`; B2 at 8k ×1 from `p2_ablate`), one panel per point with its own x scale and a shared
  completion scale. Small markers are reps, large markers the means. B1 is labelled with its K
  (★ = perf pick, ★⁻ = fallback, † = perf top-goodput K). B1 K = 5 at 16k ×1 is not drawn.
- **F2** `f2_decomposition.png`: 2×2 decomposition at 16k ×4. (a) Perf selection runs (1 rep per
  knob; tuning data, not results): overall goodput vs time-mean agents in flight for every knob of
  B1, gated_k, managed_k, adm_retain and ours; numbers = llama.cpp purges
  (`server.log`) for gated_k and adm_retain, completion % for B1 (B1 K = 10, 3.54 at 15%, is marked
  off scale, value in the note). Circles = the knob each cell ran on dev; for `managed_k` that is
  K = 8, labelled as a tie-break (the goodput rule gives K = 10). (b) Dev reps (3 per cell) at those
  knobs: mean, min–max and reps; arrows = the two lifecycle effects with Welch p and the §4.5
  verdict (S / n.e.). The report prints (b) alone one column wide (`f2b_dev.png`, colours in the
  caption) and (a) alone on the appendix page (`f2a_tuning.png`), both drawn at their printed size
  with no font below 6.5 pt.
- **F3** `f3_timeline_16k_x4.png`: GPU KV over time at 16k ×4 (dev rep 0), B1 K = 4 vs ours ω = 1,
  stacked by state against `-c` (dashed), with llama.cpp purges (▼) and failed tasks (×) marked.
  Active KV is valued at each request's final token count (an upper bound while it runs), so B1's
  stack can exceed `-c` just before its overflows: B1 rep 0 has 3 purges seen by the `/slots` reconcile
  (4 in `server.log`; the panel marks the reconcile's) and then 3 `ctx_overflow` failures near
  t = 109 s; ours has no purges and no failed tasks.
- **F4** `f4_ablations.png`: ablations at 16k ×4 (dev): overall goodput per preset (dot = mean,
  bar = min–max, grey = reps, hollow = 1 run, n in the label), including ours_h0, against ours'
  mean (line) and rep range (band) and B1 K = 4's mean (dashed); text = Δ vs ours' mean as in T5.
- `kv_states.png`: KV token·s by state per configuration at 16k ×4 (dev rep 0); B2's breakdown is
  approximate (hatched).
- `timeline_<point>.png`: GPU KV over time by state at 16k ×4 (dev rep 0) for B1 K = 4, B1 K = 6,
  B2 K = 6, ours and ours_h0.

## 11. Follow-ups for the final report (2026-11-17/18)

Sections 5–7 describe the state when this note was written, with 2 reps at 8k ×1 and 16k ×1 and
the 2×2 only at 16k ×4. Three follow-ups ran afterwards; `docs/phase2/tables.md` (T3, T4, T7) is
current and `docs/final/report.tex` uses it.

- **Third rep at 8k ×1 and 16k ×1** (`configs/sweeps/p2f_main_rep3.yaml`, `p2f_ablate_rep3.yaml`).
  8k ×1: ours 16.31 vs B1 K = 2 12.92, +26.2% (S; was +29.1% at n = 2). 16k ×1: ours vs B1 K = 5
  is now −6.1% and S (Welch p = 0.01, steady CI [−6.52, −0.18]): B1 K = 5 is faster while losing
  7.5% of its tasks, so at that point ours' advantage is completion only.
- **2×2 at 8k ×1** (`p2f_tune_8k.yaml` on perf, `p2f_2x2_8k.yaml` on dev; picks: gated_k K = 3,
  managed_k K = 3, adm_retain ω = 0.5). Gate with K re-tuned +17.0% (S); lifecycle at fixed K
  +10.3% (p = 0.04, CI [−0.66, +3.42], n.e.); lifecycle under KV admission +15.9% (p = 0.01, CI
  [−0.13, +3.88], n.e.); admission −7.0% and −2.2% (n.e.). The best dev cell is managed_k (16.68),
  as at 16k ×4 (15.18). Every gated cell loses the same ~2 kv_infeasible tasks per run.
- **Held-out test split** (`p2f_test.yaml`, test_long_500, 500 tasks, 2 reps, knobs frozen): ours
  vs B1★ +41.4% at 16k ×4 and +17.3% at 8k ×1 (both S); vs B2 +34.6% (S) at 8k ×1 and +17.2% (n.e.)
  at 16k ×4; vs gated_k +4.8% (n.e.). All six F1 comparisons are within the test margin ±0.03.

Phase 3 (TRIM) is in `docs/phase3_spec.md` §6 and `docs/phase3/tables.md`.
