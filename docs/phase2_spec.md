# Phase 2 specification: agent-level KV lifecycle at tool-call boundaries

Implementation spec for Phase 2 (written 11/15). It refines `docs/kv_lifecycle_plan.md` §3–5
using the Phase 1 measurements (`docs/phase1_details.md`), the E0b results (`results/e0b.log`,
`results/e0b/*.json`) and the llama.cpp b11960 source. The local copy of that source has the
same md5 as the copy on mem-aws. Line references to llama.cpp are `server-context.cpp` (sc),
`llama-kv-cache.cpp` (kv) and `server-queue.cpp` (sq). Line references to our code are against
commit 0667209.

Fixed for the whole phase: Qwen2.5-7B-Instruct Q4_K_M, f16 KV (56 KiB/token, 57,360 B/token
in a slot file), ReAct v3, the profiles, the frozen workloads, and llama-server controls only
(no C++).

## 1. Mechanism and expected gains

**Two capacity constraints.** Phase 1 treated the KV pool as one limit. The source shows there
are two:

- **Active requests are a hard limit.** When a ubatch cannot find cells, the server first
  purges one idle slot per retry. In unified mode only, it takes the first idle slot that holds
  tokens, in slot-id order (sc:1911–1935). If nothing can be purged it halves `n_batch`. At
  `n_batch == 1` it fails **every** processing slot with "Context size has been exceeded" and
  clears them (sc:4699–4745). This is the E2 collapse: 971 of 972 failures.
- **Idle and stale KV is a soft limit.** The server can always reclaim it, but blindly, by slot
  id.

**What Phase 2 adds.** The mechanism has three layers. Each is independently ablatable:

1. **An exact dispatch gate** (§2). The client reserves cells for every request before sending
   it, so the pool never overflows. Runs with the `full` gate must log 0 overflows and 0 server
   purges; runs with the `active` gate must log 0 overflows and expect purges (§2.4 I3).
2. **Peak-projection admission with over-commit** (§2). It decides how many agents are in flight.
   That number is no longer a fixed K sized for everyone's simultaneous peak.
3. **The KV lifecycle at tool boundaries** (§4). When memory is needed, it decides *whose* idle KV
   leaves the pool, *how* (offload to host RAM, drop to the shared prefix, erase), and *when*.
   Returning agents reclaim on demand. A new task may take an idle agent's KV only after a
   pressure-gated TTL (ski-rental rent ≥ break-even B) has expired.

**Measured costs** (E0a, E0b). These are the priors; the cost model refits them online from
server timings.

| Operation | Cost | Blocks other slots? |
|---|---|---|
| save | 33–36 µs/token (99–104 ms for 3k; 50 ms for 1.4k) | yes (main loop) |
| restore | 12–16 µs/token (37–47 ms for 3k, also into holes) | yes |
| erase | 1.4–2.5 ms | negligible |
| drop to prefix (`/completion` with the prefix only, `n_predict` 0) | 13–20 ms; keeps 1,400 cells | batched |
| re-prefill | ≈ 220 µs/token + 9.7 µs/token per 1k occupied cells of span (E0b idlecost: 2k tokens take 550 / 1,014 / 1,472 ms at 0 / 24k / 48k idle cells) | shares the GPU |

Offload is 4–7× cheaper than re-prefill per token, but it stalls everyone. The break-even
T\* = B/S ≈ 0.5–0.6 s for offloading a 3k slot at 16k. Against median tool delays of 0.8 s (×1)
and 3.2 s (×4), r = T\*/median falls inside (0.1, 10). So under plan §4.4 **the pressure-gated
TTL stays the headline**, restricted to where it matters (§4.3).

**Predictions** (pre-registered; checked in P7). These use the long mix (70% `react_long`, 30%
`react_short`), EM ≈ 0.46 throughout, and h = 1 (host tier = `-c`).

- B1 at ×1 is measured (E2). B1 at ×4 comes from the E2-calibrated active-request model: per
  task 4.0 requests and 3.0 tool calls, mean tool time 0.75 s × scale, and server request rates
  X(A) = 1.65 / 2.64 / 3.18 / 3.0 req/s at A = 1.1 / 2.5 / 6.2 / 14.3 active requests. A request
  takes r = A/X(A); B1 with K agents has A = K·4r/(4r + 3·0.75·scale) and throughput
  K/(4r + 3·0.75·scale).
- **B1 at ×4 is evaluated at the K this model says is best, not at its ×1 K.** At ×4 most of an
  agent's time is in tools, and llama.cpp already purges idle pinned slots on demand, so B1's K
  is limited only by the *active* load. By the same model, the ×1 safe points had A ≈ 1.05
  (8k K=2, 89% completed), 2.5 (16k K=4, 92%) and 6.2 (32k K=8, 100%). At ×4 the same active load is reached
  at K ≈ 4 / 8 / 12. Those B1 numbers subtract a 5–10% failure rate (as at ×1) and the purge
  re-prefills (a purged agent re-prefills its whole context, ≈ 0.5–0.8 s). P0b measures them.
- The ×4 column of the previous draft compared ours with B1 at its ×1 K (≈ 10.5 / 21 / 39) and
  predicted +45–150%. That baseline was untuned and is withdrawn.

| Point | B1 best fixed K: tasks/min (completed) | Ours tasks/min | Δ goodput vs B1 | Main source of the gain |
|---|---|---|---|---|
| 8k ×1 | K=2: 22.8 (89%) | 28–31 | +25–35% | **safety/admission result**: gate (no failures) + ~1.5 more agents; not a lifecycle headline |
| 16k ×1 | K=4: 37.8 (92%) | 40–42 | +5–10% | gate + ~1 more agent; near noise (±2.7% CV at ample memory) |
| 32k ×1 | K=8: 47.8 (100%) | 46–48 | ≈ 0 | none, by construction: K_max = 8 = B1's K (§2.3). Independent evidence of the plateau: E2 c64k K=16 gave 47.3 |
| 8k ×4 | K≈4 (model): ≈ 15–19 | 18–26 | ≈ 0 to +60% | lifecycle: prefix kept and long tails offloaded instead of id-order purge + full re-prefill; no failures |
| 16k ×4 | K≈8 (model): ≈ 28–34 | 30–36 | ≈ 0 to +25% | lifecycle; ≈ 9–10 agents in flight, 10–20% op stall |
| 32k ×4 | K≈12 (model): ≈ 44–46 | 44–47 | ≈ 0 | none: active-request compute plateau (second no-benefit regime) |

**The contribution is measured against scheduling, not against B1.** B1 has no gate, so ours
vs B1 mixes three effects. The lifecycle claim is **ours vs `sched`**, the best
scheduling-only configuration: the better of `gated_k` and `adm_retain` (§8.1), each tuned on
the same grid budget as ours. `sched` is a row of the main table at 16k ×4 and 8k ×1 and of
P1-test at 16k ×4. Expected: at 8k ×1, ours ≈ `sched` (the gain there is admission and safety);
at 16k ×4 and 8k ×4, ours > `sched` by most of the ours-vs-B1 gain.

Further pre-registered expectations:

- Drop-only (`ours_h0`, no host tier) roughly matches ours at ×1. It gets about half of ours'
  gain at ×4.
- At ×4, ours ≈ `lazy` (no TTL), because the delays are much longer than T\*. At ×1, ours > `lazy`,
  because lazy evicts agents that are about to return.
- **Iso-total-memory: we expect to lose.** Ours at 16k GPU plus 16k host < B1 at 32k GPU. The
  claim is therefore *GPU KV saved*: ours at a smaller `-c` matches B1 at 32k ×4. It is tested
  at c = 24k with controls (§8.2 P3): ours and B1 (best K) at 24k, 2 reps each. "Matches" is
  pre-registered as steady goodput within one B1 CV of B1 at 32k, with the CV taken from the
  B1-vs-B1 null runs (§8.2). Part of any match comes from the smaller pool's smaller attention
  span (E0b idlecost), so each run reports its mean span (from U, and from the kvdbg run at
  16k). `ours_h0` is the variant that uses no extra memory.

**Ruled out from the source.**

- **Sharing the ReAct prefix across slots via `n_cmpl`.** `server_slot::release()` calls
  `prompt_clear()` for every child task: "do not keep context of the child slots" (sc:666–669,
  checked). Children also cannot target a slot (`server-task.h:323`, `id_slot = -1`). No other
  HTTP route reaches `seq_cp`, so the 1.4k prefix stays duplicated per slot (E0a).
- **A compute payoff from eviction.** `get_n_kv = pad256(used_max_p1)` (kv:1255–1268). Attention
  spans up to the highest occupied cell. Erasing idle cells below active ones leaves decode at
  208 vs 202 tok/s; re-placing the active cells gives 295 (E0b highwater). The client cannot
  control placement. In tight pools the span is ≤ `-c`, so this tax is bounded and reported, not
  claimed.

## 2. Admission and dispatch gate

### 2.1 Ledger

- c = `-c`, read from `/props` and asserted equal to the sweep's `ctx`. m = 128 (margin).
- For slot k, the **charge** u_k is an upper bound on the cells the server holds for that slot:

| Slot state | u_k |
|---|---|
| ACTIVE (request in flight) | max(s_k, f_r), where f_r = tok(P_r) + n_predict_eff_r (§2.2) |
| IDLE (owner in a tool), READY (owner waiting at the gate), STALE (task done) | s_k, exact resident count |
| SAVING | s_k |
| RESTORING | f_r of the request it serves (≥ S_img) |
| UNKNOWN (op timed out, §2.4 I4) | max(s_k, S_img, f_r), an upper bound, until resolved |
| OFFLOADED, FREE | 0 |

- U = Σ_k u_k.
- s_k is exact after each response: **s_k = prompt_n + cache_n + predicted_n − 1**. The last
  sampled token is never decoded. Evidence: E0a 2,500 + 8 → `n_prompt_tokens` 2,507; truncate
  1 + 1,399 + 1 → 1,400.
- After an op, s_k comes from the response: `n_erased` → 0, `n_restored`, and `save` leaves s_k
  unchanged.
- **Charges fall only when the server's response arrives.** The max(s_k, f_r) term covers the
  interval before the server's `seq_rm` of the old tail (sc:4355), e.g. a drop-to-prefix
  request on a 5k slot.

### 2.2 Gate (request level)

All waiting requests sit in one queue. They are ordered by (priority, ready time):

- Priority 0: a returning agent (steps ≥ 2, including a pending restore).
- Priority 1: the first request of a newly admitted task.

Only the head may dispatch. One `asyncio.Lock` serializes planning, victim ops and dispatch.
Only coroutines take it (the head, `release()`, the reconcile poller); the synchronous
callbacks `on_event` and `on_response` never do (see *Wake-ups* below).

| Gate | Used by | Dispatch head h on slot k iff |
|---|---|---|
| `full` | every managed configuration | U − u_k + max(s_k, f_h) ≤ c − m |
| `active` | retain configurations (llama.cpp reclaims idle KV by purging) | Σ_ACTIVE u + f_h ≤ c − m |
| `none` | B1/B2/B3 (Phase 1 behaviour) | always |

**When the full gate fails** (deficit D > 0), the head evicts **one victim at a time**:

1. Recompute D and E(h) (§4.2) from the current ledger.
2. Take the top-ranked eligible victim and re-validate it immediately before the op: still
   eligible for this head's priority, not busy, not touched by a `release()` erase, and its
   state unchanged since the ranking.
3. Issue its action (§4.2) and await it.
4. Recompute D. Stop as soon as D ≤ 0 and dispatch; otherwise go back to 1.

A precomputed victim list would go stale during the 100–300 ms ops: active requests finish,
victims' tools end, tasks are released, and over-eviction or an ineligible eviction follows.

- If E(h) cannot cover D, the head waits (see *Wake-ups*).
- **Committed head.** Once a head has issued an eviction, a newly arrived priority-0 waiter
  queues behind it, so the cells it freed are not taken from it. The commitment lapses (and
  the queue order is restored) as soon as the head's current E(h) can no longer cover its
  remaining D, so a blocked priority-1 head never blocks a returning agent (§2.4).
- **Infeasible and truncated requests.** Let n_predict_eff = min(n_predict, c − m − tok(P)).
  The request is **kv_infeasible** only when tok(P) ≥ c − m (n_predict_eff ≤ 0); the task then
  ends with that status. Otherwise it is dispatched with n_predict_eff and charged
  f = tok(P) + n_predict_eff. This reproduces the server's own stop at
  `prompt.n_tokens() + 1 ≥ n_ctx` (process_token, sc:2188; n_ctx_slot = c in unified mode),
  and the server rejects a prompt of ≥ n_ctx tokens (sc:4081), so ours refuses only what B1
  also cannot finish, up to the margin m. A request sent with n_predict_eff <
  n_predict is logged `kv_truncated` and counted separately.
- Which tasks exceed c depends on the non-deterministic trajectory (the same task peaked at
  4,523 and 9,076 tokens in two Phase 1 runs), so different tasks are infeasible in different
  runs. kv_infeasible counts as a **failure** in completion and goodput for every
  configuration. Each run lists the task IDs that exceeded c − m, and the report gives the
  overlap of those sets across configurations.

**Wake-ups.** `on_event` (`tool_start`/`tool_end`) and `on_response` are synchronous, and
`asyncio.Condition.notify()` raises `RuntimeError` unless the caller holds the lock. The
manager therefore keeps a monotonically increasing `self._epoch` and wakes waiters with
plain futures:

- Every state change (`req_end`, op completion, `tool_start`, TTL expiry, task release, a
  change of R_sum or of the admitted count) mutates the ledger, bumps `_epoch`, and calls
  `_kick()`. `_kick()` resolves the **head's** wake future (and the admission waiter's) with
  `fut.set_result(None)` if not done, which is legal from sync code. Other waiters are not woken,
  so K = 16 waiters do not stampede.
- The head loops: snapshot `_epoch`; under the lock, check the gate, and evict or dispatch;
  if still blocked, install a fresh wake future *before* releasing the lock (no await in
  between, so no wake-up is lost); release the lock; await the future only if `_epoch` is
  unchanged; then recheck. When a head dispatches or leaves the queue, the next head's future is resolved.
- The 50 ms tick updates rent and expiry without the lock (asyncio is single-threaded, so
  attribute updates between awaits are atomic) and calls `_kick()` on any expiry.
- `EventLog.emit` calls each listener inside `try/except` and logs a `kv_violation
  listener_error`, so a manager bug cannot fail an agent's task.

**Retain plus the full gate deadlocks.** Idle agents hold memory, and their own returns are
gated. Retain configurations therefore use the `active` gate and leave idle KV to the server's
purge, which can always reclaim it.

### 2.3 Admission (task level)

Admission uses FIFO tickets, assigned at `queue.get`, with head-of-line blocking. Admit task j
iff all of these hold:

```
(a) n_admitted < K            # fixed K, or the compute cap K_max for kv admission
(b) gate_q == 0 and n_admitted_not_started == 0   # never add pressure while agents wait for memory
(c) Σ_{i admitted} R̂_i + R̂_j ≤ (1+ω)(c − m)       # kv admission only
    R̂_i = max(s_i + g_i,  P̂0_p + Ĝ_{0.9}(p, n_i))   # s_i: resident, offloaded image, or last size
```

- Ĝ_q(p, n) is the q-quantile of (final peak − size after step n) for profile p. It is fitted
  from Phase 1 perf-split traces by `scripts/fit_kvm_priors.py` into `configs/kv_priors.json`.
  At step 0 for `react_long`: P̂0 ≈ 1,449, Ĝ_0.9 ≈ 3.46k in the fit (an earlier estimate of
  4.4k is close to Ĝ_0.95 = 4,371).
- Offloaded and dropped agents still count, because they will return.
- (b) counts admitted tasks whose first request has not yet entered the gate queue. Between
  admission and the first `acquire` there is a `/tokenize` round trip, and `oneshot_long` first
  runs BM25 in a thread (agent.py:114). Without this term the gate queue looks empty in that
  window, and up to K_max tasks could be admitted in one burst. `admit()` increments
  `n_admitted_not_started`; the task's first `acquire` decrements it when it enqueues (or on
  cancellation).
- Ĝ_q is fitted only from tasks with status ok/forced, in runs with no overflow failures
  (E1 `e1_concurrency`, `e2_budget` c64k, `e2_budget_k` c32k_k8). Runs where most tasks died
  of overflow (e.g. c8k K=4: 48/200 completed) stop before the true peak and would bias
  Ĝ_0.9 low. The fit is checked against the dev traces of the G0 shakedown.
- ω is tuned on the same grid budget as K: three values, {0.5, 1, 2}. **Both K and ω are tuned
  on the perf split** (`workloads/p1_long_200.json`), with dev held out. No tuning run is
  ever reported as a result rep: every reported number is a fresh, post-freeze run on dev
  (`dev_long_200`) or test (`test_long_500`).
- K_max is the ample-memory best K: 8 at ×1 (E1). At ×4 it comes from two 131k runs with
  K ∈ {8, 16} (P0b). At 32k ×1, K_max = 8 equals B1's best K, so ours cannot admit more agents
  there; that cell's ≈ 0 is true by construction, and the plateau evidence is E2 c64k K=16
  (47.3 vs 47.8). The `kv_tick` log records whether K_max or (c) binds in each cell.

**Resolved: ω is decoupled from h.** The faithful design bound ω ≤ h, so every reservation had a
home. But then h = 0 would admit only about 3 agents at 16k, which is below fixed K = 4. Safety
comes from the gate, so ω is a pure efficiency knob. When ω > h and the host tier is full, the
action falls back to drop.

**Ablation `adm_duty`** (reviewer): Σ ρ_a·f̂_a + f_j ≤ θ(c − m) with ρ = the duty-cycle EWMA,
θ ∈ {0.6, 0.8}. It runs at 16k ×4 only.

### 2.4 Invariants and progress

- **I1.** U ≤ c − m at all times (full gate).
- **I2.** Server cells of slot k ≤ u_k. This holds because tok() is the server's own tokenizer
  (§3), generation is bounded by `n_predict`, and charges fall only after responses.
- **I3.** I1 + I2 give used ≤ c − m. `find_slot(ubatch, cont=false)` is non-contiguous
  (kv:1012–1060), so it never fails. Hence no purge and no HTTP 500.
  - With `gate == "full"`, a purge or an overflow is a `kv_violation`.
  - With `gate == "active"` (`gated_k`, `adm_retain`), purges are the intended lifecycle: they
    are recorded as `kv_state` transitions (reason `server_purge`), and only an HTTP 500 or
    overflow is a violation.
  - A violation run is **kept and its failed tasks are counted**; the report gives the number
    of violation runs per configuration. If a violation exposes a bug, the bug is fixed, the
    code is re-frozen, and **every** rep of that configuration is rerun, not only the failed
    one, so the reruns do not select for good episodes.
- **I4.** Requests and ops on a slot are mutually exclusive (`SlotKV.busy`). Ops are issued only
  on idle slots, and each op has a 5 s wall timeout.
  - Reason: deferred slot actions keep `task.id_slot = −1`, so `pop_deferred_task` releases
    them only as "first deferred task" on *some* slot's release (sq:90–110).
  - **Ops are manager-owned.** `_offload/_drop/_erase/_restore` run the whole op sequence
    (e.g. save → erase) in a task created with `asyncio.create_task`, which updates the ledger
    and clears `busy` in its own `finally`. The head awaits it through `asyncio.shield(...)`.
    The runner's `wait_for(agent.run, task_timeout)` (runner.py:196) can cancel the head, and
    a client disconnect cancels a server task only if it has not started
    (`cleanup_pending_task`, sq:368–382, noticed within `HTTP_POLLING_SECONDS` = 1 s, sc:46).
    A cancelled head therefore only abandons its wait and its queue entry; the victim's op
    still completes and the victim's charge stays exact. In-flight op tasks are kept in a set
    that `close()` awaits. `release()` erases the same way.
  - **Op timeout.** After 5 s the client cannot know whether the server ran the op (an erase
    may or may not have freed cells; a save may or may not have written the file). The slot
    becomes UNKNOWN: it stays busy, its charge is max(s_k, S_img, f_r) (an upper bound), and
    the manager polls `/slots` (with the reconcile version rule, §3) until `is_processing` is
    false. It then issues an erase and sets s from `n_erased` (or adopts `n_prompt_tokens`),
    clears busy, and keeps or unlinks the file by the outcome. A `kv_violation op_timeout` is
    logged.
- **I5.** Victims are never ACTIVE and never the head. An evicted agent holds 0 cells or only its
  prefix. So it is evicted **at most twice per idle period** (a drop, then an erase of the
  prefix), and the erase of a dropped live agent is allowed only for a priority-0 head
  (§4.2). The two count as one give-up event for that (task, step).

**No deadlock.**

- *Priority-0 head.* E(h) contains every non-active resident cell except the head's own. If no
  request is active, evicting all of E leaves U = u_k, so the head dispatches iff f_h ≤ c − m;
  otherwise it is kv_infeasible. Active requests finish in bounded time: they cannot fail (I3)
  and are capped by `n_predict`. When they finish they join E.
- *Priority-1 head.* It runs only when no priority-0 waiter exists. Its E contains free cells,
  STALE slots and *expired* idle agents. While it waits, every resident idle agent's rent grows,
  so each one either expires (finite T\*) or returns. Returning agents finish in ≤ `max_steps`+2
  requests and are released as STALE. So the head's wait is bounded.

**No livelock.** An eviction is issued by a head, and a committed head (§2.2) keeps the freed
cells until it dispatches, unless its E(h) stops covering D. A head can also lose its work by
being cancelled (task timeout) after evicting. Each dispatched request completes and advances
its agent one step. With at most two evictions per agent per idle period (I5), evictions per
dispatch are bounded by 2 × agents (plus the evictions of cancelled heads, at most one plan per
task), and dispatches per task by `max_steps`+2.

**No starvation.** FIFO inside each priority class with no bypass, and admission (b) admits only
while the gate queue is empty and every admitted task has started. Waits are reported per
profile (P6).

## 3. KV accounting

- **Prompt size.** Before every request, `POST /tokenize {content, add_special: true,
  parse_special: true}`. It runs on the HTTP thread through `tokenize_mixed` (sc:6367–6376) and
  never stalls decode. It is the same tokenizer as `/completion`. Qwen2.5 has no BOS, so
  `add_special` changes nothing. Cost: one extra call of a few ms per request.
  - The local `models/qwen2.5-tokenizer` is used only offline and in E0c's equality check.
  - Character-count estimates are rejected: 2% error is ~330 cells at 16k.
- **`tok_check`.** On every `req_end`, compare tok(P) with prompt_n + cache_n. The expected
  difference is 0, and any difference is a `kv_violation`.
- **Reconcile.** Poll `GET /slots` at 1 Hz in **every** configuration. It reports
  `n_prompt_tokens = prompt.tokens.size()` (sc:808–822) and is cheap (`only_metrics`).
  - **Race.** SLOT_GET is a high-priority task answered while a decode runs, but slot ops and
    completions are declined then and replayed after the decode (sc:3001–3005; sq: declined
    tasks are put back in order). A snapshot can therefore predate an op or request the client
    sent earlier, even if that op's response reaches the client before the poll's. The reverse
    also happens (a request completes between the snapshot and its receipt). Ops never set
    `is_processing`, so the server flag cannot catch either case.
  - **Version rule.** Every `SlotKV` has a mutation counter `ver`, bumped on dispatch, on
    response, on op send and on op completion. The poller records each slot's `ver` and its
    outstanding request/op when the poll is *sent*. At receipt it compares only slots with
    (a) the same `ver` as at send time, (b) no request or op outstanding at send time or at
    receipt, and (c) server `is_processing == false`. A missing `n_prompt_tokens` (a slot
    with neither `task` nor `task_prev`, sc:814) is *unknown*, not 0.
  - Clean slot, server < ledger: a server purge. With the `active` gate or no gate this is a
    timestamped `kv_state` (reason `server_purge`). With the `full` gate it is a violation,
    and the agent becomes DROPPED.
  - Clean slot, server > ledger: logged as a suspected undercount. Reconcile never lowers a
    charge and never raises it silently: the charge becomes max(ledger, server), and the
    `undercount` violation is raised only if the same mismatch is seen on the next clean poll.
- **Baselines run the same manager in `observe` mode** (no gate, no ops). For pinned baselines
  (B1, B3) this gives exact token·s by state and purge timestamps, which replaces Phase 1's
  upper-bound reconstruction. The one exception is the active series: during a request the
  cells grow from the reused prefix to the response's count, so a request is valued at its
  size after the response, the same per-request bound as Phase 1.
- **B2 (`--cache-ram` > 0, unified, unpinned) is approximate.** With the default
  `cache_idle_slots`, every completion launch saves and clears **every** idle slot
  (sc:3071–3086, TAG_IDLE_SLOT_CLEAR). The client also learns a request's slot only from the
  stream. Observe mode for B2 therefore:
  - holds each request's charge f_r in an *unassigned* bucket until a stream message names
    its `id_slot` (every partial carries it, server-task.cpp:1059). B2 requests are sent with
    `return_progress: true`, which changes no server state but sends a 0% progress message
    when prompt processing starts (sc:4329–4333), right after the launch, instead of the
    first data message coming at the first token;
  - on that first message, moves every slot the ledger holds as idle to HOST_CACHED (s = 0 in
    the GPU pool), and tracks host-cache bytes separately (57,360 B/token, capped at
    `--cache-ram`);
  - skips reconcile for any slot the server reports `is_processing`.
  The remaining error is bounded by one prefill per launch. B2's KV breakdown is reported as
  approximate and excluded from the exact-token·s claim; its throughput, goodput and cache
  hit rate are exact.
- **HTTP 500 in a baseline.** The server sends "Context size has been exceeded" to each slot
  that is `is_processing` and clears it (sc:4724–4731). Set s_k = 0 only for slots whose own
  stream returned that error. Other client-side ACTIVE requests may still be queued, declined
  or deferred and keep their KV; they stay at max(s_k, f_r) until their own responses arrive.
- **Cancel or timeout.** The slot's content is unknown. `release()` erases it; the server defers
  the erase until the slot is idle (E0b busy). `n_erased` is the truth.
- **Validation.** One run per budget with `LLAMA_KV_CACHE_DEBUG=1` and `-lv 4`. `find_slot` then
  logs `used_max_p1` and used cells per ubatch (kv:372, 895–903). Check used ≤ U, and record the
  span (n_kv) over time. This is not used in timed runs.
- **B2 health check.** Every B2 run logs its per-request cache hit rate (cache_n / tok(P)) and
  counts the host-cache eviction lines ("making room for prompt cache entry",
  server-task.cpp:1798, WRN level). The save/load lines are TRC level, so they are counted
  only in the E0c B2 probe (§9). A B2 run whose hit rate is below B1's at the same point is
  investigated before it is reported, so a broken baseline is not reported as a win.

## 4. Lifecycle actions at tool boundaries

### 4.1 Actions

| Action | Control | Frees | Return cost |
|---|---|---|---|
| retain | none | 0 | 0 |
| offload | `POST /slots/k?action=save {filename: slot_k.bin}` → `erase` | s | restore, (κ_s + κ_r)·s, all slots stall |
| drop to prefix | `POST /completion {prompt: prefix_ids_f, n_predict: 0, id_slot: k, cache_prompt: true}` | s − L_f | re-prefill of s − L_f |
| erase | `POST /slots/k?action=erase` | s | re-prefill of s |

**Prefix ids.** `prefix_ids_f` is the LCP of the token ids of two first prompts of family f, built
at `start()` with `prompt.react_system` and two dummy questions, then tokenized. This gives a
stable token boundary. L_f ≈ 1,400.

- Families: `react6` (short and long) and `react6x` (explain).
- `oneshot` has no family (its prefix is 55 tokens), so it is erased.

**Why drop-to-prefix works.** The server sets n_past = LCP, `n_past−−` when the prompt is fully
cached, and calls `seq_rm(slot, n_past, −1)` (sc:4103–4110, 4314–4355). E0b truncate: prompt_n 1,
cache_n 1,399, 20 ms; a 5.5k prompt then fits with no purge; the revisit gives cache_n 1,400.
`n_predict` 0 behaves like 1.

**Offload race.** If the victim's tool ends during the save, skip the erase, unlink the file
(the slot is intact), and log `wasted_save`.

**Restore** happens inside the owner's gated dispatch. The gate reserves f_h ≥ S_img first. A
failed restore ("No available space", which clears the slot, sc:3323) becomes a re-prefill and
emits `kv_op ok=false`. The file is unlinked after the restore.

**No prefetch.** The next `_generate` follows `tool_end` with no await in between
(agent.py:177–186), so prefetching at tool end gains nothing. Only the optional
`clairvoyant_pf` reference restores early.

**Host tier.** Capacity H = h·c·57,360 B in `/dev/shm/slots`, with h ∈ {0, 0.5, 1, 2} (main:
h = 1, 0.94 GB at 16k). Offload is unavailable if it would exceed H.

### 4.2 Victims

Eligible set E(h) in class order. Within IN_TOOL the rank is the expected residual tool time
Ê[r | elapsed, tool] (largest first, so the agent that returns last goes first), ties broken by
larger s.

1. STALE.
2. IN_TOOL `search` (by residual).
3. IN_TOOL `lookup` (residual ≈ 0.1·scale − elapsed, so effectively last).
4. READY agents behind the head, latest-ready first. Priority-0 heads only; this class is
   required for the deadlock argument.

- Priority-1 heads take only from FREE, STALE and IN_TOOL agents whose TTL expired, and never
  erase an agent already dropped to its prefix (I5).
- **Ê[r] is estimated online by default** (`delay_model: online`), per tool, from the tool
  durations observed in the same run (`tool_end.idle_s`):
  - Ê[r | elapsed] = mean of (d − elapsed) over observed durations d > elapsed (empirical
    residual). Below 30 observations of a tool, it blends in a fixed weak prior that does not
    depend on the run's scale (lognormal median 1 s, σ 1.0, weight 10 observations).
  - Completed durations under-represent long calls early in a run; the bias is small after
    the first few hundred calls and is the same for every configuration that ranks by residual.
  - The manager never reads `tool_start.delay_s` (the true delay) except in `clairvoyant`.
  - `oracle_prior` is the ablation that uses the generator's distribution (lognormal median
    0.8, σ 0.6, cap 5 s) × `prior_delay_scale`. P5 misspecifies that scale.

**Action per victim**, in server-seconds:

- c_off = (κ_s + κ_r)·s.
- c_drop = t_pf(span)·(s − L_f).
- Offload iff host room exists and c_off < c_drop. Otherwise drop to prefix. Erase only when the
  slot has no family, or (priority-0 heads only) when every eligible victim is already at its
  prefix and D still exceeds what is free.
- With the priors, a `react_long` tail (s = 3.5k) is offloaded (175 vs 460 ms) and a
  `react_short` slot (s = 1.8k) is dropped (90 vs 88 ms; the prefix stays).
- STALE slots are dropped to prefix. They are erased only if D needs those cells too.

**Resolved: no slot-op stall budget.** Gain proposed a 5% stall budget. The cost comparison
already charges offload's exclusive server time, and a 5% cap would force the dearer drop at ×4,
where the predicted stall is 10–20%. Stall share is reported instead (P6).

### 4.3 Timing rule (ours = `pressure_ttl`)

At `tool_start`, freeze the break-even:

- B_i = (A + S_i)·min(c_off, c_drop), with A = Σ active tokens at that moment and S_i the
  agent's resident tokens. T\*_i = B_i / S_i.
- This is plan §4.1 with stall = op time.
- Example at 16k: A ≈ 9k, S = 3k, c_off = 0.15 s → T\* ≈ 0.6 s.

**Rent.** ρ_i = ∫ S_i · p(t) dt, where **p(t) = 1 iff a priority-1 head is blocked at the gate**.
A 50 ms tick accrues it. Agent i becomes *expired* when ρ_i ≥ B_i. Expiry makes it eligible for
priority-1 heads; nothing is evicted proactively. Returning agents (priority 0) ignore TTLs and
reclaim on demand.

**Why this split** (grafted from gain, fixing faithful's near-inert TTL):

- A returning agent is in-flight work. Blocking it only delays a finish that would free memory.
- A new task taking an idle agent's KV is the ski-rental decision: start now and pay B, or wait
  for natural release.
- Without pressure, nothing is evicted. E0b highwater shows proactive eviction buys no compute
  either.

**Timing variants** (every other rule is unchanged):

| Variant | Rule |
|---|---|
| `lazy` | T\* = 0: any idle agent is eligible for everyone (the reviewer design) |
| `nosteal` | T\* = ∞: priority-1 heads take only free/STALE cells |
| `fixed_ttl` | Proactive offload at tool_start + ttl, regardless of pressure, with ttl = `ttl_x` × the online median `search` duration (`ttl_x` = 1.25: ≈ 1 s at ×1, ≈ 4 s at ×4) (B4) |
| `always` | Offload at every tool start (B5) |
| `clairvoyant` | A priority-1 head may evict agent i iff its true remaining delay > T\*_i; no prefetch, so the ours-vs-R1 gap is the value of delay knowledge (R1) |
| `clairvoyant_pf` | `clairvoyant` plus an early restore at t_end − κ_r·S − 50 ms; reported separately as the prefetch component (optional, cut first) |

## 5. Unified vs per-slot

**Managed runs: unified, `-np 64`, one slot per agent for its lifetime.**

- `-np 64` costs nothing (E0a), and at most 16 agents are admitted. So agents ≤ slots and no
  virtualization is needed.
- `SlotPool.take(family)` prefers a STALE slot of the same family (its prefix is reused), then a
  FREE slot, then any STALE slot.
- `FixedK` keeps Phase 1's mapping: worker k → slot k.

**Per-slot (`--no-kv-unified`, `-np` = c/8192) is baseline B3 only.**

- It has no span tax, and it wins with ample memory (E1b: 54.8 at K = 16). But its regions are
  static: 2 at 16k and 4 at 32k.
- Tasks that peak above 8,192 fail individually with HTTP 400 (E0b perslot). That is 1–2 of 200
  tasks; they are reported as failures, not hidden.
- Virtualizing agents over regions would work (cross-slot restore takes 38 ms, E0b), but it
  fixes idle time, not size variance. Deferred.

**Not used: `--kv-unified-per-slot`** (it keeps the span tax) and **`--cache-ram` in any pinned
or managed run.** For the latter:

- With `cache_idle_slots` on (the default, common.h:640/644), every task launch saves and clears
  all idle slots (sc:3071–3086).
- A pinned request to a cleared slot then never reloads: f_keep = 0/0 = NaN, and `NaN < 0.5` is
  false (sc:1853). E0b confirms it: pinned revisits give cache_n [0, 0, 0, 0].
- So the runner refuses `--policy` or `pin` runs whose `server_cmd.txt` has `--cache-ram` ≠ 0.

## 6. Code layout

### 6.1 New modules

**`edgeserve/kvm.py`**

```python
@dataclass
class KVConfig:
    ctx: int; margin: int = 128
    gate: str = "full"            # none | active | full
    admission: str = "kv"         # fixed | kv | duty
    k: int = 8                    # fixed K, or K_max for kv/duty
    omega: float = 1.0; theta: float = 0.8; peak_q: float = 0.9
    lifecycle: str = "managed"    # observe | server | managed
    timing: str = "pressure_ttl"  # pressure_ttl | lazy | nosteal | fixed_ttl | always | clairvoyant | clairvoyant_pf
    ttl_x: float = 1.25           # fixed_ttl: multiple of the online median search duration
    delay_model: str = "online"   # online | oracle_prior
    victim: str = "residual"      # residual | lowest_id | largest
    actions: str = "cost"         # cost | offload | drop | erase
    host_frac: float = 1.0; slot_dir: str = "/dev/shm/slots"
    tick_s: float = 0.05; reconcile_s: float = 1.0; op_timeout_s: float = 5.0
    prior_delay_scale: float | None = None   # oracle_prior only
    priors: str = "configs/kv_priors.json"

@dataclass
class SlotKV:      # one per server slot
    slot: int; family: str | None = None
    state: str = "FREE"   # FREE ACTIVE IDLE READY STALE SAVING RESTORING OFFLOADED UNKNOWN HOST_CACHED(B2 observe)
    owner: str | None = None; s: int = 0; charge: int = 0; busy: bool = False
    ver: int = 0          # bumped on dispatch, response, op send, op completion (reconcile, §3)

@dataclass
class AgentKV:     # one per admitted task
    task_id: str; profile: str; slot: int; ticket: int
    R_hat: int = 0; n_req: int = 0; step: int = 0
    tool: str | None = None; tool_t0: float = 0.0; true_delay: float = 0.0
    B: float = 0.0; t_star: float = 0.0; rent: float = 0.0; expired: bool = False
    gave_up_step: int = -1; img_path: str | None = None; img_tokens: int = 0
    img_bytes: int = 0; ready_seq: int = -1; infeasible: bool = False

class CostModel:   # EWMA from server timings; priors from §1
    def observe(self, op: str, tokens: int, server_ms: float, span: int = 0) -> None
    def c_off(self, s: int) -> float
    def c_drop(self, s: int, keep: int, span: int) -> float

class KVManager:
    def __init__(self, ctl: "SlotControl", events: EventLog, cfg: KVConfig) -> None
    async def start(self) -> None            # n_ctx check, family prefix ids, tick/reconcile tasks
    async def close(self) -> None
    async def admit(self, task_id: str, profile: str, ticket: int) -> int   # -> slot
    async def acquire(self, slot: int, prompt: str, n_predict: int) -> int   # tokenize, gate, victims, restore -> n_predict_eff
    def on_response(self, slot: int, res: LLMResult) -> None
    def on_event(self, ev: dict) -> None     # EventLog listener: tool_start / tool_end
    async def release(self, task_id: str, status: str) -> None   # shielded
    def summary(self) -> dict[str, Any]
    # internals: _gate_ok, _eligible(head), _next_victim(deficit, head) -> tuple[SlotKV, str] | None
    # (one victim at a time, §2.2), _run_op(coro) -> Task (manager-owned, kept in self._ops,
    # awaited via asyncio.shield, §2.4 I4), _offload, _drop, _erase, _restore, _resolve_unknown,
    # _kick() (bumps self._epoch, resolves the head's and the admission waiter's wake futures;
    # sync-safe), _tick, _reconcile, _emit_state

class ManagedLLM:  # same generate() signature as LlamaServerClient; agent.py unchanged
    def __init__(self, base: Any, kvm: KVManager) -> None
    async def generate(self, prompt, n_predict, stop, slot=-1, ignore_eos=False) -> LLMResult
```

Behaviour notes for `kvm.py`:

- `acquire` raises nothing and returns n_predict_eff (§2.2); `ManagedLLM` sends that value. If
  the request is infeasible (tok(P) ≥ c − m), `ManagedLLM` returns
  `LLMResult(ok=False, error="kv_infeasible: context needs N >= c-m")`. The agent maps this to
  `ctx_overflow` (agent.py:92), and the runner rewrites the status to `kv_infeasible`, which
  counts as a failure.
- On cancellation, a waiter leaves the queue (and `n_admitted_not_started`) in a `finally`,
  and resolves the next head's wake future. Victim ops it started keep running (I4).
- `on_event` and `on_response` only mutate the ledger and call `_kick()`; they never await and
  never take the lock.

**`edgeserve/policies.py`**:

- `load_policy(name, overrides: dict) -> KVConfig` reads the presets in `configs/kvm.yaml`.
- `residual(tool, elapsed, scale) -> float`.
- `rank_victims(...)` and `choose_action(...)`, both pure functions so they can be unit-tested.

**`edgeserve/admission.py`**:

- `Admission` with a ticketed FIFO.
- `can_admit(agents, R_hat_j, waiters) -> bool` for `FixedK`, `KVAdmission` and
  `DutyAdmission`.
- `SlotPool.take(family) -> int` and `SlotPool.give(slot, stale: bool)`.

**`edgeserve/mockpool.py`**: `MockPoolLLM` (§7).

**`scripts/fit_kvm_priors.py`**: reads `tasks.jsonl` of the runs with no overflow failures
(`results/e1_concurrency/*/rep*`, `results/e2_budget/c64k/rep0`,
`results/e2_budget_k/c32k_k8/rep0`), keeps only tasks with status ok/forced, and writes
`configs/kv_priors.json`: `{profile: {P0, G: {q: [per step]}, steps_mean, duty}}` (§2.3).

**`bench/e0c_runtime.py`**: §9.

### 6.2 Edits (exact hook points)

| File:line | Change |
|---|---|
| events.py:15, 20–25 | `self.listeners: list[Callable[[dict], None]] = []`; `emit` calls each listener after writing, each inside `try/except` (an exception is written as a `kv_violation listener_error` event and swallowed, so it never reaches `Agent._run_react`). The KVManager learns tool_start/tool_end from the events the agent already emits, so **agent.py is untouched**. |
| llm.py:56–57 | `httpx.Limits(..., keepalive_expiry=2.0)`, below llama-server's 5 s keep-alive. |
| llm.py:67 | `prompt: str \| list[int]` (needed for drop-to-prefix ids). |
| llm.py:81–108 | Retry once on `RemoteProtocolError` if no byte was received (the Phase 1 race: 24 failures). Optional `on_first: Callable[[int], None]` attribute, called with `id_slot` on the first data message, and a `return_progress` flag (B2 observe mode only, §3). |
| llm.py after :114 | `SlotControl(base_url)`: its own `AsyncClient(max_keepalive_connections=0)`. Methods: `tokenize(text) -> list[int]`, `slot_save(slot, fn)`, `slot_restore(slot, fn)`, `slot_erase(slot) -> SlotOpResult`, `slots() -> list[dict]`. `SlotOpResult(ok, op, slot, tokens, bytes, server_ms, wall_ms, error)`. |
| runner.py:36 | Import `KVManager`, `ManagedLLM`, `load_policy`, `Admission`. |
| runner.py:166–170 | After creating the client: if `args.policy`, `cfg = load_policy(...)`, assert `/props` n_ctx == cfg.ctx and total_slots ≥ cfg.k. |
| runner.py:172–173 | `kvm = KVManager(ctl, events, cfg)`; `events.listeners.append(kvm.on_event)`; `Agent(ManagedLLM(llm, kvm) if kvm else llm, ...)`. |
| runner.py:185–186 | One sentinel per worker; workers = cfg.k. |
| runner.py:188–193 | `ticket = next(counter)` at `queue.get`; `slot = await kvm.admit(t["task_id"], t["profile"], ticket) if kvm else (k if args.pin else -1)`; `t_admit` taken after admission. |
| runner.py:195–203 | `finally: await asyncio.shield(kvm.release(task_id, trace.status))`; status `kv_infeasible` if flagged. |
| runner.py:204–213 | Add `row["kv"] = kvm.task_stats(task_id)` (waits, give-ups, restores, recompute). |
| runner.py:218–227 | `await kvm.start()` before `t0`; `await kvm.close()` after the gather. |
| runner.py:230–242 | `meta["kvm"] = asdict(cfg) \| {"policy": args.policy}`. Refuse if `server_cmd` contains `--cache-ram` ≠ 0 and (`args.pin` or managed). |
| runner.py:243 | `summary["kv"] = kvm.summary()`. |
| runner.py:248–263 | `--policy NAME`, `--kvm JSON`, `--kvm-config configs/kvm.yaml`. `--k` is K, or K_max for kv admission. |
| sweep.py:34–35 | DEFAULTS gain `policy: None, kvm: {}, env: {}`. |
| sweep.py:64–73 | Merge `p["env"]` into `env`; append `--policy`, `--kvm json.dumps(p["kvm"])`. |
| sweep.py:94 | Count `purging slot`, `Context size has been exceeded` and `making room for prompt cache entry` lines in `server.log`, and write them into `summary.json["server_log"]`. Purges are a violation only when the run's `gate` is `full` (§2.4 I3). |
| analysis/kv_timeline.py:38 | If `kv_state` events exist, build exact intervals from them (adding `offloaded` and `host` series). Otherwise use the Phase 1 reconstruction. |

New workloads are built by the existing generator; the frozen files are untouched:

- `configs/workloads/dev_long_200.yaml`: dev, 70/30, batch.
- `test_long_500.yaml`: test, 70/30, batch.
- `dev_mixed_200.yaml`: dev, default mix.
- `dev_long_poisson_r{a,b,c}.yaml`.
- `dev_short_200.yaml`: react_short only.

### 6.3 Events (`EventLog.emit`)

| Event | Fields |
|---|---|
| `kv_admit` | task_id, profile, slot, family, ticket, wait_s, R_hat, R_sum, cap, n_agents |
| `kv_gate` | task_id, slot, req_idx, prio, f, U, cap, deficit, wait_s, victims [[slot, action, freed]] |
| `kv_op` | slot, owner, for_task, op (save/restore/erase/truncate), reason (deficit/ttl/always/release/stale/cancel), tokens, freed, bytes, server_ms, wall_ms, ok, error |
| `kv_state` | task_id, slot, frm, to, tokens, reason (incl. `server_purge`, `host_cached`, `op_timeout`) |
| `kv_truncated` | task_id, slot, req_idx, tok_P, n_predict, n_predict_eff |
| `kv_exceeded` | task_id, slot, req_idx, tok_P, cap, gate (tok(P) ≥ c − m, logged under every gate, observe included, for the cross-configuration overlap of §2.2) |
| `kv_decision` | task_id, slot, step, tool, S, A, c_off, c_drop, B, t_star |
| `kv_expire` | task_id, slot, step, rent, B, elapsed_s |
| `kv_recon` | slot, source (req_end/op/slots_poll), ledger, server, diff, action |
| `kv_violation` | kind (purge [full gate only]/overflow/undercount [repeated]/tok_mismatch/busy_op/op_timeout/resolve_failed/listener_error/loop_error), detail |
| `kv_tick` (1 Hz) | U, cap, active, idle, ready, stale, offloaded_tok, host_bytes, agents, R_sum, gate_q, admit_q, pressure, op_ms_10s |

### 6.4 Config formats

**`configs/kvm.yaml`**

```yaml
defaults: {margin: 128, gate: full, admission: kv, omega: 1.0, peak_q: 0.9, lifecycle: managed,
           timing: pressure_ttl, victim: residual, actions: cost, host_frac: 1.0}
presets:
  observe:      {lifecycle: observe, gate: none, admission: fixed}      # B1, B3
  gated_k:      {lifecycle: server, gate: active, admission: fixed}
  adm_retain:   {lifecycle: server, gate: active}
  managed_k:    {admission: fixed}
  ours:         {}
  ours_h0:      {host_frac: 0}
  ttl_off:      {timing: fixed_ttl, ttl_x: 1.25, actions: offload}    # B4 (line: ttl_x 0.6 / 1.25 / 2.5)
  always_off:   {timing: always, actions: offload}                    # B5
  clairvoyant:  {timing: clairvoyant}                                 # R1 (no prefetch)
  clairvoyant_pf: {timing: clairvoyant_pf}                            # R1 + prefetch (optional)
  oracle_prior: {delay_model: oracle_prior}                           # P5
  lazy:         {timing: lazy}
  nosteal:      {timing: nosteal}
  victim_lowid: {victim: lowest_id}
  act_erase:    {actions: erase}
  act_offload:  {actions: offload}
  adm_duty:     {admission: duty, theta: 0.8}
```

**Sweep points** (the `ctx` must equal the `kvm.ctx` that the runner derives):

```yaml
name: p1_dev
defaults: {workload: workloads/dev_long_200.json, parallel: 64, reps: 3}
points:   # k and omega are the P0b / perf-split tuning winners (placeholders here)
  - {name: c16k_d4_b1,   ctx: 16384, delay_scale: 4, k: 8, policy: observe}
  - {name: c16k_d4_b2,   ctx: 16384, delay_scale: 4, k: 9, pin: false, parallel: k, cache_ram: 896, policy: observe}
  - {name: c16k_d4_sched, ctx: 16384, delay_scale: 4, k: 16, policy: adm_retain, kvm: {omega: 1.0}}
  - {name: c16k_d4_ours, ctx: 16384, delay_scale: 4, k: 16, policy: ours, kvm: {omega: 1.0}}
  - {name: c16k_d4_kvdbg, ctx: 16384, delay_scale: 4, k: 16, policy: ours, reps: 1,
     env: {LLAMA_KV_CACHE_DEBUG: "1"}, extra: "-lv 4"}
```

**`configs/sweeps/p0_baselines.yaml`** (untracked) already runs its `cr` points unpinned with
`-np K`. Set their `cache_ram` to the matched host tier h·c: 448 / 896 / 1792 MiB at
8k / 16k / 32k, not 4096. Give B2 the same 3-value K budget as B1 (§8.2 P0/P0b).
P0/P0b are tuning runs on the perf split; none of them is reported as a result.

## 7. MockPoolLLM and tests

`MockPoolLLM(script, capacity=None, n_slots=64, kappa_s=35e-6, kappa_r=15e-6, op_fixed_s=0.002,
t_prefill=2e-5, t_decode=1e-3, tok_noise=0)`:

**Tokens.**

- A token is a 4-character chunk, matching the `MockLLM` count.
- `tokenize` returns the chunks, and `/slots` returns `n_prompt_tokens`.
- With `capacity=None` the mock is cell-unbounded, so the existing tests are unaffected.

**Requests.**

- A request on a busy slot waits until the slot is free (deferred).
- Prefill keeps the LCP, frees the tail, allocates the uncached prompt in 512-token chunks, then
  allocates one cell per decoded token. The last sampled token is not stored, so s matches the
  server's formula. A fully cached prompt re-evaluates one token.

**When cells run out.**

- Purge the first idle slot with tokens, in id order, one per retry, and log it.
- If no idle slot has tokens, fail **all** processing requests with "Context size has been
  exceeded." and clear them.

**Slot ops.**

- save, restore and erase wait until the slot is idle.
- save and restore hold a global `_loop` lock for S·κ. Decode and prefill take the same lock per
  chunk, so ops stall everyone.
- restore first clears the destination slot. If free cells < n, it returns 400 and leaves the
  slot cleared.
- An `n_cmpl` child is cleared on release (not used; documents sc:666).

**Instrumentation.** Counters `overflows`, `purges` and `max_used`, plus a `check(ledger)` hook
called on every mutation.

**Fault injection** (for tests 9, 13, 18–22): `tokenize_delay_s`; `slots_stale=True` answers
`/slots` from a snapshot taken before queued ops and requests are applied (the server's
yielding behaviour, sc:3001–3005); `op_hang={op: seconds}`; and an HTTP-500 mode that fails
only the requests currently processing.

**Tests** (`tests/test_mockpool.py`, `test_kvm.py`, `test_admission.py`, `test_policies.py`):

1. The mock reproduces E0a: 4×(2.5k + 400) in an 8k pool all fail, and the purge takes the
   lowest id.
2. The mock reproduces E0b truncate (drop to prefix keeps 1,400) and restore into a full pool
   (fails and clears).
3. `test_gate_never_overflows`: 20 seeds, c = 4k, 12 agents, +700 tokens/step, for each of ours,
   always_off, managed_k and adm_retain. Expect 0 overflows and 0 purges in managed runs, with
   U ≤ c − m and charge ≥ actual asserted at every mutation.
4. Ledger exact: s == mock cells per idle slot after every response and op, and reconcile finds
   0 differences.
5. No deadlock when all memory is held by READY agents behind the head (c = 2× the largest
   request, 10 agents, `wait_for` 30 s).
6. Synchronized returns: bounded gate wait, FIFO dispatch within each priority.
7. A priority-1 head never evicts an unexpired idle agent. Under constant pressure the TTL
   expires at T\* ± 2 ticks.
8. At most one give-up event per (task, step), where a drop followed by an erase of the same
   idle period counts as one event, and the erase is issued only by a priority-0 head. None
   at all under ample memory, and `cache_n` identical to the observe run.
9. Admission: FIFO ticket order, nothing admitted while the gate queue is non-empty or an
   admitted task has not reached its first `acquire` (a mock `/tokenize` delay of 50 ms must
   not let a burst through), and K_max respected.
10. Offload → restore: cache_n equals the pre-offload s, files are deleted, the host cap holds,
    and a full host tier forces drop.
11. A tool that ends during a save skips the erase and leaves the slot intact.
12. A failed restore becomes a re-prefill and the task completes.
13. Cancellation mid-gate, mid-request and mid-op (a `wait_for` timeout on the head between
    `save` and `erase` of a *different* agent's slot) leaves no queue entry; the victim's op
    still completes, its charge equals the mock's cells, and no orphan file remains.
14. A kv_infeasible task (tok(P) ≥ c − m) terminates without blocking the queue and counts as
    a failure; a request with c − m − tok(P) < n_predict is sent with n_predict_eff and logged
    `kv_truncated`.
15. `observe` + FixedK reproduces the Phase 1 request/slot sequence exactly. Agent traces and
    prompts are byte-identical with and without the manager.
16. `active` gate + server purge: purges are detected by reconcile as `kv_state server_purge`,
    no `kv_violation` is raised, and nothing fails.
17. Policy functions: residual ranking (online empirical estimate and oracle prior), action
    choice (long tails offloaded, short ones dropped), and that B and T\* match hand-computed
    values.
18. Wake-ups: `tool_start`/`req_end` delivered through the sync listener wake a blocked head
    (no `RuntimeError`), only the head is woken, and an exception raised inside a listener is
    logged and does not fail the agent.
19. One victim at a time: if a victim's tool ends (or an active request finishes) during the
    first victim's op, no further victim is evicted when D ≤ 0.
20. Reconcile race: the mock answers `/slots` from a snapshot taken before a queued erase or
    completion; the version rule skips the slot, and no violation or DROPPED transition
    follows. A repeated clean mismatch does raise `undercount`.
21. Op timeout: a mock op that hangs past 5 s leaves the slot UNKNOWN with an upper-bound
    charge, and it is resolved by poll + erase without an overflow.
22. HTTP 500 in observe mode zeroes only the slots whose own stream got the error.

## 8. Configurations and experiments

### 8.1 Configurations

| ID | Preset | Admission | Gate | Lifecycle | Slots / server flags | Tuned (on perf, §2.3) |
|---|---|---|---|---|---|---|
| B1 | `observe` | fixed K | none | llama.cpp purge | pinned, `-np 64`, `--cache-ram 0` | K: 3 values per budget and scale, extended at the grid edge (P0/P0b) |
| B2 | `observe` | fixed K | none | llama.cpp host prompt cache (saves and clears every idle slot) | unpinned, `-np K`, `--cache-ram` h·c | K: 3 values per budget and scale, reaching ours' expected agent count |
| B3 | `observe` | fixed K = np | none | static regions | `--no-kv-unified -np c/8192` | — |
| `sched` | better of `gated_k` / `adm_retain` | fixed K or kv | active | llama.cpp purge | pinned, `-np 64` | K or ω: 3 values (same budget as ours) |
| B4 | `ttl_off` | kv | full | offload after ttl_x × online median delay | pinned, `-np 64` | ttl_x line {0.6, 1.25, 2.5} at 16k ×4; ours' ω |
| B5 | `always_off` | kv | full | offload at tool start | same | ours' ω |
| **Ours** | `ours` | kv (ω) | full | pressure-TTL + on-demand, cost-chosen action, online delay model | same, h = 1 | ω: 3 values |
| Ours-h0 | `ours_h0` | kv | full | drop to prefix only | same, h = 0 | ours' ω (tuned itself only if drop-only ships, §8.2) |
| R1 | `clairvoyant` | kv | full | knows the true delays; no prefetch | same | — |

- **B4 and B5 are ablations of ours**, not independently tuned baselines: they share ours' ω
  and admission and change only the timing rule. B4's TTL scales with the observed delays
  (≈ 1 s at ×1, ≈ 4 s at ×4) and gets a 3-point line at the headline point.
- **`sched` is the scheduling-only reference** for the lifecycle claim (§1). It is whichever of
  `gated_k` and `adm_retain` wins the perf-split tuning, and it gets fresh reps in the main
  table at 16k ×4 and 8k ×1 and in P1-test at 16k ×4.

**2×2** at 16k ×4 and 8k ×1. Knobs are tuned on perf: `gated_k` over 3 K values (B1's best K,
1.5× and 2×), `adm_retain` over ω ∈ {0.5, 1, 2}, `managed_k` over 2 K values (B1's best K and
1.5×). The chosen knob of each cell then gets **2 fresh dev reps** (3 for the `sched` cell)
before any decomposition delta is reported. B1 (`none` gate) is shown alongside to isolate
safety.

|  | llama.cpp lifecycle (active gate) | managed lifecycle (full gate) |
|---|---|---|
| fixed K | `gated_k` | `managed_k` |
| KV admission | `adm_retain` | **ours** |

**Ablations** at 16k ×4 (1 rep each; `lazy`/`nosteal` get 2):

- `lazy`, `nosteal`
- `victim_lowid` (the server's order)
- `act_erase` (no prefix kept), `act_offload`
- h ∈ {0.5, 2}
- `adm_duty`
- R1 (clairvoyant, no prefetch) and `clairvoyant_pf` (prefetch component, optional)

### 8.2 Matrix, GPU-hours, dates

Per-run time is ≈ 200/throughput plus 1 minute: ≈ 0.13 h at ×1 and 16k/32k ×4, 0.16 h at
8k ×1, 0.3 h at 8k ×4, ≈ 0.25 h at delay ×8 and for an open-loop run, and ≈ 0.37 h for a
test-500 run. **Tuning runs (P0, P0b, ω and 2×2 knobs) use the perf split and are never
reported; every reported number is a fresh post-freeze run in observe or managed mode on dev
or test.**

| Dates | Block | Runs | GPU-h |
|---|---|---|---|
| 11/15 | Client keep-alive fix and SlotControl (first commit); start P0 on perf, ×1: B1 K between grid points (7), B2 at matched h·c with 3 K per budget (9: 8k {3,4,6}, 16k {4,6,8}, 32k {8,12,16}), B3 (2) | 18 | 2.3 |
| 11/16 | E0c probe (§9), including the B2-under-pressure probe | 1 script | 0.7 |
| 11/17 | P0b ×4 on perf: B1 K ∈ {2,3,4}/{4,6,8}/{8,12,16} at 8k/16k/32k, plus {5,6}/{10,12}/{20} if the best K is at the grid edge; B2 K ∈ {4,6,8}/{6,9,12}/{12,16,20}; K_max at 131k ×4, K ∈ {8,16} | 20–25 | ≤ 4.5 |
| 11/18 | **Decide which mechanism ships** (full ours, or drop-only if E0c's fragmented-save probe makes κ_s + κ_r ≥ t_pf); code freeze tag; G0 shakedown: one `kvdbg` run per budget (tok_check 0, reconcile clean, used ≤ U, 0 violations) | 3 | 0.5 |
| 11/18–19 | Tuning on perf, for the shipped version only: ω ∈ {0.5, 1, 2} for ours at 6 points (18); 2×2 knobs at 16k ×4 and 8k ×1 (gated_k 3, adm_retain 3, managed_k 2 per point: 16) | 34 | 5.2 |
| 11/19–21 | **P1-dev main** (6 points, fresh): B1/B2/ours 3 reps at 16k ×4 and 8k ×1, 2 reps at 8k ×4 and 16k ×1, 1 rep at 32k ×1 and 32k ×4 (36); ours_h0 2 reps at 16k/8k ×4, 1 rep at each ×1 point (7); B4 TTL line (3) + 1 rep at 16k ×4 + 2 reps at 8k ×4 (6); B5 2 reps at 16k/8k ×4 (4); B3 at 16k/32k ×1 (2) | 55 | 9.2 |
| 11/22–24 | P2: fresh reps of the chosen 2×2 knobs (3 cells × 2 points × 2 reps, +1 rep for `sched`: 14) and ablations (12) | 26 | 3.6 |
| 11/25–26 | P3: delay {0, 0.5, 2, 8} at 16k (B1, ours; ours_h0 at {2, 8}; B1 at a second K at {2, 8}) (12); default mix at 16k ×1/×4 with knobs frozen (B1, B2, ours) (6); Poisson at 16k ×4 (8, below); GPU KV saved: ours and B1 at 24k ×4, 2 reps each (4) | 30 | 5.2 |
| 11/27 | P4 no-benefit: 64k ×1 (B1 K=16, ours), short-only mix at 16k ×1 (B1, ours) (4). P5: true scale {2, 8} with priors at 4: ours (online), `oracle_prior` misspecified at 4, R1 (6) | 10 | 1.6 |
| 11/28–30 | **P1-test**: test_long_500 at 16k ×4 and 8k ×1 × {B1, B2, ours} × 2 reps (12), plus `sched` at 16k ×4 × 2 (2) | 14 | 5.2 |
| 12/1–2 | Slack for reruns (a whole-configuration rerun after a violation fix, §2.4 I3, costs 0.5–1.5 h); P6 (overhead, stall share, per-profile slowdown) and P7 (predictions vs measured) offline | — | 2.0 |
| | **Total** | ≈ 215 | **≈ 38 + 2 slack = 40**, before subtracting the E0b probe hours; any E0b overrun comes out of the slack first, then the cut order |

- **B1's K per delay scale (P3).** At scales 0 and 0.5, B1 uses its ×1 best K. At 2 and 8 it
  runs the K the active-request model predicts (§1) and one neighbor, and the better is used.
- **The shipping decision is taken on 11/18, before any ω tuning.** If drop-only ships,
  `ours_h0` becomes ours: it is the version that is tuned, B4/B5/`act_offload`/h ablations are
  dropped, and the offload step of the cut order has already been taken. ω is never tuned for
  a mechanism that is later cut.

**Metrics.**

- **Closed-loop batch runs** (almost all of the matrix): steady throughput and goodput, plus a
  per-profile slowdown CDF (service time over that profile's unloaded service time). Every
  task arrives at t = 0, so latency there is the completion time and its p95 is ≈ 0.95 ×
  makespan (E2 c8k_k2: 448 s vs 468 s); **no latency claim is made from closed-loop runs.**
- **Latency claims come only from open-loop (Poisson) runs** at 16k ×4. Rates are fractions of
  measured capacity: λ ∈ {0.6, 0.85} × B1's steady throughput at that point, where every
  compared configuration is stable, for B1, B2 and ours (6 runs); plus λ = 0.85 × ours'
  capacity for B2 and ours (2 runs), where B1 is overloaded and its queue growth is reported
  instead of a p95. SLO: p95 ≤ 3 × the unloaded median service time. p95 only: no open-loop
  run reaches n ≥ 500, so p99 is not reported (plan §5.1).
- Next to every number: completed tasks, kv_infeasible (counted as failures, with task IDs),
  `kv_truncated`, other failures, EM/F1, output tokens, and **work-normalized throughput**
  (prefill + decode tokens/s) with tokens per task, so trajectory drift between
  configurations is visible.
- KV token·s by state from `kv_state` (exact for B1, B3 and managed runs; approximate for B2,
  §3); host peak bytes; op stall share; mean attention span.
- **Noise and tests (pre-registered).** B1's 3 dev reps at 16k ×4 and 8k ×1 measure the CV
  under memory pressure (E2 points were single runs, and the ±2.7% CV was measured with ample
  memory). Throughput claims at the headline cells use a Welch t-test on per-run steady
  goodput (≥ 3 reps per cell on dev), with a bootstrap CI of Δ over steady-window completions
  (tasks resampled within runs, runs resampled across reps). The 16k ×4 headline is confirmed
  on test-500.
- **Quality equivalence.** TOST on per-question F1 averaged across reps, paired per question.
  The paired ΔF1 SD is ≈ 0.31, so the 90% CI half-width is ≈ 0.036 at n = 200 and ≈ 0.023 at
  n = 500. The margin is **0.03 on test-500** and **0.05 on dev**, and the test is calibrated
  with a B1-vs-B1 null (rep 1 vs rep 2), which must pass at the same margin.

**Cut order if behind schedule** (pre-registered; P5's ours vs `oracle_prior` comparison is
**not** cuttable): `clairvoyant_pf`, then R1 (ablation and P5) → `adm_duty` → the remaining
ablations (h ∈ {0.5, 2}, `victim_lowid`, `act_*`) → P4 short-only → offload (drop-only ours;
decided on 11/18) → TTL (lazy only) → Poisson.

**Implementation order.**

1. llm fixes.
2. MockPool and its tests.
3. Ledger, gate, admission, observe mode (incl. B2's approximate mode).
4. Drop/erase actions and on-demand victims (manager-owned ops).
5. Offload/restore and the cost model.
6. TTL, the online delay model and the variants.
7. Clairvoyant.

Code freeze is the 11/18 tag before G0.

## 9. E0b outcomes and the remaining branches

**E0b, resolved:**

| Probe | Outcome | Design consequence |
|---|---|---|
| idlecost | Idle cells cost everyone: decode 293 → 238 / 203 tok/s and 2k prefill 0.55 → 1.01 / 1.47 s at 24k / 48k idle cells, whether above or below the active ones | A span term in c_drop; reported, bounded by c |
| highwater | Cost follows the top cell (208 vs 295 tok/s) | No compute claim and no proactive eviction |
| truncate | `n_predict` 0 works: 20 ms, keeps 1,400 cells, frees the rest | Drop to prefix is the drop action |
| busy | Save is deferred until the stream ends (3.5 s wall, 50 ms server); erase takes 1.4 ms | Ops only on idle slots, with a 5 s op timeout; release erases |
| fragment | Restore into holes works (46.8 ms, cache_n 2,999, 0 purges) | Restore is feasible iff free cells ≥ S; no extra margin |
| cacheram | Pinned revisits [0,0,0,0]; unpinned A,B,C,A → [0, 2999, 0, 2999] | B2 is unpinned at matched h·c; `--cache-ram 0` is enforced elsewhere |
| perslot | 4,096 per slot, cross-slot restore in 38 ms, a 5k prompt → HTTP 400 | B3 is static; virtualization deferred |

**E0c** (`bench/e0c_runtime.py`, 11/16, 0.7 h): open questions with branches.

| Probe | Prediction (source) | If confirmed | If not |
|---|---|---|---|
| Save/restore of a **fragmented** slot (react_long-like, built under 8 concurrent decoders) | Save is ~1.5–2× slower: `state_write_data` copies one range per layer for K and V (kv:2232–2270), and decode cells interleave | The CostModel EWMA absorbs it; offload still beats drop for long tails | If κ_s + κ_r ≥ t_pf, the cost rule picks drop everywhere and the headline becomes gate + admission + victim/timing with drop (the plan §4.4 r > 10 branch); B5 and offload ablations are dropped |
| `/tokenize` == prompt_n + cache_n on 200 real dev prompts | Exact | m = 128 | m = 4 × max diff, and tok_check becomes a warning |
| Drop to prefix with **token-id** prompts (family LCP ids) | Revisit cache_n = L_f | As specified | Send the prefix as text and accept the 1-token loss |
| Op issued right after a stream's final message | Not deferred (wall ≈ server time) | — | Issue ops only after a /slots check shows `is_processing` false |
| Client disconnect mid-stream | Slot stops and becomes idle; erase works | release() as specified | release() polls /slots until idle |
| `LLAMA_KV_CACHE_DEBUG` overhead | Large | Validation runs only | — |
| **B2 under pool pressure**: unpinned, `--cache-ram` h·c, 6 agents with the shared ReAct prefix in an 8k pool, `-lv 4` to count the TRC prompt-cache save/load lines | Revisits hit the host cache by prefix (server-task.cpp:1838, f_keep ≥ 0.25), so cache_n ≈ the agent's previous context; the E0b pressured case missed only because it was pinned (f_keep NaN, sc:1853) | B2 is a strong baseline; its grid reaches ours' agent count (§8.2) | If revisits still miss, trace why before P0 results for B2 are used; B2 is reported with the cause, not dropped |
| Reconcile race | A `/slots` poll sent during a decode returns a snapshot taken before a queued erase (sc:3001–3005) | The version rule (§3) is required | The rule is kept anyway (it costs nothing) |

**Other risks.**

| Risk | Mitigation |
|---|---|
| B2 (the server's own host cache) matches ours at ×4 | The claim narrows to safety, ranked victims and pressure-gated timing; the 2×2 shows which part matters; reported honestly |
| Op stall eats the ×1 gain | The cost rule prefers drop for short tails; 16k ×1 and 32k ×1 are pre-registered as small or no-benefit |
| Accounting drift, cancellation leaks | /tokenize, reconcile with the version rule, manager-owned ops, violation counters; violation runs are kept and counted, and a bug fix reruns every rep of that configuration (§2.4 I3) |
| 16k ×1 gain within noise | Pre-registered as small; the headlines are 16k ×4 (lifecycle, vs `sched`, 3 dev reps + test-500) and 8k ×1 (safety/admission) |
| B1 retuned at ×4 leaves little gain (§1: ≈ 0 to +25% at 16k ×4) | Predictions already use the retuned B1; the claim is ours vs `sched`, and a small gain is reported as measured |
| ω does not transfer across points | Per-point tuning with the same grid budget as K; P3 reuses frozen knobs on the default mix |
| Client CPU (4 vCPUs shared) | Client CPU is logged; switch to `stream=false` if a core exceeds 60% (except B2, whose observe mode needs the stream, §3) |

## 10. Which design won, and why

**The base is the faithful design**, the winner for 2 of 3 judges. Its source reading was the
most accurate:

- the purge → halve-`n_batch` → fail-all order;
- `/slots` reporting `n_prompt_tokens`;
- the NaN f_keep bug in the pinned `--cache-ram` baseline;
- the flash-attention `scan_mask` condition;
- the active vs full gate split, with the retain + full gate deadlock.

It also had the fairest baselines and pre-registered attribution.

**Grafted from the reviewer design:**

- exact s = prompt_n + cache_n + predicted_n − 1 and the reset-on-500 rule;
- E0b-calibrated costs (erase 1.4 ms, not 20);
- READY agents in the victim set, which closes faithful's deadlock gap;
- runtime kv_infeasible (react_long reaches 11,266 tokens);
- the observe-only manager for every baseline;
- the `ManagedLLM` wrapper (here extended so agent.py is not touched at all);
- the client keep-alive fix;
- honest iso-memory framing with `ours_h0` as a main row;
- the fragmented-save E0c probe;
- one lock serializing dispatch;
- the duty-cycle admission, as an ablation.

**Grafted from the gain design:**

- the two-tier timing rule (on-demand for returning agents, pressure-TTL only for new tasks'
  first requests), which gives the TTL a non-redundant role;
- drop-to-prefix with family LCP ids;
- the per-victim offload/drop choice based on the private tail;
- the KV-debug validation gauge;
- "admit only while the gate queue is empty".

**Rejected, with reasons:**

- **Gain's shared-prefix seeding** (its largest lever): `n_cmpl` children are cleared on release
  (sc:666–669).
- **Gain's local tokenizer with a drift allowance**: /tokenize is exact and costs no main-loop
  time.
- **Gain's stall budget**: §4.2.
- **Faithful's ω ≤ h coupling**: §2.3.
- **Faithful's headroom-based pressure and proactive rent**: highwater shows there is no compute
  payoff, and it made the TTL nearly inert.
- **The reviewer's removal of TTL**: the instructor's emphasis is the lifecycle decision. The
  two-tier rule keeps the TTL where the ski-rental trade-off is real, and `lazy` tests the
  reviewer's alternative directly.
- **The reviewer's A\* = 6 frozen compute cap**: it was calibrated on 20–50k spans. K_max is the
  measured ample-memory best K instead.
