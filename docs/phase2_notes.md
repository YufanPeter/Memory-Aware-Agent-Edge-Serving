# Phase 2 notes: llama.cpp b11960 internals that the Phase 2 ideas depend on

Source: `ggml-org/llama.cpp` tag `b11960` (commit 8642541), read on 2026-11-16.
Model facts: Qwen2.5-7B-Instruct, 28 layers, 4 KV heads, head dim 128, f16 KV, flash attention on.

## 1. Attention span in the unified KV pool ("speed tax", direction 1)

**Attention spans every cell up to the highest occupied one.**
`llama_kv_cache::get_n_kv()` (`src/llama-kv-cache.cpp:1255`) returns
`min(size, max(256, PAD(cells.used_max_p1(), 256)))`, where `used_max_p1()`
(`src/llama-kv-cells.h:97`) is the index of the last used cell + 1.
Every K/V view handed to attention has `n_kv` cells. Cells that belong to other
sequences, idle or finished slots, and empty holes below the highest used cell are masked
with -inf but still part of the tensor.

**With `--kv-unified` there is one stream, so every sequence's attention covers the whole span.**
Two CUDA paths can skip masked work; neither applies to our configuration:

| Path | Where | Condition | Applies to us? |
|---|---|---|---|
| Sparse FA (gathers only visible columns) | `fattn-mma-f16.cuh:1788` `may_use_sparse` | head dim 256/512/576 only | No: Qwen2.5 head dim is 128 |
| Mask scan (`flash_attn_mask_to_KV_max`) | `fattn-common.cuh:1117` `scan_mask` | `Q->ne[1] >= 1024` (large prefill ubatch) or `Q->ne[3] > 1` (several streams) | Not for decode in unified mode; yes for per-slot streams (`--no-kv-unified`) |

Even when the mask scan runs, it only trims fully masked tiles **at the tail**
(it walks down from the end and stops at the first tile with a visible value). Holes in the
middle are always computed.

**Consequence (hypothesis H1, to be measured in X1):** in unified mode one decode step costs
roughly `a + b · B · n_kv`, where B is the number of sequences decoding and n_kv tracks the
highest occupied cell, not the tokens a sequence owns. With K agents of similar size,
n_kv ≈ K · L, so attention work grows ~K². With per-slot streams the mask scan trims each
stream to its own length, so it grows ~K. This matches Phase 1 E1/E1b (unified collapses
after K = 8; per-slot +27% at K = 16, ×2.1 at K = 32).

It also means:
- Idle (in-tool) and stale (finished-task) KV costs every decoding agent time, not just memory,
  **if it sits at or below the top of the occupied range**.
- Freeing KV shortens the span only if it lowers the highest occupied cell. Freeing cells in
  the middle leaves n_kv unchanged until the top is freed.
- n_kv is padded to multiples of 256, so the span moves in 256-cell steps.

## 2. Where new tokens go; no compaction

`find_slot()` (`src/llama-kv-cache.cpp:893`) searches from the stream's `head` (the cell after
the last placement), wrapping around. If `head > used + 2·n_tokens`, it restarts from 0
("hoping to fill it"), so with plenty of free space new tokens tend to fill low holes first.
A single sequence's tokens can be scattered (non-contiguous) for decode ubatches.

**Defragmentation is gone:** `--defrag-thold` is deprecated and a no-op
(`common/arg.cpp:2545`); the cell-move code is commented out (`llama-kv-cells.h`). Nothing
moves cells once they are placed, so one long-lived sequence at a high index pins n_kv.

Open question for X1: does save → erase → restore move a slot's cells to low free cells? It
should, because restore goes through `find_slot` from the reset head, but this is not yet
measured.

## 3. Sharing cells between slots (direction 3)

`llama_kv_cache::seq_cp(src, dst, p0, p1)` (`src/llama-kv-cache.cpp:458`): when both sequences
live in the same stream (always true with `--kv-unified`), it **only adds `dst` to each cell's
sequence set; no data is copied**. A cell stays occupied while any sequence references it, so
erasing one slot (`seq_rm` on its id) leaves the shared prefix cells in place for the others.
Across streams (per-slot mode) it copies the whole buffer, so sharing only works in unified mode.

llama-server does not expose this. Its slot actions are `save`, `restore` and `erase`
(`tools/server/server-context.cpp:5852`), each posted as a task to the main loop, which is why
they block all decoding (Phase 1 E0a). A `fork` action would follow the `erase` pattern:
`seq_rm(dst)`, `seq_cp(src → dst, 0, N)`, then set the slot's cached prompt tokens to the
first N tokens of the source, so the next request's prefix match sees them.

## 4. Observability

- `LLAMA_KV_CACHE_DEBUG=1` makes `find_slot` log `n = used_max_p1, used = <cells used>,
  head, size` for every ubatch (debug level, so the server needs verbose logging). This gives
  a direct n_kv time series for X1 and for the main runs, instead of reconstructing occupancy
  from request token counts as Phase 1 did.
- The server still has no KV-occupancy gauge in `/metrics`.

## 5. Plan impact

> Update 2026-11-18: the plan changed (docs/phase2_ideas.md). The "hygiene" policy that charged idle
> KV this tax was tried and removed: X1 shows the tax alone justifies dropping an idle agent only after
> ~30 s, and erasing stale KV lost prefix reuse. The span cost now motivates the grouped KV pool
> (idea B). The bullets below are the original 11-16 notes.

- Direction 1 is supported by the source; X1 measures the size of the effect (b in the cost
  model) and whether erasing or relocating high-index slots shrinks n_kv in practice.
- The idle-KV cost in our policy becomes rent (memory under pressure) + tax
  (`B_decoding · Δn_kv · b` per second), where Δn_kv is how much the agent's cells raise the span.
- The client cannot see cell positions over HTTP. Options: infer from the debug log, or keep
  the policy position-agnostic and evict the most recently placed idle KV first (most likely
  to sit at the top).

## 6. Measurements so far (A10G, 2026-11-16)

### Fork patch works (X3, `bench/x3_fork_check.py`, 1.5B, 16k pool, 8 slots × (1,400 shared + 1,000 own))

| Mode | First-request `cache_n` per slot | Server purges | Second pass re-prefill |
|---|---|---|---|
| copy (stock behaviour: each slot prefills the prefix) | 0 | 5 | 3 slots × 2,400 tokens |
| fork (`tools/llama_patch/fork_slot.diff`) | 1,400 on every slot | 0 | none |

Generations are identical in both modes (8/8, temperature 0, single requests), so shared cells
give the same results as per-slot copies. The patch adds 122 lines and changes no existing path;
the unpatched binaries are kept in `~/llama.cpp/build/bin-orig`.

### Base smoke test (`configs/sweeps/p2_smoke.yaml`: 7B, 16k pool, long mix, 50 dev questions)

| Config | Completed | Tasks/min | Correct/min | Server overflows | Notes |
|---|---|---|---|---|---|
| fixed K = 4 | 46/50 | 35.9 | 21.1 | 6 | 4 tasks lost to overflow |
| fixed K = 8 | 2/50 | 2.8 | 2.8 | 60 | the Phase 1 cliff |
| gate only, K = 8 | 50/50 | 19.0 | 10.7 | 0 | thrashing: 150 drops, cache hit 9% |
| admission rho = 1 | 50/50 | 24.4 | 14.2 | 0 | admission too conservative |
| base (rho = 1.5; then named l1) | 50/50 | 28.2 | 15.2 | 0 | same, plus lost prefix reuse |

Safety holds on the real server: no overflow and no server-side purge under any managed policy,
and the token estimate was never below the actual size (estimate/actual median 1.11, min 1.008).
Throughput was below fixed K = 4 for three reasons: a gate without admission thrashes; the
admission prior (Phase 1 p95 peaks) is conservative until 8 tasks of a profile finish; and with
64 workers each pinned to its own slot, new tasks landed on empty slots, losing system-prompt
reuse (cache hit 59-67% vs 80%). Fix applied: under admission the manager now picks the slot
(stale slot of the same prompt family first, then an empty one, then another family's stale slot).

### X1: idle KV slows every decoding request (`bench/x1_speed_tax.py`, 7B, `-c 131072`, slots of 2,048 tokens)

Decode time per step (ms/token) with B requests decoding and N idle slots of 2,048 tokens:

| Idle slots (idle tokens) | B = 4 | B = 8 |
|---|---|---|
| 0 | 13.9 | 20.5 |
| 8 (16k) | 15.9 | 22.6 |
| 16 (32k) | 18.0 | 24.7 |
| 32 (64k) | 21.9 (+58%) | 29.0 (+41%) |

Whether the idle KV sits above or below the active slots makes no difference (same n_kv).
What matters is how freeing it changes n_kv (B = 4, N = 32, debug pass with `LLAMA_KV_CACHE_DEBUG=1`):

| Condition | Used cells | n_kv | ms/token |
|---|---|---|---|
| no idle KV | 8,443 | 8,443 | 14.1 |
| 32 idle slots | 73,979 | 73,979 | 22.4 |
| idle slots below the active ones, then erased | 8,443 | 73,727 | 21.9 |
| ... then each active slot saved, erased, restored | 8,443 | 8,443 | 14.1 |
| idle slots above the active ones, then erased | 8,443 | 8,443 | 14.2 |

Findings:
- Step time ≈ 13.0 ms + 0.127 ms per 1,000 cells of n_kv (B = 4; 0.130 at B = 8). The slope barely
  depends on B: every decode step reads the whole span once (1,000 cells = 56 MB of f16 KV, about
  0.09 ms at the A10G's ~600 GB/s), so the cost is memory bandwidth, paid for idle and stale cells
  alike. Gate 1 passes.
- Erasing KV below the highest used cell frees memory but not time: the span stays.
- Save → erase → restore relocates a slot to the lowest free cells (restore goes through
  `find_slot`, which restarts from cell 0 when there is enough free space). That is a usable
  client-side defragmentation, at the cost of a blocking save/restore (~0.1 s per 3k tokens).
- Implication for the policy: for one idle agent of S tokens the tax is 0.127 µs × S per step,
  about 1.9% of a 20 ms step for S = 3k. Re-prefilling 3k tokens costs ~0.66 s, so the tax alone
  justifies dropping an agent only after ~30 s idle, far beyond a 0.8 s tool call. The tax matters
  for KV that stays: stale KV of finished tasks and anything that pins the top of the pool.
