# Cost model and microbenchmark plan

Goal: predict, before measuring, when KV memory (not compute) limits agent-serving
throughput, then calibrate those predictions with microbenchmarks on each machine.

- `costmodel/cost_model.py`: first-principles predictions. `--save` freezes them to `predictions/`.
- `bench/run_microbench.sh`: prefill vs length, decode vs depth, batched decode vs batch.
- `bench/slot_offload_bench.py`: real cost of llama-server slot save/restore vs re-prefill.

## Predictions (spec numbers, uncalibrated, frozen 2026-10-29)

| setting | KV/agent | max resident agents | decode ridge batch | idle in tools | ideal-offload headroom | re-prefill | offload round trip |
|---|---|---|---|---|---|---|---|
| A10G, 7B, f16, react_short | 131 MB | 148 | 95 | 64% | 1.21x | ~0.96 s | ~11 ms |
| A10G, 7B, f16, react_long | 459 MB | 42 | never | 63% | 1.32x | ~3.6 s | ~37 ms |
| A10G, 7B, q8_0, react_short | 70 MB | 279 | 43 | 64% | 1.00x | ~0.96 s | ~6 ms |
| A10G, 7B, q8_0, react_long | 244 MB | 79 | never | 64% | 1.34x | ~3.6 s | ~19 ms |
| M1 Pro, 7B, f16, react_short | 131 MB | 43 | 7 | 33% | 1.00x | slow | no separate host tier |

"Headroom" = decode throughput if every resident agent were decoding, divided by
always-retain throughput at full KV capacity. It is an upper bound for any
retain/offload/drop policy at that operating point.

What these predict:

1. **react_short on A10G needs ~150 concurrent agents before KV binds**, and q8_0 KV
   removes the headroom entirely. Phase 2 should be evaluated on react_long, or with
   `-c` capped to model a smaller device (state this explicitly in the report).
2. **At long context, decode never becomes compute-bound**: per-sequence KV reads
   exceed per-sequence FLOPs (GQA, 8k tokens). Freeing idle KV still helps, ~1.3x.
3. **Offload is ~100x cheaper than re-prefill** if the transfer path runs near PCIe
   speed. If `slot_offload_bench.py` confirms this, always-offload is the policy to
   beat and retain-vs-drop is not the interesting question. If save/restore turns
   out slow or blocking, the trade-off is real.
4. **The M1 Pro saturates compute at batch ~7**, so it cannot reproduce the
   memory-bound regime at all. Never draw performance conclusions from local runs.

Assumptions most likely to be wrong, check first: `flops_eff` (M1 prefill time
especially), `bw_eff`, the A10G FP16 peak, and the 1.5 GiB overhead.

Known discrepancy to explain: the model predicts ~68% idle time for the 1.5B model
on M1, but the Phase 1 smoke test measured 36%. Either per-step time is larger than
modelled (client overhead, longer outputs) or the measurement counts differently.

## What runs where

| Work | M1 Pro (local) | A10G (AWS) |
|---|---|---|
| Agent fix, quality gate, F1 | yes; same GGUF, quality carries over statistically | spot-check once |
| Orchestrator, policies, admission control, logging | yes; use `-c` to force pressure, 1.5B/3B model for speed | final runs |
| Correctness checks (ledger, cached counts, restore reuse) | yes | rerun once |
| "Predictions identical across K" | yes | **rerun**: CUDA batch kernels can change outputs |
| Microbenchmark scripts | develop and debug here | real numbers |
| Throughput / latency / headroom results | **no** (compute-bound at batch ~7) | yes |
| Offload cost | SSD only (unified memory) | `/dev/shm` = host RAM; instance NVMe = SSD tier |

Rule: local runs validate behaviour, AWS runs produce numbers.

## Porting checklist

1. Build llama.cpp b11960 with CUDA on the instance; record `--version` in `meta.json`.
2. Run `run_microbench.sh a10g` with `NPL=1,2,4,8,16,32,64,128`.
3. Start llama-server with `--slot-save-path /dev/shm/slots` and run
   `slot_offload_bench.py --tag a10g-shm`; repeat on the instance-store NVMe.
4. Replace `bw_eff`, `flops_eff`, overhead and `host_link_bw` in `cost_model.py`
   with fitted values. Keep the original frozen predictions and report both.
5. g5.xlarge has 16 GiB host RAM shared by the server, runner, BM25 index and
   `/dev/shm`: size the offload tier against that.
6. Stop the instance between sessions.
