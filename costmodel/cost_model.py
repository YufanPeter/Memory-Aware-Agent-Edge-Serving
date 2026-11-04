"""First-principles cost model for KV-cache decisions in agent serving.

Writes down numeric predictions BEFORE any measurement, so microbenchmarks can
confirm or refute them. Every hardware number below is a spec-sheet or assumed
value; replace the *_eff factors and budgets with fitted values from bench/.

Usage:
    python costmodel/cost_model.py                 # print tables
    python costmodel/cost_model.py --save          # also freeze to predictions/
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

GiB = 1024**3


@dataclass
class Model:
    name: str
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    n_params: float
    weight_bytes: float  # size of the GGUF file actually loaded


@dataclass
class Hardware:
    name: str
    mem_bytes: float  # memory the GPU may use for weights + KV + buffers
    mem_bw: float  # bytes/s, spec
    peak_flops: float  # FLOP/s, spec
    bw_eff: float  # assumed fraction of mem_bw reached by decode kernels
    flops_eff: float  # assumed fraction of peak_flops reached by quantized matmuls
    overhead_bytes: float  # compute buffers, context, OS share
    host_link_bw: float | None  # bytes/s to host RAM; None = unified memory


QWEN_7B = Model("Qwen2.5-7B Q4_K_M", 28, 28, 4, 128, 7.62e9, 4.68e9)
QWEN_3B = Model("Qwen2.5-3B Q4_K_M", 36, 16, 2, 128, 3.09e9, 1.93e9)
QWEN_1_5B = Model("Qwen2.5-1.5B Q4_K_M", 28, 12, 2, 128, 1.54e9, 1.1e9)

A10G = Hardware(
    name="A10G 24GB (g5.xlarge)",
    mem_bytes=23028 * 2**20,  # nvidia-smi memory.total on mem-aws
    mem_bw=600e9,
    peak_flops=125e12,  # FP16 tensor, A10 spec sheet; verify for A10G
    bw_eff=0.7,
    flops_eff=0.3,
    overhead_bytes=1.5 * GiB,
    host_link_bw=25e9,  # PCIe Gen4 x16, practical; measure it
)

M1_PRO = Hardware(
    name="M1 Pro 16-core GPU, 16GB",
    # Metal's recommendedMaxWorkingSetSize is ~2/3 of RAM on 16 GB machines;
    # llama.cpp prints the exact value at startup.
    mem_bytes=10.7 * GiB,
    mem_bw=200e9,
    peak_flops=5.2e12,
    bw_eff=0.7,
    flops_eff=0.5,
    overhead_bytes=1.0 * GiB,
    host_link_bw=None,
)

KV_ELEM_BYTES = {"f16": 2.0, "q8_0": 34 / 32}  # q8_0: 32 int8 + one fp16 scale

PROFILES = {"react_short": 2289, "react_long": 8000}  # peak context, tokens
TOOL_MEDIAN_S = 0.8
STEP_NEW_TOKENS = 60  # median prefill per step with prefix caching (Phase 1 report)
STEP_OUT_TOKENS = 37  # median decode per step


def kv_bytes_per_token(m: Model, kv_type: str) -> float:
    return 2 * m.n_layers * m.n_kv_heads * m.head_dim * KV_ELEM_BYTES[kv_type]


def kv_budget_bytes(hw: Hardware, m: Model) -> float:
    return hw.mem_bytes - m.weight_bytes - hw.overhead_bytes


def decode_step_s(hw: Hardware, m: Model, batch: int, ctx: int, kv_type: str) -> float:
    """One decode step for `batch` sequences, each with `ctx` tokens of context."""
    bytes_read = m.weight_bytes + batch * ctx * kv_bytes_per_token(m, kv_type)
    flops = batch * (2 * m.n_params + 4 * ctx * m.n_layers * m.n_q_heads * m.head_dim)
    return max(bytes_read / (hw.mem_bw * hw.bw_eff), flops / (hw.peak_flops * hw.flops_eff))


def ridge_batch(hw: Hardware, m: Model, ctx: int, kv_type: str) -> int | None:
    """Smallest batch at which decode stops being memory-bound."""
    for b in range(1, 1025):
        bytes_read = m.weight_bytes + b * ctx * kv_bytes_per_token(m, kv_type)
        flops = b * (2 * m.n_params + 4 * ctx * m.n_layers * m.n_q_heads * m.head_dim)
        if flops / (hw.peak_flops * hw.flops_eff) >= bytes_read / (hw.mem_bw * hw.bw_eff):
            return b
    return None


def prefill_s(hw: Hardware, m: Model, n_tokens: int) -> float:
    flops = 2 * m.n_params * n_tokens + 2 * n_tokens**2 * m.n_layers * m.n_q_heads * m.head_dim
    return max(flops / (hw.peak_flops * hw.flops_eff), m.weight_bytes / (hw.mem_bw * hw.bw_eff))


def offload_roundtrip_s(hw: Hardware, kv_bytes: float) -> float | None:
    return None if hw.host_link_bw is None else 2 * kv_bytes / hw.host_link_bw


def idle_fraction(hw: Hardware, m: Model, ctx: int, kv_type: str) -> float:
    """Share of an agent's residency spent waiting on a tool, at batch 1."""
    step = prefill_s(hw, m, STEP_NEW_TOKENS) + STEP_OUT_TOKENS * decode_step_s(hw, m, 1, ctx, kv_type)
    return TOOL_MEDIAN_S / (TOOL_MEDIAN_S + step)


def decode_tput(hw: Hardware, m: Model, batch: int, ctx: int, kv_type: str) -> float:
    return batch / decode_step_s(hw, m, max(batch, 1), ctx, kv_type)


def predict(hw: Hardware, m: Model, kv_type: str) -> dict:
    per_tok = kv_bytes_per_token(m, kv_type)
    budget = kv_budget_bytes(hw, m)
    out = {
        "hardware": hw.name,
        "model": m.name,
        "kv_type": kv_type,
        "kv_bytes_per_token": per_tok,
        "kv_budget_gib": budget / GiB,
        "max_resident_tokens": int(budget // per_tok),
        "profiles": {},
    }
    for prof, ctx in PROFILES.items():
        kv = ctx * per_tok
        ridge = ridge_batch(hw, m, ctx, kv_type)
        max_agents = int(budget // kv)
        tput = {b: decode_tput(hw, m, b, ctx, kv_type) for b in (1, 4, 8, 16, 32, 64)}
        idle = idle_fraction(hw, m, ctx, kv_type)
        # At full KV capacity, always-retain decodes only the non-idle agents;
        # ideal (free, overlapped) offload keeps every resident slot decoding.
        retain_batch = max(1, round(max_agents * (1 - idle)))
        headroom = decode_tput(hw, m, max_agents, ctx, kv_type) / decode_tput(hw, m, retain_batch, ctx, kv_type)
        out["profiles"][prof] = {
            "ctx_tokens": ctx,
            "kv_mb_per_agent": kv / 1e6,
            "max_resident_agents": max_agents,
            "ridge_batch": ridge,
            # ridge None: attention KV reads keep decode memory-bound at any batch.
            "memory_binds_first": ridge is None or max_agents < ridge,
            "idle_fraction_b1": idle,
            "ideal_offload_headroom": headroom,
            "decode_tok_s_by_batch": {b: round(t) for b, t in tput.items()},
            "reprefill_s": prefill_s(hw, m, ctx),
            "offload_roundtrip_s": offload_roundtrip_s(hw, kv),
            "tool_median_s": TOOL_MEDIAN_S,
        }
    return out


def fmt_s(x: float | None) -> str:
    return "n/a (unified)" if x is None else f"{x * 1000:.0f} ms"


def print_table(preds: list[dict]) -> None:
    print("| hardware | model | kv | profile | KV/agent | max agents | ridge B | mem binds first | "
          "idle | offload headroom | re-prefill | offload RT |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in preds:
        for prof, r in p["profiles"].items():
            print(f"| {p['hardware']} | {p['model']} | {p['kv_type']} | {prof} | "
                  f"{r['kv_mb_per_agent']:.0f} MB | {r['max_resident_agents']} | {r['ridge_batch']} | "
                  f"{'yes' if r['memory_binds_first'] else 'no'} | {r['idle_fraction_b1']:.0%} | "
                  f"{r['ideal_offload_headroom']:.2f}x | {fmt_s(r['reprefill_s'])} | "
                  f"{fmt_s(r['offload_roundtrip_s'])} |")
    print()
    print("Predicted decode throughput (tok/s) by batch:")
    for p in preds:
        for prof, r in p["profiles"].items():
            print(f"  {p['hardware']:<26} {p['model']:<22} {p['kv_type']:<5} {prof:<12} "
                  f"{r['decode_tok_s_by_batch']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="freeze predictions to predictions/")
    args = ap.parse_args()

    runs = [(A10G, QWEN_7B), (M1_PRO, QWEN_7B), (M1_PRO, QWEN_3B), (M1_PRO, QWEN_1_5B)]
    preds = [predict(hw, m, kv) for hw, m in runs for kv in ("f16", "q8_0")]
    print_table(preds)

    if args.save:
        out_dir = Path(__file__).resolve().parent.parent / "predictions"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.json"
        assumptions = {"hardware": [asdict(h) for h in (A10G, M1_PRO)],
                       "models": [asdict(m) for m in (QWEN_7B, QWEN_3B, QWEN_1_5B)]}
        path.write_text(json.dumps({"assumptions": assumptions, "predictions": preds}, indent=2))
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
