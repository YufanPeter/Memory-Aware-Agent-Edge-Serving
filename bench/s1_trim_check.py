"""S1 (selective KV, step 1: feasibility): does deleting token ranges from a slot's KV in place work?

Needs the patched server (tools/llama_patch/: fork + trim) and stage-1 traces for realistic
prompts (results/p2a_stage1/l0/rep0/tasks.jsonl, react_long with whole paragraphs + 9 neighbours).

For each of N prompts, cut where an agent has seen at least two observations, before its next
Thought:
  1. prefill P on slot 0;
  2. choose the spans to delete: the non-supporting sentences (obs_other) of every observation
     except the latest. The selection rule does not matter here; the mechanism does;
  3. align the spans to tokens (edgeserve/kvtrim.py), re-tokenize the trimmed text P' and check
     it equals the slot's tokens minus the deleted ranges;
  4. trim slot 0, then continue on it with P' (greedy, 48 tokens): output A;
  5. references: P' prefilled from scratch on slot 1 (B) and again on slot 2 (B2: run-to-run
     noise), and the untrimmed P on slot 3 (C).
Reported per prompt: alignment, cache reuse after the trim (cache_n of A vs len(P')), KV cells in
use before/after (LLAMA_KV_CACHE_DEBUG), trim time, and whether A == B (stale-KV effect) next to
B == B2 (noise). A last check forks a golden system prompt into two slots, trims one, and makes
sure the other still reuses the whole prefix with the same output.

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server uv run python bench/s1_trim_check.py [--n 20]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as st
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from x1_speed_tax import URL, Server  # noqa: E402

from edgeserve.kvtrim import observation_spans, plan_verified  # noqa: E402
from edgeserve.llm import LlamaServerClient  # noqa: E402

N_PREDICT = 48
STOP = ["\nObservation"]


def cut_prompts(tasks_path: Path, n: int, min_obs: int = 2) -> list[dict[str, Any]]:
    """Prompts ending at the last "\\nThought k:" with >= min_obs observations before it."""
    out = []
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        segs = [tuple(s) for s in json.loads(line)["trace"]["segments"]]
        n_obs, cut = 0, None
        for i, (kind, text) in enumerate(segs):
            if kind == "control" and text.startswith("\nObservation"):
                n_obs += 1
            elif kind == "control" and text.startswith("\nThought") and n_obs >= min_obs:
                cut = i
        if cut is not None:
            out.append({"task_id": json.loads(line)["task_id"], "segments": segs[: cut + 1]})
        if len(out) == n:
            break
    return out


def common_prefix(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


async def complete(c: httpx.AsyncClient, prompt: str, slot: int, n_predict: int = N_PREDICT) -> dict[str, Any]:
    r = await c.post("/completion", json={"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                                          "seed": 0, "id_slot": slot, "cache_prompt": True, "stop": STOP})
    r.raise_for_status()
    d = r.json()
    return {"text": d["content"], "prompt_n": d["timings"]["prompt_n"], "cache_n": d["timings"]["cache_n"]}


async def erase_all(llm: LlamaServerClient, n_slots: int) -> None:
    for s in range(n_slots):
        await llm.slot_erase(s)


async def one(srv: Server, llm: LlamaServerClient, c: httpx.AsyncClient, item: dict[str, Any]) -> dict[str, Any]:
    segs = item["segments"]
    text = "".join(t for _, t in segs)
    await erase_all(llm, 4)
    await complete(c, text, 0, n_predict=0)                       # 1. prefill P on slot 0
    toks = await llm.tokenize(text)
    plan = await plan_verified(llm.tokenize, text, observation_spans(segs), toks)   # 2-3. choose and align
    row: dict[str, Any] = {"task_id": item["task_id"], "tokens": len(toks), "spans": plan.n_spans,
                           "snapped": plan.n_snapped, "retok_dropped": plan.n_retok_dropped,
                           "skipped": plan.n_skipped, "skipped_start": plan.skipped_start,
                           "skipped_end": plan.skipped_end, "skipped_examples": plan.skipped_examples, "ranges": len(plan.ranges), "trim_tokens": plan.n_tokens}
    if not plan.ranges:
        return row
    off = srv.log_offset()
    tr = await llm.slot_trim(0, plan.ranges)                       # 4. trim, then continue
    row.update({"trim_ok": tr.ok, "trim_error": tr.error, "n_trimmed": tr.n_tokens, "trim_wall_ms": tr.wall_ms,
                "trim_server_ms": tr.server_ms})
    if not tr.ok:
        return row
    a = await complete(c, plan.new_text, 0)
    kv = srv.n_kv_since(off)
    row.update({"A_cache_n": a["cache_n"], "A_prompt_n": a["prompt_n"], "expected_tokens": len(plan.expected_ids),
                "cells_used_after": kv[0][1] if kv else None, "n_kv_after": kv[0][0] if kv else None})
    b = await complete(c, plan.new_text, 1)                        # 5. references
    b2 = await complete(c, plan.new_text, 2)
    cc = await complete(c, text, 3)
    row.update({"A": a["text"], "B": b["text"], "B2": b2["text"], "C": cc["text"],
                "A_eq_B": a["text"] == b["text"], "B_eq_B2": b["text"] == b2["text"], "A_eq_C": a["text"] == cc["text"],
                "A_B_prefix": common_prefix(a["text"], b["text"]) / max(1, len(b["text"])),
                "B_cache_n": b["cache_n"]})
    return row


async def fork_safety(llm: LlamaServerClient, c: httpx.AsyncClient, item: dict[str, Any]) -> dict[str, Any]:
    segs = item["segments"]
    text = "".join(t for _, t in segs)
    prefix = segs[0][1]                                            # the shared system prompt
    await erase_all(llm, 4)
    await complete(c, prefix, 3, n_predict=0)                      # golden slot 3
    n_prefix = len(await llm.tokenize(prefix))
    await llm.slot_fork(0, 3, n_prefix)
    first = await complete(c, text, 0)
    plan = await plan_verified(llm.tokenize, text, observation_spans(segs))
    assert plan.ranges and plan.ranges[0][0] >= n_prefix, "a range would touch the shared prefix"
    await complete(c, text, 0, n_predict=0)                        # slot 0 back to exactly P
    tr = await llm.slot_trim(0, plan.ranges)
    await complete(c, plan.new_text, 0)
    await llm.slot_fork(1, 3, n_prefix)                            # a second fork after the trim
    second = await complete(c, text, 1)
    return {"n_prefix": n_prefix, "trim_ok": tr.ok, "fork1_cache_n": first["cache_n"],
            "fork2_cache_n": second["cache_n"], "same_output": first["text"] == second["text"]}


async def run(args: argparse.Namespace, srv: Server) -> list[dict[str, Any]]:
    items = cut_prompts(Path(args.tasks), args.n)
    llm = LlamaServerClient(URL)
    rows = []
    async with httpx.AsyncClient(base_url=URL, timeout=httpx.Timeout(600, connect=10)) as c:
        for item in items:
            rows.append(await one(srv, llm, c, item))
            r = rows[-1]
            print(f"{r['task_id']}: {r['tokens']} tok, spans {r['spans']} (skipped {r['skipped']}), "
                  f"trim {r.get('n_trimmed')} tok in {r.get('trim_server_ms')} ms, A cache_n {r.get('A_cache_n')}/"
                  f"{r.get('expected_tokens')}, cells {r.get('cells_used_after')}, A==B {r.get('A_eq_B')}, "
                  f"B==B2 {r.get('B_eq_B2')}", flush=True)
        fs = await fork_safety(llm, c, items[0])
    await llm.close()
    return rows + [{"fork_safety": fs}]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fs = rows[-1]["fork_safety"]
    rows = rows[:-1]
    done = [r for r in rows if r.get("trim_ok")]
    pct = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 1)  # noqa: E731
    return {
        "prompts": len(rows),
        "spans_aligned_pct": round(100 * (1 - sum(r["skipped"] for r in rows) / max(1, sum(r["spans"] for r in rows))), 1),
        "skipped_start": sum(r["skipped_start"] for r in rows),
        "skipped_end": sum(r["skipped_end"] for r in rows),
        "spans_snapped": sum(r["snapped"] for r in rows),
        "spans_dropped_by_retokenize": sum(r["retok_dropped"] for r in rows),
        "trimmed_ok": len(done),
        "trimmed_share_of_tokens_pct": round(100 * st.mean(r["n_trimmed"] / r["tokens"] for r in done), 1) if done else None,
        "cache_reused_all_pct": pct([r["A_cache_n"] >= r["expected_tokens"] - 1 for r in done]),
        "cells_freed_pct": pct([r["cells_used_after"] is not None and r["cells_used_after"] <= r["expected_tokens"]
                                for r in done]),
        "trim_server_ms_mean": round(st.mean(r["trim_server_ms"] for r in done), 2) if done else None,
        "A_eq_B_pct": pct([r["A_eq_B"] for r in done]),
        "B_eq_B2_pct": pct([r["B_eq_B2"] for r in done]),
        "A_eq_C_pct": pct([r["A_eq_C"] for r in done]),
        "A_B_prefix_mean": round(st.mean(r["A_B_prefix"] for r in done), 3) if done else None,
        "fork_safety": fs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tasks", default="results/p2a_stage1/l0/rep0/tasks.jsonl")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--out", default="results/s1_trim")
    args = ap.parse_args()
    out = Path(args.out)
    if "LLAMA_SERVER" not in os.environ:
        raise SystemExit("set LLAMA_SERVER to the patched build (fork + trim)")
    with Server(out, args.ctx, 4, "", kv_debug=True) as srv:
        rows = asyncio.run(run(args, srv))
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")
    s = summarize(rows)
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
