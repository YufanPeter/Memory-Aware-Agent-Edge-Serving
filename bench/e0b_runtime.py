"""E0b: runtime behaviour the Phase 2 KV manager depends on.

Same harness as bench/e0_runtime.py (one server per probe, HTTP only); raw responses,
/slots snapshots and server logs go to results/e0b/.

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server MODEL=~/models/<7b>.gguf \
      python bench/e0b_runtime.py [--only idlecost,truncate,busy,fragment,cacheram,perslot]

Probes:
  idlecost  does KV held by idle slots slow other slots' decode and prefill in the unified
            pool? Idle cells allocated below the active ones, then above them, then erased.
  highwater the same with idle cells erased *below* active cells that stay in place: is the
            cost set by the number of occupied cells or by the highest occupied cell?
  truncate  does a request that sends only a slot's shared prefix shrink the slot to that
            prefix ("drop to prefix") and free the rest of its cells?
  busy      save / erase on a slot that is still generating
  fragment  restore and prefill into a pool fragmented by erasing every other slot
  cacheram  --cache-ram: are replaced or purged slot contexts kept in host RAM and reused?
  perslot   save, cross-slot restore and the region limit with --no-kv-unified
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import e0_runtime as e0
from e0_runtime import Server, client, complete, ids, slot_op, slots, stream

e0.OUT = Path("results/e0b")
save = e0.save


def purges(tag: str) -> list[str]:
    """Server log lines where the runtime evicted a slot or touched its host-RAM cache."""
    text = (e0.OUT / f"{tag}_server.log").read_text(encoding="utf-8", errors="replace")
    return [ln.strip() for ln in text.splitlines()
            if re.search(r"purg|prompt cache|cache_ram|cache state|restored|saved", ln, re.I)][:200]


async def decode_round(c: Any, active: list[int], seed: int, n_prompt: int = 1024,
                       n_gen: int = 200) -> dict[str, Any]:
    """Active slots stream n_gen tokens each from fresh prompts; aggregate decode rate."""
    times = await asyncio.gather(*(stream(c, s, ids(n_prompt, seed + s), n_gen) for s in active))
    first = max(t[0] for t in times)          # all streams past prefill
    last = min(t[-1] for t in times)          # none drained yet
    n = sum(sum(1 for x in t if first <= x <= last) for t in times)
    gaps = [b - a for t in times for a, b in zip(t, t[1:]) if a >= first and b <= last]
    return {"tok_per_s": round(n / (last - first), 1) if last > first else None,
            "gap_ms_median": round(statistics.median(gaps) * 1000, 2) if gaps else None}


async def prefill_round(c: Any, slot: int, seed: int, n: int = 2048) -> float | None:
    r = await complete(c, slot, ids(n, seed), 1)
    return r.get("prompt_ms")


async def measure(c: Any, label: str, seed: int, active: list[int], pslot: int) -> dict[str, Any]:
    reps = [await decode_round(c, active, seed + 1000 * i) for i in range(2)]
    pre = [await prefill_round(c, pslot, seed + 50 + 1000 * i) for i in range(2)]
    return {"label": label, "decode": reps,
            "tok_per_s": statistics.median(r["tok_per_s"] for r in reps if r["tok_per_s"]),
            "prefill_2k_ms": statistics.median(p for p in pre if p)}


async def probe_idlecost() -> str:
    """4 active slots decode; 0 / 8 / 16 idle slots hold 3,000 tokens each; then erased.
    Server A fills the idle slots first (their cells sit below the active ones); server B
    loads the active prompts first and keeps them (idle cells above)."""
    active, pslot, idle = [0, 1, 2, 3], 4, list(range(8, 24))
    out: dict[str, Any] = {}
    with Server("idlecost_a", 65536, 32):
        async with client() as c:
            rows = [await measure(c, "no idle", 1, active, pslot)]
            for s in idle[:8]:
                await complete(c, s, ids(3000, 700 + s), 1)
            rows.append(await measure(c, "8 idle x 3k", 2, active, pslot))
            for s in idle[8:]:
                await complete(c, s, ids(3000, 700 + s), 1)
            rows.append(await measure(c, "16 idle x 3k", 3, active, pslot))
            ops = [await slot_op(c, s, "erase") for s in idle]
            rows.append(await measure(c, "idle erased", 4, active, pslot))
            out["a"] = {"rows": rows, "erase_wall_ms": [o["wall_ms"] for o in ops]}
    with Server("idlecost_b", 65536, 32):
        async with client() as c:
            # Same prompt every round: the active slots keep their low cells and only re-decode.
            fixed = lambda: asyncio.gather(*(stream(c, s, ids(1024, 900 + s), 200) for s in active))  # noqa: E731

            async def fixed_round(label: str) -> dict[str, Any]:
                rates = []
                for _ in range(2):
                    times = await fixed()
                    first, last = max(t[0] for t in times), min(t[-1] for t in times)
                    n = sum(sum(1 for x in t if first <= x <= last) for t in times)
                    rates.append(round(n / (last - first), 1))
                return {"label": label, "tok_per_s": statistics.median(rates), "reps": rates}

            rows = [await fixed_round("no idle")]
            for s in idle:
                await complete(c, s, ids(3000, 700 + s), 1)
            rows.append(await fixed_round("16 idle x 3k above"))
            for s in idle:
                await slot_op(c, s, "erase")
            rows.append(await fixed_round("idle erased"))
            out["b"] = {"rows": rows}
    save("idlecost", out)
    fmt = lambda rows: ", ".join(f"{r['label']}: {r['tok_per_s']} tok/s" + (  # noqa: E731
        f" / prefill {r['prefill_2k_ms']:.0f} ms" if "prefill_2k_ms" in r else "") for r in rows)
    return f"A (idle below): {fmt(out['a']['rows'])} | B (idle above): {fmt(out['b']['rows'])}"


async def probe_highwater() -> str:
    """Is the cost set by the number of occupied cells or by the highest occupied cell?
    16 idle slots × 3,000 tokens are loaded first (low cells), then 4 active slots keep the
    same prompts (high cells) while the idle slots are erased underneath them."""
    active, idle = [0, 1, 2, 3], list(range(8, 24))
    with Server("highwater", 65536, 32):
        async with client() as c:
            async def fixed_round(label: str) -> dict[str, Any]:
                rates = []
                for _ in range(2):
                    times = await asyncio.gather(*(stream(c, s, ids(1024, 900 + s), 200) for s in active))
                    first, last = max(t[0] for t in times), min(t[-1] for t in times)
                    n = sum(sum(1 for x in t if first <= x <= last) for t in times)
                    rates.append(round(n / (last - first), 1))
                return {"label": label, "tok_per_s": statistics.median(rates), "reps": rates}

            rows = []
            for s in idle:
                await complete(c, s, ids(3000, 700 + s), 1)
            rows.append(await fixed_round("16 idle x 3k below"))
            for s in idle:
                await slot_op(c, s, "erase")
            rows.append(await fixed_round("idle erased, active still high"))
            for s in active:
                await slot_op(c, s, "erase")
            rows.append(await fixed_round("active re-prefilled after erase"))
    save("highwater", {"rows": rows})
    return ", ".join(f"{r['label']}: {r['tok_per_s']} tok/s" for r in rows)


async def probe_truncate() -> str:
    """-c 8192: slot 0 holds prefix (1,400) + 4,000 private tokens, then is sent the prefix
    alone. If that frees the private cells, a 5,500-token prompt on slot 1 fits without a
    purge and slot 0 still reuses its prefix afterwards."""
    prefix, private = ids(1400, 1), ids(4000, 2)
    res: dict[str, Any] = {}
    with Server("truncate", 8192, 4):
        async with client() as c:
            res["fill"] = await complete(c, 0, prefix + private, 1)
            t0 = time.monotonic()
            res["truncate_n0"] = await complete(c, 0, prefix, 0)
            res["truncate_n0"]["client_ms"] = round((time.monotonic() - t0) * 1000, 1)
            res["other_slot_5500"] = await complete(c, 1, ids(5500, 3), 1)
            res["slot0_revisit"] = await complete(c, 0, prefix + private, 1)
            # Same sequence with n_predict=1, in case n_predict=0 is special-cased.
            await slot_op(c, 1, "erase")
            res["truncate_n1"] = await complete(c, 0, prefix, 1)
            res["other_slot_5500_b"] = await complete(c, 1, ids(5500, 4), 1)
            res["slot0_revisit_b"] = await complete(c, 0, prefix + private, 1)
            res["slots"] = await slots(c)
    res["log"] = purges("truncate")
    save("truncate", res)
    return (f"n_predict=0: HTTP {res['truncate_n0']['status']} prompt_n {res['truncate_n0'].get('prompt_n')} "
            f"cache_n {res['truncate_n0'].get('cache_n')} predicted {res['truncate_n0'].get('predicted_n')} "
            f"({res['truncate_n0']['client_ms']} ms); slot0 revisit cache_n {res['slot0_revisit'].get('cache_n')} "
            f"(1400 = truncated and kept); with n_predict=1: revisit cache_n {res['slot0_revisit_b'].get('cache_n')}; "
            f"purge lines {len([x for x in res['log'] if 'purg' in x])}")


async def probe_busy() -> str:
    """Save and erase slot 0 while it streams 400 tokens."""
    with Server("busy", 32768, 8):
        async with client() as c, client() as ctl:
            async def ops() -> list[dict[str, Any]]:
                await asyncio.sleep(1.0)
                return [await slot_op(ctl, 0, "save", "busy.bin"), await slot_op(ctl, 0, "erase")]

            times, log = await asyncio.gather(stream(c, 0, ids(1000, 11), 400), ops())
            after = await complete(c, 0, ids(1000, 11), 1)
    res = {"stream_tokens": len(times), "ops": [{k: o[k] for k in ("op", "status", "wall_ms", "resp")} for o in log],
           "after": after}
    save("busy", res)
    return (f"stream got {len(times)}/400 tokens; " +
            "; ".join(f"{o['op']} HTTP {o['status']} {str(o['resp'])[:100]}" for o in res["ops"]) +
            f"; next request cache_n {after.get('cache_n')}")


async def probe_fragment() -> str:
    """-c 16384: 8 slots grow in two interleaved 950-token chunks (15,200 cells), slots
    0/2/4/6 are erased (free space in 950-token holes, largest ~1.2k), then a saved
    3,000-token image is restored and a fresh 3,000-token prompt is prefilled."""
    big = ids(3000, 77)
    res: dict[str, Any] = {}
    with Server("fragment", 16384, 16):
        async with client() as c:
            await complete(c, 15, big, 1)
            res["save"] = await slot_op(c, 15, "save", "frag.bin")
            res["erase15"] = await slot_op(c, 15, "erase")
            bases = {i: ids(1900, 300 + i) for i in range(8)}
            for r in (1, 2):
                for i in range(8):
                    await complete(c, i, bases[i][: 950 * r], 1)
            res["erase_even"] = [await slot_op(c, i, "erase") for i in (0, 2, 4, 6)]
            res["restore"] = await slot_op(c, 15, "restore", "frag.bin")
            res["restore_check"] = await complete(c, 15, big, 1)
            res["prefill_3k"] = await complete(c, 14, ids(3000, 78), 1)
            res["odd_slots_kept"] = [await complete(c, i, bases[i], 1) for i in (1, 3, 5, 7)]
    res["log"] = purges("fragment")
    for k in ("save", "erase15", "restore"):
        res[k] = {kk: res[k][kk] for kk in ("op", "status", "wall_ms", "resp")}
    res["erase_even"] = [{kk: o[kk] for kk in ("status", "resp")} for o in res["erase_even"]]
    save("fragment", res)
    return (f"restore 3k into holes: HTTP {res['restore']['status']} {str(res['restore']['resp'])[:120]}; "
            f"check cache_n {res['restore_check'].get('cache_n')}; fresh 3k prefill HTTP "
            f"{res['prefill_3k']['status']}; odd slots cache_n {[r.get('cache_n') for r in res['odd_slots_kept']]}; "
            f"purges {len([x for x in res['log'] if 'purg' in x])}")


async def probe_cacheram() -> str:
    """--cache-ram 4096 MiB, -c 8192, -np 4: (1) slot 0 alternates prompts A, B, A;
    (2) the E0a pressure pattern (4 × 2,500 pinned) with revisits; (3) unpinned A, B, C, A."""
    A, B, C = ids(3000, 21), ids(3000, 22), ids(3000, 23)
    res: dict[str, Any] = {}
    os.environ["CACHE_RAM"] = "4096"   # start_server.sh reads it; everything else as in E0a
    try:
        with Server("cacheram", 8192, 4):
            async with client() as c:
                res["alternate"] = [await complete(c, 0, p, 1) for p in (A, B, A)]
                for i in range(4):
                    await slot_op(c, i, "erase")
                res["pressure_first"] = [await complete(c, i, ids(2500, 40 + i), 1) for i in range(4)]
                res["pressure_revisit"] = [await complete(c, i, ids(2500, 40 + i), 1) for i in range(4)]
                for i in range(4):
                    await slot_op(c, i, "erase")
                res["unpinned"] = [await complete(c, -1, p, 1) for p in (A, B, C, A)]
    finally:
        os.environ.pop("CACHE_RAM", None)
    res["log"] = purges("cacheram")
    save("cacheram", res)
    cn = lambda rs: [r.get("cache_n") for r in rs]  # noqa: E731
    return (f"alternate A,B,A cache_n {cn(res['alternate'])} prompt_ms "
            f"{[r.get('prompt_ms') for r in res['alternate']]}; pressure revisit cache_n "
            f"{cn(res['pressure_revisit'])}; unpinned A,B,C,A cache_n {cn(res['unpinned'])}; "
            f"log lines {len(res['log'])}")


async def probe_perslot() -> str:
    """--no-kv-unified, -c 16384, -np 4 (4,096-token regions): save slot 0 (3,000 tokens),
    erase, restore into slot 2; and a 5,000-token prompt (larger than a region)."""
    big = ids(3000, 31)
    res: dict[str, Any] = {}
    with Server("perslot", 16384, 4, "--no-kv-unified"):
        async with client() as c:
            res["fill"] = await complete(c, 0, big, 1)
            res["save"] = await slot_op(c, 0, "save", "perslot.bin")
            res["erase"] = await slot_op(c, 0, "erase")
            res["restore_into_2"] = await slot_op(c, 2, "restore", "perslot.bin")
            res["check_2"] = await complete(c, 2, big, 1)
            res["too_big"] = await complete(c, 1, ids(5000, 32), 1)
            res["props"] = (await c.get("/props")).json().get("default_generation_settings", {}).get("n_ctx")
    for k in ("save", "erase", "restore_into_2"):
        res[k] = {kk: res[k][kk] for kk in ("op", "status", "wall_ms", "resp")}
    save("perslot", res)
    return (f"n_ctx per slot {res['props']}; save {res['save']['wall_ms']} ms; cross-slot restore HTTP "
            f"{res['restore_into_2']['status']} ({res['restore_into_2']['wall_ms']} ms), check cache_n "
            f"{res['check_2'].get('cache_n')}; 5k prompt into a 4k region: HTTP {res['too_big']['status']} "
            f"{(res['too_big'].get('error') or '')[:100]}")


PROBES = {"idlecost": probe_idlecost, "highwater": probe_highwater, "truncate": probe_truncate, "busy": probe_busy,
          "fragment": probe_fragment, "cacheram": probe_cacheram, "perslot": probe_perslot}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help=f"comma-separated subset of {','.join(PROBES)}")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(PROBES)
    for name in names:
        try:
            print(f"[{name}] {asyncio.run(PROBES[name]())}", flush=True)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
