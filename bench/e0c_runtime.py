"""E0c: the open runtime questions of docs/phase2_spec.md §9, each with a design branch.

Same harness as bench/e0_runtime.py and bench/e0b_runtime.py (one server per probe, HTTP only);
raw responses, /slots snapshots and server logs go to results/e0c/.

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server MODEL=~/models/<7b>.gguf \
      uv run python bench/e0c_runtime.py [--only fragsave,tokenize,...] [--dry]

Probes (prediction -> branch, spec §9 E0c):
  fragsave    save / restore of a fragmented react_long-like slot built under 8 concurrent
              decoders vs a contiguous one, and the re-prefill cost of the same tokens.
              Predicted ~1.5-2x slower saves; if κ_s + κ_r >= t_pf per token, drop-only ships.
  tokenize    /tokenize length == prompt_n + cache_n for the real dev prompts the agent builds
              (every step of 200 dev tasks over the four serving profiles). Exact -> margin 128;
              else margin = 4 x max diff and tok_check becomes a warning.
  dropids     drop to prefix with token-id prompts (the family LCP ids built as the manager
              does). Revisit cache_n == L_f -> as specified; else send the prefix as text.
  opafter     a save / erase issued the moment a stream's final message arrives: deferred or
              not (wall ≈ server time)? If deferred, ops wait for /slots is_processing false.
  disconnect  client closes a stream mid-generation: does the slot stop and become idle, and
              does erase work? If not, release() polls /slots until idle.
  kvdebug     decode / prefill cost of LLAMA_KV_CACHE_DEBUG=1 with -lv 4 (validation runs only).
  b2pressure  B2 under pool pressure: 6 unpinned agents sharing a 1,400-token prefix in an 8k
              pool, --cache-ram 448 (h·c), -lv 4 to count the TRC prompt-cache lines, plus a
              --cache-ram 0 control. Predicted: revisits hit the host cache by prefix
              (f_keep >= 0.25, server-task.cpp:1838), cache_n ≈ the agent's previous context.
  reconrace   a /slots poll sent while other slots decode, right after an erase of an idle
              slot: can the snapshot predate the erase (sc:3001-3005)? The version rule (§3) is
              kept either way.

`--dry` builds the dev prompts and family prefixes offline (a stand-in tokenizer, no server) and
checks the helper logic, so the script can be validated on a laptop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: edgeserve.*

import httpx  # noqa: E402

import e0_runtime as e0  # noqa: E402
from e0_runtime import Server, client, ids, slot_op, slots, stream  # noqa: E402

e0.OUT = Path("results/e0c")
save = e0.save
Tokenize = Callable[[str], Awaitable[list[int]]]

# --- helpers (pure where possible; checked by --dry and tests/test_analysis.py) ---------

TRC_PATTERNS = {
    "idle_slot_saved": "saving idle slot to prompt cache",
    "cache_update": "updating prompt cache",
    "prompt_saved": "saving prompt with length",
    "cache_lookup": "looking for better prompt",
    "cache_hit": "found better prompt",
    "cache_room": "making room for prompt cache entry",
    "cache_size_limit": "cache size limit reached",
    "cache_token_limit": "cache token limit",
    "restore_failed": "failed to restore state",
    "purge": "purging slot",
    "overflow": "Context size has been exceeded",
    "slot_by_lcp": "selected slot by lcp similarity",
    "slot_by_lru": "selected slot by LRU",
}


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def trc_counts(text: str) -> dict[str, int]:
    """Prompt-cache / purge line counts in a server log (TRC lines need -lv 4)."""
    lines = text.splitlines()
    return {k: sum(1 for ln in lines if pat in ln) for k, pat in TRC_PATTERNS.items()}


def first_react_prompt(question: str, explain: bool, few_shot: int = 6) -> str:
    """The agent's step-1 ReAct prompt (agent.py _run_react), rendered as text."""
    from edgeserve.prompt import CHATML, react_system
    return (CHATML["system"].format(content=react_system(few_shot, explain)) + CHATML["user_open"] + "Question: "
            + question + CHATML["user_close_assistant_open"] + "Thought 1:")


# Their first words differ from the first character on, so the LCP cannot run into a question
# token that a real task's question would not share (BPE merges " Which" / " Who" as words).
DUMMY_QUESTIONS = ("Which city is the capital of the country where the Eiffel Tower stands?",
                   "Are the novels Dune and Emma by authors of the same nationality?")


async def family_prefix(tokenize: Tokenize, explain: bool) -> list[int]:
    """LCP of the token ids of two first prompts of a family (spec §4.1): a token boundary
    every task of the family shares, whatever its question."""
    a = await tokenize(first_react_prompt(DUMMY_QUESTIONS[0], explain))
    b = await tokenize(first_react_prompt(DUMMY_QUESTIONS[1], explain))
    return a[: lcp(a, b)]


def equality_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diffs = [r["tok"] - (r["prompt_n"] + r["cache_n"]) for r in rows if r.get("prompt_n") is not None]
    return {"n": len(diffs), "equal": sum(1 for d in diffs if d == 0),
            "max_abs_diff": max((abs(d) for d in diffs), default=None),
            "diffs_nonzero": sorted({d for d in diffs if d})[:20]}


class RecordingLLM:
    """Stands in for the server while the real Agent runs, so the prompts are exactly the ones
    agent.py builds (real tools, real observations). Each task searches its two supporting-fact
    titles, then finishes; react_explain then requests its Explanation."""

    def __init__(self) -> None:
        self.prompts: list[dict[str, Any]] = []
        self.titles: dict[int, list[str]] = {}

    async def generate(self, prompt: str, n_predict: int, stop: list[str], slot: int = -1,
                       ignore_eos: bool = False, **_: Any) -> Any:
        from edgeserve.llm import LLMResult
        self.prompts.append({"slot": slot, "prompt": prompt})
        titles = self.titles.get(slot, [])
        n_obs = prompt.rsplit("Question: ", 1)[-1].count("\nObservation ")
        if prompt.endswith("Explanation:"):
            text = " The two facts above give the answer."
        elif prompt.endswith("finish["):
            text = "unknown]"
        elif "Context:\n" in prompt or n_obs >= len(titles):
            text = f" I have enough information.\nAction {n_obs + 1}: finish[{(titles or ['unknown'])[0]}]"
        else:
            text = f" I need to search {titles[n_obs]}.\nAction {n_obs + 1}: search[{titles[n_obs]}]"
        t = time.monotonic()
        return LLMResult(ok=True, text=text, prompt_n=len(prompt) // 4, predicted_n=len(text) // 4, slot=slot,
                         t_start=t, t_first_token=t, t_end=t)


async def dev_prompts(n_tasks: int = 200) -> list[dict[str, Any]]:
    """Every prompt of n_tasks dev questions, profiles round-robin (react_short, react_long,
    react_explain, oneshot_long), built by the real agent with zero tool delay."""
    from edgeserve.agent import Agent
    from edgeserve.corpus import Corpus
    from edgeserve.data import load_split
    from edgeserve.events import EventLog
    from edgeserve.profiles import load_profiles
    profiles = load_profiles()
    names = ["react_short", "react_long", "react_explain", "oneshot_long"]
    corpus = Corpus.load(with_bm25=True)
    llm = RecordingLLM()
    agent = Agent(llm, corpus, EventLog(None), seed=0, delay_scale=0.0)
    out: list[dict[str, Any]] = []
    for i, item in enumerate(load_split("dev")[:n_tasks]):
        prof = profiles[names[i % len(names)]]
        llm.titles[i] = list(dict.fromkeys(t for t, _ in item.supporting_facts))[:2]
        start = len(llm.prompts)
        await agent.run(f"e0c{i:04d}", item, prof, slot=i)
        for step, rec in enumerate(llm.prompts[start:], 1):
            out.append({"task": i, "profile": prof.name, "step": step, "prompt": rec["prompt"]})
    return out


async def complete_tok(c: httpx.AsyncClient, slot: int, prompt: str | list[int], n_predict: int = 1,
                       ignore_eos: bool = False) -> dict[str, Any]:
    """Non-streamed completion that also returns the generated token ids and id_slot."""
    body: dict[str, Any] = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "temperature": 0.0,
                            "ignore_eos": ignore_eos, "return_tokens": True}
    if slot >= 0:
        body["id_slot"] = slot
    t0 = time.monotonic()
    r = await c.post("/completion", json=body)
    out: dict[str, Any] = {"slot": slot, "status": r.status_code, "wall_s": round(time.monotonic() - t0, 3)}
    try:
        d = r.json()
    except ValueError:
        return out | {"error": r.text[:300]}
    if r.status_code != 200:
        return out | {"error": json.dumps(d)[:300]}
    t = d.get("timings", {})
    return out | {k: t.get(k) for k in ("prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms")} | {
        "tokens": d.get("tokens") or [], "id_slot": d.get("id_slot")}


async def tokenize(c: httpx.AsyncClient, text: str) -> list[int]:
    r = await c.post("/tokenize", json={"content": text, "add_special": True, "parse_special": True})
    r.raise_for_status()
    return [int(t) for t in r.json()["tokens"]]


def slot_tokens(snap: Any, slot: int) -> int | None:
    for s in snap if isinstance(snap, list) else []:
        if s.get("id") == slot:
            return s.get("n_prompt_tokens")
    return None


def slot_busy(snap: Any, slot: int) -> bool | None:
    for s in snap if isinstance(snap, list) else []:
        if s.get("id") == slot:
            return bool(s.get("is_processing"))
    return None


def server_log(tag: str) -> str:
    return (e0.OUT / f"{tag}_server.log").read_text(encoding="utf-8", errors="replace")


def brief(op: dict[str, Any]) -> dict[str, Any]:
    return {k: op.get(k) for k in ("op", "status", "wall_ms", "resp")}


def per_tok_us(ms: float | None, n: int | None) -> float | None:
    return round(1000 * ms / n, 2) if ms is not None and n else None


# --- probes -----------------------------------------------------------------------------

async def probe_fragsave() -> str:
    """-c 32768, -np 16. Contiguous: slot 15 prefilled with 3,800 tokens alone. Fragmented:
    slot 0 grows in 6 agent-like steps (prompt + own output + 350 new tokens), each step
    concurrent with 8 slots streaming 200 tokens, so its prefill chunks and decode cells
    interleave with theirs. Both images: save, erase, restore, check; then the fragmented
    tokens are re-prefilled from scratch (t_pf, the drop cost)."""
    res: dict[str, Any] = {}
    with Server("fragsave", 32768, 16):
        async with client() as c, client() as ctl:
            contig = ids(3800, 501)
            await complete_tok(c, 15, contig, 1)
            res["contig"] = {"save": await slot_op(ctl, 15, "save", "contig.bin"),
                             "save2": await slot_op(ctl, 15, "save", "contig2.bin")}
            res["contig"]["erase"] = await slot_op(ctl, 15, "erase")
            res["contig"]["restore"] = await slot_op(ctl, 15, "restore", "contig.bin")
            res["contig"]["check"] = await complete_tok(c, 15, contig, 1)
            await slot_op(ctl, 15, "erase")

            prompt = ids(1400, 1) + ids(100, 2)
            steps = []
            for j in range(6):
                r, *_ = await asyncio.gather(complete_tok(c, 0, prompt, 40, True),
                                             *(stream(c, s, ids(300, 600 + 10 * j + s), 200) for s in range(1, 9)))
                steps.append({k: r.get(k) for k in ("prompt_n", "cache_n", "predicted_n", "prompt_ms")})
                prompt = prompt + list(r.get("tokens") or []) + ids(350, 700 + j)
            snap = await slots(c)
            n_frag = slot_tokens(snap, 0)
            fr: dict[str, Any] = {"steps": steps, "n_tokens": n_frag}
            fr["save"] = await slot_op(ctl, 0, "save", "frag.bin")
            fr["save2"] = await slot_op(ctl, 0, "save", "frag2.bin")
            fr["erase"] = await slot_op(ctl, 0, "erase")
            fr["restore"] = await slot_op(ctl, 0, "restore", "frag.bin")
            full = prompt[: n_frag] if n_frag else prompt
            fr["check"] = await complete_tok(c, 0, full, 1)
            await slot_op(ctl, 0, "erase")
            fr["reprefill"] = await complete_tok(c, 0, full, 1)
            res["frag"] = fr

    def costs(d: dict[str, Any], n: int | None) -> dict[str, Any]:
        sv = [d[k]["resp"].get("timings", {}).get("save_ms") for k in ("save", "save2") if d[k]["status"] == 200]
        rs = d["restore"]["resp"].get("timings", {}).get("restore_ms") if d["restore"]["status"] == 200 else None
        return {"n": n, "save_us_per_tok": per_tok_us(statistics.median(sv), n) if sv else None,
                "restore_us_per_tok": per_tok_us(rs, n), "check_cache_n": d["check"].get("cache_n")}

    n_contig = res["contig"]["save"]["resp"].get("n_saved")
    out = {"contig": costs(res["contig"], n_contig), "frag": costs(res["frag"], res["frag"]["n_tokens"])}
    rp = res["frag"]["reprefill"]
    out["frag"]["reprefill_us_per_tok"] = per_tok_us(rp.get("prompt_ms"), rp.get("prompt_n"))
    f = out["frag"]
    if f["save_us_per_tok"] and f["restore_us_per_tok"] and f["reprefill_us_per_tok"]:
        k = f["save_us_per_tok"] + f["restore_us_per_tok"]
        out["offload_beats_drop"] = k < f["reprefill_us_per_tok"]
        out["kappa_sum_us"] = round(k, 2)
    for side in ("contig", "frag"):
        for op in ("save", "save2", "erase", "restore"):
            res[side][op] = brief(res[side][op])
    save("fragsave", {"summary": out, "raw": res})
    return json.dumps(out)


async def probe_tokenize() -> str:
    """Every prompt of 200 dev tasks: /tokenize vs prompt_n + cache_n of the same prompt sent as
    text on a pinned slot (steps of a task in order, so the prefix is cached as in a run)."""
    prompts = await dev_prompts(200)
    rows: list[dict[str, Any]] = []
    local = None
    try:   # the offline tokenizer, only for the equality check (spec §3)
        from tokenizers import Tokenizer
        local = Tokenizer.from_file("models/qwen2.5-tokenizer/tokenizer.json")
    except Exception:
        pass
    with Server("tokenize", 65536, 32):
        async with client() as c:
            for p in prompts:
                tok = await tokenize(c, p["prompt"])
                r = await complete_tok(c, p["task"] % 32, p["prompt"], 1)
                row = {k: p[k] for k in ("task", "profile", "step")} | {
                    "tok": len(tok), "prompt_n": r.get("prompt_n"), "cache_n": r.get("cache_n"),
                    "status": r["status"], "chars": len(p["prompt"])}
                if local is not None:
                    row["local"] = len(local.encode(p["prompt"], add_special_tokens=True).ids)
                rows.append(row)
    stats = equality_stats(rows)
    by_profile = {prof: equality_stats([r for r in rows if r["profile"] == prof])
                  for prof in sorted({r["profile"] for r in rows})}
    if local is not None:
        stats["local_equal"] = sum(1 for r in rows if r.get("local") == r["tok"])
    save("tokenize", {"summary": stats, "by_profile": by_profile, "rows": rows})
    margin = 128 if stats["max_abs_diff"] == 0 else 4 * (stats["max_abs_diff"] or 0)
    return f"{stats['equal']}/{stats['n']} prompts exact, max |diff| {stats['max_abs_diff']} -> margin m = {margin}"


async def probe_dropids() -> str:
    """-c 8192: a 5.5k react6 prompt (real prefix + filler) on slot 0, then the family prefix
    ids alone with n_predict 0, then a revisit with the full prompt as text. Expected: drop keeps
    L_f cells, the revisit reuses L_f (cache_n == L_f), a 5k prompt fits on slot 1 meanwhile."""
    res: dict[str, Any] = {}
    with Server("dropids", 8192, 4):
        async with client() as c:
            pref = {"react6": await family_prefix(lambda t: tokenize(c, t), False),
                    "react6x": await family_prefix(lambda t: tokenize(c, t), True)}
            res["L_f"] = {k: len(v) for k, v in pref.items()}
            # Real dev questions: every first prompt must start with the family prefix ids.
            from edgeserve.data import load_split
            qs = [it.question for it in load_split("dev")[:100]]
            ok = [(await tokenize(c, first_react_prompt(q, False)))[: len(pref["react6"])] == pref["react6"]
                  for q in qs]
            res["dev_checked"], res["dev_start_with_prefix"] = len(ok), sum(ok)
            text = first_react_prompt("Which band released the album that the song Hey Jude closed?", False)
            text += " I need to search the album." + " Observation filler sentence." * 700
            full = await tokenize(c, text)
            res["full_tokens"] = len(full)
            res["fill"] = await complete_tok(c, 0, text, 1)
            t0 = time.monotonic()
            res["drop"] = await complete_tok(c, 0, pref["react6"], 0)
            res["drop"]["client_ms"] = round((time.monotonic() - t0) * 1000, 1)
            res["slots_after_drop"] = slot_tokens(await slots(c), 0)
            res["other_5k"] = await complete_tok(c, 1, ids(5000, 9), 1)
            res["revisit"] = await complete_tok(c, 0, text, 1)
    for k in ("fill", "drop", "other_5k", "revisit"):
        res[k].pop("tokens", None)
    res["log"] = trc_counts(server_log("dropids"))
    save("dropids", res)
    lf = res["L_f"]["react6"]
    return (f"L_f react6 {lf}, react6x {res['L_f']['react6x']}; drop HTTP {res['drop']['status']} prompt_n "
            f"{res['drop'].get('prompt_n')} cache_n {res['drop'].get('cache_n')} ({res['drop']['client_ms']} ms), "
            f"slot 0 holds {res['slots_after_drop']}; 5k on slot 1 HTTP {res['other_5k']['status']}; revisit "
            f"cache_n {res['revisit'].get('cache_n')} ({'==' if res['revisit'].get('cache_n') == lf else '!='} L_f); "
            f"{res['dev_start_with_prefix']}/{res['dev_checked']} dev first prompts start with the prefix ids; "
            f"purges {res['log']['purge']}")


async def stream_then(c: httpx.AsyncClient, slot: int, prompt: list[int], n_predict: int,
                      on_stop: Callable[[], Awaitable[dict[str, Any]]]) -> tuple[int, dict[str, Any] | None]:
    """Stream a completion; the moment the final message is parsed, run on_stop (before the
    stream is closed). Returns (tokens received, on_stop's result)."""
    body = {"prompt": prompt, "n_predict": n_predict, "id_slot": slot, "cache_prompt": True, "temperature": 0.0,
            "ignore_eos": True, "stream": True}
    n, out = 0, None
    async with c.stream("POST", "/completion", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            msg = json.loads(line[6:])
            n += bool(msg.get("content"))
            if msg.get("stop"):
                out = await on_stop()
    return n, out


async def probe_opafter() -> str:
    """-c 32768: slot 0 streams 300 tokens; save (odd trials) or erase (even trials) is sent the
    moment the final message arrives, on a separate connection. Deferred if wall >> server ms."""
    trials = []
    with Server("opafter", 32768, 8):
        async with client() as c, client() as ctl:
            for i in range(6):
                op = "save" if i % 2 == 0 else "erase"
                fn = f"opafter{i}.bin" if op == "save" else None
                n, o = await stream_then(c, 0, ids(1500, 40 + i), 300,
                                         lambda op=op, fn=fn: slot_op(ctl, 0, op, fn))
                assert o is not None
                server_ms = o["resp"].get("timings", {}).get("save_ms") if op == "save" else None
                trials.append({"op": op, "tokens": n, "status": o["status"], "wall_ms": o["wall_ms"],
                               "server_ms": server_ms, "resp": o["resp"]})
                await slot_op(ctl, 0, "erase")
    save("opafter", {"trials": trials})
    return "; ".join(f"{t['op']} wall {t['wall_ms']} ms" + (f" (server {t['server_ms']})" if t["server_ms"] else "")
                     for t in trials)


async def probe_disconnect() -> str:
    """-c 32768: slot 0 streams up to 2,000 tokens; the client closes the stream after 50.
    /slots is then polled every 50 ms until slot 0 is idle; erase and a new request follow."""
    res: dict[str, Any] = {}
    with Server("disconnect", 32768, 8):
        async with client() as c, client() as ctl:
            body = {"prompt": ids(1000, 51), "n_predict": 2000, "id_slot": 0, "cache_prompt": True,
                    "temperature": 0.0, "ignore_eos": True, "stream": True}
            n = 0
            async with c.stream("POST", "/completion", json=body) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and json.loads(line[6:]).get("content"):
                        n += 1
                        if n >= 50:
                            break
            t_close = time.monotonic()
            polls = []
            while time.monotonic() - t_close < 10:
                snap = await slots(ctl)
                polls.append({"t_ms": round((time.monotonic() - t_close) * 1000, 1), "busy": slot_busy(snap, 0),
                              "n_tokens": slot_tokens(snap, 0)})
                if polls[-1]["busy"] is False:
                    break
                await asyncio.sleep(0.05)
            res["received"] = n
            res["polls"] = polls
            res["idle_after_ms"] = polls[-1]["t_ms"] if polls and polls[-1]["busy"] is False else None
            res["erase"] = brief(await slot_op(ctl, 0, "erase"))
            after = await complete_tok(c, 0, ids(1000, 51), 1)
            after.pop("tokens", None)
            res["after"] = after
    save("disconnect", res)
    return (f"closed after {res['received']} tokens; slot idle after {res['idle_after_ms']} ms "
            f"(holding {res['polls'][-1]['n_tokens'] if res['polls'] else None}); erase HTTP {res['erase']['status']} "
            f"{res['erase']['resp']}; next request HTTP {res['after']['status']} cache_n {res['after'].get('cache_n')}")


async def decode_round(c: httpx.AsyncClient, active: list[int], seed: int, n_prompt: int = 1024,
                       n_gen: int = 200) -> float | None:
    """Aggregate decode tok/s of concurrent streams while all are past prefill (as in E0b)."""
    times = await asyncio.gather(*(stream(c, s, ids(n_prompt, seed + s), n_gen) for s in active))
    first, last = max(t[0] for t in times), min(t[-1] for t in times)
    n = sum(sum(1 for x in t if first <= x <= last) for t in times)
    return round(n / (last - first), 1) if last > first else None


async def probe_kvdebug() -> str:
    """Same server flags with and without LLAMA_KV_CACHE_DEBUG=1 and -lv 4: 4-stream decode rate
    and a 2k prefill, twice each."""
    out: dict[str, Any] = {}
    for tag, dbg in (("kvdebug_off", False), ("kvdebug_on", True)):
        if dbg:
            os.environ["LLAMA_KV_CACHE_DEBUG"] = "1"
        try:
            with Server(tag, 32768, 16, "-lv 4" if dbg else ""):
                async with client() as c:
                    dec = [await decode_round(c, [0, 1, 2, 3], 100 * i) for i in range(2)]
                    pre = [(await complete_tok(c, 8 + i, ids(2048, 900 + i), 1)).get("prompt_ms") for i in range(2)]
        finally:
            os.environ.pop("LLAMA_KV_CACHE_DEBUG", None)
        log = server_log(tag)
        out[tag] = {"decode_tok_per_s": dec, "prefill_2k_ms": pre, "log_lines": log.count("\n"),
                    "find_slot_lines": log.count("find_slot:")}
    save("kvdebug", out)
    a, b = out["kvdebug_off"], out["kvdebug_on"]
    med = lambda xs: statistics.median(x for x in xs if x) if any(xs) else None  # noqa: E731
    return (f"decode {med(a['decode_tok_per_s'])} -> {med(b['decode_tok_per_s'])} tok/s, prefill 2k "
            f"{med(a['prefill_2k_ms'])} -> {med(b['prefill_2k_ms'])} ms, find_slot lines {b['find_slot_lines']}, "
            f"log {a['log_lines']} -> {b['log_lines']} lines")


async def b2_agents(c: httpx.AsyncClient, n_agents: int = 6, n_steps: int = 5,
                    seed: int = 0) -> list[dict[str, Any]]:
    """Agent-like unpinned traffic: a shared 1,400-token prefix, a private question, then per
    step 30 generated tokens and a 400-token observation, with seeded tool delays (lognormal,
    median 0.8 s, sigma 0.6, as the profiles' default)."""
    prefix = ids(1400, 1)
    rows: list[dict[str, Any]] = []

    async def agent(a: int) -> None:
        rng = random.Random(seed * 1000 + a)
        prompt = prefix + ids(50, 100 + a)
        prev_ctx = len(prefix)   # what a revisit could reuse at best
        for j in range(n_steps):
            r = await complete_tok(c, -1, prompt, 30, True)
            gen = list(r.get("tokens") or [])
            rows.append({"agent": a, "step": j + 1, "status": r["status"], "id_slot": r.get("id_slot"),
                         "prompt_tokens": len(prompt), "reusable": prev_ctx, "cache_n": r.get("cache_n"),
                         "prompt_n": r.get("prompt_n"), "prompt_ms": r.get("prompt_ms"),
                         "hit": round((r.get("cache_n") or 0) / prev_ctx, 4) if prev_ctx else None})
            if r["status"] != 200:
                return
            prev_ctx = len(prompt) + len(gen)
            prompt = prompt + gen + ids(400, 1000 + 10 * a + j)
            await asyncio.sleep(min(5.0, 0.8 * rng.lognormvariate(0.0, 0.6)))

    await asyncio.gather(*(agent(a) for a in range(n_agents)))
    return rows


async def probe_b2pressure() -> str:
    """-c 8192, -np 6, unpinned, -lv 4: --cache-ram 448 (B2 at h·c) and --cache-ram 0 (control)."""
    out: dict[str, Any] = {}
    for tag, cr in (("b2pressure", "448"), ("b2pressure_cr0", "0")):
        os.environ["CACHE_RAM"] = cr
        try:
            with Server(tag, 8192, 6, "-lv 4"):
                async with client() as c:
                    rows = await b2_agents(c)
        finally:
            os.environ.pop("CACHE_RAM", None)
        revisits = [r for r in rows if r["step"] > 1 and r["status"] == 200]
        hits = [r["hit"] for r in revisits if r["hit"] is not None]
        out[tag] = {"rows": rows, "revisits": len(revisits), "failed": sum(1 for r in rows if r["status"] != 200),
                    "hit_mean": round(statistics.mean(hits), 4) if hits else None,
                    "hit_ge_0.9": sum(1 for h in hits if h >= 0.9), "log": trc_counts(server_log(tag))}
    save("b2pressure", out)
    a, b = out["b2pressure"], out["b2pressure_cr0"]
    return (f"cache-ram 448: revisit hit mean {a['hit_mean']} ({a['hit_ge_0.9']}/{a['revisits']} >= 0.9), failed "
            f"{a['failed']}, log {a['log']} | cache-ram 0: hit mean {b['hit_mean']} "
            f"({b['hit_ge_0.9']}/{b['revisits']}), failed {b['failed']}, purges {b['log']['purge']}")


async def probe_reconrace() -> str:
    """-c 32768: slot 0 idle with 2,000 tokens while slots 1-8 decode 300 tokens. An erase of
    slot 0 is sent, then (2 ms later) GET /slots. 20 trials, slot 0 refilled each time. Counts
    snapshots that still show slot 0's tokens although the erase was sent first, and those
    answered after the erase's own response (the case the version rule must catch)."""
    trials = []
    with Server("reconrace", 32768, 16):
        async with client() as c, client() as ctl, client() as poll:
            for i in range(20):
                await complete_tok(c, 0, ids(2000, 5), 1)
                bg = asyncio.gather(*(stream(c, s, ids(200, 300 + 20 * i + s), 300) for s in range(1, 9)))
                await asyncio.sleep(0.5)
                t0 = time.monotonic()
                er = asyncio.create_task(slot_op(ctl, 0, "erase"))
                await asyncio.sleep(0.002)
                ts = time.monotonic()
                snap = await slots(poll)
                t_snap = time.monotonic()
                e = await er
                await bg
                trials.append({"erase_wall_ms": e["wall_ms"], "erase_done_before_snap": e["t1"] < t_snap,
                               "snap_sent_ms": round((ts - t0) * 1000, 2),
                               "snap_wall_ms": round((t_snap - ts) * 1000, 2),
                               "slot0_tokens_in_snap": slot_tokens(snap, 0), "n_erased": e["resp"].get("n_erased")})
    stale = [t for t in trials if (t["slot0_tokens_in_snap"] or 0) > 0]
    stale_after_ack = [t for t in stale if t["erase_done_before_snap"]]
    save("reconrace", {"trials": trials})
    return (f"{len(stale)}/{len(trials)} snapshots show slot 0 still holding tokens after the erase was sent; "
            f"{len(stale_after_ack)} of them arrived after the erase response; erase wall median "
            f"{statistics.median(t['erase_wall_ms'] for t in trials)} ms under decode")


PROBES = {"fragsave": probe_fragsave, "tokenize": probe_tokenize, "dropids": probe_dropids,
          "opafter": probe_opafter, "disconnect": probe_disconnect, "kvdebug": probe_kvdebug,
          "b2pressure": probe_b2pressure, "reconrace": probe_reconrace}


# --- offline check ----------------------------------------------------------------------

async def dry() -> str:
    """No server: the agent-built dev prompts, the family prefixes under a stand-in tokenizer
    (4-character chunks, as MockPoolLLM), and the log counters on synthetic lines."""
    vocab: dict[str, int] = {}

    async def fake_tok(text: str) -> list[int]:
        return [vocab.setdefault(text[i:i + 4], len(vocab)) for i in range(0, len(text) - len(text) % 4, 4)]

    prompts = await dev_prompts(20)
    pref6, pref6x = await family_prefix(fake_tok, False), await family_prefix(fake_tok, True)
    by_prof: dict[str, int] = {}
    for p in prompts:
        by_prof[p["profile"]] = by_prof.get(p["profile"], 0) + 1
    # Every react prompt of the family starts with the family prefix (append-only prompts).
    react6 = [p for p in prompts if p["profile"] in ("react_short", "react_long")]
    starts = 0
    for p in react6:
        starts += (await fake_tok(p["prompt"]))[: len(pref6)] == pref6
    log = "\n".join(["0.01.000.000 W srv  try_clear_id: purging slot 0 with 1470 tokens",
                     "0.01.000.100 D slot update_slots: id  1 | task 3 | saving idle slot to prompt cache",
                     "0.01.000.200 W srv          load:  - making room for prompt cache entry, removing oldest"])
    counts = trc_counts(log)
    assert counts["purge"] == 1 and counts["idle_slot_saved"] == 1 and counts["cache_room"] == 1, counts
    assert lcp([1, 2, 3], [1, 2, 4]) == 2
    assert equality_stats([{"tok": 5, "prompt_n": 1, "cache_n": 4}])["equal"] == 1
    return (f"{len(prompts)} prompts from 20 dev tasks {by_prof}; stand-in L_f react6 {len(pref6)}, react6x "
            f"{len(pref6x)}; {starts}/{len(react6)} react6 prompts start with the prefix; helpers ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help=f"comma-separated subset of {','.join(PROBES)}")
    ap.add_argument("--dry", action="store_true", help="offline check only (no server)")
    args = ap.parse_args()
    if args.dry:
        print(f"[dry] {asyncio.run(dry())}", flush=True)
        return
    names = args.only.split(",") if args.only else list(PROBES)
    for name in names:
        try:
            print(f"[{name}] {asyncio.run(PROBES[name]())}", flush=True)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
