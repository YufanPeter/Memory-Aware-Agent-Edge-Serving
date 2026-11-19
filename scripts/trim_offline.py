"""Offline TRIM estimate (docs/phase3_spec.md Q1), with no LLM: how much observation text each
selector keeps and how many supporting-fact sentences survive.

A. Gold searches. For every dev question, the observation `search[gold title]` returns under the
   react_long profile (whole paragraph plus 9 BM25 neighbours), one per supporting-fact title.
   `referenced` needs the Thought the model writes after reading, so it is not in this part.

B. Trace replay (--traces). The observations of finished react_long tasks in real runs, with the
   Thought/Action the model actually wrote after each, so `referenced` is measured too. The
   trajectory is held fixed (a trimmed run would make different later steps), so the context
   sizes are a first-order estimate: lazy TRIM compacts every consumed observation before the
   next request; eager also compacts the newest consumed one during the tool wait.

    uv run --no-sync python -m scripts.trim_offline [--split dev] [--traces 'results/p2_main/c32k_d4_b1_k8/rep*']
        [--json out.json]

Prints markdown tables; writes nothing unless --json is given.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from edgeserve.corpus import Corpus
from edgeserve.data import load_split
from edgeserve.profiles import load_profiles
from edgeserve.tools import ToolBox
from edgeserve.trim import SENTENCE_KINDS, TrimContext, compact_pieces, idf_table, keep_relevance, parse_mode

Pieces = list[tuple[str, str]]
Selector = Callable[[Pieces, TrimContext], set[int]]
GOLD_MODES = ["none", "oracle", "drop", "relevance:1", "relevance:2", "relevance:3", "relevance:5"]
TRACE_MODES = ["none", "oracle", "drop", "referenced", "relevance:1", "relevance:2", "relevance:3"]
KEEP_STATUS = ("ok", "forced")


def selectors(modes: list[str], ablate_title: bool = True) -> dict[str, Selector]:
    """The parsed modes, plus `relevance:2 (title kept)`: the same ranking without removing the
    page's own title tokens, to show what the spec's salient-token rule costs relevance."""
    out: dict[str, Selector] = {m: parse_mode(m).kept for m in modes}
    if ablate_title:
        out["relevance:2 (title excluded)"] = lambda p, c: keep_relevance(p, c, 2, exclude_title=True)
    return out


class Tally:
    """Character and sentence counts of one selector over many observations."""

    def __init__(self) -> None:
        self.c = defaultdict(int)

    def add(self, pieces: Pieces, kept: set[int]) -> None:
        sents = [i for i, (k, _) in enumerate(pieces) if k in SENTENCE_KINDS]
        kept &= set(sents)
        new = compact_pieces(pieces, kept)
        sf = [i for i in sents if pieces[i][0] == "obs_sf"]
        c = self.c
        c["obs"] += 1
        c["chars"] += sum(len(t) for _, t in pieces)
        c["chars_kept"] += sum(len(t) for _, t in new)
        c["sent_chars"] += sum(len(pieces[i][1]) for i in sents)
        c["sent_chars_kept"] += sum(len(pieces[i][1]) for i in kept)
        c["sents"] += len(sents)
        c["sents_kept"] += len(kept)
        c["sf"] += len(sf)
        c["sf_kept"] += sum(1 for i in sf if i in kept)
        if sf:
            c["obs_with_sf"] += 1
            c["obs_sf_all_kept"] += all(i in kept for i in sf)

    def row(self) -> dict[str, Any]:
        c = self.c
        r = lambda a, b: round(c[a] / c[b], 4) if c[b] else None  # noqa: E731
        return {"obs": c["obs"], "obs_chars_kept": r("chars_kept", "chars"),
                "sentence_chars_kept": r("sent_chars_kept", "sent_chars"), "sentences_kept": r("sents_kept", "sents"),
                "sf_recall": r("sf_kept", "sf"), "obs_all_sf_kept": r("obs_sf_all_kept", "obs_with_sf"),
                "sents_per_obs": round(c["sents"] / c["obs"], 1) if c["obs"] else None,
                "chars_per_obs": round(c["chars"] / c["obs"]) if c["obs"] else None}


# --- A. gold searches ------------------------------------------------------------------------

def gold_searches(split: str, corpus: Corpus, profile_name: str = "react_long") -> dict[str, dict[str, Any]]:
    prof = load_profiles()[profile_name]
    idf = idf_table(corpus)
    sel = selectors(GOLD_MODES)
    tallies = {name: Tally() for name in sel}
    for item in load_split(split):
        tools = ToolBox(corpus, item.sf_set, prof.search_sentences, prof.neighbors)
        for title in dict.fromkeys(t for t, _ in item.supporting_facts):
            obs = tools.search(title)
            ctx = TrimContext(question=item.question, idf=idf)
            for name, f in sel.items():
                tallies[name].add(obs.pieces, f(obs.pieces, ctx))
    return {name: t.row() for name, t in tallies.items()}


# --- B. trace replay -------------------------------------------------------------------------

def observation_spans(segs: Pieces) -> list[tuple[int, int, int | None]]:
    """(start, end, index of the Thought/Action that consumed it, or None) per observation."""
    spans = []
    i = 0
    while i < len(segs):
        kind, text = segs[i]
        if kind == "control" and text.startswith("\nObservation "):
            j = i + 1
            while j < len(segs) and segs[j][0].startswith("obs_"):
                j += 1
            nxt = next((k for k in range(j, len(segs)) if segs[k][0] == "thought_action"), None)
            spans.append((i + 1, j, nxt))
            i = j
        else:
            i += 1
    return spans


def replay(rows: list[dict[str, Any]], questions: dict[str, str], idf: dict[str, float],
           modes: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Per-observation tallies and per-task context estimates for each selector."""
    sel = selectors(modes)
    tallies = {name: Tally() for name in sel}
    ctx_stats: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in sel}
    for row in rows:
        segs = [tuple(s) for s in row["trace"]["segments"]]
        spans = observation_spans(segs)
        steps = [s for s in row["trace"]["steps"] if s["kind"] != "trim"]
        ta = [k for k, (kind, _) in enumerate(segs) if kind == "thought_action"]
        if not spans or len(ta) != len(steps):   # an empty output would misalign requests and segments
            continue
        shared = len(segs[0][1]) if segs and segs[0][0] == "shared_prefix" else 0
        for name, f in sel.items():
            kept_len: dict[int, int] = {}
            for start, end, consumer in spans:
                pieces = list(segs[start:end])
                gen = segs[consumer][1] if consumer is not None else ""
                kept = f(pieces, TrimContext(question=questions.get(row["qid"], ""), generated=gen, idf=idf))
                if consumer is not None:
                    tallies[name].add(pieces, kept)
                kept_len[start] = sum(len(t) for _, t in compact_pieces(pieces, kept))
            _contexts(segs, spans, ta, steps, kept_len, shared, ctx_stats[name])
    out = {}
    base = ctx_stats["none"]
    for name, st in ctx_stats.items():
        out[name] = {k: round(statistics.mean(v), 1) for k, v in st.items()}
        for k in ("mean_ctx_tok", "peak_tok", "idle_tok_lazy", "idle_tok_eager"):
            if base.get(k) and st.get(k):
                out[name][f"{k}_ratio"] = round(sum(st[k]) / sum(base[k]), 4)
    return {n: t.row() for n, t in tallies.items()}, out


def _contexts(segs: Pieces, spans: list[tuple[int, int, int | None]], ta: list[int], steps: list[dict[str, Any]],
              kept_len: dict[int, int], shared: int, acc: dict[str, list[float]]) -> None:
    """Request j's prompt is segs[:ta[j]]; an observation consumed by a Thought/Action before that
    point is compacted (lazy). Chars are scaled to tokens with the request's own token count."""
    lens = [len(t) for _, t in segs]
    prompt_tok, idle_lazy, idle_eager = [], [], []
    for j, b in enumerate(ta):
        full = sum(lens[:b])
        cut = sum(sum(lens[s:e]) - kept_len[s] for s, e, c in spans if c is not None and c < b)
        tok = int(steps[j]["llm"].get("prompt_n") or 0) + int(steps[j]["llm"].get("cache_n") or 0)
        scale = tok / full if full else 0.25
        out_tok = int(steps[j]["llm"].get("predicted_n") or 0)
        prompt_tok.append((full - cut - shared) * scale)
        after = (full - cut) * scale + out_tok
        idle_lazy.append(after)
        # eager: the observation this request consumed is compacted before the tool wait too
        extra = sum(sum(lens[s:e]) - kept_len[s] for s, e, c in spans if c == b)
        idle_eager.append(after - extra * scale)
    acc["mean_ctx_tok"].append(statistics.mean(prompt_tok))       # private (non-shared) prompt tokens per request
    acc["peak_tok"].append(max(idle_lazy))                          # slot cells after the largest request
    tool_waits = [j for j, s in enumerate(steps) if s.get("action") in ("search", "lookup")]
    if tool_waits:
        acc["idle_tok_lazy"].append(statistics.mean(idle_lazy[j] for j in tool_waits))
        acc["idle_tok_eager"].append(statistics.mean(idle_eager[j] for j in tool_waits))


def load_trace_rows(patterns: list[str], profile: str) -> tuple[list[dict[str, Any]], list[str]]:
    runs = sorted({Path(p) for pat in patterns for p in glob.glob(pat)})
    rows: list[dict[str, Any]] = []
    for run in runs:
        with open(run / "tasks.jsonl", encoding="utf-8") as f:
            rows += [r for line in f if (r := json.loads(line))["profile"] == profile and r["status"] in KEEP_STATUS]
    return rows, [str(r) for r in runs]


# --- output ------------------------------------------------------------------------------------

def table(rows: dict[str, dict[str, Any]], cols: list[str]) -> str:
    out = ["| selector | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in rows.items():
        out.append(f"| {name} | " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--profile", default="react_long")
    ap.add_argument("--traces", nargs="*", default=[], help="run directory globs to replay (part B)")
    ap.add_argument("--json", help="also write the numbers here")
    args = ap.parse_args()
    if not Path("data/corpus/corpus.jsonl").exists():
        raise SystemExit("data/corpus missing: run scripts.build_corpus first")
    corpus = Corpus.load(with_bm25=True)
    cols = ["obs", "chars_per_obs", "sents_per_obs", "obs_chars_kept", "sentence_chars_kept", "sentences_kept",
            "sf_recall", "obs_all_sf_kept"]
    result: dict[str, Any] = {"split": args.split, "profile": args.profile}
    result["gold"] = gold_searches(args.split, corpus, args.profile)
    print(f"## A. search[gold title], {args.profile} settings, split {args.split}\n")
    print(table(result["gold"], cols))
    if args.traces:
        rows, runs = load_trace_rows(args.traces, args.profile)
        questions = {it.id: it.question for it in load_split(args.split)}
        obs, ctx = replay(rows, questions, idf_table(corpus), TRACE_MODES)
        result.update({"trace_runs": runs, "trace_tasks": len(rows), "trace_obs": obs, "trace_context": ctx})
        print(f"\n## B. replay of {len(rows)} {args.profile} tasks (ok/forced) from {len(runs)} runs\n")
        print(table(obs, cols))
        print("\nContext per task (tokens, trajectory held fixed; ratio = vs none):\n")
        print(table(ctx, ["mean_ctx_tok", "mean_ctx_tok_ratio", "peak_tok", "peak_tok_ratio", "idle_tok_lazy",
                          "idle_tok_lazy_ratio", "idle_tok_eager", "idle_tok_eager_ratio"]))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
