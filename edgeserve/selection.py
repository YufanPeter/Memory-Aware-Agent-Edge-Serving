"""Which sentences of an agent's older observations it no longer needs (selective KV).

`select_drop` picks the observation sentences to delete; edgeserve/kvtrim.SelectiveKV then deletes
them from the slot's KV in place (mechanism "kv") or from the text only, re-prefilling (mechanism
"text", the baseline). Never touched: the shared prefix, the question, every Thought/Action the model
wrote, the framing, and the newest `keep_last` observations. Each observation is processed once
(a zero-length marker segment records it).

Scorers (higher = keep):
  oracle   HotpotQA supporting-fact labels (obs_sf). An upper bound, not a deployable method.
  bm25     BM25 of the sentence against the question plus the model's own thoughts so far.
  entity   capitalised words of the sentence that the model later mentioned in a thought/action.
  recency  whole observations: keep the most recent ones, drop older ones entirely.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from edgeserve.prompt import Prompt

_WORD = re.compile(r"[a-z0-9]+")
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9'-]+")
_STOP = frozenset("the a an of in on at to and or is was were be by for with as from that this which who "
                  "what when where it its his her their he she they i we you are has have had not".split())
SENTENCE_KINDS = ("obs_sf", "obs_other")


def _observation_spans(p: Prompt) -> list[tuple[int, int]]:
    """[start, end) segment index ranges of each observation's body, oldest first."""
    spans, start = [], None
    for i, (kind, text) in enumerate(p.segments):
        if kind == "control":
            if start is not None:
                spans.append((start, i))
                start = None
            if text.lstrip("\n").startswith("Observation "):
                start = i + 1
    if start is not None:
        spans.append((start, len(p.segments)))
    return spans


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _bm25(query: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    n = len(docs)
    avg = sum(map(len, docs)) / n if n else 0.0
    df = Counter(w for d in docs for w in set(d))
    q = Counter(query)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / (avg or 1)))
        scores.append(s)
    return scores


def select_drop(p: Prompt, question: str, scorer: str = "bm25", keep: float = 0.3, keep_last: int = 1,
                marker: str = "obs_trimmed") -> tuple[set[int], list[tuple[int, int]]]:
    """Segment indices of the observation sentences to remove, and the observations considered.

    Only observations older than the latest `keep_last` ones and not yet processed (no `marker`
    segment inside) are considered; within them the scorer keeps a `keep` share of sentences
    (recency: of observations)."""
    spans = _observation_spans(p)
    spans = spans[: len(spans) - keep_last] if keep_last > 0 else spans
    spans = [(a, b) for a, b in spans if not any(k == marker for k, _ in p.segments[a:b])]
    if not spans:
        return set(), []
    sent_idx = [i for a, b in spans for i in range(a, b) if p.segments[i][0] in SENTENCE_KINDS]
    if not sent_idx:
        return set(), spans

    drop: set[int] = set()
    if scorer == "oracle":
        drop = {i for i in sent_idx if p.segments[i][0] != "obs_sf"}
    elif scorer == "recency":
        n_keep = int(keep * len(spans))
        for a, b in spans[: len(spans) - n_keep]:
            drop |= {i for i in range(a, b) if p.segments[i][0] in SENTENCE_KINDS}
    else:
        thoughts = " ".join(t for k, t in p.segments if k == "thought_action")
        if scorer == "bm25":
            scores = _bm25(_words(question + " " + thoughts), [_words(p.segments[i][1]) for i in sent_idx])
        elif scorer == "entity":
            mentioned = set(_ENTITY.findall(thoughts)) | set(_ENTITY.findall(question))
            scores = [float(len(set(_ENTITY.findall(p.segments[i][1])) & mentioned)) for i in sent_idx]
        else:
            raise ValueError(f"unknown scorer {scorer!r}")
        n_keep = max(1, math.ceil(keep * len(sent_idx)))
        ranked = sorted(range(len(sent_idx)), key=lambda j: (-scores[j], j))
        drop = {sent_idx[j] for j in ranked[n_keep:]}
    return drop, spans

