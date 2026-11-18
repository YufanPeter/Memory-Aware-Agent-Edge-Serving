"""Context compaction ("slim instead of evict", direction 2 of the Phase 2 plan).

When an agent's KV has to go, it can come back smaller: before the next request, the
sentences of its *older* observations that a scorer deems unhelpful are removed and each
removed run is replaced by " [...]". Everything else stays byte-identical: the shared prefix,
the question, every Thought/Action the model wrote, the framing, and the latest observation
(the model has not read it yet). The prompt then diverges from the slot's cached KV at the first
removed sentence, so llama-server re-prefills only from there (or from scratch after a drop),
and the context stays smaller for the rest of the task.

An observation is compacted at most once (its " [...]" marker shows it was), so repeated
compaction does not keep eating the same text.

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
from dataclasses import dataclass

from edgeserve.prompt import Prompt

ELIDED = " [...]"
_WORD = re.compile(r"[a-z0-9]+")
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9'-]+")
_STOP = frozenset("the a an of in on at to and or is was were be by for with as from that this which who "
                  "what when where it its his her their he she they i we you are has have had not".split())
SENTENCE_KINDS = ("obs_sf", "obs_other")


@dataclass
class CompactResult:
    chars_removed: int = 0
    sentences_removed: int = 0
    sf_removed: int = 0           # supporting-fact sentences removed (quality risk)
    observations: int = 0         # observations compacted this time


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


def compact(p: Prompt, question: str, scorer: str = "bm25", keep: float = 0.3) -> CompactResult:
    """Compact p in place. Returns what was removed."""
    res = CompactResult()
    spans = _observation_spans(p)[:-1]                      # never the latest observation
    spans = [(a, b) for a, b in spans if not any(k == "obs_elided" for k, _ in p.segments[a:b])]
    if not spans:
        return res
    sent_idx = [i for a, b in spans for i in range(a, b) if p.segments[i][0] in SENTENCE_KINDS]
    if not sent_idx:
        return res

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
    span_ends = {b - 1 for a, b in spans}
    new: list[tuple[str, str]] = []
    for i, (kind, text) in enumerate(p.segments):
        if i in drop:
            res.chars_removed += len(text)
            res.sentences_removed += 1
            res.sf_removed += kind == "obs_sf"
            if not new or new[-1] != ("obs_elided", ELIDED):
                new.append(("obs_elided", ELIDED))
                res.chars_removed -= len(ELIDED)
        else:
            new.append((kind, text))
        if i in span_ends:
            # zero-length marker: changes no prompt bytes, but this observation is never compacted again
            new.append(("obs_elided", ""))
    res.observations = len(spans)
    p.segments[:] = new
    return res
