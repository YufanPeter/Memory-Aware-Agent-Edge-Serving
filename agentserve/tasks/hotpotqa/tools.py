"""Wikipedia-style tools over one HotpotQA item's paragraphs (ReAct paper style)."""
from __future__ import annotations

import difflib
import re

from agentserve.agents.tools.base import Tool
from agentserve.tasks.hotpotqa.dataset import HotpotItem


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


class Corpus:
    """Shared state between search and lookup for one task execution."""

    def __init__(self, item: HotpotItem) -> None:
        self.docs = dict(zip(item.titles, item.paragraphs))
        self.current: list[str] | None = None
        self.lookup_kw: str | None = None
        self.lookup_hits: list[str] = []
        self.lookup_idx = 0

    def find(self, query: str) -> tuple[str, list[str]] | None:
        q = _norm(query)
        by_norm = {_norm(t): t for t in self.docs}
        if q in by_norm:
            t = by_norm[q]
            return t, self.docs[t]
        # substring then fuzzy match on titles
        subs = [t for n, t in by_norm.items() if q and (q in n or n in q)]
        if subs:
            t = min(subs, key=len)
            return t, self.docs[t]
        close = difflib.get_close_matches(q, list(by_norm), n=1, cutoff=0.6)
        if close:
            t = by_norm[close[0]]
            return t, self.docs[t]
        return None


class SearchTool(Tool):
    name = "search"
    description = "search: get the Wikipedia paragraph about a person, place or thing. Example: Action: search[Barack Obama]"

    def __init__(self, corpus: Corpus, n_sentences: int = 5) -> None:
        self.corpus, self.n = corpus, n_sentences

    async def call(self, arg: str) -> str:
        hit = self.corpus.find(arg)
        if hit is None:
            similar = difflib.get_close_matches(arg, list(self.corpus.docs), n=5, cutoff=0.0)
            return f"Could not find [{arg}]. Similar: {similar}."
        title, sents = hit
        self.corpus.current, self.corpus.lookup_kw = sents, None
        return f"[{title}] " + "".join(sents[: self.n]).strip()


class LookupTool(Tool):
    name = "lookup"
    description = "lookup: find the next sentence containing a keyword in the last searched paragraph. Example: Action: lookup[born]"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    async def call(self, arg: str) -> str:
        c = self.corpus
        if c.current is None:
            return "No paragraph searched yet. Use search first."
        if arg != c.lookup_kw:
            c.lookup_kw = arg
            c.lookup_hits = [s.strip() for s in c.current if arg.lower() in s.lower()]
            c.lookup_idx = 0
        if c.lookup_idx >= len(c.lookup_hits):
            return "No more results."
        c.lookup_idx += 1
        return f"(Result {c.lookup_idx} / {len(c.lookup_hits)}) {c.lookup_hits[c.lookup_idx - 1]}"
