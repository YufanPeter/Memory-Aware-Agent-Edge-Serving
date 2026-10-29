"""search / lookup tools over the global corpus (ReAct-paper style).

Every observation is built from labelled pieces so the agent can record which
context tokens are supporting facts (obs_sf), other retrieved text (obs_other),
or formatting / messages (obs_meta). `Observation.text` is exactly the
concatenation of its pieces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from edgeserve.corpus import Corpus


@dataclass
class Observation:
    pieces: list[tuple[str, str]] = field(default_factory=list)   # (kind, text)
    sf_hits: set[tuple[str, int]] = field(default_factory=set)    # supporting facts shown

    def add(self, kind: str, text: str) -> None:
        if text:
            self.pieces.append((kind, text))

    @property
    def text(self) -> str:
        return "".join(t for _, t in self.pieces)


class ToolBox:
    """Tools for one task execution. Holds lookup state (the last searched paragraph)."""

    names = ("search", "lookup")

    def __init__(self, corpus: Corpus, sf: set[tuple[str, int]],
                 search_sentences: int = 3, neighbors: int = 0) -> None:
        self.corpus = corpus
        self.sf = sf
        self.search_sentences = search_sentences   # 0 = whole paragraph
        self.n_neighbors = neighbors
        self._current: str | None = None
        self._lookup_kw: str | None = None
        self._lookup_hits: list[int] = []
        self._lookup_idx = 0

    def call(self, name: str, arg: str) -> Observation:
        if name == "search":
            return self.search(arg)
        if name == "lookup":
            return self.lookup(arg)
        obs = Observation()
        obs.add("obs_meta", f"Unknown action '{name}'. Use search[entity], lookup[keyword] or finish[answer].")
        return obs

    def _add_sentences(self, obs: Observation, title: str, idxs: list[int]) -> None:
        sents = self.corpus.docs[title]
        for i in idxs:
            is_sf = (title, i) in self.sf
            text = sents[i] if i else sents[i].lstrip()
            obs.add("obs_sf" if is_sf else "obs_other", text)
            if is_sf:
                obs.sf_hits.add((title, i))

    def _add_paragraph(self, obs: Observation, title: str, limit: int) -> None:
        n = len(self.corpus.docs[title])
        obs.add("obs_meta", f"[{title}] ")
        self._add_sentences(obs, title, list(range(n if limit <= 0 else min(limit, n))))

    def paragraphs(self, titles: list[str]) -> Observation:
        """Whole paragraphs, one per line (the long-context one-shot profile's context)."""
        obs = Observation()
        for t in titles:
            self._add_paragraph(obs, t, 0)
            obs.add("obs_meta", "\n")
        return obs

    def search(self, arg: str) -> Observation:
        obs = Observation()
        title = self.corpus.find_title(arg.strip())
        if title is None:
            similar = self.corpus.similar_titles(arg)
            obs.add("obs_meta", f"Could not find [{arg}]. Similar: {similar}.")
            return obs
        self._current, self._lookup_kw = title, None
        self._add_paragraph(obs, title, self.search_sentences)
        for nb in self.corpus.neighbors(title, self.n_neighbors):
            obs.add("obs_meta", "\n")
            self._add_paragraph(obs, nb, self.search_sentences)
        return obs

    def lookup(self, arg: str) -> Observation:
        obs = Observation()
        if self._current is None:
            obs.add("obs_meta", "No paragraph searched yet. Use search first.")
            return obs
        kw = arg.strip().lower()
        if kw != self._lookup_kw:
            self._lookup_kw = kw
            self._lookup_hits = [i for i, s in enumerate(self.corpus.docs[self._current]) if kw and kw in s.lower()]
            self._lookup_idx = 0
        if self._lookup_idx >= len(self._lookup_hits):
            obs.add("obs_meta", "No more results.")
            return obs
        i = self._lookup_hits[self._lookup_idx]
        self._lookup_idx += 1
        obs.add("obs_meta", f"(Result {self._lookup_idx} / {len(self._lookup_hits)}) ")
        self._add_sentences(obs, self._current, [i])
        obs.pieces[-1] = (obs.pieces[-1][0], obs.pieces[-1][1].lstrip())
        return obs
