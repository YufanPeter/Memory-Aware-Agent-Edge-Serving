"""Global offline Wikipedia corpus: every paragraph of the HotpotQA validation set,
deduplicated by title.

Title lookup (what `search[entity]` uses) goes exact -> normalized -> normalized
without a trailing "(...)" qualifier -> fuzzy. BM25 over paragraph text supplies
the suggestions of a failed search and the extra "related" paragraphs of the
long-context profile.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz, process

CORPUS_DIR = Path("data/corpus")
FUZZY_CUTOFF = 85


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _strip_qualifier(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", s)


class Corpus:
    def __init__(self, docs: dict[str, list[str]], bm25=None) -> None:
        self.docs = docs
        self.titles = list(docs)
        self._norm_titles = [norm(t) for t in self.titles]
        self._by_norm: dict[str, str] = {}
        self._by_base: dict[str, list[str]] = {}
        for t, n in zip(self.titles, self._norm_titles):
            self._by_norm.setdefault(n, t)
            self._by_base.setdefault(norm(_strip_qualifier(t)), []).append(t)
        self._bm25 = bm25

    @classmethod
    def load(cls, path: str | Path = CORPUS_DIR, with_bm25: bool = False) -> Corpus:
        path = Path(path)
        docs = {}
        for line in (path / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            docs[d["title"]] = d["sentences"]
        bm25 = None
        if with_bm25:
            import bm25s
            bm25 = bm25s.BM25.load(str(path / "bm25"))
        return cls(docs, bm25)

    def find_title(self, query: str) -> str | None:
        if query in self.docs:
            return query
        q = norm(query)
        if not q:
            return None
        if q in self._by_norm:
            return self._by_norm[q]
        base = self._by_base.get(norm(_strip_qualifier(query)))
        if base:
            return min(base, key=len)
        hit = process.extractOne(q, self._norm_titles, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
        return self.titles[hit[2]] if hit else None

    def text_search(self, query: str, n: int = 5) -> list[str]:
        """Titles of the n best full-text BM25 matches (empty without an index)."""
        if self._bm25 is None:
            return []
        import bm25s

        ids, scores = self._bm25.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False),
                                          k=n, show_progress=False)
        return [self.titles[i] for i, s in zip(ids[0], scores[0]) if s > 0]

    def similar_titles(self, query: str, n: int = 5) -> list[str]:
        """Suggestions for a failed search: full-text BM25 hits (like Wikipedia's search
        results in the ReAct paper's tool), or fuzzy title matches without an index."""
        if hits := self.text_search(query, n):
            return hits
        hits = process.extract(norm(query), self._norm_titles, scorer=fuzz.WRatio, limit=n)
        return [self.titles[i] for _, _, i in hits]

    def neighbors(self, title: str, k: int) -> list[str]:
        """k most BM25-similar paragraphs to `title`'s paragraph, excluding itself."""
        if k <= 0:
            return []
        if self._bm25 is None:
            raise RuntimeError("corpus loaded without BM25; use Corpus.load(with_bm25=True)")
        import bm25s

        query = title + " " + " ".join(self.docs[title][:2])
        ids, _ = self._bm25.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False),
                                     k=k + 1, show_progress=False)
        out = [self.titles[i] for i in ids[0] if self.titles[i] != title]
        return out[:k]
