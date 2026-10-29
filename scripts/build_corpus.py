"""Merge all validation paragraphs into one corpus (deduplicated by title) and build
its BM25 index.

    uv run python -m scripts.build_corpus
"""
from __future__ import annotations

import json

import bm25s

from edgeserve.corpus import CORPUS_DIR
from edgeserve.data import load_raw


def main() -> None:
    docs: dict[str, list[str]] = {}
    n_para = 0
    for it in load_raw():
        for t, sents in zip(it.titles, it.paragraphs):
            n_para += 1
            docs.setdefault(t, sents)   # a few titles have slightly different text; keep the first

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_DIR / "corpus.jsonl", "w") as f:
        for t, sents in docs.items():
            f.write(json.dumps({"title": t, "sentences": sents}, ensure_ascii=False) + "\n")

    texts = [t + " " + "".join(s) for t, s in docs.items()]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en"))
    retriever.save(str(CORPUS_DIR / "bm25"))
    print(f"{n_para} paragraphs -> {len(docs)} unique titles; written to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
