"""HotpotQA (distractor setting) loader. Each item carries its own 10 paragraphs
(2 gold + 8 distractors), which serve as an offline, deterministic corpus."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_PATH = "data/hotpotqa/distractor/validation-00000-of-00001.parquet"


@dataclass
class HotpotItem:
    id: str
    question: str
    answer: str
    qtype: str
    level: str
    titles: list[str]
    paragraphs: list[list[str]]       # sentences per paragraph
    supporting_titles: list[str]


def load_items(path: str | Path = DEFAULT_PATH) -> list[HotpotItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/download_hotpotqa.sh")
    df = pd.read_parquet(path)
    items = []
    for r in df.itertuples(index=False):
        ctx = r.context
        items.append(HotpotItem(
            id=r.id, question=r.question, answer=r.answer, qtype=r.type, level=r.level,
            titles=[str(t) for t in ctx["title"]],
            paragraphs=[[str(s) for s in sents] for sents in ctx["sentences"]],
            supporting_titles=sorted({str(t) for t in r.supporting_facts["title"]}),
        ))
    return items
