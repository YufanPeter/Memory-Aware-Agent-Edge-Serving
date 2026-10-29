"""HotpotQA items: loading the raw parquet and the frozen dev/test splits."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

RAW_PATH = Path("data/hotpotqa/distractor/validation-00000-of-00001.parquet")
SPLIT_DIR = Path("data/splits")


@dataclass
class HotpotItem:
    id: str
    question: str
    answer: str
    qtype: str                                   # bridge | comparison
    titles: list[str]                            # the item's own 10 paragraphs (2 gold + 8 distractors)
    paragraphs: list[list[str]]                  # sentences per paragraph, as in the dataset
    supporting_facts: list[tuple[str, int]] = field(default_factory=list)  # (title, sentence index)

    @property
    def sf_set(self) -> set[tuple[str, int]]:
        return {(t, i) for t, i in self.supporting_facts}

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> HotpotItem:
        d = dict(d)
        d["supporting_facts"] = [(t, int(i)) for t, i in d["supporting_facts"]]
        return cls(**d)


def load_raw(path: str | Path = RAW_PATH) -> list[HotpotItem]:
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/download_hotpotqa.sh")
    items = []
    for r in pd.read_parquet(path).itertuples(index=False):
        items.append(HotpotItem(
            id=str(r.id), question=str(r.question), answer=str(r.answer), qtype=str(r.type),
            titles=[str(t) for t in r.context["title"]],
            paragraphs=[[str(s) for s in sents] for sents in r.context["sentences"]],
            supporting_facts=[(str(t), int(i)) for t, i in
                              zip(r.supporting_facts["title"], r.supporting_facts["sent_id"])],
        ))
    return items


def save_split(items: list[HotpotItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(it.to_json() + "\n" for it in items), encoding="utf-8", newline="\n")


def load_split(name_or_path: str | Path) -> list[HotpotItem]:
    path = Path(name_or_path)
    if not path.suffix:
        path = SPLIT_DIR / f"{name_or_path}.jsonl"
    return [HotpotItem.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
