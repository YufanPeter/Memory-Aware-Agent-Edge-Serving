"""Answer quality: official HotpotQA EM/F1, plus supporting-fact recall of retrieval."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    # As in the official script: yes/no/noanswer only score on an exact match.
    if (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def sf_recall(seen: set[tuple[str, int]], gold: set[tuple[str, int]]) -> float:
    return len(seen & gold) / len(gold) if gold else 0.0
