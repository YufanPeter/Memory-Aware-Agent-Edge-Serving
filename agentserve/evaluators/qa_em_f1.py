"""Exact match / token F1 with the official HotpotQA (SQuAD-style) normalization."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from agentserve.core.registry import register
from agentserve.core.types import TaskResult
from agentserve.evaluators.base import Evaluator


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        if p != g:
            return 0.0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


@register("evaluator", "qa_em_f1")
class QAEmF1(Evaluator):
    def score(self, result: TaskResult, gold: Any) -> dict[str, float]:
        golds = gold if isinstance(gold, list) else [gold]
        pred = result.prediction or ""
        em = max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)
        f1 = max(f1_score(pred, g) for g in golds)
        return {"em": em, "f1": f1}
