"""Task profiles: how one HotpotQA question is turned into an agent task.

The profile mix of a workload is what gives Phase 1 its short/long contexts,
short/long outputs and single/multi-step tasks. Values live in configs/profiles.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROFILES_PATH = Path("configs/profiles.yaml")


@dataclass
class Profile:
    name: str
    mode: str = "react"            # react | oneshot | closedbook
    few_shot: int = 6              # number of examples in the shared prefix (react only)
    explain_tokens: int = 0        # >0: after finish, decode exactly this many Explanation tokens (long output)
    search_sentences: int = 3      # sentences returned per paragraph by search; 0 = whole paragraph
    neighbors: int = 0             # extra BM25-related paragraphs returned by search
    extra_paragraphs: int = 0      # oneshot: BM25 distractors added to the question's own 10 paragraphs
    max_steps: int = 6
    n_predict: int = 128           # generation cap per step
    max_parse_errors: int = 2
    forced_finish_tokens: int = 32
    delay: dict[str, Any] = field(default_factory=lambda: {"type": "none"})
    per_tool_delay: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_profiles(path: str | Path = PROFILES_PATH) -> dict[str, Profile]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {})
    known = {f.name for f in fields(Profile)}
    out = {}
    for name, over in cfg["profiles"].items():
        merged = {**defaults, **(over or {})}
        unknown = set(merged) - known
        if unknown:
            raise ValueError(f"profile {name}: unknown keys {sorted(unknown)}")
        out[name] = Profile(name=name, **merged)
    return out
