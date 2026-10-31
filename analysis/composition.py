"""Token-level composition of agent contexts (E5): what the KV of a finished task holds.

Each task's prompt ledger (tasks.jsonl, trace.segments) is tokenized with the Qwen2.5
tokenizer and split by segment type: shared prefix (system prompt + few-shot examples),
question, the model's thoughts/actions, observations that are supporting facts, other
observation text, observation framing, and template/control tokens.

    uv run --with tokenizers python -m analysis.composition results/e1_concurrency/k16/rep0 [...]
      [--tokenizer models/qwen2.5-tokenizer/tokenizer.json]

Writes composition.json next to each run; prints shares per profile.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

KINDS = ["shared_prefix", "question", "thought_action", "obs_sf", "obs_other", "obs_meta", "control"]


def compose(run: Path, tok: Tokenizer) -> dict[str, Any]:
    per_profile: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_tasks: dict[str, int] = defaultdict(int)
    with open(run / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            segs = row["trace"]["segments"]
            if not segs:
                continue
            n_tasks[row["profile"]] += 1
            for kind, text in segs:
                per_profile[row["profile"]][kind] += len(tok.encode(text, add_special_tokens=False).ids)
    out: dict[str, Any] = {}
    for prof, counts in per_profile.items():
        total = sum(counts.values())
        private = total - counts["shared_prefix"]
        obs = counts["obs_sf"] + counts["obs_other"] + counts["obs_meta"]
        out[prof] = {
            "tasks": n_tasks[prof],
            "mean_tokens": round(total / n_tasks[prof]),
            "share_of_total": {k: round(counts[k] / total, 4) for k in KINDS},
            "share_of_private": {k: round(counts[k] / private, 4) for k in KINDS if k != "shared_prefix"},
            "sf_share_of_observation": round(counts["obs_sf"] / obs, 4) if obs else None,
        }
    (run / "composition.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tokenizer", default="models/qwen2.5-tokenizer/tokenizer.json")
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    for r in args.runs:
        res = compose(Path(r), tok)
        for prof, d in sorted(res.items()):
            s = d["share_of_total"]
            print(f"{r} {prof:<14} n={d['tasks']:<4} mean {d['mean_tokens']:>5} tok | shared {s['shared_prefix']:.0%} "
                  f"thought {s['thought_action']:.0%} obs_sf {s['obs_sf']:.0%} obs_other {s['obs_other']:.0%} "
                  f"obs_meta {s['obs_meta']:.0%} | SF share of obs {d['sf_share_of_observation']}")


if __name__ == "__main__":
    main()
