"""Freeze dev/test splits from the validation set, stratified by question type.

    uv run python -m scripts.build_splits [--dev 200] [--test 500] [--seed 0]

dev is for tuning (prompts, best fixed K); test is only for the final report.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from edgeserve.data import SPLIT_DIR, load_raw, save_split


def stratified(items, n, rng):
    by_type = defaultdict(list)
    for it in items:
        by_type[it.qtype].append(it)
    total = len(items)
    picked = []
    for qtype, group in sorted(by_type.items()):
        k = round(n * len(group) / total)
        picked += rng.sample(group, k)
    rng.shuffle(picked)
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = load_raw()
    rng = random.Random(args.seed)
    dev = stratified(items, args.dev, rng)
    dev_ids = {it.id for it in dev}
    test = stratified([it for it in items if it.id not in dev_ids], args.test, rng)

    save_split(dev, SPLIT_DIR / "dev.jsonl")
    save_split(test, SPLIT_DIR / "test.jsonl")
    for name, s in [("dev", dev), ("test", test)]:
        n_cmp = sum(it.qtype == "comparison" for it in s)
        print(f"{name}: {len(s)} items ({n_cmp} comparison)")


if __name__ == "__main__":
    main()
