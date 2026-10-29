"""Workloads: which questions arrive when, with which profile.

A workload is generated once from a YAML spec and frozen to JSON, then replayed
by the runner, so every scheduler sees exactly the same tasks and arrivals.

Spec:
    name: mixed_poisson
    split: dev
    n_tasks: 40
    seed: 0
    mix: {react_short: 0.4, react_long: 0.3, react_explain: 0.2, oneshot_long: 0.1}
    arrival: {type: poisson, rate: 0.5}      # tasks/s
           | {type: batch}                   # all at t=0
           | {type: uniform, interval: 2.0}
           | {type: bursty, rate_on: 2.0, rate_off: 0.1, mean_on_s: 10, mean_off_s: 30}
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml

from edgeserve.data import load_split


def arrivals(cfg: dict[str, Any], n: int, rng: random.Random) -> list[float]:
    kind = cfg.get("type", "batch")
    if kind == "batch":
        return [0.0] * n
    if kind == "uniform":
        return [i * cfg["interval"] for i in range(n)]
    if kind == "poisson":
        t, out = 0.0, []
        for _ in range(n):
            out.append(t)
            t += rng.expovariate(cfg["rate"])
        return out
    if kind == "bursty":   # two-state MMPP: alternating on/off periods with different rates
        t, out, on = 0.0, [], True
        period_end = rng.expovariate(1.0 / cfg["mean_on_s"])
        while len(out) < n:
            gap = rng.expovariate(cfg["rate_on"] if on else cfg["rate_off"])
            if t + gap > period_end:
                t, on = period_end, not on
                period_end = t + rng.expovariate(1.0 / (cfg["mean_on_s"] if on else cfg["mean_off_s"]))
                continue
            t += gap
            out.append(t)
        return [x - out[0] for x in out]
    raise ValueError(f"unknown arrival type {kind!r}")


def make_workload(spec: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(spec.get("seed", 0))
    items = load_split(spec.get("split", "dev"))
    n = spec["n_tasks"]
    if n > len(items):
        raise ValueError(f"n_tasks={n} > {len(items)} items in split")
    chosen = rng.sample(items, n)
    names, weights = zip(*sorted(spec["mix"].items()))
    # Exact proportions (largest remainder), then shuffled, so small workloads keep the mix.
    raw = [w / sum(weights) * n for w in weights]
    counts = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[i] += 1
    profiles = [name for name, c in zip(names, counts) for _ in range(c)]
    rng.shuffle(profiles)
    times = arrivals(spec.get("arrival", {"type": "batch"}), n, rng)
    tasks = [{"task_id": f"t{i:04d}", "qid": it.id, "profile": prof, "arrival_s": round(at, 4)}
             for i, (it, prof, at) in enumerate(zip(chosen, profiles, times))]
    return {"meta": spec, "tasks": tasks}


def save_workload(wl: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(wl, indent=1, sort_keys=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_workload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_spec(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())
