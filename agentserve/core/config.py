"""Experiment config loading.

An experiment YAML has one section per layer. A section may reference a
reusable component file with `ref:` (path relative to the configs/ root or to
the experiment file) and override any of its keys:

    backend: {ref: backends/llamacpp.yaml, ctx_size: 16384}

`sweep:` expands into a grid. A key that is a comma-joined list of dotted paths
binds several paths to the same value (e.g. scheduler K and server slots):

    sweep:
      "scheduler.max_concurrency,backend.parallel": [1, 2, 4]
      workload.arrival.rate: [0.2, 0.5]
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Iterator

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_refs(node: Any, here: Path) -> Any:
    if isinstance(node, dict):
        if "ref" in node:
            ref = Path(node["ref"])
            path = next(
                (p for p in (here / ref, CONFIG_ROOT / ref) if p.exists()), None
            )
            if path is None:
                raise FileNotFoundError(f"config ref not found: {ref}")
            base = _resolve_refs(yaml.safe_load(path.read_text()) or {}, path.parent)
            return _merge_over(base, {k: v for k, v in node.items() if k != "ref"}, here)
        return {k: _resolve_refs(v, here) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, here) for v in node]
    return node


def _merge_over(base: dict, over: dict, here: Path) -> dict:
    """Merge raw `over` into resolved `base`. A sub-dict that carries its own `ref`,
    or switches to a different `type`, replaces the inherited section instead of
    being merged into it (params of one implementation are not valid for another)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        old = out.get(k)
        if k == "sweep":  # a sweep is a whole experiment design; never merge two of them
            out[k] = _resolve_refs(v, here)
        elif (isinstance(v, dict) and "ref" not in v and isinstance(old, dict)
                and v.get("type", old.get("type")) == old.get("type")):
            out[k] = _merge_over(out[k], v, here)
        else:
            out[k] = _resolve_refs(v, here)
    return out


def _label(val: Any) -> str:
    if isinstance(val, list) and all(isinstance(v, dict) for v in val):
        return "+".join(str(v.get("name", i)) for i, v in enumerate(val))
    if isinstance(val, dict) and ("name" in val or "type" in val):
        if "name" in val:
            return str(val["name"])
        parts = [str(val["type"])]
        if "order" in val:
            parts.append(str(val["order"]))
        if val.get("max_concurrency"):
            parts.append(f"K{val['max_concurrency']}")
        if "mode" in val:
            parts.append(str(val["mode"]))
        ret = val.get("retention")
        if isinstance(ret, dict) and ret.get("type", "server") != "server":
            parts.append(f"{ret['type']}-{ret.get('mode', 'erase')}")
        return "-".join(parts)
    if isinstance(val, (list, dict)):
        return str(abs(hash(yaml.safe_dump(val))) % 10**6)
    return str(val)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = _resolve_refs(raw, path.parent)
    cfg.setdefault("name", path.stem)
    return cfg


def resolve(node: Any, here: Path | None = None) -> Any:
    """Resolve `ref:` entries in a value supplied after loading (e.g. a CLI override)."""
    return _resolve_refs(node, here or CONFIG_ROOT)


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def expand_sweep(cfg: dict) -> Iterator[tuple[str, dict]]:
    """Yield (point_id, concrete_config) for each grid point x seed."""
    sweep = cfg.get("sweep") or {}
    seeds = cfg.get("seeds") or [cfg.get("seed", 0)]
    base = {k: v for k, v in cfg.items() if k not in ("sweep", "seeds")}
    keys = list(sweep)
    # unique, readable label per value of each swept key
    labels_for: dict[str, list[str]] = {}
    for k in keys:
        raw = [_label(v) for v in sweep[k]]
        labels_for[k] = [f"{lab}#{i}" if raw.count(lab) > 1 else lab for i, lab in enumerate(raw)]
    for idx in itertools.product(*(range(len(sweep[k])) for k in keys)):
        values = [sweep[k][i] for k, i in zip(keys, idx)]
        for seed in seeds:
            point = copy.deepcopy(base)
            parts, labels = [], {}
            for key, val, i in zip(keys, values, idx):
                for dotted in key.split(","):
                    set_dotted(point, dotted.strip(), val)
                short = key.split(",")[0].split(".")[-1]
                lab = labels_for[key][i]
                labels[short] = val if not isinstance(val, (list, dict)) else lab
                parts.append(f"{short}={lab}")
            point["seed"] = seed
            parts.append(f"seed={seed}")
            point["point"] = {**labels, "seed": seed}
            yield "__".join(parts), point
