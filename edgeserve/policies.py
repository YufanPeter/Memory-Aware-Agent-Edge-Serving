"""Policy presets and the pure decision functions of the KV manager (docs/phase2_spec.md §4).

load_policy() turns a preset of configs/kvm.yaml plus overrides into a KVConfig. Everything
else here is a pure function, so the decisions can be unit-tested without a server:

- the expected residual tool time that ranks idle victims (largest first: the agent that
  returns last goes first). Online by default, from the tool durations observed in the same
  run (DelayModel); `oracle_prior` instead uses the generator's own distribution;
- the victim order (rank_victims) and the action per victim (choose_action: offload to the
  host tier, drop to the family prefix, or erase);
- the ski-rental break-even B and T* = B/S frozen at tool start (break_even).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

KVM_CONFIG_PATH = Path("configs/kvm.yaml")

# configs/profiles.yaml: tool delay = scale x min(lognormal(median 0.8 s, sigma 0.6), 5 s); lookup 0.1 s.
ORACLE = {"median": 0.8, "sigma": 0.6, "cap": 5.0}
ORACLE_FIXED = {"lookup": 0.1}
# The online estimator's prior: weak and independent of the run's delay scale (spec §4.2).
WEAK_PRIOR = {"median": 1.0, "sigma": 1.0}
PRIOR_WEIGHT = 10        # pseudo-observations
PRIOR_UNTIL = 30         # from this many observations of a tool on, the prior is dropped

CHOICES = {
    "gate": ("none", "active", "full"),
    "admission": ("fixed", "kv", "duty"),
    "lifecycle": ("observe", "server", "managed"),
    "timing": ("pressure_ttl", "lazy", "nosteal", "fixed_ttl", "always", "clairvoyant", "clairvoyant_pf"),
    "delay_model": ("online", "oracle_prior"),
    "victim": ("residual", "lowest_id", "largest"),
    "actions": ("cost", "offload", "drop", "erase"),
}
# Each lifecycle has exactly one safe gate: retain + full gate deadlocks (idle agents hold memory
# and their own returns are gated), so llama.cpp's purge needs the active gate (spec §2.2).
GATE_OF = {"observe": "none", "server": "active", "managed": "full"}


def load_policy(name: str, overrides: dict[str, Any] | None = None, path: str | Path = KVM_CONFIG_PATH) -> Any:
    """KVConfig for preset `name`: yaml defaults, then the preset, then `overrides`."""
    from edgeserve.kvm import KVConfig   # kvm imports this module

    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    presets = spec.get("presets") or {}
    if name not in presets:
        raise ValueError(f"unknown policy {name!r}; presets in {path}: {sorted(presets)}")
    merged = {**(spec.get("defaults") or {}), **(presets[name] or {}), **(overrides or {})}
    unknown = set(merged) - {f.name for f in fields(KVConfig)}
    if unknown:
        raise ValueError(f"policy {name}: unknown keys {sorted(unknown)}")
    cfg = KVConfig(**merged)
    validate(cfg)
    return cfg


def validate(cfg: Any) -> None:
    for key, allowed in CHOICES.items():
        if getattr(cfg, key) not in allowed:
            raise ValueError(f"{key}={getattr(cfg, key)!r}, expected one of {allowed}")
    if cfg.gate != GATE_OF[cfg.lifecycle]:
        raise ValueError(f"lifecycle {cfg.lifecycle} needs gate {GATE_OF[cfg.lifecycle]}, got {cfg.gate}")
    if cfg.k < 1 or cfg.margin < 0 or cfg.host_frac < 0:
        raise ValueError(f"bad k/margin/host_frac: {cfg.k}/{cfg.margin}/{cfg.host_frac}")


# --- residual tool time ----------------------------------------------------------------------

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lognormal_residual(elapsed: float, median: float, sigma: float, cap: float | None = None) -> float:
    """E[D - e | D > e] for D = min(X, cap), X ~ lognormal(ln median, sigma)."""
    if cap is not None and elapsed >= cap:
        return 0.0
    mu, e = math.log(median), max(elapsed, 1e-9)
    surv = 1.0 - _phi((math.log(e) - mu) / sigma)
    if surv < 1e-12:
        return 0.0
    m = math.exp(mu + sigma * sigma / 2)
    tail = m * _phi((mu + sigma * sigma - math.log(e)) / sigma)        # E[X; X > e]
    if cap is not None:
        tail -= m * _phi((mu + sigma * sigma - math.log(cap)) / sigma)  # minus E[X; X > cap]
        tail += cap * (1.0 - _phi((math.log(cap) - mu) / sigma))        # plus cap * P(X > cap)
    return max(0.0, tail / surv - e)


def lognormal_survival(elapsed: float, median: float, sigma: float) -> float:
    return 1.0 - _phi((math.log(max(elapsed, 1e-9)) - math.log(median)) / sigma)


def residual(tool: str | None, elapsed: float, scale: float) -> float:
    """Oracle prior (the generator's distribution x `scale`): `oracle_prior` and its P5
    misspecification. Never the true delay of this call."""
    if tool in ORACLE_FIXED:
        return max(0.0, ORACLE_FIXED[tool] * scale - elapsed)
    if scale <= 0:
        return 0.0
    return scale * lognormal_residual(elapsed / scale, ORACLE["median"], ORACLE["sigma"], ORACLE["cap"])


@dataclass
class DelayModel:
    """Online residual estimate per tool from the durations observed in this run (tool_end.idle_s):
    the mean of (d - elapsed) over observed d > elapsed, blended below PRIOR_UNTIL observations
    with PRIOR_WEIGHT pseudo-observations from the weak prior (conditioned on d > elapsed too)."""

    obs: dict[str, list[float]] = field(default_factory=dict)   # sorted durations per tool

    def observe(self, tool: str, d: float) -> None:
        bisect.insort(self.obs.setdefault(tool, []), d)

    def count(self, tool: str) -> int:
        return len(self.obs.get(tool, []))

    def median(self, tool: str) -> float | None:
        xs = self.obs.get(tool)
        return xs[len(xs) // 2] if xs else None

    def residual(self, tool: str | None, elapsed: float) -> float:
        xs = self.obs.get(tool or "", [])
        i = bisect.bisect_right(xs, elapsed)
        n_tail = len(xs) - i
        tail = sum(xs[i:]) - elapsed * n_tail
        w = 0.0
        if len(xs) < PRIOR_UNTIL:
            w = PRIOR_WEIGHT * lognormal_survival(elapsed, WEAK_PRIOR["median"], WEAK_PRIOR["sigma"])
        if n_tail == 0 and w < 1e-9:
            return 0.0   # overdue against everything seen: expected back any moment
        prior = lognormal_residual(elapsed, WEAK_PRIOR["median"], WEAK_PRIOR["sigma"]) if w else 0.0
        return (tail + w * prior) / (n_tail + w)


# --- victims and actions ---------------------------------------------------------------------

@dataclass
class Victim:
    """One evictable slot, as seen by one head (spec §4.2)."""
    slot: int
    cls: int                    # 0 the head's own stale tail, 1 STALE, 2 in search (or another tool),
                                # 3 in lookup, 4 READY behind the head (priority-0 heads only)
    s: int                      # resident cells
    keep: int = 0               # cells a drop to the family prefix keeps (0: no family)
    residual: float = 0.0       # classes 2-3: expected remaining tool time
    ready_seq: int = -1         # class 4: latest-ready goes first
    owner: str | None = None
    erase_ok: bool = True       # may its prefix be erased (I5: a dropped agent only for priority 0)
    owned: bool = True          # the cells are the owner's own context (not a previous task's)

    @property
    def at_prefix(self) -> bool:
        return 0 < self.keep and self.s <= self.keep


def rank_victims(cands: list[Victim], victim: str = "residual") -> list[Victim]:
    """Eviction order. `residual`: class order, within IN_TOOL by expected residual (largest
    first) then larger s; READY agents latest-ready first. `lowest_id`: the server's own order.
    `largest`: most cells first. The head's own stale tail (class 0) always goes first."""
    own = [v for v in cands if v.cls == 0]
    rest = [v for v in cands if v.cls != 0]
    if victim == "lowest_id":
        return own + sorted(rest, key=lambda v: v.slot)
    if victim == "largest":
        return own + sorted(rest, key=lambda v: (-v.s, v.slot))

    def key(v: Victim) -> tuple:
        if v.cls in (2, 3):
            return (v.cls, -v.residual, -v.s, v.slot)
        if v.cls == 4:
            return (v.cls, -v.ready_seq, -v.s, v.slot)
        return (v.cls, -v.s, v.slot)
    return own + sorted(rest, key=key)


def choose_action(s: int, keep: int, host_room: bool, c_off: float, c_drop: float,
                  actions: str = "cost", owned: bool = True) -> str:
    """offload | drop | erase for one victim (spec §4.2). Unowned (stale) KV is never offloaded:
    nobody returns for it. Without a family prefix, or already at it, only erase frees cells."""
    if actions == "erase" or keep <= 0 or s <= keep:
        return "erase"
    if not owned or actions == "drop":
        return "drop"
    if actions == "offload":
        return "offload" if host_room else "drop"
    return "offload" if host_room and c_off < c_drop else "drop"


def break_even(active: int, s: int, c_off: float, c_drop: float, host_room: bool) -> tuple[float, float]:
    """B = (A + S) * min(c_off, c_drop) token-seconds (the op stalls the A active tokens too),
    T* = B / S: how long retaining S tokens under pressure costs as much as giving them up."""
    c = min(c_off if host_room else math.inf, c_drop)
    b = (active + s) * c
    return b, (b / s if s > 0 else 0.0)
