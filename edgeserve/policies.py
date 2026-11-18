"""Phase 2 serving configurations. Each is a set of switches for the KV manager (kvm.py).

Plan: docs/phase2_ideas.md (base -> selective KV -> grouped KV pool).

Baselines:
  fixed_k        Phase 1 runner: K workers, KV always retained, no manager at all
  per_slot       fixed_k on a server with --no-kv-unified (the sweep sets the server flag)
  llamacpp       fixed_k without slot pinning (--no-pin); the server places and evicts KV
  always_offload admission + gate; every tool call saves the KV to host RAM, restored on return
                 (InferCept-style offload)

The base (KV-budget admission, never overflow the pool):
  base           admission (overcommit rho) + pre-request gate + pressure-gated drop at tool calls
  base_fork      base + one shared copy of each system prompt via the patched server's fork action
  base_compact   base + text compaction: an agent whose KV was dropped comes back with its older
                 observations trimmed and re-prefilled. The re-prefill baseline for selective KV.

Decomposition and references:
  admit_only     admission (rho = 1) + gate, KV always retained
  ttl_fixed_k    fixed K workers + gate + pressure-gated drop (no admission)
  base_oracle    base, but each give-up decision knows the true tool delay
  compact_static no manager decisions; every request compacts older observations (quality runs)
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Policy:
    name: str
    managed: bool = True           # False: no KV manager (Phase 1 behaviour)
    admission: bool = False        # admit by projected peak KV instead of a fixed number of workers
    overcommit: float = 1.0        # rho: admit while sum of projected peaks <= rho * pool
    gate: bool = True              # check that a request fits before sending it
    lifecycle: str = "retain"      # retain | pressure | fixed_ttl | always | oracle
    give_up: str = "drop"          # drop | offload: what releasing an idle agent's KV means
    ttl_s: float = 1.0             # fixed_ttl only
    pressure: float = 0.85         # occupancy share of the pool above which memory is "under pressure"
    margin: float = 0.03           # pool share the gate keeps free (token estimates are approximate)
    compact: str = "never"         # never | on_drop | always
    compact_keep: float = 0.3      # share of older observation sentences kept by compaction
    compact_scorer: str = "bm25"   # bm25 | recency | entity | oracle
    prefetch: bool = True          # restore offloaded KV as soon as the tool returns
    fork_prefix: bool = False      # share the system prefix through the patched server
    pin: bool = True               # worker k uses slot k


_BASE = Policy("base", admission=True, overcommit=1.5, lifecycle="pressure")

POLICIES: dict[str, Policy] = {p.name: p for p in [
    Policy("fixed_k", managed=False),
    Policy("per_slot", managed=False),
    Policy("llamacpp", managed=False, pin=False),
    Policy("always_offload", admission=True, overcommit=1.5, lifecycle="always", give_up="offload"),
    _BASE,
    replace(_BASE, name="base_fork", fork_prefix=True),
    replace(_BASE, name="base_compact", compact="on_drop"),
    Policy("admit_only", admission=True, overcommit=1.0, lifecycle="retain"),
    Policy("ttl_fixed_k", lifecycle="pressure"),
    replace(_BASE, name="base_oracle", lifecycle="oracle"),
    Policy("compact_static", gate=False, compact="always"),
]}


def get_policy(name: str, **overrides: object) -> Policy:
    if name not in POLICIES:
        raise SystemExit(f"unknown policy {name!r}; choose from {', '.join(POLICIES)}")
    return replace(POLICIES[name], **{k: v for k, v in overrides.items() if v is not None})
