"""Check a KV-managed run against the server: was the manager's accounting safe?

For every request the manager reserves an *estimate* of the slot's size after the request
(kv_state -> active) and later records the *actual* size the server reported
(kv_state active -> ready, prompt_n + cache_n + predicted_n). The gate is only safe if estimates
are not below actual sizes, so this reports how often and by how much they fall short.
It also counts what the server did on its own: idle-slot purges ("purging slot") and context
overflows, both of which a working manager should prevent.

    uv run python -m analysis.check_kvm results/p2_smoke/base/rep0 [more run dirs]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def check(run: Path) -> dict[str, Any]:
    pending: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    states: dict[str, int] = {}
    with open(run / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"kv_state"' not in line:
                continue
            e = json.loads(line)
            states[e["to"]] = states.get(e["to"], 0) + 1
            if e["to"] == "active":
                pending[e["slot"]] = e["tokens"]
            elif e["frm"] == "active" and e["to"] == "ready" and e["slot"] in pending and e["tokens"] > 0:
                pairs.append((pending.pop(e["slot"]), e["tokens"]))
    log = (run / "server.log").read_text(encoding="utf-8", errors="replace") if (run / "server.log").exists() else ""
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "run": str(run), "statuses": summary["statuses"],
        "server_purges": log.count("purging slot"),
        "server_overflows": log.count("Context size has been exceeded"),
        "requests": len(pairs),
    }
    if pairs:
        short = [a - e for e, a in pairs if e < a]
        ratio = sorted(e / a for e, a in pairs)
        out.update({
            "estimate_below_actual": len(short),
            "max_shortfall_tokens": max(short) if short else 0,
            "estimate_over_actual_p50": round(statistics.median(ratio), 3),
            "estimate_over_actual_min": round(ratio[0], 3),
            "estimate_over_actual_max": round(ratio[-1], 3),
        })
    out["kv_transitions"] = states
    return out


def main() -> None:
    for arg in sys.argv[1:]:
        print(json.dumps(check(Path(arg)), indent=1))


if __name__ == "__main__":
    main()
