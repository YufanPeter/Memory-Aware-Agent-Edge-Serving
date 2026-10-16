"""Per-group latency/quality breakdown, e.g. to check that long tasks are not
unfairly delayed by a policy."""
from __future__ import annotations

from typing import Any

import pandas as pd

from agentserve.metrics.system import FINISHED


def by_group(tasks: pd.DataFrame, col: str) -> dict[str, dict[str, Any]]:
    if tasks.empty or col not in tasks:
        return {}
    out = {}
    for g, df in tasks.groupby(col):
        fin = df[df.status.isin(FINISHED)]
        row: dict[str, Any] = {
            "n": len(df), "n_finished": len(fin),
            "em": float(df.em.fillna(0).mean()), "f1": float(df.f1.fillna(0).mean()),
        }
        if not fin.empty:
            row.update(latency_p50=float(fin.latency.median()), latency_p95=float(fin.latency.quantile(0.95)))
            if "slowdown" in fin:
                row.update(slowdown_p50=float(fin.slowdown.median()), slowdown_p95=float(fin.slowdown.quantile(0.95)))
        out[str(g)] = row
    return out
