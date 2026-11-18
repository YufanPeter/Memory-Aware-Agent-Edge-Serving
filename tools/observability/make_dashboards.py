"""Generate the two Grafana dashboards (live and replay) from one panel layout.

    uv run python tools/observability/make_dashboards.py

Writes grafana/dashboards/edgeserve_live.json and edgeserve_replay.json. Each selected run gets
its own row (Grafana repeats the row for every value of $run), so replayed runs can be
compared one above the other on the same time axis.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = {"type": "prometheus", "uid": "prom"}
R = 'run=~"$run"'

# Slot states: code -> (name, colour). Codes match STATE in exporter.py.
STATES = [(0, "empty", "#3a3a3a"), (1, "prefill", "#ff9830"), (2, "decode", "#73bf69"),
          (3, "active", "#56a64b"), (4, "ready", "#8ab8ff"), (5, "tool wait", "#b877d9"),
          (6, "gate wait", "#fade2a"), (7, "stale", "#8f6b4a"), (8, "shared prefix", "#ffcb7d"),
          (9, "offloaded", "#5794f2"), (10, "dropped", "#f2495c"), (11, "reserved", "#9fa7b3"),
          (12, "error", "#c4162a")]
POOL_COLOURS = {"active": "#73bf69", "tool_wait": "#b877d9", "stale": "#8f6b4a", "shared_prefix": "#ffcb7d"}


def target(expr: str, legend: str, ref: str = "A") -> dict:
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "range": True}


def ts_panel(title: str, targets: list[dict], x: int, y: int, w: int, h: int, unit: str = "short",
             stack: bool = False, fill: int = 10, overrides: list | None = None, desc: str = "") -> dict:
    return {
        "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
        "fieldConfig": {
            "defaults": {"unit": unit, "min": 0, "custom": {
                "drawStyle": "line", "lineWidth": 1, "fillOpacity": fill, "showPoints": "never",
                "lineInterpolation": "stepAfter",
                "stacking": {"mode": "normal" if stack else "none", "group": "A"}}},
            "overrides": overrides or []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }


def stat(title: str, expr: str, x: int, y: int, w: int, unit: str = "short", decimals: int | None = None,
         text_mode: str = "value", display: str = "", colour: str = "blue") -> dict:
    d = {"unit": unit, "color": {"mode": "fixed", "fixedColor": colour}}
    if decimals is not None:
        d["decimals"] = decimals
    if display:
        d["displayName"] = display
    return {"type": "stat", "title": title, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": 3},
            "targets": [{**target(expr, ""), "instant": True, "range": False}],
            "fieldConfig": {"defaults": d, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "textMode": text_mode, "colorMode": "value", "graphMode": "none"}}


def colour_override(name: str, colour: str) -> dict:
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": colour}}]}


def panels() -> list[dict]:
    row = {"type": "row", "title": "$run", "repeat": "run", "collapsed": False,
           "gridPos": {"x": 0, "y": 0, "w": 24, "h": 1}, "panels": []}
    y = 1
    p = [
        stat("Run", f"edgeserve_run_info{{{R}}}", 0, y, 6, text_mode="name",
             display="${__field.labels.policy} · K=${__field.labels.k} · ${__field.labels.workload}"),
        stat("Pool budget", f"edgeserve_pool_budget_tokens{{{R}}}", 6, y, 3, colour="purple"),
        stat("Elapsed", f"edgeserve_run_elapsed_seconds{{{R}}}", 9, y, 3, unit="s", decimals=0, colour="text"),
        stat("Tasks done", f"edgeserve_tasks_done_total{{{R}}}", 12, y, 3, colour="green"),
        stat("Correct (EM)", f"edgeserve_tasks_em_total{{{R}}}", 15, y, 3, colour="green"),
        stat("Mean F1", f"edgeserve_tasks_f1_total{{{R}}} / clamp_min(edgeserve_tasks_done_total{{{R}}}, 1)",
             18, y, 3, decimals=3, colour="green"),
        stat("Failed tasks", f"edgeserve_tasks_failed_total{{{R}}}", 21, y, 3, colour="red"),
    ]
    y += 3
    hq = lambda q, h: f"histogram_quantile({q}, {h}_bucket{{{R}}})"   # noqa: E731  (whole run so far)
    el = f"clamp_min(edgeserve_run_elapsed_seconds{{{R}}}, 1)"
    p += [
        stat("Goodput (avg/min)", f"60 * edgeserve_tasks_em_total{{{R}}} / {el}", 0, y, 3, decimals=1, colour="green"),
        stat("Throughput (avg/min)", f"60 * edgeserve_tasks_done_total{{{R}}} / {el}", 3, y, 3, decimals=1),
        stat("Task latency p95", hq(0.95, "edgeserve_task_latency_seconds"), 6, y, 3, unit="s", decimals=1,
             colour="orange"),
        stat("Service time p95", hq(0.95, "edgeserve_task_service_seconds"), 9, y, 3, unit="s", decimals=1,
             colour="orange"),
        stat("TTFT p50", hq(0.5, "edgeserve_request_ttft_seconds"), 12, y, 3, unit="s", decimals=2, colour="yellow"),
        stat("TTFT p95", hq(0.95, "edgeserve_request_ttft_seconds"), 15, y, 3, unit="s", decimals=2, colour="yellow"),
        stat("TPOT p50", hq(0.5, "edgeserve_request_tpot_ms"), 18, y, 3, unit="ms", decimals=1, colour="purple"),
        stat("TPOT p95", hq(0.95, "edgeserve_request_tpot_ms"), 21, y, 3, unit="ms", decimals=1, colour="purple"),
    ]
    y += 3
    budget = {"matcher": {"id": "byName", "options": "budget"}, "properties": [
        {"id": "custom.stacking", "value": {"mode": "none"}}, {"id": "custom.fillOpacity", "value": 0},
        {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 10]}}, {"id": "custom.lineWidth", "value": 2},
        {"id": "color", "value": {"mode": "fixed", "fixedColor": "#f2495c"}}]}
    p += [
        ts_panel("KV pool occupancy (tokens)", [
            target(f"edgeserve_pool_kv_tokens{{{R}}}", "{{state}}"),
            target(f"edgeserve_pool_budget_tokens{{{R}}}", "budget", "B")],
            0, y, 12, 9, stack=True, fill=60,
            overrides=[budget] + [colour_override(s, c) for s, c in POOL_COLOURS.items()],
            desc="KV tokens held in the pool, by what their owner is doing: active (prefill/decode or about "
                 "to send), tool_wait (agent waiting on a tool), stale (task finished, KV not yet reused), "
                 "shared_prefix (forked system prompt). Dashed line: pool size (-c). Above it = overflow."),
        ts_panel("KV tokens per slot", [target(f"edgeserve_slot_kv_tokens{{{R}}}", "slot {{slot}}")],
                 12, y, 12, 9, stack=True, fill=40,
                 desc="Each slot's KV tokens, stacked. Without a KV manager these are rebuilt from per-request "
                      "token counts; llama-server purges are not visible."),
    ]
    y += 9
    mappings = [{"type": "value", "options": {str(c): {"text": n, "color": col, "index": i}
                                              for i, (c, n, col) in enumerate(STATES)}}]
    p.append({
        "type": "state-timeline", "title": "Slot timeline", "datasource": DS,
        "description": "One row per slot. prefill / decode split only in replay (live shows 'active').",
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 11},
        "targets": [target(f"edgeserve_slot_state{{{R}}}", "slot {{slot}}")],
        # colour mode must be "fixed": with "thresholds" the state timeline ignores the mapping colours
        "fieldConfig": {"defaults": {"mappings": mappings, "color": {"mode": "fixed", "fixedColor": "#3a3a3a"},
                                     "custom": {"lineWidth": 0, "fillOpacity": 85}}, "overrides": []},
        "options": {"mergeValues": True, "showValue": "never", "rowHeight": 0.85, "alignValue": "left",
                    "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "single"}},
    })
    y += 11
    p += [
        ts_panel("Tasks", [
            target(f"edgeserve_tasks{{{R}}}", "{{phase}}"),
            target(f"edgeserve_requests_inflight{{{R}}}", "LLM requests in flight", "B")], 0, y, 8, 8,
            desc="queued = arrived but not admitted; running = admitted, not finished."),
        ts_panel("Throughput and goodput (tasks/min, 30 s window)", [
            target(f"rate(edgeserve_tasks_done_total{{{R}}}[30s]) * 60", "throughput"),
            target(f"rate(edgeserve_tasks_em_total{{{R}}}[30s]) * 60", "goodput (EM-correct)", "B")],
            8, y, 8, 8, overrides=[colour_override("throughput", "#5794f2"),
                                   colour_override("goodput (EM-correct)", "#73bf69")]),
        ts_panel("Failures and slot operations (cumulative)", [
            target(f"edgeserve_requests_failed_total{{{R}}}", "failed requests"),
            target(f"edgeserve_tasks_failed_total{{{R}}}", "failed tasks", "B"),
            target(f"edgeserve_slot_ops_total{{{R}}}", "slot {{op}}", "C"),
            target(f"edgeserve_compactions_total{{{R}}}", "compactions", "D")], 16, y, 8, 8, fill=0),
    ]
    y += 8
    hr = lambda q, h: f"histogram_quantile({q}, rate({h}_bucket{{{R}}}[30s]))"   # noqa: E731  (30 s window)
    p += [
        ts_panel("Task latency (s, 30 s window)", [
            target(hr(0.5, "edgeserve_task_latency_seconds"), "latency p50"),
            target(hr(0.95, "edgeserve_task_latency_seconds"), "latency p95", "B"),
            target(hr(0.5, "edgeserve_task_service_seconds"), "service p50", "C"),
            target(hr(0.95, "edgeserve_task_service_seconds"), "service p95", "D")], 0, y, 8, 8, unit="s", fill=0,
            desc="Latency = arrival to done (includes queueing); service = admission to done. Quantiles are "
                 "interpolated inside histogram buckets."),
        ts_panel("TTFT (s, 30 s window)", [
            target(hr(0.5, "edgeserve_request_ttft_seconds"), "p50"),
            target(hr(0.95, "edgeserve_request_ttft_seconds"), "p95", "B")], 8, y, 8, 8, unit="s", fill=0,
            desc="Time to first token per LLM request: waiting in the server + prefill."),
        ts_panel("TPOT (ms per output token, 30 s window)", [
            target(hr(0.5, "edgeserve_request_tpot_ms"), "p50"),
            target(hr(0.95, "edgeserve_request_tpot_ms"), "p95", "B")], 16, y, 8, 8, unit="ms", fill=0,
            desc="Decode time per generated token. Grows with the number of decoding slots and with the "
                 "unified-pool attention span."),
    ]
    y += 8
    p += [
        ts_panel("GPU memory and server RSS (MiB)", [
            target(f"edgeserve_gpu_mem_mib{{{R}}}", "GPU memory used"),
            target(f"edgeserve_server_rss_mib{{{R}}}", "llama-server RSS", "B")], 0, y, 8, 8, unit="decmbytes",
            desc="GPU memory is fixed at startup (weights + preallocated KV pool), so this line is flat; "
                 "what changes is how the pool is used (panels above)."),
        ts_panel("GPU utilisation (%)", [target(f"edgeserve_gpu_util_percent{{{R}}}", "GPU util")],
                 8, y, 8, 8, unit="percent"),
        ts_panel("Server token rates (tokens/s, 10 s window)", [
            target(f"rate(edgeserve_llama_prompt_tokens_total{{{R}}}[10s])", "prefill"),
            target(f"rate(edgeserve_llama_tokens_predicted_total{{{R}}}[10s])", "decode", "B"),
            target(f"rate(edgeserve_llama_prompt_tokens_cached_total{{{R}}}[10s])", "prefix-cache reuse", "C")],
            16, y, 8, 8, fill=0),
    ]
    return [row] + p


def dashboard(uid: str, title: str, live: bool) -> dict:
    d = {
        "uid": uid, "title": title, "tags": ["edgeserve"], "schemaVersion": 39, "version": 1,
        "editable": True, "graphTooltip": 1,
        "timezone": "browser" if live else "utc",
        "time": {"from": "now-10m", "to": "now"} if live else
                {"from": "2026-01-01T00:00:00.000Z", "to": "2026-01-01T00:08:00.000Z"},
        "refresh": "2s" if live else "",
        "timepicker": {"refresh_intervals": ["1s", "2s", "5s", "10s", "30s"]},
        "templating": {"list": [{
            "name": "run", "label": "run", "type": "query", "datasource": DS,
            "query": {"query": "label_values(edgeserve_run_info, run)", "refId": "run"},
            "definition": "label_values(edgeserve_run_info, run)",
            "refresh": 2, "multi": True, "includeAll": True, "sort": 1,
            "current": {"selected": True, "text": ["All"], "value": ["$__all"]}}]},
        "annotations": {"list": []},
        "description": ("Live view of the run currently being written to results/ (exporter.py live)." if live else
                        "Replayed runs (exporter.py replay). Time axis = time since the run started, "
                        "shown from 00:00 (2026-01-01 UTC anchor). Pick runs with the $run variable."),
        "panels": panels(),
    }
    for i, pnl in enumerate(d["panels"], start=1):   # stable ids (needed for single-panel links/renders)
        pnl["id"] = i
    if not live:   # replay data has 0.25 s resolution; let panels use it
        for pnl in d["panels"]:
            if pnl["type"] in ("timeseries", "state-timeline"):
                pnl["interval"] = "250ms"
    return d


def main() -> None:
    out = HERE / "grafana" / "dashboards"
    out.mkdir(parents=True, exist_ok=True)
    for uid, title, live in (("edgeserve-live", "EdgeServe live", True),
                             ("edgeserve-replay", "EdgeServe replay", False)):
        path = out / f"{uid.replace('-', '_')}.json"
        path.write_text(json.dumps(dashboard(uid, title, live), indent=1) + "\n", encoding="utf-8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()
