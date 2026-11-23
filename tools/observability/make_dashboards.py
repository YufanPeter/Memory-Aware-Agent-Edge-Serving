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


def axis_right(name: str, unit: str) -> dict:
    """Override: draw series `name` on a right-hand axis with its own unit."""
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "custom.axisPlacement", "value": "right"}, {"id": "unit", "value": unit}]}


def main_panels(y: int) -> list[dict]:
    """What the project is judged on, per run: results, guardrails, and the few signals that explain them."""
    hq = lambda q, h: f"histogram_quantile({q}, {h}_bucket{{{R}}})"   # noqa: E731  (whole run so far)
    hr = lambda q, h: f"histogram_quantile({q}, rate({h}_bucket{{{R}}}[30s]))"   # noqa: E731  (30 s window)
    el = f"clamp_min(edgeserve_run_elapsed_seconds{{{R}}}, 1)"
    done_failed = stat("完成 / 失败任务", f"edgeserve_tasks_done_total{{{R}}}", 21, y, 3, colour="green")
    done_failed["targets"] = [
        {**target(f"edgeserve_tasks_done_total{{{R}}} - edgeserve_tasks_failed_total{{{R}}}", "完成"),
         "instant": True, "range": False},
        {**target(f"edgeserve_tasks_failed_total{{{R}}}", "失败", "B"), "instant": True, "range": False}]
    done_failed["fieldConfig"]["overrides"] = [colour_override("失败", "#f2495c")]
    done_failed["options"]["textMode"] = "value_and_name"
    done_failed["description"] = "约束：失败（overflow、超时等）必须为 0 或接近 0，不能靠丢任务换速度。"
    p = [
        stat("配置", f"edgeserve_run_info{{{R}}}", 0, y, 6, text_mode="name",
             display="${__field.labels.policy} · K=${__field.labels.k} · ${__field.labels.workload}"),
        stat("Goodput", f"60 * edgeserve_tasks_em_total{{{R}}} / {el}", 6, y, 3, decimals=1,
             colour="green"),
        stat("任务延迟 p95", hq(0.95, "edgeserve_task_latency_seconds"), 9, y, 3, unit="s", decimals=1,
             colour="orange"),
        stat("任务延迟 p99", hq(0.99, "edgeserve_task_latency_seconds"), 12, y, 3, unit="s", decimals=1,
             colour="orange"),
        stat("F1", f"edgeserve_tasks_f1_total{{{R}}} / clamp_min(edgeserve_tasks_done_total{{{R}}}, 1)",
             15, y, 3, decimals=3, colour="green"),
        stat("吞吐", f"60 * edgeserve_tasks_done_total{{{R}}} / {el}", 18, y, 3, decimals=1),
        done_failed,
    ]
    p[1]["description"] = "主指标（答对/分钟）：每分钟答对（EM）的任务数，按整次运行平均。项目的目标就是在固定显存下提高它。"
    p[2]["description"] = "主指标：任务从到达到完成的时间（含排队），95% 的任务不超过这个值。"
    p[3]["description"] = "主指标：尾延迟，99% 的任务不超过这个值。"
    p[4]["description"] = "约束：平均 F1。和对照配置相比变化在 ±0.03 内算持平。"
    p[5]["description"] = "任务/分钟：每分钟完成的任务数（不论对错），帮助区分'做得少'和'做得错'。"
    y += 3
    budget = {"matcher": {"id": "byName", "options": "预算（-c）"}, "properties": [
        {"id": "custom.stacking", "value": {"mode": "none"}}, {"id": "custom.fillOpacity", "value": 0},
        {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 10]}}, {"id": "custom.lineWidth", "value": 2},
        {"id": "color", "value": {"mode": "fixed", "fixedColor": "#f2495c"}}]}
    p += [
        ts_panel("KV pool 占用 vs 预算（token）", [
            target(f"edgeserve_pool_kv_tokens{{{R}}}", "{{state}}"),
            target(f"edgeserve_pool_budget_tokens{{{R}}}", "预算（-c）", "B")],
            0, y, 12, 9, stack=True, fill=60,
            overrides=[budget] + [colour_override(s, c) for s, c in POOL_COLOURS.items()],
            desc="memory-aware 的核心视图：KV pool 里的 token 按用途堆叠。active = 正在生成或马上要发请求；"
                 "tool_wait = agent 在等 tool，KV 闲置；stale = 任务已结束、KV 还留着；shared_prefix = 共享的 "
                 "system prompt。红色虚线是 pool 大小，超过它就会 overflow。selective KV 删掉的部分会让这里变低。"),
        ts_panel("Goodput 与吞吐随时间（任务/分钟，30 秒窗口）", [
            target(f"rate(edgeserve_tasks_em_total{{{R}}}[30s]) * 60", "goodput（答对）"),
            target(f"rate(edgeserve_tasks_done_total{{{R}}}[30s]) * 60", "吞吐（完成）", "B")],
            12, y, 12, 9, overrides=[colour_override("goodput（答对）", "#73bf69"),
                                     colour_override("吞吐（完成）", "#5794f2")],
            desc="主指标随时间的变化：负载忙的时段 goodput 能不能跟上，闲的时段是否稳定。"),
    ]
    y += 9
    p += [
        ts_panel("任务延迟 p95 / p99（秒）与排队任务数", [
            target(hr(0.95, "edgeserve_task_latency_seconds"), "延迟 p95"),
            target(hr(0.99, "edgeserve_task_latency_seconds"), "延迟 p99", "B"),
            target(f"edgeserve_tasks{{{R}}}", "{{phase}} 任务数", "C")], 0, y, 8, 8, unit="s", fill=0,
            overrides=[axis_right("queued 任务数", "short"), axis_right("running 任务数", "short")],
            desc="延迟（左轴，30 秒窗口，含排队）和任务数（右轴）：queued = 到达了但还没被准入，running = 正在执行。"
                 "延迟升高时看排队数，可以判断时间是花在等待准入还是执行上。"),
        ts_panel("TTFT p95（秒，左）/ TPOT p95（毫秒，右）", [
            target(hr(0.95, "edgeserve_request_ttft_seconds"), "TTFT p95"),
            target(hr(0.95, "edgeserve_request_tpot_ms"), "TPOT p95", "B")], 8, y, 8, 8, unit="s", fill=0,
            overrides=[axis_right("TPOT p95", "ms")],
            desc="解释延迟从哪来：TTFT = 请求在服务器排队 + prefill 的时间，重算多时会升高；TPOT = 每生成一个 "
                 "token 的时间，同时 decode 的 agent 多、KV 跨度长时会升高。"),
        ts_panel("selective KV：删掉的 KV vs 新做的 prefill（token/秒）", [
            target(f"rate(edgeserve_trimmed_tokens_total{{{R}}}[30s])", "删掉的 KV"),
            target(f"rate(edgeserve_llama_prompt_tokens_total{{{R}}}[30s])", "prefill（新内容 + 重算）", "B")],
            16, y, 8, 8, fill=0, overrides=[colour_override("删掉的 KV", "#ff9830"),
                                             colour_override("prefill（新内容 + 重算）", "#5794f2")],
            desc="selective KV 的作用：橙线是从 KV 里直接删掉的 token；蓝线是服务器实际做的 prefill。"
                 "在 KV 里删（kv）不会引起重算；只改文本（text）删掉的内容不会出现在橙线上，但蓝线会更高。"),
    ]
    return p


def detail_panels(y: int) -> list[dict]:
    """Diagnostics, folded away by default: slots, GPU, server rates, slot operations."""
    hr = lambda q, h: f"histogram_quantile({q}, rate({h}_bucket{{{R}}}[30s]))"   # noqa: E731
    p = [
        stat("Pool 大小", f"edgeserve_pool_budget_tokens{{{R}}}", 0, y, 4, colour="purple"),
        stat("已运行", f"edgeserve_run_elapsed_seconds{{{R}}}", 4, y, 4, unit="s", decimals=0, colour="text"),
        stat("答对（EM）", f"edgeserve_tasks_em_total{{{R}}}", 8, y, 4, colour="green"),
        stat("服务时间 p95", f"histogram_quantile(0.95, edgeserve_task_service_seconds_bucket{{{R}}})",
             12, y, 4, unit="s", decimals=1, colour="orange"),
        stat("TTFT p50", f"histogram_quantile(0.5, edgeserve_request_ttft_seconds_bucket{{{R}}})",
             16, y, 4, unit="s", decimals=2, colour="yellow"),
        stat("TPOT p50", f"histogram_quantile(0.5, edgeserve_request_tpot_ms_bucket{{{R}}})",
             20, y, 4, unit="ms", decimals=1, colour="purple"),
    ]
    y += 3
    mappings = [{"type": "value", "options": {str(c): {"text": n, "color": col, "index": i}
                                              for i, (c, n, col) in enumerate(STATES)}}]
    p.append({
        "type": "state-timeline", "title": "Slot 时间线", "datasource": DS,
        "description": "每个 slot 一行。prefill / decode 只在回放里区分（实时视图显示 active）。",
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
        ts_panel("每个 slot 的 KV（token）", [target(f"edgeserve_slot_kv_tokens{{{R}}}", "slot {{slot}}")],
                 0, y, 12, 8, stack=True, fill=40,
                 desc="不带 KV 管理器时由每次请求的 token 数重建，看不到服务器自己的清理。"),
        ts_panel("失败、slot 操作与删除次数（累计）", [
            target(f"edgeserve_requests_failed_total{{{R}}}", "失败的请求"),
            target(f"edgeserve_tasks_failed_total{{{R}}}", "失败的任务", "B"),
            target(f"edgeserve_slot_ops_total{{{R}}}", "slot {{op}}", "C"),
            target(f"edgeserve_trims_total{{{R}}}", "selective KV 删除次数", "D")], 12, y, 12, 8, fill=0),
    ]
    y += 8
    p += [
        ts_panel("延迟细节（秒，30 秒窗口）", [
            target(hr(0.5, "edgeserve_task_latency_seconds"), "任务延迟 p50"),
            target(hr(0.5, "edgeserve_task_service_seconds"), "服务时间 p50", "B"),
            target(hr(0.95, "edgeserve_task_service_seconds"), "服务时间 p95", "C"),
            target(hr(0.5, "edgeserve_request_ttft_seconds"), "TTFT p50", "D")], 0, y, 8, 8, unit="s", fill=0,
            desc="服务时间 = 从准入到完成（不含排队）。"),
        ts_panel("GPU 显存与服务器内存（MiB）", [
            target(f"edgeserve_gpu_mem_mib{{{R}}}", "GPU 显存"),
            target(f"edgeserve_server_rss_mib{{{R}}}", "llama-server 内存", "B")], 8, y, 8, 8, unit="decmbytes",
            desc="GPU 显存在启动时就分配好了（权重 + 整个 KV pool），所以这条线是平的；真正变化的是 pool 内部的占用。"),
        ts_panel("GPU 利用率（%）与服务器 token 速率", [
            target(f"edgeserve_gpu_util_percent{{{R}}}", "GPU 利用率"),
            target(f"rate(edgeserve_llama_tokens_predicted_total{{{R}}}[10s])", "decode token/s", "B"),
            target(f"rate(edgeserve_llama_prompt_tokens_cached_total{{{R}}}[10s])", "缓存复用 token/s", "C")],
            16, y, 8, 8, unit="percent", fill=0,
            overrides=[axis_right("decode token/s", "short"), axis_right("缓存复用 token/s", "short")]),
    ]
    return p


def panels() -> list[dict]:
    """Per run: a main row (expanded) with what matters, then a folded row with the diagnostics."""
    main = {"type": "row", "title": "$run", "repeat": "run", "collapsed": False,
            "gridPos": {"x": 0, "y": 0, "w": 24, "h": 1}, "panels": []}
    body = main_panels(1)
    y = max(pn["gridPos"]["y"] + pn["gridPos"]["h"] for pn in body)
    details = {"type": "row", "title": "细节 · $run（点开查看 slot、GPU、服务器速率）", "repeat": "run",
               "collapsed": True, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
               "panels": detail_panels(y + 1)}
    return [main] + body + [details]


def run_variable(live: bool) -> dict:
    """The $run variable; each selected run gets its own row.

    Live: only runs the exporter reports right now (an instant query; a run's series go stale as
    soon as the exporter moves to the next run), so the dashboard shows the current run, not every
    run Prometheus has seen. Replay: every run Prometheus holds, to pick from."""
    if live:
        q, regex = "query_result(edgeserve_run_info)", '/run="([^"]+)"/'
    else:
        q, regex = "label_values(edgeserve_run_info, run)", ""
    return {"name": "run", "label": "run", "type": "query", "datasource": DS,
            "query": {"query": q, "refId": "run"}, "definition": q, "regex": regex,
            "refresh": 2, "multi": True, "includeAll": True, "sort": 1,
            "current": {"selected": True, "text": ["All"], "value": ["$__all"]}}


def dashboard(uid: str, title: str, live: bool) -> dict:
    d = {
        "uid": uid, "title": title, "tags": ["edgeserve"], "schemaVersion": 39, "version": 1,
        "editable": True, "graphTooltip": 1,
        "timezone": "browser" if live else "utc",
        "time": {"from": "now-10m", "to": "now"} if live else
                {"from": "2026-01-01T00:00:00.000Z", "to": "2026-01-01T00:08:00.000Z"},
        "refresh": "2s" if live else "",
        "timepicker": {"refresh_intervals": ["1s", "2s", "5s", "10s", "30s"]},
        "templating": {"list": [run_variable(live)]},
        "annotations": {"list": []},
        "description": ("Live view of the run currently being written to results/ (exporter.py live)." if live else
                        "Replayed runs (exporter.py replay). Time axis = time since the run started, "
                        "shown from 00:00 (2026-01-01 UTC anchor). Pick runs with the $run variable."),
        "panels": panels(),
    }
    flat = [q for pnl in d["panels"] for q in [pnl] + pnl.get("panels", [])]   # incl. panels in folded rows
    for i, pnl in enumerate(flat, start=1):   # stable ids (needed for single-panel links/renders)
        pnl["id"] = i
    if not live:   # replay data has 0.25 s resolution; let panels use it
        for pnl in flat:
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
