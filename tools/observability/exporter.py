"""Prometheus exporter for EdgeServe runs: KV pool, slots, tasks, GPU and server, from events.jsonl.

It only reads the files the runner writes (events.jsonl, server_cmd.txt, meta.json); it never
talks to llama-server, so watching a run does not change it.

Live: follow the newest results/**/events.jsonl and serve its current state on :9101/metrics
(Prometheus scrapes it every second):

    uv run python tools/observability/exporter.py live [--results results] [--port 9101]

Replay: turn finished runs into Prometheus history. Every replayed run is placed at the same
anchor time (2026-01-01 00:00:00 UTC + seconds since the run started), so in the Grafana
"EdgeServe replay" dashboard all runs share one time axis and can be compared side by side:

    uv run python tools/observability/exporter.py replay results/p2a_stage1/l0/rep0 results/p2a_stage1/l3/rep0

Run state is rebuilt from events, as in analysis/kv_timeline.py:
- runs without a KV manager (fixed_k etc.): a slot holds prompt_n + cache_n + predicted_n tokens
  after each request; between requests it is in a tool call; after task_done it is stale
  (llama-server keeps it until the slot's next task). Server-side purges are not visible.
- managed runs: the KV manager's own kv_state events give each slot's state and tokens.
Replay also splits each request into prefill and decode (t_first_token); live mode only knows
"active" because the client logs a request once it ends.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ANCHOR = 1767225600.0          # 2026-01-01 00:00:00 UTC: t = 0 of every replayed run
REPLAY_STEP = 0.25             # seconds between replay snapshots
PROM_IMAGE = "prom/prometheus:v3.1.0"
HERE = Path(__file__).resolve().parent

# Slot state codes (the Grafana state-timeline panel maps them to names and colours).
STATE = {"empty": 0, "prefill": 1, "decode": 2, "active": 3, "ready": 4, "tool_wait": 5,
         "gate_wait": 6, "stale": 7, "golden": 8, "offloaded": 9, "dropped": 10, "reserved": 11,
         "error": 12}
KVM_STATE = {"free": "empty", "reserved": "reserved", "ready": "ready", "active": "active",
             "idle": "tool_wait", "waiting": "gate_wait", "stale": "stale", "golden": "golden",
             "offloaded": "offloaded", "dropped": "dropped"}
# Which pool share a slot's tokens count toward.
POOL_OF = {"prefill": "active", "decode": "active", "active": "active", "ready": "active",
           "gate_wait": "active", "reserved": "active", "tool_wait": "tool_wait", "stale": "stale",
           "golden": "shared_prefix"}
POOL_STATES = ("active", "tool_wait", "stale", "shared_prefix")
SAMPLE_GAUGES = {"gpu_mem_mib": "edgeserve_gpu_mem_mib", "gpu_util": "edgeserve_gpu_util_percent",
                 "server_rss_mib": "edgeserve_server_rss_mib",
                 "requests_processing": "edgeserve_llama_requests_processing",
                 "requests_deferred": "edgeserve_llama_requests_deferred",
                 "n_busy_slots_per_decode": "edgeserve_llama_busy_slots_per_decode"}
SAMPLE_COUNTERS = {"prompt_tokens_total": "edgeserve_llama_prompt_tokens_total",
                   "prompt_tokens_cached_total": "edgeserve_llama_prompt_tokens_cached_total",
                   "tokens_predicted_total": "edgeserve_llama_tokens_predicted_total",
                   "n_decode_total": "edgeserve_llama_decode_calls_total"}
HELP = {
    "edgeserve_run_info": ("gauge", "Run metadata (labels); value 1"),
    "edgeserve_run_elapsed_seconds": ("gauge", "Seconds since the run started"),
    "edgeserve_pool_budget_tokens": ("gauge", "KV pool size (-c) in tokens"),
    "edgeserve_pool_kv_tokens": ("gauge", "KV tokens held, by what their owner is doing"),
    "edgeserve_pool_resident_tokens": ("gauge", "KV tokens resident in the pool (shared prefix counted once)"),
    "edgeserve_pool_free_tokens": ("gauge", "Budget minus resident tokens"),
    "edgeserve_slot_state": ("gauge", "Slot state code (see STATE in exporter.py)"),
    "edgeserve_slot_kv_tokens": ("gauge", "KV tokens held by the slot"),
    "edgeserve_tasks": ("gauge", "Tasks queued (arrived, not admitted) or running"),
    "edgeserve_tasks_arrived_total": ("counter", "Tasks arrived"),
    "edgeserve_tasks_done_total": ("counter", "Tasks finished (any status)"),
    "edgeserve_tasks_failed_total": ("counter", "Tasks that ended without an answer"),
    "edgeserve_tasks_em_total": ("counter", "Exact-match correct tasks (goodput numerator)"),
    "edgeserve_tasks_f1_total": ("counter", "Sum of task F1"),
    "edgeserve_requests_inflight": ("gauge", "LLM requests in flight"),
    "edgeserve_requests_total": ("counter", "LLM requests finished"),
    "edgeserve_requests_failed_total": ("counter", "LLM requests that failed (e.g. context overflow)"),
    "edgeserve_slot_ops_total": ("counter", "Slot save/restore/erase/fork operations"),
    "edgeserve_trims_total": ("counter", "Selective KV trims (observations processed during tool calls)"),
    "edgeserve_trimmed_tokens_total": ("counter", "Tokens deleted from slots by selective KV (in place)"),
}
# Latency histograms: Grafana takes p50/p95 with histogram_quantile().
HISTS = {
    "edgeserve_request_ttft_seconds": ("Time to first token per LLM request (queueing in the server + prefill)",
                                       (0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 20)),
    "edgeserve_request_tpot_ms": ("Decode time per output token (server timings)",
                                  (10, 12.5, 15, 17.5, 20, 25, 30, 40, 50, 75, 100, 150, 250, 500)),
    "edgeserve_task_latency_seconds": ("Task latency, arrival to done (includes queueing)",
                                       (1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600)),
    "edgeserve_task_service_seconds": ("Task service time, admission to done",
                                       (1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600)),
}
for _name, (_help, _) in HISTS.items():
    HELP[_name] = ("histogram", _help)
for _name in list(SAMPLE_GAUGES.values()):
    HELP[_name] = ("gauge", "Runner sample (nvidia-smi, process RSS or llama-server /metrics)")
for _name in list(SAMPLE_COUNTERS.values()):
    HELP[_name] = ("counter", "llama-server /metrics counter")


def server_ctx(run: Path) -> int | None:
    f = run / "server_cmd.txt"
    if not f.exists():
        return None
    args = shlex.split(f.read_text(encoding="utf-8"))
    for flag in ("-c", "--ctx-size"):
        if flag in args:
            return int(args[args.index(flag) + 1])
    return None


def runner_args(run: Path) -> dict[str, str]:
    """Options of the edgeserve.runner process writing to `run` (Linux /proc), {} if none."""
    target = run.resolve()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            cwd = (proc / "cwd").resolve()
        except OSError:
            continue
        if "edgeserve.runner" not in argv or "--out" not in argv:
            continue
        out = Path(argv[argv.index("--out") + 1])
        if (out if out.is_absolute() else cwd / out).resolve() == target:
            return {a: b for a, b in zip(argv, argv[1:]) if a.startswith("--") and not b.startswith("--")}
    return {}


def run_label(run: Path, results: Path | None = None) -> str:
    try:
        return str(run.resolve().relative_to((results or Path("results")).resolve()))
    except ValueError:
        return run.name


class RunState:
    """State of one run, updated event by event."""

    def __init__(self, run: Path, label: str) -> None:
        self.label, self.run = label, run
        self.load_meta()
        self.managed = False          # becomes True on the first kv_state event
        self.t = 0.0
        self.slot_state: dict[int, str] = {}
        self.slot_tokens: dict[int, int] = defaultdict(int)
        self.slot_shared: dict[int, int] | None = None   # per-slot shared-prefix tokens (kv_state "shared")
        self.slot_task: dict[int, str] = {}
        self.task_slot: dict[str, int] = {}
        self.resident: int | None = None
        self.queued = self.running = self.inflight = 0
        self.counters: dict[str, float] = defaultdict(float)
        self.slot_ops: dict[str, int] = defaultdict(int)
        self.sample: dict[str, float] = {}
        # histogram state: cumulative bucket counts, sum, count
        self.hist: dict[str, list[float]] = {h: [0.0] * (len(b) + 2) for h, (_, b) in HISTS.items()}
        self.t_arrive: dict[str, float] = {}
        self.t_admit: dict[str, float] = {}

    def load_meta(self) -> None:
        """Run labels from meta.json (the runner writes it when the run ends), else from the
        command line of the runner process still writing this run; pool size from server_cmd.txt."""
        f = self.run / "meta.json"
        meta = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if meta:
            pol = (meta.get("policy") or {}).get("name", "?")
            sel = meta.get("selective") or {}
            k, workload = meta.get("k", "?"), meta.get("workload", "?")
            sel_tag = f"{sel['mechanism']}:{sel['scorer']}" if sel else ""
        else:
            a = runner_args(self.run)
            pol, k, workload = a.get("--policy", "fixed_k" if a else "?"), a.get("--k", "?"), a.get("--workload", "?")
            sel_tag = f"{a['--selective']}:{a.get('--sel-scorer', 'oracle')}" if a.get("--selective", "none") != "none" else ""
        self.info = {"run": self.label, "policy": str(pol) + (f"+{sel_tag}" if sel_tag else ""), "k": str(k),
                     "workload": Path(str(workload)).stem}
        self.budget = server_ctx(self.run) or meta.get("pool_tokens") or 0

    # --- events -------------------------------------------------------------------------------
    def feed(self, ev: dict[str, Any]) -> None:
        self.t = max(self.t, float(ev.get("t", 0.0)))
        kind = ev["type"]
        slot = ev.get("slot")
        slot = int(slot) if slot is not None else self.task_slot.get(ev.get("task_id", ""))
        if kind == "task_arrive":
            self.t_arrive[ev["task_id"]] = ev["t"]
            self.queued += 1
            self.counters["arrived"] += 1
        elif kind == "task_admit":
            self.t_admit[ev["task_id"]] = ev["t"]
            self.queued -= 1
            self.running += 1
            if slot is not None:
                self.task_slot[ev["task_id"]] = slot
                self.slot_task[slot] = ev["task_id"]
                if not self.managed:
                    self._state(slot, "ready")
        elif kind == "req_start":
            self.inflight += 1
            if slot is not None and not self.managed:
                self._state(slot, "active")
        elif kind == "req_first_token":          # synthetic, replay only
            if slot is not None and self.slot_state.get(slot) in ("active", "prefill"):
                self._state(slot, "decode")
        elif kind == "req_prefill":              # synthetic, replay only
            if slot is not None and self.slot_state.get(slot) == "active":
                self._state(slot, "prefill")
        elif kind == "req_end":
            self.inflight -= 1
            self.counters["requests"] += 1
            if not ev.get("ok"):
                self.counters["requests_failed"] += 1
                if slot is not None and not self.managed:
                    self._state(slot, "error")
            else:
                if ev.get("ttft_s") is not None:
                    self.observe("edgeserve_request_ttft_seconds", float(ev["ttft_s"]))
                if ev.get("predicted_ms") is not None and (ev.get("predicted_n") or 0) > 0:
                    self.observe("edgeserve_request_tpot_ms", ev["predicted_ms"] / ev["predicted_n"])
            if ev.get("ok") and slot is not None:
                if not self.managed:
                    self.slot_tokens[slot] = int(ev["prompt_n"]) + int(ev["cache_n"]) + int(ev["predicted_n"])
                    self._state(slot, "ready")
                elif self.slot_state.get(slot) in ("prefill", "decode"):
                    self._state(slot, "active")
        elif kind == "tool_start":
            if slot is not None and not self.managed:
                self._state(slot, "tool_wait")
        elif kind == "tool_end":
            if slot is not None and not self.managed:
                self._state(slot, "ready")
        elif kind == "task_done":
            self.running -= 1
            self.counters["done"] += 1
            tid = ev.get("task_id", "")
            if tid in self.t_arrive:
                self.observe("edgeserve_task_latency_seconds", ev["t"] - self.t_arrive[tid])
            if tid in self.t_admit:
                self.observe("edgeserve_task_service_seconds", ev["t"] - self.t_admit[tid])
            self.counters["em"] += float(ev.get("em") or 0.0)
            self.counters["f1"] += float(ev.get("f1") or 0.0)
            if ev.get("status") not in ("ok", "forced"):
                self.counters["failed"] += 1
            if slot is not None and not self.managed:
                self._state(slot, "stale" if self.slot_tokens[slot] else "empty")
            self.task_slot.pop(ev.get("task_id", ""), None)
        elif kind == "kv_state":
            if not self.managed:      # switch to the manager's view of every slot
                self.managed = True
            if slot is not None:
                st = KVM_STATE.get(ev["to"], "ready")
                if st == "active" and self.slot_state.get(slot) in ("prefill", "decode"):
                    st = self.slot_state[slot]
                self._state(slot, st)
                self.slot_tokens[slot] = int(ev.get("tokens") or 0) if st not in ("empty", "offloaded", "dropped") else 0
                if "shared" in ev:
                    if self.slot_shared is None:
                        self.slot_shared = defaultdict(int)
                    self.slot_shared[slot] = int(ev["shared"] or 0) if self.slot_tokens[slot] else 0
            if ev.get("resident") is not None:
                self.resident = int(ev["resident"])
        elif kind == "slot_op":
            self.slot_ops[str(ev.get("op"))] += 1
        elif kind == "trim":
            self.counters["trims"] += 1
            n = int(ev.get("tokens") or 0)
            self.counters["trimmed_tokens"] += n
            if slot is not None and not self.managed and n:     # managed runs: kv_state carries it
                self.slot_tokens[slot] = max(0, self.slot_tokens[slot] - n)
        elif kind == "sample":
            self.sample = {k: float(v) for k, v in ev.items() if isinstance(v, (int, float)) and k != "t"}

    def observe(self, name: str, v: float) -> None:
        h, bounds = self.hist[name], HISTS[name][1]
        for i, b in enumerate(bounds):
            if v <= b:
                h[i] += 1
        h[-2] += v          # sum
        h[-1] += 1          # count (= +Inf bucket)

    def _state(self, slot: int, st: str) -> None:
        self.slot_state[slot] = st

    # --- metrics ------------------------------------------------------------------------------
    def metrics(self) -> list[tuple[str, dict[str, str], float]]:
        run = {"run": self.label}
        out: list[tuple[str, dict[str, str], float]] = [
            ("edgeserve_run_info", self.info, 1.0),
            ("edgeserve_run_elapsed_seconds", run, self.t),
            ("edgeserve_pool_budget_tokens", run, float(self.budget)),
        ]
        pool = dict.fromkeys(POOL_STATES, 0.0)
        for slot, st in self.slot_state.items():
            cat = POOL_OF.get(st)
            if cat:
                own = self.slot_tokens[slot] - (self.slot_shared or {}).get(slot, 0) if cat != "shared_prefix" else self.slot_tokens[slot]
                pool[cat] += max(0, own)
        resident = self.resident if self.managed and self.resident is not None else sum(pool.values())
        excess = sum(pool.values()) - resident
        if self.managed and excess > 0:
            # Older runs log no per-slot "shared": forked slots each count the shared prefix, so the sum
            # exceeds the manager's resident total. Take the excess out of the non-prefix shares pro rata.
            rest = sum(v for k, v in pool.items() if k != "shared_prefix")
            if rest > 0:
                for k in pool:
                    if k != "shared_prefix":
                        pool[k] -= excess * pool[k] / rest
        out += [("edgeserve_pool_kv_tokens", {**run, "state": s}, round(v)) for s, v in pool.items()]
        out += [("edgeserve_pool_resident_tokens", run, float(resident)),
                ("edgeserve_pool_free_tokens", run, float(max(0, self.budget - resident)))]
        for slot in sorted(self.slot_state):
            lab = {**run, "slot": f"{slot:02d}"}
            out.append(("edgeserve_slot_state", lab, float(STATE[self.slot_state[slot]])))
            out.append(("edgeserve_slot_kv_tokens", lab, float(self.slot_tokens[slot])))
        out += [("edgeserve_tasks", {**run, "phase": "queued"}, float(self.queued)),
                ("edgeserve_tasks", {**run, "phase": "running"}, float(self.running)),
                ("edgeserve_tasks_arrived_total", run, self.counters["arrived"]),
                ("edgeserve_tasks_done_total", run, self.counters["done"]),
                ("edgeserve_tasks_failed_total", run, self.counters["failed"]),
                ("edgeserve_tasks_em_total", run, self.counters["em"]),
                ("edgeserve_tasks_f1_total", run, self.counters["f1"]),
                ("edgeserve_requests_inflight", run, float(self.inflight)),
                ("edgeserve_requests_total", run, self.counters["requests"]),
                ("edgeserve_requests_failed_total", run, self.counters["requests_failed"]),
                ("edgeserve_trims_total", run, self.counters["trims"]),
                ("edgeserve_trimmed_tokens_total", run, self.counters["trimmed_tokens"])]
        out += [("edgeserve_slot_ops_total", {**run, "op": op}, float(n)) for op, n in self.slot_ops.items()]
        for name, (_, bounds) in HISTS.items():
            h = self.hist[name]
            for i, b in enumerate(bounds):
                out.append((f"{name}_bucket", {**run, "le": f"{b:g}"}, h[i]))
            out.append((f"{name}_bucket", {**run, "le": "+Inf"}, h[-1]))
            out.append((f"{name}_sum", run, h[-2]))
            out.append((f"{name}_count", run, h[-1]))
        for src, name in {**SAMPLE_GAUGES, **SAMPLE_COUNTERS}.items():
            if src in self.sample:
                out.append((name, run, self.sample[src]))
        return out


def fmt_labels(lab: dict[str, str]) -> str:
    esc = {k: str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") for k, v in lab.items()}
    return "{" + ",".join(f'{k}="{v}"' for k, v in esc.items()) + "}"


def family(name: str) -> str:
    """Metric family of a sample name (histogram samples carry _bucket/_sum/_count)."""
    for suffix in ("_bucket", "_sum", "_count"):
        if name.endswith(suffix) and name[: -len(suffix)] in HISTS:
            return name[: -len(suffix)]
    return name


def exposition(rows: list[tuple[str, dict[str, str], float]], ts: float | None = None) -> str:
    """Prometheus text format (live) or OpenMetrics sample lines with timestamps (replay)."""
    lines, seen = [], set()
    for name, lab, val in rows:
        fam = family(name)
        if ts is None and fam not in seen:
            typ, help_ = HELP.get(fam, ("gauge", fam))
            lines += [f"# HELP {fam} {help_}", f"# TYPE {fam} {typ}"]
            seen.add(fam)
        lines.append(f"{name}{fmt_labels(lab)} {val:g}" + (f" {ts:.3f}" if ts is not None else ""))
    return "\n".join(lines) + "\n"


# --- live ---------------------------------------------------------------------------------------
class Live:
    def __init__(self, results: Path) -> None:
        self.results = results
        self.lock = threading.Lock()
        self.state: RunState | None = None
        self.path: Path | None = None
        self.offset = 0
        self.buf = ""
        self.meta_checked = 0.0
        self.meta_final = False

    def newest(self) -> Path | None:
        files = list(self.results.glob("**/events.jsonl"))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def poll(self) -> None:
        cur = self.newest()
        if cur is None:
            return
        if cur != self.path:
            with self.lock:
                self.path, self.offset, self.buf = cur, 0, ""
                self.meta_checked, self.meta_final = 0.0, False
                self.state = RunState(cur.parent, run_label(cur.parent, self.results))
            print(f"following {cur}", flush=True)
        with open(self.path, encoding="utf-8") as f:
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()
        now = time.monotonic()
        if (self.state.info["policy"] == "?" or not self.meta_final) and now - self.meta_checked > 5:
            self.meta_checked = now      # until meta.json exists: retry from the runner's command line
            self.meta_final = (self.path.parent / "meta.json").exists()
            with self.lock:
                self.state.load_meta()
        if not chunk:
            return
        self.buf += chunk
        *complete, self.buf = self.buf.split("\n")
        with self.lock:
            for line in complete:
                if line.strip():
                    self.state.feed(json.loads(line))

    def render(self) -> str:
        with self.lock:
            return exposition(self.state.metrics()) if self.state else ""


def serve_live(results: Path, host: str, port: int) -> None:
    live = Live(results)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") not in ("/metrics", ""):
                self.send_error(404)
                return
            body = live.render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: Any) -> None:
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"exporter on http://{host}:{port}/metrics, watching {results}/**/events.jsonl", flush=True)
    while True:
        try:
            live.poll()
        except (OSError, json.JSONDecodeError) as e:
            print(f"poll error: {e}", file=sys.stderr, flush=True)
        time.sleep(0.5)


# --- replay -------------------------------------------------------------------------------------
def replay_events(run: Path) -> list[dict[str, Any]]:
    evs = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    extra = []
    for ev in evs:   # prefill / decode split, known only once the request ended
        if ev["type"] == "req_end" and ev.get("ok") and ev.get("t_first_token") is not None:
            extra.append({"t": ev["t_start"] + 1e-6, "type": "req_prefill", "slot": ev.get("slot")})
            extra.append({"t": ev["t_first_token"], "type": "req_first_token", "slot": ev.get("slot")})
    return sorted(evs + extra, key=lambda e: e["t"])


def openmetrics(run: Path, label: str) -> str:
    st = RunState(run, label)
    evs = replay_events(run)
    end = evs[-1]["t"] if evs else 0.0
    parts, i, t = [], 0, 0.0
    while t <= end + REPLAY_STEP:
        while i < len(evs) and evs[i]["t"] <= t:
            st.feed(evs[i])
            i += 1
        parts.append(exposition(st.metrics(), ts=ANCHOR + t))
        t += REPLAY_STEP
    # OpenMetrics wants each metric family contiguous, with its TYPE line first.
    fam: dict[str, list[str]] = defaultdict(list)
    for line in "".join(parts).splitlines():
        fam[family(re.match(r"[a-zA-Z_:][a-zA-Z0-9_:]*", line).group(0))].append(line)
    out = []
    for name, lines in fam.items():
        typ = HELP.get(name, ("gauge", ""))[0]
        if typ == "counter":   # OpenMetrics counters: family name without _total, samples with it
            out.append(f"# TYPE {name.removesuffix('_total')} counter")
        else:
            out.append(f"# TYPE {name} {typ}")
        out += lines
    return "\n".join(out) + "\n# EOF\n"


def delete_run(label: str, prom: str = "http://127.0.0.1:9090") -> None:
    """Remove a run's earlier replay from Prometheus, so replaying it again does not duplicate it."""
    q = urllib.parse.urlencode({"match[]": f'{{__name__=~"edgeserve_.*",run="{label}"}}'})
    for path in (f"/api/v1/admin/tsdb/delete_series?{q}", "/api/v1/admin/tsdb/clean_tombstones"):
        urllib.request.urlopen(urllib.request.Request(prom + path, method="POST"), timeout=60).read()


def replay(runs: list[Path], results: Path, compose_dir: Path) -> None:
    backfill = compose_dir / "backfill"
    backfill.mkdir(exist_ok=True)
    for run in runs:
        if not (run / "events.jsonl").exists():
            print(f"skip {run}: no events.jsonl")
            continue
        label = run_label(run, results)
        delete_run(label)
        f = backfill / (label.replace("/", "__") + ".om")
        f.write_text(openmetrics(run, label), encoding="utf-8", newline="\n")
        print(f"{label}: {f.stat().st_size / 1e6:.1f} MB of samples -> Prometheus", flush=True)
        r = subprocess.run(["docker", "compose", "exec", "-T", "prometheus", "promtool", "tsdb",
                            "create-blocks-from", "openmetrics", f"/backfill/{f.name}", "/prometheus"],
                           cwd=compose_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr, file=sys.stderr)
            raise SystemExit(f"backfill of {label} failed")
        f.unlink()
    subprocess.run(["docker", "compose", "restart", "prometheus"], cwd=compose_dir, check=True,
                   capture_output=True)
    print("done; open the 'EdgeServe replay' dashboard (time axis = seconds since run start, from 00:00)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lv = sub.add_parser("live")
    lv.add_argument("--results", default="results")
    lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=9101)
    rp = sub.add_parser("replay")
    rp.add_argument("runs", nargs="+", help="run directories (containing events.jsonl)")
    rp.add_argument("--results", default="results")
    rp.add_argument("--print", action="store_true", help="print OpenMetrics of the first run instead")
    args = ap.parse_args()
    if args.cmd == "live":
        serve_live(Path(args.results), args.host, args.port)
    elif args.print:
        run = Path(args.runs[0])
        sys.stdout.write(openmetrics(run, run_label(run, Path(args.results))))
    else:
        replay([Path(r) for r in args.runs], Path(args.results), HERE)


if __name__ == "__main__":
    main()
