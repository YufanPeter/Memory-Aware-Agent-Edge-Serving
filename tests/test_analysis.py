"""Phase 2 tooling: exact KV timeline, server-log counts, sweep argument building, the priors
fit, the statistics helpers and the E0c helpers. All offline."""
import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from analysis import kv_timeline, server_log, stats
from scripts import fit_kvm_priors as fit
from scripts import sweep

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {"observe", "gated_k", "adm_retain", "managed_k", "ours", "ours_h0", "ttl_off", "always_off", "clairvoyant",
           "clairvoyant_pf", "oracle_prior", "lazy", "nosteal", "victim_lowid", "act_erase", "act_offload", "adm_duty"}


def kv_state(t, slot, frm, to, tokens, reason="x", task="t0"):
    return {"t": t, "type": "kv_state", "task_id": task, "slot": slot, "frm": frm, "to": to, "tokens": tokens,
            "reason": reason}


# --- kv_timeline ------------------------------------------------------------------------

def test_exact_intervals_from_kv_state():
    ev = [
        kv_state(0.0, 0, "FREE", "ACTIVE", 1000, "dispatch"),
        kv_state(1.0, 0, "ACTIVE", "IDLE", 1100, "response"),
        kv_state(3.0, 0, "IDLE", "SAVING", 1100, "deficit"),
        kv_state(3.5, 0, "SAVING", "OFFLOADED", 1100, "deficit"),
        kv_state(5.0, 0, "OFFLOADED", "RESTORING", 1100, "return"),
        kv_state(5.5, 0, "RESTORING", "ACTIVE", 1100, "dispatch"),
        kv_state(6.0, 0, "ACTIVE", "STALE", 1200, "release"),
        kv_state(2.0, 1, "FREE", "IDLE", 500, "response", "t1"),
        kv_state(4.0, 1, "IDLE", "IDLE", 0, "server_purge", "t1"),
        {"t": 8.0, "type": "task_done", "task_id": "t0"},
    ]
    ivs, st = kv_timeline.exact_intervals(ev)
    by = {}
    for slot, t0, t1, s, tok in ivs:
        by.setdefault((slot, s), 0.0)
        by[(slot, s)] += tok * (t1 - t0)
    assert by[(0, "active")] == pytest.approx(1000 * 1 + 1100 * 0.5)
    assert by[(0, "idle_tool")] == pytest.approx(1100 * 2)
    assert by[(0, "op")] == pytest.approx(1100 * 0.5 + 1100 * 0.5)
    assert by[(0, "offloaded")] == pytest.approx(1100 * 1.5)
    assert by[(0, "stale")] == pytest.approx(1200 * 2)          # closed at the last task_done
    assert by[(1, "idle_tool")] == pytest.approx(500 * 2)       # purged at t=4: 0 tokens after
    assert st["server_purges"] == 1 and st["server_purged_tokens"] == 500

    grid = kv_timeline.timeline(ivs, 1.0, kv_timeline.EXACT_STATES)
    s = kv_timeline.summarize(ivs, grid, st, kv_timeline.EXACT_STATES, kv_timeline.GPU_STATES)
    assert set(s["share"]) == set(kv_timeline.GPU_STATES)       # offloaded tokens are not in the pool
    assert sum(v for v in s["share"].values() if v) == pytest.approx(1.0, abs=1e-3)
    assert s["peak_offloaded_tokens"] == 1100


def test_analyze_picks_mode_and_manager_stats(tmp_path):
    ev = [{"t": 0.0, "type": "req_start", "task_id": "t0", "step": 1},
          kv_state(0.1, 0, "FREE", "ACTIVE", 100),
          {"t": 1.0, "type": "req_end", "task_id": "t0", "ok": True, "slot": 0, "prompt_n": 90, "cache_n": 0,
           "predicted_n": 11, "t_start": 0.1, "t_end": 1.0},
          kv_state(1.0, 0, "ACTIVE", "IDLE", 100),
          {"t": 1.5, "type": "kv_op", "op": "save", "reason": "deficit", "server_ms": 500.0, "wall_ms": 520.0,
           "ok": True, "tokens": 100},
          {"t": 1.6, "type": "kv_violation", "kind": "tok_mismatch", "detail": "x"},
          {"t": 1.7, "type": "kv_truncated", "task_id": "t0"},
          {"t": 1.8, "type": "kv_tick", "U": 300, "cap": 8064, "pressure": True, "gate_q": 2, "agents": 3},
          {"t": 1.9, "type": "kv_gate", "prio": 0, "wait_s": 0.25, "victims": [[1, "drop", 200]]},
          {"t": 10.0, "type": "task_done", "task_id": "t0"}]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev), encoding="utf-8")
    s = kv_timeline.analyze(tmp_path, 0.5)
    assert s["mode"] == "exact" and s["timeline_version"] == kv_timeline.VERSION
    m = s["manager"]
    assert m["violations"] == {"tok_mismatch": 1} and m["kv_truncated"] == 1
    assert m["op_stall_s"] == 0.5 and m["op_stall_share"] == pytest.approx(0.05)
    assert m["U_peak"] == 300 and m["pressure_share"] == 1.0 and m["gate_evictions"] == 1
    header = (tmp_path / "kv_timeline.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == ["t", *kv_timeline.EXACT_STATES]


def test_active_valued_at_the_size_after_the_request():
    # A fresh slot's first request: the ACTIVE event carries the pre-request size (0); the
    # request is valued at the response's count, Phase 1's per-request bound.
    ev = [{**kv_state(0.0, 0, "IDLE", "ACTIVE", 0, "dispatch"), "resident": 0, "charge": 1529},
          {**kv_state(1.0, 0, "ACTIVE", "IDLE", 1411, "req_end"), "resident": 1411, "charge": 1411},
          {**kv_state(2.0, 0, "IDLE", "ACTIVE", 1411, "dispatch"), "resident": 1411, "charge": 2200},
          {**kv_state(3.0, 0, "ACTIVE", "IDLE", 0, "overflow"), "resident": 0, "charge": 0},
          {**kv_state(4.0, 0, "IDLE", "ACTIVE", 0, "dispatch"), "resident": 0, "charge": 900},
          {"t": 5.0, "type": "task_done", "task_id": "t0"}]
    ivs, _ = kv_timeline.exact_intervals(ev)
    act = [(t0, t1, tok) for _, t0, t1, s, tok in ivs if s == "active"]
    assert act == [(0.0, 1.0, 1411), (2.0, 3.0, 2200), (4.0, 5.0, 900)]   # overflow / open: the charge


def test_unpinned_run_is_approximate(tmp_path):
    ev = [kv_state(0.0, 0, "FREE", "ACTIVE", 100), kv_state(1.0, 0, "ACTIVE", "IDLE", 110),
          {"t": 2.0, "type": "task_done", "task_id": "t0"}]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev), encoding="utf-8")
    (tmp_path / "meta.json").write_text(json.dumps({"kvm": {"pinned": True}}), encoding="utf-8")
    assert kv_timeline.analyze(tmp_path)["mode"] == "exact"
    (tmp_path / "meta.json").write_text(json.dumps({"kvm": {"pinned": False}}), encoding="utf-8")
    assert kv_timeline.analyze(tmp_path)["mode"] == "approximate"
    (tmp_path / "meta.json").unlink()
    ev.insert(2, kv_state(1.5, 0, "IDLE", "HOST_CACHED", 110, "host_cached"))
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev), encoding="utf-8")
    s = kv_timeline.analyze(tmp_path)
    assert s["mode"] == "approximate" and s["active_tokens"] == "post_request_bound"


def test_reconstruction_unchanged_without_kv_state():
    ev = [{"t": 1.0, "type": "req_end", "task_id": "a", "ok": True, "slot": 0, "prompt_n": 100, "cache_n": 0,
           "predicted_n": 10, "t_start": 0.0, "t_end": 1.0},
          {"t": 3.0, "type": "req_end", "task_id": "a", "ok": True, "slot": 0, "prompt_n": 120, "cache_n": 0,
           "predicted_n": 5, "t_start": 2.0, "t_end": 3.0},
          {"t": 4.0, "type": "task_done", "task_id": "a"}]
    assert not kv_timeline.has_exact(ev)
    ivs, st = kv_timeline.intervals(ev)
    assert [(s, t0, t1, tok) for _, t0, t1, s, tok in ivs] == [
        ("active", 0.0, 1.0, 110), ("idle_tool", 1.0, 2.0, 110), ("active", 2.0, 3.0, 125),
        ("idle_tool", 3.0, 4.0, 125)]
    assert st["evictions"] == 1 and st["evicted_tokens"] == 100   # cache_n 0 on a follow-up


# --- server log -------------------------------------------------------------------------

LOG = """0.01.000.000 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.02.000.000 W srv  try_clear_id: purging slot 3 with 1470 tokens
0.02.500.000 W srv  update_slots: failed to find free space in the KV cache, retrying with smaller batch size
0.03.000.000 E srv        decode: Context size has been exceeded. off = 605, n_batch = 1, ret = 1
0.03.000.001 E srv    send_error: task id = 0, error: Context size has been exceeded.
0.03.000.002 E srv    send_error: task id = 2, error: Context size has been exceeded.
0.03.100.000 W srv          load:  - making room for prompt cache entry, removing oldest entry (size = 80 MiB)
0.03.200.000 D find_slot: stream[0], n =  1000, used =   900, head =     0, size = 8192, n_swa =     0
0.03.300.000 D find_slot: stream[0], n =  1300, used =  1200, head =     0, size = 8192, n_swa =     0
"""


def test_server_log_counts(tmp_path):
    p = tmp_path / "server.log"
    p.write_text(LOG, encoding="utf-8")
    c = server_log.count(p)
    assert (c["purges"], c["purged_tokens"]) == (1, 1470)
    assert (c["overflow_lines"], c["overflow_events"], c["overflow_task_errors"]) == (3, 1, 2)
    assert c["batch_halvings"] == 1 and c["host_cache_room_lines"] == 1 and c["host_cache_evictions"] == 1
    assert "kvdbg" not in c                      # kv_timeline.kvdbg_check is the single source
    assert server_log.first_launch_s(p) == pytest.approx(1.0)
    assert server_log.find_slot_samples(p)[0] == (pytest.approx(3.2), 1000, 900)
    assert server_log.count(tmp_path / "missing.log") == {}


def test_server_log_judge():
    c = {"purges": 2, "overflow_events": 1}
    for gate, want in (("full", (2, 1)), ("active", (0, 1)), ("none", (0, 0)), (None, (0, 0))):
        j = server_log.judge(dict(c), gate)
        assert (j["purge_violations"], j["overflow_violations"]) == want and j["gate"] == gate
    assert server_log.judge(server_log.judge(dict(c), "full"), "full")["purge_violations"] == 2   # idempotent
    assert server_log.judge({}, "full") == {}


def test_kvdbg_check_aligns_clocks(tmp_path):
    p = tmp_path / "server.log"
    p.write_text(LOG, encoding="utf-8")
    ev = [{"t": 0.5, "type": "req_start"}, {"t": 2.0, "type": "kv_tick", "U": 1000, "cap": 1100},
          {"t": 2.5, "type": "kv_tick", "U": 1100, "cap": 1100}]
    d = kv_timeline.kvdbg_check(ev, p)   # offset 0.5 s: samples at runner time 2.7 and 2.8
    assert d["used_gt_U_approx"] == 1 and d["used_gt_cap"] == 1 and d["span_max"] == 1536


# --- sweep ------------------------------------------------------------------------------

def test_sweep_phase1_point_unchanged(tmp_path):
    (_, p), = sweep.points({"name": "s", "defaults": {"ctx": 8192}, "points": [{"name": "a", "workload": "w.json",
                                                                            "k": 4}]})
    args = sweep.runner_args(p, tmp_path)
    assert args[3:] == ["--workload", "w.json", "--out", str(tmp_path), "--k", "4", "--base-url",
                        "http://127.0.0.1:8080", "--delay-scale", "1.0", "--seed", "0", "--task-timeout", "900"]
    env = sweep.server_env(p, tmp_path)
    assert (env["CTX"], env["PARALLEL"], env["CACHE_RAM"], env["EXTRA"]) == ("8192", "64", "0", "")


def test_sweep_policy_kvm_env(tmp_path):
    spec = {"name": "s", "defaults": {"kvm": {"omega": 1.0}, "env": {"A": "1"}},
            "points": [{"name": "ours", "workload": "w.json", "ctx": 16384, "k": 16, "policy": "ours",
                        "kvm": {"host_frac": 0.5}, "env": {"LLAMA_KV_CACHE_DEBUG": 1}, "extra": "-lv 4"},
                       {"name": "b3", "workload": "w.json", "ctx": 32768, "k": 4, "parallel": "k",
                        "extra": "--no-kv-unified", "policy": "observe"}]}
    (_, ours), (_, b3) = sweep.points(spec)
    args = sweep.runner_args(ours, tmp_path, mock=True)
    i = args.index("--kvm")
    assert args[args.index("--policy") + 1] == "ours" and "--mock" in args
    assert args[args.index("--mock-slots") + 1] == "64" and "--mock-cache-ram" not in args
    b2 = {**ours, "pin": False, "parallel": "k", "cache_ram": 896, "policy": "observe"}
    b2_args = sweep.runner_args(b2, tmp_path, mock=True)
    assert b2_args[b2_args.index("--mock-slots") + 1] == "16" and b2_args[-1] == "896"
    assert json.loads(args[i + 1]) == {"ctx": 16384, "omega": 1.0, "host_frac": 0.5}
    env = sweep.server_env(ours, tmp_path)
    assert env["LLAMA_KV_CACHE_DEBUG"] == "1" and env["A"] == "1" and env["EXTRA"] == "-lv 4"
    assert json.loads(sweep.runner_args(b3, tmp_path)[-1])["ctx"] == 8192   # per-slot n_ctx without --kv-unified


def test_sweep_server_log_judged_by_gate(tmp_path):
    (tmp_path / "server.log").write_text(LOG, encoding="utf-8")
    for gate, purge_v, over_v in (("full", 1, 1), ("active", 0, 1), ("none", 0, 0)):
        (tmp_path / "meta.json").write_text(json.dumps({"kvm": {"gate": gate}}), encoding="utf-8")
        s = sweep.server_log_summary(tmp_path)
        assert (s["purge_violations"], s["overflow_violations"]) == (purge_v, over_v)
    (tmp_path / "summary.json").write_text(json.dumps({"completed": 1}), encoding="utf-8")
    assert sweep.add_server_log(tmp_path)["server_log"]["purges"] == 1


@pytest.mark.parametrize("path", sorted((ROOT / "configs/sweeps").glob("*.yaml")), ids=lambda p: p.stem)
def test_sweep_files_are_consistent(path):
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    kvm_yaml = ROOT / "configs/kvm.yaml"
    presets = set(yaml.safe_load(kvm_yaml.read_text(encoding="utf-8"))["presets"]) if kvm_yaml.exists() else PRESETS
    names = [pt["name"] for pt in spec["points"]]
    assert len(names) == len(set(names))
    for name, p in sweep.points(spec):
        assert (ROOT / p["workload"]).exists(), (name, p["workload"])
        assert sweep.n_parallel(p) >= p["k"] or "--no-kv-unified" in p["extra"], name
        if p["policy"]:
            assert p["policy"] in presets, (name, p["policy"])
            assert sweep.kvm_overrides(p)["ctx"] == sweep.expected_ctx(p)
        if p["cache_ram"]:   # B2: --cache-ram only unpinned (spec §5: the runner refuses it pinned/managed)
            assert not p["pin"] and p["policy"] in (None, "observe"), name


# --- priors fit -------------------------------------------------------------------------

def row(profile, sizes, status="ok"):
    steps = [{"llm": {"ok": True, "prompt_n": p, "cache_n": 0, "predicted_n": d, "latency_s": 1.0},
              "tool_delay_s": 1.0} for p, d in sizes]
    return {"profile": profile, "status": status, "service_s": 10.0, "_delay_scale": 2.0, "trace": {"steps": steps}}


def test_fit_growth_and_quantiles():
    p0, sizes = fit.task_sizes(row("x", [(100, 11), (200, 21)]))
    assert (p0, sizes) == (100, [110, 220])
    assert fit.growth(p0, sizes) == [120, 110]          # step 0 from P0, step 1 from size 110
    rows = [row("x", [(100, 11), (200 + 10 * i, 21)]) for i in range(20)]
    pri = fit.fit(rows, quantiles=(0.5, 1.0), min_n=10)["x"]
    assert pri["P0"] == 100 and pri["steps_mean"] == 2 and pri["n_at_step"] == [20, 20]
    assert pri["G"]["1.0"] == [310, 300] and pri["duty"] == 0.2 and pri["tool_s_unit_mean"] == 1.0
    assert fit.g_at(pri, 1.0, 9) == 300
    assert fit.coverage({"x": pri}, rows, 1.0)["x"]["covered"] == 1.0


def test_fit_excludes_overflow_runs(tmp_path):
    for name, statuses in (("good", {"ok": 1}), ("bad", {"ok": 1, "ctx_overflow": 3})):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"statuses": statuses}), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"delay_scale": 1.0}), encoding="utf-8")
        r = row("x", [(100, 11)])
        (d / "tasks.jsonl").write_text(json.dumps(r) + "\n" + json.dumps(row("x", [(5, 1)], "timeout")) + "\n",
                                       encoding="utf-8")
    rows, excluded = fit.load_tasks([tmp_path / "good", tmp_path / "bad"])
    assert len(rows) == 1 and excluded[0]["run"].endswith("bad")


# --- statistics -------------------------------------------------------------------------

def test_t_distribution_and_tests():
    assert stats.t_cdf(2.0, 10) == pytest.approx(0.963306, abs=1e-5)
    assert stats.t_ppf(0.975, 10) == pytest.approx(2.228139, abs=1e-4)
    w = stats.welch([1.0, 2.0, 3.0], [2.0, 3.0, 5.0])
    assert w["diff"] == pytest.approx(4 / 3) and w["df"] == pytest.approx(3.4483, abs=1e-3)
    assert w["p"] == pytest.approx(0.2846, abs=1e-3)
    assert stats.tost_paired([0.01, -0.01] * 50, 0.05)["equivalent"]
    assert not stats.tost_paired([0.1, 0.0] * 50, 0.05)["equivalent"]
    assert stats.cv([10.0, 10.0]) == 0.0


def test_bootstrap_and_steady_window():
    tasks = [{"status": "ok", "arrival_s": 0.0, "latency_s": float(i), "em": i % 2} for i in range(1, 21)]
    em, w = stats.steady_window(tasks)
    assert w == 19 - 3 and len(em) == 17    # completions between the 10th and 90th percentile
    a = [([0.0, 1.0] * 10, 60.0)] * 3
    b = [([1.0, 1.0] * 10, 60.0)] * 3
    d = stats.bootstrap_delta(a, b, "goodput", n_boot=300)
    assert d["delta"] == pytest.approx(10.0) and d["ci"][0] > 0
    assert stats.bootstrap_delta(a, b, "throughput", n_boot=50)["delta"] == 0.0


# --- E0c helpers ------------------------------------------------------------------------

def test_e0c_helpers():
    sys.path.insert(0, str(ROOT / "bench"))
    try:
        e0c = importlib.import_module("e0c_runtime")
    finally:
        sys.path.remove(str(ROOT / "bench"))
    assert e0c.lcp([1, 2, 3], [1, 2, 4]) == 2 and e0c.lcp([], [1]) == 0
    c = e0c.trc_counts(LOG)
    assert c["purge"] == 1 and c["overflow"] == 3 and c["cache_room"] == 1
    assert e0c.equality_stats([{"tok": 5, "prompt_n": 1, "cache_n": 4}, {"tok": 7, "prompt_n": 1, "cache_n": 4}]) \
        == {"n": 2, "equal": 1, "max_abs_diff": 2, "diffs_nonzero": [2]}
    vocab: dict[str, int] = {}

    async def tok(text):
        return [vocab.setdefault(text[i:i + 4], len(vocab)) for i in range(0, len(text) - len(text) % 4, 4)]

    pref = asyncio.run(e0c.family_prefix(tok, False))
    for q in ("Who directed Jaws?", "Is Paris in France?", "1984 or Brave New World?"):
        assert asyncio.run(tok(e0c.first_react_prompt(q, False)))[: len(pref)] == pref
    assert len(pref) > 1000


# --- aggregate ----------------------------------------------------------------------------

def test_aggregate_phase2_columns(tmp_path):
    from analysis import aggregate
    run = tmp_path / "p1_dev" / "c8k_d1_ours" / "rep0"
    run.mkdir(parents=True)
    summary = {"n_tasks": 2, "statuses": {"ok": 1, "kv_infeasible": 1}, "completed": 1, "failed": 1,
               "makespan_s": 10.0, "kv": {"admitted": 2, "violations": {"purge": 0}, "kv_exceeded": ["t1"]},
               "server_log": {"purges": 2, "overflow_events": 0, "gate": "full"}}
    meta = {"k": 8, "pin_slots": True, "cache_prompt": True, "delay_scale": 1.0, "workload": "workloads/w.json",
            "server_cmd": "llama-server -c 8192 -np 64 --cache-ram 0",
            "kvm": {"policy": "ours", "gate": "full", "admission": "kv", "omega": 1.0, "host_frac": 1.0}}
    llm = {"ok": True, "prompt_n": 90, "cache_n": 10, "predicted_n": 10, "ttft_s": 0.1}
    tasks = [{"task_id": "t0", "status": "ok", "service_s": 5.0, "trace": {"steps": [{"llm": llm}]}},
             {"task_id": "t1", "status": "kv_infeasible", "service_s": 0.1, "trace": {"steps": []}}]
    ev = [kv_state(0.0, 0, "FREE", "ACTIVE", 100), kv_state(1.0, 0, "ACTIVE", "STALE", 109),
          {"t": 2.0, "type": "task_done", "task_id": "t0"}]
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev), encoding="utf-8")
    (run / "kv_summary.json").write_text(json.dumps({"peak_tokens": 1}), encoding="utf-8")   # pre-Phase-2 format
    r = aggregate.run_row(run)
    assert (r["policy"], r["gate"], r["omega"], r["ctx"]) == ("ours", "full", 1.0, "8192")
    assert r["kv_infeasible"] == 1 and r["kv_infeasible_ids"] == "t1" and r["status_kv_infeasible"] == 1
    assert r["kv_exceeded"] == 1 and r["kv_exceeded_ids"] == "t1"
    assert r["mgr_admitted"] == 2 and r["mgr_violations_purge"] == 0
    assert r["slog_purges"] == 2 and r["slog_purge_violations"] == 2 and r["violation_run"]
    assert r["kv_mode"] == "exact"                       # the stale kv_summary.json was recomputed
    assert r["work_tok_per_s"] == 10.0 and r["final_context_mean"] == 110 and r["req_hit_rate_mean"] == 0.1
    p = aggregate.point_rows([r, {**r, "rep": "rep1", "steady_goodput_per_min": 2.0}], list(r))[0]
    assert p["reps"] == 2 and p["violation_runs"] == 2 and p["kv_infeasible_ids_union"] == "t1"
    assert p["kv_exceeded_ids_union"] == "t1"
