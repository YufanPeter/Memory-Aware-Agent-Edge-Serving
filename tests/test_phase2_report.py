"""Phase 2 report helpers on synthetic rows: configuration and metric selection, the tuning rule
(best K), deltas and the end-to-end report on a tiny results root. Offline, no matplotlib."""
import csv
import json
from pathlib import Path

import pytest

from analysis import aggregate
from analysis import phase2_report as R


def row(**kw):
    base = {"sweep": "p2_main", "point": "c16k_d4_b1_k4", "rep": "rep0", "k": "4", "ctx": "16384", "np": "64",
            "cache_ram": "0", "pin": "True", "unified": "True", "delay_scale": "4.0", "workload": "dev_long_200",
            "n_tasks": "200", "policy": "observe", "admission": "fixed", "omega": "1.0", "completed": "190",
            "failed": "10", "goodput_tasks_per_min": "9.0", "throughput_tasks_per_min": "20.0",
            "steady_throughput_per_min": "21.0", "steady_goodput_per_min": "9.5", "em": "0.42", "f1": "0.55",
            "makespan_s": "500", "status_ok": "150", "status_forced": "40", "status_ctx_overflow": "10",
            "kvx_agents_mean": "4.0", "slog_purges": "3", "slog_overflow_events": "1", "violation_run": "False"}
    base.update(kw)
    return base


def run(tmp_path=Path("/nonexistent"), **kw):
    return R.derive(row(**kw), tmp_path)


def test_config_from_flags():
    assert R.config(row()) == "B1"
    assert R.config(row(policy="")) == "B1"   # Phase 1 runs carry no policy
    assert R.config(row(pin="False", cache_ram="896")) == "B2"
    assert R.config(row(pin="False", cache_ram="0")) == "unpinned"
    assert R.config(row(unified="False")) == "B3"
    assert R.config(row(policy="ours")) == "ours"


def test_derive_metrics():
    r = run()
    assert r["completion"] == pytest.approx(0.95)
    assert r["failures"] == {"ctx_overflow": 10}
    assert r["omega"] is None                      # fixed K: the recorded default ω is not a knob
    assert r["give_drop"] is None                  # no manager summary: no give-ups column
    assert r["op_share"] is None and r["recompute"] is None
    m = run(policy="ours", admission="kv", omega="0.5", mgr_give_ups="30", mgr_evictions_drop="20",
            mgr_evictions_offload="10", mgr_recompute_tokens="4000", mgr_op_server_s="50", status_ctx_overflow="",
            status_kv_infeasible="2")
    assert m["omega"] == 0.5
    assert (m["ops_offload"], m["ops_drop"], m["ops_erase"]) == (10, 20, 0)   # eviction ops; blank erase = none
    assert m["recompute"] == pytest.approx(20.0)   # older runs.csv: the manager's count (same definition)
    assert m["op_share"] == pytest.approx(0.1)
    assert m["failures"] == {"kv_infeasible": 2}
    # Phase 1 runs: never the same-slot kv_evicted_tokens, which misses cross-slot losses
    assert run(kv_evicted_tokens="1000")["recompute"] is None


def test_give_ups_and_recompute_from_tasks():
    """Give-ups by action come from the per-task counters, not summary.kv.evictions (which counts
    eviction ops: wasted saves and stale clean-ups too); recompute is one trace estimator."""
    m = run(policy="ours", admission="kv", mgr_give_ups="240", mgr_evictions_offload="125", mgr_evictions_drop="191",
            mgr_evictions_erase="106", recompute_tokens="2000", mgr_recompute_tokens="2000",
            gave_up_offloaded="117", gave_up_dropped="123", gave_up_erased="61")
    c = R.make_cell([m])
    assert R.gives(c) == "240 (117/123/61)"
    assert R.ops(c) == "125/191/106"
    assert m["recompute"] == pytest.approx(10.0)
    b1 = run(recompute_tokens="600", kv_evicted_tokens="100", gave_up_offloaded="")
    assert b1["recompute"] == pytest.approx(3.0) and R.gives(R.make_cell([b1])) == "–"

    def step(p, c, d, ok=True):
        return {"llm": {"ok": ok, "prompt_n": p, "cache_n": c, "predicted_n": d}}
    tasks = [{"trace": {"steps": [step(1000, 0, 41), step(60, 1040, 30),     # cache kept: 0 lost
                                  step(1130, 0, 20, ok=False),                # failed request: skipped
                                  step(500, 630, 10)]}}]                      # 1129 held, 630 found: 499 lost
    assert aggregate.recompute_tokens(tasks) == 499
    assert aggregate.recompute_tokens([{"trace": {"steps": []}}]) is None
    assert aggregate.give_up_row([{"kv": {"offloaded": 1, "dropped": 2}}, {"kv": {"erased": 3}}]) == \
        {"gave_up_offloaded": 1, "gave_up_dropped": 2, "gave_up_erased": 3}
    assert aggregate.give_up_row([{}]) == {}


def test_excluded_first_adm_retain():
    assert R.excluded({"sweep": "p2_tune", "point": "c16k_d4_admret_w1"})
    assert not R.excluded({"sweep": "p2_tune", "point": "c16k_d4_admret_v2_w1"})
    assert not R.excluded({"sweep": "p2_2x2", "point": "c16k_d4_admret_w1"})
    assert not R.excluded({"sweep": "p2_tune", "point": "c8k_d1_admret_w0.5"})   # 8k x1 ran on the fixed code


def cell(k, completion, goodput, cfg="B1"):
    return {"cfg": cfg, "k": k, "completion": completion, "goodput": goodput}


def test_pick_best_k_rule():
    cs = [cell(4, 0.98, 8.7), cell(6, 0.81, 10.5), cell(8, 0.46, 8.4), cell(3, 1.0, 8.0)]
    assert R.pick(cs)["k"] == 4        # highest goodput among >= 95%
    assert R.top(cs)["k"] == 6         # highest goodput whatever it completes
    # nobody reaches 95%: highest completion, goodput breaks ties
    cs = [cell(2, 0.89, 10.2), cell(3, 0.62, 10.9), cell(5, 0.89, 10.5)]
    assert R.pick(cs)["k"] == 5
    assert R.pick([]) is None and R.top([]) is None
    assert R.star(R.pick(cs)) == "★⁻"   # fallback: marked apart from a pick that meets the floor
    assert R.star(cell(4, 0.98, 8.7)) == "★"


def test_cells_and_deltas():
    base = R.make_cell([run(rep="rep0", goodput_tasks_per_min="8.0"), run(rep="rep1", goodput_tasks_per_min="10.0")])
    assert base["n"] == 2 and base["goodput"] == pytest.approx(9.0)
    assert (base["goodput_lo"], base["goodput_hi"]) == (8.0, 10.0)
    assert base["failures"] == {"ctx_overflow": 10.0}   # per run
    new = R.make_cell([run(policy="ours", admission="kv", rep=f"rep{i}", goodput_tasks_per_min=g, completed="200",
                           status_ctx_overflow="") for i, g in enumerate(("11.0", "12.0"))])
    d = R.delta(base, new)
    assert d["goodput"] == pytest.approx(2.5)
    assert d["goodput_pct"] == pytest.approx(100 * 2.5 / 9.0)
    assert d["completion_pp"] == pytest.approx(5.0)
    assert d["n"] == (2, 2) and 0 < d["p"] < 1
    assert R.rel(0, 5) is None
    assert R.fmt(base, "goodput", 2) == "9.00 (8.00–10.00)"
    assert R.fmt(new, "completion", 0, 100) == "100"   # equal reps: no range
    one = R.make_cell([run(completed="199")])
    assert R.fmt(one, "completion", 1, 100) == "99.5"   # a lost task stays visible


def test_knob_key_merges_sweeps_and_names():
    a = run(sweep="e2_budget_k", point="c16k_k4", workload="p1_long_200")
    b = run(sweep="p0x4_tune", point="c16k_k4", workload="p1_long_200")
    cells = R.group([a, b], R.knob_key)
    assert len(cells) == 1 and next(iter(cells.values()))["n"] == 2
    assert R.budget(131072, 1.0) == "131k ×1" and R.budget(8192, 4.0) == "8k ×4"


def write_root(root: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(root / "runs.csv", "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:   # per-question F1 needs tasks.jsonl; one task per question
        write_tasks(root / r["sweep"] / r["point"] / r["rep"], float(r["f1"]), int(r.get("n_failed_tasks", 0)),
                    n=int(r.get("n_task_lines", 20)), half_em=r.get("half_em") == "True")


def write_tasks(d: Path, f1: float, n_failed: int = 0, n: int = 20, half_em: bool = False) -> None:
    """n tasks, the last n_failed of them failed (status ctx_overflow, scored 0); half_em: only
    every other answered task is EM-correct, so the steady goodput is half (the bootstrap CI of a
    Δ against a full-EM run then excludes 0; two full-EM runs give a CI of exactly [0, 0])."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "tasks.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            bad = i >= n - n_failed
            em = 0.0 if bad or (half_em and i % 2) else 1.0
            f.write(json.dumps({"qid": f"q{i}", "status": "ctx_overflow" if bad else "ok", "em": em,
                                "f1": 0.0 if bad else f1 + 0.001 * (i % 3), "arrival_s": 0.0,
                                "latency_s": 10.0 + i, "service_s": 5.0}) + "\n")


def test_f1_tost_pairs_answered_questions(tmp_path):
    """A baseline that fails tasks must not look worse on quality: the verdict pairs only the
    questions answered in every run, and the failures-as-0 version is reported apart."""
    write_tasks(tmp_path / "b1", 0.6, n_failed=5)
    write_tasks(tmp_path / "ours", 0.6)
    q = R.f1_tost([tmp_path / "b1"], [tmp_path / "ours"])
    assert q["n"] == 15 and q["equivalent"]
    q0 = R.f1_tost([tmp_path / "b1"], [tmp_path / "ours"], answered=False)
    assert q0["n"] == 20 and not q0["equivalent"]


def test_steady_ci_needs_two_runs_a_side(tmp_path):
    rs = []
    for i in range(3):
        write_tasks(tmp_path / f"rep{i}", 0.6)
        rs.append(run(tmp_path, rep=f"rep{i}"))
    for r, i in zip(rs, range(3), strict=True):
        r["dir"] = tmp_path / f"rep{i}"
    one, two = R.make_cell(rs[:1]), R.make_cell(rs[1:])
    assert R.steady_ci(one, two, 20) is None            # one run: resampling runs is degenerate
    assert R.steady_ci(two, R.make_cell(rs[:2]), 20) is not None


def test_dev_2x2_uses_the_perf_knob(tmp_path):
    perf = {"workload": "p1_long_200", "sweep": "p2_tune", "policy": "gated_k", "admission": "fixed"}
    rs = [run(**perf, point="c16k_d4_gatedk_k4", k="4", completed="200", goodput_tasks_per_min="9.0"),
          run(**perf, point="c16k_d4_gatedk_k6", k="6", completed="200", goodput_tasks_per_min="11.5"),
          run(sweep="p2_2x2", policy="gated_k", point="c16k_d4_gatedk_k4", k="4", goodput_tasks_per_min="12"),
          run(sweep="p2_2x2", policy="gated_k", point="c16k_d4_gatedk_k6", k="6", goodput_tasks_per_min="10")]
    chosen, notes = R.dev_2x2(rs, R.tuning_cells(rs))
    assert chosen["gated_k"]["k"] == 6                 # the perf pick, although K=4 does better on dev
    assert any("also ran K=4" in n for n in notes)
    chosen, notes = R.dev_2x2(rs[:1] + rs[3:], R.tuning_cells(rs[:1] + rs[3:]))
    assert chosen["gated_k"] is None and any("picks K=4, which p2_2x2 has not run" in n for n in notes)


def test_report_on_partial_data(tmp_path):
    perf = {"workload": "p1_long_200", "sweep": "p0x4_tune"}
    rows = [row(**perf, point="c16k_k4", k="4", completed="196", goodput_tasks_per_min="8.7"),
            row(**perf, point="c16k_k6", k="6", completed="162", goodput_tasks_per_min="10.5"),
            row(point="c16k_d4_b1_k4", rep="rep0", goodput_tasks_per_min="8.8"),
            row(point="c16k_d4_b1_k4", rep="rep1", goodput_tasks_per_min="9.0", f1="0.56"),
            row(point="c16k_d4_ours", rep="rep0", policy="ours", admission="kv", k="16", completed="200",
                goodput_tasks_per_min="11.0", f1="0.57", mgr_give_ups="5", mgr_op_server_s="20"),
            row(point="c16k_d4_ours", rep="rep1", policy="ours", admission="kv", k="16", completed="200",
                goodput_tasks_per_min="11.4", f1="0.57", mgr_give_ups="5", mgr_op_server_s="20")]
    write_root(tmp_path, rows)
    md = R.report(tmp_path, n_boot=50)
    assert "**T1." in md and "| 16k ×4 | B1 | K=4 | 1 | 98.0 |" in md
    assert "| B1★ | K=4 | 2 |" in md                     # main-table B1 marked as the tuning pick
    assert "| ours | B1 K=4 | +25.8 |" in md            # (11.2 - 8.9) / 8.9
    assert "_T5: no ablation runs" in md and "_32k ×4 (p2_main): not run yet._" in md
    assert "eq." in md                                   # F1 TOST row


def test_verdict_pval_and_test_margins():
    """S needs both Welch p < 0.05 and a steady CI that excludes 0; p never prints as 0.00; the
    test-split TOST names the tightest margin its CI fits in."""
    ci = {"delta": 1.0, "ci": [0.2, 1.8]}
    assert R.verdict({"p": 0.01}, ci) == "S"
    assert R.verdict({"p": 0.01}, {"delta": -1.0, "ci": [-1.8, -0.2]}) == "S"   # a supported loss
    assert R.verdict({"p": 0.01}, {"delta": 1.0, "ci": [-0.1, 1.8]}) == "n.e."  # CI includes 0
    assert R.verdict({"p": 0.09}, ci) == "n.e."
    assert R.verdict({"p": None}, ci) == "n.e." and R.verdict({"p": 0.01}, None) == "n.e."
    assert R.pval({"p": 0.0042, "n": (3, 3)}) == "0.004 (n 3/3)"
    assert R.pval({"p": 0.0004, "n": (3, 3)}) == "<0.001 (n 3/3)"
    assert R.pval({"p": 0.012, "n": (2, 2)}) == "0.01 (n 2/2)"
    m = (0.03, 0.05)
    q = {"n": 400, "mean": 0.0, "equivalent": True}
    assert R.tost_cell({**q, "ci90": [-0.02, 0.02]}, margins=m) == "+0.000 [-0.020, +0.020] eq. (n 400)"
    assert R.tost_cell({**q, "ci90": [-0.01, 0.04]}, margins=m).endswith("eq. ±0.05 only (n 400)")
    assert R.tost_cell({**q, "ci90": [-0.01, 0.06]}, margins=m).endswith("not eq. (n 400)")


PT = {(16384, 4.0): ("c16k_d4", "16384", "4.0"), (8192, 1.0): ("c8k_d1", "8192", "1.0")}


def at(pt, **kw):
    """A row at one 2x2 point (budget x delay), named <prefix>_<point>."""
    pre, ctx, delay = PT[pt]
    kw["point"] = f"{pre}_{kw['point']}"
    return row(ctx=ctx, delay_scale=delay, **kw)


def two_by_two_rows():
    """Perf tuning, dev 2x2 and p2_main rows at both 2x2 points, plus 16k x4 ablations. At 8k x1
    managed_k also ran K=8 on perf (not the rule's pick there), and adm_retain beats gated_k on
    perf, so sched is adm_retain at 8k x1 and gated_k at 16k x4."""
    perf = {"workload": "p1_long_200", "sweep": "p2_tune", "completed": "200", "status_ctx_overflow": ""}
    kv = {"admission": "kv"}
    rows = []
    # the 16k x4 adm_retain tuning that counts is the post-fix admret_v2 (c16k_d4_admret_w* is excluded)
    for pt, (k_safe, k_top, k_gate, k_man, gate_g, adm_g, adm_pt) in {
            (16384, 4.0): (4, 6, 6, 8, "11.5", "10.6", "admret_v2_w0.5"),
            (8192, 1.0): (2, 3, 3, 3, "11.0", "11.4", "admret_w0.5")}.items():
        rows += [at(pt, sweep="p0x4_tune", workload="p1_long_200", point=f"b1_k{k_safe}", k=str(k_safe),
                    completed="196" if k_safe == 4 else "178", goodput_tasks_per_min="8.7"),   # 8k x1: 89%
                 at(pt, sweep="p0x4_tune", workload="p1_long_200", point=f"b1_k{k_top}", k=str(k_top),
                    completed="160", goodput_tasks_per_min="10.5"),
                 at(pt, **perf, policy="gated_k", point=f"gatedk_k{k_safe}", k=str(k_safe), goodput_tasks_per_min="9.0"),
                 at(pt, **perf, policy="gated_k", point=f"gatedk_k{k_gate}", k=str(k_gate), goodput_tasks_per_min=gate_g),
                 at(pt, **perf, policy="managed_k", point=f"managedk_k{k_man}", k=str(k_man),
                    goodput_tasks_per_min="11.68"),
                 at(pt, **perf, **kv, policy="adm_retain", point=adm_pt, omega="0.5", goodput_tasks_per_min=adm_g),
                 at(pt, **perf, **kv, policy="ours", point="ours_w1", omega="1.0", goodput_tasks_per_min="11.4")]
        for rep in ("rep0", "rep1"):
            dev = {"rep": rep, "completed": "200", "status_ctx_overflow": ""}
            rows += [at(pt, point=f"b1_k{k_safe}", k=str(k_safe), rep=rep, goodput_tasks_per_min=f"{11 + 0.1 * int(rep[-1])}",
                        half_em="True"),
                     at(pt, point=f"b1_k{k_top}", k=str(k_top), rep=rep, goodput_tasks_per_min="13.9", completed="177"),
                     at(pt, **dev, point="ours", policy="ours", admission="kv", k="16",
                        goodput_tasks_per_min=f"{14.6 + 0.1 * int(rep[-1])}"),
                     at(pt, **dev, sweep="p2_2x2", point=f"gatedk_k{k_gate}", policy="gated_k", k=str(k_gate),
                        goodput_tasks_per_min=f"{13.7 + 0.1 * int(rep[-1])}"),
                     at(pt, **dev, sweep="p2_2x2", point=f"managedk_k{k_man}", policy="managed_k", k=str(k_man),
                        goodput_tasks_per_min=f"{15.2 + 0.1 * int(rep[-1])}"),
                     at(pt, **dev, **kv, sweep="p2_2x2", point="admret_w0.5", policy="adm_retain", omega="0.5",
                        goodput_tasks_per_min=f"{12.8 + 0.1 * int(rep[-1])}")]
    eight = (8192, 1.0)   # the 16k x4 tie-break must not carry over: managed_k K=8 is not 8k x1's pick
    rows.append(at(eight, **perf, policy="managed_k", point="managedk_k8", k="8", goodput_tasks_per_min="11.0"))
    head = (16384, 4.0)   # ablations at 16k x4: p2_ablate single runs and p2_2x2 follow-up reps
    rows += [at(head, sweep="p2_ablate", point="lazy", policy="lazy", admission="kv", completed="200",
                status_ctx_overflow="", goodput_tasks_per_min="14.2"),
             at(head, sweep="p2_2x2", point="act_offload", policy="act_offload", admission="kv", completed="200",
                status_ctx_overflow="", goodput_tasks_per_min="15.3")]
    return rows


def test_two_by_two_at_both_points(tmp_path):
    write_root(tmp_path, two_by_two_rows())
    runs, dropped = R.load_runs(tmp_path)
    assert not dropped                                   # c8k_d1_admret_w0.5 in p2_tune is kept
    cells = R.tuning_cells(runs)
    head, eight = R.TWO_POINTS
    chosen, notes = R.dev_2x2(runs, cells, head)
    assert chosen["managed_k"]["k"] == 8 and any("at 16k ×4, managed_k: K=8 is used" in n for n in notes)
    chosen, notes = R.dev_2x2(runs, cells, eight)
    assert (chosen["gated_k"]["k"], chosen["managed_k"]["k"], chosen["adm_retain"]["omega"]) == (3, 3, 0.5)
    assert all(c["ctx"] == 8192 for c in chosen.values()) and not notes   # rule's picks, no 16k tie-break
    assert R.sched(R.perf_2x2(cells, head))["cfg"] == "gated_k"
    assert R.sched(R.perf_2x2(cells, eight))["cfg"] == "adm_retain"

    two, names, cnotes = R.effect_cells(runs, cells, eight)
    assert two["B1 same K"]["k"] == 3 and two["B1★"]["k"] == 2
    assert names == {"B1 same K": "B1†", "B1★": "B1★⁻"}  # no B1 K completes 95% on perf here: fallback
    rows = R.two_by_two(two, 50, names, cnotes)
    assert [r[0] for r in rows] == ["gate, same K", "gate, K re-tuned", "lifecycle at fixed K",
                                    "lifecycle at KV admission", "admission, llama.cpp lifecycle",
                                    "admission, managed lifecycle", "ours vs gated_k"]
    assert rows[1][1] == "gated_k (K=3) − B1★⁻ (K=2)"
    assert rows[1][8] == "S"                             # p < 0.05 and B1's half-EM window: CI excludes 0
    assert rows[2][8] == "n.e."                          # full EM on both sides: the CI is [0, 0]
    assert "sched on perf is adm_retain" in rows[6][9]
    assert "gated_k is sched" in R.effect_cells(runs, cells, head)[2]["ours vs gated_k"]
    two["managed_k"] = None
    assert R.two_by_two(two, 0, names)[2][-1] == "missing: managed_k"

    md = R.t4(runs, cells, 50)
    assert md.index("*16k ×4*") < md.index("*8k ×1*")
    assert md.count("| gate, same K |") == 2 and md.count("| ours vs gated_k |") == 2
    # T5 is 16k x4 only: the 8k x1 2x2 runs and the 2x2 cells never enter the ablations
    assert sorted(R.cell_name(c) for c in R.ablation_cells(runs)) == ["act_offload", "lazy"]


def test_t7_test_split(tmp_path):
    """The held-out test table: 500 tasks per run (completion over n_tasks, not 200), 2 reps, ours
    vs every row with the verdict, the ±0.03 test margin, and none of it leaking into T3."""
    head, eight = (16384, 4.0), (8192, 1.0)
    test = {"sweep": "p2_test", "workload": "test_long_500", "n_tasks": "500", "n_task_lines": "500",
            "status_ctx_overflow": ""}
    rows = [at(head, sweep="p0x4_tune", workload="p1_long_200", point="b1_k4", completed="196",
               goodput_tasks_per_min="8.7")]
    for rep in ("rep0", "rep1"):
        i = int(rep[-1])
        rows += [at(head, **test, rep=rep, point="b1_k4", completed="496", f1="0.56", half_em="True",
                    goodput_tasks_per_min=f"{10.0 + 0.1 * i}"),
                 at(head, **test, rep=rep, point="gatedk_k6", policy="gated_k", k="6", completed="500", f1="0.60",
                    goodput_tasks_per_min=f"{13.5 + 0.1 * i}"),
                 at(head, **test, rep=rep, point="ours", policy="ours", admission="kv", k="16", completed="499",
                    f1="0.60", goodput_tasks_per_min=f"{14.2 + 0.1 * i}"),
                 at(eight, **test, rep=rep, point="b1_k2", k="2", completed="475", f1="0.60",
                    goodput_tasks_per_min=f"{12.8 + 0.1 * i}"),
                 at(eight, **test, rep=rep, point="ours", policy="ours", admission="kv", k="8", completed="497",
                    f1="0.60", goodput_tasks_per_min=f"{15.0 + 0.1 * i}")]
    write_root(tmp_path, rows)
    runs, _ = R.load_runs(tmp_path)
    assert not R.main_cells(runs) and not R.ablation_cells(runs)   # test runs are T7's alone
    assert all(c["split"] == "perf" for c in R.tuning_cells(runs).values())
    cells = R.test_cells(runs)
    assert sorted(cells) == [eight, head]
    b1 = cells[head][0]
    assert b1["cfg"] == "B1" and b1["n"] == 2 and b1["completion"] == pytest.approx(496 / 500)
    md = R.t7(runs, R.picks(R.tuning_cells(runs), "B1"), 50)
    assert "test_long_500; 500 tasks per run, 2 reps" in md
    assert md.index("*16k ×4*") < md.index("*8k ×1*")
    assert "| B1★† | K=4 | 2 | 99.2 |" in md
    # ΔF1 0.04 vs B1 K=4 fits the dev margin only; Welch p < 0.05 and B1's half-EM window: S
    line = {r.split(" | ")[1]: r for r in md.splitlines() if r.startswith("| ours | ")}
    assert line["B1 K=4"].startswith("| ours | B1 K=4 | +41.8 | +0.6 |")
    assert line["B1 K=4"].endswith("eq. ±0.05 only (n 500) | S |")
    assert line["gated_k K=6"].endswith("eq. (n 500) | n.e. |")   # full EM both sides: CI [0, 0]
    assert "| B1 K=4 rep1 | rep0 (null) |" in md
    full = R.report(tmp_path, n_boot=50)
    assert "## T7" in full and "_T3: no p2_main runs yet._" in full
