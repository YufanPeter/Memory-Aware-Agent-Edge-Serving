"""Phase 3 report helpers on synthetic data: labels, the selector rule, trace metrics (peak context,
KV held at tool start after the eager compaction) and the effect list. Offline, no matplotlib."""
import json
from pathlib import Path

import pytest

from analysis import phase3_report as P


def llm(prompt, cache, pred, latency=0.1):
    return {"ok": True, "prompt_n": prompt, "cache_n": cache, "predicted_n": pred, "latency_s": latency}


def task(steps, status="ok", profile="react_long", qid="q1"):
    return {"task_id": "t0", "qid": qid, "profile": profile, "status": status, "em": 1.0, "f1": 1.0,
            "sf_recall": 0.5, "trace": {"steps": steps}}


def write_run(d: Path, tasks):
    d.mkdir(parents=True)
    with open(d / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    return d


def test_trace_metrics_use_the_compacted_slot_during_the_tool_wait(tmp_path):
    steps = [
        {"step": 1, "kind": "react", "action": "search", "llm": llm(1400, 0, 40)},     # 1440 cells, tool
        {"step": 2, "kind": "react", "action": "lookup", "llm": llm(500, 1439, 30)},   # 1969 cells
        {"step": 2, "kind": "trim", "action": None, "trimmed_obs": 1, "llm": llm(150, 1500, 1, 0.05)},  # 1651
        {"step": 3, "kind": "react", "action": "finish", "llm": llm(60, 1650, 20)},
    ]
    run = write_run(tmp_path / "r", [task(steps), task(steps[:1] + [steps[3]], status="forced", profile="react_short",
                                                       qid="q2")])
    m = P.trace_metrics(str(run))
    # held at each search/lookup start: step 1 (no compaction) 1439, step 2 after its compaction 1650
    assert m["held"] == pytest.approx((1439 + 1650 + 1439) / 3)
    assert m["peak"] == pytest.approx((1969 + 1730) / 2)
    assert m["peak_long"] == 1969
    assert m["forced"] == 0.5
    assert m["calls"] == pytest.approx(2.5)          # trim requests are not agent calls
    assert m["trim_lat"] == pytest.approx(0.05)


def test_labels_and_naive_marking():
    base = {"cfg": "ours", "trim": "oracle", "fixed": False}
    assert P.label(base) == "ours+oracle (naive)"
    assert P.label({**base, "fixed": True}) == "ours+oracle"
    assert P.label({**base, "cfg": "B1"}) == "B1+oracle"          # B1 has no gate: nothing to fix
    assert P.label({**base, "trim": "none"}) == "ours"
    assert P.label({"cfg": "B1", "trim": "relevance:2", "fixed": False}) == "B1+rel2"


def cell(trim, f1, peak):
    return {"trim": trim, "f1": f1, "peak": peak}


def test_selector_rule():
    cs = {"none": cell("none", 0.60, 3000), "referenced": cell("referenced", 0.59, 2600),
          "relevance:2": cell("relevance:2", 0.58, 2500), "relevance:3": cell("relevance:3", 0.62, 2550),
          "oracle": cell("oracle", 0.70, 2000)}            # oracle never qualifies: not realistic
    assert P.chosen(cs) == "relevance:2"                    # 0.58 is within 0.03 of 0.60, smallest peak
    cs["relevance:2"]["f1"] = 0.56                          # now outside: next smallest peak that qualifies
    assert P.chosen(cs) == "relevance:3"
    for m in ("referenced", "relevance:3"):
        cs[m]["f1"] = 0.50
    assert P.chosen(cs) == "relevance:2"                    # nobody qualifies: highest F1 among realistic


def test_effects_cover_both_selectors_and_skip_missing_cells():
    cs = {"B1 K=4": {"star": True, "k": 4}, "B1 K=6": {"k": 6}, "ours": {}, "B1+oracle": {"k": 6},
          "B1+rel2": {"k": 6}, "ours+oracle": {}, "ours+rel2": {}, "ours+oracle (naive)": {}, "ours+rel2 (naive)": {}}
    eff = P.effects(cs)
    names = [e[0] for e in eff]
    assert ("TRIM for B1, same K (oracle)", "B1 K=6", "B1+oracle") in eff
    assert ("TRIM for B1, K retuned (rel2)", "B1 K=4", "B1+rel2") in eff
    assert ("shrink-fast fix (rel2)", "ours+rel2 (naive)", "ours+rel2") in eff
    assert names[-1] == "ours vs B1, untrimmed (Phase 2, T3)"
    # at 8k x1 the retuned K equals Phase 2's pick: no separate same-K row
    cs8 = {"B1 K=2": {"star": True, "k": 2}, "ours": {}, "B1+oracle": {"k": 2}}
    assert not any(n.startswith("TRIM for B1, same K") for n, _, _ in P.effects(cs8))
