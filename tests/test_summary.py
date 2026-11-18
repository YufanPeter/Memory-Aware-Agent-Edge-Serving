"""runner.summarize: request-level TTFT / TPOT next to the task-level metrics."""
from edgeserve.runner import summarize


def _row(i: int, ttfts: list[float], tpots_ms: list[float], em: float = 1.0) -> dict:
    steps = [{"llm": {"ok": True, "ttft_s": a, "predicted_ms": b * 10, "predicted_n": 10,
                      "prompt_n": 100, "cache_n": 0}} for a, b in zip(ttfts, tpots_ms)]
    steps.append({"llm": {"ok": False, "ttft_s": 99.0, "predicted_ms": None, "predicted_n": 0}})  # failed: ignored
    return {"task_id": f"t{i}", "status": "ok", "em": em, "f1": em, "sf_recall": 1.0, "arrival_s": 0.0,
            "latency_s": 10.0 + i, "service_s": 10.0 + i, "queue_wait_s": 0.0, "n_llm_calls": len(steps),
            "trace": {"steps": steps}}


def test_ttft_tpot_quantiles():
    rows = [_row(i, [0.1 * (i + 1), 0.2], [20.0 + i, 30.0]) for i in range(10)]
    s = summarize(rows, makespan=60.0)
    assert s["n_requests"] == 20
    assert s["ttft_p50_s"] == 0.2
    assert s["ttft_p99_s"] == 1.0            # the failed request (99 s) is not counted
    assert s["tpot_p50_ms"] == 30.0          # values 20..29 and ten 30s: the median falls on 30
    assert s["tpot_p99_ms"] == 30.0
