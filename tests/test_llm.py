"""LlamaServerClient retry/stream parsing, SlotControl field parsing, EventLog listeners.
Bodies are the ones llama-server b11960 sent in E0b (results/e0b/*.json)."""
import asyncio
import json

import httpx

from edgeserve.events import EventLog
from edgeserve.llm import KEEPALIVE_EXPIRY_S, LlamaServerClient, SlotControl

FINAL = {"content": "", "stop": True, "id_slot": 3, "stop_type": "eos", "truncated": False,
         "timings": {"prompt_n": 5, "cache_n": 1395, "predicted_n": 2, "prompt_ms": 3.0, "predicted_ms": 9.0}}


def sse(*msgs: dict) -> bytes:
    return "".join(f"data: {json.dumps(m)}\n\n" for m in msgs).encode()


def client_with(handler, cls=LlamaServerClient, **kw):
    c = cls("http://mock", **kw)
    c._client = httpx.AsyncClient(base_url="http://mock", transport=httpx.MockTransport(handler))
    return c


async def test_retry_once_when_no_response_byte_arrived():
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, content=sse({"content": "Hi", "stop": False, "id_slot": 3}, FINAL))
    c = client_with(handler, return_progress=True)
    first = []
    res = await c.generate([1, 2, 3], 0, [], slot=3, on_first=first.append)
    assert res.ok and res.retries == 1 and res.text == "Hi" and first == [3]
    assert (res.prompt_n, res.cache_n, res.predicted_n, res.slot) == (5, 1395, 2, 3)
    assert res.kv_tokens == 1401
    assert calls[1]["prompt"] == [1, 2, 3] and calls[1]["return_progress"] is True and calls[1]["id_slot"] == 3
    assert KEEPALIVE_EXPIRY_S < 5.0


async def test_second_disconnect_is_an_error_and_errors_are_not_retried():
    n = [0]

    def handler(req):
        n[0] += 1
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
    res = await client_with(handler).generate("p", 1, [])
    assert not res.ok and res.retries == 1 and n[0] == 2 and "RemoteProtocolError" in res.error

    def overflow(req):
        n[0] += 1
        return httpx.Response(200, content=sse({"error": {"code": 500, "message": "Context size has been exceeded.",
                                                          "type": "server_error"}}))
    n[0] = 0
    res = await client_with(overflow).generate("p", 1, [], slot=2)
    assert not res.ok and n[0] == 1 and "Context size has been exceeded" in res.error and res.slot == -1


async def test_slot_control_parses_server_bodies():
    seen = []
    bodies = {
        "save": {"id_slot": 0, "filename": "busy.bin", "n_saved": 1399, "n_written": 80247364,
                 "timings": {"save_ms": 50.258}},
        "restore": {"id_slot": 15, "filename": "frag.bin", "n_restored": 3000, "n_read": 172080724,
                    "timings": {"restore_ms": 46.803}},
        "erase": {"id_slot": 0, "n_erased": 1399},
    }

    def handler(req):
        seen.append((req.method, req.url.path, dict(req.url.params), json.loads(req.content or b"{}")))
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [151644, 8948, 198]})
        if req.url.path == "/slots":
            return httpx.Response(200, json=[{"id": 0, "n_ctx": 8192, "is_processing": False, "n_prompt_tokens": 5400}])
        action = req.url.params["action"]
        if action == "restore" and req.url.path == "/slots/9":
            return httpx.Response(400, json={"error": {"code": 400, "type": "invalid_request_error", "message":
                                  "Unable to restore slot: No available space in KV cache or invalid slot save file"}})
        return httpx.Response(200, json=bodies[action])
    ctl = client_with(handler, SlotControl)
    assert await ctl.tokenize("<|im_start|>system\n") == [151644, 8948, 198]
    assert seen[0][1:] == ("/tokenize", {}, {"content": "<|im_start|>system\n", "add_special": True,
                                             "parse_special": True})
    s = await ctl.slot_save(0, "busy.bin")
    assert s.ok and (s.tokens, s.bytes, s.server_ms) == (1399, 80247364, 50.258)
    assert seen[-1][1:3] == ("/slots/0", {"action": "save"}) and seen[-1][3] == {"filename": "busy.bin"}
    r = await ctl.slot_restore(15, "frag.bin")
    assert r.ok and (r.tokens, r.bytes, r.server_ms) == (3000, 172080724, 46.803)
    e = await ctl.slot_erase(0)
    assert e.ok and e.tokens == 1399 and e.server_ms is None and seen[-1][3] == {}
    bad = await ctl.slot_restore(9, "frag.bin")
    assert not bad.ok and bad.status == 400 and "No available space" in bad.error and not bad.timeout
    assert (await ctl.slots())[0]["n_prompt_tokens"] == 5400


async def test_slot_control_op_timeout():
    async def slow(req):
        await asyncio.sleep(1.0)
        return httpx.Response(200, json={"id_slot": 0, "n_erased": 1})
    ctl = SlotControl("http://mock", op_timeout_s=0.05)
    ctl._client = httpx.AsyncClient(base_url="http://mock", transport=httpx.MockTransport(slow))
    r = await ctl.slot_erase(0)
    assert r.timeout and not r.ok and r.wall_ms < 500


def test_listener_errors_are_logged_not_raised():
    ev = EventLog()
    got = []

    def bad(e):
        raise ValueError("bug")
    ev.listeners += [bad, got.append]
    out = ev.emit("tool_start", task_id="t", step=1)
    assert out["type"] == "tool_start" and got == [out]
    assert [e["type"] for e in ev.events] == ["tool_start", "kv_violation"]
    v = ev.events[1]
    assert v["kind"] == "listener_error" and "ValueError: bug" in v["detail"]
