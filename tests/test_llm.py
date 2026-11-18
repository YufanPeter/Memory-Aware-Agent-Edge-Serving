"""MockLLM's shared pool (behaviour measured on llama-server in Phase 1 E0a), slot ops, and the
client's retry on the keep-alive race."""
import asyncio
import json

import httpx
import pytest

from edgeserve.llm import LlamaServerClient, MockLLM


def tok(n: int, c: str = "a") -> str:
    """A prompt of exactly n mock tokens (4 chars each)."""
    return c * (4 * n)


def silent(prompt: str) -> str:
    return ""


async def test_idle_slots_purged_lowest_id_first():
    llm = MockLLM(silent, pool_tokens=100)
    await llm.generate(tok(40, "x"), 8, [], slot=1)
    await llm.generate(tok(40, "y"), 8, [], slot=0)
    res = await llm.generate(tok(40, "z"), 8, [], slot=2)
    assert res.ok
    assert llm.purges == [0]                      # slot 0 goes first although slot 1 is older
    assert sorted(llm.slots) == [1, 2]


async def test_overflow_fails_every_request_in_flight():
    llm = MockLLM(lambda p: "w" * 40, decode_s_per_token=0.01, pool_tokens=100)
    a, b = await asyncio.gather(llm.generate(tok(50), 10, [], slot=0),
                                llm.generate(tok(50, "b"), 10, [], slot=1))
    assert not a.ok and not b.ok                  # 120 tokens in flight > 100, though one would fit
    assert "Context size has been exceeded" in (a.error or "")
    assert llm.overflows == 1
    assert llm.resident_tokens() == 0


async def test_requests_that_fit_succeed():
    llm = MockLLM(lambda p: "w" * 40, decode_s_per_token=0.01, pool_tokens=100)
    a, b = await asyncio.gather(llm.generate(tok(30), 10, [], slot=0),
                                llm.generate(tok(30, "b"), 10, [], slot=1))
    assert a.ok and b.ok
    assert llm.resident_tokens() == 80


async def test_save_erase_restore_keeps_the_cache():
    llm = MockLLM(silent, pool_tokens=100)
    await llm.generate(tok(40), 8, [], slot=0)
    assert (await llm.slot_save(0, "s0.bin")).n_tokens == 40
    await llm.slot_erase(0)
    assert llm.resident_tokens() == 0
    assert (await llm.slot_restore(0, "s0.bin")).ok
    res = await llm.generate(tok(40) + tok(4, "q"), 8, [], slot=0)
    assert res.cache_n == 40 and res.prompt_n == 4


async def test_restore_fails_without_room():
    llm = MockLLM(silent, pool_tokens=100)
    await llm.generate(tok(40), 8, [], slot=0)
    await llm.slot_save(0, "s0.bin")
    await llm.slot_erase(0)
    await llm.generate(tok(70, "b"), 8, [], slot=1)
    r = await llm.slot_restore(0, "s0.bin")
    assert not r.ok and r.error


async def test_speed_tax_slows_decode():
    fast = MockLLM(lambda p: "w" * 40, decode_s_per_token=0.001)
    taxed = MockLLM(lambda p: "w" * 40, decode_s_per_token=0.001, tax_s_per_kv_token=1e-5)
    taxed.slots[5] = tok(1000)                    # an idle slot holding 1,000 tokens
    a = await fast.generate(tok(10), 10, [], slot=0)
    b = await taxed.generate(tok(10), 10, [], slot=0)
    assert (b.t_end - b.t_first_token) > 2 * (a.t_end - a.t_first_token)


def _sse(content: str) -> bytes:
    final = {"content": "", "stop": True, "id_slot": 0, "stop_type": "eos",
             "timings": {"prompt_n": 3, "cache_n": 0, "predicted_n": 1, "prompt_ms": 1.0, "predicted_ms": 1.0}}
    return (f"data: {json.dumps({'content': content})}\n\n" f"data: {json.dumps(final)}\n\n").encode()


@pytest.mark.parametrize("fail_times, ok", [(1, True), (2, False)])
async def test_retry_on_keepalive_race(fail_times, ok):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, content=_sse("hi"))

    client = LlamaServerClient("http://test")
    client._client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    res = await client.generate("p", 4, [], slot=0)
    await client.close()
    assert res.ok is ok
    assert calls["n"] == 2                        # retried exactly once
    if ok:
        assert res.text == "hi"


async def test_slot_ops_parse_server_replies():
    def handler(request: httpx.Request) -> httpx.Response:
        action = request.url.params["action"]
        if action == "save":
            return httpx.Response(200, json={"id_slot": 3, "filename": "a.bin", "n_saved": 2048,
                                             "n_written": 117000000, "timings": {"save_ms": 85.5}})
        if action == "erase":
            return httpx.Response(200, json={"id_slot": 3, "n_erased": 2048})
        return httpx.Response(400, json={"error": {"message": "bad"}})

    client = LlamaServerClient("http://test")
    client._ctl = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    s = await client.slot_save(3, "a.bin")
    e = await client.slot_erase(3)
    r = await client.slot_restore(3, "missing.bin")
    await client.close()
    assert s.ok and s.n_tokens == 2048 and s.server_ms == 85.5 and s.n_bytes == 117000000
    assert e.ok and e.n_tokens == 2048
    assert not r.ok and "HTTP 400" in r.error
