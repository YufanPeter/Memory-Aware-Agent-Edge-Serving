"""MockPoolLLM reproduces the llama-server behaviours the KV manager relies on (E0a, E0b)."""
import asyncio
import time

from edgeserve.agent import Agent
from edgeserve.events import EventLog
from edgeserve.llm import MockLLM
from edgeserve.mockpool import EXCEEDED, MockPoolLLM
from edgeserve.profiles import Profile
from edgeserve.runner import mock_script


def pool(script=lambda p: "y" * 4000, **kw) -> MockPoolLLM:
    kw = {"capacity": 8192, "n_slots": 16, "t_prefill": 0.0, "t_decode": 0.0, "op_fixed_s": 0.0} | kw
    return MockPoolLLM(script, **kw)


async def ids(p: MockPoolLLM, seed: int, n: int) -> list[int]:
    """n distinct 4-char tokens; prompts with different seeds differ at the first token."""
    return await p.tokenize("".join(f"{seed:02d}{i % 100:02d}" for i in range(n)))


async def test_e0a_overflow_fails_every_processing_request():
    # 4 concurrent x (2,500 prompt + 400 generated) in an 8k pool: all fail, all slots cleared.
    p = pool(lambda _: "")
    prompts = [await ids(p, 10 + i, 2500) for i in range(4)]
    res = await asyncio.gather(*(p.generate(pr, 400, [], slot=i, ignore_eos=True) for i, pr in enumerate(prompts)))
    assert [r.ok for r in res] == [False] * 4
    assert all("Context size has been exceeded" in r.error and "context" in r.error.lower() for r in res)
    assert p.snapshot() == {} and p.used == 0
    assert p.overflows == 4 and p.overflow_events == 1 and p.purges == 0


async def test_e0a_pressure_purges_lowest_idle_slot_first():
    # 4 pinned x 2,500 sent one after another (n_predict 8), then revisited (n_predict 1).
    p = pool()
    first = [await p.generate(await ids(p, i, 2500), 8, [], slot=i) for i in range(4)]
    assert all(r.ok and r.kv_tokens == 2507 for r in first)
    assert p.snapshot() == {1: 2507, 2: 2507, 3: 2507}          # slot 3's prefill purged slot 0
    revisit = [await p.generate(await ids(p, i, 2500), 1, [], slot=i) for i in range(4)]
    assert [r.cache_n for r in revisit] == [0, 0, 2499, 2499]   # E0a pressure.json
    assert [(s, n) for _, s, n in p.purge_log] == [(0, 2507), (1, 2507), (0, 2500)]   # E0a server log
    assert p.log[0] == "purging slot 0 with 2507 tokens"
    assert p.overflows == 0 and p.max_used <= 8192


async def test_s_formula_and_tokenize_match_the_server():
    p = pool(lambda _: "y" * 4000)
    prompt = await ids(p, 1, 2500)
    r = await p.generate(prompt, 8, [], slot=0)
    assert (r.prompt_n, r.cache_n, r.predicted_n) == (2500, 0, 8)
    assert p.cells(0) == r.prompt_n + r.cache_n + r.predicted_n - 1 == 2507   # E0a: 2,500 + 8 -> 2,507
    text = "abcd" * 10 + "xy"                                                 # trailing partial chunk dropped
    r2 = await p.generate(text, 0, [], slot=1)
    assert len(await p.tokenize(text)) == r2.prompt_n + r2.cache_n == 10
    assert r2.predicted_n == 1 and p.cells(1) == 10                           # n_predict 0 behaves like 1


async def test_e0b_truncate_keeps_prefix_and_frees_the_tail():
    p = pool(lambda _: "")
    big = await ids(p, 0, 5400)
    assert (await p.generate(big, 1, [], slot=0)).kv_tokens == 5400
    t = await p.generate(big[:1400], 0, [], slot=0)                           # drop to prefix, token ids
    assert (t.prompt_n, t.cache_n, t.predicted_n) == (1, 1399, 1)
    assert p.cells(0) == 1400
    assert (await p.generate(await ids(p, 1, 5500), 1, [], slot=1)).ok and p.purges == 0   # 5.5k fits now
    rev = await p.generate(big, 1, [], slot=0)
    assert (rev.prompt_n, rev.cache_n) == (4000, 1400)
    assert p.purges == 1 and p.purge_log[0][1] == 1


async def test_e0b_restore_into_full_pool_fails_and_clears():
    p = pool(lambda _: "")
    a = await ids(p, 0, 3000)
    await p.generate(a, 1, [], slot=0)
    s = await p.slot_save(0, "a.bin")
    assert s.ok and s.tokens == 3000 and s.bytes == 172_080_724                 # E0b fragment n_written
    assert (await p.slot_erase(0)).tokens == 3000
    for slot, n in ((1, 4000), (2, 3000), (3, 500)):
        await p.generate(await ids(p, slot, n), 1, [], slot=slot)
    r = await p.slot_restore(3, "a.bin")
    assert not r.ok and r.status == 400 and "No available space" in r.error
    assert p.cells(3) == 0 and p.purges == 0                                   # cleared, no purge
    slots = {d["id"]: d for d in await p.slots()}
    assert slots[3]["n_prompt_tokens"] == 0


async def test_e0b_restore_into_holes_and_cross_slot():
    p = pool(lambda _: "")
    prompts = [await ids(p, i, 1000) for i in range(8)]
    for i, pr in enumerate(prompts):
        await p.generate(pr, 1, [], slot=i)
    assert (await p.slot_save(7, "frag.bin")).tokens == 1000
    await p.slot_erase(7)
    for i in (0, 2, 4, 6):
        await p.slot_erase(i)
    assert p.used == 3000
    r = await p.slot_restore(7, "frag.bin")
    assert r.ok and r.tokens == 1000 and r.server_ms is not None
    assert p.used_max_p1() <= 8000                                             # placed in the holes
    chk = await p.generate(prompts[7], 1, [], slot=7)
    assert (chk.prompt_n, chk.cache_n) == (1, 999)
    assert [(await p.generate(prompts[i], 1, [], slot=i)).cache_n for i in (1, 3, 5)] == [999] * 3
    x = await p.slot_restore(9, "frag.bin")                                    # another slot (E0b perslot)
    assert x.ok and (await p.generate(prompts[7], 1, [], slot=9)).cache_n == 999
    assert p.purges == 0 and p.overflows == 0


async def test_restore_of_missing_file_fails_and_clears():
    p = pool(lambda _: "")
    await p.generate(await ids(p, 0, 100), 1, [], slot=0)
    r = await p.slot_restore(0, "nope.bin")
    assert not r.ok and r.status == 400 and p.cells(0) == 0
    assert not (await p.slot_save(0, "../x.bin")).ok                           # fs_validate_filename


async def test_files_on_disk_with_slot_dir(tmp_path):
    p = pool(lambda _: "", slot_dir=tmp_path)
    pr = await ids(p, 0, 300)
    await p.generate(pr, 1, [], slot=0)
    await p.slot_save(0, "s0.bin")
    assert (tmp_path / "s0.bin").exists()
    await p.slot_erase(0)
    assert (await p.slot_restore(0, "s0.bin")).tokens == 300
    (tmp_path / "s0.bin").unlink()                                             # the manager unlinks after restore
    assert not (await p.slot_restore(1, "s0.bin")).ok


async def test_busy_slot_defers_ops_and_requests():
    p = pool(lambda _: "", t_decode=0.002)
    pr = await ids(p, 0, 400)
    stream = asyncio.create_task(p.generate(pr, 100, [], slot=0, ignore_eos=True))
    await asyncio.sleep(0.02)
    assert p.is_processing(0)
    assert {d["id"]: d for d in await p.slots()}[0]["is_processing"] is True
    save = asyncio.create_task(p.slot_save(0, "busy.bin"))                     # deferred until the stream ends
    await asyncio.sleep(0.001)
    second = asyncio.create_task(p.generate(pr, 1, [], slot=0))               # deferred behind both, FIFO
    first, s = await stream, await save
    assert s.ok and s.tokens == first.kv_tokens == 499                         # E0b busy: n_saved = s
    assert s.wall_ms > s.server_ms
    r2 = await second
    assert (r2.prompt_n, r2.cache_n) == (1, 399)                               # ran after both, on the stream's KV
    assert (await p.slot_erase(0)).tokens == p.ops[-1]["tokens"] == 400
    assert (await p.generate(pr, 1, [], slot=0)).cache_n == 0                  # E0b busy: next request cache_n 0


async def test_save_and_restore_stall_every_slot():
    p = pool(lambda _: "", kappa_s=2e-4, t_decode=1e-3)
    await p.generate(await ids(p, 0, 1000), 1, [], slot=0)
    other = await ids(p, 1, 100)
    save = asyncio.create_task(p.slot_save(0, "x.bin"))
    await asyncio.sleep(0)                                                     # the save holds the main loop
    r = await p.generate(other, 20, [], slot=1, ignore_eos=True)
    s = await save
    assert s.server_ms >= 199
    assert r.t_end - r.t_start >= 0.19                                         # stalled behind the save
    e = asyncio.create_task(p.slot_erase(0))                                   # erase: op_fixed_s only
    r = await p.generate(other, 1, [], slot=1)
    await e
    assert r.t_end - r.t_start < 0.1


async def test_slots_reports_n_prompt_tokens_like_the_server():
    p = pool(lambda _: "")
    pr = await ids(p, 0, 700)
    slots = await p.slots()
    assert len(slots) == 16 and "n_prompt_tokens" not in slots[0]             # never ran a task (sc:814)
    r = await p.generate(pr, 1, [], slot=0)
    assert (await p.slots())[0]["n_prompt_tokens"] == r.kv_tokens == 700
    await p.slot_save(0, "f.bin")
    await p.slot_erase(0)
    assert (await p.slots())[0]["n_prompt_tokens"] == 0                        # task_prev remains
    await p.slot_restore(5, "f.bin")
    assert "n_prompt_tokens" not in (await p.slots())[5]                       # a restore is not a task
    assert p.cells(5) == 700
    assert (await p.props())["default_generation_settings"]["n_ctx"] == 8192


async def test_stale_slots_snapshot_predates_a_queued_erase():
    for stale in (True, False):
        p = pool(lambda _: "", t_decode=0.05, slots_stale=stale)
        await p.generate(await ids(p, 1, 100), 1, [], slot=1)
        long = asyncio.create_task(p.generate(await ids(p, 0, 50), 6, [], slot=0, ignore_eos=True))
        await asyncio.sleep(0.01)                                              # a decode holds the main loop
        erase = asyncio.create_task(p.slot_erase(1))
        await asyncio.sleep(0.002)                                             # the erase waits for the loop
        snap = {d["id"]: d for d in await p.slots()}
        assert snap[1]["n_prompt_tokens"] == (100 if stale else 0)
        assert (await erase).tokens == 100
        assert {d["id"]: d for d in await p.slots()}[1]["n_prompt_tokens"] == 0
        await long
    p.freeze_slots()
    await p.generate(await ids(p, 2, 10), 1, [], slot=2)
    assert "n_prompt_tokens" not in (await p.slots())[2]
    p.thaw_slots()
    assert (await p.slots())[2]["n_prompt_tokens"] == 10


async def test_op_timeout_leaves_the_outcome_unknown_to_the_client():
    p = pool(lambda _: "", op_timeout_s=0.1, op_hang={"save": 0.3})
    await p.generate(await ids(p, 0, 200), 1, [], slot=0)
    r = await p.slot_save(0, "h.bin")
    assert r.timeout and not r.ok and 90 <= r.wall_ms < 250 and "h.bin" not in p.files
    await asyncio.sleep(0.35)
    assert "h.bin" in p.files and p.ops[-1]["op"] == "save" and p.ops[-1]["ok"]   # the server ran it later
    assert (await p.slot_erase(0)).tokens == 200


async def test_inject_500_fails_only_processing_requests():
    p = pool(lambda _: "", t_decode=0.005)
    await p.generate(await ids(p, 2, 300), 1, [], slot=2)                      # idle, keeps its KV
    a = asyncio.create_task(p.generate(await ids(p, 0, 100), 50, [], slot=0, ignore_eos=True))
    b = asyncio.create_task(p.generate(await ids(p, 1, 100), 50, [], slot=1, ignore_eos=True))
    queued = asyncio.create_task(p.generate(await ids(p, 0, 100), 1, [], slot=0))   # deferred
    await asyncio.sleep(0.05)
    assert sorted(p.inject_500()) == [0, 1]
    ra, rb, rq = await a, await b, await queued
    assert not ra.ok and not rb.ok and EXCEEDED in ra.error and ra.slot == 0  # first token was already sent
    assert rq.ok and rq.cache_n == 0                                           # its slot was cleared
    assert p.cells(2) == 300 and p.http500 == 2


async def test_context_limit_stops_generation_and_refuses_long_prompts():
    p = pool(lambda _: "", capacity=1000)
    r = await p.generate(await ids(p, 0, 990), 50, [], slot=0, ignore_eos=True)
    assert r.ok and r.predicted_n == 10 and r.truncated and r.stop_type == "limit"   # n_tokens + 1 >= n_ctx
    assert p.cells(0) == 999
    big = await p.generate(await ids(p, 1, 1000), 1, [], slot=1)
    assert not big.ok and big.error.startswith("HTTP 400") and "exceeds the available context size" in big.error
    assert p.cells(0) == 999 and p.purges == 0


async def test_unpinned_requests_prefer_the_longest_common_prefix():
    p = pool(lambda _: "")
    a, b = await ids(p, 0, 400), await ids(p, 1, 400)
    ra = await p.generate(a, 1, [], slot=-1)
    rb = await p.generate(b, 1, [], slot=-1)
    assert ra.slot != rb.slot
    assert (await p.generate(a + b, 1, [], slot=-1)).slot == ra.slot


async def test_cache_ram_emulation_reproduces_e0b():
    p = pool(lambda _: "", n_slots=4, cache_ram_mib=4096)
    A, B, C = [await ids(p, s, 3000) for s in (21, 22, 23)]
    alt = [(await p.generate(x, 1, [], slot=0)).cache_n for x in (A, B, A)]
    for i in range(4):
        await p.slot_erase(i)
    first = [(await p.generate(await ids(p, 40 + i, 2500), 1, [], slot=i)).cache_n for i in range(4)]
    revisit = [(await p.generate(await ids(p, 40 + i, 2500), 1, [], slot=i)).cache_n for i in range(4)]
    for i in range(4):
        await p.slot_erase(i)
    unpinned = [(await p.generate(x, 1, [], slot=-1)).cache_n for x in (A, B, C, A)]
    assert alt == [0, 0, 2999] and first == [0] * 4
    assert revisit == [0, 0, 0, 0]                    # pinned revisit of a cleared slot never reloads
    assert unpinned == [0, 2999, 0, 2999]             # E0b cacheram
    assert p.purges == 0 and p.idle_clears > 0


async def test_check_hook_sees_every_mutation():
    p = pool(lambda _: "y" * 100, capacity=4096)
    seen = []

    def check(pool_):
        seen.append(pool_.used)
        assert pool_.used <= pool_.capacity
        assert pool_.used == sum(pool_.snapshot().values())
    p.check = check
    await p.generate(await ids(p, 0, 1000), 5, [], slot=0)
    await p.slot_save(0, "c.bin")
    await p.slot_erase(0)
    await p.slot_restore(1, "c.bin")
    assert p.check_errors == [] and len(seen) > 5 and seen[-1] == 1004
    p.check = lambda _: (_ for _ in ()).throw(AssertionError("boom"))
    await p.slot_erase(1)
    assert p.check_errors == ["AssertionError: boom"]


async def test_tok_noise_and_tokenize_delay():
    p = pool(lambda _: "", tok_noise=2, tokenize_delay_s=0.02)
    text = "abcd" * 50
    r = await p.generate(text, 1, [], slot=0)
    t0 = time.monotonic()
    toks = await p.tokenize(text)
    assert time.monotonic() - t0 >= 0.015
    assert len(toks) == r.prompt_n + r.cache_n - 2


async def test_agent_behaves_as_with_mockllm(corpus, item):
    plain, pooled = MockLLM(mock_script), pool(mock_script, capacity=None)
    prof = Profile(name="p", few_shot=2)
    a = await Agent(plain, corpus, EventLog()).run("t", item, prof, slot=0)
    b = await Agent(pooled, corpus, EventLog()).run("t", item, prof, slot=0)
    assert a.status == b.status == "ok" and a.prediction == b.prediction
    assert [s.output for s in a.steps] == [s.output for s in b.steps]
    assert [c["prompt"] for c in plain.calls] == [c["prompt"] for c in pooled.calls]
    prev_s = 0
    for step in b.steps:                                    # append-only prompt: each step reuses the last KV
        llm = step.llm
        assert llm["cache_n"] == prev_s or prev_s == 0
        prev_s = llm["prompt_n"] + llm["cache_n"] + llm["predicted_n"] - 1
    assert pooled.cells(0) == prev_s
    for c, step in zip(pooled.calls, b.steps):
        assert len(await pooled.tokenize(c["prompt"])) == step.llm["prompt_n"] + step.llm["cache_n"]


async def test_agent_prompt_beyond_n_ctx_is_a_ctx_overflow(corpus, item):
    trace = await Agent(pool(mock_script, capacity=200), corpus, EventLog()).run("t", item, Profile(name="p"), slot=0)
    assert trace.status == "ctx_overflow"
