"""KV manager: agent-level KV lifecycle at tool-call boundaries (docs/phase2_spec.md §2-6).

One KVManager sits between the agents and llama-server. ManagedLLM wraps the LLM client, so
agent.py is unchanged: before every request it calls acquire() (tokenize, gate, evict,
restore), after it on_response(). Tool boundaries arrive through the EventLog listener
on_event(), from the tool_start / tool_end events the agent already emits.

Three layers, each ablatable from configs/kvm.yaml:

1. **Dispatch gate (§2.2).** Every slot carries a charge u_k, an upper bound on the cells the
   server holds for it (§2.1). `full`: a request is sent only if U - u_k + max(s_k, f) <= c - m,
   so the pool can never overflow and the server never purges. `active`: only requests in
   flight are counted, and llama.cpp's own purge reclaims idle KV (retain configurations).
   `none`: Phase 1 behaviour; the manager only observes. Charges fall only when the server's
   response arrives, never on send.
2. **Admission (§2.3, admission.py):** FIFO tickets, projected peaks with over-commit omega.
3. **Lifecycle (§4).** When the full gate fails, the head of the request queue evicts one
   victim at a time (offload = save + erase to a bounded host tier, drop to the family prefix,
   or erase), ranked by expected residual tool time, and re-plans after every op. Returning
   agents (priority 0) reclaim on demand; a new task's first request (priority 1) may take an
   idle agent's KV only once that agent's pressure-gated TTL expired (rent >= break-even B).

Concurrency. One asyncio.Lock serializes planning, victim ops and dispatch; only the head and
the manager's own proactive ops take it. The synchronous callbacks (on_event, on_response,
on_first) only mutate the ledger, bump `_epoch` and resolve the head's (and the admission
head's) wake future: asyncio.Condition.notify() would raise without the lock. Slot ops run in
manager-owned tasks awaited through asyncio.shield, so a cancelled head abandons only its
wait and the victim's ledger entry stays exact (§2.4 I4). An op that times out leaves its slot
UNKNOWN with an upper-bound charge until /slots and an erase resolve it.

Accounting (§3). Prompt sizes come from the server's own /tokenize; s_k = prompt_n + cache_n
+ predicted_n - 1 after each response; /slots is polled at 1 Hz with a per-slot version rule,
because a snapshot can predate an op or a request the client already sent.
"""
from __future__ import annotations

import asyncio
import contextvars
import itertools
import json
import math
import statistics
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from edgeserve.admission import Admission, SlotPool, Ticket, make_admission
from edgeserve.events import EventLog
from edgeserve.llm import LLMResult, SlotOpResult
from edgeserve.policies import WEAK_PRIOR, DelayModel, Victim, break_even, choose_action, rank_victims, residual
from edgeserve.profiles import PROFILES_PATH, Profile, load_profiles
from edgeserve.prompt import CHATML, react_system

BYTES_PER_TOKEN = 57_360          # f16 KV of Qwen2.5-7B per token in a slot file (E0b n_written)
FILE_HEADER_BYTES = 724
EXCEEDED = "context size has been exceeded"
REPORT_S = 1.0                    # kv_tick period
OP_WINDOW_S = 10.0                # kv_tick op_ms_10s window
RESOLVE_POLL_S = 0.1
RESOLVE_TRIES = 5                 # erase attempts before an unreachable server's slot is left UNKNOWN
RESOLVE_BACKOFF_S = 0.25          # first wait between them (doubling, at most 2 s)

FREE, ACTIVE, IDLE, READY, STALE = "FREE", "ACTIVE", "IDLE", "READY", "STALE"
SAVING, RESTORING, OFFLOADED, DROPPED = "SAVING", "RESTORING", "OFFLOADED", "DROPPED"
UNKNOWN, HOST_CACHED = "UNKNOWN", "HOST_CACHED"
SERVER_BUSY = (ACTIVE, RESTORING, UNKNOWN)   # the server may be writing these cells: the active gate counts them

# Which task a request belongs to when its slot does not say (unpinned B2 runs): set by
# admit() in the worker's context, which asyncio copies into the agent's task.
CURRENT_TASK: contextvars.ContextVar[str | None] = contextvars.ContextVar("kvm_task", default=None)

# Fallback when configs/kv_priors.json is missing: the fit of scripts/fit_kvm_priors.py on the
# Phase 1 runs without overflow (spec §2.3). G[q][n] = q-quantile of the growth still to come
# after n requests.
DEFAULT_PRIORS: dict[str, dict[str, Any]] = {
    "react_short": {"P0": 1449, "G": {"0.9": [932, 891, 727, 732, 556, 406, 193]}, "duty": 0.794},
    "react_long": {"P0": 1449, "G": {"0.9": [3460, 3428, 2498, 2927, 2551, 1794, 914]}, "duty": 0.833},
    "react_explain": {"P0": 1661, "G": {"0.9": [1117, 1074, 945, 822, 893, 708, 524, 302]}, "duty": 0.903},
    "oneshot_long": {"P0": 3265, "G": {"0.9": [62]}, "duty": 0.999},
}
GENERIC_PRIOR: dict[str, Any] = {"P0": 1500, "G": {"0.9": [3500]}, "duty": 0.8}
DUMMY_QUESTIONS = ("Which river is longer, the Nile or the Amazon?",   # first characters differ, so
                   "Are lions and tigers both big cats?")              # the LCP ends before the question


@dataclass
class KVConfig:
    ctx: int = 0                  # -c; 0 = take it from /props
    margin: int = 128
    gate: str = "full"            # none | active | full
    admission: str = "kv"         # fixed | kv | duty
    k: int = 8                    # fixed K, or K_max for kv/duty
    omega: float = 1.0
    theta: float = 0.8
    peak_q: float = 0.9
    lifecycle: str = "managed"    # observe | server | managed
    timing: str = "pressure_ttl"  # pressure_ttl | lazy | nosteal | fixed_ttl | always | clairvoyant | clairvoyant_pf
    ttl_x: float = 1.25           # fixed_ttl: multiple of the online median search duration
    delay_model: str = "online"   # online | oracle_prior
    victim: str = "residual"      # residual | lowest_id | largest
    actions: str = "cost"         # cost | offload | drop | erase
    host_frac: float = 1.0
    slot_dir: str = "/dev/shm/slots"
    tick_s: float = 0.05
    reconcile_s: float = 1.0
    op_timeout_s: float = 5.0
    prior_delay_scale: float | None = None   # oracle_prior only (None: the run's delay scale)
    priors: str = "configs/kv_priors.json"
    # Set by the runner from its own arguments, not by presets.
    pinned: bool = True           # False: the server picks slots (B2 observe)
    cache_ram_mib: float = 0.0    # the server's --cache-ram (B2 HOST_CACHED sub-mode when unpinned)
    delay_scale: float = 1.0


@dataclass
class SlotKV:      # one per server slot
    slot: int
    family: str | None = None     # prompt family of the content (what a drop to prefix keeps)
    state: str = FREE
    owner: str | None = None
    s: int = 0                    # resident cells, exact after every response and op
    charge: int = 0
    busy: bool = False            # an op is in flight (I4: requests and ops are exclusive)
    ver: int = 0                  # bumped on dispatch, response, op send, op completion (reconcile, §3)
    f: int = 0                    # reservation of the request in flight (or being restored for)
    req_out: bool = False
    op: str | None = None
    pending_free: int = 0         # charge the op in flight is expected to release
    bound: int = 0                # UNKNOWN: upper bound max(s, S_img, f)
    floor: int = 0                # reconcile: the server reported more than the ledger
    mismatch: int | None = None   # last clean server count that disagreed (undercount on repeat)
    host_tok: int = 0             # HOST_CACHED: tokens moved to llama.cpp's host cache (approximate)
    host_id: int = -1             # HOST_CACHED: the _HostEntry holding them
    op_failed: bool = False       # the last op on this slot failed or timed out
    t_state: float = 0.0
    op_task: asyncio.Task | None = field(default=None, repr=False)


@dataclass
class AgentKV:     # one per admitted task
    task_id: str
    profile: str
    slot: int
    ticket: int
    R_hat: int = 0
    n_req: int = 0                # requests dispatched; the idle period index for I5
    step: int = 0
    tool: str | None = None
    tool_t0: float = 0.0
    true_delay: float = 0.0       # tool_start.delay_s: read only by the clairvoyant timings
    B: float = 0.0
    t_star: float = 0.0
    rent: float = 0.0
    expired: bool = False
    gave_up_step: int = -1
    img_path: str | None = None
    img_tokens: int = 0
    img_bytes: int = 0
    ready_seq: int = -1
    infeasible: bool = False
    family: str | None = None
    S: int = 0                    # resident tokens frozen at tool start
    last_s: int = 0               # size after the last response (R̂ keeps using it after an eviction)
    tok_P: int = 0
    pressure_s: float = 0.0       # pressure time during the current tool call
    evictions: int = 0            # give-ups in the current idle period (I5: at most 2, a drop then an erase)
    op_tried: int = -1            # n_req of the idle period in which a manager op last ran on it
    side_err_step: int = -1       # step whose proactive op or prefetch raised (not respawned)
    started: bool = False         # reached its first acquire
    waiting: bool = False         # in the gate queue
    released: bool = False
    proactive: bool = False
    t_admit: float = 0.0
    busy_s: float = 0.0           # time with a request in flight (duty cycle)


@dataclass
class _Waiter:
    agent: AgentKV
    slot: int
    prio: int                     # 0: returning agent, 1: a new task's first request
    seq: int
    f: int                        # tok(P) + n_predict_eff
    n_eff: int
    t0: float
    fut: asyncio.Future | None = field(default=None, repr=False)
    blocked: bool = False
    deficit0: int | None = None
    victims: list[list[Any]] = field(default_factory=list)
    wakeups: int = 0
    dispatched: bool = False
    failed: dict[int, int] = field(default_factory=dict)   # slot -> ver after an op on it failed


@dataclass
class _HostEntry:  # B2: one entry of llama.cpp's --cache-ram prompt cache, as the ledger models it
    id: int
    owner: str | None             # the task whose context it is
    slot: int                     # the slot it was saved from
    tokens: int


@dataclass
class CostModel:
    """Server-seconds per op; priors from E0a/E0b (spec §1), refit online by EWMA from the
    server's own timings (save_ms, restore_ms, prompt_ms; wall time for erase and truncate)."""

    kappa_s: float = 35e-6        # save, s/token (E0b: 33-36 µs)
    kappa_r: float = 15e-6        # restore (12-16 µs)
    t_pf: float = 220e-6          # re-prefill s/token with an empty pool
    t_pf_span: float = 9.7e-6     # + s/token per 1k occupied cells of span (E0b idlecost)
    erase_s: float = 0.002
    truncate_s: float = 0.015
    alpha: float = 0.2
    min_tokens: dict[str, int] = field(default_factory=lambda: {"save": 256, "restore": 256, "prefill": 64})
    n_obs: dict[str, int] = field(default_factory=dict)

    def observe(self, op: str, tokens: int, server_ms: float | None, span: int = 0) -> None:
        if server_ms is None or server_ms < 0:
            return
        x = server_ms / 1000.0
        if op in ("erase", "truncate"):
            setattr(self, f"{op}_s", self._ewma(getattr(self, f"{op}_s"), x))
        elif tokens >= self.min_tokens.get(op, 1):
            per = x / tokens
            if op == "save":
                self.kappa_s = self._ewma(self.kappa_s, per)
            elif op == "restore":
                self.kappa_r = self._ewma(self.kappa_r, per)
            elif op == "prefill":
                self.t_pf = self._ewma(self.t_pf, max(1e-9, per - self.t_pf_span * span / 1000.0))
            else:
                return
        else:
            return
        self.n_obs[op] = self.n_obs.get(op, 0) + 1

    def _ewma(self, old: float, new: float) -> float:
        return (1 - self.alpha) * old + self.alpha * new

    def c_off(self, s: int) -> float:
        return (self.kappa_s + self.kappa_r) * s

    def c_drop(self, s: int, keep: int, span: int) -> float:
        return (self.t_pf + self.t_pf_span * span / 1000.0) * max(0, s - keep)

    def params(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("min_tokens")
        return d


def load_priors(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_PRIORS)
    d = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def family_name(p: Profile) -> str | None:
    """ReAct prompts with the same rules and examples share their first ~1.4k tokens."""
    if p.mode != "react":
        return None
    return f"react{p.few_shot}{'x' if p.explain_tokens > 0 else ''}"


def default_families(profiles: dict[str, Profile]) -> tuple[dict[str, str | None], dict[str, tuple[str, str]]]:
    """profile -> family, and two first prompts per family (agent.py's layout with dummy
    questions); their token LCP is the family prefix a drop keeps."""
    family_of: dict[str, str | None] = {}
    prompts: dict[str, tuple[str, str]] = {}
    for name, p in profiles.items():
        fam = family_of[name] = family_name(p)
        if fam is None or fam in prompts:
            continue
        head = CHATML["system"].format(content=react_system(p.few_shot, p.explain_tokens > 0)) \
            + CHATML["user_open"] + "Question: "
        tail = CHATML["user_close_assistant_open"] + "Thought 1:"
        prompts[fam] = (head + DUMMY_QUESTIONS[0] + tail, head + DUMMY_QUESTIONS[1] + tail)
    return family_of, prompts


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _num(x: float) -> float | None:
    return round(x, 6) if math.isfinite(x) else None


def _wait_stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    return {"n": len(ys), "mean_s": round(statistics.mean(ys), 4), "p95_s": round(ys[int(0.95 * (len(ys) - 1))], 4),
            "max_s": round(ys[-1], 4)}


class KVManager:
    def __init__(self, ctl: Any, events: EventLog, cfg: KVConfig, llm: Any = None, *,
                 profiles: dict[str, Profile] | None = None, priors: dict[str, Any] | None = None,
                 families: tuple[dict[str, str | None], dict[str, tuple[str, str]]] | None = None,
                 cost: CostModel | None = None) -> None:
        """`ctl`: SlotControl (or MockPoolLLM); `llm`: the LLM client the drop-to-prefix request
        goes through (ManagedLLM sets it if None). `families` overrides the prompt families
        derived from `profiles` (tests use synthetic prompts)."""
        self.ctl, self.events, self.cfg, self.llm = ctl, events, cfg, llm
        if profiles is None:
            profiles = load_profiles(PROFILES_PATH)
        self.family_of, self.family_prompts = families if families is not None else default_families(profiles)
        self.priors = priors if priors is not None else load_priors(cfg.priors)
        self.cost = cost or CostModel()
        self.delays = DelayModel()
        self.cap = cfg.ctx - cfg.margin
        self.observe = cfg.lifecycle == "observe"
        self.managed = cfg.lifecycle == "managed"
        self.host_cached = self.observe and not cfg.pinned and cfg.cache_ram_mib > 0
        self.host_cap = int(cfg.host_frac * cfg.ctx * BYTES_PER_TOKEN)
        self.adm: Admission = make_admission(cfg, self._duty)
        self.slots: dict[int, SlotKV] = {}
        self.agents: dict[str, AgentKV] = {}
        self.pool = SlotPool(())
        self.prefix_ids: dict[str, list[int]] = {}
        self.n_ctx = 0

        self._lock = asyncio.Lock()
        self._epoch = 0
        self._waiters: list[_Waiter] = []
        self._seq = itertools.count()
        self._committed: _Waiter | None = None
        self._ops: set[asyncio.Task] = set()        # manager-owned slot ops (awaited by close)
        self._side: set[asyncio.Task] = set()       # proactive ops and prefetches
        self._bg: list[asyncio.Task] = []
        self._closed = False
        self._unassigned: dict[str, int] = {}       # B2: request charges whose slot is not known yet
        self._host: list[_HostEntry] = []           # B2: the server's host prompt cache, oldest first
        self._host_ids = itertools.count()
        self._host_unmatched: set[str] = set()      # B2: launched on an empty slot without an own entry
        self.host_saves = self.host_loads = self.host_evictions = 0
        self._files: set[str] = set()
        self.n_not_started = 0
        self.host_bytes = self.host_reserved = self.host_peak = 0
        self._off_tokens, self._off_t, self._off_tok_s = 0, time.monotonic(), 0.0

        # counters for summary()
        self.violations: Counter[str] = Counter()
        self.loop_errors: Counter[str] = Counter()
        self.op_stats: dict[str, dict[str, float]] = {}
        self.op_reasons: Counter[str] = Counter()
        self.evictions: Counter[str] = Counter()
        self.give_ups = self.wasted_saves = self.restores = self.restore_failed = self.resolved = 0
        self.overflows = self.purges_seen = self.truncated = self.tokenize_fallbacks = self.expiries = 0
        self.recompute_tokens = self.admitted = self.max_agents = self.max_U = self.recon_errors = 0
        self.infeasible: list[str] = []
        self.exceeded: list[str] = []               # tok(P) >= c - m under any gate (spec §2.2 overlap)
        self.would_truncate = 0                     # observe: requests a gated run would send truncated
        self.pressure_s = self.op_server_s = self.op_wall_s = 0.0
        self._tok_s: dict[str, float] = {}
        self._gate_waits: dict[int, list[float]] = {0: [], 1: []}
        self._admit_waits: list[float] = []
        self._bind_s: Counter[str] = Counter()
        self._op_window: deque[tuple[float, float]] = deque()
        self._duty_ewma: dict[str, float] = {}
        self._stats: dict[str, dict[str, Any]] = {}

    # --- lifecycle of the manager ---------------------------------------------------------

    async def start(self) -> None:
        """Slots from /props, family prefix ids, tick and reconcile loops."""
        if self.cfg.ctx <= 0:
            raise ValueError("KVConfig.ctx is not set")
        props = await self.ctl.props()
        self.n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
        n_slots = int(props.get("total_slots") or 64)
        now = time.monotonic()
        self.slots = {k: SlotKV(k, t_state=now) for k in range(n_slots)}
        self.pool = SlotPool(range(n_slots))
        for fam, (pa, pb) in self.family_prompts.items():
            ia, ib = await self.ctl.tokenize(pa), await self.ctl.tokenize(pb)
            self.prefix_ids[fam] = ia[:_lcp(ia, ib)]
        self.events.emit("kv_start", ctx=self.cfg.ctx, cap=self.cap, n_slots=n_slots, n_ctx=self.n_ctx,
                         families={f: len(ids) for f, ids in self.prefix_ids.items()}, host_cap=self.host_cap)
        loop = asyncio.get_running_loop()
        self._bg = [loop.create_task(self._tick_loop()), loop.create_task(self._reconcile_loop())]

    async def close(self) -> None:
        """Stop the loops, let every op finish (they own their slots' ledger entries), and
        delete any slot file still on disk."""
        self._closed = True
        for t in self._bg:
            t.cancel()
        await asyncio.gather(*self._bg, return_exceptions=True)
        while self._ops or self._side:
            await asyncio.gather(*list(self._ops | self._side), return_exceptions=True)
        for a in list(self.agents.values()):
            self._drop_image(a)
        for fn in list(self._files):
            self._unlink(fn)
        self._flush()

    # --- admission ------------------------------------------------------------------------

    async def admit(self, task_id: str, profile: str, ticket: int, worker: int | None = None) -> int:
        """Wait for this ticket's turn and conditions (a)-(c); return the task's slot (-1 when
        the server picks slots). `worker` keeps Phase 1's mapping (worker k -> slot k) for
        fixed-K admission."""
        t0 = time.monotonic()
        fam = self.family_of.get(profile)
        R_j = self._R_hat(profile, 0, 0)
        tk = Ticket(ticket, task_id, R_j, t0=t0)
        self.adm.push(tk)
        try:
            while not (self.adm.head() is tk and self._can_admit(R_j)):
                tk.fut = asyncio.get_running_loop().create_future()
                await tk.fut
        finally:
            self.adm.remove(tk)
            self._kick()
        gate_q, not_started = len(self._waiters), self.n_not_started
        slot = self._pick_slot(fam, worker)
        now = time.monotonic()
        a = AgentKV(task_id, profile, slot, ticket, R_hat=R_j, family=fam, t_admit=now)
        self.agents[task_id] = a
        self.n_not_started += 1
        self.admitted += 1
        self.max_agents = max(self.max_agents, len(self.agents))
        self._admit_waits.append(now - t0)
        self._stats[task_id] = {"admit_wait_s": round(now - t0, 4), "gate_wait_s": 0.0, "n_req": 0, "give_ups": 0,
                                "offloaded": 0, "dropped": 0, "erased": 0, "restores": 0, "restore_failed": 0,
                                "wasted_saves": 0, "recompute_tokens": 0, "truncated": 0, "infeasible": False,
                                "exceeded": False, "would_truncate": 0, "peak_tokens": 0}
        CURRENT_TASK.set(task_id)
        if slot >= 0:
            sk = self.slots[slot]
            sk.owner = task_id
            self._set(sk, IDLE, f=0, reason="admit")
        self.events.emit("kv_admit", task_id=task_id, profile=profile, slot=slot, family=fam, ticket=ticket,
                         wait_s=round(now - t0, 4), R_hat=R_j, R_sum=self._R_sum(), cap=_num(self.adm.limit()),
                         n_agents=len(self.agents), gate_q=gate_q, not_started=not_started)
        self._bump()
        return slot

    def _can_admit(self, R_j: int) -> bool:
        return self.adm.can_admit([a for a in self.agents.values() if not a.released], R_j,
                                  len(self._waiters), self.n_not_started)

    def _pick_slot(self, family: str | None, worker: int | None) -> int:
        if not self.cfg.pinned:
            return -1
        if self.cfg.admission == "fixed" and worker is not None:
            return self.pool.claim(worker)
        return self.pool.take(family, busy=lambda k: self.slots[k].busy or self.slots[k].state == UNKNOWN,
                              size=lambda k: self.slots[k].s)

    def _R_hat(self, profile: str, n: int, s: int) -> int:
        """max(s + Ĝ_q(p, n), P̂0 + Ĝ_q(p, n)): the projected peak after n requests."""
        pr = self.priors.get(profile) or GENERIC_PRIOR
        G = pr["G"]
        key = str(self.cfg.peak_q)
        if key not in G:
            key = min(G, key=lambda q: abs(float(q) - self.cfg.peak_q))
        vals = G[key]
        g = int(vals[min(n, len(vals) - 1)]) if vals else 0
        return int(max(s + g, int(pr["P0"]) + g))

    def _R_sum(self) -> int:
        return sum(a.R_hat for a in self.agents.values() if not a.released)

    def _duty(self, profile: str) -> float:
        if profile in self._duty_ewma:
            return self._duty_ewma[profile]
        pr = self.priors.get(profile) or GENERIC_PRIOR
        llm_s, tool_s = pr.get("llm_s_mean"), pr.get("tool_s_unit_mean")
        if llm_s and tool_s is not None:   # the prior's duty at this run's delay scale
            return llm_s / (llm_s + self.cfg.delay_scale * tool_s)
        return float(pr.get("duty", 0.8))

    # --- the request path -----------------------------------------------------------------

    def agent_for(self, slot: int) -> AgentKV | None:
        if slot >= 0 and (sk := self.slots.get(slot)) is not None and sk.owner in self.agents:
            return self.agents[sk.owner]
        tid = CURRENT_TASK.get()
        return self.agents.get(tid) if tid else None

    async def acquire(self, slot: int, prompt: str | list[int], n_predict: int) -> int:
        """Tokenize, wait at the gate (evicting victims as needed), restore an offloaded
        context, and return n_predict_eff (<= 0: kv_infeasible). Raises nothing of its own."""
        a = self.agent_for(slot)
        if a is None:
            return n_predict
        first = not a.started
        try:
            n_tok = await self._tokenize(prompt)
        finally:
            if first:   # the task reached its first acquire (or died trying): admission (b)
                a.started = True
                self.n_not_started -= 1
        a.tok_P = n_tok
        if n_tok >= self.cap and a.task_id not in self.exceeded:   # listed for every configuration (spec §2.2)
            self.exceeded.append(a.task_id)
            self._stats[a.task_id]["exceeded"] = True
            self.events.emit("kv_exceeded", task_id=a.task_id, slot=slot, req_idx=a.n_req + 1, tok_P=n_tok,
                             cap=self.cap, gate=self.cfg.gate)
        if self.cfg.gate == "none":   # what is sent stays exactly Phase 1's
            if n_tok < self.cap < n_tok + n_predict:
                self.would_truncate += 1
                self._stats[a.task_id]["would_truncate"] += 1
            self._dispatch_observe(a, slot, n_tok + n_predict)
            return n_predict
        n_eff = min(n_predict, self.cap - n_tok)
        if n_eff <= 0:
            a.infeasible = True
            self._stats[a.task_id]["infeasible"] = True
            if a.task_id not in self.infeasible:
                self.infeasible.append(a.task_id)
            self.events.emit("kv_infeasible", task_id=a.task_id, slot=slot, tok_P=n_tok, cap=self.cap)
            self._bump()
            return n_eff
        if n_eff < n_predict:
            self.truncated += 1
            self._stats[a.task_id]["truncated"] += 1
            self.events.emit("kv_truncated", task_id=a.task_id, slot=slot, req_idx=a.n_req + 1, tok_P=n_tok,
                             n_predict=n_predict, n_predict_eff=n_eff)
        w = _Waiter(a, slot, 1 if a.n_req == 0 else 0, next(self._seq), n_tok + n_eff, n_eff, time.monotonic())
        a.waiting, a.ready_seq = True, w.seq
        sk = self.slots[slot]
        self._waiters.append(w)
        if sk.state == IDLE:
            self._set(sk, READY, reason="acquire")
        self._bump()
        try:
            await self._gate(w)
        finally:
            a.waiting = False
            if w in self._waiters:
                self._waiters.remove(w)
            if self._committed is w:
                self._committed = None
            if not w.dispatched and sk.state == READY:
                self._set(sk, IDLE, reason="cancel")
            self._bump()
        return w.n_eff

    async def _tokenize(self, prompt: str | list[int]) -> int:
        if not isinstance(prompt, str):
            return len(prompt)
        try:
            return len(await self.ctl.tokenize(prompt))
        except Exception:   # every token is at least one character: an upper bound keeps I2
            self.tokenize_fallbacks += 1
            return len(prompt)

    def _dispatch_observe(self, a: AgentKV, slot: int, f: int) -> None:
        a.n_req += 1
        self._stats[a.task_id]["n_req"] = a.n_req
        if slot < 0:
            self._unassigned[a.task_id] = f
            return
        sk = self.slots[slot]
        sk.req_out = True
        self._set(sk, ACTIVE, f=f, reason="dispatch")

    def on_first(self, task_id: str, k: int) -> None:
        """B2 observe: the first stream message names the request's slot. With --cache-ram and
        cache_idle_slots the launch has just saved and cleared every other idle slot."""
        f = self._unassigned.pop(task_id, 0)
        sk = self.slots.get(k)
        if sk is None:
            return
        if self.host_cached:
            self._host_launch(sk, task_id)
        a = self.agents.get(task_id)
        if a is not None:
            a.slot = k
        sk.owner = task_id
        sk.req_out = True
        self._set(sk, ACTIVE, f=f, reason="dispatch")
        self._bump()

    # --- B2: llama.cpp's host prompt cache (approximate: the ledger has owners, not tokens) ---

    def _host_launch(self, sk: SlotKV, task_id: str) -> None:
        """get_available_slot (sc:1891) saves the chosen slot's other content and loads the best
        entry, which leaves the cache (server-task.cpp:1912); the launch then saves and clears
        every other idle slot (sc:3071, TAG_IDLE_SLOT_CLEAR)."""
        if sk.s > 0 and sk.owner != task_id and sk.state in (IDLE, READY, STALE):
            self._host_save(sk)
        mine = [e for e in self._host if e.owner == task_id]   # its own entry: f_keep = 1, highest f_sim
        if mine:
            self._host_remove(max(mine, key=lambda e: e.tokens), "host_loaded", keep_slot=sk.slot)
            self.host_loads += 1
        elif sk.s == 0 or sk.state == HOST_CACHED:
            self._host_unmatched.add(task_id)   # a cache hit is a load of another agent's entry
        for o in self.slots.values():
            if o is not sk and o.state in (IDLE, READY, STALE) and o.s > 0:
                e = self._host_save(o)
                if e is not None:
                    o.host_tok, o.host_id = o.s, e.id
                    self._set(o, HOST_CACHED, s=0, reason="host_cached")
                else:   # over the limit or already cached: cleared all the same
                    self._set(o, FREE, s=0, reason="host_skipped")
        self._host_total()

    def _host_save(self, o: SlotKV) -> _HostEntry | None:
        """server_prompt_cache::alloc (server-task.cpp:1756): an entry the same agent saved earlier is
        contained in this one (append-only prompts) and removed; the oldest entries make room."""
        limit = int(self.cfg.cache_ram_mib * 2**20)
        n = o.s
        if n <= 0 or n * BYTES_PER_TOKEN > limit:
            return None
        if o.owner is not None and any(e.owner == o.owner and e.tokens >= n for e in self._host):
            return None
        for e in [e for e in self._host if o.owner is not None and e.owner == o.owner]:
            self._host_remove(e, "host_obsolete")
        while self._host and (sum(e.tokens for e in self._host) + n) * BYTES_PER_TOKEN > limit:
            self._host_remove(self._host[0], "host_evicted")
            self.host_evictions += 1
        e = _HostEntry(next(self._host_ids), o.owner, o.slot, n)
        self._host.append(e)
        self.host_saves += 1
        return e

    def _host_remove(self, e: _HostEntry, reason: str, keep_slot: int = -1) -> None:
        """An entry leaves the cache: the slot that still shows it as HOST_CACHED becomes FREE
        (unless it is the launching slot, which goes ACTIVE right after)."""
        self._host.remove(e)
        o = self.slots.get(e.slot)
        if o is not None and o.slot != keep_slot and o.state == HOST_CACHED and o.host_id == e.id:
            o.host_id = -1
            self._set(o, FREE, reason=reason)

    def _host_total(self) -> None:
        self.host_bytes = sum(e.tokens for e in self._host) * BYTES_PER_TOKEN
        self.host_peak = max(self.host_peak, self.host_bytes)

    def on_cancel(self, task_id: str) -> None:
        """The client gave up on a request in flight: what the slot holds is unknown until an
        erase (release) or, in observe mode, a clean /slots poll."""
        self._unassigned.pop(task_id, None)
        a = self.agents.get(task_id)
        sk = self.slots.get(a.slot) if a is not None else None
        if sk is not None and sk.req_out:
            sk.req_out = False
            sk.bound = max(sk.s, sk.f)
            self._set(sk, UNKNOWN, reason="cancel")
        self._bump()

    def on_response(self, slot: int, res: LLMResult, task_id: str | None = None) -> None:
        """Ledger update from a finished request. Sync: never awaits, never takes the lock."""
        a = self.agents.get(task_id) if task_id else self.agent_for(slot)
        if a is not None:
            self._unassigned.pop(a.task_id, None)
            a.busy_s += max(0.0, res.t_end - res.t_start)
        k = slot if slot >= 0 else res.slot
        sk = self.slots.get(k)
        if sk is None:
            self._bump()
            return
        sk.req_out = False
        if a is not None and a.task_id in self._host_unmatched:
            self._host_unmatched.discard(a.task_id)
            if res.ok and res.cache_n > 0 and self._host:   # load picks the largest f_keep: the shortest entry
                self._host_remove(min(self._host, key=lambda e: e.tokens), "host_loaded", keep_slot=k)
                self.host_loads += 1
                self._host_total()
        if res.ok:
            n_p = res.prompt_n + res.cache_n
            if a is not None and a.tok_P and n_p != a.tok_P:
                self._violation("tok_mismatch", f"{a.task_id} slot {k}: tokenize {a.tok_P} vs prompt {n_p}")
                self._recon(sk, "req_end", a.tok_P, n_p, "tok_mismatch")
            s = res.kv_tokens
            if a is not None:
                st = self._stats[a.task_id]
                if a.n_req > 1 and a.last_s:
                    lost = max(0, a.last_s - res.cache_n)
                    st["recompute_tokens"] += lost
                    self.recompute_tokens += lost
                a.last_s = s
                st["peak_tokens"] = max(st["peak_tokens"], s)
                a.R_hat = self._R_hat(a.profile, a.n_req, s)
                sk.family = a.family
            self.cost.observe("prefill", res.prompt_n, res.prompt_ms, self._span())
            self._set(sk, IDLE, s=s, f=0, reason="req_end")
        else:
            err = (res.error or "").lower()
            if EXCEEDED in err:   # the server cleared every slot whose own stream got this (sc:4724)
                self.overflows += 1
                if self.cfg.gate != "none":
                    self._violation("overflow", f"slot {k}: {res.error}"[:300])
                self._set(sk, IDLE, s=0, f=0, reason="overflow")
            elif err.startswith("http 4"):   # refused before it ran: the slot is unchanged
                self._set(sk, IDLE, f=0, reason="req_error")
            else:
                sk.bound = max(sk.s, sk.f)
                self._set(sk, UNKNOWN, reason="req_error")
        self._bump()

    # --- the gate -------------------------------------------------------------------------

    def _head(self) -> _Waiter | None:
        c = self._committed
        if c is not None and c in self._waiters:
            return c
        return min(self._waiters, key=lambda w: (w.prio, w.seq), default=None)

    def _kick(self) -> None:
        """Wake the head and the admission head only (no stampede). Sync-safe."""
        h = self._head()
        if h is not None and h.fut is not None and not h.fut.done():
            h.wakeups += 1
            h.fut.set_result(None)
        t = self.adm.head()
        if t is not None and t.fut is not None and not t.fut.done():
            t.fut.set_result(None)

    def _bump(self) -> None:
        self._epoch += 1
        self._kick()

    async def _gate(self, w: _Waiter) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if self._head() is w:
                async with self._lock:
                    if self._head() is w and await self._try_dispatch(w):
                        return
                    snap = self._epoch            # no await between the last check and the future
                    w.fut = loop.create_future()
            else:
                snap = self._epoch
                w.fut = loop.create_future()
            if self._epoch == snap:
                await w.fut

    def _deficit(self, w: _Waiter) -> int:
        sk = self.slots[w.slot]
        if self.cfg.gate == "full":
            return self._U() - sk.charge + max(sk.s, w.f) - self.cap
        # A slot keeps its old tail (e.g. a STALE slot's foreign context) until the server batches
        # the new prompt, and a processing slot cannot be purged, so the request costs max(s, f).
        return self.active_charge(exclude=sk) + max(sk.s, w.f) - self.cap

    async def _try_dispatch(self, w: _Waiter) -> bool:
        """Under the lock: evict one victim at a time until D <= 0, then dispatch. False: wait."""
        sk = self.slots[w.slot]
        while True:
            w.blocked = False
            if sk.state == UNKNOWN and not sk.busy and not sk.req_out:   # a resolve gave up: try again
                sk.busy, sk.op = True, "resolve"
                self._run_op(sk, self._resolve_unknown(sk, "recheck", None, None))
            if sk.busy or sk.state == UNKNOWN:   # its own slot has an op in flight
                w.blocked = True
                return False
            D = self._deficit(w)
            if w.deficit0 is None:
                w.deficit0 = D
            if D <= 0:
                break
            if not self.managed:                 # active gate: wait for requests in flight
                w.blocked = True
                return False
            cands = self._eligible(w)
            pend = sum(o.pending_free for o in self.slots.values() if o.busy and o is not sk)
            if sum(self._max_free(v, w) for v in cands) + pend < D:
                if self._committed is w:         # the commitment lapses: queue order is restored
                    self._committed = None
                    self._kick()
                w.blocked = True
                return False
            if pend >= D:                        # ops already in flight will cover it
                w.blocked = True
                return False
            v, action = self._next_victim(w, cands)
            self._committed = w
            freed = await self._evict(self.slots[v.slot], action, "deficit" if v.owned else "stale", w,
                                      own=not v.owned)
            w.victims.append([v.slot, action, freed])
            vs = self.slots[v.slot]
            if vs.op_failed:   # not retried by this head until the slot changes (a wasted save is no failure)
                w.failed[v.slot] = vs.ver
        if w.agent.img_path is not None and w.agent.img_tokens > w.f:   # never restore past the reservation
            self._violation("restore_bound", f"{w.agent.task_id}: image {w.agent.img_tokens} > f {w.f}")
            self._drop_image(w.agent)
        if w.agent.img_path is not None:
            await self._restore_for(w)
            if sk.state == UNKNOWN:
                w.blocked = True
                return False
        self._dispatch(w)
        return True

    def _dispatch(self, w: _Waiter) -> None:
        sk, a = self.slots[w.slot], w.agent
        now = time.monotonic()
        a.n_req += 1
        a.evictions = 0                          # a new idle period starts with this request's response
        a.waiting = False
        self._stats[a.task_id]["n_req"] = a.n_req
        sk.req_out = True
        self._set(sk, ACTIVE, f=w.f, reason="dispatch")
        w.dispatched = True
        self._waiters.remove(w)
        if self._committed is w:
            self._committed = None
        wait = now - w.t0
        self._gate_waits[w.prio].append(wait)
        self._stats[a.task_id]["gate_wait_s"] = round(self._stats[a.task_id]["gate_wait_s"] + wait, 4)
        self.events.emit("kv_gate", task_id=a.task_id, slot=w.slot, req_idx=a.n_req, prio=w.prio, seq=w.seq,
                         f=w.f, U=self._U(), cap=self.cap, deficit=w.deficit0, wait_s=round(wait, 4),
                         victims=w.victims)
        self._bump()

    # --- victims --------------------------------------------------------------------------

    def _keep(self, sk: SlotKV) -> int:
        ids = self.prefix_ids.get(sk.family) if sk.family else None
        return len(ids) if ids else 0

    def _eligible(self, w: _Waiter) -> list[Victim]:
        """E(h) from the current ledger (spec §4.2), never precomputed: ops take 100-300 ms."""
        out: list[Victim] = []
        a_h = w.agent
        behind = {x.agent.task_id for x in self._waiters if x is not w}
        now = time.monotonic()
        for sk in self.slots.values():
            if (sk.busy or sk.s <= 0 or sk.state in SERVER_BUSY or sk.state in (SAVING, HOST_CACHED)
                    or w.failed.get(sk.slot) == sk.ver):
                continue
            keep = self._keep(sk)
            if sk.slot == w.slot:   # a first request's slot may still hold a finished task's KV
                if w.prio == 1 and sk.s > w.f:
                    out.append(Victim(sk.slot, 0, sk.s, keep if sk.family == a_h.family else 0, owned=False))
                continue
            if sk.state == STALE:
                out.append(Victim(sk.slot, 1, sk.s, keep, owned=False))
                continue
            a = self.agents.get(sk.owner or "")
            if a is None or a.released or a.evictions >= 2:
                continue
            if a.tool is not None and not a.waiting:
                if w.prio == 1 and (a.evictions > 0 or not self._p1_eligible(a, now)):
                    continue
                out.append(Victim(sk.slot, 3 if a.tool == "lookup" else 2, sk.s, keep,
                                  residual=self._residual(a, now), owner=a.task_id, erase_ok=w.prio == 0))
            elif a.waiting and a.task_id in behind and w.prio == 0:
                # before its first request the slot still holds a finished task's KV: stale, never offloaded
                out.append(Victim(sk.slot, 4, sk.s, keep, ready_seq=a.ready_seq, owner=a.task_id,
                                  owned=a.n_req > 0))
        return out

    def _max_free(self, v: Victim, w: _Waiter) -> int:
        """What this head could free from v at most (coverage of D)."""
        if v.cls == 0:
            return max(0, v.s - w.f)
        if v.at_prefix:
            return v.s if v.erase_ok else 0
        if v.erase_ok or v.keep <= 0:
            return v.s
        return v.s if self._action(v) in ("offload", "erase") else v.s - v.keep

    def _action(self, v: Victim) -> str:
        span = self._span()
        return choose_action(v.s, v.keep, self._host_room(v.s), self.cost.c_off(v.s),
                             self.cost.c_drop(v.s, v.keep, span), self.cfg.actions, owned=v.owned)

    def _next_victim(self, w: _Waiter, cands: list[Victim]) -> tuple[Victim, str]:
        """In rank order: give up a tail (drop / offload, or erase without a family). A STALE slot
        already at its prefix is erased in its place: dead KV goes before any live agent (spec
        §4.2, class 1). Only when every candidate is at its prefix: erase a dropped agent's prefix
        (priority-0 heads, I5). _try_dispatch has checked that the candidates cover D."""
        ranked = rank_victims(cands, self.cfg.victim)
        for v in ranked:
            if not v.at_prefix:
                return v, self._action(v)
            if v.cls == 1 and v.erase_ok:
                return v, "erase"
        for v in ranked:
            if v.erase_ok:
                return v, "erase"
        raise AssertionError("no victim although E(h) covers D")

    def _p1_eligible(self, a: AgentKV, now: float) -> bool:
        """May a priority-1 head take this idle agent's KV (spec §4.3)?"""
        t = self.cfg.timing
        if t in ("clairvoyant", "clairvoyant_pf"):
            return a.true_delay - (now - a.tool_t0) > a.t_star
        if t == "fixed_ttl":
            return now - a.tool_t0 >= a.t_star
        if t == "always":
            return True
        return a.expired

    def _residual(self, a: AgentKV, now: float) -> float:
        elapsed = now - a.tool_t0
        if self.cfg.timing in ("clairvoyant", "clairvoyant_pf"):
            return max(0.0, a.true_delay - elapsed)
        if self.cfg.delay_model == "oracle_prior":
            return residual(a.tool, elapsed, self.cfg.prior_delay_scale or self.cfg.delay_scale)
        return self.delays.residual(a.tool, elapsed)

    def _host_room(self, s: int) -> bool:
        return self.host_bytes + self.host_reserved + s * BYTES_PER_TOKEN + FILE_HEADER_BYTES <= self.host_cap

    # --- slot ops (manager-owned) ---------------------------------------------------------

    def _run_op(self, sk: SlotKV, coro: Any) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(self._op_wrapper(sk, coro))
        sk.op_task = task
        self._ops.add(task)
        task.add_done_callback(self._ops.discard)
        return task

    async def _op_wrapper(self, sk: SlotKV, coro: Any) -> None:
        try:
            await coro
        except Exception as e:   # a bug or a transport error: the outcome is unknown
            self._violation("op_error", f"slot {sk.slot} {sk.op}: {type(e).__name__}: {e}"[:300])
            if sk.state != UNKNOWN and not self._closed:
                self._to_unknown(sk, sk.s, sk.op or "op")
        finally:
            if sk.op_task is asyncio.current_task():
                sk.op_task = None
            if sk.state != UNKNOWN:
                sk.busy, sk.op, sk.pending_free = False, None, 0
            self._bump()

    async def _evict(self, sk: SlotKV, action: str, reason: str, w: _Waiter | None, own: bool = False) -> int:
        """Give up one victim's KV and wait for it (shielded: a cancelled head leaves the op running).
        own: the cells are not the owner's context (a finished task's KV in a newly taken slot), so
        it is no give-up of the owner's and the owner's state is kept."""
        if sk.busy or sk.req_out:   # I4: never two ops, or an op and a request, on one slot
            self._violation("busy_op", f"slot {sk.slot}: {action} while {sk.op or 'a request'} is in flight")
            return 0
        a = self.agents.get(sk.owner or "")
        before = sk.charge
        sk.busy, sk.op, sk.op_failed = True, action, False
        keep = self._keep(sk) if action == "drop" else 0
        sk.pending_free = max(0, sk.s - keep)
        if a is not None and not own:   # a wasted save or a failed op is no give-up: only _gave_up counts
            a.op_tried = a.n_req
        self.evictions[action] += 1
        coro = {"offload": self._op_offload, "drop": self._op_drop, "erase": self._op_erase}[action]
        task = self._run_op(sk, coro(sk, None if own else a, reason, w))
        await asyncio.shield(task)
        return max(0, before - sk.charge)

    async def _op_offload(self, sk: SlotKV, a: AgentKV | None, reason: str, w: _Waiter | None) -> None:
        fn = f"slot_{sk.slot}.bin"
        in_tool = a is not None and a.tool is not None and not a.waiting
        period = a.n_req if a is not None else -1
        prev, s0 = sk.state, sk.s
        est = s0 * BYTES_PER_TOKEN + FILE_HEADER_BYTES
        self.host_reserved += est
        self._set(sk, SAVING, reason=reason)
        try:
            r = await self.ctl.slot_save(sk.slot, fn)
        finally:
            self.host_reserved -= est
        wasted = r.ok and in_tool and a is not None and (a.tool is None or a.waiting or a.n_req != period)
        self._log_op(sk, r, reason, w, freed=0, wasted=wasted)
        self.cost.observe("save", r.tokens, r.server_ms)
        if r.timeout or r.status == 0:
            self._to_unknown(sk, s0, "save", fn, a)
            return
        if not r.ok:
            self._set(sk, prev, reason="save_failed")
            return
        self._files.add(fn)
        self._check_op_tokens(sk, r.tokens, "save")
        if wasted or a is None or a.released:   # its tool ended during the save: the slot is intact
            self._unlink(fn)
            if wasted and a is not None:
                self.wasted_saves += 1
                self._stats[a.task_id]["wasted_saves"] += 1
            self._set(sk, READY if a is not None and a.waiting else IDLE, reason="wasted_save")
            return
        a.img_path, a.img_tokens, a.img_bytes = fn, r.tokens, r.bytes
        self.host_bytes += r.bytes
        self.host_peak = max(self.host_peak, self.host_bytes)
        self._offloaded_delta(r.tokens)
        e = await self.ctl.slot_erase(sk.slot)
        self._log_op(sk, e, reason, w, freed=e.tokens if e.ok else 0)
        self.cost.observe("erase", e.tokens, e.server_ms if e.server_ms is not None else e.wall_ms)
        if e.timeout or e.status == 0:
            self._to_unknown(sk, s0, "erase", None, a)   # the image is valid; the resolver erases again
            return
        if not e.ok:
            self._drop_image(a)
            self._set(sk, prev, reason="erase_failed")
            return
        self._check_op_tokens(sk, e.tokens, "erase")
        self._set(sk, OFFLOADED, s=0, reason=reason)
        self._gave_up(a, "offloaded")

    async def _op_drop(self, sk: SlotKV, a: AgentKV | None, reason: str, w: _Waiter | None) -> None:
        """Drop to the family prefix: /completion with the prefix ids and n_predict 0 keeps
        LCP cells and frees the tail (E0b truncate: 1,400 kept, 20 ms)."""
        ids = self.prefix_ids[sk.family or ""]
        prev, s0 = sk.state, sk.s
        t0 = time.monotonic()
        try:
            res: LLMResult | None = await asyncio.wait_for(self.llm.generate(ids, 0, [], sk.slot),
                                                           self.cfg.op_timeout_s)
        except asyncio.TimeoutError:
            res = None
        wall_ms = (time.monotonic() - t0) * 1000
        ok = res is not None and res.ok
        s_new = res.kv_tokens if ok else s0
        self._log_op(sk, None, reason, w, op="truncate", ok=ok, tokens=s_new, freed=max(0, s0 - s_new) if ok else 0,
                     server_ms=(res.prompt_ms + res.predicted_ms) if ok else None, wall_ms=wall_ms,
                     error=None if ok else ("timeout" if res is None else res.error), timeout=res is None)
        if not ok:
            if res is not None and (res.error or "").lower().startswith("http 4"):
                self._set(sk, prev, reason="truncate_failed")
            else:
                self._to_unknown(sk, s0, "truncate")
            return
        if res.prompt_n + res.cache_n != len(ids):
            self._violation("tok_mismatch", f"slot {sk.slot} truncate: {len(ids)} ids vs {res.prompt_n + res.cache_n}")
        self.cost.observe("truncate", s_new, wall_ms)
        owner = self.agents.get(sk.owner or "")
        if sk.owner is None:
            to = STALE
        elif a is None and owner is not None and owner.waiting:   # the head's own stale tail
            to = READY
        else:
            to = DROPPED
        self._set(sk, to, s=s_new, reason=reason)
        if a is not None:
            self._gave_up(a, "dropped")

    async def _op_erase(self, sk: SlotKV, a: AgentKV | None, reason: str, w: _Waiter | None) -> None:
        prev, s0 = sk.state, sk.s
        r = await self.ctl.slot_erase(sk.slot)
        self._log_op(sk, r, reason, w, freed=r.tokens if r.ok else 0)
        self.cost.observe("erase", r.tokens, r.server_ms if r.server_ms is not None else r.wall_ms)
        if r.timeout or r.status == 0:
            self._to_unknown(sk, s0, "erase", None, a)
            return
        if not r.ok:
            self._set(sk, prev, reason="erase_failed")
            return
        self._check_op_tokens(sk, r.tokens, "erase")
        owner = self.agents.get(sk.owner or "")
        if sk.owner is None or owner is None or owner.released:
            to = FREE
        elif a is None and owner.waiting:
            to = READY
        else:
            to = OFFLOADED if owner.img_path else DROPPED
        self._set(sk, to, s=0, reason=reason)
        if sk.owner is None:
            self.pool.erased(sk.slot)
        if a is not None:
            self._gave_up(a, "erased")

    async def _restore_for(self, w: _Waiter) -> None:
        sk, a = self.slots[w.slot], w.agent
        sk.busy, sk.op, sk.pending_free = True, "restore", 0
        self._set(sk, RESTORING, f=w.f, reason="restore")
        await asyncio.shield(self._run_op(sk, self._op_restore(sk, a)))

    async def _op_restore(self, sk: SlotKV, a: AgentKV) -> None:
        """Into the agent's own slot, inside its gated dispatch (or a clairvoyant prefetch). A
        failed restore clears the slot (sc:3323) and becomes a re-prefill."""
        fn = a.img_path or ""
        r = await self.ctl.slot_restore(sk.slot, fn)
        self._log_op(sk, r, "restore", None, freed=0)
        self.cost.observe("restore", r.tokens, r.server_ms)
        if r.timeout or r.status == 0:
            self._to_unknown(sk, max(a.img_tokens, sk.f), "restore", fn, a)
            return
        self._drop_image(a)
        to = READY if a.waiting else IDLE
        if r.ok:
            self.restores += 1
            self._stats[a.task_id]["restores"] += 1
            self._set(sk, to, s=r.tokens, reason="restore")
        else:
            self.restore_failed += 1
            self._stats[a.task_id]["restore_failed"] += 1
            self._set(sk, to, s=0, reason="restore_failed")

    def _to_unknown(self, sk: SlotKV, bound: int, op: str, fn: str | None = None, a: AgentKV | None = None) -> None:
        """Op timeout (§2.4 I4): the server may or may not have run it. The slot stays busy with
        an upper-bound charge until _resolve_unknown has polled /slots and erased it."""
        sk.bound = max(sk.s, bound, sk.f)
        sk.busy = True
        self._set(sk, UNKNOWN, reason="op_timeout")
        self._violation("op_timeout", f"slot {sk.slot} {op}")
        self._run_op(sk, self._resolve_unknown(sk, op, fn, a))

    async def _resolve_unknown(self, sk: SlotKV, op: str, fn: str | None, a: AgentKV | None) -> None:
        sk.op = "resolve"
        srv: dict[str, Any] = {}
        for _ in range(int(60 / RESOLVE_POLL_S)):
            try:
                srv = next((d for d in await self.ctl.slots() if d.get("id") == sk.slot), {})
            except Exception:
                srv = {}
            if not srv.get("is_processing"):
                break
            await asyncio.sleep(RESOLVE_POLL_S)
        # The erase queues behind whatever the server still has pending on the slot, so its
        # answer reflects the timed-out op's outcome.
        for i in range(RESOLVE_TRIES):
            r = await self.ctl.slot_erase(sk.slot)
            if r.ok or not (r.timeout or r.status == 0):
                break
            await asyncio.sleep(min(RESOLVE_BACKOFF_S * 2 ** i, 2.0))
        self._log_op(sk, r, "resolve", None, freed=r.tokens if r.ok else 0)   # one kv_op per resolve
        if not r.ok and (r.timeout or r.status == 0):
            # Unreachable or hung server: the slot stays UNKNOWN at its upper bound (I1 holds), so
            # release() and the run can finish; reconcile adopts n_prompt_tokens if /slots answers.
            self._violation("resolve_failed", f"slot {sk.slot} {op}: {r.error}"[:300])
            if op == "restore" and a is not None:
                self._drop_image(a)
            elif fn and (a is None or a.img_path != fn):
                self._unlink(fn)            # a save that may be partial; an offload's valid image is kept
            sk.busy, sk.op, sk.pending_free = False, None, 0
            self._bump()
            return
        s_new = 0 if r.ok else int(srv.get("n_prompt_tokens", sk.bound))
        if op == "save" and fn and a is not None and not a.released and r.ok and self._file_exists(fn):
            a.img_path, a.img_tokens = fn, sk.s                 # the save ran: a valid image of s cells
            a.img_bytes = sk.s * BYTES_PER_TOKEN + FILE_HEADER_BYTES
            self._files.add(fn)
            self.host_bytes += a.img_bytes
            self.host_peak = max(self.host_peak, self.host_bytes)
            self._offloaded_delta(a.img_tokens)
        elif op == "restore" and a is not None:
            self._drop_image(a)
        elif fn and (a is None or a.img_path != fn):
            self._unlink(fn)
        owner = self.agents.get(sk.owner or "")
        if sk.owner is None or owner is None or owner.released:
            to = FREE if s_new == 0 else STALE
        elif owner.img_path:
            to = OFFLOADED
        else:
            to = READY if owner.waiting and s_new > 0 else DROPPED
        self.resolved += 1
        self._set(sk, to, s=s_new, reason="resolved")
        if sk.owner is None and to == FREE:
            self.pool.erased(sk.slot)

    def _gave_up(self, a: AgentKV, kind: str) -> None:
        """At most one give-up event per (task, idle period): a drop then an erase is one. Called only
        once the owner's cells are freed, so a.evictions counts what I5 bounds."""
        a.evictions += 1
        self._stats[a.task_id][kind] += 1
        if a.gave_up_step != a.n_req:
            a.gave_up_step = a.n_req
            self.give_ups += 1
            self._stats[a.task_id]["give_ups"] += 1

    def _check_op_tokens(self, sk: SlotKV, n: int, op: str) -> None:
        """An op's own count (n_saved, n_erased) is authoritative for an idle slot. An UNKNOWN slot
        (a cancelled request: the server may have run part of it, or truncated the old tail at
        launch) is only bounded by its charge, so only a count above the charge breaks I2."""
        if sk.state == UNKNOWN:
            if n > sk.charge:
                self._violation("undercount", f"slot {sk.slot}: {op} found {n} > bound {sk.charge}")
                self._recon(sk, "op", sk.charge, n, "undercount")
            elif n != sk.s:
                self._recon(sk, "op", sk.s, n, "resolved")
            return
        if n == sk.s:
            return
        if n < sk.s:
            self.purges_seen += 1
            if self.cfg.gate == "full":
                self._violation("purge", f"slot {sk.slot}: {op} found {n} < ledger {sk.s}")
            self._recon(sk, "op", sk.s, n, "purge")
        else:
            self._violation("undercount", f"slot {sk.slot}: {op} found {n} > ledger {sk.s}")
            self._recon(sk, "op", sk.s, n, "undercount")

    def _log_op(self, sk: SlotKV, r: SlotOpResult | None, reason: str, w: _Waiter | None, *, freed: int,
                op: str | None = None, ok: bool | None = None, tokens: int | None = None,
                server_ms: float | None = None, wall_ms: float | None = None, error: str | None = None,
                timeout: bool = False, wasted: bool = False) -> None:
        if r is not None:
            op, ok, tokens, server_ms, wall_ms = r.op, r.ok, r.tokens, r.server_ms, r.wall_ms
            error, timeout = r.error, r.timeout
        assert op is not None
        if not ok:
            sk.op_failed = True
        st = self.op_stats.setdefault(op, {"n": 0, "ok": 0, "failed": 0, "timeouts": 0, "tokens": 0,
                                           "server_ms": 0.0, "wall_ms": 0.0})
        st["n"] += 1
        st["ok" if ok else "failed"] += 1
        st["timeouts"] += int(timeout)
        st["tokens"] += int(tokens or 0)
        st["server_ms"] = round(st["server_ms"] + (server_ms or 0.0), 3)
        st["wall_ms"] = round(st["wall_ms"] + (wall_ms or 0.0), 3)
        self.op_reasons[reason] += 1
        now = time.monotonic()
        busy_ms = server_ms if server_ms is not None else (wall_ms or 0.0)
        self.op_server_s += (server_ms or 0.0) / 1000
        self.op_wall_s += (wall_ms or 0.0) / 1000
        self._op_window.append((now, busy_ms))
        owner = self.agents.get(sk.owner or "")
        self.events.emit("kv_op", slot=sk.slot, owner=sk.owner, for_task=w.agent.task_id if w else None,
                         prio=w.prio if w else None, step=owner.n_req if owner else None, op=op, reason=reason,
                         tokens=tokens, freed=freed, bytes=r.bytes if r is not None else 0, server_ms=server_ms,
                         wall_ms=round(wall_ms or 0.0, 3), ok=ok, error=error, wasted=wasted or None)

    # --- files ----------------------------------------------------------------------------

    def _file_exists(self, fn: str) -> bool:
        files = getattr(self.ctl, "files", None)   # MockPoolLLM without slot_dir keeps files in memory
        return Path(self.cfg.slot_dir, fn).exists() or (isinstance(files, dict) and fn in files)

    def _unlink(self, fn: str) -> None:
        try:
            Path(self.cfg.slot_dir, fn).unlink(missing_ok=True)
        except OSError:
            pass
        files = getattr(self.ctl, "files", None)
        if isinstance(files, dict):
            files.pop(fn, None)
        self._files.discard(fn)

    def _drop_image(self, a: AgentKV) -> None:
        if a.img_path is None:
            return
        self._unlink(a.img_path)
        self.host_bytes -= a.img_bytes
        self._offloaded_delta(-a.img_tokens)
        a.img_path, a.img_tokens, a.img_bytes = None, 0, 0

    def _offloaded_delta(self, d: int) -> None:
        now = time.monotonic()
        self._off_tok_s += self._off_tokens * (now - self._off_t)
        self._off_t, self._off_tokens = now, self._off_tokens + d

    # --- tool boundaries and timing -------------------------------------------------------

    def on_event(self, ev: dict[str, Any]) -> None:
        """EventLog listener: tool_start / tool_end of admitted tasks. Sync."""
        typ = ev.get("type")
        if typ not in ("tool_start", "tool_end"):
            return
        a = self.agents.get(ev.get("task_id") or "")
        if a is None or a.released:
            return
        now = time.monotonic()
        if typ == "tool_start":
            a.tool = str(ev.get("tool") or "search")
            a.step, a.tool_t0 = int(ev.get("step") or 0), now
            a.true_delay = float(ev.get("delay_s") or 0.0)
            a.rent, a.expired, a.pressure_s = 0.0, False, 0.0
            if self.managed:
                self._decide(a)
        else:
            if a.tool is not None:
                self.delays.observe(a.tool, float(ev.get("idle_s", now - a.tool_t0)))
            a.tool = None
        self._bump()

    def _decide(self, a: AgentKV) -> None:
        """Freeze the break-even at tool start (spec §4.3): B = (A + S) min(c_off, c_drop), T* = B/S."""
        sk = self.slots.get(a.slot)
        if sk is None:
            return
        S = a.S = sk.s
        A = sum(o.charge for o in self.slots.values() if o.state == ACTIVE)
        keep = 0 if self.cfg.actions == "erase" else self._keep(sk)
        host = self.cfg.actions in ("cost", "offload") and self._host_room(S)
        c_off, c_drop = self.cost.c_off(S), self.cost.c_drop(S, keep, self._span())
        B, t_star = break_even(A, S, c_off, c_drop, host)
        t = self.cfg.timing
        if t == "lazy":
            B = t_star = 0.0
        elif t == "nosteal":
            B = t_star = math.inf
        elif t == "fixed_ttl":
            t_star = self._ttl()
            B = t_star * S
        a.B, a.t_star = B, t_star
        a.expired = t in ("lazy", "always")
        self.events.emit("kv_decision", task_id=a.task_id, slot=a.slot, step=a.step, tool=a.tool, S=S, A=A,
                         c_off=_num(c_off), c_drop=_num(c_drop), B=_num(B), t_star=_num(t_star))
        if t == "always" and S > 0:
            self._spawn(self._proactive(a, "always"), a)

    def _ttl(self) -> float:
        med = self.delays.median("search")
        return self.cfg.ttl_x * (med if med is not None else WEAK_PRIOR["median"])

    def _spawn(self, coro: Any, a: AgentKV) -> None:
        a.proactive = True
        task = asyncio.get_running_loop().create_task(coro)
        self._side.add(task)
        task.add_done_callback(self._side_done)

    def _side_done(self, t: asyncio.Task) -> None:
        self._side.discard(t)
        if not t.cancelled() and (e := t.exception()) is not None:
            self._loop_error("side", e)

    def _loop_error(self, where: str, e: BaseException) -> None:
        """A manager bug in a background loop or side task: counted every time, logged once per
        site and type as a kv_violation (so the run is a violation run), and the loop goes on."""
        key = f"{where}:{type(e).__name__}"
        self.loop_errors[key] += 1
        if self.loop_errors[key] == 1:
            self._violation("loop_error", f"{where}: {type(e).__name__}: {e}"[:300])

    async def _proactive(self, a: AgentKV, reason: str) -> None:
        """always / fixed_ttl: give up an idle agent's KV regardless of pressure."""
        try:
            async with self._lock:
                sk = self.slots.get(a.slot)
                if (sk is None or a.released or a.tool is None or a.waiting or sk.busy or sk.state != IDLE
                        or sk.s <= 0 or a.evictions > 0 or a.op_tried == a.n_req):
                    return
                keep = self._keep(sk)
                action = choose_action(sk.s, keep, self._host_room(sk.s), self.cost.c_off(sk.s),
                                       self.cost.c_drop(sk.s, keep, self._span()), self.cfg.actions)
                await self._evict(sk, action, reason, None)
        except Exception:
            a.side_err_step = a.step
            raise
        finally:
            a.proactive = False

    async def _prefetch(self, a: AgentKV) -> None:
        """clairvoyant_pf: restore shortly before the true tool end, if the full gate allows. The
        restore acts as the priority-0 request it precedes: it never takes cells ahead of a queued
        priority-0 waiter or a committed head (spec §2.2), so they are not left short. A skipped
        prefetch is retried by the next tick."""
        try:
            async with self._lock:
                sk = self.slots.get(a.slot)
                c = self._committed
                if (sk is None or a.released or a.img_path is None or a.waiting or a.tool is None or sk.busy
                        or sk.state != OFFLOADED or self._U() - sk.charge + a.img_tokens > self.cap
                        or (c is not None and c in self._waiters) or any(x.prio == 0 for x in self._waiters)):
                    return
                sk.busy, sk.op = True, "restore"
                self._set(sk, RESTORING, f=a.img_tokens, reason="prefetch")
                await asyncio.shield(self._run_op(sk, self._op_restore(sk, a)))
        except Exception:
            a.side_err_step = a.step
            raise
        finally:
            a.proactive = False

    async def _tick_loop(self) -> None:
        last = last_report = time.monotonic()
        while True:
            await asyncio.sleep(self.cfg.tick_s)
            now = time.monotonic()
            try:
                self._tick(now, now - last)
            except Exception as e:   # CancelledError is no Exception: close() still stops the loop
                self._loop_error("tick", e)
            last = now
            if now - last_report >= REPORT_S:
                try:
                    self._report(now, now - last_report)
                except Exception as e:
                    self._loop_error("report", e)
                last_report = now

    def _tick(self, now: float, dt: float) -> None:
        """Rent and expiry without the lock (spec §4.3): rho_i = ∫ S_i p(t) dt, where p(t) = 1
        iff a priority-1 head is blocked at the gate. Expiry only makes an agent eligible."""
        h = self._head()
        pressure = h is not None and h.prio == 1 and h.blocked
        if pressure:
            self.pressure_s += dt
        t = self.cfg.timing
        expired = False
        if not self.managed:   # observe / server lifecycles: no break-even was frozen, nothing expires
            return
        for a in list(self.agents.values()):
            if a.tool is None or a.waiting or a.released or a.proactive:
                continue
            sk = self.slots.get(a.slot)
            if sk is None or sk.busy:
                continue
            resident = sk.state == IDLE and sk.s > 0 and a.evictions == 0
            if t == "pressure_ttl" and pressure and resident and not a.expired:
                a.rent += a.S * dt
                a.pressure_s += dt
                if a.rent >= a.B:
                    a.expired = expired = True
                    self.expiries += 1
                    self.events.emit("kv_expire", task_id=a.task_id, slot=a.slot, step=a.step, rent=round(a.rent, 3),
                                     B=_num(a.B), elapsed_s=round(now - a.tool_t0, 4),
                                     pressure_s=round(a.pressure_s, 4), t_star=_num(a.t_star))
            elif (t == "fixed_ttl" and resident and a.op_tried != a.n_req and a.side_err_step != a.step
                  and now - a.tool_t0 >= a.t_star):
                self._spawn(self._proactive(a, "ttl"), a)
            elif (t == "clairvoyant_pf" and a.img_path and sk.state == OFFLOADED and a.side_err_step != a.step
                  and now >= a.tool_t0 + a.true_delay - self.cost.kappa_r * a.img_tokens - 0.05):
                self._spawn(self._prefetch(a), a)
        if expired:
            self._bump()

    def _report(self, now: float, dt: float) -> None:
        while self._op_window and now - self._op_window[0][0] > OP_WINDOW_S:
            self._op_window.popleft()
        by: Counter[str] = Counter()
        for sk in self.slots.values():
            by[sk.state] += sk.charge
        h = self._head()
        bind = self.adm.blocked if self.adm.queue else None
        if bind:
            self._bind_s[bind] += dt
        self.events.emit("kv_tick", U=self._U(), cap=self.cap, active=by[ACTIVE] + by[RESTORING],
                         idle=by[IDLE] + by[DROPPED] + by[SAVING], ready=by[READY], stale=by[STALE],
                         unknown=by[UNKNOWN], offloaded_tok=self._off_tokens, host_bytes=self.host_bytes,
                         agents=len(self.agents), R_sum=self._R_sum(), gate_q=len(self._waiters),
                         admit_q=len(self.adm.queue), pressure=bool(h is not None and h.prio == 1 and h.blocked),
                         op_ms_10s=round(sum(ms for _, ms in self._op_window), 3), bind=bind)

    # --- reconcile ------------------------------------------------------------------------

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.reconcile_s)
            try:
                await self._reconcile_once()
            except Exception as e:
                self._loop_error("reconcile", e)

    async def _reconcile_once(self) -> None:
        """Version rule (§3): compare only slots unchanged since the poll was sent, with no
        request or op outstanding at send or receipt, that the server reports idle."""
        sent = {k: (sk.ver, sk.req_out or sk.busy) for k, sk in self.slots.items()}
        try:
            data = await self.ctl.slots()
        except Exception:
            self.recon_errors += 1
            return
        for d in data:
            sk = self.slots.get(d.get("id", -1))
            if sk is None or sk.slot not in sent:
                continue
            ver0, out0 = sent[sk.slot]
            if sk.ver != ver0 or out0 or sk.req_out or sk.busy or d.get("is_processing"):
                continue
            if "n_prompt_tokens" not in d:   # a slot that never ran a task: unknown, not 0
                continue
            self._compare(sk, int(d["n_prompt_tokens"]))

    def _compare(self, sk: SlotKV, srv: int) -> None:
        led = sk.s
        if srv == led:
            if sk.floor or sk.mismatch is not None:
                sk.floor, sk.mismatch = 0, None
                sk.charge = self._charge(sk)
            return
        owner = self.agents.get(sk.owner or "")
        if sk.state == UNKNOWN or (self.observe and srv > led):   # observe: adopt the server's count
            self._recon(sk, "slots_poll", led, srv, "adopt")
            to = (STALE if srv > 0 else FREE) if owner is None else IDLE
            self._set(sk, to, s=srv, reason="adopt")
            return
        if srv < led:   # the server purged an idle slot
            self.purges_seen += 1
            self._recon(sk, "slots_poll", led, srv, "purge")
            if self.cfg.gate == "full":
                self._violation("purge", f"slot {sk.slot}: server {srv} < ledger {led}")
            if owner is None:
                to = STALE if srv > 0 else FREE
            else:
                to = DROPPED if sk.state in (IDLE, READY, DROPPED) else sk.state
            self._set(sk, to, s=srv, reason="server_purge")
            if to == FREE:
                self.pool.erased(sk.slot)
            return
        # The server holds more than the ledger: never lowered, never raised silently.
        action = "undercount" if sk.mismatch == srv else "raise"
        if action == "undercount":
            self._violation("undercount", f"slot {sk.slot}: server {srv} > ledger {led} on two clean polls")
        sk.mismatch, sk.floor = srv, srv
        sk.charge = self._charge(sk)
        self._recon(sk, "slots_poll", led, srv, action)

    def _recon(self, sk: SlotKV, source: str, ledger: int, server: int, action: str) -> None:
        self.events.emit("kv_recon", slot=sk.slot, source=source, ledger=ledger, server=server,
                         diff=server - ledger, action=action)

    # --- release --------------------------------------------------------------------------

    async def release(self, task_id: str, status: str) -> None:
        """Task done (any status). Ops on the slot finish first; a cancelled request's slot is
        erased (n_erased is the truth); otherwise the KV stays as STALE for the next task of
        the same family. The runner calls this through asyncio.shield."""
        a = self.agents.get(task_id)
        if a is None:
            return
        a.released, a.tool = True, None
        if not a.started:
            a.started = True
            self.n_not_started -= 1
        self._stats[task_id]["status"] = status
        sk = self.slots.get(a.slot)
        while sk is not None and sk.op_task is not None and not sk.op_task.done():
            await asyncio.shield(sk.op_task)
        self._drop_image(a)
        if sk is not None and sk.owner == task_id:
            if sk.state == UNKNOWN and not self.observe and not sk.busy:
                sk.busy, sk.op, sk.pending_free = True, "erase", sk.charge
                await asyncio.shield(self._run_op(sk, self._op_erase(sk, None, "cancel", None)))
                while sk.op_task is not None and not sk.op_task.done():
                    await asyncio.shield(sk.op_task)
            if sk.state != UNKNOWN:
                to = STALE if sk.s > 0 else FREE
                self._set(sk, to, f=0, reason="release")
            sk.owner = None
            if self.cfg.pinned:
                self.pool.give(sk.slot, stale=sk.s > 0 or sk.state == UNKNOWN, family=sk.family)
        for o in self.slots.values():   # B2: slots this task's requests landed on
            if o.owner == task_id:
                if o.state in (IDLE, READY):
                    self._set(o, STALE if o.s > 0 else FREE, reason="release")
                o.owner = None
        span = time.monotonic() - a.t_admit
        if span > 0:
            d = min(1.0, a.busy_s / span)
            old = self._duty_ewma.get(a.profile)
            self._duty_ewma[a.profile] = d if old is None else 0.8 * old + 0.2 * d
        del self.agents[task_id]
        self._bump()

    # --- ledger ---------------------------------------------------------------------------

    def _charge(self, sk: SlotKV) -> int:
        if sk.state in (ACTIVE, RESTORING):
            c = max(sk.s, sk.f)
        elif sk.state == UNKNOWN:
            c = max(sk.s, sk.bound, sk.f)
        else:
            c = sk.s
        return max(c, sk.floor)

    def _U(self) -> int:
        return sum(sk.charge for sk in self.slots.values()) + sum(self._unassigned.values())

    @property
    def U(self) -> int:
        return self._U()

    def active_charge(self, exclude: SlotKV | None = None) -> int:
        """What the active gate counts: requests in flight at their charge max(s, f), unknown slots
        at their bound. A reused slot's old tail is removed only when the server adds that slot's
        prompt to a batch (sc:4351); if earlier slots fill n_batch first, the tail stays while
        the slot is processing and cannot be purged (seen in p2_tune adm_retain, 3 overflows)."""
        return sum(max(sk.charge, sk.floor) if sk.state == ACTIVE else sk.charge
                   for sk in self.slots.values() if sk.state in SERVER_BUSY and sk is not exclude)

    def _span(self) -> int:
        return min(self.cfg.ctx, self._U())

    def _set(self, sk: SlotKV, state: str | None = None, *, s: int | None = None, f: int | None = None,
             reason: str = "") -> None:
        """Every ledger mutation: token-seconds by state, version, charge, kv_state."""
        now = time.monotonic()
        old_state, old_s = sk.state, sk.s
        self._tok_s[old_state] = self._tok_s.get(old_state, 0.0) + sk.charge * (now - sk.t_state)
        sk.t_state = now
        if state is not None:
            sk.state = state
        if s is not None:   # an exact count replaces any reconcile floor
            sk.s, sk.floor, sk.mismatch = s, 0, None
        if f is not None:
            sk.f = f
        if sk.state != HOST_CACHED:
            sk.host_tok, sk.host_id = 0, -1
        sk.ver += 1
        sk.charge = self._charge(sk)
        self.max_U = max(self.max_U, self._U())
        if sk.state != old_state or sk.s != old_s:
            owner = self.agents.get(sk.owner or "")
            if sk.state in (OFFLOADED, RESTORING):   # host-tier tokens: kv_timeline's offloaded series
                tokens = owner.img_tokens if owner is not None and owner.img_path else sk.s
            elif sk.state == HOST_CACHED:
                tokens = sk.host_tok
            else:
                tokens = sk.s
            self.events.emit("kv_state", task_id=sk.owner, slot=sk.slot, frm=old_state, to=sk.state, tokens=tokens,
                             resident=sk.s, charge=sk.charge, reason=reason)

    def _violation(self, kind: str, detail: str) -> None:
        self.violations[kind] += 1
        self.events.emit("kv_violation", kind=kind, detail=detail)

    def _flush(self) -> None:
        now = time.monotonic()
        for sk in self.slots.values():
            self._tok_s[sk.state] = self._tok_s.get(sk.state, 0.0) + sk.charge * (now - sk.t_state)
            sk.t_state = now
        self._offloaded_delta(0)

    # --- reporting ------------------------------------------------------------------------

    def task_stats(self, task_id: str) -> dict[str, Any]:
        return dict(self._stats.get(task_id, {}))

    def summary(self) -> dict[str, Any]:
        self._flush()
        # EventLog writes a listener's exception itself (the manager never emits that kind)
        viol = self.violations + Counter(e["kind"] for e in self.events.events
                                         if e.get("type") == "kv_violation" and e.get("kind") == "listener_error")
        return {
            "ctx": self.cfg.ctx, "cap": self.cap, "n_ctx": self.n_ctx,
            "violations": dict(viol), "loop_errors": dict(self.loop_errors),
            "overflows": self.overflows, "server_purges": self.purges_seen,
            "kv_infeasible": list(self.infeasible), "kv_truncated": self.truncated,
            "kv_exceeded": list(self.exceeded), "kv_would_truncate": self.would_truncate,
            "admitted": self.admitted, "max_agents": self.max_agents,
            "give_ups": self.give_ups, "evictions": dict(self.evictions), "wasted_saves": self.wasted_saves,
            "restores": self.restores, "restore_failed": self.restore_failed, "unknown_resolved": self.resolved,
            "ops": self.op_stats, "op_reasons": dict(self.op_reasons),
            "op_server_s": round(self.op_server_s, 3), "op_wall_s": round(self.op_wall_s, 3),
            "expiries": self.expiries, "pressure_s": round(self.pressure_s, 3),
            "recompute_tokens": self.recompute_tokens, "tokenize_fallbacks": self.tokenize_fallbacks,
            "reconcile_errors": self.recon_errors,
            "max_U": self.max_U, "host_cap_bytes": self.host_cap, "host_peak_bytes": self.host_peak,
            **({"host_cache": {"saves": self.host_saves, "loads": self.host_loads, "evictions": self.host_evictions,
                               "entries": len(self._host), "bytes": self.host_bytes}} if self.host_cached else {}),
            "token_s": {k: round(v, 1) for k, v in sorted(self._tok_s.items())},
            "offloaded_token_s": round(self._off_tok_s, 1),
            "gate_wait": {f"p{p}": _wait_stats(v) for p, v in self._gate_waits.items()},
            "admit_wait": _wait_stats(self._admit_waits),
            "admission_bind_s": {k: round(v, 2) for k, v in self._bind_s.items()},
            "cost_model": self.cost.params(),
            "delay_obs": {t: len(v) for t, v in self.delays.obs.items()},
            "families": {f: len(ids) for f, ids in self.prefix_ids.items()},
        }


class ManagedLLM:
    """Same generate() as LlamaServerClient, routed through the manager, so agent.py is unchanged."""

    def __init__(self, base: Any, kvm: KVManager) -> None:
        self.base, self.kvm = base, kvm
        if kvm.llm is None:
            kvm.llm = base

    async def props(self) -> dict[str, Any]:
        return await self.base.props()

    async def close(self) -> None:
        await self.base.close()

    async def generate(self, prompt: str | list[int], n_predict: int, stop: list[str], slot: int = -1,
                       ignore_eos: bool = False) -> LLMResult:
        kvm = self.kvm
        a = kvm.agent_for(slot)
        if a is None:
            return await self.base.generate(prompt, n_predict, stop, slot, ignore_eos)
        t0 = time.monotonic()
        n_eff = await kvm.acquire(slot, prompt, n_predict)
        if n_eff <= 0:
            return LLMResult(ok=False, error=f"kv_infeasible: context needs {a.tok_P} >= c-m = {kvm.cap}",
                             slot=slot, t_start=t0, t_end=time.monotonic())
        if slot < 0:   # unpinned (B2): the stream names the slot
            task_id = a.task_id
            call = self.base.generate(prompt, n_eff, stop, slot, ignore_eos,
                                      on_first=lambda k: kvm.on_first(task_id, k))
        else:
            call = self.base.generate(prompt, n_eff, stop, slot, ignore_eos)
        try:
            res = await call
        except BaseException:
            kvm.on_cancel(a.task_id)
            raise
        kvm.on_response(slot, res, a.task_id)
        return res
