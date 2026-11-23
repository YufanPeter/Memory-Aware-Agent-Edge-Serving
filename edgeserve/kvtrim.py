"""Selective KV: map text spans to delete onto the token ranges of a slot's KV (patched server's trim).

The client holds text; the server's slot holds tokens. To delete a sentence from a slot's KV in
place (`LlamaServerClient.slot_trim`), the client must name the token range that sentence
occupies, and afterwards send prompts whose tokenization equals the slot's remaining tokens, or
the prefix cache misses and the rest is recomputed. So:

1. tokenize the slot's text with pieces (`LlamaServerClient.tokenize`) and turn the pieces into
   byte boundaries;
2. a span can be deleted only if both its ends fall on token boundaries (a sentence usually
   starts with a space, which starts a token). Two snaps fix the common misses: a start just
   after a space moves onto the space (the space was merged into the next word's token), and a
   span covering whole lines takes its trailing newline (". \n" is often one token). Spans that
   still miss are skipped and counted;
3. the trimmed text is the text without the accepted spans, and the expected tokens are the
   original tokens without the deleted ranges; `check()` compares them with the re-tokenized
   trimmed text (BPE could merge tokens across a new junction), and `plan_verified()` drops the
   spans around a mismatch and plans again.
"""
from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

Token = tuple[int, bytes]   # (id, bytes of its piece)


@dataclass
class TrimPlan:
    ranges: list[tuple[int, int]] = field(default_factory=list)   # token ranges [p0, p1), merged
    spans: list[tuple[int, int]] = field(default_factory=list)    # char spans actually removed
    new_text: str = ""
    expected_ids: list[int] = field(default_factory=list)
    n_spans: int = 0              # spans asked for
    n_skipped: int = 0            # spans whose ends are not token boundaries
    skipped_start: int = 0        # ... because the start is inside a token
    skipped_end: int = 0          # ... because the end is inside a token
    skipped_examples: list[str] = field(default_factory=list)
    n_snapped: int = 0            # spans aligned only after a snap
    n_retok_dropped: int = 0      # spans dropped by plan_verified because re-tokenization differed
    chars_asked: int = 0
    chars_removed: int = 0

    @property
    def n_tokens(self) -> int:
        return sum(b - a for a, b in self.ranges)


class AlignmentError(ValueError):
    """The tokens do not spell the text (e.g. a BOS token the text does not contain)."""


def char_to_byte(text: str) -> list[int]:
    """Byte offset of every character position 0..len(text) in the UTF-8 encoding."""
    out = [0]
    for ch in text:
        out.append(out[-1] + len(ch.encode("utf-8")))
    return out


def plan_trim(text: str, spans: list[tuple[int, int]], tokens: list[Token]) -> TrimPlan:
    """Token ranges for the char spans of `text` that fall on token boundaries."""
    byte_of = char_to_byte(text)
    bounds = [0]
    for _, piece in tokens:
        bounds.append(bounds[-1] + len(piece))
    if bounds[-1] != byte_of[-1]:
        raise AlignmentError(f"tokens cover {bounds[-1]} bytes, text has {byte_of[-1]}")
    tok_at = {b: i for i, b in enumerate(bounds)}

    plan = TrimPlan(n_spans=len(spans))
    accepted: list[tuple[int, int, int, int]] = []     # (p0, p1, c0, c1)
    for c0, c1 in sorted(spans):
        if c1 <= c0:
            continue
        plan.chars_asked += c1 - c0
        p0, p1 = tok_at.get(byte_of[c0]), tok_at.get(byte_of[c1])
        snapped = False
        if p0 is None and c0 > 0 and text[c0 - 1] == " " and byte_of[c0 - 1] in tok_at:
            c0, p0, snapped = c0 - 1, tok_at[byte_of[c0 - 1]], True
        line_start = c0 == 0 or text[c0 - 1] == "\n"
        if p1 is None and line_start and text[c1:c1 + 1] == "\n" and byte_of[c1 + 1] in tok_at:
            c1, p1, snapped = c1 + 1, tok_at[byte_of[c1 + 1]], True
        if p0 is None or p1 is None:
            plan.n_skipped += 1
            plan.skipped_start += p0 is None
            plan.skipped_end += p1 is None
            if len(plan.skipped_examples) < 5:
                plan.skipped_examples.append(repr(text[max(0, c0 - 6):c0] + "|" + text[c0:c0 + 12]) + " ... "
                                             + repr(text[c1 - 12:c1] + "|" + text[c1:c1 + 6]))
            continue
        plan.n_snapped += snapped
        accepted.append((p0, p1, c0, c1))
    # merge adjacent or overlapping ranges (adjacent sentences become one range)
    for p0, p1, c0, c1 in accepted:
        if plan.ranges and p0 <= plan.ranges[-1][1]:
            q0, q1 = plan.ranges[-1]
            s0, s1 = plan.spans[-1]
            plan.ranges[-1] = (q0, max(q1, p1))
            plan.spans[-1] = (s0, max(s1, c1))
        else:
            plan.ranges.append((p0, p1))
            plan.spans.append((c0, c1))
    pieces, last = [], 0
    for c0, c1 in plan.spans:
        pieces.append(text[last:c0])
        plan.chars_removed += c1 - c0
        last = c1
    pieces.append(text[last:])
    plan.new_text = "".join(pieces)
    ids = [t for t, _ in tokens]
    keep, last = [], 0
    for p0, p1 in plan.ranges:
        keep += ids[last:p0]
        last = p1
    keep += ids[last:]
    plan.expected_ids = keep
    return plan


def check(plan: TrimPlan, retokenized: list[Token]) -> int:
    """-1 if the trimmed text tokenizes to the expected tokens, else the first differing index."""
    got = [t for t, _ in retokenized]
    if got == plan.expected_ids:
        return -1
    n = min(len(got), len(plan.expected_ids))
    return next((i for i in range(n) if got[i] != plan.expected_ids[i]), n)


async def plan_verified(tokenize: Callable[[str], Awaitable[list[Token]]], text: str,
                        spans: list[tuple[int, int]], tokens: list[Token] | None = None,
                        max_rounds: int = 4) -> TrimPlan:
    """plan_trim, then re-tokenize the trimmed text; where it differs from the expected tokens,
    drop the spans of the deleted range nearest the difference and plan again."""
    tokens = tokens if tokens is not None else await tokenize(text)
    spans = sorted(spans)
    dropped = 0
    for _ in range(max_rounds):
        plan = plan_trim(text, spans, tokens)
        plan.n_retok_dropped = dropped
        if not plan.ranges:
            return plan
        m = check(plan, await tokenize(plan.new_text))
        if m == -1:
            return plan
        # junction of each deleted range in the trimmed token sequence
        junctions, removed = [], 0
        for p0, p1 in plan.ranges:
            junctions.append(p0 - removed)
            removed += p1 - p0
        k = min(range(len(junctions)), key=lambda i: abs(junctions[i] - m))
        s0, s1 = plan.spans[k]
        keep = [(a, b) for a, b in spans if not (a >= s0 - 1 and b <= s1)]
        dropped += len(spans) - len(keep)
        spans = keep
    plan = plan_trim(text, [], tokens)
    plan.n_retok_dropped = dropped
    return plan


_TITLE = re.compile(r"^\[.+\] $")


def segment_spans(segments: list[tuple[str, str]], drop: set[int],
                  whole_paragraphs: bool = True) -> list[tuple[int, int]]:
    """Char spans (in the concatenated segment text) of the segments in `drop`. With
    whole_paragraphs, a paragraph ("[Title] " + sentences) whose sentences are all dropped goes as
    one span with its title and the "\\n" separator after it, which aligns to tokens far better
    than its sentences one by one."""
    starts, pos = [], 0
    for _, text in segments:
        starts.append(pos)
        pos += len(text)
    end_of = lambda k: starts[k] + len(segments[k][1])  # noqa: E731
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(segments):
        kind, text = segments[i]
        if whole_paragraphs and kind == "obs_meta" and _TITLE.match(text):
            j = i + 1
            while j < len(segments) and segments[j][0] in ("obs_sf", "obs_other"):
                j += 1
            content = range(i + 1, j)
            if content and all(k in drop for k in content):
                last = j if j < len(segments) and segments[j] == ("obs_meta", "\n") else j - 1
                spans.append((starts[i], end_of(last)))
            else:
                spans += [(starts[k], end_of(k)) for k in content if k in drop]
            i = j
            continue
        if i in drop:
            spans.append((starts[i], end_of(i)))
        i += 1
    return spans


def observation_spans(segments: list[tuple[str, str]], kinds: tuple[str, ...] = ("obs_other",),
                      keep_last: int = 1, whole_paragraphs: bool = True) -> list[tuple[int, int]]:
    """segment_spans of the given segment kinds inside every observation except the last
    `keep_last` ones (observations start at a control segment beginning with "\\nObservation")."""
    obs_of, cur, n_obs = [], -1, 0
    for kind, text in segments:
        if kind == "control":
            cur, n_obs = (n_obs, n_obs + 1) if text.startswith("\nObservation") else (-1, n_obs)
            obs_of.append(-1)
        else:
            obs_of.append(cur)
    drop = {i for i, ((kind, _), o) in enumerate(zip(segments, obs_of))
            if kind in kinds and 0 <= o < n_obs - keep_last}
    return segment_spans(segments, drop, whole_paragraphs)


def apply_spans(segments: list[tuple[str, str]], spans: list[tuple[int, int]]) -> list[tuple[str, str]]:
    """The segments with the char spans removed; segments that become empty are dropped
    (zero-length marker segments are kept)."""
    out, pos, spans = [], 0, sorted(spans)
    for kind, text in segments:
        a, b = pos, pos + len(text)
        pos = b
        if not text:
            out.append((kind, text))
            continue
        keep, last = [], a
        for s0, s1 in spans:
            if s1 <= a or s0 >= b:
                continue
            keep.append(text[last - a:max(s0, a) - a])
            last = min(s1, b)
        keep.append(text[last - a:])
        new = "".join(keep)
        if new:
            out.append((kind, new))
    return out


# --- selective KV for a running agent ------------------------------------------------------------

TRIMMED = "obs_trimmed"   # zero-length marker segment: this observation was already processed


@dataclass
class TrimResult:
    mechanism: str
    ok: bool = True
    error: str | None = None
    observations: int = 0         # observations processed this time
    sentences: int = 0            # sentences chosen for deletion
    sf_sentences: int = 0         # ... of which supporting facts (quality risk)
    spans: int = 0
    skipped: int = 0              # spans not on token boundaries (kv) — left in place
    retok_dropped: int = 0        # spans dropped because re-tokenization differed (kv)
    tokens: int = 0               # tokens deleted from the slot (kv)
    chars: int = 0                # chars removed from the prompt
    server_ms: float = 0.0
    wall_ms: float = 0.0


class SelectiveKV:
    """Deletes the unhelpful sentences of an agent's older observations while it waits on a tool.

    mechanism "kv":   delete their tokens from the slot's KV in place (patched server's trim) and
                      remove the same text from the agent's prompt, so the next request reuses the
                      whole remaining slot;
    mechanism "text": only remove the text; the next request re-prefills from the first change
                      (the usual "trim old tool outputs" approach, the baseline).
    Sentences are chosen by edgeserve.selection.select_drop (scorer, keep share, keep_last newest
    observations untouched); each observation is processed once.

    trigger "always":   trim at every tool call;
    trigger "pressure": trim only while the KV manager reports memory pressure (tasks waiting for
                        admission or room, or the pool above its pressure share). Observations left
                        alone stay unprocessed and are trimmed at a later tool call under pressure."""

    def __init__(self, llm: Any, mechanism: str = "kv", scorer: str = "oracle", keep: float = 0.3,
                 keep_last: int = 1, trigger: str = "always") -> None:
        if mechanism not in ("kv", "text"):
            raise ValueError(f"unknown mechanism {mechanism!r}")
        if trigger not in ("always", "pressure"):
            raise ValueError(f"unknown trigger {trigger!r}")
        self.llm, self.mechanism, self.scorer, self.keep, self.keep_last = llm, mechanism, scorer, keep, keep_last
        self.trigger = trigger

    async def trim(self, slot: int, p: Any) -> TrimResult | None:
        from edgeserve.selection import select_drop   # local import: selection imports prompt, not us
        question = next((t for k, t in p.segments if k == "question"), "")
        drop, obs = select_drop(p, question, self.scorer, self.keep, self.keep_last, marker=TRIMMED)
        if not obs:
            return None
        t0 = time.monotonic()
        res = TrimResult(self.mechanism, observations=len(obs), sentences=len(drop),
                         sf_sentences=sum(p.segments[i][0] == "obs_sf" for i in drop))
        # zero-length markers change no offsets: insert them first, then cut by char spans
        ends = {b for _, b in obs}
        segs = []
        for i, seg in enumerate(p.segments):
            if i in ends:
                segs.append((TRIMMED, ""))
            segs.append(seg)
        if len(p.segments) in ends:
            segs.append((TRIMMED, ""))
        shift = lambda i: i + sum(e <= i for e in ends)  # noqa: E731  (index after marker insertion)
        spans = segment_spans(segs, {shift(i) for i in drop})
        res.spans = len(spans)
        if spans and self.mechanism == "kv" and slot >= 0:
            text = "".join(t for _, t in segs)
            try:
                plan = await plan_verified(self.llm.tokenize, text, spans)
            except Exception as e:   # noqa: BLE001  tokenizer unreachable or text/tokens disagree
                res.ok, res.error, spans = False, f"{type(e).__name__}: {e}", []
            else:
                res.skipped, res.retok_dropped = plan.n_skipped, plan.n_retok_dropped
                spans = plan.spans
                if plan.ranges:
                    r = await self.llm.slot_trim(slot, plan.ranges)
                    res.server_ms = r.server_ms
                    if r.ok:
                        res.tokens = plan.n_tokens
                    else:   # nothing was deleted on the server: keep the text as it is
                        res.ok, res.error, spans = False, r.error, []
        res.chars = sum(b - a for a, b in spans)
        p.segments[:] = apply_spans(segs, spans)
        res.wall_ms = round((time.monotonic() - t0) * 1000, 2)
        return res
