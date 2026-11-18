"""edgeserve.kvtrim: text spans -> token ranges for the patched server's trim action."""
import asyncio
import re

import pytest

from edgeserve.kvtrim import AlignmentError, check, observation_spans, plan_trim, plan_verified


def toks(text: str) -> list[tuple[int, bytes]]:
    """Toy tokenizer: a token is an optional leading space plus a run of non-spaces."""
    return [(hash(p) % 10_000, p.encode()) for p in re.findall(r" ?\S+|\s", text)]


TEXT = "Obs: A one. B two. C three. D four."


def span(sub: str) -> tuple[int, int]:
    i = TEXT.index(sub)
    return i, i + len(sub)


def test_aligned_spans_become_token_ranges():
    plan = plan_trim(TEXT, [span(" B two."), span(" D four.")], toks(TEXT))
    assert plan.n_skipped == 0
    assert plan.new_text == "Obs: A one. C three."
    assert check(plan, toks(plan.new_text)) == -1
    assert plan.n_tokens == 4 and plan.chars_removed == len(" B two.") + len(" D four.")


def test_adjacent_spans_merge():
    plan = plan_trim(TEXT, [span(" B two."), span(" C three.")], toks(TEXT))
    assert len(plan.ranges) == 1 and plan.new_text == "Obs: A one. D four."


def test_misaligned_span_is_skipped():
    i = TEXT.index("wo.")                # starts inside the token " two."
    plan = plan_trim(TEXT, [(i, i + len("wo. C three."))], toks(TEXT))
    assert plan.n_skipped == 1 and plan.skipped_start == 1 and plan.ranges == [] and plan.new_text == TEXT


def test_start_after_a_space_snaps_onto_it():
    i = TEXT.index("B two.")             # the space belongs to the token " B"
    plan = plan_trim(TEXT, [(i, i + len("B two."))], toks(TEXT))
    assert plan.n_snapped == 1 and plan.new_text == "Obs: A one. C three. D four."


def test_whole_line_takes_its_newline():
    text = "Obs: [T1] a.\n[T2] b c.\nend"
    def nl_toks(t):                      # "." and "\n" merge into one token, like Qwen's ".\n"
        return [(hash(p) % 10_000, p.encode()) for p in re.findall(r"\.\n| ?[^\s.]+|\.|\s", t)]
    i = text.index("[T2]")
    plan = plan_trim(text, [(i, text.index("\nend"))], nl_toks(text))
    assert plan.n_snapped == 1 and plan.new_text == "Obs: [T1] a.\nend"


def test_unicode_offsets():
    text = "Obs: Zürich é. B two."
    i = text.index(" B two.")
    plan = plan_trim(text, [(i, len(text))], toks(text))
    assert plan.new_text == "Obs: Zürich é." and check(plan, toks(plan.new_text)) == -1


def test_tokens_must_spell_the_text():
    with pytest.raises(AlignmentError):
        plan_trim(TEXT, [], [(1, b"<s>")] + toks(TEXT))


def test_check_reports_first_difference():
    plan = plan_trim(TEXT, [span(" B two.")], toks(TEXT))
    other = toks(plan.new_text)
    other[3] = (other[3][0] + 1, other[3][1])
    assert check(plan, other) == 3


def test_observation_spans_skip_the_latest_observation():
    segs = [("shared_prefix", "SYS"), ("control", "\nThought 1:"), ("thought_action", " t"),
            ("control", "\nObservation 1: "), ("obs_sf", "keep."), ("obs_other", " drop1."),
            ("control", "\nThought 2:"), ("thought_action", " t"),
            ("control", "\nObservation 2: "), ("obs_other", " latest."), ("control", "\nThought 3:")]
    text = "".join(t for _, t in segs)
    spans = observation_spans(segs)
    assert [text[a:b] for a, b in spans] == [" drop1."]
    assert [text[a:b] for a, b in observation_spans(segs, keep_last=0)] == [" drop1.", " latest."]


def test_whole_paragraph_goes_with_its_title():
    segs = [("control", "\nObservation 1: "), ("obs_meta", "[A] "), ("obs_sf", "keep."), ("obs_other", " x."),
            ("obs_meta", "\n"), ("obs_meta", "[B] "), ("obs_other", "y1."), ("obs_other", " y2."),
            ("obs_meta", "\n"), ("obs_meta", "[C] "), ("obs_other", "z."),
            ("control", "\nThought 2:"), ("control", "\nObservation 2: "), ("obs_other", " new."),
            ("control", "\nThought 3:")]
    text = "".join(t for _, t in segs)
    got = [text[a:b] for a, b in observation_spans(segs)]
    assert got == [" x.", "[B] y1. y2.\n", "[C] z."]


def test_plan_verified_drops_the_span_that_retokenizes_differently():
    text = "Obs: A one. B two. C three."
    def tok(t):
        return [(hash(p) % 10_000, p.encode()) for p in re.findall(r" ?\S+|\s", t)]
    async def tokenize(t):               # pretend deleting " B two." merges "one." and " C" into one token
        out = tok(t)
        if "one. C" in t:
            i = next(k for k, (_, p) in enumerate(out) if p == b" C")
            out[i - 1:i + 1] = [(1, b"one. C")]
        return out
    spans = [(text.index(" B two."), text.index(" C")), (text.index(" C"), len(text))]
    plan = asyncio.run(plan_verified(tokenize, text, spans[:1]))
    assert plan.ranges == [] and plan.n_retok_dropped == 1
    plan = asyncio.run(plan_verified(tokenize, text, spans[1:]))
    assert plan.new_text == "Obs: A one. B two." and plan.n_retok_dropped == 0
