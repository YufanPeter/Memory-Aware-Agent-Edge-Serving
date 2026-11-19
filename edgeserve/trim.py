"""TRIM (Phase 3): keep only part of a consumed observation (docs/phase3_spec.md).

The model reads observation n at full length exactly once, in the request that writes
Thought/Action n+1. After that only its useful sentences matter, so the agent rewrites it in
the prompt to the sentences a selector keeps. This is a text-level change: llama-server's
prefix cache keeps the common prefix and re-prefills from the first changed byte, so it needs
no runtime support beyond what the server already exposes.

Selectors work sentence by sentence on one observation's pieces, the (kind, text) list of
tools.Observation as stored in Prompt.segments:

  none         keep everything (the Phase 2 baseline)
  oracle       keep the HotpotQA supporting facts (obs_sf): the bound on free selection
  drop         keep no sentence: the agent relies on its own Thoughts
  referenced   keep sentences sharing a salient token with what the model wrote right after
               reading the observation (its Thought/Action)
  relevance:m  keep the m sentences most similar to the question (IDF-weighted overlap of
               salient tokens; IDF over the corpus paragraphs)

Framing (obs_meta: page titles, "Could not find ...", lookup counters) is always kept, and
each run of dropped sentences becomes one obs_trim piece "…", so the observation still reads
as text: "[Ed Wood] Edward Davis Wood Jr. was an American filmmaker. …".

Salient tokens: lowercase word tokens (accents folded), stopwords removed, length >= 3 or
containing a digit. `referenced` also drops the tokens of the sentence's own page title (they
appear in almost every sentence of the page, so they would make every sentence look
referenced); `relevance:m` keeps them (see relevance_scores).
"""
from __future__ import annotations

import math
import re
import unicodedata
import weakref
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from edgeserve.corpus import Corpus
    from edgeserve.prompt import Prompt

SENTENCE_KINDS = ("obs_sf", "obs_other")
OBS_KINDS = ("obs_sf", "obs_other", "obs_meta", "obs_trim")
TRIM_KIND = "obs_trim"
MARKER = "…"
MODES = ("none", "oracle", "drop", "referenced", "relevance")

# A standard English list plus the words the ReAct format itself puts into every Thought and
# Action ("I need to search ...", "Action 2: lookup[...]"): they say nothing about content.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren as at be because been before being
below between both but by can cannot could couldn did didn do does doesn doing don down during each
few for from further had hadn has hasn have haven having he her here hers herself him himself his how
i if in into is isn it its itself just let ll me more most mustn my myself no nor not now of off on
once only or other ought our ours ourselves out over own re same shan she should shouldn so some such
than that the their theirs them themselves then there these they this those through to too under until
up very was wasn we were weren what when where which while who whom why will with won would wouldn you
your yours yourself yourselves also however since may might must shall still yet
thought action observation search lookup finish answer question need find information paragraph result
results closest similar next know
""".split())

_WORD = re.compile(r"[a-z0-9]+")
_TITLE_PIECE = re.compile(r"^\[(.+)\] $", re.S)   # tools.ToolBox._add_paragraph's "[title] " piece
# The step labels the model writes ("Action 2:"): their number would match any sentence with a 2.
_LABEL = re.compile(r"\b(?:Thought|Action|Observation)\s*\d*\s*:", re.I)


def tokens(text: str) -> list[str]:
    """Lowercase word tokens with accents folded (García -> garcia), as the model often drops them."""
    text = text.lower()
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _WORD.findall(text)


def salient(text: str, exclude: frozenset[str] | set[str] = frozenset()) -> set[str]:
    return {t for t in tokens(text)
            if t not in STOPWORDS and t not in exclude and (len(t) >= 3 or any(c.isdigit() for c in t))}


def title_tokens(title: str | None) -> frozenset[str]:
    return frozenset(tokens(title)) if title else frozenset()


# --- IDF over the corpus paragraphs, computed once per Corpus -------------------------------

_IDF: weakref.WeakKeyDictionary[Corpus, dict[str, float]] = weakref.WeakKeyDictionary()


def idf_table(corpus: Corpus) -> dict[str, float]:
    """BM25-style IDF, log(1 + (N - df + 0.5) / (df + 0.5)), with df counted over paragraphs.
    The first call on a corpus takes seconds (every paragraph is tokenized), so the runner
    calls it before the run's clock starts; later calls return the cached table."""
    if (cached := _IDF.get(corpus)) is not None:
        return cached
    df: Counter[str] = Counter()
    for sents in corpus.docs.values():
        df.update(set(tokens(" ".join(sents))))
    n = len(corpus.docs)
    table = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    table[""] = math.log(1 + (n + 0.5) / 0.5)   # weight of a token the corpus never saw
    _IDF[corpus] = table
    return table


# --- selectors ----------------------------------------------------------------------------

@dataclass
class TrimContext:
    question: str = ""
    generated: str = ""            # the model's text written right after reading the observation
    page: str | None = None        # page of sentences shown without a "[title] " piece (lookup)
    idf: Mapping[str, float] | None = None


def sentence_titles(pieces: list[tuple[str, str]], page: str | None = None) -> list[str | None]:
    """The page each piece belongs to: the latest "[title] " framing piece, else `page`
    (a lookup result shows one sentence of the last searched page without its title)."""
    out: list[str | None] = []
    cur = page
    for kind, text in pieces:
        if kind == "obs_meta" and (m := _TITLE_PIECE.match(text)):
            cur = m.group(1)
        out.append(cur)
    return out


def _sentences(pieces: list[tuple[str, str]]) -> list[int]:
    return [i for i, (k, _) in enumerate(pieces) if k in SENTENCE_KINDS]


def keep_oracle(pieces: list[tuple[str, str]], ctx: TrimContext) -> set[int]:
    return {i for i in _sentences(pieces) if pieces[i][0] == "obs_sf"}


def reference_tokens(generated: str) -> set[str]:
    """Salient tokens of the model's Thought/Action, without its step labels."""
    return salient(_LABEL.sub(" ", generated))


def keep_referenced(pieces: list[tuple[str, str]], ctx: TrimContext) -> set[int]:
    ref = reference_tokens(ctx.generated)
    titles = sentence_titles(pieces, ctx.page)
    return {i for i in _sentences(pieces) if salient(pieces[i][1], title_tokens(titles[i])) & ref}


def relevance_scores(pieces: list[tuple[str, str]], ctx: TrimContext,
                     exclude_title: bool = False) -> dict[int, float]:
    """IDF-weighted overlap of each sentence's salient tokens with the question's. Without an
    IDF table every shared token weighs 1. Unlike `referenced`, page-title tokens count by
    default: the question usually names the page's entity, and on the perf split keeping them
    raised relevance:2's supporting-fact recall from 0.42 to 0.64 at the same kept size
    (scripts/trim_offline.py, trace replay of p0_baselines c32k runs)."""
    q = salient(ctx.question)
    idf = ctx.idf
    default = idf.get("", 1.0) if idf is not None else 1.0
    titles = sentence_titles(pieces, ctx.page)
    out = {}
    for i in _sentences(pieces):
        shared = salient(pieces[i][1], title_tokens(titles[i]) if exclude_title else frozenset()) & q
        out[i] = sum(idf.get(t, default) for t in shared) if idf is not None else float(len(shared))
    return out


def keep_relevance(pieces: list[tuple[str, str]], ctx: TrimContext, m: int,
                   exclude_title: bool = False) -> set[int]:
    """The m best-scoring sentences of the observation; ties go to the earlier sentence. A
    sentence that shares no salient token with the question is never kept (its rank among the
    zero scores would be arbitrary)."""
    scores = relevance_scores(pieces, ctx, exclude_title)
    ranked = sorted((i for i, s in scores.items() if s > 0), key=lambda i: (-scores[i], i))
    return set(ranked[:m])


@dataclass(frozen=True)
class TrimMode:
    """A parsed --trim value. str() gives the canonical mode string back."""
    name: str = "none"
    m: int = 0

    def __str__(self) -> str:
        return f"relevance:{self.m}" if self.name == "relevance" else self.name

    @property
    def active(self) -> bool:
        return self.name != "none"

    @property
    def needs_idf(self) -> bool:
        return self.name == "relevance"

    def kept(self, pieces: list[tuple[str, str]], ctx: TrimContext) -> set[int]:
        """Indices of the sentence pieces this selector keeps (framing is kept regardless)."""
        if self.name == "none":
            return set(_sentences(pieces))
        if self.name == "oracle":
            return keep_oracle(pieces, ctx)
        if self.name == "drop":
            return set()
        if self.name == "referenced":
            return keep_referenced(pieces, ctx)
        return keep_relevance(pieces, ctx, self.m)


def parse_mode(mode: str | TrimMode) -> TrimMode:
    """'none' | 'oracle' | 'drop' | 'referenced' | 'relevance:m' (m >= 1)."""
    if isinstance(mode, TrimMode):
        return mode
    s = str(mode).strip().lower()
    name, _, arg = s.partition(":")
    if name not in MODES:
        raise ValueError(f"unknown trim mode {mode!r}: use one of none, oracle, drop, referenced, relevance:m")
    if name == "relevance":
        if not arg.isdigit() or int(arg) < 1:
            raise ValueError(f"trim mode {mode!r}: relevance needs a sentence count, e.g. relevance:2")
        return TrimMode("relevance", int(arg))
    if arg:
        raise ValueError(f"trim mode {mode!r}: only relevance takes an argument")
    return TrimMode(name)


# --- compaction ---------------------------------------------------------------------------

def compact_pieces(pieces: list[tuple[str, str]], kept: set[int]) -> list[tuple[str, str]]:
    """Framing and kept sentences unchanged; each run of dropped sentences becomes one marker.

    Whitespace: a sentence piece carries its own leading space, except a paragraph's first
    sentence (lstripped after "[title] ") and a lookup result. The marker therefore takes a
    leading space only when the text before it does not already end in whitespace ("[T] …",
    "S0 …", "(Result 1 / 2) …"), and a trailing space only when the next kept sentence has no
    leading space of its own (a few corpus sentences start mid-quote)."""
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(pieces):
        kind, text = pieces[i]
        if kind not in SENTENCE_KINDS or i in kept:
            out.append((kind, text))
            i += 1
            continue
        j = i
        while j < len(pieces) and pieces[j][0] in SENTENCE_KINDS and j not in kept:
            j += 1
        before = out[-1][1] if out else " "   # the observation follows "Observation n: "
        lead = "" if before[-1:].isspace() else " "
        nxt = pieces[j] if j < len(pieces) else None
        tail = " " if nxt is not None and nxt[0] in SENTENCE_KINDS and not nxt[1][:1].isspace() else ""
        out.append((TRIM_KIND, lead + MARKER + tail))
        i = j
    return out


@dataclass
class TrimStats:
    chars_before: int = 0
    chars_after: int = 0
    sents_kept: int = 0
    sents_dropped: int = 0
    sf_kept: int = 0
    sf_dropped: int = 0
    pieces_before: int = 0
    pieces_after: int = 0
    changed: bool = False

    def to_dict(self) -> dict[str, int]:
        d = asdict(self)
        for k in ("pieces_before", "pieces_after", "changed"):
            d.pop(k)
        return d


def trim_pieces(pieces: list[tuple[str, str]], mode: TrimMode,
                ctx: TrimContext) -> tuple[list[tuple[str, str]], TrimStats]:
    """The compacted pieces of one observation and what was kept."""
    sents = _sentences(pieces)
    kept = mode.kept(pieces, ctx) & set(sents)
    new = compact_pieces(pieces, kept)
    sf = [i for i in sents if pieces[i][0] == "obs_sf"]
    st = TrimStats(chars_before=sum(len(t) for _, t in pieces), chars_after=sum(len(t) for _, t in new),
                   sents_kept=len(kept), sents_dropped=len(sents) - len(kept),
                   sf_kept=sum(1 for i in sf if i in kept), sf_dropped=sum(1 for i in sf if i not in kept),
                   pieces_before=len(pieces), pieces_after=len(new), changed=new != pieces)
    return new, st


def compact_observation(prompt: Prompt, start: int, end: int, mode: TrimMode, ctx: TrimContext) -> TrimStats:
    """Rewrite prompt.segments[start:end] (one observation) in place. The slice is replaced,
    not the list, because TaskTrace.segments aliases it. The caller shifts the spans of later
    observations by pieces_after - pieces_before."""
    pieces = prompt.segments[start:end]
    if bad := [k for k, _ in pieces if k not in OBS_KINDS]:
        raise ValueError(f"segments {start}:{end} are not one observation (found {bad})")
    new, st = trim_pieces(pieces, mode, ctx)
    if st.changed:
        prompt.segments[start:end] = new
    return st
