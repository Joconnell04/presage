"""presage/digest.py: the post-call digest (experiment E44).

The cards are the live half of presage; this is the half the consultant reads once the call is
over, in the two minutes before the next one. It answers six questions about the call that just
happened, and every answer is anchored in a line of the transcript:

  questions they asked        with who asked and when, where speaker labels exist
  what was answered on screen and whether the consultant used the card
  what is still open          asked and never answered, or answered with no figure
  what was promised           "I'll send you the model", "we'll come back on pricing"
  objections                  and how each one landed
  figures said aloud          listed only; E9b is what verifies them
  next steps                  who owns each

**Three inputs, one model call.** The transcript (`Line`), the intervention log's card rows
(`Card`: what was shown, and `used` / `ignored` / `wrong` / `good` where the consultant marked
it), and a deterministic pre-pass that marks candidate questions, promises, objections and
spoken figures. The model confirms and phrases; it never has to search, which is what keeps this
to one subscription call with a JSON schema instead of a map-reduce over an hour of talk.

**Grounding by line id, not by timestamp.** Every line the model sees is labelled `L12`, and
every item it returns names the line it came from. The timestamps, the speaker labels and the
"who said it" in the output are then resolved here from that id, so a model that hallucinates a
time cannot put one in the digest: an item whose `line` is not a real line is dropped, and an
item on the wrong side of the call (a promise attributed to the client) is dropped too.

**Windowing.** An hour of talk is 400-700 final lines. The prompt carries the anchors (every
candidate question, promise and objection) plus the 30 s around each, merged, and is trimmed to
`MAX_PROMPT_TOKENS` by dropping context lines furthest from their anchor: anchors themselves are
never dropped. `meta["prompt_tokens"]` reports the estimate that was actually sent.

Use:

    from presage import digest
    digest.recorder().observe(frame)                    # one line in server.broadcast
    out = await digest.build(llm)                       # {"digest", "markdown", "meta"}
    path = digest.save(out["markdown"], out["meta"]["started_at"])

Offline, over a simulator run's frames (this is E44's test bed):

    uv run python -m presage.digest docs/experiments/sim/e17/jev-r1/big-01-capital
    uv run python -m presage.digest <run_dir> --provider gemini --dry-run

Nothing here is imported by `brain.py` and nothing here touches the answer path.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("presage.digest")

DIGEST_DIR = Path(os.environ.get("PRESAGE_DIGEST_DIR")
                  or (Path.home() / ".cache" / "presage" / "digests"))

WINDOW_S = 30.0              # seconds of context kept either side of an anchor line
WINDOW_LINES = 3             # ...and at most this many lines either side, so a burst cannot flood it
MAX_PROMPT_TOKENS = 12000    # the budget the brief sets for an hour-long call
CHARS_PER_TOKEN = 4          # the estimator; no tokenizer is installed and none is worth adding
MAX_LINES = 4000             # ceiling on what the recorder keeps: ~6 h of talk
MAX_CARDS = 400
LINE_CHARS = 240             # a single line is clipped to this in the prompt
MAX_FIGURES = 40             # spoken-figure candidates handed to the model
MAX_ITEMS = 60               # per section, after validation

# What the gate calls a line that is worth a card. A line the gate already marked is a candidate
# whatever the regexes below think.
GATE_ASKS = ("question", "what-if", "whatif", "request", "objection")


# ----------------------------------------------------------------------------- data

@dataclass
class Line:
    """One final transcript line. `kind` is the gate's verdict when this line has one."""
    id: str
    source: str                  # "mic" (Me, the consultant) | "system" (Them, the client)
    text: str
    ts: float                    # wall clock
    t0: float | None = None      # seconds into the audio stream, when the STT provider gives it
    speaker: str | None = None   # "A" / "B" from diarization, far side only
    utt: int | None = None
    kind: str | None = None      # gate verdict for this line: question / what-if / ... / none
    label: str = ""              # what the model sees: L1, L2, ... (see `label_lines`)

    @property
    def me(self) -> bool:
        return self.source == "mic"

    @property
    def who(self) -> str:
        if self.me:
            return "Me"
        return f"Them {self.speaker}" if self.speaker else "Them"

    def at(self, started_at: float | None) -> float:
        """Seconds into the call. `t0` is the stream position and is preferred; the wall clock
        against `started_at` is the fallback, and 0.0 when neither is knowable."""
        if self.t0 is not None and 0 <= self.t0 < 86400:
            return float(self.t0)
        if started_at and self.ts >= started_at:
            return round(self.ts - started_at, 2)
        return 0.0


@dataclass
class Card:
    """One card presage put on screen, folded to its final state (E12's `card` rows)."""
    id: str
    ts: float
    question: str = ""
    answer: str = ""
    say: str = ""
    figure: str | None = None
    source: str = ""             # prefetch | live | manual
    answered_from: str = ""      # files | web | model | derived | none
    verdict: str | None = None   # used | ignored | wrong | good, from POST /api/feedback
    implied: bool = False        # replaced with no verdict: E12's weak "ignored"
    utt: int | None = None
    citations: list = field(default_factory=list)

    @property
    def mark(self) -> str:
        return self.verdict or ("ignored?" if self.implied else "unmarked")


def mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"


def label_lines(lines: list[Line]) -> dict[str, Line]:
    """Number the lines L1, L2, ... in call order, and hand back the map.

    The transcript's own ids are the page's (`ts3` for Them, `tm3` for Me) and differ by one
    character across the two sides, which is exactly the confusion a model should not be able to
    make when the rule is "promises come from Me only". `L12` is unambiguous, short, and resolves
    back to the real line here rather than in the model.
    """
    out = {}
    for i, l in enumerate(lines, 1):
        l.label = f"L{i}"
        out[l.label] = l
    return out


# ----------------------------------------------------------------------------- the recorder

class Recorder:
    """The whole call, kept for the digest.

    `brain.transcript` holds ten minutes and `Hub.transcript` holds eighty lines, because both
    exist to answer the next question rather than to describe the call. The digest needs all of
    it, so this keeps its own copy: final lines only, id-keyed so a retracted final that gets
    continued updates the line rather than duplicating it.
    """

    def __init__(self) -> None:
        self.lines: dict[str, Line] = {}
        self.cards: dict[str, Card] = {}
        self.started_at: float | None = None
        self.title: str = ""

    # -- ingest ---------------------------------------------------------------

    def observe(self, msg: dict) -> None:
        """One broadcast frame. Never raises: the digest must not cost a card or a line."""
        try:
            kind = msg.get("type")
            if kind == "transcript":
                if msg.get("final"):
                    self.line(msg)
            elif kind == "answer":
                self.card(msg)
            elif kind == "cleared":
                self.clear()
        except Exception as e:  # noqa: BLE001
            log.debug("digest recorder: %s", e)

    def line(self, msg: dict) -> None:
        text = str(msg.get("text") or "").strip()
        if not text:
            return
        lid = str(msg.get("id") or "")
        if not lid:
            lid = f"l{len(self.lines)}"
        t0 = msg.get("t0")
        self.lines[lid] = Line(
            id=lid, source=str(msg.get("source") or "system"), text=text,
            ts=float(msg.get("ts") or time.time()),
            t0=float(t0) if isinstance(t0, (int, float)) else None,
            speaker=(msg.get("speaker") or None), utt=msg.get("utt"))
        if self.started_at is None:
            self.started_at = self.lines[lid].ts - (self.lines[lid].t0 or 0.0)
        if len(self.lines) > MAX_LINES:
            for k in list(self.lines)[: len(self.lines) - MAX_LINES]:
                self.lines.pop(k, None)

    def card(self, msg: dict) -> None:
        cid = str(msg.get("id") or "")
        if not cid:
            return
        was = self.cards.get(cid)
        c = Card(id=cid, ts=float(msg.get("ts") or time.time()),
                 question=str(msg.get("question") or ""), answer=str(msg.get("answer") or ""),
                 say=str(msg.get("say") or ""), figure=msg.get("figure"),
                 source=str(msg.get("source") or ""),
                 answered_from=str(msg.get("answered_from") or ""),
                 utt=msg.get("utt"), citations=list(msg.get("citations") or []))
        if was is not None:                       # a re-send is an update, not a second card
            c.verdict, c.implied, c.ts = was.verdict, was.implied, was.ts
        self.cards[cid] = c
        if len(self.cards) > MAX_CARDS:
            for k in list(self.cards)[: len(self.cards) - MAX_CARDS]:
                self.cards.pop(k, None)

    def mark(self, card_id: str, verdict: str) -> None:
        c = self.cards.get(str(card_id))
        if c is not None:
            c.verdict = verdict

    def clear(self) -> None:
        self.lines.clear()
        self.cards.clear()
        self.started_at = None

    # -- read -----------------------------------------------------------------

    def ordered(self) -> list[Line]:
        return sorted(self.lines.values(), key=lambda l: (l.at(self.started_at), l.ts))

    def shown(self) -> list[Card]:
        return sorted(self.cards.values(), key=lambda c: c.ts)

    def stats(self) -> dict:
        lines = self.ordered()
        return {"lines": len(lines), "cards": len(self.cards),
                "started_at": self.started_at,
                "call_s": round(lines[-1].at(self.started_at), 1) if lines else 0.0}

    # -- the intervention log -------------------------------------------------

    def fold_log(self, rows: Iterable[dict]) -> int:
        """Take verdicts (and the weak "replaced with nothing said") off E12's log rows.

        The log is the only place a `used` / `wrong` mark exists, and it is written by a
        different module for a different reason, so this reads it rather than duplicating it.
        Rows for cards this recorder never saw are ignored: a digest describes one call.
        """
        n = 0
        for r in rows or []:
            cid = str(r.get("id") or "")
            c = self.cards.get(cid)
            if c is None:
                continue
            if r.get("event") == "feedback" and r.get("verdict"):
                c.verdict, n = str(r["verdict"]), n + 1
            elif r.get("event") == "replaced" and not c.verdict:
                c.implied, n = True, n + 1
        return n

    def gate_kinds(self, rows: Iterable[dict]) -> int:
        """Attach the gate's verdict to the line it judged, from the log's `card` rows.

        Only lines that produced a card carry one, which is why the pre-pass has its own
        heuristics; where a verdict exists it is better evidence than any regex.
        """
        by_utt = {(l.source, l.utt): l for l in self.lines.values() if l.utt is not None}
        n = 0
        for r in rows or []:
            if r.get("event") != "card":
                continue
            kind = ((r.get("gate") or {}).get("verdict") or "").strip().lower()
            if not kind:
                continue
            line = by_utt.get(("system", r.get("utt")))
            if line is not None and not line.kind:
                line.kind, n = kind, n + 1
        return n


_recorder: Recorder | None = None


def recorder() -> Recorder:
    global _recorder
    if _recorder is None:
        _recorder = Recorder()
    return _recorder


# ----------------------------------------------------------------------------- the pre-pass

_Q_MARK = re.compile(r"\?\s*$")
_Q_START = re.compile(
    r"^(what|what's|whats|how|why|when|where|which|who|whose|can|could|do|does|did|is|are|was|"
    r"were|will|would|should|have|has|had|am|any|anything|remind|walk|talk|tell|give|show|"
    r"help|explain|run|break)\b")
_Q_PHRASES = (
    "walk me through", "talk me through", "remind me", "what about", "how much", "how many",
    "do we have", "do you have", "can you", "could you", "what's the", "whats the", "what is",
    "tell me", "give me", "any idea", "where are we", "how does that compare", "is that in",
    "curious about", "help me understand", "pull up", "can we see", "i'd love to see",
    "what does", "how did", "break that down", "say more", "go back to", "what if",
)

# First-person future: the consultant committing to do something after the call. "we can" and
# "I can" are in on purpose (they carry most real commitments in speech) and are the reason the
# model confirms rather than the reason the list is trusted: this is a recall-first candidate
# generator, and precision is measured on what comes back, not on what goes in.
_PROMISE_SUBJECT = re.compile(
    r"\b(i'?ll|i will|we'?ll|we will|i'?m going to|im going to|we'?re going to|were going to|"
    r"i can|we can|i could|we could|let me|let us|let'?s|i'?d be happy to|i want to|we want to|"
    r"i'?ll have|we'?ll have|i shall|we shall|happy to|i'?d have to|we'?d have to|"
    r"i need to|we need to|i have to|we have to)\b")
_PROMISE_VERB = (
    "send", "share", "get back", "come back", "follow up", "circle back", "write up", "write it up",
    "pull", "put together", "run the", "rerun", "re-run", "add", "update", "check", "confirm",
    "look into", "dig into", "take that away", "set up", "schedule", "walk you through",
    "show you", "give you", "email", "forward", "draft", "model", "sort out", "come to you",
    "revert", "let you know", "find out", "have a look", "review", "regroup", "reconvene",
    "catch up", "circle", "dig it up", "get you", "bring", "loop",
)
_PROMISE_BARE = ("i'll send", "i'll share", "get back to you", "come back to you", "follow up",
                 "circle back", "let you know", "take that away", "leave that with me",
                 "i'll find out", "we'll revert", "i'll email")

_OBJECTION_MARKERS = (
    "seems high", "seems low", "sounds high", "sounds low", "too high", "too low", "that's high",
    "thats high", "that's low", "i'm not sure", "im not sure", "not sure i", "not sure that",
    "i don't think", "i dont think", "doesn't feel", "doesnt feel", "hard to believe",
    "doesn't match", "doesnt match", "that's not what", "thats not what", "concerned",
    "worried", "skeptical", "sceptical", "aggressive", "optimistic", "conservative", "disagree",
    "pushback", "push back", "risky", "i'd challenge", "id challenge", "that can't be",
    "that cant be", "don't buy", "dont buy", "struggle with", "my board", "won't fly",
    "wont fly", "that feels", "surely", "really?", "come on",
)

# A number said out loud: currency, a percent, a multiple, or a scale word. Bare integers under
# four digits are left alone on purpose ("two retail partners", "Q3", "twenty people" are not figures).
_FIGURE = re.compile(
    r"(?<![\w.])(?:"
    r"\$\s?\d[\d,]*(?:\.\d+)?\s*(?:k|m|bn|b|mm|million|billion|thousand)?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:%|percent|per cent|bps|basis points)"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:x|times)\b"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand|k\b|m\b|bn\b|b\b)"
    r"|\d{4,}(?:\.\d+)?"
    r")", re.I)

# The transcriber spells small numbers out as often as it writes digits ("two million", "twelve
# percent", "one point two five"), so a digits-only extractor finds almost nothing in real
# speech. A unit or a scale word is still required: "two retail partners" is not a figure.
_UNITS = r"percent|per cent|%|million|billion|thousand|basis points|bps|times|x\b|dollars|cents"
_WORDS = (r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
          r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
          r"sixty|seventy|eighty|ninety|hundred|a half|half")
_FIGURE_HEDGE = re.compile(
    r"\b(?:about|around|roughly|closer to|close to|north of|south of|under|over|nearly|almost|"
    r"approximately|just over|just under)\s+\$?\d[\d,]*(?:\.\d+)?"
    r"(?![\d.]|\s*(?:%|percent|per cent|bps|x\b|k\b|m\b|b\b|bn\b|million|billion|thousand))", re.I)
_FIGURE_WORDS = re.compile(
    r"\b((?:%s)(?:[\s-]+(?:point|and|a|%s))*)\s+(%s)" % (_WORDS, _WORDS, _UNITS), re.I)


def is_question(line: Line) -> bool:
    """A client line that asks for something: the gate's verdict OR the shape of the line.

    The gate only ever adds. It missed 1 in 8 of the simulator's client questions (E17), so
    letting a `none` verdict veto a line that plainly ends in a question mark would import the
    gate's misses into the digest, which is the one place they can still be caught.

    `objection` is deliberately not a question here even though the gate answers it with a card:
    it has its own section. An objection phrased as a question still lands in both, because the
    regexes run on it.
    """
    if line.me:
        return False
    t = line.text.strip().lower()
    if not t:
        return False
    kind = (line.kind or "").lower()
    if kind and kind != "objection" and kind in GATE_ASKS:
        return True
    return bool(_Q_MARK.search(t) or _Q_START.match(t) or any(p in t for p in _Q_PHRASES))


def is_promise(line: Line) -> bool:
    """A consultant line that commits to doing something."""
    if not line.me:
        return False
    t = " " + line.text.strip().lower()
    if any(p in t for p in _PROMISE_BARE):
        return True
    if not _PROMISE_SUBJECT.search(t):
        return False
    return any(v in t for v in _PROMISE_VERB)


def is_objection(line: Line) -> bool:
    if line.me:
        return False
    if (line.kind or "").lower() == "objection":
        return True
    t = line.text.strip().lower()
    return any(m in t for m in _OBJECTION_MARKERS)


def figures_in(text: str) -> list[str]:
    """Every number said out loud in one line, digits or words, as it was said."""
    found: list[str] = []
    for rx in (_FIGURE, _FIGURE_HEDGE, _FIGURE_WORDS):
        for m in rx.finditer(text or ""):
            f = " ".join(m.group(0).split()).strip(" .,")
            if f:
                found.append(f)
    out, keys = [], []
    for f in sorted(found, key=len, reverse=True):      # "about 2.5x" beats "about 2"
        k = f.lower().replace(" ", "")
        if any(k in seen for seen in keys):
            continue
        keys.append(k)
        out.append(f)
    return sorted(out, key=lambda f: (text or "").find(f))


@dataclass
class Prepass:
    questions: list[str] = field(default_factory=list)      # line ids
    promises: list[str] = field(default_factory=list)
    objections: list[str] = field(default_factory=list)
    figures: list[tuple[str, str]] = field(default_factory=list)   # (line id, figure)

    @property
    def anchors(self) -> set[str]:
        return set(self.questions) | set(self.promises) | set(self.objections)

    def counts(self) -> dict:
        return {"questions": len(self.questions), "promises": len(self.promises),
                "objections": len(self.objections), "figures": len(self.figures)}


def prepass(lines: list[Line], carded: set[str] | None = None) -> Prepass:
    """Candidate questions, promises, objections and spoken figures, by line id.

    Deterministic and cheap (a few hundred regex matches on an hour of talk). `carded` is the
    set of line ids that put a card on screen: presage already decided those were worth
    answering, so they are questions whatever the regexes say.
    """
    p = Prepass()
    carded = carded or set()
    for l in lines:
        lid = l.label or l.id
        if is_question(l) or (lid in carded and not l.me):
            p.questions.append(lid)
        if is_promise(l):
            p.promises.append(lid)
        if is_objection(l):
            p.objections.append(lid)
        for f in figures_in(l.text)[:3]:
            p.figures.append((lid, f))
    return p


# ----------------------------------------------------------------------------- windowing

def est_tokens(text: str) -> int:
    return (len(text or "") + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def window(lines: list[Line], anchors: set[str], started_at: float | None,
           ctx_lines: int = WINDOW_LINES, ctx_s: float = WINDOW_S) -> list[Line]:
    """The anchor lines plus the talk around each, merged. Anchors are never dropped here.

    `ctx_lines` and `ctx_s` are the two knobs the token budget turns: context is the only thing
    in the prompt that can be spared without losing an item from the digest.
    """
    idx = {(l.label or l.id): i for i, l in enumerate(lines)}
    keep: set[int] = set()
    for aid in anchors:
        i = idx.get(aid)
        if i is None:
            continue
        keep.add(i)
        at = lines[i].at(started_at)
        for j in range(max(0, i - ctx_lines), min(len(lines), i + ctx_lines + 1)):
            if j != i and abs(lines[j].at(started_at) - at) <= ctx_s:
                keep.add(j)
    return [lines[i] for i in sorted(keep)]


def rank_anchors(lines: list[Line], pre: "Prepass", carded: set[str]) -> list[str]:
    """Anchors worst first, for the one rung of the ladder that has to drop an item.

    An hour of a real call holds 50-80 of these and never reaches this rung; a two-hour workshop
    where every second line is a question does. When something must go, it is a plain question
    with no card, from early in the call, because a promise, an objection and a question presage
    already answered are each harder to reconstruct from memory than a question that was asked
    and left alone.
    """
    rank = {}
    for lid in pre.questions:
        rank[lid] = 3 if lid in carded else 4
    for lid in pre.objections:
        rank[lid] = min(rank.get(lid, 9), 1)
    for lid in pre.promises:
        rank[lid] = min(rank.get(lid, 9), 2)
    order = {(l.label or l.id): i for i, l in enumerate(lines)}
    return sorted(rank, key=lambda lid: (-rank[lid], order.get(lid, 0)))


def render_lines(lines: list[Line], all_lines: list[Line], started_at: float | None,
                 chars: int = LINE_CHARS) -> str:
    """The transcript block: `[L12 03:41 Them A] text`, with `…` where a gap was cut out."""
    out: list[str] = []
    prev = None
    index = {(l.label or l.id): i for i, l in enumerate(all_lines)}
    for l in lines:
        i = index.get(l.label or l.id, -1)
        if prev is not None and i - prev > 1:
            out.append("  …")
        prev = i
        text = l.text if len(l.text) <= chars else l.text[: chars - 1] + "…"
        out.append(f"[{l.label or l.id} {mmss(l.at(started_at))} {l.who}] {text}")
    return "\n".join(out)


# ----------------------------------------------------------------------------- the one call

SYSTEM = """You write the digest a consultant reads in the two minutes after a client call.

You are given the call's transcript in windows, the cards a live copilot put on their screen
during the call, and a list of candidate lines a deterministic pass already found. Your job is
to confirm and phrase, not to search: nearly every item you need is in the candidate lists.

Rules, all of them hard:
- Every item names the line id it comes from, exactly as written in the transcript (e.g. "L42").
  Never invent a line id and never name a line you were not shown. An item you cannot anchor to
  a line does not go in the digest.
- Questions come from Them (the client) only. Promises come from Me (the consultant) only.
- Use their own words, shortened. Never add a number, a name or a commitment that is not in the
  line you are citing.
- A question that the transcript never answers, or answers with no number, is not a failure to
  report around: it is the most useful part of the digest. Mark it honestly.
- Skip small talk, agreement, and lines that only look like questions ("that's the number,
  right?"). Skip a "promise" that is only a figure of speech ("let me think").
- A promise is a commitment the client could hold them to after the call ("I'll send the model",
  "we'll come back on pricing"). Something the consultant did inside the call ("let me pull that
  up") is not a promise, unless the transcript shows it never happened.
- Card ids: put the card id on a question when a card on screen answered that question, and on
  a figure when the same number was also on a card. Otherwise leave it empty; a card id you are
  unsure of is better left out than guessed.
- Figures are numbers a person SAID OUT LOUD. The candidate figure list is exhaustive: take them
  from it, say what each one refers to, and add nothing. A number that was only ever on a card
  is not a figure said aloud. An empty candidate list means an empty section.
- Terse. No preamble, no advice, no interpretation the transcript does not carry."""

DIGEST_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string",
                    "description": "One sentence, at most 24 words: what the call was about and where it ended."},
        "questions": {
            "type": "array",
            "description": "Every question the client asked, in the order they were asked.",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "string", "description": "the transcript line id, e.g. 'L42'"},
                    "question": {"type": "string", "description": "their question, at most 16 words"},
                    "status": {"type": "string",
                               "enum": ["answered_on_screen", "answered_by_me",
                                        "answered_without_figure", "unanswered"],
                               "description": "answered_on_screen: a card answered it with a figure; "
                                              "answered_by_me: the consultant answered it out loud; "
                                              "answered_without_figure: answered but no number was given; "
                                              "unanswered: nobody answered it."},
                    "answer": {"type": "string",
                               "description": "what the answer was, at most 14 words, or \"\" if none"},
                    "card": {"type": "string", "description": "the card id that answered it, or \"\""},
                },
                "required": ["line", "question", "status", "answer", "card"],
                "additionalProperties": False,
            },
        },
        "promises": {
            "type": "array",
            "description": "What the consultant committed to do after the call.",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "string"},
                    "promise": {"type": "string", "description": "at most 12 words, starting with a verb"},
                    "due": {"type": "string", "description": "when they said they would do it, or \"\""},
                },
                "required": ["line", "promise", "due"],
                "additionalProperties": False,
            },
        },
        "objections": {
            "type": "array",
            "description": "Pushback from the client on a number, a plan or a price.",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "string"},
                    "objection": {"type": "string", "description": "at most 14 words"},
                    "response": {"type": "string", "description": "how the consultant answered it, at most 14 words, or \"\""},
                    "landed": {"type": "string", "enum": ["resolved", "parked", "unresolved"]},
                },
                "required": ["line", "objection", "response", "landed"],
                "additionalProperties": False,
            },
        },
        "figures": {
            "type": "array",
            "description": "Numbers said out loud in the call. List them; do not check them.",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "string"},
                    "figure": {"type": "string", "description": "the number as said, e.g. '$16.4M'"},
                    "about": {"type": "string", "description": "what it refers to, at most 8 words"},
                    "card": {"type": "string", "description": "the card id it came off, or \"\""},
                },
                "required": ["line", "figure", "about", "card"],
                "additionalProperties": False,
            },
        },
        "next_steps": {
            "type": "array",
            "description": "What happens after this call, from what was actually said.",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "string"},
                    "step": {"type": "string", "description": "at most 12 words, starting with a verb"},
                    "owner": {"type": "string", "enum": ["me", "them", "both"]},
                },
                "required": ["line", "step", "owner"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "questions", "promises", "objections", "figures", "next_steps"],
    "additionalProperties": False,
}

SECTIONS = ("questions", "promises", "objections", "figures", "next_steps")


# How the prompt shrinks when a call is too long to send whole, worst case last. Each rung is
# (context lines either side, context seconds, characters per line, cards written out in full).
# Rung 0 is what a normal call uses; the E17 runs and a realistic hour never leave it.
LADDER = [(WINDOW_LINES, WINDOW_S, LINE_CHARS, True),
          (1, 15.0, LINE_CHARS, True),
          (0, 0.0, LINE_CHARS, True),
          (0, 0.0, 140, False),
          (0, 0.0, 100, False)]


def card_block(cards: list[Card], started_at: float | None, full: bool = True) -> str:
    if not cards:
        return "\nCards the copilot showed: none."
    head = ("\nCards the copilot showed (id, time, the question it answered, its figure, and what "
            "the consultant did with it):" if full else
            "\nCards the copilot showed (id, time, figure, what the consultant did with it):")
    out = [head]
    for c in cards:
        at = mmss(c.ts - started_at if started_at else 0)
        fig = f" → {c.figure}" if c.figure else ""
        if full:
            out.append(f"  {c.id} [{at}] {(c.question or c.say or c.answer)[:90]}{fig} · {c.mark}")
        else:
            out.append(f"  {c.id} [{at}]{fig} · {c.mark}")
    return "\n".join(out)


def build_prompt(lines: list[Line], cards: list[Card], pre: Prepass, started_at: float | None,
                 max_tokens: int = MAX_PROMPT_TOKENS, carded: set[str] | None = None
                 ) -> tuple[str, dict]:
    """The user half of the one call, and what it cost. The system half is fixed (it caches).

    An hour of a normal call is about 3-6k tokens here. The ladder exists for the calls that are
    not normal: it gives up context first, then the cards' question text, and only at the last
    rung gives up an anchor, which is the only thing that loses the digest an item. `meta` says
    which rung was used and what, if anything, was dropped.
    """
    call_s = lines[-1].at(started_at) if lines else 0.0
    anchors, dropped_anchors = set(pre.anchors), 0

    def assemble(rung: int) -> tuple[str, int]:
        ctx_lines, ctx_s, chars, full = LADDER[rung]
        head = [f"Call: {mmss(call_s)} long, {len(lines)} lines. Me = the consultant, "
                f"Them = the client. Times are minutes into the call.",
                card_block(cards, started_at, full),
                "\nCandidate lines the deterministic pass found (confirm or discard each one; "
                "add anything it missed):",
                f"  questions:  {', '.join(q for q in pre.questions if q in anchors) or 'none'}",
                f"  promises:   {', '.join(q for q in pre.promises if q in anchors) or 'none'}",
                f"  objections: {', '.join(q for q in pre.objections if q in anchors) or 'none'}",
                "  figures:    " + ("; ".join(f"{lid} {f}" for lid, f in pre.figures[:MAX_FIGURES])
                                    or "none")]
        body = render_lines(window(lines, anchors, started_at, ctx_lines, ctx_s), lines,
                            started_at, chars)
        text = "\n".join(head) + "\n\nTranscript (oldest first; … marks talk that was cut out):\n" + body
        return text, est_tokens(SYSTEM) + est_tokens(text) + 200      # 200: the schema's own words

    rung = 0
    user, tokens = assemble(rung)
    while tokens > max_tokens and rung < len(LADDER) - 1:
        rung += 1
        user, tokens = assemble(rung)
    if tokens > max_tokens:
        worst = rank_anchors(lines, pre, carded or set())
        for lid in worst:
            if tokens <= max_tokens or len(anchors) <= 1:
                break
            anchors.discard(lid)
            dropped_anchors += 1
            if dropped_anchors % 10 == 0 or len(anchors) <= 1:
                user, tokens = assemble(rung)
        user, tokens = assemble(rung)

    meta = {"prompt_tokens": tokens, "prompt_chars": len(user), "rung": rung,
            "dropped_anchors": dropped_anchors, "truncated": dropped_anchors > 0,
            "lines_sent": user.count("\n[") , "lines_total": len(lines), "cards": len(cards),
            **pre.counts()}
    return user, meta


# ----------------------------------------------------------------------------- validation

def _clip(text: Any, n: int = 200) -> str:
    return " ".join(str(text or "").split())[:n]


def _said(figure: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "", str(figure or "").lower())


def validate(data: dict, lines: list[Line], cards: list[Card], started_at: float | None,
             spoken: dict[str, list[str]] | None = None) -> dict:
    """Resolve every item against the transcript, and drop what will not resolve.

    This is what makes a line id better than a timestamp: the digest's times, speakers and
    quotes are read out of the transcript here, so the model's only job was to point.
    """
    by_id = {(l.label or l.id): l for l in lines}
    card_ids = {c.id for c in cards}
    # Which card answered which line is not a judgement call: a card carries the far-side
    # utterance counter it was drafted for, so the link is exact wherever the brain set one
    # (78 of 84 cards in the E17 runs). The model is asked for it anyway, and is overridden
    # here, because "every card is accounted for" should not depend on a model remembering.
    by_line: dict[str, list[Card]] = {}
    utts = {l.utt: l for l in lines if not l.me and l.utt is not None}
    for c in cards:
        line = utts.get(c.utt) if c.utt is not None else None
        if line is not None:
            by_line.setdefault(line.label or line.id, []).append(c)
    dropped = {"no_line": 0, "wrong_side": 0, "duplicate": 0, "bad_card": 0, "empty": 0,
               "not_spoken": 0}
    out: dict[str, Any] = {"summary": _clip(data.get("summary"), 240)}

    def resolve(item: dict, text_key: str, side: str | None) -> dict | None:
        lid = str(item.get("line") or "").strip().strip(".,")
        line = by_id.get(lid)
        if line is None:
            dropped["no_line"] += 1
            return None
        if side == "me" and not line.me:
            dropped["wrong_side"] += 1
            return None
        if side == "them" and line.me:
            dropped["wrong_side"] += 1
            return None
        if not _clip(item.get(text_key)):
            dropped["empty"] += 1
            return None
        at = line.at(started_at)
        row = {"line": lid, "line_id": line.id, "at": mmss(at), "at_s": round(at, 2),
               "ts": round(line.ts, 3), "who": line.who, "speaker": line.speaker,
               "quote": _clip(line.text, 300)}
        card = str(item.get("card") or "").strip()
        if card and card not in card_ids:
            dropped["bad_card"] += 1
            card = ""
        for k, v in item.items():
            if k in ("line", "card"):
                continue
            row[k] = _clip(v, 200)
        if "card" in item or card:
            row["card"] = card
        return row

    plan = {"questions": ("question", "them"), "promises": ("promise", "me"),
            "objections": ("objection", "them"), "figures": ("figure", None),
            "next_steps": ("step", None)}
    for section, (key, side) in plan.items():
        rows, seen = [], set()
        for item in (data.get(section) or [])[: MAX_ITEMS * 2]:
            if not isinstance(item, dict):
                continue
            row = resolve(item, key, side)
            if row is None:
                continue
            if section == "figures" and spoken is not None:
                # E9b verifies figures against the files; it can only do that for numbers a
                # person actually said, so a "figure" that was only ever on a card is dropped
                # here rather than passed on as if it had been spoken.
                said = [_said(f) for f in spoken.get(row["line"], [])]
                mine = _said(row.get("figure"))
                if not any(mine and (mine in f or f in mine) for f in said):
                    dropped["not_spoken"] += 1
                    continue
            sig = (row["line"], row[key].lower()[:40])
            if sig in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(sig)
            rows.append(row)
        out[section] = sorted(rows, key=lambda r: r["at_s"])[:MAX_ITEMS]

    linked = fixed = 0
    for q in out["questions"]:
        q.setdefault("status", "unanswered")
        if q["status"] not in ("answered_on_screen", "answered_by_me", "answered_without_figure",
                               "unanswered"):
            q["status"] = "unanswered"
        mine = by_line.get(q["line"]) or []
        if mine and not q.get("card"):
            best = next((c for c in mine if c.figure), mine[0])
            q["card"], linked = best.id, linked + 1
        with_figure = next((c for c in mine if c.figure), None)
        if with_figure is not None and q["status"] in ("unanswered", "answered_without_figure"):
            # a figure was on the screen while they were asking, whatever the transcript shows
            q["status"], fixed = "answered_on_screen", fixed + 1
            q["answer"] = q.get("answer") or with_figure.figure or ""
    referenced = {r.get("card") for r in out["questions"] + out["figures"] if r.get("card")}
    out["cards"] = [{"id": c.id, "at": mmss(c.ts - started_at if started_at else 0),
                     "question": _clip(c.question, 160), "figure": c.figure,
                     "say": _clip(c.say, 160), "answer": _clip(c.answer, 200),
                     "source": c.source, "answered_from": c.answered_from, "mark": c.mark,
                     "in_digest": c.id in referenced} for c in cards]
    out["open"] = [q for q in out["questions"]
                   if q["status"] in ("unanswered", "answered_without_figure")]
    out["dropped"] = dropped
    out["counts"] = {s: len(out[s]) for s in SECTIONS}
    out["counts"]["open"] = len(out["open"])
    out["counts"]["cards_shown"] = len(cards)
    out["counts"]["cards_in_digest"] = len(referenced)
    out["counts"]["cards_linked_here"] = linked      # card ids the utt map supplied, not the model
    out["counts"]["status_fixed"] = fixed
    return out


# ----------------------------------------------------------------------------- markdown

def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


STATUS_WORD = {"answered_on_screen": "card", "answered_by_me": "said",
               "answered_without_figure": "no figure", "unanswered": "open"}


def render(digest: dict, title: str = "", started_at: float | None = None) -> str:
    """The digest as the consultant reads it. Bottom line first: what is still open."""
    when = datetime.fromtimestamp(started_at).strftime("%Y-%m-%d %H:%M") if started_at else ""
    c = digest.get("counts") or {}
    out = [f"# Post-call digest{(': ' + title) if title else ''}",
           f"_{when}{' · ' if when else ''}{plural(c.get('questions', 0), 'question')} asked · "
           f"{c.get('open', 0)} still open · {plural(c.get('promises', 0), 'promise')} · "
           f"{plural(c.get('cards_shown', 0), 'card')}_", ""]
    if digest.get("summary"):
        out += [digest["summary"], ""]

    if digest.get("open"):
        out += ["## Still open", "",
                "| at | who | question | why |", "|---|---|---|---|"]
        for q in digest["open"]:
            why = "nobody answered" if q["status"] == "unanswered" else "answered, no figure"
            out.append(f"| {q['at']} | {q['who']} | {q['question']} | {why} |")
        out.append("")

    if digest.get("promises"):
        out += ["## I promised", "", "| at | promise | when |", "|---|---|---|"]
        for p in digest["promises"]:
            out.append(f"| {p['at']} | {p['promise']} | {p.get('due') or '-'} |")
        out.append("")

    if digest.get("questions"):
        out += ["## Questions they asked", "",
                "| at | who | question | answer | how | card |", "|---|---|---|---|---|---|"]
        for q in digest["questions"]:
            out.append(f"| {q['at']} | {q['who']} | {q['question']} | {q.get('answer') or '-'} | "
                       f"{STATUS_WORD.get(q['status'], q['status'])} | {q.get('card') or '-'} |")
        out.append("")

    if digest.get("objections"):
        out += ["## Objections", "", "| at | who | objection | response | landed |",
                "|---|---|---|---|---|"]
        for o in digest["objections"]:
            out.append(f"| {o['at']} | {o['who']} | {o['objection']} | "
                       f"{o.get('response') or '-'} | {o['landed']} |")
        out.append("")

    if digest.get("figures"):
        out += ["## Figures said aloud", "",
                "Listed, not checked.", "",
                "| at | who | figure | about | card |", "|---|---|---|---|---|"]
        for f in digest["figures"]:
            out.append(f"| {f['at']} | {f['who']} | {f['figure']} | {f.get('about') or '-'} | "
                       f"{f.get('card') or '-'} |")
        out.append("")

    if digest.get("next_steps"):
        out += ["## Next steps", ""]
        for s in digest["next_steps"]:
            out.append(f"- **{s['owner']}** {s['step']}  ({s['at']})")
        out.append("")

    if digest.get("cards"):
        out += [f"## Cards shown ({len(digest['cards'])})", "",
                "| at | id | question | figure | marked | in digest |",
                "|---|---|---|---|---|---|"]
        for card in digest["cards"]:
            out.append(f"| {card['at']} | {card['id']} | {card['question'] or '-'} | "
                       f"{card['figure'] or '-'} | {card['mark']} | "
                       f"{'yes' if card['in_digest'] else 'no'} |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def save(markdown: str, started_at: float | None, folder: Path | str | None = None) -> Path:
    """`~/.cache/presage/digests/<started_at>.md`, the start time as local `YYYY-MM-DDTHH-MM-SS`."""
    f = Path(folder) if folder else DIGEST_DIR
    f.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(started_at or time.time()).strftime("%Y-%m-%dT%H-%M-%S")
    path = f / f"{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- build

async def _noop_tool(name: str, args: dict) -> str:
    return json.dumps({"error": "the digest call has no tools"})


async def build(llm: Any = None, rec: Recorder | None = None, model: str | None = None,
                title: str = "", max_tokens: int = MAX_PROMPT_TOKENS,
                log_rows: Iterable[dict] | None = None, dry_run: bool = False) -> dict:
    """One digest. Returns `{"digest", "markdown", "meta"}`.

    `llm` is any `presage.llm.LLM`; the server passes the brain's own, so the digest runs on the
    same subscription the call ran on. One `answer()` call, no tools, no web, JSON schema out,
    which is also what gives the token and latency record the experiment reports.
    """
    rec = rec or recorder()
    if log_rows is not None:
        rec.fold_log(log_rows)
        rec.gate_kinds(log_rows)
    lines, cards = rec.ordered(), rec.shown()
    label_lines(lines)
    started = rec.started_at
    carded = {l.label for l in lines
              for c in cards if c.utt is not None and c.utt == l.utt and not l.me}
    pre = prepass(lines, carded)
    user, meta = build_prompt(lines, cards, pre, started, max_tokens=max_tokens,
                              carded=carded)
    meta.update({"started_at": started, "title": title or rec.title,
                 "call_s": round(lines[-1].at(started), 1) if lines else 0.0})
    spoken: dict[str, list[str]] = {}
    for lid, f in pre.figures:
        spoken.setdefault(lid, []).append(f)
    if not lines:
        empty = validate({}, lines, cards, started)
        return {"digest": empty, "markdown": render(empty, meta["title"], started), "meta": meta}
    if dry_run:
        return {"digest": None, "markdown": "", "meta": meta, "prompt": user, "system": SYSTEM}

    if llm is None:
        from presage.llm import make_llm
        llm = make_llm(os.environ.get("PRESAGE_LLM_PROVIDER", "claude_code"))
    model = model or getattr(llm, "answer_model", None) or "claude-sonnet-5"
    t0 = time.time()
    data, record = await llm.answer(model, SYSTEM, user, [], DIGEST_SCHEMA, _noop_tool,
                                    on_partial=None, web=False)
    meta["seconds"] = round(time.time() - t0, 2)
    meta["provider"] = getattr(llm, "provider", "?")
    meta["model"] = model
    meta["usage"] = {k: record.get(k) for k in
                     ("in", "out", "cache_read", "cache_write", "total_s", "stop", "tool_rounds")}
    digest = validate(data if isinstance(data, dict) else {}, lines, cards, started, spoken)
    return {"digest": digest, "markdown": render(digest, meta["title"], started), "meta": meta}


# ----------------------------------------------------------------------------- the test bed

def from_frames(run_dir: Path | str, log_dir: Path | str | None = None) -> Recorder:
    """A recorder rebuilt from one simulator run: `frames.jsonl` plus `score.json`'s gate lines.

    The E17 runs under `docs/experiments/sim/e17/` hold every frame the server broadcast, which
    is the same stream `Recorder.observe` reads live, so the test bed and the live path share
    one code path from here down. The gate verdicts live in `score.json` (the sim reads them out
    of the server log) and are attached to the line each one judged, by time.
    """
    run = Path(run_dir)
    rec = Recorder()
    rec.title = run.name
    for raw in (run / "frames.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec.observe(json.loads(raw))
        except json.JSONDecodeError:
            continue
    rows = log_rows_for(set(rec.cards), log_dir)
    if rows:
        rec.fold_log(rows)
        rec.gate_kinds(rows)
    score = run / "score.json"
    if score.exists():
        try:
            rows = json.loads(score.read_text(encoding="utf-8")).get("rows") or []
        except json.JSONDecodeError:
            rows = []
        lines = rec.ordered()
        for r in rows:                      # score.json aligns each script turn to its gate line
            gate = r.get("gate") or {}
            kind = str(gate.get("kind") or "").strip().lower()
            if not kind or r.get("speaker") != "them":
                continue
            end = float(r.get("end_s") or 0.0)
            near = [l for l in lines if not l.me and abs(l.at(rec.started_at) - end) < 6.0]
            if near:
                best = min(near, key=lambda l: abs(l.at(rec.started_at) - end))
                if not best.kind:
                    best.kind = kind
    return rec


def log_rows_for(card_ids: set[str], folder: Path | str | None = None) -> list[dict]:
    """E12's rows for these cards, out of the intervention log's day files.

    A simulator run wrote real `card` and `replaced` rows at the time, so the offline digest can
    read the same third input the live one does: the gate verdict per card, and the weak
    "replaced with nothing said" that stands in for ignored until the page has keycaps.
    """
    if not card_ids:
        return []
    from presage import feedback
    out = []
    for day in feedback.days(folder):
        out += [r for r in feedback.read(day) if str(r.get("id")) in card_ids]
    return out


async def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a post-call digest from a simulator run.")
    ap.add_argument("run_dir", help="a directory holding frames.jsonl (and score.json)")
    ap.add_argument("--provider", default=os.environ.get("PRESAGE_LLM_PROVIDER", "claude_code"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None, help="write the JSON and the markdown here")
    ap.add_argument("--max-tokens", type=int, default=MAX_PROMPT_TOKENS)
    ap.add_argument("--dry-run", action="store_true", help="build the prompt, do not call the model")
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.environ.get("PRESAGE_LOG", "WARNING"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    rec = from_frames(args.run_dir)
    llm = None
    if not args.dry_run:
        from dotenv import load_dotenv
        from presage.llm import make_llm
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        llm = make_llm(args.provider)
    out = await build(llm, rec, model=args.model, title=rec.title, max_tokens=args.max_tokens,
                      dry_run=args.dry_run)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
        if out["markdown"]:
            p.with_suffix(".md").write_text(out["markdown"], encoding="utf-8")
        print(f"wrote {p.with_suffix('.json')}")
    print(json.dumps(out["meta"], indent=1, default=str))
    if out["markdown"]:
        print()
        print(out["markdown"])
    elif args.dry_run:
        print("\n--- system ---\n" + out["system"] + "\n--- user ---\n" + out["prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
