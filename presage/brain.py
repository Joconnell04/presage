"""Speculative prefetch engine.

Flow:
  transcript (interim or final) -> topic hits -> speculative answer drafted now, kept warm
  final utterance that is a question   -> warm hit rendered instantly, live answer fired in parallel
  manual ask                            -> live answer

Every answer comes from the files: pre-packed evidence (cells + document chunks) plus a
read-only `sql` tool and a `search` tool. The model is told to say "Not in files" otherwise.
"""
from __future__ import annotations

import asyncio
import json
import math
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Awaitable, Callable

from presage.llm import LLM, ToolSpec, make_llm
from presage.llm.base import DraftDropped

try:   # typed-decision gate (Jev) with a Haiku fallback; brain still works without the module
    from presage.llm.jev import Q as DQ, make_decider
    from presage.llm import intent
    from presage.llm import ask as askmod          # the typed ask (E38); `ask` is a method here
    from presage.verify import verify
    from presage import prerun
    from presage import spoken                     # the spoken-figure guard (E9b); no network, no LLM
    from presage import drift as driftmod          # the slide-against-the-model check (E63); same
except Exception:  # noqa: BLE001
    DQ, make_decider, intent, askmod, verify, prerun, spoken, driftmod = (None,) * 8

log = logging.getLogger("presage.brain")

# Which provider answers, and which models it uses. An unset env var means the provider's own
# defaults (presage/llm/<provider>.py: default_models), which is also what the UI displays.
PROVIDER = os.environ.get("PRESAGE_LLM_PROVIDER", "claude_code")   # the Claude subscription unless told otherwise
ENV_MODELS = {role: os.environ[var] for role, var in
              (("answer", "PRESAGE_ANSWER_MODEL"), ("hedge", "PRESAGE_HEDGE_MODEL"),
               ("classifier", "PRESAGE_CLASSIFIER_MODEL")) if var in os.environ}

# The Anthropic defaults by name, because the README quotes them. Measured on this account
# (docs/research-llm.md): Opus 5 4.3 s, Sonnet 5 2.2 s, Haiku 4.5 1.2 s per answer.
ANSWER_MODEL = ENV_MODELS.get("answer", "claude-sonnet-5")
HEDGE_MODEL = ENV_MODELS.get("hedge", "claude-haiku-4-5-20251001")  # "" disables hedging
CLASSIFIER_MODEL = ENV_MODELS.get("classifier", "claude-haiku-4-5-20251001")
EFFORT = os.environ.get("PRESAGE_EFFORT", "low")   # Anthropic only, see presage/llm/anthropic.py
PARTIAL_EVERY_S = 0.12

# The spoken-figure guard (E9b, wired in E9c). A Me final is checked against the cells before it
# is dropped, and a figure the files contradict raises one quiet correction card.
SPOKEN_MIN_WORDS = 3       # "nine million" on its own names no metric; the guard would be guessing
SPOKEN_COOLDOWN_S = 60.0   # the same cell is corrected once a minute, not once a sentence: a
                           # consultant who says the figure again has already read the card
SPOKEN_BUDGET_MS = 25.0    # the check is 0.88 ms p50 / 4.6 ms p95 (E9b) so it runs inline on the
                           # transcript thread; a folder that makes it slower than this sends
                           # every later line to a worker thread instead of holding audio up

# The slide-drift check (E63). A page change raises at most one card, and a page already
# flagged is never flagged again this session: the deck does not get louder the second time
# the presenter steps back to slide 4.
DRIFT_COOLDOWN_S = 60.0    # E9c's shape, keyed on the cell: two pages quoting the same stale
                           # number inside a minute are one piece of news, not two

# E68: every question registered, whoever asks it; a `question` frame the moment it is; and a
# registered question is never met with silence (docs/experiments/E68-questions.md).
NO_ANSWER_S = 6.0          # with nothing on screen this long after a question registered, a
                           # no-answer card goes up. The transcript final lands a p50 464 ms after
                           # the question ends (E51), so this is ~6.5 s into E45's 8 s window: the
                           # card is read before the next turn typically ends (7.0 s, E45), and it
                           # sits 1.5x above the p90 first text of a live answer at E52's priority
                           # lane (3.9 s), so an answer on its way rarely gets one first. The
                           # answer call keeps running; its card replaces this one when it lands.
LIVE_TIMEOUT_S = 25.0      # and when still nothing has reached the page this long after it
                           # registered, the answer call is given up and the card says so. The API
                           # provider's own request timeout (llm/base.py TIMEOUT_S); the claude_code
                           # provider has none, so without this "Still looking" could stand forever.
                           # A call that is already streaming text onto the page is never cut off
SECOND_OPINION_P = 0.7     # the pre-gate's confident `none` on a question-shaped client line is
                           # overturned only when the tier-2 gate reads a question, request or
                           # objection that needs a lookup at this confidence, the same bar the
                           # gate-wait veto uses. E17's two "walk me through the X line next" misses
                           # read 0.94 and 0.98; the one tag statement it would also let in, 0.52
DUPLICATE_SIM = 0.80       # a question within 20 s about the same topics is skipped as a repeat
                           # only when the words say the same thing too (model2vec cosine); E17's
                           # only duplicate skip ever was a different question about the same region
SURE_P = 0.6               # a line that waited for the gate's verdict gets the no-answer card only
                           # when the verdict is a question or request at this confidence. On the
                           # E17 frames with the card modelled: 0.5 lets one cut-off line Jev read
                           # at 0.51 put up a card that is noise under scorer v2, 0.6 none, and 0.7
                           # takes it off two real questions as well (presage/sim/e68.py rescore)
ME_KIND_P = 0.5            # the consultant's own line counts as a question when the gate, reading
                           # its words as if the client had said them, calls it a question or a
                           # what-if at this probability or more (the Me policy, `me_route`)


def _saved_pref(key: str) -> Any:
    """One key out of settings.json, for a choice the server does not hand to the Brain.

    The page's own settings arrive through `set_setting` and `set_decider`; a key it has no
    control for yet (the `intent` choice, E21a) would otherwise be write-only. Never raises: a
    Brain built in a test has no settings file and gets None.
    """
    try:
        from presage import settings as _prefs
        return _prefs.load().get(key)
    except Exception:  # noqa: BLE001
        return None

DEFAULT_SETTINGS: dict[str, Any] = {
    "mode": "call",              # see MODES below
    "web_mode": "auto",          # "auto": live answers escalate to web search when the files miss; "off": never
    "qa_threshold": 0.60,        # cosine similarity for an instant answer from the precomputed Q&A cache (paraphrases 0.5-0.85, unrelated < 0.25)
    "warm_sim_threshold": 0.56,  # cosine at which a warm card is shown without asking (E7)
    "warm_sim_ask": 0.30,        # from here up to the threshold a typed decision confirms it first  # cosine similarity for matching a warm card to the landed question
    "topic_threshold": 0.6,      # min topic-match score to prefetch ("revenue" ≈ 0.8, "APAC" = 2.0, noise < 0.5)
    "question_threshold": 0.5,   # min classifier confidence to answer live
    "prefetch_cooldown_s": 8,    # don't re-prefetch the same topic within this window
    "max_warm": 6,               # ring buffer of warm candidates
    "answer_ttl_s": 90,          # candidates expire after this
    "transcript_window_s": 45,   # how much recent transcript the model sees
    "hedge_delay_s": 4.0,        # the hedge waits this long behind the primary; 0 fires both at once. 2x the measured p50 first text: cuts hedge requests 2.00 -> 1.23 per question at the same p50 (E39)
    "screen_bias": 1.0,          # E43a: the page on screen is packed first. 0 is the off switch, the same shape as web_mode
    "screen_ttl_s": 10.0,        # a reading older than this counts as no screen at all, so a poller that died cannot pin the evidence to a slide from ten minutes ago
    "drift": 1.0,                # E63: a page whose printed figure no longer matches its cell raises one quiet card. 0 is the off switch, the same shape as screen_bias
}

Emit = Callable[[dict], Awaitable[None]]


@dataclass(frozen=True)
class Mode:
    id: str
    label: str
    hint: str                       # empty-state text on the page
    prompt: str                     # appended to the system prompt
    prefetch: bool                  # speculative answers on topic hits
    auto: bool                      # answer automatically at all
    kinds: frozenset                # classifier kinds that trigger an answer
    model_knowledge: bool           # answers may be the model's own suggestion (answered_from "model")
    cues: tuple = ()                # phrases that trigger an answer without waiting for the classifier
    question_forming: bool = False  # draft as soon as a question from Them starts forming
    notes: bool = False             # keep a rolling notes card
    word_cap: int = 25


_CALL_KINDS = frozenset({"question", "request", "objection"})
MODES: dict[str, Mode] = {
    "call": Mode(
        "call", "Call", "Listening. Facts from your files appear here, cited.",
        "", prefetch=True, auto=True, kinds=_CALL_KINDS, model_knowledge=False),
    "sales": Mode(
        "sales", "Sales", "Objections and pricing come up: what to say next appears here.",
        "MODE: Sales call. The consultant is selling. When the client raises an objection, a pricing or competitor "
        "point, a risk, or hesitates, answer with what the consultant should SAY next: one or two sentences of talk "
        "track, then the strongest supporting fact from the files with its citation when one exists. General sales "
        "judgement is welcome for the framing: set answered_from=\"model\" when the substance is your suggestion, "
        "\"files\" when it rests on a file fact. Numbers still only from the files or cited web pages. Up to 40 words. "
        "In this mode never answer \"Not in files\" and never use answered_from=\"none\": there is always something "
        "useful to say next; when the files hold nothing, say it from sales judgement with answered_from=\"model\".\n\n",
        prefetch=True, auto=True, kinds=_CALL_KINDS, model_knowledge=True, word_cap=40,
        cues=("too expensive", "expensive", "pricing", "the price", "discount", "budget", "competitor", "cheaper",
              "not sure", "not convinced", "concern", "risky", "risk", "why should we", "what makes you different",
              "we already have", "think about it", "get back to you", "timeline", "contract", "roi", "payback")),
    "interview": Mode(
        "interview", "Interview", "Every question you're asked gets a suggested answer here.",
        "MODE: Interview. \"Me\" is being interviewed; \"Them\" is the interviewer. For every question asked, draft "
        "the answer Me should give: at most 40 words, first person, specific, confident. Use the files (CV, notes, "
        "portfolio, prep) whenever they hold relevant material and cite them; otherwise draw on strong general "
        "knowledge of the subject and set answered_from=\"model\". Never invent personal history the files do not "
        "support: when a question is about Me's experience and the files are silent, give the shape of a good "
        "answer (\"situation, what you did, result, about <topic>\") rather than invented facts. In this mode never "
        "answer \"Not in files\" and never use answered_from=\"none\": an outline the candidate can fill in the next "
        "five seconds is the job, with answered_from=\"model\".\n\n",
        prefetch=False, auto=True, kinds=frozenset({"question", "request"}), model_knowledge=True,
        question_forming=True, word_cap=40),
    "pitch": Mode(
        "pitch", "Pitch", "Presenting. Facts and responses to tough questions appear here.",
        "MODE: Pitch. The consultant is presenting. Answer factual questions from the files as usual. For "
        "challenges and tough questions (\"why would this work\", \"what about <risk>\"), suggest a concise response: "
        "the best supporting fact from the files plus a one-sentence framing; set answered_from=\"model\" when the "
        "framing is the substance. Up to 35 words. For a challenge, never answer \"Not in files\": if the files hold no "
        "supporting fact, suggest the response from judgement with answered_from=\"model\".\n\n",
        prefetch=True, auto=True, kinds=_CALL_KINDS, model_knowledge=True, word_cap=35,
        cues=("what about", "how would", "why would", "concern", "risk", "assumption", "what if", "pushback")),
    "notes": Mode(
        "notes", "Notes", "Taking notes. A running summary, actions and open questions appear here.",
        "", prefetch=False, auto=False, kinds=frozenset(), model_knowledge=False, notes=True),
    "quiet": Mode(
        "quiet", "Quiet", "Quiet. Nothing happens until you type a question.",
        "", prefetch=False, auto=False, kinds=frozenset(), model_knowledge=False),
}

NOTES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "array", "items": {"type": "string"}, "description": "What has been discussed, at most 5 bullets, newest developments last, each under 20 words."},
        "actions": {"type": "array", "items": {"type": "string"}, "description": "Commitments and next steps with owner when said, at most 5."},
        "questions": {"type": "array", "items": {"type": "string"}, "description": "Open questions or things to follow up, at most 3."},
    },
    "required": ["summary", "actions", "questions"],
    "additionalProperties": False,
}


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# ----------------------------------------------------------------------------- data

@dataclass
class Utterance:
    id: str
    source: str        # "mic" (Me) | "system" (Them)
    text: str
    final: bool
    ts: float          # wall clock when received
    utt: int | None = None      # per-source utterance counter from the STT provider
    retracts: bool = False      # this continues an utterance whose eager final was premature
    question_forming: bool = False   # the transcriber thinks a question is in progress (partials)


@dataclass
class Answer:
    id: str
    question: str
    answer: str
    figure: str | None
    citations: list[dict]
    source: str        # "prefetch" | "live" | "manual"
    latency_ms: int
    in_files: bool
    ts: float
    model: str
    topic: str | None = None
    answered_from: str = "files"          # "files" | "web" | "model" | "none"
    web_citations: list = field(default_factory=list)   # [{"title","url"}] when answered_from == "web"
    mode: str = "call"
    say: str = ""                         # 4-8 words the consultant can say verbatim
    cached: bool = False                  # served instantly from the precomputed Q&A cache
    utt: int | None = None                # the utterance this answers (per-source counter), when known
    working: str | None = None            # how a derived number was reached, inputs included
    derived: bool = False                 # the figure came out of whatif/calc, not out of a cell
    verdict: dict | None = None           # the verifier's result: ok, level, problems, reason (E23)
    who: str | None = None                # E68: who asked, "Them" or "Me"; None on a card nobody asked
    line: str | None = None               # E68: the transcript line id it answers ("ts7", "tm3"). `utt`
                                          # stays the client's counter only, as every reader assumes
    no_answer: dict | None = None         # E68: set on the card a registered question gets when no
                                          # answer came: {"reason", "after_ms", "detail"}

    def msg(self) -> dict:
        return {"type": "answer", **asdict(self)}


@dataclass
class Asked:
    """One registered question (E68): the line the brain decided is a question, with who asked
    it and what it will do about it, from the moment it registered until a card answers it.

    `policy` is what the brain does with it: `answer` (every client question, and a consultant
    lookup) gets a card whatever happens, a no-answer card at the worst; `check` (the consultant
    asking the client for a fact) is answered only when the files hold it, so the consultant can
    check the reply; `heard` (the consultant asking for a view, a plan, a feeling) is registered
    and nothing more. The page learns of it through the `question` frame (docs/ui-protocol.md).
    """
    qid: str
    aid: str                        # the card id that will answer it; the page can hold a slot for it
    text: str
    who: str                        # "Them" | "Me"
    policy: str                     # "answer" | "check" | "heard"
    kind: str                       # question | what-if | request | objection
    via: str                        # heuristic | second-opinion | gate | interview | me | typed | joined
    t_q: float
    utt: int | None = None
    line: str | None = None         # the transcript line id ("ts7", "tm3")
    utts: list = field(default_factory=list)   # every utterance folded into it (a joined final)
    status: str = "heard"           # heard | answered | no-answer | quiet
    sure: bool = True               # something read it as a question: it gets the no-answer card
    shown: bool = False             # a card (partial or settled) for `aid` has reached the page
    hits: list = field(default_factory=list)   # its topic hits, which a follow-up borrows
    watch: asyncio.Task | None = None
    task: asyncio.Task | None = None   # the live answer call, so the deadline can give it up

    def msg(self) -> dict:
        return {"type": "question", "id": self.qid, "aid": self.aid, "text": self.text,
                "who": self.who, "source": "mic" if self.who == "Me" else "system",
                "utt": self.utt, "utts": list(self.utts), "line": self.line, "kind": self.kind, "policy": self.policy,
                "via": self.via, "sure": self.sure, "status": self.status, "ts": self.t_q}


@dataclass
class Candidate:
    """One speculative answer, built from the top topic hits of one utterance."""
    id: str
    topic: str                     # display label, e.g. "FY25 actual Q3 · Revenue"
    keys: frozenset[str]           # normalized topic keys it covers
    status: str                    # "warming" | "ready" | "shown" | "expired"
    created: float
    hits: list = field(default_factory=list)
    question_guess: str = ""
    answer: Answer | None = None
    task: asyncio.Task | None = None
    answer_id: str | None = None   # set when a question landed while still warming
    asked: str = ""                # the line that landed while this was still warming
    asked_kind: str = ""           # the pre-gate's kind for that line, once it has been read
    utt_id: str | None = None      # interview mode: the utterance whose question is still forming
    tentative: bool = False        # cosine in the ask band: confirm with a typed decision before showing

    @property
    def key(self) -> str:
        return "|".join(sorted(self.keys))

    def msg(self) -> dict:
        preview = self.answer.answer if self.answer else self.question_guess
        return {   # "shown" is internal; the page only knows warming / ready / expired
            "type": "candidate", "id": self.id, "topic": self.topic,
            "status": "expired" if self.status == "shown" else self.status,
            "preview": preview, "question": self.question_guess, "ts": self.created,
            "figure": self.answer.figure if self.answer else None,
        }


def topic_key(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", topic.lower()).strip()


# ----------------------------------------------------------------------------- heuristics

_Q_START = re.compile(
    r"^(what|what's|whats|how|why|when|where|which|who|can|could|do|does|did|is|are|was|were|"
    r"will|would|should|have|has|remind|walk|tell|give|show|any|anything|talk)\b"
)
_Q_PHRASES = (
    "walk me through", "talk me through", "remind me", "what about", "how much", "how many",
    "do we have", "do you have", "can you", "could you", "what's the", "whats the", "what is",
    "what was", "what were", "what did", "tell me", "give me", "any idea", "where are we",
    "how does that compare", "is that in", "what's our", "what are", "do we know", "i'd love to see",
    "curious about", "help me understand", "pull up", "can we see", "what does", "how did",
)


def questionish(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    if _Q_START.match(t):
        return True
    return any(p in t for p in _Q_PHRASES)


_WHATIF_PHRASES = (
    "what if", "what would happen", "what happens if", "suppose", "supposing", "instead of",
    "rather than", "hypothetical", "if we charged", "if we charge", "if we changed", "if we change",
    "if we raised", "if we raise", "if we cut", "if we dropped", "if we drop", "if we moved",
    "if we move", "if we took", "if we went", "if it were", "if that were", "sensitivity",
)
# "at 1%", "to 1.25 percent", "2x instead": a rate, not just any number, or every figure said out
# loud would look like a hypothetical
_AT_RATE = re.compile(r"\b(?:at|to)\s+\$?\d[\d,.]*\s*(?:%|percent|per cent|bps|x)(?![a-z0-9])")
_RATE_INSTEAD = re.compile(r"\d[\d,.]*\s*(?:%|percent|per cent|bps|x)?\s+instead\b")
_HAS_NUMBER = re.compile(r"\d")


TOOL_BUDGET = 3              # tool calls per answer; the prompt asks for one


def whatif_ish(text: str) -> bool:
    """A hypothetical that moves a number: a recalculation is wanted, not a lookup."""
    t = text.strip().lower()
    if not t:
        return False
    if any(p in t for p in _WHATIF_PHRASES):
        return True
    return bool(_AT_RATE.search(t) or _RATE_INSTEAD.search(t))


# E68. The consultant's own questions. The gate reads the words as if the client had said them
# (`Brain._me_reading`), and these three patterns route what it calls a question: a tag or a check
# that the room follows is no question at all; a recall marker, "our" figures or the materials named,
# with nobody addressed, is a lookup the copilot answers like a client question; a question about
# the other side's views, plans or feelings is registered and nothing more. Measured on 2,394 real
# consultant lines (data/gate/real, labelled by data/gate/me_label.py) and the synthetic set.
_SECOND_PERSON = re.compile(r"\b(you|your|yours|you're|youre|you've|youve|you'd|you'll|y'all|yall|ya)\b", re.I)
_ME_CONFIRM = re.compile(
    r"(,|\s)(right|correct|yeah|yes|no|okay|ok|agreed|fair|huh|isn'?t it|doesn'?t it|don'?t you think|"
    r"you know|wouldn'?t you say|aren'?t they|isn'?t that)\s*[?.!]*\s*$|"
    r"\b(does|did) (that|this|it) make sense\b|\bmakes? sense\b\s*[?.!]*$|\bsounds? good\b|"
    r"\bcan (you|everyone|everybody|y'all) (see|hear)\b|\bany (questions|thoughts)\b|\bshall we\b|"
    r"\bshould (we|i) (move|go|keep|start|jump|continue|dive|pause|stop|wrap)\b|\bready to\b|"
    r"\b(are|is) (you|everyone|everybody) (ready|good|okay|ok|there|still there|with me)\b|"
    r"\bwho (wouldn'?t|doesn'?t|isn'?t|'?s going to|would|could)\b", re.I)
_ME_FIGURE = (r"(number|numbers|figure|figures|data|breakdown|split|count|total|totals|rate|rates|revenue|"
              r"cost|costs|margin|margins|churn|price|prices|fee|fees|headcount|subscribers|estimate|forecast|"
              r"budget|actuals|percent|percentage|share|growth|handy)")
_ME_LOOKUP = re.compile(
    r"(?<!\bsay that )(?<!\bsay it )(?<!\bcome )(?<!\brepeat that )\bagain\b|\bremind me\b|"
    r"\brefresh my memory\b|\b(where|what) did we (land|end up|settle)\b|"
    r"\bwhat did we (assume|use|put|budget|plan|say) (for|on|in)\b|\bwhat('?s| is| was| were| are) our\b|"
    r"\bhow (much|many)\b[^?.]{0,30}\b(do|did|have|had|are|were) we\b|"
    r"\b(do|did) we (have|know)\b[^?.]{0,25}\b" + _ME_FIGURE + r"\b|"
    r"\b(in|on) (the )?(model|deck|memo|workbook|spreadsheet|sheet|slide|page)\b|"
    r"\b(model|deck|memo|workbook|sheet) (say|says|show|shows|have|has)\b|"
    r"\bthe (current|latest|going) (rate|price)\b|\bthese days\b", re.I)
_ME_VIEW = re.compile(
    r"\b(think|thinking|feel|feeling|expect|expected|expects|expecting|want|wanted|prefer|rather|"
    r"comfortable|confident|worried|worry|worries|concern|concerned|sense|agree|happy|plan to|planning|"
    r"intend|like to|board|opinion|view|take on|sure|priority|priorities|goal|goals|"
    r"approach|decide|decision|appetite|open to|excited|hope|hoping|believe|seeing|noticed|hearing)\b|"
    r"\bwork for\b|\bline up\b|\bmatch\b|\bmake sense\b|\bhow (is|are|was|were) (it|things|that) going\b",
    re.I)


# E68. "And the year after that." / "And the retail partner." / "Okay, and the platform fee itself.":
# a client's elliptical follow-up, straight after a question. The transcriber writes a full stop,
# the pre-gate reads a fragment with no verb as `none`, and E68's big-04 run left all three of
# these silent. Verbless, short, opening on "and" / "what about" / "same question for", and only
# within FOLLOW_UP_S of a client question that registered: on the 1,110 real client lines that are
# no question the pattern alone matches 16, and 2 of them follow a question-shaped client line.
_FOLLOW_UP = re.compile(
    r"^\s*(?:(?:okay|ok|so|right|alright|all right|and|um|uh)[\s,]+)*"
    r"(?:and|what about|how about|same (?:question|thing) for|same for)\b", re.I)
_FOLLOW_VERB = re.compile(
    r"\b(is|are|was|were|be|been|being|am|'s|'re|'m|do|does|did|have|has|had|can|could|will|would|"
    r"should|shall|may|might|must|'ll|'d|'ve|go|goes|went|get|gets|got|make|makes|made|think|thinks|"
    r"jump|jumps|say|says|said|know|mean|means|want|wants|need|needs|look|looks|seem|seems|feel|feels|"
    r"take|takes|took|come|comes|came|keep|keeps|let|let's|see|sees|thank|thanks)\b", re.I)
FOLLOW_UP_S = 30.0         # a follow-up leans on a question this recent (a chain of short turns)


def follow_up_shaped(text: str) -> bool:
    """An elliptical follow-up question: short, verbless, "and ..." / "what about ..."."""
    t = (text or "").strip()
    return (bool(t) and len(t.split()) <= 9 and bool(_FOLLOW_UP.search(t)) and not _FOLLOW_VERB.search(t)
            and not re.search(r",\s*(right|correct|yeah|okay|ok)\s*[?.!]*$", t, re.I))


def me_candidate(text: str) -> bool:
    """Worth asking the gate about at all: shaped like a question. 0.01 ms, and it keeps the
    tier-2 read off the nine consultant lines in ten that are statements."""
    t = (text or "").strip()
    return bool(t) and (questionish(t) or whatif_ish(t) or t.endswith("?"))


def me_route(text: str, kind: str | None, p_kind: float, needs: bool) -> str:
    """The Me policy (E68): what the brain does with a consultant's line, given the gate's reading
    of its words as if the client had said them. "" is nothing at all.

      answer  a lookup ("what was the Q3 number again?"): answered like a client question
      check   a fact asked of the client ("what's your target margin?"): a card only when the files
              hold it, so the consultant can check the reply against them
      heard   a view, a plan, a feeling asked of the client: registered, no card
    """
    if kind not in ("question", "what-if") or float(p_kind or 0.0) < ME_KIND_P:
        return ""
    t = (text or "").strip()
    if _ME_CONFIRM.search(t):
        return ""
    if _ME_LOOKUP.search(t) and not _SECOND_PERSON.search(t):
        return "answer"
    if (kind == "what-if" or whatif_ish(t)) and not _SECOND_PERSON.search(t):
        # "would we need to raise in year zero if we took out five million?": a hypothetical on
        # the model, answered like a client's what-if (the view words "need", "want" used to
        # send these to `heard`; Jackson's test, 2026-09-24)
        return "answer"
    if needs and _SECOND_PERSON.search(t):
        # a fact asked OF the client addresses them ("what's your target margin?"). Without a
        # "you" the line is as likely asking the room for a view, and E68's big-04 run put a noise
        # card on exactly that ("is that high against the tools they use now, or against the
        # channels?" answered with the fee); on the real lines it removes all four checks that
        # landed on a line that asked nothing
        return "check"
    if needs and not _SECOND_PERSON.search(t):
        # "What is five year revenue?", "what would the equity value be?": a factual question
        # with nobody addressed. Jackson (2026-09-23): register it AND try the files, and when
        # the files hold nothing show the no-answer card rather than nothing. The E68 policy kept
        # these at `heard`; on the real consultant lines that moves the 18 false registrations
        # in 2,304 from silent to a possible no-answer card, which is the trade that was asked for.
        return "answer"
    return "heard"


# E43a. A pointing word with no metric of its own is a question about whatever is on screen.
# "this number", "that slide": the words name something the page is showing, not something the
# workbook holds.
_POINTS_AT = re.compile(
    r"\b(?:this|that|these|those)\s+(?:one|ones|thing|things|bit|part|number|numbers|figure|"
    r"figures|total|totals|split|splits|breakdown|chart|charts|graph|bar|bars|line|lines|row|"
    r"rows|column|columns|cell|cells|table|tables|slide|slides|page|pages|deck|memo|section|"
    r"box|screen|exhibit|panel|tab|math|calc|calculation|build|waterfall|bridge)\b", re.I)
# "what's this based on", "where does that come from", "how does it tie out": the pointing word is
# a bare pronoun, so the line names nothing to look up either.
_BARE_POINT = re.compile(
    r"\b(?:this|that|it)\b\s*(?:[?.,!]|$|\b(?:is|are|was|were|come|comes|came|coming|based|from|"
    r"tie|ties|tied|mean|means|meant|work|works|worked|show|shows|do|does|did|go|goes|get|gets|"
    r"look|looks|seem|seems|say|says|include|includes|assume|assumes|reflect|reflects|roll|rolls|"
    r"square|squares|reconcile|reconciles|add|adds|sit|sits|hold|holds|break|breaks|compare|"
    r"compares|versus|vs|against|actually|really|even|right|correct|built|calculated|computed|"
    r"derived|made|sourced|driven|produced|rounded|stated|shown|the|a|an|our|your|their)\b)", re.I)
# "the figure here", "what's the number there?" -- "there are four regions" is not pointing
_HERE_POINT = re.compile(r"\bhere\b|\b(?:up|down|over|in|on|out)\s+there\b|\bthere\s*[?.!]?\s*$", re.I)


def deictic(text: str, ask: Any = None) -> bool:
    """"what's this number based on", "walk me through this slide", "how does that tie to the model".

    False when the line names a metric of its own ("what is this quarter's churn" points at a
    period, not at the screen) or carries its own figure, because then the words are the referent
    and the page is only background. The three patterns are what separates a pointing word from a
    determiner: a screen noun after it, a bare pronoun, or "here".
    """
    if not text or _HAS_NUMBER.search(text):
        return False
    if not (_POINTS_AT.search(text) or _BARE_POINT.search(text) or _HERE_POINT.search(text)):
        return False
    if ask is not None and getattr(ask, "slots", None):
        try:
            slots = ask.slots()
        except Exception:  # noqa: BLE001
            return True
        if slots.get("metric") or slots.get("row"):
            return False
    return True


_NUM = re.compile(r"(\$?)(\d[\d,]*(?:\.\d+)?)\s*(%|percent|k|m|million|thousand|x)?(?![a-z])", re.I)


_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMWORD = re.compile(r"\b((?:(?:%s)(?:[\s-]+|$))+(?:point(?:\s+(?:%s))+\s*)?)" % (
    "|".join(list(_ONES) + list(_TENS) + list(_SCALE) + ["and", "a"]), "|".join(_ONES)), re.I)


def _words_to_digits(text: str) -> str:
    """"twelve percent" -> "12 percent", "one point two five" -> "1.25", "two million" -> "2000000".
    The transcriber spells small numbers out some of the time and writes digits the rest."""
    def convert(m: re.Match) -> str:
        words = re.split(r"[\s-]+", m.group(1).strip().lower())
        total = cur = 0
        seen = False
        frac = ""
        i = 0
        while i < len(words):
            w = words[i]
            if w == "point":
                for d in words[i + 1:]:
                    if d in _ONES and _ONES[d] < 10:
                        frac += str(_ONES[d])
                break
            if w in ("and", "a"):
                i += 1
                continue
            if w in _ONES:
                cur += _ONES[w]; seen = True
            elif w in _TENS:
                cur += _TENS[w]; seen = True
            elif w == "hundred":
                cur = (cur or 1) * 100; seen = True
            elif w in _SCALE:
                total += (cur or 1) * _SCALE[w]; cur = 0; seen = True
            i += 1
        if not seen or all(w in _SCALE or w in ("and", "a") for w in words if w != "point"):
            return m.group(0)                     # "220 thousand": the digits carry the number
        val = total + cur
        out = f"{val}.{frac}" if frac else str(val)
        return out + (" " if m.group(0).endswith((" ", "-")) else "")
    return _NUMWORD.sub(convert, text or "")


def _numbers_in(text: str) -> set[float]:
    """Every number the asker said, as the value a cell would hold (12% -> 0.12, $4.2M -> 4200000)."""
    out: set[float] = set()
    text = _words_to_digits(text)
    for dollar, num, unit in _NUM.findall(text or ""):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        u = (unit or "").lower()
        if u in ("%", "percent"):
            out.add(v / 100.0)
        elif u in ("k", "thousand"):
            out.add(v * 1e3)
        elif u in ("m", "million"):
            out.add(v * 1e6)
        else:
            out.add(v)
    return out


_ANAPHORA = re.compile(r"^(and|what about|how about)\b|\b(that|this|it|those|these|the same)\b", re.I)


def _anaphoric(text: str) -> bool:
    """A short follow-up that leans on what was just said cannot be answered from a cached entry."""
    t = (text or "").strip()
    return len(t.split()) <= 12 and bool(_ANAPHORA.search(t))


_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "at", "by", "with", "as", "is", "are",
         "were", "was", "be", "that", "this", "it", "we", "our", "if", "what", "how", "would", "happen", "instead",
         "current", "went", "goes", "go", "to", "from", "case", "under", "structure", "did", "do", "does"}


def _word_set(text: str) -> set[str]:
    """Lower-case content words, plurals folded, for lever matching."""
    out = set()
    for w in re.findall(r"[a-z0-9%.]+", (text or "").lower()):
        w = w.strip(".")
        if w in _STOP or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def _asked_values(text: str) -> list[tuple[float, bool]]:
    """Every number the asker said with whether it was a percentage, as a cell would hold it."""
    out: list[tuple[float, bool]] = []
    text = _words_to_digits(text)
    if prerun is not None:                     # "Q3", "FY26", "third quarter" are columns (E70)
        text = prerun.blank_periods(text)
    for dollar, num, unit in _NUM.findall(text or ""):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        u = (unit or "").lower()
        if u in ("%", "percent"):
            out.append((v / 100.0, True))
        elif u in ("k", "thousand"):
            out.append((v * 1e3, False))
        elif u in ("m", "million"):
            out.append((v * 1e6, False))
        else:
            out.append((v, False))
    return out


def _same_value(a: float, b: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9 + 0.005 * max(abs(float(a)), abs(float(b)))
    except (TypeError, ValueError):
        return False


@dataclass
class _LeverHit:
    """TopicHit's shape, for when presage.ingest.topics cannot be imported."""
    topic: str
    score: float
    kind: str
    targets: list[dict] = field(default_factory=list)


_ASKED = re.compile(r"\s*what if (?:the |we |our )?(.+?)\s+(?:were|was|is|goes to|went to|becomes)\b", re.I)
_MOVED = re.compile(r"(?:→|->)\s*([^\s,;]+)")


def _asked_label(question: str | None) -> str | None:
    """The lever out of a grid question ("What if the GMV fee were 1.25%?" -> "GMV fee")."""
    m = _ASKED.match(question or "")
    return m.group(1).strip() if m else None


_WORK = re.compile(r"^\s*(?P<cell>.+?)\s+(?P<was>\S+)\s*(?:→|->)\s*(?P<now>[^,;:]+)")


def _moved_to(working: str | None) -> str:
    """What a working line says the lever moved to ("B48 0.85% → 1.25%, ..." -> "1.25%")."""
    m = _MOVED.search(working or "")
    return m.group(1) if m else ""


def cell_map(v: Any) -> dict:
    """`sets` / `inputs` as a plain dict, however the model shaped it: {"Sheet!A1": "1%"} or
    [{"cell": "Sheet!A1", "value": "1%"}] / [{"name": "assets", "ref": "Sheet!B57"}]."""
    if isinstance(v, dict):
        return {str(k): (x if isinstance(x, (int, float)) else str(x)) for k, x in v.items()}
    out: dict[str, Any] = {}
    for item in v if isinstance(v, list) else []:
        if not isinstance(item, dict):
            continue
        key = item.get("cell") or item.get("name") or item.get("ref")
        val = item.get("value", item.get("ref"))
        if key is not None and val is not None:
            out[str(key)] = val if isinstance(val, (int, float)) else str(val)
    return out


def lever_hit(lever: dict, score: float = 1.2):
    """A topic hit for a lever the utterance named. The topic index only knows row labels, so a
    spoken alias like "GMV fee" has to become a hit here."""
    target = {"kind": "lever", "file": lever.get("file"), "sheet": lever.get("sheet"),
              "cell": lever.get("cell"), "label": lever.get("label"), "value": lever.get("value"),
              "fmt": lever.get("fmt"), "outputs": list(lever.get("outputs") or [])[:6]}
    topic = str(lever.get("label") or lever.get("cell") or "lever")
    try:
        from presage.ingest.topics import TopicHit
        return TopicHit(topic, score, "lever", [target])
    except Exception:  # noqa: BLE001
        return _LeverHit(topic, score, "lever", [target])


# ----------------------------------------------------------------------------- prompts

_QUERY_SCHEMA: dict = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

_SETS_ITEMS = {"type": "array", "description": "Cells to change: [{\"cell\": \"Sheet!A1\", \"value\": \"1%\"}]; a text "
                                                  "lever takes one of its choices as the value.",
               "items": {"type": "object", "properties": {"cell": {"type": "string"}, "value": {"type": "string"}},
                         "required": ["cell", "value"], "additionalProperties": False}}

SCRATCH_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="goalseek",
        description="Find the input value that makes a target cell hit a value: \"what fee gets us to $10M\", "
                    "\"how many subscribers for breakeven\". Searches the client's own model (bisection on the lever) "
                    "and returns the lever value found, the value achieved and whether the relationship is monotone.",
        input_schema={"type": "object",
                      "properties": {"file": {"type": "string", "description": "Workbook; omit when only one is loaded."},
                                     "target": {"type": "string", "description": "Output cell, \"Sheet!A1\"."},
                                     "value": {"type": "string", "description": "The value the target should reach, e.g. \"$10M\"."},
                                     "lever": {"type": "string", "description": "The input cell to move, from the LEVERS table."}},
                      "required": ["target", "value", "lever"], "additionalProperties": False},
    ),
    ToolSpec(
        name="explain",
        description="Why an output moved (or did not): walks the formula chain from an output cell back to the "
                    "changed input, up to six hops, with each intermediate cell's label, formula, old and new value. "
                    "Use for \"why did costs go up\", \"how did you get that\", \"what feeds this number\".",
        input_schema={"type": "object",
                      "properties": {"file": {"type": "string"}, "cell": {"type": "string", "description": "The output cell, \"Sheet!A1\"."},
                                     "sets": _SETS_ITEMS},
                      "required": ["cell"], "additionalProperties": False},
    ),
    ToolSpec(
        name="compare",
        description="Several what-if cases side by side: \"premium vs market at 1%\", \"base vs downside\". Each case "
                    "is its own list of cell changes; returns every watched output per case with best, worst and spread.",
        input_schema={"type": "object",
                      "properties": {"file": {"type": "string"},
                                     "cases": {"type": "array", "items": {"type": "object",
                                               "properties": {"label": {"type": "string"}, "sets": _SETS_ITEMS},
                                               "required": ["label", "sets"], "additionalProperties": False}},
                                     "watch": {"type": "array", "items": {"type": "string"}}},
                      "required": ["cases"], "additionalProperties": False},
    ),
    ToolSpec(
        name="sweep",
        description="One input over several values (\"at 0.5, 1 and 1.5 percent\", \"from 5% to 15% churn\"): the "
                    "watched outputs at each point.",
        input_schema={"type": "object",
                      "properties": {"file": {"type": "string"}, "cell": {"type": "string"},
                                     "values": {"type": "array", "items": {"type": "string"}},
                                     "watch": {"type": "array", "items": {"type": "string"}}},
                      "required": ["cell", "values"], "additionalProperties": False},
    ),
]

TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="find_cells",
        description="Find every spreadsheet cell whose row label, column header or text contains the pattern "
                    "(case-insensitive substring, e.g. 'gross margin', 'Q3', 'APAC'). Returns up to 40 hits with "
                    "file › sheet › cell, header, row label and value. The fastest way to locate a number anywhere "
                    "in the workbooks; do this before writing SQL when the evidence lacks the number.",
        input_schema={"type": "object", "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"], "additionalProperties": False},
    ),
    ToolSpec(
        name="read",
        description="Read one whole unit of a file to see the context around a hit: for documents give the "
                    "file name and 'page 3', 'slide 4' or a section heading; for spreadsheets give the file "
                    "name and a sheet name (returns that sheet's rows, up to 60).",
        input_schema={"type": "object", "properties": {"file": {"type": "string"}, "loc": {"type": "string"}},
                      "required": ["file", "loc"], "additionalProperties": False},
    ),
    ToolSpec(
        name="find_numbers",
        description="Find numbers stated in the documents (deck, memo, pdf, notes) with the sentence they appear "
                    "in, e.g. 'growth', '23%', 'churn'. Use it when a figure is quoted in prose rather than in a "
                    "spreadsheet cell.",
        input_schema={"type": "object", "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"], "additionalProperties": False},
    ),
    ToolSpec(
        name="list_files",
        description="List the context files with their sheets or pages/slides and what each contains. "
                    "Use it when unsure which file could hold the answer.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    ToolSpec(
        name="search",
        description="Full-text search over the document chunks (pdf, pptx, docx, txt, md). "
                    "Returns the best matching passages with file and page/slide citations.",
        input_schema=_QUERY_SCHEMA,
    ),
    ToolSpec(
        name="sql",
        description="Run ONE read-only SQLite SELECT against the spreadsheet tables in the schema. "
                    "Use it for filters, aggregates, comparisons, or when the pre-fetched evidence "
                    "does not contain the number. Returns up to 50 rows as JSON.",
        input_schema=_QUERY_SCHEMA,
    ),
    ToolSpec(
        name="whatif",
        description="Recalculate the client's own model with one or more input cells changed, and read back "
                    "what those cells drive. Use it whenever the question moves a number the model already "
                    "holds: 'what if we charged 1%', 'suppose the fee went to 1.25%', '400 subscribers instead of "
                    "250'. `sets` is the list of cells to change, e.g. "
                    "[{\"cell\": \"Subscriber Fee Model!B48\", \"value\": \"1%\"}]: a cell is \"Sheet!A1\" and a "
                    "value may be written \"1%\", \"$4.2M\", \"1.2x\" or as a plain number. Change only cells "
                    "the LEVERS table lists; a formula cell is rejected. `file` is the workbook name, needed "
                    "only when several are loaded. `watch` names output cells to report on top of the lever's "
                    "own. Returns {changes, outputs: [{cell, label, was, now, delta, pct}], partial, ms}; when "
                    "`partial` is not empty some cells kept a cached value, so say 'partial recalc'.",
        input_schema={
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Workbook file name; omit when only one is loaded."},
                "sets": {
                    "type": "array",
                    "description": "The cells to change and what to change them to.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cell": {"type": "string", "description": "\"Sheet!A1\", from the LEVERS table."},
                            "value": {"type": "string", "description": "The new value, e.g. \"1%\" or \"250\"."},
                        },
                        "required": ["cell", "value"],
                        "additionalProperties": False,
                    },
                },
                "watch": {"type": "array", "items": {"type": "string"},
                          "description": "Extra output cells to report, e.g. [\"Summary Scenarios!G57\"]."},
            },
            "required": ["sets"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="calc",
        description="Arithmetic the model does not contain, over inputs bound to cells you cite. Use it when "
                    "the answer needs a number no cell holds and no lever moves: a share, a per-unit figure, a "
                    "growth rate, the asker's own rate applied to a file number. Each input binds a name to a "
                    "cell reference (\"Subscriber Fee Model!B57\", or \"file!Sheet!A1\" across workbooks), to a "
                    "document number reference from find_numbers, or to a literal number the asker gave, which "
                    "the answer must attribute to them. `expr` uses those names with + - * / ** %, parentheses, "
                    "and min max abs round sum avg cagr(start, end, years) pct(a, b); nothing else parses. "
                    "Example: expr \"assets * rate\", inputs [{\"name\": \"assets\", \"ref\": \"Subscriber Fee "
                    "Model!B57\"}, {\"name\": \"rate\", \"ref\": \"0.01\"}]. Returns {value, expr, inputs}.",
        input_schema={
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "e.g. \"assets * rate\" or \"pct(enterprise, total)\"."},
                "inputs": {
                    "type": "array",
                    "description": "Every name the expression uses, bound to where its value comes from.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The name used in expr, e.g. \"assets\"."},
                            "ref": {"type": "string",
                                    "description": "A cell reference, a find_numbers reference, or a literal number."},
                        },
                        "required": ["name", "ref"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["expr", "inputs"],
            "additionalProperties": False,
        },
    ),
]
TOOLS += SCRATCH_TOOLS
SCRATCH_TOOL_NAMES = ("whatif", "calc", "goalseek", "explain", "compare", "sweep")

ANSWER_REQUIRED = ("answer", "say", "figure", "citations", "answered_from")
ANSWER_OPTIONAL = ("working", "web_citations", "question")


def answer_schema(speculative: bool = False, web: bool = False, derived: bool = False) -> dict:
    """The answer JSON. One shape for every call, because the shape is a cached prefix.

    The three variable fields used to be added per call, which made four schema variants; the
    Agent SDK fixes `output_format` at connect time and the session pool key carries the schema,
    so each variant was its own prompt-cache prefix and the first call on a cold one paid a
    31-41k token write (E39). They are now always present and never required: the model omits
    the ones that do not apply, so the output stays as short as the trimmed variants were, and
    speculative, live and derived answers share one prefix per model (E42).

    Which optional field to fill is said in the system prompt and in the per-call trigger, both
    of which the model reads anyway: the trigger is user text, so saying it there costs no cache.
    The arguments are kept so call sites read as before; they no longer change the schema.
    """
    props = {k: ANSWER_SCHEMA["properties"][k] for k in ANSWER_REQUIRED + ANSWER_OPTIONAL}
    return {"type": "object", "properties": props, "required": list(ANSWER_REQUIRED),
            "additionalProperties": False}


def complete_answer(data: dict) -> dict:
    """Fill the fields a trimmed schema left out, so the rest of the brain sees one shape."""
    from_ = data.get("answered_from") or ("files" if data.get("in_files") else "none")
    data["answered_from"] = from_
    data.setdefault("in_files", from_ in ("files", "derived"))
    data.setdefault("working", None)
    data.setdefault("web_citations", [])
    data.setdefault("question", "")
    data.setdefault("citations", [])
    return data


ANSWER_SCHEMA: dict = {   # "answer" first on purpose: it streams to the screen before the rest is generated
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "At most 25 words for a number; up to 45 for scope, context, a method or a list, naming the items. Plain statement, no preamble."},
        "say": {"type": "string", "description": "4 to 12 words the consultant can say out loud verbatim, e.g. 'Q3 came in at 14.0 million' or 'Priorities: pipeline, pricing, partners, hiring'. For a list, the items themselves. No numbers that are not in `answer`."},
        "figure": {"anyOf": [{"type": "string"}, {"type": "null"}],
                   "description": "The single key number with unit, e.g. '$4.2M' or '12.5%', else null. Null for scope, context, meaning, a summary, a why or a list: a count is not a figure."},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "loc": {"type": "string"}},
                "required": ["file", "loc"],
                "additionalProperties": False,
            },
        },
        "working": {"anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "The arithmetic or the lever change, with the input values, e.g. "
                                   "'$2.60M assets × 1% = $26,050' or 'Subscriber Fee Model B48 0.85% → 1%, model "
                                   "recalculated'. At most 14 words. OPTIONAL: leave the field out "
                                   "entirely when nothing was computed."},
        "in_files": {"type": "boolean", "description": "true only when the answer comes from the files."},
        "answered_from": {"type": "string", "enum": ["files", "web", "model", "derived", "none"],
                          "description": "files: from the context files; web: from cited web pages; model: your own "
                                         "suggestion (only when the mode allows it); derived: computed by the whatif "
                                         "or calc tool from file numbers; none: not found."},
        "web_citations": {
            "type": "array",
            "description": "Web pages the answer relies on. OPTIONAL: leave the field out entirely "
                           "unless answered_from is web.",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
                "required": ["title", "url"],
                "additionalProperties": False,
            },
        },
        "question": {"type": "string",
                     "description": "OPTIONAL: fill it only on a speculative trigger, with your best guess at "
                                    "the question; leave it out when the trigger carries the question."},
    },
    "required": ["answer", "say", "figure", "citations", "working", "in_files", "answered_from",
                 "web_citations", "question"],
    "additionalProperties": False,
}

_PARTIAL_ANSWER = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)')


def partial_answer(buf: str) -> str | None:
    """The `answer` string from a partially generated JSON object, or None if it hasn't started."""
    m = _PARTIAL_ANSWER.search(buf)
    if not m:
        return None
    try:
        return json.loads('"' + m.group(1) + '"')
    except json.JSONDecodeError:
        return m.group(1)

CLASSIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["question", "request", "objection", "none"]},
        "needs_files": {"type": "boolean", "description": "true if answering needs a looked-up number, date, name or fact (from the consultant's files or public sources); false for small talk, opinions and things answered from the head"},
        "confidence": {"type": "number"},
        "question": {"type": "string", "description": "The ask rephrased as one standalone question, or empty"},
    },
    "required": ["kind", "needs_files", "confidence", "question"],
    "additionalProperties": False,
}

SYSTEM_HEAD = """You are presage, a silent assistant for a strategy consultant ("Me") on a live client call ("Them"). \
You see the recent transcript and a folder of context files: spreadsheets loaded as SQL tables, documents as text. \
The consultant reads your answer on a second screen while talking, so it must be short and exact.

Rules:
- Answer in at most 25 words, then one key figure if there is one. No preamble, no hedging, no restating the question.
- A question about scope, context, meaning, method, a summary or a why is answered in words: `figure` is null \
and `answer` may run to 45 words. Never manufacture a headline number for it. A count of things ("4 \
priorities") is never the answer by itself: name the things in `say` or `answer`, since the card is read \
mid-sentence by someone who cannot click through.
- Evidence rows marked "input" are assumptions typed into the model; a share, a total or a result comes from \
a row that is that result, or is computed from the two file numbers with the working shown. Never present an \
assumption as a market fact.
- Every number must come from the files (the pre-fetched evidence or a tool result) or, when web tools are \
offered, from a web page you cite. Never compute from memory or general knowledge. Simple arithmetic on file \
numbers is fine if you show both inputs.
- Search the files properly before giving up: `find_cells` locates a label or number anywhere in the workbooks, \
`read` shows a whole sheet, page or slide, `list_files` shows what exists, `sql` aggregates and filters, `search` \
finds passages in documents. Usually 0-2 tool calls; up to 6 when the first lookups miss.
- A number that is in no cell and no sentence must come from `whatif` or `calc`. Never work it out yourself. \
Use `whatif` when the question changes an input the LEVERS table lists, so the model recalculates itself; \
`calc` otherwise, with every input bound to a cell you cite. For a what-if, call `whatif` first and at once, \
with the LEVERS cell and the asker's value; no lookups before it. One tool call is the norm, three is the most. \
The LEVERS table is a sample: an input that is not listed may still be a lever, so pass its spoken name as the \
cell ("subscriber churn") and the server resolves it; never say something cannot be changed without trying. \
`goalseek` answers "what X gets us to Y", `explain` answers "why did it move" or "how did you get that", \
`compare` puts scenarios side by side, `sweep` runs one input over several values; a scenario name \
("market", "downside") is a value for a text lever.
- For a derived number: answered_from="derived", `working` filled with the inputs and the operation, and the \
citations are the cells the inputs came from. Name the asker's own number as theirs ("at your 1%"). \
If `whatif` returns a non-empty `partial`, say "partial recalc".
- If the files do not contain the answer and no web tools are offered: in_files=false, answered_from="none", \
answer "Not in files: <what you looked for>". Never guess.
- answered_from="model" is allowed only when the MODE section below says so; in plain call mode every answer is \
files, web or none.
- Web tools (only when offered in the request): use them only after the files have been searched and do not hold \
the answer, and only for public facts: market data, benchmarks, definitions, public company figures, current \
events. Then set answered_from="web", in_files=false, list every page you relied on in web_citations with its URL, \
and say the figure's date if the page gives one. Never use the web for something the files should contain.
- An objection ("that feels high", "I'm not sure the board will go for that") is answered with the basis of the \
figure being challenged: the cell it comes from, how it was built, and the comparable the files hold, framed as a \
reply the consultant can say; never with an unrelated neighbouring number (E34: three of sixteen wrong cards).
- Answer the metric the question names. When the files hold a neighbouring figure but not the one \
named (a different period, segment, structure or total), say the named one is not in the files and name what \
is there; never substitute the neighbour as if it were the answer (measured as a top cause of wrong cards).
- When the files hold the same metric under two qualifiers (top-down and bottom-up, gross and net, \
low, middle and ceiling), the figure is the one the question fits best, the qualifier is said in \
the say-line, and the other value follows in the sentence, so the consultant is never handed one \
of two TAMs as if it were the only one.
- Cite where each number came from: spreadsheets as "<file> › <sheet> › <cell or row label>", \
documents as "<file> › page N" or "<file> › slide N". Use the file names exactly as given.
- Prefer the pre-fetched evidence; it already holds the rows the topic points at. Be fast.
- Speculative triggers: no question has been asked yet, a topic just came up. Predict the single most likely \
question about that topic given the transcript and answer it. Always send the `question` field on a speculative \
trigger, holding the predicted question; leave it out on every other trigger, where the question is already known.
- Fill `figure` with the one number the consultant should say out loud, formatted the way a human would say it.
- Fill `say` with the 4-8 words to speak first: the consultant glances at the screen for a second and a half while \
talking, so `say` is what they read; `answer` is the detail if they look longer.

Style examples (answer → figure):
- "Q3 FY25 revenue was $14.0M, up 6.6% on Q2 and 3% ahead of the $13.6M plan." → "$14.0M"
- "Headcount plan reaches 152 FTE by June, up from 119 in January; Engineering takes half the adds." → "152 FTE"
- "Enterprise churn is 8 customers of 62, 12.9%, versus 6.5% mid-market." → "12.9%"
- "Not in files: no cost per lead figure in the deck, model or memo." → null
- "Fed funds target range is 4.25-4.50% as of 18 Sep 2026 (federalreserve.gov)." → "4.25-4.50%"  (answered_from web)
- "At your 1%, year 5 subscriber revenue is $9.59M, $1.44M above the 0.85% in the model." → "$9.59M" \
(answered_from derived, `whatif` on Subscriber Fee Model!B48, working "Subscriber Fee Model B48 0.85% → 1%, model recalculated")
- "1% on the $2.60M of subscriber volume is $26,050 a year, against $22,142 at 0.85%." → "$26,050" \
(answered_from derived, `calc`, working "$2,604,971 × 1% = $26,050")

Glossary the transcript may use (map spoken forms to file labels): revenue = top line = sales = bookings if the file \
says so; COGS = cost of goods sold = cost of sales; gross margin = GM = gross profit; opex = operating expenses = \
S&M + R&D + G&A; EBITDA = operating profit before D&A; ARR = annual recurring revenue = run-rate; NRR = net revenue \
retention; churn = logo churn or revenue churn, say which; headcount = HC = FTE = heads; plan = budget = target = \
forecast; FY = fiscal year; Q1..Q4 = quarters; H1/H2 = halves; YoY = year over year; QoQ = quarter over quarter; \
bps = basis points; CAC = customer acquisition cost; LTV = lifetime value; TAM/SAM/SOM = market sizing tiers; \
capex = capital expenditure; D&A = depreciation and amortisation; WC = working capital; FCF = free cash flow.
Spoken numbers: "fourteen million" = 14,000 in a $ thousands sheet; check the sheet's unit line before quoting.

"""


# ----------------------------------------------------------------------------- brain

class Brain:
    def __init__(self, index: Any, emit: Emit, settings: dict | None = None, llm: LLM | None = None):
        self.index = index
        self.emit = emit
        self.settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
        if settings:
            self.settings.update(settings)
        self.llm: LLM = llm or make_llm(PROVIDER, models=ENV_MODELS)
        # Billing-aware routing: speculative drafts (the bulk) run on self.llm; the one latency-critical
        # live answer per question can run on a second provider (the API key is ~2x faster than the
        # Claude Code subscription route under load). live_provider: "same" | "anthropic" | "claude_code" | "gemini".
        self.llm_live: LLM | None = None
        self._anthropic_key: str | None = os.environ.get("ANTHROPIC_API_KEY") or None   # snapshot before any spawn hides it
        self.set_live_provider(os.environ.get("PRESAGE_LIVE_PROVIDER") or (settings or {}).get("live_choice") or "same")
        self.decider = None
        self.decider_choice = (settings or {}).get("decider") or os.environ.get("PRESAGE_DECIDER") or "auto"
        # the two typed-intent questions are their own choice: the local E21 checkpoint beats Jev
        # on both (E21a), while the gate's checkpoint answers neither. "auto" is local when the
        # weights are here, else Jev with a key, else off.
        self.intent_choice = str((settings or {}).get("intent") or _saved_pref("intent") or "auto").lower()
        self.slot_decider = None
        self.set_decider(self.decider_choice)
        self._utt_answers: dict[tuple[str, int], str] = {}   # (source, utt) -> answer id, for retracted finals
        self.transcript: list[Utterance] = []
        self.candidates: dict[str, Candidate] = {}      # key: normalized topic
        self.answers: dict[str, Answer] = {}
        self._last_prefetch: dict[str, float] = {}
        self._sem = asyncio.Semaphore(16)
        self._system_text: str | None = None
        self._system_version: Any = None
        self.stats = {"prefetches": 0, "live": 0, "cache_hits": 0, "classified": 0, "errors": 0,
                      "cache_read_tokens": 0, "cache_write_tokens": 0, "late_held": 0,
                      "corrections": 0, "drift": 0,
                      # E52: drafts that gave their session slot back, here and at the pool, and
                      # eager finals the settled one moved under
                      "drafts_cancelled": 0, "drafts_dropped": 0, "final_reconciled": 0,
                      # E68: questions registered, by who asked and what the brain does about
                      # them; the no-answer cards and why; vetoes the tier-2 gate overturned;
                      # questions skipped as a repeat of one a moment earlier
                      "questions": {"Them": 0, "Me": 0}, "me_routes": {"answer": 0, "check": 0, "heard": 0},
                      "no_answer": 0, "no_answer_reasons": {}, "second_opinion": 0, "duplicates": 0}
        self._asked: dict[str, Asked] = {}          # E68: card id -> the question it answers
        self._last_client_line = ""   # the newest thing Them said, "" while a line has no settled word
        self.notes: dict | None = None
        self._notes_seen = 0          # transcript length when notes were last produced
        self._notes_at = 0.0
        self._notes_busy = False
        # scratch paper: the last 5 what-ifs, so "and at 1.25%?" resolves and the page can show them
        self.scratch: list[dict] = []
        self._last_whatif: dict | None = None   # the most recent whatif tool result
        # the typed ask (E38): the workbook's own vocabulary, cached per index build, and the last
        # ask for 90 s so "and in the first quarter?" inherits the four slots it did not restate
        self._vocab: Any = None
        self._vocab_version: Any = None
        self._vocab_busy = False
        self._last_ask: dict | None = None
        # E43a: what the presenter has on screen, as a callable the server sets
        # (`brain.screen_state = screenmod.snapshot`). A callable, not an import: the brain never
        # owns the poller thread and keeps working when there is none.
        self.screen_state: Any = None
        # the spoken-figure guard (E9c): built off the transcript thread, one correction per cell
        # per minute, and a fall back to a worker thread if a folder ever makes the check slow
        self._spoken_busy = False
        self._spoken_slow = False
        self._spoken_none: Any = None     # an index build the guard could not be made for
        self._spoken_at: dict[str, float] = {}
        # the slide-drift check (E63): the pages already judged this session, and when each
        # stale cell last raised a card
        self._drift_seen: set[tuple[str, str]] = set()
        self._drift_at: dict[str, float] = {}
        self._drift_page: tuple[str, str] | None = None   # the page the last reading was on

    # ------------------------------------------------------------------ public API

    async def on_transcript(self, u: Utterance) -> None:
        text = u.text.strip()
        if u.source == "system":
            self._last_client_line = text     # "" when the line has opened with nothing settled (E40)
        if not text:
            return
        if u.final:
            self.transcript.append(u)
            cutoff = time.time() - 600
            self.transcript = [t for t in self.transcript if t.ts >= cutoff]
        mode = self.mode
        hits = self._topic_hits(text)
        if hits and mode.prefetch:
            self._maybe_prefetch(hits[:4], u)
        if not u.final and u.source == "system" and mode.auto:
            low = text.lower()
            # a what-if cue with a number in it is specific enough to draft on its own, in any mode
            # that answers: the recalculation is the slow part and it is already half-spoken
            spec = whatif_ish(text) and bool(_HAS_NUMBER.search(text))
            if spec or (len(text.split()) >= 5 and
                        (u.question_forming or questionish(text) or
                         (mode.cues and any(p in low for p in mode.cues)))):
                self._prefetch_question(u)
        if u.final:
            await self._on_final(u, hits)

    async def ask(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        hits = self._topic_hits(text)
        warm = self._best_warm(text, hits)
        await self._answer_question(text, hits, warm, source="manual")

    @property
    def mode(self) -> Mode:
        return MODES.get(str(self.settings.get("mode", "call")), MODES["call"])

    @staticmethod
    def modes() -> list[dict]:
        return [{"id": m.id, "label": m.label, "hint": m.hint} for m in MODES.values()]

    def set_setting(self, key: str, value: Any) -> None:
        if key == "mode":
            new = str(value).lower()
            if new not in MODES or new == self.settings.get("mode"):
                return
            self.settings["mode"] = new
            self._system_text = None          # the mode section is part of the cached prompt
            log.info("mode -> %s", new)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                asyncio.create_task(self.prewarm())
        elif key == "web_mode":
            self.settings[key] = "off" if str(value).lower() in ("off", "0", "false", "no") else "auto"
        elif key in self.settings:
            self.settings[key] = float(value)

    def set_provider(self, provider: str, api_key: str | None = None, models: dict | None = None) -> dict:
        """Swap providers mid-call. The env model overrides still win over the new provider's
        defaults unless `models` names them. Raises ValueError on an unknown provider; a bad or
        missing key surfaces from `check_key()`, not here."""
        self.llm = make_llm(provider, api_key=api_key, models={**ENV_MODELS, **(models or {})})
        if provider == "anthropic" and api_key:
            self._anthropic_key = api_key
        self.set_live_provider(getattr(self, "live_choice", "auto"))
        self.set_decider(getattr(self, "decider_choice", "auto"))
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            asyncio.create_task(self.prewarm())
        log.info("provider -> %s %s", self.llm.provider, self.llm.models())
        return self.llm.state()

    def set_live_provider(self, choice: str | None) -> str:
        """'auto': API key for live answers when one exists and the main provider is the subscription;
        'same': everything on the main provider; or a provider name."""
        choice = (choice or "auto").lower()
        want = None
        key = self._anthropic_key_now()
        if choice == "auto":
            if self.llm.provider == "claude_code" and key:
                want = "anthropic"
        elif choice != "same" and choice != self.llm.provider:
            want = choice
        self.live_choice = choice
        if want is None:
            self.llm_live = None
        else:
            try:
                # capture the key now: the claude_code provider hides ANTHROPIC_API_KEY from the environment
                # while it spawns, and a lazily resolving client would race with that
                self.llm_live = make_llm(want, api_key=(key if want == "anthropic" else None), models=ENV_MODELS)
            except Exception as e:  # noqa: BLE001
                log.warning("live provider %s unavailable (%s); using %s", want, e, self.llm.provider)
                self.llm_live = None
        log.info("live answers on %s", self.llm_live.provider if self.llm_live else self.llm.provider)
        return self.llm_live.provider if self.llm_live else self.llm.provider

    def set_decider(self, choice: str | None = None, intent_choice: str | None = None) -> str:
        """Pick the question gate: auto (Jev with a key, else local Laya if its weights are here,
        else Haiku), or one of them by name. Returns the kind that is now in place.

        `intent_choice` is the separate `intent` setting (`auto`, `local`, `jev`, `off`) for the
        two typed-intent questions: which lever a spoken what-if means, and whether a warm card
        answers the question that landed. They are a different decision from the gate because they
        now have a different best answer -- the E21 multi checkpoint beats Jev on both, locally
        (E21a) -- while the gate's own checkpoint answers neither.
        """
        self.decider_choice = (choice or "auto").lower()
        if intent_choice:
            self.intent_choice = str(intent_choice).lower()
        self.decider = None
        if make_decider is None:
            return "none"
        try:
            self.decider = make_decider(self.llm, prefer=self.decider_choice)
        except Exception as e:  # noqa: BLE001
            log.warning("decider %s unavailable (%s); using Haiku", self.decider_choice, e)
            try:
                self.decider = make_decider(self.llm, prefer="haiku")
            except Exception as e2:  # noqa: BLE001
                log.warning("decider unavailable: %s", e2)
        kind = str(getattr(self.decider, "kind", "") or type(self.decider).__name__.replace("Decider", "").lower())
        # A Jev client, if there is one: the gate's own when the gate is Jev, otherwise a fresh
        # one. The E38a slot request stays on it whatever the intent questions do, because the
        # multi checkpoint was not trained on the slots (E21b is the experiment that re-cuts them).
        slow = getattr(self.decider, "slow", None)          # the cascade's real decider
        self.slot_decider = (slow if slow is not None and getattr(slow, "kind", "") == "jev"
                             else (self.decider if kind == "jev" else None))
        if self.slot_decider is None and self.decider_choice != "haiku":
            try:
                from presage.llm.jev import JevDecider, jev_key
                if jev_key():
                    self.slot_decider = JevDecider()
            except Exception as e:  # noqa: BLE001
                log.debug("jev client unavailable: %s", e)
        self.intent_decider, self.intent_kind = self.slot_decider, "jev" if self.slot_decider else "off"
        try:
            from presage.llm import intent_local
            self.intent_decider = intent_local.make(self.intent_choice, jev=self.slot_decider)
            self.intent_kind = intent_local.describe(self.intent_decider)
        except Exception as e:  # noqa: BLE001  the local checkpoint is optional
            log.debug("local typed intent unavailable: %s", e)
        log.info("decider: %s (choice %s); typed intent: %s (choice %s); slots: %s",
                 kind or "none", self.decider_choice, self.intent_kind, self.intent_choice,
                 "jev" if self.slot_decider is not None else "off")
        return kind

    def _anthropic_key_now(self) -> str | None:
        """The API key, immune to the claude_code provider hiding it from the environment for a spawn:
        prefer the snapshot, otherwise re-read the environment a few times over ~150 ms."""
        for _ in range(4):
            k = os.environ.get("ANTHROPIC_API_KEY")
            if k:
                self._anthropic_key = k
                return k
            if self._anthropic_key:
                return self._anthropic_key
            time.sleep(0.05)
        return self._anthropic_key

    async def check_key(self) -> tuple[bool, str]:
        """One cheap request against the current provider: (ok, reason)."""
        try:
            return await self.llm.validate_key()
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {str(e)[:120]}"

    def _recent_requests(self) -> list:
        return self.stats.setdefault("requests", [])

    def clear(self) -> None:
        for c in self.candidates.values():
            if c.task and not c.task.done():
                c.task.cancel()
        self.candidates.clear()
        self.answers.clear()
        self._last_prefetch.clear()
        self.notes = None
        self.scratch.clear()
        self._last_whatif = None
        self._last_ask = None
        self._notes_seen = len(self.transcript)
        for q in self._asked.values():          # E68: nothing still waiting may raise a card later
            if q.watch and not q.watch.done():
                q.watch.cancel()
        self._asked.clear()
        # a cleared deck is a new session: every page may raise its drift card once more
        self._drift_seen.clear()
        self._drift_at.clear()
        self._drift_page = None

    def state(self) -> dict:
        return {
            "settings": self.settings,
            "mode": self.mode.id,
            "modes": self.modes(),
            "notes": self.notes,
            "scratch": self.scratch_items(),
            "answers": [a.msg() for a in sorted(self.answers.values(), key=lambda a: a.ts, reverse=True)[:30]],
            "candidates": [c.msg() for c in self.candidates.values() if c.status != "expired"],
            "models": self.llm.state(),   # provider + answer/hedge/classifier + that provider's defaults
            "live_provider": self.llm_live.provider if self.llm_live else self.llm.provider,
            "live_choice": getattr(self, "live_choice", "auto"),
            "screen": self._screen() or {},   # E43a: the page on screen, so `hello` shows it without polling
            # E68: the last registered questions, newest first, so a page that reconnects can
            # redraw a question that is still waiting for its card
            "questions": [q.msg() for q in sorted(self._asked.values(), key=lambda q: q.t_q, reverse=True)[:10]],
            "stats": self.stats,
        }

    async def housekeeping(self) -> None:
        """Expire stale candidates; refresh the notes card in Notes mode. Run as a background task."""
        while True:
            await asyncio.sleep(3)
            now = time.time()
            if self._vocab_version != getattr(self.index, "version", None):
                asyncio.create_task(self._refresh_vocab())     # the folder was rebuilt
            if self.mode.notes and not self._notes_busy:
                new = len(self.transcript) - self._notes_seen
                if new >= 6 or (new >= 1 and now - self._notes_at >= 45):
                    asyncio.create_task(self._refresh_notes())
            ttl = self.settings["answer_ttl_s"]
            for key, c in list(self.candidates.items()):
                if c.status != "expired" and now - c.created > ttl:
                    c.status = "expired"
                    if c.task and not c.task.done():
                        c.task.cancel()
                    await self.emit(c.msg())
                    del self.candidates[key]

    async def prewarm(self) -> None:
        """Warm everything before the first question: the provider writes the system prompt + tools
        into its prompt cache and pings every model so connections and routing are hot. Measured:
        the first call to a model after idle takes 7-19 s on this account, warm calls 1-3 s."""
        await self._refresh_vocab()
        for llm in ([self.llm] + ([self.llm_live] if self.llm_live else [])):
            try:
                await llm.prewarm(self.system_text(), TOOLS)
            except Exception as e:  # noqa: BLE001
                log.warning("prewarm (%s) failed: %s", llm.provider, e)

    async def keepalive(self, every_s: float = 240) -> None:
        """Re-run prewarm periodically so the cache and connections never go cold mid-call."""
        while True:
            await asyncio.sleep(every_s)
            await self.prewarm()

    # ------------------------------------------------------------------ prompt assembly

    def system_text(self) -> str:
        deep = getattr(self.index, "deep_stats", None)
        deep_key = (deep() or {}).get("finished") if deep else None
        version = (getattr(self.index, "version", None), self.mode.id, deep_key)
        if self._system_text is None or version != self._system_version:
            schema = self.index.schema_summary()
            summ = ""
            fn = getattr(self.index, "summaries", None)
            if fn is not None:
                try:
                    summ = fn() or ""
                except Exception as e:  # noqa: BLE001
                    log.debug("summaries unavailable: %s", e)
            extra = ("\n\nWhat each file contains (from the deep index):\n" + summ) if summ.strip() else ""
            levers = ""
            fn = getattr(self.index, "levers_block", None)
            if fn is not None:
                try:
                    levers = fn(limit=100) or ""      # ~5k tokens on a 250-lever model, cached
                except Exception as e:  # noqa: BLE001
                    log.debug("levers_block unavailable: %s", e)
            if levers.strip():
                extra += "\n\n" + levers.strip()
            self._system_text = SYSTEM_HEAD + self.mode.prompt + "Context files:\n" + schema + extra
            self._system_version = version
        return self._system_text

    async def _refresh_notes(self) -> None:
        """Rolling meeting notes from the recent transcript (Notes mode)."""
        self._notes_busy = True
        try:
            cutoff = time.time() - 300
            lines = [f"{'Me' if u.source == 'mic' else 'Them'}: {u.text}" for u in self.transcript if u.ts >= cutoff]
            if not lines:
                return
            prev = json.dumps(self.notes, ensure_ascii=False) if self.notes else "none yet"
            prompt = ("You keep running notes for a strategy consultant (Me) on a call with a client (Them). "
                      "Update the notes from the transcript below: keep what is still true from the previous notes, "
                      "add what is new, drop what was resolved. Be concrete: names, numbers, decisions. "
                      "No filler, no interpretation.\n\n"
                      f"Previous notes: {prev}\n\nTranscript (last 5 minutes, oldest first):\n" + "\n".join(lines[-40:]))
            data = await self.classify(prompt, NOTES_SCHEMA, model=self.llm.answer_model)
            if not data:
                return
            self.notes = {"type": "notes", "summary": list(data.get("summary") or [])[:5],
                          "actions": list(data.get("actions") or [])[:5],
                          "questions": list(data.get("questions") or [])[:3],
                          "ts": time.time(), "utterances": len(self.transcript)}
            self._notes_seen = len(self.transcript)
            self._notes_at = time.time()
            await self.emit(self.notes)
        except Exception as e:  # noqa: BLE001
            log.warning("notes failed: %s", e)
        finally:
            self._notes_busy = False

    def _context_lines(self) -> str:
        cutoff = time.time() - self.settings["transcript_window_s"]
        lines = [f"{'Me' if u.source == 'mic' else 'Them'}: {u.text}" for u in self.transcript if u.ts >= cutoff]
        return "\n".join(lines[-14:]) or "(nothing yet)"

    def _user_prompt(self, trigger: str, evidence: str, current: str) -> str:
        scratch = ""
        if self.scratch:
            lines = [f"- {s.get('label')} → {s.get('value')}: {s.get('output')}" +
                     (f" ({s['working']})" if s.get("working") else "") +
                     ("; also " + ", ".join(f"{o.get('label')} {o.get('now')}" for o in s["outputs"])
                      if s.get("outputs") else "")
                     for s in self.scratch[-5:]]
            scratch = ("Scratch so far (what-ifs already run this call; a follow-up like \"and at 1.25%?\" "
                       "moves the same lever):\n" + "\n".join(lines) + "\n\n")
        # E43a: what "this" refers to. The system prompt already asks for the file and the slide in
        # the citation; this says which slide is in the room.
        on_screen = ""
        s = self._screen()
        if s:
            on_screen = f"On screen: {s['file']}" + (f" › {s['loc']}" if s.get("loc") else "")
            if s.get("sheet"):
                on_screen += f" › sheet {s['sheet']}"
            if s.get("title"):
                on_screen += f" ({s['title']})"
            on_screen += ". A question that says \"this\" or \"that\" probably means something on it.\n\n"
        return (
            f"Recent transcript (oldest first):\n{self._context_lines()}\n\n"
            f"Current utterance: {current}\n\n"
            f"Trigger: {trigger}\n\n"
            f"{on_screen}"
            f"{scratch}"
            f"Pre-fetched evidence:\n{evidence or '(none)'}\n\n"
            "Respond with the JSON object only."
        )

    # ------------------------------------------------------------------ scratch paper

    def scratch_items(self) -> list[dict]:
        """The scratch rows the page shows: label, value, output, working, ts."""
        keys = ("label", "value", "output", "working", "ts")
        return [{k: s.get(k) for k in keys} for s in self.scratch]

    async def _note_scratch(self, a: Answer) -> None:
        """Record a derived answer on the scratch pad and push the list to the page. A what-if row
        is the lever, the value it was set to and the headline output; a calc row is the question
        and its figure. A newer result for the same card (the live answer after the instant one, or
        the primary model after the hedge) replaces the row instead of adding one."""
        working = a.working or ""
        m = _WORK.match(working)
        if m and "recalculated" in working.lower():
            cell = m.group("cell").strip()
            label = self._lever_label(cell) or cell
            value = m.group("now").strip()
        else:
            q = re.sub(r"^\s*(and|so|then)\s+", "", (a.question or "").strip(), flags=re.I)
            label = (q[:46] + "…") if len(q) > 47 else (q or "calc")
            value = ""
        out = a.figure or (a.say or a.answer or "")
        w = self._last_whatif or {}
        outputs = []
        if m and w.get("changes") and _WORK.match(working) and \
                re.sub(r"[^a-z0-9]+", " ", str(w["changes"][0].get("cell", "")).lower()).strip() == \
                re.sub(r"[^a-z0-9]+", " ", m.group("cell").lower()).strip():
            outputs = [{"label": o.get("label"), "now": o.get("now_s", o.get("now"))}
                       for o in (w.get("outputs") or [])[:6]]
        item = {"aid": a.id, "ts": a.ts, "working": working, "outputs": outputs,
                "label": str(label)[:64], "value": str(value)[:24], "output": str(out)[:72]}
        replaced = False
        for i, row in enumerate(self.scratch):
            if row.get("aid") == a.id or (value and row.get("label") == item["label"] and row.get("value") == value):
                self.scratch[i] = item
                replaced = True
                break
        if not replaced:
            self.scratch.append(item)
        del self.scratch[:-5]
        log.info("scratch %s: %s → %s: %s (%d rows)", "replace" if replaced else "add", item["label"][:40],
                 item["value"], item["output"][:30], len(self.scratch))
        await self.emit({"type": "scratch", "items": self.scratch_items()})

    def _lever_label(self, cell: str) -> str:
        """"Subscriber Fee Model!B48" or "Subscriber Fee Model B48" -> the lever's row label."""
        fn = getattr(self.index, "levers", None)
        if fn is None:
            return ""
        want = re.sub(r"[^a-z0-9]+", " ", cell.lower()).strip()
        try:
            for lv in fn() or []:
                key = re.sub(r"[^a-z0-9]+", " ", f"{lv.get('sheet')} {lv.get('cell')}".lower()).strip()
                if key == want:
                    return str(lv.get("label") or "")
        except Exception:  # noqa: BLE001
            return ""
        return ""

    async def _emit_answer(self, a: Answer, strict: bool = False) -> bool:
        """Every card passes here, live and prefetched. Returns whether it reached the page.

        E68: `strict` drops a live card that fails too, which is what a consultant's `check` wants
        (a flagged figure is no use for checking the client's reply against), and a card that goes
        out marks its registered question answered (`_shown`). The verifier (E23: 36/40 corrupted cards
        caught, 1 false positive in 733, 0.14 ms) checks each number against the cells and pages
        the card cites; a speculative or cached card that fails is dropped, a live one is shown
        flagged so the consultant does not read the figure out.

        A spoken correction (E9c) skips it and keeps the guard's own verdict: its figure was read
        out of the cell it cites a moment ago, and the number the verifier would argue with is the
        one the consultant said, which is the point of the card. A drift card (E63) skips it for
        the same reason in the other direction: it prints the slide's number precisely because no
        cell holds it any more, so the verifier would reject the one card whose whole content is
        that disagreement."""
        if verify is not None and a.answer and a.source != "manual" \
                and a.answered_from not in ("correction", "drift"):
            try:
                v = verify(a, self.index, whatif=getattr(self, "_last_whatif", None))
                a.verdict = v.as_dict() if hasattr(v, "as_dict") else asdict(v)
                if not v.ok:
                    self.stats["rejected"] = self.stats.get("rejected", 0) + 1
                    log.warning("card %s failed verification: %s", a.id, v.reason or v.problems[:1])
                    if a.source == "prefetch" or a.cached or strict:
                        return False                 # nobody asked; a contradicted figure is worth nothing
            except Exception as e:  # noqa: BLE001
                log.debug("verify failed: %s", e)
        await self.emit(a.msg())
        await self._shown(a.id)
        if a.derived:
            await self._note_scratch(a)
        return True

    # ------------------------------------------------------------------ topics + evidence

    def _topic_hits(self, text: str) -> list:
        try:
            hits = self.index.match_topics(text)
        except Exception as e:  # noqa: BLE001
            log.warning("match_topics failed: %s", e)
            hits = []
        thr = self.settings["topic_threshold"]
        hits = [h for h in hits if getattr(h, "score", 0) >= thr]
        return hits or self._lever_hits(text)

    def _lever_matches(self, text: str, k: int = 5) -> list[tuple[float, dict]]:
        """Levers the utterance names, by label or spoken alias, matched on words weighted by how
        rare they are across the lever table ("gmv" counts, "fee" barely), so "1.25% of GMV" finds
        the alias "GMV fee" and "subscriber churn" finds "Annual subscriber churn". A scenario word in the
        utterance ("base case") picks that column; a hardcoded zero ranks last."""
        fn = getattr(self.index, "levers", None)
        if fn is None:
            return []
        try:
            levers = fn() or []
        except Exception as e:  # noqa: BLE001
            log.debug("levers unavailable: %s", e)
            return []
        toks = _word_set(text)
        if not toks or not levers:
            return []
        df: dict[str, int] = {}
        names_of: list[list[set[str]]] = []
        for lv in levers:
            names = [_word_set(str(n)) for n in [lv.get("label") or ""] + list(lv.get("aliases") or [])]
            names = [n for n in names if n]
            names_of.append(names)
            for w in set().union(*names) if names else set():
                df[w] = df.get(w, 0) + 1
        weight = {w: 1.0 / math.log(2.0 + c) for w, c in df.items()}
        scored: list[tuple[float, int, dict]] = []
        for lv, names in zip(levers, names_of):
            best, rare = 0.0, False
            # E36a: names that come from the sheet's section title describe the block, not the
            # row, so they count at half weight and only when the row's own names found nothing
            sec = _word_set(str(lv.get("section") or ""))
            row_names = [nt for nt in names if not (sec and nt <= sec)]
            block_names = [nt for nt in names if sec and nt <= sec]
            for group, scale in ((row_names, 1.0), (block_names, 0.5)):
                if group is block_names and best > 0:
                    break
                for nt in group:
                    total = sum(weight[w] for w in nt)
                    if total <= 0:
                        continue
                    matched = nt & toks
                    frac = scale * sum(weight[w] for w in matched) / total
                    if frac > best:
                        best, rare = frac, any(df.get(w, 0) <= 3 for w in matched)
            # a rare word alone ("GMV") is enough; common words need most of the name
            if best < 0.5 and not (rare and best >= 0.35):
                continue
            head = _word_set(str(lv.get("header") or ""))
            if head and head <= toks:
                best += 0.2                      # "base case" names the Base column
            if not lv.get("value"):
                best -= 0.3                      # a hardcoded 0 is a placeholder, not a lever
            if lv.get("active") is False:
                best -= 0.5                      # an input the summary does not read right now (the
                                                 # unselected price point, an unused scenario column)
            scored.append((best, len(lv.get("outputs") or []), lv))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [(round(sc, 2), lv) for sc, _, lv in scored[:k]]

    def _lever_hits(self, text: str) -> list:
        out = [lever_hit(lv, score=sc) for sc, lv in self._lever_matches(text, k=3)]
        if out:
            log.info("lever hits: %s", [f"{h.topic} ({h.score})" for h in out])
        return out

    def _whatif_prior(self) -> dict | None:
        """The last what-if as `{"question", "lever"}` for follow-up resolution, within 90 s."""
        w = getattr(self, "_last_whatif", None) or {}
        if not w or time.time() - float(w.get("ts") or 0) > 90:
            return None
        lever = w.get("lever")
        if lever is None:
            cell = next(iter(w.get("sets") or {}), None)
            fn = getattr(self.index, "levers", None)
            if cell and fn is not None:
                want = re.sub(r"[^a-z0-9]+", " ", str(cell).lower()).strip()
                for lv in fn() or []:
                    if re.sub(r"[^a-z0-9]+", " ", f"{lv.get('sheet')}!{lv.get('cell')}".lower()).strip() == want:
                        lever = lv
                        break
        if lever is None:
            return None
        return {"question": str(w.get("question") or ""), "lever": lever, "sets": dict(w.get("sets") or {}),
                "kind": w.get("kind")}

    # ------------------------------------------------------------------ the typed ask (E38)

    async def _refresh_vocab(self) -> None:
        """What the folder lets a question name, off the transcript thread and once per index
        build (1 ms on sample_context, 13 ms on the big test model). Called at open, from prewarm and
        whenever housekeeping sees `index.version` move; never from `on_transcript`.

        The spoken guard (E9c) is built in the same thread hop: it is that vocabulary plus the
        rarity tables, and the Me line it runs on must never wait for either."""
        version = getattr(self.index, "version", None)
        if askmod is None or self._vocab_busy or (self._vocab is not None and self._vocab_version == version):
            return
        self._vocab_busy = True
        t0 = time.time()

        def build() -> Any:
            v = askmod.build_vocab(self.index)
            got = spoken.guard_for(self.index, v) if spoken is not None else None
            return v, got is not None                   # the guard caches itself on the index

        try:
            v, guarded = await asyncio.to_thread(build)
            self._vocab, self._vocab_version = (v if v.metrics or v.periods else None), version
            log.info("ask vocabulary v%s: %s in %d ms%s", version, v.sizes(),
                     int((time.time() - t0) * 1000), "; spoken guard up" if guarded else "")
        except Exception as e:  # noqa: BLE001
            self._vocab, self._vocab_version = None, version
            log.info("ask vocabulary unavailable: %s", e)
        finally:
            self._vocab_busy = False

    def _ask_prior(self) -> dict | None:
        """The last typed ask as `{"line", "slots", "p"}`, within 90 s -- the same window
        `_whatif_prior` keeps the last what-if in. This is the whole of E38's follow-up gain:
        cell reach on a follow-up 58.9% without it, 90.0% with it, at 0 ms."""
        w = getattr(self, "_last_ask", None) or {}
        if not w or time.time() - float(w.get("ts") or 0) > 90:
            return None
        slots = {k: v for k, v in (w.get("slots") or {}).items() if v}
        return {"line": str(w.get("line") or ""), "slots": slots, "p": dict(w.get("p") or {})} if slots else None

    def _keep_ask(self, ask: Any) -> None:
        """Hold an ask that resolved something; a line that named nothing leaves the prior standing."""
        if ask is not None and ask.filled():
            self._last_ask = {"line": ask.line, "slots": ask.slots(), "p": ask.probs(), "ts": time.time()}

    def _ask_heads(self):
        """The two local heads for `basis` and `compare_to`, loaded once: 91.7% and 95.8% held out
        against Jev's 75.0% and 90.4% (E38a, `data/gate/ask/train_heads.py`), for 0.04 ms including
        the encode. False once, when they are not built."""
        h = self.__dict__.get("_askheads")
        if h is None:
            try:
                from presage.llm.pregate import AskHeads
                h = AskHeads.load() if AskHeads.available() else False
            except Exception as e:  # noqa: BLE001
                log.info("ask heads unavailable: %s", e)
                h = False
            self._askheads = h
        return h or None

    def _head_slots(self, text: str) -> dict:
        """The closed slots the heads are confident about. Those two then come off the request."""
        heads = self._ask_heads()
        if heads is None or askmod is None:
            return {}
        try:
            v = self._embed([text])
            got = heads.slots(v[0]) if v is not None else {}
        except Exception as e:  # noqa: BLE001
            log.debug("ask heads failed: %s", e)
            return {}
        return {name: askmod.Slot(value, p, "head") for name, (value, p) in got.items()}

    def _ask_decider(self) -> tuple[Any, bool]:
        """(the client the slot questions go to, does it also gate). Jev gating -> the slots ride
        in the batch it already sends, for nothing (E38a). A local gate (E17) cannot take a choice
        over the workbook, so they go to the Jev client beside it. Neither -> None.

        `slot_decider`, not `intent_decider`: since E21a the lever pick and the warm-card check run
        on a local checkpoint that was not trained on the five slots, so the slots keep their own
        handle on the hosted model until E21b re-cuts them."""
        gate = getattr(self.decider, "slow", None) or self.decider
        if str(getattr(gate, "provider", "") or getattr(gate, "kind", "")) == "jev":
            return gate, True
        return getattr(self, "slot_decider", None), False

    def _ask_questions(self, text: str, prior: dict | None) -> tuple[list, dict]:
        """The slot questions for one line, minus the ones a head already answered."""
        v = self._vocab
        if askmod is None or v is None or not (v.metrics or v.periods):
            return [], {}
        heads = self._head_slots(text)
        slots = tuple(s for s in askmod.Ask.SLOTS if s not in heads)
        return askmod.ask_questions(v, text, prior, self._embed, slots=slots), heads

    def _ask_state(self, text: str, prior: dict | None, gate_state: str = "") -> str:
        """What the slots are answered against: the gate's own state when they ride in its batch
        (it carries the mode and the speaker), `ask.state_for` when they go on their own."""
        if not gate_state:
            return askmod.state_for(text, prior)
        return "\n\n".join(p for p in (gate_state, askmod.prior_state(prior), f'THE LINE: "{text}"') if p)

    # How long the local gate will wait on a slot request running beside it before it goes without
    # one. Jev's own timeout is 2 s, which is fine for a request the gate IS, and far too long for
    # one it is only travelling with: the local gate answers in 76 ms (E38a) and a card on the gate
    # branch waits for whatever this returns. 0.5 s is above the slot request's measured p90 (339 ms
    # here, 387 ms in E38), so nothing that was going to arrive is cut off.
    ASK_WAIT_S = 0.5

    async def _slot_request(self, dec: Any, text: str, prior: dict | None, qs: list, heads: dict,
                            gate_state: str = "", wait: float | None = None):
        """The five slots on Jev. Never raises: a slot request that fails leaves the pre-pack
        exactly as it is today, which is also what E38 says to do when the metric comes back NONE."""
        try:
            call = dec.decide(self._ask_state(text, prior, gate_state), qs)
            d = await (asyncio.wait_for(call, wait) if wait else call)
            return askmod.read_ask(d, text, self._vocab, prior, extra=heads)
        except Exception as e:  # noqa: BLE001
            log.info("typed ask failed: %s", e)
            return None

    async def _extract_ask(self, text: str):
        """One fresh ask for a typed question: the manual path has no gate verdict to ride in and
        is not latency-critical (E38 wiring step 4)."""
        dec, _ = self._ask_decider()
        prior = self._ask_prior()
        qs, heads = self._ask_questions(text, prior)
        if dec is None or not qs:
            return None
        ask = await self._slot_request(dec, text, prior, qs, heads)
        self._keep_ask(ask)
        return ask

    async def _typed_lever(self, text: str, prior: dict | None):
        """One typed choice over the lever table, shortlisted by the decider that is answering it:
        50 levers on Jev, 20 on the local E21 checkpoint, never the whole table on a local one
        (at 251 options it answers the same option every time, E7 and E21). None when
        unavailable."""
        dec = getattr(self, "intent_decider", None)
        if dec is None or intent is None:
            return None
        fn = getattr(self.index, "levers", None)
        if fn is None:
            return None
        try:
            levers = fn() or []
            if not levers:
                return None
            pick = await intent.pick_lever(dec, text, levers, prior=prior)
            log.info("typed lever pick: %s p=%.2f follow=%s %dms", pick.cell, pick.p, pick.follow, pick.latency_ms)
            return pick
        except Exception as e:  # noqa: BLE001
            log.info("typed lever pick failed: %s", e)
            return None

    async def _typed_ask(self, text: str, levers: list[dict], prior: dict | None):
        """When the parser finds no lever, the typed choice over the whole table (95.7% on E7's
        300 lines, 250 ms) picks it from spoken aliases the labels do not carry ("1% of GMV"),
        for a plain what-if with the value the line says; a follow-up rides on the prior lever."""
        follow_up = bool(prior) and _anaphoric(text)
        if not whatif_ish(text) and not follow_up:
            return None
        asked = _asked_values(text)
        if not asked:
            return None
        pick = await self._typed_lever(text, prior)
        lv = None
        if pick is not None:
            if pick.follow == "not-whatif" and pick.p >= 0.5:
                return None
            if pick.follow == "same" and prior:
                lv = prior.get("lever")
            elif pick.lever is not None and pick.p >= 0.5:
                lv = pick.lever
        elif follow_up and prior:
            lv = prior.get("lever")
        if lv is None:
            return None
        fmt, cur = lv.get("fmt") or "number", lv.get("value")
        for v, is_pct in asked:
            if fmt == "pct":
                if not is_pct and v > 1:
                    v = v / 100.0
                if v > 1:
                    continue
            elif is_pct:
                continue
            if cur is not None and _same_value(v, cur):
                continue
            cell = f"{lv.get('sheet')}!{lv.get('cell')}"
            return prerun.Ask(sets={cell: v}, kind="whatif", confidence=float(pick.p if pick else 0.5),
                              why="typed lever pick", levers=[lv])
        return None

    async def _prerun_whatif(self, text: str, hits: list) -> tuple[str, dict | None]:
        """A what-if, goal seek, comparison, sweep or explanation the client just asked is run
        here, before any model call (E32: 67% of 156 labelled asks answered with no model, parse
        2 ms, recalculation 9 ms p50), and handed over as evidence so the answer needs no tool
        round. Returns (evidence block, record) or ("", None)."""
        if prerun is None:
            return "", None
        fn = getattr(self.index, "levers", None)
        try:
            levers = (fn() or []) if fn else []
        except Exception as e:  # noqa: BLE001
            log.debug("levers unavailable: %s", e)
            return "", None
        if not levers:
            return "", None
        prior = self._whatif_prior()
        ask = prerun.parse_ask(text, levers, prior=prior)
        if not ask.kind and prerun.lookup_cue(text) and not prerun.change_cue(text):
            return "", None                        # a lookup, never a what-if (E70)
        if not ask.kind or ask.confidence < 0.4:
            ask = await self._typed_ask(text, levers, prior)     # E7's Jev pick: aliases the parser lacks
            if ask is None:
                return "", None
        try:
            out = await asyncio.to_thread(prerun.run_ask, self.index, ask, None)
        except Exception as e:  # noqa: BLE001
            log.info("prerun %s failed: %s", ask.kind, e)
            return "", None
        block = prerun.evidence_block(ask, out or {})
        if not block:
            return "", None
        self._last_whatif = {"file": (out or {}).get("file"), "sets": dict(ask.sets), "kind": ask.kind,
                             "lever": (ask.levers or [None])[0], "question": text, "ask": ask,
                             "outputs": (out or {}).get("outputs") or [],
                             "changes": (out or {}).get("changes") or [],
                             "partial": list((out or {}).get("partial") or []), "ts": time.time()}
        log.info("prerun %s conf=%.2f sets=%s in %s ms", ask.kind, ask.confidence, ask.sets, (out or {}).get("ms"))
        return block, self._last_whatif

    # How much of today's pre-pack still follows a typed ask: (topic hits, targets each). Measured
    # in E38a through this method on 240 labelled lines -- (0,0) reaches the gold cell 82.9% of the
    # time and hands the model 12 numeric cells at the median, (1,1) 83.8% with 13, (2,1) 85.0%
    # with 21 -- against 66.7% and 38 cells for the pre-pack alone. The tail is not zero because it
    # is what catches a line whose metric came back empty; it is not two because the count of
    # neighbouring figures is the mechanism behind E34's wrong cards, and the last 1.2 points of
    # reach cost 8 more of them.
    ASK_TAIL = (1, 1)

    def _evidence(self, hits: list, text: str, ask: Any = None) -> str:
        """Deterministic pre-pack, runs before any LLM call: the spreadsheet rows/columns the topic
        hits point at (with cell refs) plus the best document chunks. Cheap (sub-millisecond).

        With a typed ask (E38) the row and column the line named are packed FIRST and the topic
        hits become a short tail. Topic matching is good at putting the right row somewhere in the
        block and bad at putting only the right row there: on 240 labelled lines it reaches the
        gold cell 66.7% of the time and hands the model 38 numeric cells at the median, against
        82.9% and 11 from the ask. An ask that resolved nothing changes nothing here -- the topic
        path runs at full width, which is what the gate saying `needs_lookup` on an empty metric
        slot has to fall through to.

        With a page on screen (E43a) that page's own chunk and the model cells behind its figures
        are packed BEFORE the ask targets: the presenter is looking at slide 4, so the question is
        about slide 4, and its eight linked cells are the answer set. The screen is a prior of the
        same kind as the ask, so it buys the same narrowing of the topic tail. A deictic line
        ("what's this based on") also drops the content-word LIKE, which on a line with no content
        words of its own matches half the workbook.

        The speculative path passes no ask: `_run_candidate` fires mid-sentence, before any gate
        verdict exists. It does see the screen, which costs it nothing: the block is a dict read
        and one SELECT, and the poller never fires a prefetch of its own.
        """
        parts: list[str] = []
        seen: set[str] = set()
        seen_chunks: set[int] = set()

        screen = self._screen()
        on_screen = ""
        if screen is not None:
            try:
                on_screen = self._screen_block(screen, seen_chunks)
            except Exception as e:  # noqa: BLE001
                log.debug("screen block failed: %s", e)
            if on_screen:
                parts.append(on_screen)
                log.info("screen evidence: %s › %s (%d chars)", screen.get("file"), screen.get("loc"),
                         len(on_screen))
        # a pointing word with a page behind it: the page IS the metric, so the tail is one hit
        # and the LIKE is skipped whether or not the ask typed
        point = bool(on_screen) and deictic(text, ask)

        def pack(tgt: dict) -> bool:
            key = json.dumps(tgt, sort_keys=True, default=str)
            if key in seen:
                return False
            seen.add(key)
            try:
                block = self._evidence_for_target(tgt, seen_chunks)
            except Exception as e:  # noqa: BLE001
                log.debug("evidence for %s failed: %s", tgt, e)
                return False
            if block:
                parts.append(block)
            return bool(block)

        typed = 0
        if ask is not None and askmod is not None and ask.filled():
            try:
                targets = askmod.evidence_targets(ask, self.index)
            except Exception as e:  # noqa: BLE001
                log.debug("ask targets failed: %s", e)
                targets = []
            typed = sum(1 for tgt in targets if pack(tgt))
            log.info("typed ask evidence: %s -> %d blocks", {k: v for k, v in ask.slots().items() if v}, typed)
        n_hits, n_tgts = (1, 1) if point else (self.ASK_TAIL if (typed or on_screen) else (4, 2))
        for h in hits[:n_hits]:
            for tgt in (h.targets or [])[:n_tgts]:
                pack(tgt)
        # direct cell lookup on the utterance's content words: robust when a word like "revenue"
        # names dozens of rows and the topic ranking cannot pick one. A typed ask has already
        # named the row, so the LIKE is the noise it exists to remove, and so has a page on screen
        # when the line points at it rather than naming anything.
        try:
            from presage.ingest.topics import normalize
            words = ([] if (typed or point) else
                     [w for w in dict.fromkeys(normalize(text)) if len(w) >= 4 and not w.isdigit()][:4])
        except Exception:  # noqa: BLE001
            words = []
        cell_lines: list[str] = []
        for w in words:
            like = "%" + w.replace("%", "").replace("'", "") + "%"
            rows = self.index.sql(
                "SELECT file, sheet, cell_ref, header, row_label, COALESCE(value_num, value_text) AS v, is_formula "
                f"FROM cells WHERE value_num IS NOT NULL AND (lower(row_label) LIKE '{like}' OR lower(header) LIKE '{like}') "
                "ORDER BY file, sheet, row, col", limit=12)
            rows = [r for r in rows if "error" not in r]
            if rows:
                cell_lines.append(f"# cells matching '{w}' (file › sheet › cell | header | row label | value | input or result)")
                cell_lines += [f"{r['file']} › {r['sheet']} › {r['cell_ref']} | {r['header']} | {r['row_label']} | {r['v']} | "
                               f"{'result' if r.get('is_formula') else 'input'}" for r in rows]
        if cell_lines:
            parts.append("\n".join(cell_lines)[:3000])
        try:
            chunks = self.index.search_text(text, k=3)
        except Exception as e:  # noqa: BLE001
            chunks = []
            log.warning("search_text failed: %s", e)
        for ch in chunks:
            if ch.id in seen_chunks:
                continue
            seen_chunks.add(ch.id)
            parts.append(f"# {ch.file} › {ch.loc}\n{re.sub(r'\s+', ' ', ch.text)[:600]}")
        return "\n".join(parts)[:8000]

    def _columns(self, table: str) -> list[dict]:
        cache = self.__dict__.setdefault("_col_cache", {})
        if table not in cache:
            rows = self.index.sql(f"SELECT name, label, letter FROM _columns WHERE table_name = '{table}' ORDER BY ord", limit=200)
            cache[table] = [r for r in rows if "error" not in r]
        return cache[table]

    def _links_rows(self, file: str, loc: str, exact: bool = False, limit: int = 8) -> list[dict]:
        """`Index.links_for`, with the prefix trap closed.

        `links_for` runs `loc = ? OR loc LIKE ? || '%'`, so the head `"slide 4"` also matches
        `slide 40` through `slide 49` on any deck with forty slides or more, and the screen's own
        `loc` IS that head. `exact=True` keeps only the rows whose `loc` is the string asked for,
        which is what the screen path passes after `_chunks_at` has turned the head into the
        chunk's own citation (`slide 4 'FY26 plan by region'`).
        """
        fn = getattr(self.index, "links_for", None)
        if fn is None:
            return []
        try:
            links = fn(file, loc, limit=limit if not exact else max(limit, 24)) or []
        except Exception as e:  # noqa: BLE001
            log.debug("links_for failed: %s", e)
            return []
        out = [lk for lk in links
               if isinstance(lk, dict) and not lk.get("error") and lk.get("cell_ref")]
        if exact:
            out = [lk for lk in out if str(lk.get("loc") or "") == loc]
        return out[:limit]

    def _links_lines(self, file: str, loc: str, exact: bool = False) -> list[str]:
        """The model cells behind the figures on a slide or page (E16: 42-69% of figures resolve
        at 96-100% precision), so a card can cite the cell instead of only the slide. A figure
        whose cell moved loses its link at the next rebuild, so a missing link on a stale slide
        is the drift signal."""
        where = "slide" if "slide" in str(loc).lower() else "page"
        return [f"  {lk.get('value_text')} on this {where} comes from {lk.get('cell_file') or ''} › "
                f"{lk.get('cell_sheet')}!{lk.get('cell_ref')} ({lk.get('label') or ''})"
                for lk in self._links_rows(file, loc, exact=exact)]

    # ------------------------------------------------------------------ the screen prior (E43a)

    # How much of the 8,000-character evidence budget the page on screen may take. The ask targets
    # and the ASK_TAIL run to about 2,000 characters on the two bench folders, so this leaves both
    # of them room even when the page is a dense one.
    SCREEN_CAP = 2400
    SCREEN_CHUNKS = 2          # a slide is one chunk; a slide with speaker notes is two
    SCREEN_LINKS = 8           # E16 caps a slide at eight figures anyway

    def _screen(self) -> dict | None:
        """The current screen reading, or None. A plain dict read of the poller's last tick: it
        does no I/O, so calling it inside the answer path inherits none of the poller's failure
        modes. `file` is required, because a reading that did not resolve to an indexed file has
        nothing to pack, and a reading older than `screen_ttl_s` counts as no screen at all."""
        fn = self.screen_state
        if fn is None or not self.settings.get("screen_bias", 1.0):
            return None
        try:
            s = fn()
        except Exception as e:  # noqa: BLE001
            log.debug("screen_state failed: %s", e)
            return None
        if not isinstance(s, dict) or s.get("status") != "ok" or not s.get("file"):
            return None
        age = s.get("age")
        if age is not None and float(age) > float(self.settings.get("screen_ttl_s", 10.0)):
            return None
        return s

    def _chunks_at(self, file: str, loc: str | None) -> list:
        """The indexed chunks for the page on screen, with their own full citations.

        The screen gives a head (`"slide 4"`); the index stores `slide 4 'FY26 plan by region'`
        and `... notes`. Matching on the head plus a space is what keeps slide 4 off slide 40.
        """
        chunks = getattr(self.index, "chunks", None) or []
        out = []
        head = (loc or "").strip()
        for ch in chunks:
            if getattr(ch, "file", None) != file:
                continue
            cloc = str(getattr(ch, "loc", "") or "")
            if head and not (cloc == head or cloc.startswith(head + " ")):
                continue
            out.append(ch)
            if len(out) >= self.SCREEN_CHUNKS:
                break
        return out

    def _screen_figures(self, s: dict) -> list[dict]:
        """Every figure on the page on screen with the model cell behind it, exact `loc` only."""
        rows: list[dict] = []
        for ch in self._chunks_at(s["file"], s.get("loc")):
            rows.extend(self._links_rows(ch.file, ch.loc, exact=True, limit=self.SCREEN_LINKS))
        return rows[:self.SCREEN_LINKS]

    def _screen_block(self, s: dict, seen_chunks: set[int]) -> str:
        """What the page on screen is worth as evidence: its own chunk text and, for a deck or a
        memo page, the model cells behind its figures. One indexed SELECT, no model call."""
        head = f"# on screen now: {s.get('app')} › {s['file']}"
        if s.get("loc"):
            head += f" › {s['loc']}"
        if s.get("title"):
            head += f" ({s['title']})"
        if s.get("sheet"):
            head += f" › sheet {s['sheet']}" + (f" › {s['cell']}" if s.get("cell") else "")
        head += "  (the question is probably about this)"
        lines = [head]
        for ch in self._chunks_at(s["file"], s.get("loc")):
            if ch.id in seen_chunks:
                continue
            seen_chunks.add(ch.id)
            lines.append(f"# {ch.file} › {ch.loc}\n{re.sub(r'\s+', ' ', ch.text)[:600]}")
            # the chunk's OWN citation, never the screen's head: see `_links_rows`
            lines.extend(self._links_lines(ch.file, ch.loc, exact=True))
        return "\n".join(lines)[:self.SCREEN_CAP]

    def _screen_ask(self, text: str, ask: Any = None) -> Any:
        """The deictic rule: "what's this number based on" has no metric of its own, so the page
        on screen is the referent and the ask is filled from its figures rather than from the
        workbook's vocabulary.

        The metric comes off the cell one of the page's figures resolved to, which is a row the
        index holds, so `ask.evidence_targets` packs that row exactly as a spoken metric would:
        the figure's own neighbours across every period, which is what "what's this based on"
        wants. With no page known this is silent: the ask is returned untouched and today's
        pre-pack runs at full width.

        Only the metric. Filling `period` from the cell's own header as well costs 16 more numeric
        cells at the median on `data/screen/deictic.jsonl` (25.5 against 9.0) and buys no reach,
        because a period slot makes `evidence_targets` pack that whole column: thirty rows of a
        period nobody asked about.
        """
        if askmod is None or (ask is not None and not isinstance(ask, askmod.Ask)):
            return ask
        if not deictic(text, ask):
            return ask
        s = self._screen()
        if s is None:
            log.debug("screen ask: silent (no page on screen) %r", text[:50])
            return ask
        figs = self._screen_figures(s)
        if not figs:
            log.debug("screen ask: silent (no figures on %s › %s)", s.get("file"), s.get("loc"))
            return ask
        row, header, cell_file = None, None, None
        for lk in figs:
            row, header = self._cell_labels(lk)
            cell_file = lk.get("cell_file")
            if row:
                break
        if not row:
            log.debug("screen ask: silent (no row label behind %s › %s)", s.get("file"), s.get("loc"))
            return ask
        out = ask if ask is not None else askmod.Ask(line=text)
        out.metric = askmod.Slot(row, 1.0, "screen")
        out.file = out.file or cell_file
        log.info("deictic ask from the screen: %s › %s -> metric=%r (header %r)",
                 s.get("file"), s.get("loc"), row, header)
        return out

    def _cell_labels(self, lk: dict) -> tuple[str | None, str | None]:
        """The row label and column header behind one linked figure, so the ask can name them."""
        ref, sheet = lk.get("cell_ref"), lk.get("cell_sheet")
        if not ref or not sheet:
            return None, None
        q = ("SELECT row_label, header FROM cells WHERE sheet = '{s}' AND cell_ref = '{r}'"
             .format(s=str(sheet).replace("'", "''"), r=str(ref).replace("'", "''")))
        if lk.get("cell_file"):
            q += " AND file = '{f}'".format(f=str(lk["cell_file"]).replace("'", "''"))
        try:
            rows = [r for r in self.index.sql(q, limit=1) if "error" not in r]
        except Exception as e:  # noqa: BLE001
            log.debug("cell labels failed: %s", e)
            return None, None
        if not rows:
            return None, None
        return (rows[0].get("row_label") or None), (rows[0].get("header") or None)

    def _evidence_for_target(self, tgt: dict, seen_chunks: set[int]) -> str:
        kind = tgt.get("kind")
        if kind == "lever":
            ref = f"{tgt.get('sheet')}!{tgt.get('cell')}"
            lines = [f"# lever (whatif can change it): {tgt.get('label')} "
                     f"[{tgt.get('file')} › {ref} = {tgt.get('value')}]"]
            lines += [f"  moves {o.get('label')} [{o.get('cell')}]" for o in (tgt.get("outputs") or [])[:6]]
            return "\n".join(lines)
        if kind == "chunk":
            lines = []
            for cid in (tgt.get("chunk_ids") or [])[:2]:
                if cid in seen_chunks:
                    continue
                seen_chunks.add(cid)
                ch = self.index.chunks[cid] if 0 <= cid < len(self.index.chunks) else None
                if ch:
                    lines.append(f"# {ch.file} › {ch.loc}\n{re.sub(r'\s+', ' ', ch.text)[:600]}")
                    lines.extend(self._links_lines(ch.file, ch.loc))
            return "\n".join(lines)
        table = tgt.get("table")
        if not table:
            return ""
        where = f"{tgt.get('file')} › {tgt.get('sheet') or table}"
        cols = self._columns(table)
        if not cols:
            return ""
        label_of = {c["name"]: (c["label"] or c["name"]) for c in cols}
        letter_of = {c["name"]: c["letter"] for c in cols}
        first = cols[0]["name"]

        def render_row(r: dict) -> str:
            row = r.get("_row")
            cells = []
            for name, val in r.items():
                if name.startswith("_") or val is None or name == first:
                    continue
                ref = f"{letter_of.get(name) or '?'}{row}" if row is not None else "?"
                cells.append(f"{label_of.get(name, name)}={val} [{ref}]")
            head = r.get(first)
            return f"{where} › row {row} ({head}): " + ", ".join(cells)

        if kind == "row":
            rows = self.index.sql(f'SELECT * FROM "{table}" WHERE _row = {int(tgt["row"])}', limit=1)
            rows = [r for r in rows if "error" not in r]
            return "\n".join(render_row(r) for r in rows)
        if kind == "column":
            col = tgt["column"]
            rows = self.index.sql(f'SELECT "{first}", "{col}", _row FROM "{table}" WHERE "{col}" IS NOT NULL', limit=30)
            rows = [r for r in rows if "error" not in r]
            out = [f"# {where} › column '{tgt.get('label') or col}' ({letter_of.get(col)})"]
            out += [f"  {r.get(first)}: {r.get(col)} [{letter_of.get(col)}{r.get('_row')}]" for r in rows]
            return "\n".join(out)
        if kind == "value":
            col, val = tgt["column"], str(tgt["value"]).replace("'", "''")
            rows = self.index.sql(f'SELECT * FROM "{table}" WHERE "{col}" = \'{val}\'', limit=8)
            rows = [r for r in rows if "error" not in r]
            return "\n".join(render_row(r) for r in rows)
        if kind == "table":
            count = self.index.sql(f'SELECT COUNT(*) AS n FROM "{table}"', limit=1)
            n = count[0].get("n", 0) if count and "error" not in count[0] else 0
            if n and n <= 16:
                rows = [r for r in self.index.sql(f'SELECT * FROM "{table}"', limit=16) if "error" not in r]
                return "\n".join(render_row(r) for r in rows)
            return f"# table {table} ({where}) has {n} rows; use sql to query it"
        return ""

    # ------------------------------------------------------------------ speculative path

    def _maybe_prefetch(self, hits: list, u: Utterance) -> None:
        keys = frozenset(topic_key(h.topic) for h in hits if topic_key(h.topic))
        if not keys:
            return
        now = time.time()
        ttl = self.settings["answer_ttl_s"]
        # an existing warm candidate that already covers these topics makes this one redundant
        for c in self.candidates.values():
            if c.status != "expired" and now - c.created < ttl and keys <= c.keys:
                return
        key = "|".join(sorted(keys))
        if now - self._last_prefetch.get(key, 0) < self.settings["prefetch_cooldown_s"]:
            return
        self._last_prefetch[key] = now
        # E52: the mirror of the check above. A draft still warming on a strict subset of these
        # topics is superseded by this one, and `_best_warm` would pass over it anyway, so it stops
        # holding a session slot for a card nothing will ask for.
        for old in list(self.candidates.values()):
            if old.utt_id is None and old.keys < keys:
                self._cancel_draft(old, "superseded")
        label = " · ".join(dict.fromkeys(h.topic for h in hits[:3]))
        c = Candidate(id=new_id("c"), topic=label, keys=keys, status="warming", created=now, hits=list(hits))
        self.candidates[key] = c
        self._trim_warm()
        self.stats["prefetches"] += 1
        c.task = asyncio.create_task(self._run_candidate(c, u))

    def _cancel_draft(self, c: Candidate, why: str) -> bool:
        """Stop a speculative draft nobody is waiting for, and give its session slot back (E52).

        A draft holds a slot in the provider's pool for its whole run, and E51 priced a live answer
        that starts beside four other calls at 5.28 s of first text against 1.86 s beside none. A
        draft whose topics a newer one already covers, or whose line the gate has just called
        `none`, is holding that slot for a card that will never be drawn.

        Two things are never cancelled: a candidate a question has already landed on (`answer_id`),
        which is a card the page is waiting for, and one that has already answered, whose slot is
        back in the pool and whose text `_best_warm` can still hand to a later question.
        """
        if c.answer_id or c.status != "warming" or c.task is None or c.task.done():
            return False
        c.task.cancel()
        c.status = "expired"
        self.candidates.pop(c.key, None)
        self.stats["drafts_cancelled"] = self.stats.get("drafts_cancelled", 0) + 1
        log.info("skip cancelled draft: %s %r", why, (c.question_guess or c.topic)[:60])
        asyncio.create_task(self.emit(c.msg()))
        return True

    def _drop_utt_drafts(self, u: Utterance, why: str) -> None:
        """The line is over and the gate says it was not a question: the draft that fired
        mid-sentence for it is answering nobody (E52)."""
        for c in list(self.candidates.values()):
            if c.utt_id == u.id:
                self._cancel_draft(c, why)

    def _trim_warm(self) -> None:
        live = [c for c in self.candidates.values() if c.status != "expired"]
        overflow = len(live) - int(self.settings["max_warm"])
        if overflow <= 0:
            return
        for c in sorted(live, key=lambda c: c.created)[:overflow]:
            c.status = "expired"
            if c.task and not c.task.done():
                c.task.cancel()
            self.candidates.pop(c.key, None)
            asyncio.create_task(self.emit(c.msg()))

    def _prefetch_question(self, u: Utterance) -> None:
        """Interview mode: a question from Them is still being spoken; start drafting the answer now."""
        key = f"utt:{u.id}"
        if key in self.candidates:
            return
        now = time.time()
        label = (u.text[:48] + "…") if len(u.text) > 48 else u.text
        c = Candidate(id=new_id("c"), topic=label, keys=frozenset({key}), status="warming", created=now,
                      hits=self._topic_hits(u.text)[:4], utt_id=u.id)
        self.candidates[key] = c
        self._trim_warm()
        self.stats["prefetches"] += 1
        c.task = asyncio.create_task(self._run_candidate(c, u))

    async def _run_candidate(self, c: Candidate, u: Utterance) -> None:
        await self.emit(c.msg())
        evidence = self._evidence(c.hits, u.text)
        pre_block, prerun = await self._prerun_whatif(u.text, c.hits)
        if pre_block:
            evidence = pre_block + "\n\n" + evidence
        if c.utt_id:
            trigger = (f"SPECULATIVE: the other side is mid-sentence: \"{u.text}\". Predict where it is going and "
                       "draft the response now; it will be shown the moment they finish.")
        else:
            trigger = (f"SPECULATIVE: these topics just came up in the current utterance: {c.topic}. "
                       "No question has been asked yet. Predict the single most likely question and answer it from the files.")

        async def on_result(data: dict, model: str, ms: int, is_primary: bool) -> None:
            if c.status == "expired":
                return
            c.question_guess = data.get("question") or c.question_guess
            from_ = data.get("answered_from") or ("files" if data.get("in_files") else "none")
            if prerun and from_ in ("files", "model"):
                from_ = "derived"
                if not data.get("working"):
                    data["working"] = self._working_from_last() or None
            c.answer = Answer(
                id=c.answer_id or new_id("a"), question=c.question_guess, answer=data.get("answer", ""),
                figure=data.get("figure"), citations=data.get("citations", []), source="prefetch",
                latency_ms=ms, in_files=bool(data.get("in_files", False)), ts=time.time(), model=model, topic=c.topic,
                answered_from=from_,
                web_citations=data.get("web_citations") or [], mode=self.mode.id, say=data.get("say") or "",
                working=data.get("working") or None, derived=(from_ == "derived"),
            )
            if c.status == "warming":
                c.status = "ready"
            await self.emit(c.msg())
            if c.answer_id:   # a question landed while we were warming: show it as an answer now
                hold = await self._late_draft_hold(c)
                if hold:
                    # held, not expired: it stays a warm candidate the consultant can open, and
                    # the live call already running on this id fills the card (E40)
                    self.stats["late_held"] = self.stats.get("late_held", 0) + 1
                    log.info("skip held draft: %s %r for %r", hold,
                             (c.question_guess or "")[:50], c.asked[:50])
                    return
                asked = self._asked.get(c.answer_id)
                if asked is not None:
                    c.answer.who = asked.who     # E68; `utt` stays unset on this path (E40 reads that)
                self.answers[c.answer_id] = c.answer
                c.status = "shown"
                await self._emit_answer(c.answer)

        try:
            await self._hedged(trigger, evidence, u.text, on_result, tools=([] if prerun else None),
                               speculative=True, derived=bool(prerun) or whatif_ish(u.text))
        except asyncio.CancelledError:
            raise
        except DraftDropped as e:
            # E52: the pool gave this draft's turn to a live answer. Not an error, and not the
            # brain's to retry: the question it was guessing at is being answered for real.
            self.stats["drafts_dropped"] = self.stats.get("drafts_dropped", 0) + 1
            log.info("skip dropped draft: %s %r", e, c.topic[:60])
            c.status = "expired"
            await self.emit(c.msg())
        except Exception as e:  # noqa: BLE001
            self.stats["errors"] += 1
            log.exception("candidate %s failed: %s", c.topic, e)
            c.status = "expired"
            await self.emit(c.msg())

    # ------------------------------------------------------------------ question path

    async def _on_final(self, u: Utterance, hits: list) -> None:
        mode = self.mode
        # notes / quiet: no card is ever automatic, but a question is still registered and shown
        # as heard (Jackson, 2026-09-24: a Quiet page that swallows "what is five year valuation"
        # without a trace reads as broken); `_answer_question` turns every route into `heard`
        register_only = not mode.auto
        text = u.text
        # the transcriber splits at sentence pauses; two finals from one source within a second are one question
        prev = self.__dict__.setdefault("_last_final", {}).get(u.source)
        joined_from: tuple | None = None
        # E52: two finals on the *same* utterance are a retraction, not a split. The transcriber
        # re-decodes the same audio plus whatever came next, so the second text already contains
        # the first and joining them would say it twice; that case is reconciled below instead.
        same_utt = bool(prev and u.utt is not None and prev.get("utt") == u.utt)
        # E68: not when the earlier final is a finished question. A transcriber that stalls flushes
        # several turns inside one second (E41's stalled runs: 9 sub-second pairs, never one in
        # E17's twenty or E68's two), and joining those folded three client questions into one
        if prev and u.ts - prev["ts"] < 1.0 and prev.get("utt") is not None and u.utt is not None \
                and not same_utt and not u.retracts and not str(prev.get("text") or "").rstrip().endswith("?"):
            text = f"{prev['text']} {text}".strip()
            joined_from = (u.source, prev["utt"])
            hits = self._topic_hits(text) or hits
        self._last_final[u.source] = {"ts": u.ts, "text": u.text, "utt": u.utt}
        low = text.lower()
        heur = questionish(text) or whatif_ish(text)
        words = len(text.split())
        who = "Me" if u.source == "mic" else "Them"
        warm = self._best_warm(text, hits, utt_id=u.id)
        if joined_from and joined_from in self._utt_answers:
            # the first half was already answered: re-answer the whole question into that card,
            # under the policy it registered with (E68: a consultant's check stays a check)
            aid = self._utt_answers[joined_from]
            was = self._asked.get(aid)
            await self._answer_question(text, hits, warm, source="live", utt=(u.source, u.utt), aid=aid,
                                        who=who, policy=was.policy if was else "answer",
                                        kind=was.kind if was else None, via="joined", line=u.id)
            return
        if mode.question_forming:
            # interview: every question from the interviewer gets a drafted answer, no gating on files
            if u.source == "system" and heur and words >= 3:
                await self._answer_question(text, hits, warm, source="live", via="interview", line=u.id)
            return
        cue = any(p in low for p in mode.cues) and words >= 4
        prior_aid = self._utt_answers.get((u.source, u.utt)) if u.utt is not None else None
        # E52. The route already starts on the eager final (`stt/local.py _maybe_eager_final`):
        # 198 of the 205 E17 turns finalised inside its window, well under the 900 ms of
        # `LONG_SILENCE_MS` the settled final waits for. What the eager final costs is that the
        # transcriber can retract it and finalise the same utterance again with more words: 4 of
        # 388 finals over the 20 runs, 1.0%. Those stopped here, so a card drawn from the half
        # question was never corrected. When the words have moved, the question is re-routed into
        # the same card id, which is what `aid=prior_aid` below already does for a redraw.
        restated = bool(prior_aid and same_utt and prev["text"].strip() != text.strip())
        if restated:
            self.stats["final_reconciled"] = self.stats.get("final_reconciled", 0) + 1
            log.info("settled final moved, re-answering %s: %r -> %r", prior_aid,
                     prev["text"][:40], text[:60])
        if prior_aid and not u.retracts and not restated:
            return                       # same utterance finalized twice with nothing new
        if u.source == "mic":
            # The consultant's own lines. Until E68 none of them was gated: E6a measured every gate
            # half-misreading the consultant's questions to the client as questions for the
            # copilot. What reads them now is a policy of their own (`_me_question`): the gate on
            # the words alone, then `me_route`. The spoken-figure guard (E9b) still reads every
            # line on the way past; it costs no model call and is the only other card this side
            # produces.
            if not prior_aid:
                if not register_only:
                    await self._check_spoken(text)
            await self._me_question(u, text, hits, prior_aid)
            return
        gate_task = None
        # only the client's lines are gated here: the consultant's have their own path above
        if words >= 3:
            gate_task = asyncio.create_task(self._gate(u))
        if (heur and (hits or warm)) or cue or prior_aid:
            # a statement shaped like a question was the main source of wrong cards on the
            # simulator (E5: 20 of 61 client turns, most after the gate had said "none"); the
            # 0.2 ms pre-gate tier vetoes those before a card is drafted, and the full gate keeps
            # running in the background so the intervention log (E12) has a verdict for every card
            veto = await self._pregate_veto(u) if not prior_aid else None
            typed = None
            kind, via = None, "heuristic"
            # E68: does this registration carry the no-answer guarantee? Only when something read
            # the line as a question: the pre-gate deciding a question, request or what-if, the
            # gate's verdict clearing `SURE_P` when the line waited for it, or the second opinion. A question-shaped line the gate doubts still registers (the page
            # shows it was heard), but a "Couldn't answer" card on it would be the noise E5 counted
            sure = veto is None and u.__dict__.get("_pregate_kind") in ("question", "what-if", "request")
            if veto == "unsure" and gate_task is not None:
                # the local tier could not tell: wait for the full gate (about 200 ms) before a card
                # shows, since E34 measured prefetch cards on "none" and unfinished lines as the
                # largest remaining source of wrong cards (4 of 16)
                try:
                    verdict = await gate_task
                    gate_task = None
                    typed = verdict.get("ask")     # this line waited for the gate, so the ask is here
                    kind = verdict.get("kind")
                    if verdict.get("kind") == "none" and float(verdict.get("confidence", 0)) >= 0.7:
                        veto = ("none", float(verdict.get("confidence", 0)))
                    elif verdict.get("complete") is False and float(verdict.get("confidence", 0)) >= 0.7 \
                            and not text.rstrip().endswith("?"):
                        # E68: a line that ends in "?" is a finished question whatever the gate's
                        # `complete` head says (9 of the 1,707 real cut-offs in data/gate/real end
                        # in one); E17's Jev runs lost "what's the design and UX piece run us
                        # separately?" twice to exactly this
                        veto = ("incomplete", float(verdict.get("confidence", 0)))
                    else:
                        veto = None
                        sure = kind in ("question", "request") and float(verdict.get("confidence", 0)) >= SURE_P
                except Exception as e:  # noqa: BLE001
                    log.debug("gate wait failed: %s", e)
                    veto = None
            if veto and veto != "unsure" and veto[0] == "none" and heur:
                # E68: the pre-gate's confident "none" is its own call, and on a question-shaped
                # line it is the one the tier-2 gate disagrees with most usefully. Both E17 misses
                # that shape stood for ("Okay, walk me through the security and compliance line
                # next", nine turn-runs) read `request` at 0.94 and 0.98 on tier 2. One call, only
                # on the lines this veto was about to drop.
                second = await self._second_opinion(u)
                if second is not None:
                    log.info("veto overturned: pre-gate none p=%.2f, tier 2 %s p=%.2f %r", veto[1],
                             second.get("kind"), float(second.get("confidence", 0)), text[:60])
                    veto, via, kind, sure = None, "second-opinion", second.get("kind"), True
            if veto and veto != "unsure":
                if await self._follow_up(u, text, hits, warm):
                    return
                log.info("heuristic card vetoed: %s p=%.2f %r", veto[0], veto[1], text[:60])
                self.stats["vetoed"] = self.stats.get("vetoed", 0) + 1
                self._drop_utt_drafts(u, veto[0])     # E52: nothing will show this line's draft
                return
            log.info("heuristic answer: heur=%s hits=%d warm=%s cue=%s %r", heur, len(hits), bool(warm), cue, text[:60])
            kind = kind or u.__dict__.get("_pregate_kind")
            await self._answer_question(text, hits, warm, source="live", utt=(u.source, u.utt), aid=prior_aid,
                                        ask=typed, kind=("question" if kind in (None, "none", "what-if") else kind),
                                        via=via, line=u.id, sure=sure)
            return
        if not gate_task:
            return
        try:
            verdict = await gate_task
        except Exception as e:  # noqa: BLE001
            log.warning("gate failed: %s", e)
            return
        kind = verdict.get("kind")
        wanted = verdict.get("needs_files") or (mode.model_knowledge and kind in ("objection", "request"))
        if kind in mode.kinds and wanted and float(verdict.get("confidence", 0)) >= self.settings["question_threshold"]:
            q = (verdict.get("question") or text).strip()
            qhits = hits or self._topic_hits(q)
            warm = self._best_warm(q, qhits) or warm
            await self._answer_question(q, qhits, warm, source="live", utt=(u.source, u.utt),
                                        ask=verdict.get("ask"), kind=kind, via="gate", line=u.id)
            return
        if await self._follow_up(u, text, hits, warm):
            return
        self._drop_utt_drafts(u, f"gate {kind}")      # E52: the line is over and got no card

    # ------------------------------------------------------------------ the consultant's questions (E68)

    async def _me_question(self, u: Utterance, text: str, hits: list, prior_aid: str | None = None) -> None:
        """A consultant's line that may be a question, and what to do about it (E68).

        Shape first (`me_candidate`, 0.01 ms), so the nine lines in ten that are statements cost
        nothing. Then the gate reads the words as if the client had said them (`_me_reading`, one
        tier-2 call), and `me_route` turns that reading into a policy: `answer` for a lookup the
        consultant wants the copilot to make, `check` for a fact asked of the client (a card only
        when the files hold it), `heard` for a view or a plan asked of the client, or nothing for
        a statement, a tag or a check that the room is following. Measured on 2,394 real
        consultant lines: 58 of 90 questions registered, 18 of 2,304 other lines, none of them
        routed to `answer` (docs/experiments/E68-questions.md).

        A consultant's line the gate calls a question with `needs_lookup` false is still
        registered: the page shows it was heard. It is the model call, never the registration,
        that `needs_lookup` and the view words decide.
        """
        if prior_aid:
            was = self._asked.get(prior_aid)
            if was is not None:          # the settled final moved: same registration, new words
                await self._answer_question(text, hits, None, source="live", utt=(u.source, u.utt),
                                            aid=prior_aid, who="Me", policy=was.policy, kind=was.kind,
                                            via=was.via, line=u.id)
            return
        if len(text.split()) < 3 or not me_candidate(text):
            return
        v = await self._me_reading(u)
        if not v:
            return
        route = me_route(text, v.get("kind"), float(v.get("p_kind", 0.0)), bool(v.get("needs_files")))
        log.info("me question: %s kind=%s p=%.2f needs=%s %r", route or "none", v.get("kind"),
                 float(v.get("p_kind", 0.0)), v.get("needs_files"), text[:60])
        if not route:
            return
        self.stats["me_routes"][route] = self.stats["me_routes"].get(route, 0) + 1
        warm = self._best_warm(text, hits, utt_id=u.id) if route == "answer" else None
        await self._answer_question(text, hits, warm, source="live", utt=(u.source, u.utt), who="Me",
                                    policy=route, kind=v.get("kind") or "question", via="me", line=u.id,
                                    ask=v.get("ask"))

    async def _me_reading(self, u: Utterance) -> dict | None:
        """The gate's verdict on a consultant's line as if the client had said it (E68).

        The gate was trained to answer `none` for anything Me says (data/gate/README.md, rule 1),
        so asked straight it can only say the line is not the copilot's business. Asked about the
        words alone it reads a question the way it reads a client's: on 24 hand-written lookups it
        calls all 24 a question at 0.99, and on the E17/E65 consultant lines every question the
        scripts put in the consultant's mouth. Tier 2 only: the pre-gate's Me/Them feature is most
        of its judgement (E22) and it defers on half of these.
        """
        if self.decider is None or DQ is None:
            return None
        try:
            return await self._gate_decider(u, tier2=True, who="Them", tag="me reading")
        except Exception as e:  # noqa: BLE001
            log.debug("me reading failed: %s", e)
            return None

    async def _follow_up(self, u: Utterance, text: str, hits: list, warm: Candidate | None) -> bool:
        """Register a client's elliptical follow-up that nothing else did (E68, `follow_up_shaped`).

        Only within `FOLLOW_UP_S` of the last client question that registered, and it borrows that
        question's topic hits when it names none of its own, so the evidence has the row the chain
        is about; the model resolves "and the retail partner" against the transcript it is shown.
        """
        if not follow_up_shaped(text):
            return False
        last = max((q for q in self._asked.values() if q.who == "Them" and q.via != "repeat"),
                   key=lambda q: q.t_q, default=None)
        if last is None or time.time() - last.t_q > FOLLOW_UP_S:
            return False
        log.info("follow-up of %r: %r", last.text[:40], text[:60])
        self.stats["follow_ups"] = self.stats.get("follow_ups", 0) + 1
        await self._answer_question(text, hits or last.hits, warm, source="live", utt=(u.source, u.utt),
                                    kind="question", via="follow-up", line=u.id)
        return True

    async def _second_opinion(self, u: Utterance) -> dict | None:
        """Tier 2 alone, on a question-shaped client line the pre-gate just vetoed as `none` (E68).

        The verdict when tier 2 reads a question, request or objection that needs a lookup at
        `SECOND_OPINION_P` or more, else None. With no second tier behind the pre-gate there is
        nobody to ask, and the veto stands exactly as it did before.
        """
        if getattr(self.decider, "slow", None) is None or DQ is None:
            return None
        try:
            v = await self._gate_decider(u, tier2=True, tag="second opinion")
        except Exception as e:  # noqa: BLE001
            log.debug("second opinion failed: %s", e)
            return None
        if v.get("kind") in ("question", "request", "objection") and v.get("needs_files") \
                and float(v.get("confidence", 0)) >= SECOND_OPINION_P:
            self.stats["second_opinion"] = self.stats.get("second_opinion", 0) + 1
            return v
        return None

    # ------------------------------------------------------------------ the spoken guard (E9c)

    def _spoken_guard(self) -> Any:
        """The guard for this index build, or None while there is not one yet.

        Never built here: `spoken.guard_for` costs one `ask.build_vocab` plus the rarity tables,
        which is thirty times the check it enables. `_refresh_vocab` builds it in the thread hop
        it already makes; this picks that up, and starts one background build when the index has
        moved on without it (which is also the only path when the E38 modules failed to import).
        """
        version = getattr(self.index, "version", 0)
        if spoken is None or self._spoken_none == version:
            return None                  # a folder with no cells to argue with: asked once, not once a line
        g = getattr(self.index, "_spoken_guard", None)
        if g is not None and getattr(g, "version", None) == version:
            return g
        if not self._spoken_busy:
            self._spoken_busy = True
            asyncio.create_task(self._build_spoken())
        return None

    async def _build_spoken(self) -> None:
        version = getattr(self.index, "version", 0)
        t0 = time.time()
        try:
            g = await asyncio.to_thread(spoken.guard_for, self.index,
                                        self._vocab if self._vocab_version == version else None)
            log.info("spoken guard v%s in %d ms%s", version, int((time.time() - t0) * 1000),
                     "" if g is not None else " (unavailable)")
            if g is None:
                self._spoken_none = version
        except Exception as e:  # noqa: BLE001
            self._spoken_none = version
            log.info("spoken guard unavailable: %s", e)
        finally:
            self._spoken_busy = False

    async def _check_spoken(self, text: str) -> None:
        """A figure the consultant just said that the files contradict (E9b: 87 of 99 caught, 2
        false alarms in 164, 0.88 ms p50). No model call, no gate, no network.

        `text` rather than `u.text`: `_on_final` has already joined the two finals a sentence
        pause split, so "so that's about" and "nine million in fees" arrive here as one line.

        Inline on the transcript thread, because a thread hop costs more than the check does.
        `SPOKEN_BUDGET_MS` is the escape hatch and nothing else: one slow check and every later
        line goes to a worker thread, so a folder we have not measured cannot hold audio up.
        """
        if len(text.split()) < SPOKEN_MIN_WORDS:
            return                       # "nine million" alone names no metric to check it against
        g = self._spoken_guard()
        if g is None:
            return                       # still building: this line goes by unchecked, once
        # the what-if and the scratch pad are the whole of the silence on a recalculated figure: a
        # number the consultant derived a moment ago is in no cell by construction, so without them
        # every sentence describing a scenario is a contradiction
        kw = {"whatif": self._last_whatif, "scratch": self.scratch, "prior": self._ask_prior()}
        try:
            sp = await asyncio.to_thread(g.check, text, **kw) if self._spoken_slow else g.check(text, **kw)
        except Exception as e:  # noqa: BLE001
            log.debug("spoken guard failed: %s", e)
            return
        if sp.ms > SPOKEN_BUDGET_MS and not self._spoken_slow:
            self._spoken_slow = True
            log.info("spoken guard took %.0f ms (budget %.0f): later lines go off the transcript thread",
                     sp.ms, SPOKEN_BUDGET_MS)
        if sp.verdict != "contradicts":
            return
        key = sp.cell or sp.label or sp.said
        now = time.time()
        self._spoken_at = {k: t for k, t in self._spoken_at.items() if now - t < SPOKEN_COOLDOWN_S}
        if key in self._spoken_at:
            return          # said again inside the minute: the consultant has read the card once
        self._spoken_at[key] = now
        self.stats["corrections"] = self.stats.get("corrections", 0) + 1
        log.info("spoken guard: contradicts %s said %s actual %s", key, sp.said, sp.actual_s)
        await self._emit_answer(self._correction(sp))

    def _correction(self, sp: Any) -> Answer:
        """The correction card, in the answer frame the page already draws (docs/ui-protocol.md).

        `answered_from="correction"` is the only new value anywhere in the frame, and the current
        page tolerates an unknown one: `answeredFrom` falls through to the Files dot instead of
        breaking, so the GUI session can style it as a quiet correction whenever it gets to it.

        `question` carries the figure that was said and not the line it came from. The
        intervention log keeps no mic side (E12) and the said figure is the card's own content.
        """
        file_, _sep, loc = sp.cell.partition(" › ")
        a = Answer(
            id=new_id("s"), question=f"said {sp.said}".strip(), answer=sp.reason or sp.working,
            figure=sp.actual_s, citations=[{"file": file_, "loc": loc}] if sp.cell else [],
            source="live", latency_ms=int(round(sp.ms)), in_files=True, ts=time.time(),
            model="", topic=sp.label or None, answered_from="correction",
            mode=str(self.settings.get("mode", "call")),
            say=(f"{sp.label} is {sp.actual_s}" if sp.label else f"Actually {sp.actual_s}"),
            working=sp.working or None,
            # the guard's own verdict, in the shape E23's verifier already puts here. `ok` is true
            # because the figure on the card *is* the cell: what is contradicted is the room.
            verdict={"ok": True, "level": "spoken", "problems": [], "reason": sp.reason,
                     "spoken": sp.verdict, "said": sp.said, "actual": sp.actual_s,
                     "cell": sp.cell, "how": sp.how, "ms": round(sp.ms, 2)})
        self.answers[a.id] = a
        return a

    # ------------------------------------------------------- the slide against the model (E63)

    async def on_screen_change(self, s: dict | None = None) -> Answer | None:
        """A new page came up: does anything printed on it still match the model?

        The server hands this the poller's own `on_change`, so it arrives on the poller thread
        and hops to the loop; the transcript thread is never on this path, and the check itself
        (a few indexed SELECTs, 0.7 to 4 ms a page) goes to a worker thread so the loop is not
        holding the audio callback up either.

        Two things keep it quiet. A page is judged **once a session**: stepping back to slide 4
        for the third time does not raise its card again. And a cell that has already produced a
        card is silent for `DRIFT_COOLDOWN_S`, which is E9c's rule, because two pages quoting the
        same stale number are one piece of news. A page suppressed by the cooldown still counts
        as judged, so the news does not reappear a minute later halfway through the call.
        """
        if driftmod is None or not self.settings.get("drift", 1.0):
            return None
        s = s if isinstance(s, dict) else self._screen()
        if not s or s.get("status") != "ok" or not s.get("file"):
            return None
        head = driftmod.head_of(s.get("loc"))
        if not head:
            return None                       # a workbook, or a deck with no slide index
        key = (str(s["file"]), head)
        if key == self._drift_page:
            return None      # the reading changed for some other reason; the page did not
        self._drift_page = key
        if key in self._drift_seen:
            return None
        try:
            rep = await asyncio.to_thread(driftmod.check, self.index, s)
        except Exception as e:  # noqa: BLE001  a drift check that fails is not a dead call
            log.debug("drift check failed: %s", e)
            return None
        self.stats["drift_ms"] = round(rep.ms, 2)
        if not rep.drifts:
            log.debug("drift: %s › %s clean (%d links, %.2f ms)", key[0], head, rep.links, rep.ms)
            return None
        d = rep.drifts[0]                     # the most trustworthy link, then the biggest move
        self._drift_seen.add(key)
        now = time.time()
        self._drift_at = {k: t for k, t in self._drift_at.items() if now - t < DRIFT_COOLDOWN_S}
        cell_key = f"{d.cell_file} › {d.cell}"
        if cell_key in self._drift_at:
            log.info("drift: %s › %s held back, %s raised a card %.0fs ago",
                     key[0], head, cell_key, now - self._drift_at[cell_key])
            return None
        self._drift_at[cell_key] = now
        self.stats["drift"] = self.stats.get("drift", 0) + 1
        log.info("drift: %s (%d of %d links on the page, %.2f ms)", d.line, len(rep.drifts),
                 rep.links, rep.ms)
        a = self._drift_card(d, rep)
        await self._emit_answer(a)
        return a

    def _drift_card(self, d: Any, rep: Any) -> Answer:
        """The drift card, in the answer frame the page already draws (docs/ui-protocol.md).

        `answered_from="drift"` is the only new value in the frame, the same shape E9c's
        correction took. `figure` is the model's number, because that is the one worth reading
        out; `question` is what the page says, because nobody asked anything and the slide's own
        claim is the card's subject.
        """
        a = Answer(
            id=new_id("d"), question=f"{d.head} says {d.value_text}",
            answer=d.line, figure=d.now_text,
            citations=[{"file": d.cell_file, "loc": f"{d.cell_sheet} › {d.cell_ref}"},
                       {"file": d.file, "loc": d.loc}],
            source="live", latency_ms=int(round(rep.ms)), in_files=True, ts=time.time(),
            model="", topic=(d.label or None), answered_from="drift",
            mode=str(self.settings.get("mode", "call")), say=d.say, working=d.working,
            # the check's own verdict, in the shape E23's verifier puts here. `ok` is true: the
            # figure on the card IS the cell. What disagrees with it is the slide.
            verdict={"ok": True, "level": "drift", "problems": [], "reason": d.line,
                     "cell": f"{d.cell_file} › {d.cell}", "was": d.was, "now": d.now,
                     "slide": d.value_text, "score": d.score, "n_alt": d.n_alt,
                     "others": len(rep.drifts) - 1, "ms": round(rep.ms, 2)})
        self.answers[a.id] = a
        return a

    async def _pregate_veto(self, u: Utterance):
        """The local pre-gate tier alone (0.2 ms): (kind, p) when it is sure the line is not a
        question, request, objection or what-if; None when it is unsure or absent."""
        fast = getattr(self.decider, "fast", None)
        if fast is None or DQ is None:
            return None
        try:
            mode = self.mode
            recent = self.transcript[-4:]
            lines = "\n".join(f"{'Me' if t.source == 'mic' else 'Them'}: {t.text}" for t in recent) or u.text
            who = "Me" if u.source == "mic" else "Them"
            state = ("A strategy consultant (Me) is on a call with a client (Them). Recent transcript, oldest first; "
                     f"judge ONLY the last line. Mode: {mode.label}. The last line is spoken by {who}: a line from Me "
                     "(the consultant) is never a question for the copilot, so answer kind=none and needs_lookup=false "
                     f"for it unless the mode is Interview.\n\n{lines}")
            qs = [DQ("kind", "choice", "What is the last line?",
                     options=["question", "what-if", "request", "objection", "none"]),
                  DQ("needs_lookup", "noul", "Answering the last line well needs a looked-up fact."),
                  DQ("complete", "noul", "The last line is a complete thought (not cut off mid-sentence).")]
            got = await fast.decide(state, qs)
            if not got or "kind" not in got:
                return "unsure"                    # the local tier deferred: the full gate decides
            kind, p = got["kind"].value, float(got["kind"].p)
            u.__dict__["_pregate_kind"] = kind      # E68: the registration carries what the line is
            if kind == "none" and p >= 0.9:
                return (kind, p)
            if "complete" in got and got["complete"].value is False and float(got["complete"].p) >= 0.9:
                return ("incomplete", float(got["complete"].p))
        except Exception as e:  # noqa: BLE001
            log.debug("pregate veto failed: %s", e)
        return None

    async def _late_draft_hold(self, c: Candidate) -> str:
        """Why a speculative draft that finished late must not push itself to the screen, or "".

        A draft on this path answers a question that landed while it was still warming, a median
        of 4.4 s earlier, and carries the candidate's guessed question rather than the asked one.
        E40's rule k2d, inverted so the held line carries its reason: hold when the pre-gate reads
        the line it answers as an objection, or when what the client is saying now does not read
        as a question, what-if, request or objection at P >= 0.50 -- a line that has opened with
        no settled word counting as failing that test. Over the 20 E17 runs it holds 45 of the 121
        late drafts, removes 18 bad cards of 83, keeps 106 of 108 hits and leaves the median first
        card at 1,936 ms.

        Fails open: no pre-gate, no reading from it, or a throw all show the card, as today.
        """
        q_kinds = ("question", "what-if", "request", "objection")
        asked = (c.asked or "").strip()
        if not asked or getattr(self.decider, "fast", None) is None:
            return ""
        try:
            if not c.asked_kind:
                # the line the draft answers is judged last, with up to three lines of context
                recent = [t for t in self.transcript[-4:] if t.text.strip() != asked][-3:]
                got = self._pregate_kind(
                    [f"{'Me' if t.source == 'mic' else 'Them'}: {t.text}" for t in recent]
                    + [f"Them: {asked}"])
                c.asked_kind = got[0] if got else ""
            if c.asked_kind == "objection":
                return "objection"
            line = (self._last_client_line or "").strip()
            if not line:
                return "still talking"       # a line has opened with nothing settled in it yet
            got = self._pregate_kind([f"Them: {line}"])
            if got is None:
                return ""                    # no pre-gate: show it, as today
            p_q = sum(float(got[1].get(k, 0.0)) for k in q_kinds)
            return "" if p_q >= 0.50 else "moved on"
        except Exception as e:  # noqa: BLE001
            log.debug("late draft rule failed: %s", e)
            return ""

    def _pregate_kind(self, lines: list[str]) -> tuple[str, dict] | None:
        """(kind, the kind distribution) for the last of `lines`, each already "Me: " / "Them: "
        prefixed, or None when there is no pre-gate. 0.2 ms, local, nothing hosted.

        This reads the pre-gate's weights through `PreGateDecider.gate` with `force=True` rather
        than through its `Decider` contract, which answers nothing below its own 0.75 threshold.
        That threshold protects a veto -- `_pregate_veto` kills a card nobody else will draft.
        Holding a late draft is a delay, not a veto: the live call on the same card id is already
        running and fills the card. 42 of the 121 late drafts E40 measured sit under the
        threshold on the line they answer and 47 on the line the client is on, and deferring on
        them costs 2 of the 18 bad cards the rule removes.
        """
        fast = getattr(self.decider, "fast", None)
        if fast is None or not lines or not getattr(fast, "ready", lambda: False)():
            return None
        got = fast.gate.decide(lines, self.mode.id, "Them", force=True)
        if not got:
            return None
        return str(got.get("kind") or ""), dict(got.get("probs") or {})

    async def _gate(self, u: Utterance) -> dict:
        """Should this utterance get an answer? Jev answers in ~100 ms when a key is set; otherwise
        the Haiku classifier. Returns the classifier-shaped verdict either way."""
        if self.decider is not None and DQ is not None:
            try:
                return await self._gate_decider(u)
            except Exception as e:  # noqa: BLE001
                log.info("decider failed (%s); falling back to classifier", e)
        return await self._classify(u)

    def _gate_lines(self, u: Utterance, who: str | None = None) -> str:
        """The judged line and the three transcript lines before it, oldest first, each prefixed
        "Me: " or "Them: ".

        Found by identity rather than taken as the transcript's last four: `on_transcript` runs
        one task per event, so a later final can be appended while this line is still being
        gated, and the gate would then judge the wrong line. `who` relabels the judged line only,
        which is how E68 reads a consultant's words as if the client had said them.
        """
        idx = None
        for i in range(len(self.transcript) - 1, max(-1, len(self.transcript) - 13), -1):
            if self.transcript[i] is u:
                idx = i
                break
        before = (self.transcript[max(0, idx - 3):idx] if idx is not None
                  else [t for t in self.transcript[-4:] if t is not u][-3:])
        name = who or ("Me" if u.source == "mic" else "Them")
        return "\n".join([f"{'Me' if t.source == 'mic' else 'Them'}: {t.text}" for t in before]
                         + [f"{name}: {u.text}"])

    async def _gate_decider(self, u: Utterance, tier2: bool = False, who: str | None = None,
                            tag: str = "gate") -> dict:
        """The gate's verdict on one line. `tier2` skips the pre-gate and asks the second tier
        alone, with no slot request riding along; `who` relabels the judged line (both E68). `tag`
        names the reading in the log: only "gate" lines are the gate proper, which is what
        `presage.sim.run.GATE_RE` reads and what `stats["gated"]` counts."""
        mode = self.mode
        lines = self._gate_lines(u, who)
        who = who or ("Me" if u.source == "mic" else "Them")
        state = ("A strategy consultant (Me) is on a call with a client (Them). Recent transcript, oldest first; "
                 f"judge ONLY the last line. Mode: {mode.label}. The last line is spoken by {who}: a line from Me "
                 "(the consultant) is never a question for the copilot, so answer kind=none and needs_lookup=false "
                 f"for it unless the mode is Interview.\n\n{lines}")
        qs = [
            DQ("kind", "choice", "What is the last line? question = asks for information; what-if = asks what a "
               "number would become under a different assumption (\"what if we charged 1%\", \"at 400 subscribers "
               "instead\"); request = asks Me to show, walk through or send something; objection = pushes back on "
               "a number, price, claim or plan, raises a risk or competitor, or hesitates; none = small talk, "
               "statements, agreement, filler.",
               options=["question", "what-if", "request", "objection", "none"]),
            DQ("needs_lookup", "noul", "Answering the last line well needs a looked-up number, date, name or fact "
               "(from the consultant's decks, models, documents or public sources), or, in Sales/Pitch/Interview "
               "mode, a suggested response. False for small talk, pleasantries and things answered from the head."),
            DQ("complete", "noul", "The last line is a complete thought (not cut off mid-sentence)."),
        ]
        t0 = time.time()
        d, ask = await self._gate_answers(u, state, qs, tier2=tier2)
        ms = int((time.time() - t0) * 1000)
        kind = d["kind"].value if "kind" in d else "none"
        needs = bool(d["needs_lookup"].value) if "needs_lookup" in d else False
        p_kind = float(d["kind"].p) if "kind" in d else 0.0
        if kind == "what-if":
            kind, needs = "question", True    # always needs the files: the recalculation runs on them
        conf = float(min(d["kind"].p if "kind" in d else 0.0, d["needs_lookup"].p if "needs_lookup" in d else 0.0))
        if tag == "gate":
            self.stats["gated"] = self.stats.get("gated", 0) + 1
            self.stats["gate_ms"] = ms
        log.info("%s %s kind=%s p=%.2f needs=%s complete=%s %dms", tag, type(self.decider).__name__, kind, conf,
                 needs, d["complete"].value if "complete" in d else None, ms)
        return {"kind": kind, "needs_files": needs, "confidence": conf, "question": "", "p_kind": p_kind,
                "complete": d["complete"].value if "complete" in d else True, "ask": ask}

    async def _gate_answers(self, u: Utterance, state: str, qs: list, tier2: bool = False) -> tuple[dict, Any]:
        """The gate's three answers, and the typed ask when this line paid for a hosted call anyway.

        Tier 1 is the local pre-gate, run here rather than inside the cascade so the tier is known:
        it answers most lines at 0.2 ms and its verdict is the one that ships, so those lines are
        exactly as they were and spend nothing on slots. A line that reaches tier 2 gets the five
        slots as well (E38):

          the gate is Jev    one request carries all eight questions: +0.4 ms p50 on 224 (E38a),
                             which is E2 proposal 4 -- only the round trip is expensive
          the gate is local  the slots go to the typed-intent Jev client BESIDE the gate, not
                             after it (E17 put the local Core ML gate in front). The verdict then
                             costs the difference, 76 ms -> 289 ms p50, bounded by `ASK_WAIT_S`;
                             only the gate branch of `_on_final` waits for it, the heuristic path
                             has already drafted its card by then.
        """
        cascade = self.decider if hasattr(self.decider, "fast") and hasattr(self.decider, "slow") else None
        if cascade is not None and not tier2:
            try:
                got = await cascade.fast.decide(state, qs)
            except Exception as e:  # noqa: BLE001
                log.debug("pregate tier failed: %s", e)
                got = None
            if got is not None:
                cascade.stats["fast"] = cascade.stats.get("fast", 0) + 1
                return got, None
            cascade.stats["slow"] = cascade.stats.get("slow", 0) + 1
        gate = (cascade.slow if cascade is not None else self.decider)
        if tier2:
            return await gate.decide(state, qs), None     # E68: a reading on its own, no slots with it
        prior = self._ask_prior()
        slot_qs, heads = self._ask_questions(u.text, prior)
        dec, batched = self._ask_decider()
        if not slot_qs or dec is None:
            return await gate.decide(state, qs), None
        if batched:
            try:
                d = await dec.decide(self._ask_state(u.text, prior, state), qs + slot_qs)
                ask = askmod.read_ask(d, u.text, self._vocab, prior, extra=heads)
            except Exception as e:  # noqa: BLE001  the slots must never cost the gate its verdict
                log.info("typed ask batch failed (%s); gate only", e)
                return await gate.decide(state, qs), None
        else:
            d, ask = await asyncio.gather(
                gate.decide(state, qs),
                self._slot_request(dec, u.text, prior, slot_qs, heads, wait=self.ASK_WAIT_S))
        self._keep_ask(ask)
        return d, ask

    async def _classify(self, u: Utterance) -> dict:
        recent = self.transcript[-3:]
        lines = "\n".join(f"{'Me' if t.source == 'mic' else 'Them'}: {t.text}" for t in recent) or u.text
        prompt = (
            "A strategy consultant (Me) is on a call with a client (Them). Classify ONLY the last line below.\n"
            "kind: question = asks for information; request = asks Me to show/walk through/send something; "
            "objection = pushes back on a number, price, claim or plan, raises a risk or competitor, or hesitates; "
            "none = anything else (small talk, statements, agreement).\n"
            "needs_files: true when answering well needs a looked-up number, date, name or fact, whether from the "
            "consultant's decks, models and documents (revenue, margins, headcount, churn, plan, assumptions) or from "
            "public sources (market size, benchmarks, rates, competitors, regulation). false for small talk "
            "(weather, greetings, scheduling, pleasantries), opinions, and things the consultant answers from the head.\n"
            "question: rephrase the ask as one standalone question that could be looked up (empty if none).\n\n"
            f"{lines}\n"
        )
        self.stats["classified"] += 1
        verdict = await self.classify(prompt, CLASSIFY_SCHEMA)
        return verdict or {"kind": "none", "needs_files": False, "confidence": 0, "question": ""}

    def _embed(self, texts: list[str]):
        """model2vec static embeddings (potion-base-8M, cached locally, ~0.03 ms). None if unavailable."""
        m = self.__dict__.get("_embedder")
        if m is None:
            try:
                from model2vec import StaticModel
                m = StaticModel.from_pretrained("minishlab/potion-base-8M")
            except Exception as e:  # noqa: BLE001
                log.info("embeddings unavailable: %s", e)
                m = False
            self._embedder = m
        if not m:
            return None
        import numpy as np
        v = m.encode(texts)
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v / n

    def _best_warm(self, text: str, hits: list, utt_id: str | None = None) -> Candidate | None:
        """The warm candidate that answers the question: the one drafted for this very utterance wins;
        else the best topic-key overlap; else the closest drafted question by embedding similarity."""
        if utt_id:
            c = self.candidates.get(f"utt:{utt_id}")
            if c and c.status in ("ready", "warming", "shown"):
                return c
        by_keys = self._best_warm_keys(text, hits)
        if by_keys:
            return by_keys
        live = [c for c in self.candidates.values() if c.status in ("ready", "shown") and c.question_guess]
        if not live:
            return None
        v = self._embed([text] + [c.question_guess for c in live])
        if v is None:
            return None
        sims = v[1:] @ v[0]
        i = int(sims.argmax())
        self._warm_sim = float(sims[i])
        if sims[i] >= self.settings.get("warm_sim_threshold", 0.56):
            log.info("warm hit by similarity %.2f: %s", sims[i], live[i].question_guess[:60])
            return live[i]
        if sims[i] >= self.settings.get("warm_sim_ask", 0.30):
            # the band where cosine is a coin flip (56% precision at 0.45, E7): hand it to a typed
            # decision in _answer_question before it is shown
            live[i].tentative = True
            return live[i]
        return None

    def _best_warm_keys(self, text: str, hits: list) -> Candidate | None:
        qkeys = {topic_key(h.topic) for h in hits} - {""}
        low = text.lower()
        best: tuple | None = None
        for c in self.candidates.values():
            if c.status not in ("ready", "warming", "shown"):
                continue
            overlap = len(qkeys & c.keys) + sum(1 for k in c.keys if len(k) > 3 and k in low and k not in qkeys)
            if overlap == 0:
                continue
            rank = (overlap, c.status in ("ready", "shown"), c.created)
            if best is None or rank > best[0]:
                best = (rank, c)
        return best[1] if best else None

    async def _answer_question(self, q: str, hits: list, warm: Candidate | None, source: str,
                               utt: tuple[str, int | None] | None = None, aid: str | None = None,
                               ask: Any = None, who: str | None = None, policy: str = "answer",
                               kind: str | None = None, via: str | None = None,
                               line: str | None = None, sure: bool = True) -> None:
        """Answer a question the brain has decided to act on. Every path that registers a question
        comes through here, so this is where the `question` frame goes out (E68, `_register`).

        `policy` is the E68 registration's (`Asked`): `answer` as always; `check` for a consultant
        asking the client a fact, which takes no warm or cached card (they answer a guess at some
        question, not this one) and shows only what the files hold; `heard` registers and stops.
        """
        t_q = time.time()
        who = who or ("Me" if source == "manual" else "Them")
        via = via or ("typed" if source == "manual" else "heuristic")
        if not self.mode.auto and via != "typed":
            policy = "heard"                 # notes / quiet: registered on the page, never answered
        qkeys = frozenset(topic_key(h.topic) for h in hits) - {""}
        recent = self.__dict__.setdefault("_recent_questions", [])
        if source != "manual" and qkeys and aid is None:
            dup = self._repeat_of(q, qkeys, t_q)
            if dup is not None:
                # E68: still registered, and pointed at the card that already answers it
                log.info("skip duplicate question about %s", sorted(qkeys))
                self.stats["duplicates"] = self.stats.get("duplicates", 0) + 1
                was = self._asked.get(dup)
                rep = Asked(qid=new_id("q"), aid=dup, text=q, who=who, policy=policy,
                            kind=kind or "question", via="repeat", t_q=t_q,
                            utt=(utt[1] if utt else None), line=line,
                            status=(was.status if was is not None else "answered"))
                await self.emit(rep.msg())
                return
        aid = aid or new_id("a")
        recent.append((t_q, qkeys, q, aid))
        del recent[:-20]
        utt_no = utt[1] if utt else None
        if utt and utt[1] is not None:
            self._utt_answers[utt] = aid
            if len(self._utt_answers) > 200:
                for k in list(self._utt_answers)[:100]:
                    del self._utt_answers[k]
        registered = await self._register(aid, q, who, policy, kind, via, utt_no, line, t_q, sure)
        if hits and not registered.hits:
            registered.hits = list(hits)[:4]
        if policy == "heard":
            return                   # registered, and no card is coming: the page shows it was heard
        card_utt = utt_no if who == "Them" else None    # `utt` on a card is always the client's counter
        if policy == "check":
            warm = None
        if warm is not None and getattr(warm, "tentative", False):
            warm.tentative = False
            ok = await self._warm_confirmed(warm, q)
            if not ok:
                warm = None
        if policy != "check" and not (warm and warm.answer):
            inst = self._instant_from_cache(q, aid, source, hits)
            if inst is not None:
                warm = inst          # a cached answer beats a candidate that is still warming
        shown_text: str | None = None
        if warm and warm.answer:
            self.stats["cache_hits"] += 1
            a = replace(warm.answer, id=aid, question=q, source="prefetch",
                        latency_ms=int((time.time() - t_q) * 1000), ts=time.time(), utt=card_utt,
                        who=who, line=line)
            self.answers[aid] = a
            warm.status = "shown"
            # a card the verifier dropped never reached the page: the live call must not treat
            # it as on screen (it used to hold the partials back and skip an identical answer)
            if await self._emit_answer(a):
                shown_text = a.answer
            await self.emit(warm.msg())
        elif warm:
            warm.answer_id = aid   # when it lands it will render as an answer
            warm.asked = q         # ... if the rule in _late_draft_hold still lets it
        live = asyncio.create_task(self._run_live(aid, q, hits, t_q, source, shown_text, warm, utt_no, ask,
                                                  who=who, policy=policy, line=line))
        asked = self._asked.get(aid)
        if asked is not None:
            asked.task = live

    # ------------------------------------------------------------------ registration (E68)

    def _repeat_of(self, q: str, qkeys: frozenset, t_q: float) -> str | None:
        """The card id of a question asked in the last 20 s that this one repeats, or None.

        Topic keys alone used to decide it, and E17's only duplicate skip ever (sample-03 t10, all
        four runs) was a different question about the same region: "how does that APAC figure
        compare to what our closest competitor booked" after "walk me through what APAC brought
        in". The words have to say the same thing too.
        """
        recent = self.__dict__.get("_recent_questions") or []
        for entry in reversed(recent):
            if len(entry) < 4:
                continue
            ts, keys, text, aid = entry
            if t_q - ts >= 20 or not qkeys <= keys:
                continue
            v = self._embed([q, text])
            if v is not None:
                if float(v[0] @ v[1]) >= DUPLICATE_SIM:
                    return aid
            elif _word_set(q) == _word_set(text):
                return aid
        return None

    async def _register(self, aid: str, q: str, who: str, policy: str, kind: str | None, via: str,
                        utt: int | None, line: str | None, t_q: float, sure: bool = True) -> Asked:
        """Record a question the brain is acting on and tell the page, before any answer exists.

        One registration per card id: the same question re-routed with more words (a joined
        final, a settled final that moved) updates the text and keeps its clock, so its no-answer
        deadline is still counted from when it was first heard.
        """
        e = self._asked.get(aid)
        if e is not None:
            e.text = q
            if utt is not None and utt not in e.utts:
                e.utts.append(utt)
            await self.emit(e.msg())
            return e
        e = Asked(qid="q" + aid[1:], aid=aid, text=q, who=who, policy=policy, kind=kind or "question",
                  via=via, t_q=t_q, utt=utt, line=line, sure=sure, utts=[utt] if utt is not None else [])
        self._asked[aid] = e
        if len(self._asked) > 60:
            for old in sorted(self._asked.values(), key=lambda x: x.t_q)[:len(self._asked) - 60]:
                if old.watch and not old.watch.done():
                    old.watch.cancel()
                self._asked.pop(old.aid, None)
        counts = self.stats.setdefault("questions", {})
        key = "typed" if via == "typed" else who            # a typed ask is counted apart from speech
        counts[key] = counts.get(key, 0) + 1
        log.info("question registered: %s %s %s via %s %r", who, policy, e.kind, via, q[:60])
        await self.emit(e.msg())
        if policy in ("answer", "check") and via != "typed" and e.kind != "objection" and sure:
            # a typed ask has the page's own pending spinner; an objection is not a question and
            # a pushback that meets no card has lost nothing (E68: on the E17 frames the deadline
            # card's only noise was two such lines, a tag statement and a cut-off, both read as
            # objections); everything else gets a deadline
            e.watch = asyncio.create_task(self._no_answer_watch(e))
        return e

    async def _shown(self, aid: str) -> None:
        """A card for `aid` reached the page, partial or settled: the question it answers is
        answered, whatever came before it (a no-answer card included)."""
        e = self._asked.get(aid)
        if e is None or e.shown:
            return
        e.shown = True
        if e.watch and not e.watch.done():
            e.watch.cancel()
        e.status = "answered"
        await self.emit(e.msg())

    async def _no_answer_watch(self, e: Asked) -> None:
        try:
            await asyncio.sleep(NO_ANSWER_S)
            # a call that already ended with nothing has put up its own card, with the real reason
            if not e.shown and e.status != "no-answer" and self._asked.get(e.aid) is e:
                await self._no_answer(e, "slow")
            await asyncio.sleep(max(0.0, LIVE_TIMEOUT_S - NO_ANSWER_S))
        except asyncio.CancelledError:
            return
        if e.shown or self._asked.get(e.aid) is not e:
            return
        if e.task is not None and not e.task.done():
            e.task.cancel()                  # `_hedged` cancels its model calls on the way out
            log.info("live answer given up after %.0f s: %r", LIVE_TIMEOUT_S, e.text[:60])
        await self._no_answer(e, "timeout")

    _NO_ANSWER_TEXT = {
        "slow": ("Couldn't answer: nothing came back in {s:g} s. Still looking.", "No answer yet"),
        "error": ("Couldn't answer: {detail}.", "Couldn't answer"),
        "empty": ("Couldn't answer: the answer came back empty.", "Couldn't answer"),
        "timeout": ("Couldn't answer: no answer in {t:g} s.", "Couldn't answer"),
        "none": ("Not in the files.", "Not in the files"),
    }

    async def _no_answer(self, e: Asked, reason: str, detail: str = "") -> None:
        """The card a registered question gets instead of silence (E68).

        Same id as the answer it stands in for, so the answer replaces it in place whenever it
        lands, which is what `answer` frames already do for a redraw. `answered_from="none"` is
        what the page already draws as NOT IN FILES; `no_answer` says why, so a page can word a
        slow answer differently from an empty one. Emitted directly, never through
        `_emit_answer`: the verifier has no number to check on it and must not be the thing that
        drops it, and a late draft the hold keeps back never touches it.

        Only for a question the brain owes an answer (`policy == "answer"`): a consultant's check
        the files cannot answer ends quiet, which is the point of a check. And not for an
        objection, which registers (the page shows it was heard) but asked nothing.
        """
        if e.shown or e.policy != "answer" or e.kind == "objection" or not e.sure:
            return
        if reason not in ("slow", "timeout") and e.watch and not e.watch.done():
            e.watch.cancel()             # the call is over: "still looking" would no longer be true
        after_ms = int((time.time() - e.t_q) * 1000)
        text, say = self._NO_ANSWER_TEXT.get(reason, self._NO_ANSWER_TEXT["error"])
        text = text.format(s=NO_ANSWER_S, t=LIVE_TIMEOUT_S, detail=(detail or "the answer call failed").rstrip("."))
        a = Answer(id=e.aid, question=e.text, answer=text, figure=None, citations=[],
                   source="manual" if e.via == "typed" else "live", latency_ms=after_ms, in_files=False,
                   ts=time.time(), model="", answered_from="none", mode=self.mode.id, say=say,
                   utt=e.utt if e.who == "Them" else None, who=e.who, line=e.line,
                   verdict={"ok": True, "level": "none", "problems": [], "reason": text},
                   no_answer={"reason": reason, "after_ms": after_ms, "detail": (detail or "")[:160]})
        first = e.status != "no-answer"
        self.answers[e.aid] = a
        e.status = "no-answer"
        if first:
            self.stats["no_answer"] = self.stats.get("no_answer", 0) + 1
        why = self.stats.setdefault("no_answer_reasons", {})
        why[reason] = why.get(reason, 0) + 1
        log.info("no answer: %s after %d ms for %s %r", reason, after_ms, e.who, e.text[:60])
        await self.emit(a.msg())
        await self.emit(e.msg())

    async def _warm_confirmed(self, warm: Candidate, q: str) -> bool:
        """Does this warm card answer the question that landed? One typed noul for the cosine band
        where the embedding alone is a coin flip (56% precision at 0.45, E7). Local since E21a:
        98.8% precision and 97.6% recall in 28 ms on the E21 checkpoint, against 94.5 / 72.9 at
        220 ms on Jev, so the card no longer waits on a round trip to be shown or dropped."""
        dec = getattr(self, "intent_decider", None)
        if dec is None or intent is None or not warm.question_guess:
            return False
        try:
            v = await intent.card_answers(dec, warm.question_guess, q)
            log.info("warm card check: %s p=%.2f %dms for %r", v.answers, v.p, v.latency_ms, warm.question_guess[:50])
            return bool(v.answers) and v.p >= 0.5
        except Exception as e:  # noqa: BLE001
            log.info("warm card check failed: %s", e)
            return False

    def _instant_from_cache(self, q: str, aid: str, source: str, hits: list | None = None) -> Candidate | None:
        """A precomputed Q&A entry close enough to the question renders at once (the live call still
        runs and replaces it if it disagrees). Returns a synthetic ready candidate or None."""
        fn = getattr(self.index, "qa_matches", None)
        if fn is None:
            return None
        try:
            matches = fn(q, k=5)
        except Exception as e:  # noqa: BLE001
            log.debug("qa_matches failed: %s", e)
            return None
        # a hypothetical is answered only by a derived entry at the value the asker named; a plain
        # question never by a derived one (the grid answers "what if", not "what is")
        if _anaphoric(q):
            return None          # "and what does that come to a month?" needs the transcript, not a lookup
        hypothetical = whatif_ish(q)
        asked = _numbers_in(q) if hypothetical else set()
        eligible = []
        for m in matches:
            if bool(getattr(m, "derived", False)) != hypothetical:
                continue
            if hypothetical:
                lv = getattr(m, "lever_value", None)
                if lv is None or not any(_same_value(lv, a) for a in asked):
                    continue
            eligible.append(m)
        if not eligible:
            return None
        m = eligible[0]
        score = float(getattr(m, "score", 0.0))
        # a topic hit pointing at the same file as the cached entry is corroboration worth a tenth
        files = {str((t or {}).get("file", "")).lower() for h in (hits or []) for t in (h.targets or [])}
        if files and str(getattr(m, "file", "")).lower() in files:
            score += 0.10
        if score < self.settings.get("qa_threshold", 0.60):
            return None
        self.stats["qa_hits"] = self.stats.get("qa_hits", 0) + 1
        derived = bool(getattr(m, "derived", False))
        a = Answer(id=aid, question=getattr(m, "question", q), answer=getattr(m, "answer", ""), figure=getattr(m, "figure", None),
                   citations=list(getattr(m, "citations", []) or []), source="prefetch", latency_ms=0, in_files=True,
                   ts=time.time(), model="cache", topic=getattr(m, "loc", None),
                   answered_from="derived" if derived else "files",
                   mode=self.mode.id, say=getattr(m, "say", "") or "", cached=True,
                   working=(getattr(m, "working", "") or None), derived=derived)
        c = Candidate(id=new_id("c"), topic=f"cache · {getattr(m, 'loc', '')}", keys=frozenset(), status="ready",
                      created=time.time(), question_guess=a.question, answer=a)
        log.info("instant answer from Q&A cache (%.2f): %s", float(getattr(m, "score", 0.0)), a.question[:60])
        return c

    async def _run_live(self, aid: str, q: str, hits: list, t_q: float, source: str,
                        shown_text: str | None, warm: Candidate | None, utt_no: int | None = None,
                        ask: Any = None, who: str = "Them", policy: str = "answer",
                        line: str | None = None) -> None:
        """Files first, always. A manual ask may use the web in the same pass (it is deliberate and not
        latency-critical); a live question escalates to a second, web-enabled pass only when the
        files-only answer came back empty and web_mode is auto.

        `ask` is the typed ask the gate verdict carried (E38); a typed question has no gate verdict
        to ride in and extracts its own, which costs a round trip nobody is listening through.

        A line that points rather than names ("what's this number based on") has its metric filled
        from the page on screen before the evidence is built (E43a), and nothing happens when no
        page is known.

        E68: a question that registered is never left with nothing on screen. An empty answer is
        not drawn as a blank card; a call that ends with nothing shown, or dies, raises the
        no-answer card (`_no_answer`) where the old code drew an "Error:" card claiming to come
        from the files. A consultant's `check` streams nothing and shows only an answer from the
        files that passes the verifier, and escalates to nobody."""
        self.stats["live"] += 1
        check = policy == "check"
        card_utt = utt_no if who == "Them" else None
        if ask is None and source == "manual":
            ask = await self._extract_ask(q)
        ask = self._screen_ask(q, ask)
        evidence = self._evidence(hits, q, ask)
        pre_block, prerun = await self._prerun_whatif(q, hits)
        if pre_block:
            evidence = pre_block + "\n\n" + evidence
        final_shown = shown_text is not None
        web_auto = self.settings.get("web_mode", "auto") == "auto"
        if web_auto and source == "manual":
            trigger = (f"QUESTION (typed by the consultant): \"{q}\". Answer from the files first. If the files do "
                       "not contain it and it is a public fact, use the web tools now and cite the pages.")
        elif check:
            trigger = (f"QUESTION (the consultant asked the client this, final): \"{q}\". Answer only when the "
                       "files state the specific fact asked for, so the consultant can check the client's reply "
                       "against it. The client's own views, plans, expectations or preferences are not in the "
                       "files: then answered_from=\"none\".")
        elif who == "Me":
            trigger = f"QUESTION (asked aloud by the consultant, final): \"{q}\". Answer exactly this question from the files."
        else:
            trigger = f"QUESTION (final): \"{q}\". Answer exactly this question from the files."
        if prerun:
            # `working` is optional in the schema now (E42), so the call that needs it asks for it
            # here: the trigger is user text, so this costs no cache prefix.
            trigger += (" The what-if has already been run; phrase its result (no tools are offered), "
                        "with the lever change in `working`.")
        last: dict = {}

        async def on_partial(text: str, model: str, more: dict | None = None) -> None:
            # stream words onto the screen until any complete answer has landed. `more` carries
            # whatever else has closed in the streamed JSON: since E52 that is `figure`, which the
            # schema writes third and which used to wait for the settled frame (docs/ui-protocol.md).
            if final_shown or not text or check:
                return                   # a check shows nothing until the files are known to hold it
            await self.emit({"type": "answer", "id": aid, "question": q, "answer": text,
                             "figure": (more or {}).get("figure"),
                             "citations": [], "source": source, "latency_ms": int((time.time() - t_q) * 1000),
                             "in_files": True, "answered_from": "files", "web_citations": [], "mode": self.mode.id,
                             "utt": card_utt, "ts": time.time(), "model": model, "topic": None, "partial": True,
                             "working": None, "derived": False, "who": who, "line": line})
            await self._shown(aid)

        async def on_result(data: dict, model: str, ms: int, is_primary: bool) -> None:
            nonlocal shown_text, final_shown
            last.clear(); last.update(data)
            answer_text = data.get("answer", "") or ""
            from_ = data.get("answered_from") or ("files" if data.get("in_files") else "none")
            if not answer_text.strip():
                # E68: never a blank card. What the page gets instead is decided when the call is
                # over, because the other model may still land something worth showing
                log.info("empty answer from %s for %r", model, q[:60])
                return
            if check and (from_ not in ("files", "derived") or prerun and not data.get("figure")):
                log.info("check: not in the files (%s) for %r", from_, q[:60])
                return
            final_shown = True
            if shown_text and answer_text.strip() == shown_text.strip():
                return   # prefetched answer confirmed, nothing to redraw
            if prerun and from_ in ("files", "model"):
                from_ = "derived"          # the figure came out of the pre-run recalculation
                data.setdefault("working", None)
                if not data.get("working"):
                    data["working"] = self._working_from_last() or None
            a = Answer(
                id=aid, question=q, answer=answer_text, figure=data.get("figure"),
                citations=data.get("citations", []), source=source,
                latency_ms=int((time.time() - t_q) * 1000), in_files=bool(data.get("in_files", False)),
                ts=time.time(), model=model, topic=warm.topic if warm else None,
                answered_from=from_,
                web_citations=data.get("web_citations") or [], mode=self.mode.id, say=data.get("say") or "",
                utt=card_utt, working=data.get("working") or None, derived=(from_ == "derived"),
                who=who, line=line,
            )
            self.answers[aid] = a
            if await self._emit_answer(a, strict=check):
                shown_text = answer_text

        try:
            await self._hedged(trigger, evidence, q, on_result, on_partial, web=(web_auto and source == "manual"),
                               live=True, tools=([] if prerun else None),
                               derived=bool(prerun) or whatif_ish(q) or source == "manual")
            if web_auto and source != "manual" and not check and last \
                    and (last.get("answered_from") or "none") == "none" and not self.mode.model_knowledge:
                self.stats["web_escalations"] = self.stats.get("web_escalations", 0) + 1
                trigger2 = (f"QUESTION (final): \"{q}\". The files were searched and do not contain the answer. "
                            "Web search is now available: answer from public sources if this is a public fact, "
                            "with citations; otherwise keep answered_from=\"none\".")
                await self._hedged(trigger2, evidence, q, on_result, None, web=True, hedge=False, live=True,
                                   derived=whatif_ish(q))
        except Exception as e:  # noqa: BLE001
            self.stats["errors"] += 1
            log.exception("live answer failed: %s", e)
            asked = self._asked.get(aid)
            if not shown_text and asked is not None:
                await self._no_answer(asked, "error", f"{type(e).__name__}: {str(e)[:100]}")
            return
        # E68: the call is over. A question that still has nothing on screen is told so now rather
        # than waiting out the deadline; a check the files could not answer ends quiet
        asked = self._asked.get(aid)
        if asked is not None and not asked.shown:
            if asked.policy == "answer":
                # "none": the model searched and said so with nothing to show for it; "empty":
                # nothing came back at all, or an answer with no words in it
                said_none = bool(last) and (last.get("answered_from") or "none") == "none"
                await self._no_answer(asked, "none" if said_none else "empty")
            elif asked.status == "heard":
                asked.status = "quiet"
                await self.emit(asked.msg())

    # ------------------------------------------------------------------ LLM calls

    async def _hedged(self, trigger: str, evidence: str, current: str,
                      on_result: Callable[[dict, str, int, bool], Awaitable[None]],
                      on_partial: Callable[[str, str], Awaitable[None]] | None = None,
                      web: bool = False, hedge: bool = True, live: bool = False,
                      tools: list | None = None, speculative: bool = False, derived: bool = False,
                      force_llm: LLM | None = None) -> None:
        """Fire the primary model and (optionally) a faster hedge model in parallel.
        The first result renders; the primary's result replaces it if it arrives later.
        `tools=[]` answers from the evidence alone (a pre-run what-if): one round, no exploring.
        `force_llm` pins the provider (the fallback path); no further fallback runs from it."""
        llm = force_llm or (self.llm_live if (live and self.llm_live) else self.llm)
        primary = llm.answer_model
        models = [primary]
        if hedge and llm.hedge_model and llm.hedge_model != primary:
            models.append(llm.hedge_model)
        delay = float(self.settings.get("hedge_delay_s") or 0)
        # E52: which queue this call joins at the provider. A live answer and its hedge are both
        # "live": somebody asked the question and is waiting for the card. The hedge only fires on
        # the tail anyway, so the pair costs the reserve at most once per question.
        lane = "live" if live else ("draft" if speculative else "bg")

        async def run(model: str) -> tuple[dict, int]:
            # A hedge fired at t=0 doubles the cost of every answer and races the primary for the
            # session pool. Held back by `hedge_delay_s`, it only fires on the tail: the primary
            # cancels a still-sleeping hedge before it ever becomes a request (E39).
            if delay > 0 and model != primary:
                await asyncio.sleep(delay)
            return await self._llm_answer(model, trigger, evidence, current, on_partial, web, llm,
                                          tools=tools, speculative=speculative, derived=derived,
                                          lane=lane)

        tasks = {asyncio.create_task(run(m)): m for m in models}
        primary_task = next(t for t, m in tasks.items() if m == primary)
        pending = set(tasks)
        errors: list[Exception] = []
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    model = tasks[t]
                    try:
                        data, ms = t.result()
                    except asyncio.CancelledError:
                        continue
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)
                        log.warning("%s failed: %s", model, e)
                        continue
                    await on_result(data, model, ms, t is primary_task)
                    if t is primary_task:
                        for p in pending:
                            p.cancel()
                        pending = set()
        finally:
            # E52. `asyncio.wait` does not cancel what it was waiting on, so cancelling a
            # speculative draft used to leave its model call running to the end, still holding the
            # session slot the cancellation was meant to give back. Every path leaves here with no
            # request of ours outstanding.
            for t in tasks:
                if not t.done():
                    t.cancel()
        if errors and len(errors) == len(models):
            if all(isinstance(e, DraftDropped) for e in errors):
                # E52: the pool dropped this draft so a live answer could go first. That is a
                # decision, not an outage, so no retry and no second provider.
                raise errors[0]
            # the live provider (an API key by default) can die mid-call: no credit, a revoked key,
            # an outage. One retry on the main provider keeps the card coming; a credit or auth
            # error also flips routing so the next question does not pay the same failure.
            if force_llm is not None:
                raise errors[0]
            if live and llm is self.llm_live and self.llm_live is not self.llm:
                text = " ".join(str(e) for e in errors).lower()
                if any(w in text for w in ("credit", "billing", "authentication", "api key", "401", "403")):
                    log.warning("live provider %s failed (%s); routing live answers to %s until fixed",
                                llm.provider, str(errors[0])[:120], self.llm.provider)
                    self.llm_live = None
                else:
                    log.warning("live provider %s failed (%s); retrying once on %s",
                                llm.provider, str(errors[0])[:120], self.llm.provider)
                await self._hedged(trigger, evidence, current, on_result, on_partial, web=web, hedge=hedge,
                                   live=False, tools=tools, speculative=speculative, derived=derived)
                return
            fb = self._fallback_llm(exclude=llm)
            if fb is not None:
                # the subscription itself failed (login expired, outage, rate limit): one answer on
                # the fallback provider, which Jackson allowed as second choice (Gemini key in .env)
                log.warning("%s failed (%s); answering on %s", llm.provider, str(errors[0])[:120], fb.provider)
                await self._hedged(trigger, evidence, current, on_result, on_partial, web=False, hedge=False,
                                   live=False, tools=tools, speculative=speculative, derived=derived, force_llm=fb)
                return
            raise errors[0]

    def _fallback_llm(self, exclude: LLM | None = None) -> LLM | None:
        """The provider to use when the main one fails: Gemini when its key is set, never the
        Anthropic API (its credit is Jackson's call), never the provider that just failed."""
        cached = getattr(self, "_fallback", None)
        if cached is None:
            try:
                from presage.llm.gemini import _env_key as gemini_key
                if gemini_key():
                    cached = make_llm("gemini", models=ENV_MODELS)
            except Exception as e:  # noqa: BLE001
                log.debug("fallback provider unavailable: %s", e)
            self._fallback = cached or False
        if not cached:
            return None
        if exclude is not None and getattr(exclude, "provider", "") == cached.provider:
            return None
        return cached

    async def classify(self, prompt: str, schema: dict, model: str | None = None) -> dict:
        """Structured JSON from the main provider, on the fallback provider when the main one fails."""
        try:
            return await self.llm.classify(prompt, schema, model=model)
        except TypeError:
            return await self.llm.classify(prompt, schema)
        except Exception as e:  # noqa: BLE001
            fb = self._fallback_llm(exclude=self.llm)
            if fb is None:
                raise
            log.warning("%s classify failed (%s); using %s", self.llm.provider, str(e)[:120], fb.provider)
            try:
                return await fb.classify(prompt, schema, model=None)
            except TypeError:
                return await fb.classify(prompt, schema)

    async def _llm_answer(self, model: str, trigger: str, evidence: str, current: str,
                          on_partial: Callable[[str, str], Awaitable[None]] | None = None,
                          web: bool = False, llm: LLM | None = None,
                          tools: list | None = None, speculative: bool = False,
                          derived: bool = False, lane: str = "bg") -> tuple[dict, int]:
        """One model's answer: the provider runs the tool loop and streams `answer` as it grows."""
        llm = llm or self.llm
        toolset = TOOLS if tools is None else tools
        schema = answer_schema(speculative=speculative, web=web, derived=derived)
        t0 = time.time()
        used: set[str] = set()          # scratch tools that returned a result in this call
        calls = [0]                     # every provider gets the same budget: three calls, then answer

        def run_tool(name: str, args: Any):
            calls[0] += 1
            if calls[0] > TOOL_BUDGET:
                return json.dumps({"error": "tool budget spent: answer now from the results you already have"})
            out = self._run_tool(name, args)
            if name not in SCRATCH_TOOL_NAMES:
                return out
            if isinstance(out, str):
                if not out.startswith('{"error"'):
                    used.add(name)
                return out

            async def tracked():
                r = await out
                if isinstance(r, str) and not r.startswith('{"error"'):
                    used.add(name)
                return r
            return tracked()

        kwargs = dict(model=model, system=self.system_text(), user=self._user_prompt(trigger, evidence, current),
                      tools=toolset, schema=schema, run_tool=run_tool, on_partial=on_partial,
                      lane=lane)
        async with self._sem:
            if web:
                try:
                    data, rec = await llm.answer(**kwargs, web=True)
                except TypeError as e:   # provider without web support yet
                    log.warning("%s has no web support (%s); answering from files only", llm.provider, e)
                    data, rec = await llm.answer(**kwargs)
            else:
                data, rec = await llm.answer(**kwargs)
        rec["web"] = web
        self._account(rec)
        data = complete_answer(data)
        if web and (data.get("answered_from") == "web") and not data.get("web_citations"):
            data["web_citations"] = list(rec.get("web_urls") or [])[:5]
        if used and data.get("answer") and data.get("answered_from") in ("files", "model", None):
            # the number came out of the scratch tools whatever the model labelled it (the hedge
            # model sometimes says "model"); the page must show it as derived, with its working
            data["answered_from"] = "derived"
            if not data.get("working"):
                data["working"] = self._working_from_last() or None
        return data, int((time.time() - t0) * 1000)

    def _working_from_last(self) -> str:
        """A working line rebuilt from the last what-if when the model left `working` empty."""
        w = getattr(self, "_last_whatif", None) or {}
        ch = w.get("changes") or []
        if not ch:
            return ""
        parts = [f"{c.get('cell')} {c.get('was_s', c.get('was'))} → {c.get('now_s', c.get('now'))}" for c in ch[:2]]
        line = ", ".join(parts) + ", model recalculated"
        outs = w.get("outputs") or []
        if outs:
            o = outs[0]
            line += f": {o.get('label')} {o.get('was_s', o.get('was'))} → {o.get('now_s', o.get('now'))}"
        return line

    def _account(self, rec: dict) -> None:
        """One record per answer (model, timings, tokens, tool rounds) for the stats panel and
        for docs/research-llm.md."""
        self.stats["cache_read_tokens"] += rec.get("cache_read", 0) or 0
        self.stats["cache_write_tokens"] += rec.get("cache_write", 0) or 0
        pool = getattr(self.llm, "pool_stats", None)       # E52: the lane's own counters
        if pool is not None:
            try:
                self.stats["pool"] = pool()
            except Exception as e:  # noqa: BLE001
                log.debug("pool stats unavailable: %s", e)
        reqs = self._recent_requests()
        reqs.append(rec)
        del reqs[:-500]
        log.info("llm %s", json.dumps(rec, default=str))

    def _run_tool(self, name: str, args: Any) -> Any:
        """Dispatch one tool call. Returns the JSON string the provider sends back, except for the
        two scratch-paper tools when this runs on the event loop: `whatif` and `calc` are sync and
        CPU-bound (a workbook's first compile costs about 2 s), so they come back as an awaitable
        that runs them in a thread. Providers await a result that is awaitable; claude_code already
        calls this inside `asyncio.to_thread`, where there is no loop and nothing changes."""
        args = args if isinstance(args, dict) else {}
        if name in SCRATCH_TOOL_NAMES:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return self._run_scratch_tool(name, args)
            return asyncio.to_thread(self._run_scratch_tool, name, args)
        try:
            if name == "sql":
                return json.dumps(self.index.sql(str(args.get("query", "")), limit=50), default=str)[:8000]
            if name == "search":
                chunks = self.index.search_text(str(args.get("query", "")), k=6)
                return json.dumps([{"file": c.file, "loc": c.loc, "text": c.text[:800]} for c in chunks])[:8000]
            if name == "find_cells":
                return self._tool_find_cells(str(args.get("pattern", "")))
            if name == "read":
                return self._tool_read(str(args.get("file", "")), str(args.get("loc", "")))
            if name == "list_files":
                return self._tool_list_files()
            if name == "find_numbers":
                fn = getattr(self.index, "find_numbers", None)
                if fn is None:
                    return json.dumps({"error": "numbers index not built yet"})
                rows = fn(str(args.get("pattern", "")), limit=40)
                return json.dumps({"hits": len(rows), "numbers": rows}, default=str)[:8000]
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        return json.dumps({"error": f"unknown tool {name}"})

    def _resolve_cell(self, key: str) -> str:
        """A "Sheet!A1" reference as given, or a lever's spoken name resolved to its cell."""
        k = str(key).strip().strip("'")
        if re.match(r"^[^!]+![A-Za-z]{1,3}\d{1,7}$", k):
            return k
        best = self._lever_matches(k, k=1)             # "subscriber churn" -> Assumptions!C39
        if not best:
            raise ValueError(f"no lever matches {key!r}; pick one from the LEVERS table")
        lv = best[0][1]
        return f"{lv.get('sheet')}!{lv.get('cell')}"

    def _resolve_sets(self, sets: dict) -> dict:
        return {self._resolve_cell(k): v for k, v in (sets or {}).items()}

    def _run_scratch_tool(self, name: str, args: dict) -> str:
        """`whatif` / `calc` against the Index. Blocking and CPU-bound: never on the event loop."""
        fn = getattr(self.index, name, None)
        if fn is None:
            return json.dumps({"error": "scratch paper not built yet"})
        try:
            if name == "calc":
                return json.dumps(fn(str(args.get("expr", "")), cell_map(args.get("inputs"))), default=str)[:8000]
            if name == "goalseek":
                out = fn(args.get("file") or None, str(args.get("target", "")), str(args.get("value", "")),
                         self._resolve_cell(str(args.get("lever", ""))))
                return json.dumps(out, default=str)[:8000]
            if name == "explain":
                out = fn(args.get("file") or None, str(args.get("cell", "")),
                         self._resolve_sets(cell_map(args.get("sets"))) or None)
                return json.dumps(out, default=str)[:8000]
            if name == "compare":
                cases, labels = [], []
                for c in args.get("cases") or []:
                    if isinstance(c, dict):
                        cases.append(self._resolve_sets(cell_map(c.get("sets"))))
                        labels.append(str(c.get("label", "")) or f"case {len(cases)}")
                watch = [str(c) for c in (args.get("watch") or []) if str(c).strip()] or None
                out = fn(args.get("file") or None, cases, watch, labels)
                return json.dumps(out, default=str)[:8000]
            if name == "sweep":
                watch = [str(c) for c in (args.get("watch") or []) if str(c).strip()] or None
                out = fn(args.get("file") or None, self._resolve_cell(str(args.get("cell", ""))),
                         [str(v) for v in (args.get("values") or [])], watch)
                return json.dumps(out, default=str)[:8000]
            sets = cell_map(args.get("sets"))
            if not sets:
                return json.dumps({"error": 'sets is empty; give [{"cell": "Sheet!A1", "value": "1%"}] '
                                            'with a cell from the LEVERS table, or the lever\'s spoken name'})
            sets = self._resolve_sets(sets)
            watch = [str(c) for c in (args.get("watch") or []) if str(c).strip()] or None
            out = fn(args.get("file") or None, sets, watch)
            if isinstance(out, dict) and "error" not in out:
                self._last_whatif = {"file": out.get("file") or args.get("file"), "sets": sets,
                                     "changes": list(out.get("changes") or []),
                                     "outputs": list(out.get("outputs") or []),
                                     "partial": list(out.get("partial") or []), "ts": time.time()}
            return json.dumps(out, default=str)[:8000]
        except ValueError as e:
            return json.dumps({"error": str(e)[:300]})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _tool_find_cells(self, pattern: str) -> str:
        pat = pattern.strip().lower().replace("'", "''").replace("%", "")
        if len(pat) < 2:
            return json.dumps({"error": "pattern too short"})
        like = f"%{pat}%"
        rows = self.index.sql(
            "SELECT file, sheet, cell_ref, header, row_label, COALESCE(value_num, value_text) AS value "
            f"FROM cells WHERE lower(row_label) LIKE '{like}' OR lower(header) LIKE '{like}' "
            f"OR lower(value_text) LIKE '{like}' ORDER BY file, sheet, row, col", limit=40)
        rows = [r for r in rows if "error" not in r] if rows else []
        if rows and "error" in (rows[0] if rows else {}):
            return json.dumps(rows[0])
        return json.dumps({"hits": len(rows), "cells": rows}, default=str)[:8000]

    def _tool_read(self, file: str, loc: str) -> str:
        fl, ll = file.strip().lower(), loc.strip().lower()
        m = re.search(r"(page|slide|section)\s*(\d+)", ll)
        want_unit = int(m.group(2)) if m else None
        chunks = [c for c in getattr(self.index, "chunks", []) if fl and fl in c.file.lower()]
        if chunks:
            picked = [c for c in chunks if (want_unit is not None and c.unit == want_unit)
                      or (ll and (ll in c.loc.lower() or ll in (c.heading or "").lower()))]
            if not picked and want_unit is None and ll:
                picked = [c for c in chunks if any(w in c.text.lower() for w in ll.split()[:3])][:3]
            if picked:
                text = "\n\n".join(f"[{c.file} › {c.loc}]\n{c.text}" for c in picked[:6])
                return json.dumps({"file": picked[0].file, "units": len(picked), "text": text[:7000]})
        tables = self.index.sql("SELECT name, file, sheet, n_rows FROM _tables", limit=200)
        tables = [t for t in tables if "error" not in t and fl in str(t["file"]).lower()]
        hit = [t for t in tables if ll and ll in str(t["sheet"] or "").lower()] or \
              [t for t in tables if ll and ll.replace(" ", "_") in t["name"]]
        if hit:
            t = hit[0]
            cols = self._columns(t["name"])
            label_of = {c["name"]: (c["label"] or c["name"]) for c in cols}
            rows = self.index.sql(f'SELECT * FROM "{t["name"]}"', limit=60)
            out = []
            for r in rows:
                if "error" in r:
                    return json.dumps(r)
                out.append({"_row": r.get("_row"), **{label_of.get(k, k): v for k, v in r.items()
                                                       if not k.startswith("_") and v is not None}})
            return json.dumps({"file": t["file"], "sheet": t["sheet"], "table": t["name"], "rows": out}, default=str)[:8000]
        fq, lq = fl.replace("'", "''"), ll.replace("'", "''")
        cells = self.index.sql(
            "SELECT sheet, cell_ref, header, row_label, COALESCE(value_num, value_text) AS value FROM cells "
            f"WHERE lower(file) LIKE '%{fq}%' AND lower(sheet) LIKE '%{lq}%' "
            "ORDER BY row, col", limit=120)
        if cells and "error" not in cells[0]:
            return json.dumps({"file": file, "sheet": loc, "cells": cells}, default=str)[:8000]
        return json.dumps({"error": f"nothing found for {file!r} at {loc!r}; try list_files"})

    def _tool_list_files(self) -> str:
        files = self.index.sql("SELECT path, kind, n_tables, n_chunks, error FROM _files ORDER BY path", limit=200)
        tables = self.index.sql("SELECT file, sheet, name, title, n_rows FROM _tables ORDER BY file, sheet", limit=500)
        docs = self.index.sql("SELECT file, kind, n_units, unit FROM _documents", limit=200)
        by_file: dict[str, dict] = {}
        for f in files:
            if "error" in f and len(f) == 1:
                return json.dumps(f)
            by_file[f["path"]] = {"file": f["path"], "kind": f["kind"], "sheets": [], "units": None, "error": f.get("error")}
        for t in tables:
            if "error" in t:
                continue
            by_file.setdefault(t["file"], {"file": t["file"], "kind": "xlsx", "sheets": [], "units": None, "error": None})
            by_file[t["file"]]["sheets"].append({"sheet": t["sheet"], "table": t["name"], "rows": t["n_rows"],
                                                 "about": (t.get("title") or "")[:120]})
        for d in docs:
            if "error" in d:
                continue
            if d["file"] in by_file:
                by_file[d["file"]]["units"] = f'{d["n_units"]} {d["unit"]}s'
        headings: dict[str, list[str]] = {}
        for c in getattr(self.index, "chunks", []):
            if c.heading and c.heading not in headings.setdefault(c.file, []):
                headings[c.file].append(c.heading)
        for name, info in by_file.items():
            if headings.get(name):
                info["headings"] = headings[name][:12]
        return json.dumps(list(by_file.values()), default=str)[:8000]


# ----------------------------------------------------------------------------- self test

WHATIF_Q = "what if we charged 1% of GMV under the GMV fee structure"


def _num(v: Any) -> float:
    """"1%" -> 0.01, "$4.2M" -> 4200000.0, "1.2x" -> 1.2. Only the fake index uses it."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace(",", "").replace("$", "").replace("x", "")
    mult = 1.0
    if s.endswith("%"):
        s, mult = s[:-1], 0.01
    elif s.endswith("percent"):
        s, mult = s[:-7], 0.01
    elif s.endswith("m"):
        s, mult = s[:-1], 1e6
    elif s.endswith("k"):
        s, mult = s[:-1], 1e3
    return float(s.strip()) * mult


@dataclass
class _FakeQA:
    """What `Index.qa_matches` returns once the grid ships: a derived entry."""
    question: str
    answer: str
    say: str = ""
    figure: str = ""
    citations: list = field(default_factory=list)
    file: str = ""
    loc: str = ""
    score: float = 0.0
    derived: bool = False
    working: str = ""


class _FakeIndex:
    """Stand-in for the Index while the ingest side is still being built (`--fake`).

    Only the scratch-paper surface is real, with the numbers measured on the big model in
    docs/superpowers/specs/2026-09-21-scratch-paper-design.md, so the tool path, the answer
    schema, the badge and the scratch emission can be exercised end to end against a live model.
    """
    FILE = "Plan_Model.xlsx"
    version = 1
    chunks: list = []
    BASE_RATE = 0.0085
    G57, B58 = 8_148_348.58, 22_142.25          # at the 0.85% in the workbook
    ASSETS = 2_604_971.00                        # Subscriber Fee Model!B57

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    # -- the parts brain reads for every answer ------------------------------
    def schema_summary(self) -> str:
        return (f"{self.FILE} (22 sheets, 3,468 formulas)\n"
                "  member_fee_model (Subscriber Fee Model): fee assumptions, subscriber volume, fee revenue\n"
                "  summary_scenarios (Summary Scenarios): five-year revenue by scenario\n")

    def summaries(self) -> str:
        return f"{self.FILE}: the client's subscription and fee model, five-year projections."

    def sql(self, query: str, limit: int = 50) -> list:
        return []

    def search_text(self, text: str, k: int = 3) -> list:
        return []

    def match_topics(self, text: str, limit: int = 12) -> list:
        return []                                # forces the lever-alias path

    def qa_matches(self, text: str, k: int = 3) -> list:
        if "1.25" not in text:
            return []                            # only the grid point is cached
        return [_FakeQA(question="What if the GMV fee were 1.25%?",
                        answer="At 1.25% year 5 subscriber revenue is $11.98M, up $3.84M on the 0.85% in the model.",
                        say="Eleven ninety-eight at one and a quarter", figure="$11.98M",
                        citations=[{"file": self.FILE, "loc": "Subscriber Fee Model › B48"},
                                   {"file": self.FILE, "loc": "Summary Scenarios › G57"}],
                        file=self.FILE, loc="Summary Scenarios › G57", score=0.82, derived=True,
                        working="Subscriber Fee Model B48 0.85% → 1.25%, model recalculated")]

    # -- scratch paper --------------------------------------------------------
    def levers(self, file: str | None = None) -> list[dict]:
        return [{"file": self.FILE, "sheet": "Subscriber Fee Model", "cell": "B48",
                 "label": "Annual fee as a share of gross volume", "header": "Assumption",
                 "value": self.BASE_RATE, "fmt": "pct",
                 "aliases": ["GMV fee", "platform fee", "GMV fee"],
                 "outputs": [{"cell": "Summary Scenarios!G57", "label": "Year 5 subscriber revenue"},
                             {"cell": "Subscriber Fee Model!B58", "label": "Fee revenue per subscriber"}]}]

    def levers_block(self, limit: int = 40) -> str:
        return ("== LEVERS (inputs whatif can change) ==\n"
                "Annual fee as a share of gross volume [GMV fee; platform fee] "
                f"({self.FILE} › Subscriber Fee Model!B48 = 0.85%) moves: Year 5 subscriber revenue "
                "(Summary Scenarios!G57), Fee revenue per subscriber (Subscriber Fee Model!B58)")

    def whatif(self, file: str | None, sets: dict, watch: list | None = None) -> dict:
        self.calls.append(("whatif", {"file": file, "sets": dict(sets), "watch": list(watch or [])}))
        time.sleep(0.06)                          # the measured 57 ms, so the thread hop is real
        cell, raw = next(iter(sets.items()))
        if "B48" not in str(cell).upper():
            raise ValueError(f"{cell} is not a lever; pick one from the LEVERS table")
        rate = _num(raw)
        exact = abs(rate - 0.01) < 1e-9
        k = rate / self.BASE_RATE
        outs = [("Summary Scenarios!G57", "Year 5 subscriber revenue", self.G57, 9_586_292.44 if exact else self.G57 * k),
                ("Subscriber Fee Model!B58", "Fee revenue per subscriber", self.B58, 26_049.71 if exact else self.B58 * k)]
        return {"file": file or self.FILE,
                "changes": [{"cell": "Subscriber Fee Model!B48",
                             "label": "Annual fee as a share of gross volume",
                             "was": "0.85%", "now": f"{rate * 100:g}%"}],
                "outputs": [{"cell": c, "label": lab, "was": round(was, 2), "now": round(now_, 2),
                             "delta": round(now_ - was, 2), "pct": round((now_ / was - 1) * 100, 2)}
                            for c, lab, was, now_ in outs],
                "partial": [], "ms": 57}

    def calc(self, expr: str, inputs: dict) -> dict:
        self.calls.append(("calc", {"expr": expr, "inputs": dict(inputs)}))
        time.sleep(0.01)
        known = {"Subscriber Fee Model!B57": (self.ASSETS, "Subscriber volume"),
                 "Subscriber Fee Model!B48": (self.BASE_RATE, "Annual fee as a share of gross volume")}
        vals, shown = {}, []
        for name, ref in inputs.items():
            key = str(ref)
            hit = known.get(key) or known.get("Subscriber Fee Model!" + key.split("!")[-1])
            if hit is None:
                try:
                    hit = (_num(ref), "from the asker")
                except ValueError:
                    return {"error": f"cannot resolve input {name!r} = {key!r}"}
            vals[name], label = hit[0], hit[1]
            shown.append({"name": name, "ref": key, "value": hit[0], "label": label})
        m = re.match(r"^\s*([a-z_]\w*)\s*([*/+-])\s*([a-z_]\w*|[\d.]+)\s*$", expr, re.I)
        if not m:
            return {"error": f"the fake calc only does 'a * b': {expr!r}"}
        a = vals.get(m.group(1))
        b = vals.get(m.group(3), None)
        if b is None:
            b = _num(m.group(3))
        if a is None:
            return {"error": f"unknown name {m.group(1)!r}"}
        value = {"*": a * b, "/": (a / b if b else 0), "+": a + b, "-": a - b}[m.group(2)]
        return {"value": round(value, 2), "expr": expr, "inputs": shown}


FILE_LOC = (_FakeIndex.FILE, "Subscriber Fee Model › B48")


async def _selftest(folder: str | None, dry: bool, provider: str | None = None, fake: bool = False) -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if fake:
        index: Any = _FakeIndex()
    else:
        from presage.ingest import Index
        index = Index()
        index.build(folder or ".")

    async def emit(msg: dict) -> None:
        kind = msg.get("type")
        if kind == "candidate":
            print(f"  [candidate] {msg['topic']:<28} {msg['status']:<8} {msg.get('preview', '')[:80]}")
        elif kind == "answer" and msg.get("partial"):
            print(f"  [partial:{msg['source']}] {msg['latency_ms']} ms  {msg['answer']}")
        elif kind == "answer":
            print(f"  [answer:{msg['source']}] {msg['latency_ms']} ms  {msg['answer']}  fig={msg['figure']}  "
                  f"cites={[c['file'] + ' › ' + c['loc'] for c in msg['citations']]}  in_files={msg['in_files']}  ({msg['model']})")
            print(f"      answered_from={msg.get('answered_from')} derived={msg.get('derived')} "
                  f"cached={msg.get('cached')} working={msg.get('working')!r}")
        elif kind == "scratch":
            for it in msg.get("items", []):
                print(f"  [scratch] {it['label']} → {it['value']}: {it['output']}")
        elif kind == "question":
            print(f"  [question:{msg['status']}] {msg['who']} {msg['policy']} via {msg['via']}  {msg['text'][:70]}")

    brain = Brain(index, emit, llm=make_llm(provider, models=ENV_MODELS) if provider else None)
    print("provider:", json.dumps(brain.llm.state()))
    script = [
        ("system", "Thanks for walking us through the deck so far.", True),
        ("system", "So on the revenue side", False),
        ("system", "So on the revenue side, what did Q3 actually look like versus plan?", True),
        ("mic", "Sure, let me pull that up.", True),
        ("system", "And remind me, how many heads are in the headcount plan for next year?", True),
        ("system", "What's the weather like where you are?", True),
        # E68: the consultant's own questions, one routed each way the Me policy can route
        ("mic", "Sorry, remind me what Q3 revenue came in at again?", True),       # a lookup: answered
        ("mic", "Does that hiring pace work for your board?", True),              # a view: heard, no card
    ]
    if fake:
        script = [("system", "so what if we charged 1% of", False),
                  ("system", WHATIF_Q + "?", True)]
    if dry:
        print(brain.system_text()[:3000])
        print("...")
        for src, text, final in script + [("system", WHATIF_Q, True)]:
            hits = brain._topic_hits(text)
            print(f"\n{src}: {text!r} final={final} questionish={questionish(text)} "
                  f"whatif_ish={whatif_ish(text)}")
            print("  hits:", [(h.topic, round(h.score, 2)) for h in hits[:5]])
            if hits:
                print("  evidence:\n   " + brain._evidence(hits, text)[:1200].replace("\n", "\n   "))
        return
    ok, why = await brain.check_key()
    print(f"key: {'ok' if ok else 'NOT USABLE - ' + why}")
    if not ok:
        return
    await brain.prewarm()
    hk = asyncio.create_task(brain.housekeeping())
    if fake:
        # scratch paper end to end: spoken what-if (speculative draft), then the typed question,
        # then a follow-up that lands on the derived Q&A cache entry
        for src, text, final in script:
            print(f"\n{src}: {text!r} (final={final})")
            await brain.on_transcript(Utterance(id=new_id("t"), source=src, text=text, final=final,
                                                ts=time.time(), question_forming=not final))
            await asyncio.sleep(1.2 if not final else 18)
        print(f"\ntyped ask: {WHATIF_Q!r}")
        await brain.ask(WHATIF_Q)
        await asyncio.sleep(22)
        print("\ntyped follow-up: 'and at 1.25%?'")
        await brain.ask("and at 1.25%?")
        await asyncio.sleep(22)
        hk.cancel()
        print("\ntool calls:", json.dumps(index.calls, default=str))
        print("scratch:", json.dumps(brain.state()["scratch"], default=str))
        print("stats:", json.dumps(brain.stats))
        sys.stdout.flush()
        return
    for src, text, final in script:
        print(f"\n{src}: {text!r} (final={final})")
        await brain.on_transcript(Utterance(id=new_id("t"), source=src, text=text, final=final, ts=time.time()))
        await asyncio.sleep(2.5 if final else 0.8)
    for q in brain.state()["questions"][::-1]:
        print(f"  [question] {q['who']:<4} {q['policy']:<6} {q['status']:<9} via {q['via']:<10} {q['text'][:60]}")
    await asyncio.sleep(8)
    print("\nmanual ask: 'churned ARR in enterprise'")
    await brain.ask("How much ARR churned in the enterprise segment?")
    await asyncio.sleep(10)
    print("\nlive question not in the files (web escalation when the provider supports it)")
    await brain.on_transcript(Utterance(id=new_id("t"), source="system",
                                        text="What is the current US federal funds rate?", final=True, ts=time.time()))
    await asyncio.sleep(25)
    print(f"\ntyped what-if: {WHATIF_Q!r}  (needs the lever table from the deep index)")
    await brain.ask(WHATIF_Q)
    await asyncio.sleep(25)
    hk.cancel()
    print("\nscratch:", json.dumps(brain.state()["scratch"], default=str))
    print("stats:", json.dumps(brain.stats))
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    ap = argparse.ArgumentParser(description="presage brain self-test")
    ap.add_argument("folder", nargs="?", help="context folder to index (not needed with --fake)")
    ap.add_argument("--dry", action="store_true", help="no API calls: print prompts, topic hits and evidence")
    ap.add_argument("--provider", help="claude_code (default), anthropic or gemini; overrides PRESAGE_LLM_PROVIDER")
    ap.add_argument("--fake", action="store_true",
                    help="scratch paper only, against a fake index carrying the the big model numbers from the spec")
    a = ap.parse_args()
    if not a.folder and not a.fake:
        ap.error("a context folder is required unless --fake is given")
    asyncio.run(_selftest(a.folder, a.dry, a.provider, a.fake))
