"""Typed intent: which lever a spoken what-if means, and whether a warm card answers the question.

Two decisions, both on the `Decider` contract in `presage/llm/jev.py`, so Jev, Haiku and the local
Laya gate all drive them unchanged:

    pick_lever(decider, line, levers, prior=None)  -> Pick     one choice over the lever table
    card_answers(decider, card_q, question)        -> Verdict  one noul over a (card, question) pair

`pick_lever` replaces the rarity-weighted word matching in `Brain._lever_matches`. Measured on 300
labelled spoken lines over the big test model's 250 levers (`docs/experiments/E7-typed-intent.md`):
95.7% against the matcher's 56.7%, at 252 ms p50 and $0.00054 a call on Jev. Act on the answer when
`pick.p >= 0.5` and 98.8% of what it acts on is right, on 95.4% of the lines that name a lever.
The lever table is the record `presage.ingest.scratch` builds: label, header (the scenario column),
section (the block title above the row) and its `context` (sheet + section), sheet, cell, value,
fmt, aliases.

**The choice is asked over a shortlist above 50 levers** (`SHORTLIST_K`), drawn by
`presage.llm.shortlist`: a model2vec rank over the same option text, fused by reciprocal rank with
the pre-run's word score. E7 ruled a shortlist out because the word matcher could not draw one --
its top 20 held the answer on 81.3% of lines and the two-stage path was capped at 78.3% -- and E20
found that was the matcher, not the shape. The fused rank holds the gold in its top 50 on 97.0% of
the same 300 lines, which is every line whose gold cell still exists, and the choice over that top
50 scores 93.0% against 92.3% over the whole 250-lever table, in 233 ms against 387. Past 254
levers it is the only path there is, because one `choice` takes 255 options.

**Both questions also run locally**, on the E21 multi checkpoint, which is what
`presage/llm/intent_local.py` loads and what `k_for` sizes the shortlist for: 94.3% of E7's 300
lines to the right cell in 43 ms against Jev's 91.0% in 241, and 98.8 / 97.6 precision / recall on
the warm card in 28 ms against Jev's 94.5 / 72.9. It must never be handed the whole table: at 251
options the sequence leaves 9 tokens for the line and it answers the same option every time
(14.9%, the NONE option, on all 47 held-out lines). The gate checkpoint is not the multi one and
answers neither question; `k_for` only sizes the question, it does not choose the weights.

Two things make the difference, and both are in `lever_options` rather than in the model:

1. **One option per lever, carrying what tells two levers apart.** The same label appears twice in
   a scenario model -- churn under Base and churn under Downside, fees under Premium and fees under
   Market -- so an option is `label · header (sheet › section) = value`, and the answer comes back
   as a cell. The section is what separates two cells that a label and a value cannot: the big model holds
   0.85% under `Structure 2: Percentage of GMV` and 0.85% again under `Subscriber Volume Inputs`,
   both called an platform fee (E36).
2. **A NONE option.** Most spoken lines name no lever at all. Without somewhere to put them a
   choice question has to answer something, and every "what if they just say no?" lands on a lever.

`prior` carries the previous ask. On Jev the same request also asks whether the line is a
follow-up to it ("and at 1.25?", "and per month?"), and a follow-up resolves to the previous
lever instead of to whatever the words alone suggest; it costs no extra round trip, because Jev
answers every question in one request. A local checkpoint is not asked that question at all (see
`asks_follow`): it was trained on the lever choice with the previous ask in the state, the
previous lever is pinned into its shortlist, and the verdict is read off which lever it picked.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from presage.llm.jev import MAX_OPTIONS, Decider, DecisionError, Q

log = logging.getLogger("presage.llm.intent")

NONE_OPTION = "none of these — the line does not name an input of the model"
FOLLOW_SAME = "same input as the previous ask, at a different value"
FOLLOW_NEW = "a different input of the model"
FOLLOW_NONE = "not a what-if at all"
FOLLOW_OPTIONS = [FOLLOW_SAME, FOLLOW_NEW, FOLLOW_NONE]


SHORTLIST_K = 50           # levers put in front of Jev when the table is longer (E20)
LOCAL_SHORTLIST_K = 20     # levers put in front of a local Laya checkpoint, always (E21)
AUTO_K = "auto"            # `k=AUTO_K`: let the decider decide, which is what every caller wants


def is_local(decider: Any) -> bool:
    """True for a Laya-backed decider, hosted or otherwise: torch, Core ML, the gate, the E21
    multi checkpoint. They all share one sequence budget and one 255-option ceiling that bites
    long before Jev's."""
    return str(getattr(decider, "kind", "") or "") == "laya"


def k_for(decider: Any, k: int | str | None = AUTO_K) -> int | None:
    """How many levers this decider gets to choose from.

    Jev takes 50 (E20: the same accuracy as the whole 250-lever table, 154 ms faster, and the only
    path past 254 levers). A local checkpoint takes 20 and **never the whole table**: at 251
    options `laya.build_sequence` cuts each option to four tokens, leaves nine for the line, and
    the checkpoint answers the same option to everything (14.9%, E21). So `k=None`, which means
    "no shortlist" for Jev, still means 20 here, and an explicit larger k is cut down to it.
    """
    if is_local(decider):
        return LOCAL_SHORTLIST_K if (k is None or k == AUTO_K) else min(int(k), LOCAL_SHORTLIST_K)
    return SHORTLIST_K if k == AUTO_K else k


def asks_follow(decider: Any) -> bool:
    """Whether to put the three-way follow-up question in the request as well.

    Jev answers it in the same round trip for nothing, so it always gets asked. The local E21
    checkpoint was never trained on it: E21 turned the follow-up pairs into *lever* rows whose
    state carries the previous ask, and left the three-way question itself out of the mix. Asked
    it anyway, it scores **36.0%** on E7's 100 follow-ups against Jev's 94.0% (E21a).

    So a local follow-up is answered by the question the checkpoint was trained on: the same
    lever choice, with the previous ask in the state and the previous lever pinned into the
    shortlist. Picking that lever again *is* "same", picking another *is* "new", and NONE -- "the
    line does not name an input of the model" -- is "not a what-if". `_follow_from` reads it off.
    """
    return not is_local(decider)


def _follow_from(prior: dict | None, lever: dict | None) -> str | None:
    """The follow-up verdict implied by which lever was picked, for a decider that was not asked."""
    if not prior:
        return None
    if lever is None:
        return "not-whatif"
    return "same" if cell_of(lever) == cell_of(prior.get("lever") or {}) else "new"


@dataclass
class Pick:
    """The lever a line means. `lever` is None when the line names none, or is not a what-if."""
    lever: dict | None
    cell: str | None                  # "Sheet!A1"
    p: float = 0.0                    # the probability on the chosen option
    confidence: float = 0.0
    follow: str | None = None         # "same" / "new" / "not-whatif" when `prior` was given
    latency_ms: int = 0
    probs: dict[str, float] | None = None
    sent: list[str] | None = None     # the cells offered, when the table was shortlisted
    shortlist_ms: float = 0.0


@dataclass
class Verdict:
    """Whether a warm card answers the question that landed."""
    answers: bool
    p: float = 0.0
    confidence: float = 0.0
    latency_ms: int = 0


# ----------------------------------------------------------------------------- the options
def cell_of(lever: dict) -> str:
    return f'{lever.get("sheet")}!{lever.get("cell")}'


def _fmt_value(lever: dict) -> str:
    try:
        from presage.ingest.scratch import fmt_value
        return fmt_value(lever.get("value"), lever.get("fmt") or "number")
    except Exception:  # noqa: BLE001  the option text must never take a decision down
        v = lever.get("value")
        return "n/a" if v is None else f"{v:g}" if isinstance(v, (int, float)) else str(v)


MAX_OPTION_ALIASES = 6     # the names printed per option, of the eight a lever may carry

# A block title ends in how the block is laid out ("... by Market Scenario") and in what the
# rows under it are ("... Comps"). Neither says what the input is, and both are words the
# choice can match on by accident, so they come off before the option is printed.
_SECTION_TAIL = re.compile(r"\s+by\s+.*$|[\s.,;:-]+(?:comps?|inputs?|notes?|data|table|"
                           r"assumptions?|detail|details|summary)\s*$", re.I)


def _section(lever: dict) -> str:
    return _SECTION_TAIL.sub("", str(lever.get("section") or "").strip()).strip(" .,:;-")


def _context(lever: dict) -> str:
    """`Sheet › block title`, which is where the option says the input lives."""
    sheet, section = str(lever.get("sheet") or "").strip(), _section(lever)
    if not section:
        ctx = str(lever.get("context") or "").strip()
        return ctx or sheet
    return f"{sheet} › {section}" if section.lower() != sheet.lower() else sheet


def option_text(lever: dict, aliases: bool = True) -> str:
    """One option: `label · header (sheet › section) = value [spoken names]`.

    The header is the scenario column the input sits under and is the only thing separating the
    two churn cells or the two mentor salaries, so it goes in front of the sheet name. The
    section is the block title above the row, and it is the only thing separating two cells
    that hold the same number under the same words: the big model carries a 0.85% "platform fee" under
    `Subscriber Fee Model › Structure 2: Percentage of GMV` and another under `Subscriber Ramp ›
    Subscriber Volume Inputs`, and without the block a chooser is guessing between them (E36).
    """
    label = str(lever.get("label") or "").strip() or cell_of(lever)
    head = str(lever.get("header") or "").strip()
    context = _context(lever)
    parts = [label]
    # the header is dropped only when it says the same thing as the block, never because its
    # letters are inside it: "Score" sits inside "Scorecard", and dropping it took the score
    # column off four of this set's scenario lines (E36)
    dup = head.lower() in (str(lever.get("sheet") or "").lower(), _section(lever).lower())
    if head and head.lower() not in label.lower() and not dup:
        parts.append(f"· {head}")
    parts.append(f'({context}) = {_fmt_value(lever)}')
    names = ([a for a in (lever.get("aliases") or []) if a][:MAX_OPTION_ALIASES]
             if aliases else [])
    if names:
        parts.append("[" + "; ".join(names) + "]")
    return " ".join(parts)


def lever_options(levers: list[dict], aliases: bool = True) -> tuple[list[str], dict[str, dict]]:
    """Option strings plus the map back to the lever. Uniqueness is guaranteed by the cell ref,
    which is appended only when two levers would otherwise print the same option."""
    texts = [option_text(lv, aliases) for lv in levers]
    seen: dict[str, int] = {}
    for t in texts:
        seen[t] = seen.get(t, 0) + 1
    options: list[str] = []
    back: dict[str, dict] = {}
    for lv, t in zip(levers, texts):
        if seen[t] > 1 or t in back:
            t = f"{t} [{lv.get('cell')}]"
        options.append(t)
        back[t] = lv
    return options, back


def _cut(levers: list[dict]) -> list[dict]:
    """The table as a choice question can take it, when there is no shortlist to be had."""
    return levers[:MAX_OPTIONS - 1] if len(levers) > MAX_OPTIONS - 1 else levers


def shortlisted(line: str, levers: list[dict], prior: dict | None = None,
                k: int | None = SHORTLIST_K, vecs: Any = None) -> tuple[list[dict], float]:
    """The levers to ask about, and what drawing them cost, in ms.

    Everything shorter than `k` is passed through, so a small workbook asks the question it
    always did. Above it, `shortlist.rank` fuses a model2vec cosine over the option text with
    `prerun.match_levers`; the vectors are memoised per lever table, so the cost on a call is the
    word matcher's (3 ms over 250 levers, 8 ms over 600) and not the embedding's.

    The previous ask's lever is pinned into the list whatever it ranks, because a follow-up
    ("and at 1.25?") has to be able to resolve to it and its own words name nothing.

    Every failure here falls back to the front of the table rather than raising: a shortlist that
    cannot be drawn must not take a decision down.
    """
    if k is None or len(levers) <= k:
        return _cut(levers), 0.0
    t0 = time.perf_counter()
    try:
        from presage.llm import shortlist as _sl
        keep = [cell_of(prior["lever"])] if (prior or {}).get("lever") else []
        out = _sl.shortlist(line, levers, k,
                            vecs if vecs is not None else _sl.vecs_for_table(levers), keep=keep)
    except Exception as e:  # noqa: BLE001
        log.debug("shortlist unavailable, asking the front of the table: %s", e)
        return _cut(levers), (time.perf_counter() - t0) * 1000
    return (out or _cut(levers)), (time.perf_counter() - t0) * 1000


def _state(line: str, prior: dict | None) -> str:
    who = ("A consultant is presenting a financial model to a client. The client just said the "
           "line below. Judge only that line.")
    if prior:
        prev_q = str(prior.get("question") or "").strip()
        lv = prior.get("lever") or {}
        prev_lever = option_text(lv) if lv else "(unknown)"
        return (f"{who}\n\nThe client's PREVIOUS ask was: \"{prev_q}\"\n"
                f"It was answered by changing this input: {prev_lever}\n\n"
                f"THE LINE: \"{line}\"")
    return f'{who}\n\nTHE LINE: "{line}"'


INSTRUCTIONS = (
    "Which input of the model does the line ask to change? Pick the one the speaker named, in the "
    "words they would say out loud rather than the words on the sheet: the line is a spoken "
    "paraphrase, never a quotation of the label. A line that gives a value for an input is asking "
    "to change it however it is phrased (\"run X at 9\", \"I want to see X at 9\", \"what would 9 "
    "do for X\", \"X at 9 instead\"). Where the same input appears under two scenario columns, "
    "pick the column the line names, and the first one listed when it names none. Pick the last "
    "option only when no input in the list is named at all: the line asks for a figure as the "
    "model already stands, it names something the model computes rather than something you could "
    "type over, or it is not about the model."
)
FOLLOW_INSTRUCTIONS = (
    "Is the line a follow-up to the previous ask? Answer \"same input\" when it asks for the same "
    "input again at a different value or in different units (\"and at 1.25?\", \"and per month?\", "
    "\"try double that\"), \"a different input\" when it names or implies another one (including "
    "the same input under a different scenario column), and \"not a what-if\" when there is "
    "nothing to recalculate."
)


# ----------------------------------------------------------------------------- pick the lever
async def pick_lever(decider: Decider, line: str, levers: list[dict], prior: dict | None = None,
                     aliases: bool = True, k: int | str | None = AUTO_K, vecs: Any = None) -> Pick:
    """The lever a spoken what-if means, as one typed decision over the lever table.

    `levers` is the whole table from `Index.levers()`, however long it is: anything past `k`
    levers is ranked and cut to `k` here (see `shortlisted`), which is both what makes a workbook
    with more than 254 levers answerable at all and, on the one workbook where both can be
    measured, 154 ms faster at the same accuracy. `k` defaults to whatever this decider can take
    (`k_for`): 50 for Jev, 20 for a local checkpoint, which is a ceiling rather than a default.
    Pass `k=None` for the old behaviour on Jev, the front of the table truncated to what a choice
    question takes.

    `prior` is `{"question": str, "lever": dict}` for the ask before this one, which turns on the
    follow-up question in the same request and pins that lever into the shortlist.
    """
    if not (line or "").strip():
        return Pick(None, None)
    if not levers:
        raise DecisionError("no levers")
    levers, short_ms = shortlisted(line, levers, prior, k_for(decider, k), vecs)
    options, back = lever_options(levers, aliases)
    qs = [Q("lever", "choice", INSTRUCTIONS, options=options + [NONE_OPTION])]
    if prior and asks_follow(decider):
        qs.insert(0, Q("follow", "choice", FOLLOW_INSTRUCTIONS, options=FOLLOW_OPTIONS))
    t0 = time.time()
    out = await decider.decide(_state(line, prior), qs)
    ms = int((time.time() - t0) * 1000)

    sent = [cell_of(lv) for lv in levers]
    follow = None
    if prior and "follow" in out:
        value = str(out["follow"].value)
        follow = {FOLLOW_SAME: "same", FOLLOW_NEW: "new", FOLLOW_NONE: "not-whatif"}.get(value, "new")
        if follow == "same":
            lv = prior.get("lever")
            d = out["follow"]
            return Pick(lv, cell_of(lv) if lv else None, d.p, d.confidence, follow, ms, d.probs,
                        sent, short_ms)
        if follow == "not-whatif":
            d = out["follow"]
            return Pick(None, None, d.p, d.confidence, follow, ms, d.probs, sent, short_ms)
    d = out["lever"]
    lever = back.get(str(d.value))
    return Pick(lever, cell_of(lever) if lever else None, d.p, d.confidence,
                follow if follow is not None else _follow_from(prior, lever), ms, d.probs,
                sent, short_ms)


# ----------------------------------------------------------------------------- the warm card
CARD_INSTRUCTIONS = (
    "A card has already been drafted, answering the question in DRAFTED. The question in ASKED is "
    "the one that was actually asked. Would the drafted card answer it? True only when the same "
    "fact answers both: the same figure, for the same thing, on the same basis. False when the "
    "card holds a near neighbour -- another scenario column, another year, another line on the "
    "same sheet, the level when what was asked is what it would become under a change -- and "
    "false when the card is about something else entirely."
)


async def card_answers(decider: Decider, card_q: str, question: str) -> Verdict:
    """Does the warm card drafted for `card_q` answer `question`? One noul.

    Measured on 200 labelled pairs (E7 doc, section 3): 93.0% precision and 88.9% recall on Jev at
    168 ms, against 56.3% / 84.4% for the 0.45 cosine in `Brain._best_warm`. On E21's 377 held-out
    pairs the local multi checkpoint answers it in 28 ms at **98.8% precision and 97.6% recall**,
    against Jev's 94.5 / 72.9 and the shipped cosine's 61.3 / 89.8, which is why the confirmation
    step does not need a round trip any more (E21a).
    """
    if not (card_q or "").strip() or not (question or "").strip():
        return Verdict(False)
    state = f'DRAFTED: "{card_q}"\nASKED: "{question}"'
    t0 = time.time()
    out = await decider.decide(state, [Q("answers", "noul", CARD_INSTRUCTIONS)])
    ms = int((time.time() - t0) * 1000)
    d = out["answers"]
    return Verdict(bool(d.value), d.p, d.confidence, ms)


# ----------------------------------------------------------------------------- both at once
async def pick_and_check(decider: Decider, line: str, levers: list[dict], card_q: str | None = None,
                         prior: dict | None = None, k: int | str | None = AUTO_K,
                         vecs: Any = None) -> tuple[Pick, Verdict | None]:
    """The lever and the warm-card check in one request, for the live path where both are wanted."""
    if not card_q:
        return await pick_lever(decider, line, levers, prior, k=k, vecs=vecs), None
    levers, short_ms = shortlisted(line, levers, prior, k_for(decider, k), vecs)
    options, back = lever_options(levers)
    qs: list[Q] = [Q("lever", "choice", INSTRUCTIONS, options=options + [NONE_OPTION]),
                   Q("answers", "noul", CARD_INSTRUCTIONS + f'\n\nDRAFTED: "{card_q}"')]
    if prior and asks_follow(decider):
        qs.insert(0, Q("follow", "choice", FOLLOW_INSTRUCTIONS, options=FOLLOW_OPTIONS))
    t0 = time.time()
    out = await decider.decide(_state(line, prior), qs)
    ms = int((time.time() - t0) * 1000)
    a = out["answers"]
    verdict = Verdict(bool(a.value), a.p, a.confidence, ms)
    sent = [cell_of(lv) for lv in levers]
    follow = None
    if prior and "follow" in out:
        follow = {FOLLOW_SAME: "same", FOLLOW_NEW: "new", FOLLOW_NONE: "not-whatif"}.get(
            str(out["follow"].value), "new")
        if follow == "same":
            lv = prior.get("lever")
            return Pick(lv, cell_of(lv) if lv else None, out["follow"].p,
                        out["follow"].confidence, follow, ms, None, sent, short_ms), verdict
        if follow == "not-whatif":
            return Pick(None, None, out["follow"].p, out["follow"].confidence, follow, ms,
                        None, sent, short_ms), verdict
    d = out["lever"]
    lever = back.get(str(d.value))
    return Pick(lever, cell_of(lever) if lever else None, d.p, d.confidence,
                follow if follow is not None else _follow_from(prior, lever), ms, d.probs,
                sent, short_ms), verdict


__all__ = ["Pick", "Verdict", "pick_lever", "card_answers", "pick_and_check", "lever_options",
           "option_text", "cell_of", "shortlisted", "is_local", "k_for", "asks_follow",
           "NONE_OPTION", "SHORTLIST_K", "LOCAL_SHORTLIST_K", "AUTO_K"]


def _levers_from(index: Any) -> list[dict]:
    """The lever table off an Index, empty when the workbook will not compile."""
    try:
        return list(index.levers() or [])
    except Exception:  # noqa: BLE001
        return []
