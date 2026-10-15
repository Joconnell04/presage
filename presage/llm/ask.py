"""The typed ask: which row, which column, which grid -- decided with the gate, not after it.

`Brain._evidence` turns a spoken line into evidence by matching topics on row labels and packing
whatever they point at: four topic hits, two targets each, plus a LIKE over the four longest
content words. It is sub-millisecond and it is why E34's two commonest wrong cards are "an
objection is answered with a neighbouring figure" and "a plain question gets a neighbouring
figure": the pre-pack hands the answer model a block holding the right row *and* its neighbours,
the right column *and* the other five, and the model picks one.

This module asks for the cell instead. Five slots, every one of them a choice over the workbook's
own vocabulary, so nothing has to be generated and nothing can be invented:

    metric      a row label (`cells.row_label` over numeric cells, plus deep definition names)
    period      a column header (`cells.header` over numeric cells, plus `_columns.label`)
    segment     the grid the figure lives in: a sheet, a table, or a filter value in a text column
    compare_to  what it is being held against: the plan, the prior period, another segment, none
    basis       actual / plan / scenario / per unit / percent / rate -- a closed five-way

`grain` is not asked. A period is a column of the index and the index already knows whether that
column is a month, a quarter, a year or a total, so `grain_of` reads it off the chosen option for
nothing. One fewer choice question is one fewer option list in the request.

Two calls:

    extract_ask(decider, line, vocab, prior=None, gate=True)  -> Ask
    evidence_targets(ask, index)                              -> [target, ...]

`extract_ask` is meant to run **inside the batch `Brain._gate_decider` already sends** (E2
proposal 4): `gate=True` adds `kind`, `needs_lookup` and `complete` to the same request, so the
slots cost the difference between a three-question request and an eight-question one, not a second
round trip. `evidence_targets` returns targets in exactly the shape `Brain._evidence_for_target`
already consumes (`row`, `column`, `value`, `table`), so the pre-pack changes where it points
without changing what it knows how to render.

`prior` is the ask before this one. A follow-up restates one slot and leans on the rest -- "and for
Q2?", "same for EMEA", "what about the plan?" -- and `inherit` fills the ones it did not restate
from the prior ask. That is E2 proposal 6: the slots a follow-up does not say are already typed, so
carrying them costs 0 ms against 312 to 1,867 ms for a query rewriter.

Measured in `docs/experiments/E38-typed-ask.md` on `data/gate/ask/asks.jsonl`.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from presage.llm.jev import MAX_OPTIONS, Decider, Q

log = logging.getLogger("presage.llm.ask")

NONE = "none — the line does not name one"
MAX_CHOICES = MAX_OPTIONS - 1          # a NONE option always rides along

# `basis` and `compare_to` are the two slots the workbook does not supply a vocabulary for, so
# they are closed sets written once. Both are small enough for a pre-gate style logistic head.
# "actual" is not an option: it is the default, nobody says it out loud, and a slot holds what
# the line names. A line that asks for a figure on no particular basis answers NONE here.
BASES = ["plan or budget, what the figure is meant to be",
         "a scenario or case (downside, base, upside, low, high)",
         "per unit (per subscriber, per customer, per head, per deal)",
         "a percent, a rate or a margin"]
BASIS_KEYS = ["plan", "scenario", "per_unit", "rate"]
COMPARES = ["the plan or budget for the same thing",
            "the same thing in an earlier period",
            "another segment, region, department or scenario",
            "an outside benchmark or a competitor"]
COMPARE_KEYS = ["plan", "prior", "other_segment", "benchmark"]

# a column label the shaper could not name, and block titles that are not a period
_DEAD_COL = re.compile(r"^col\s+[A-Z]{1,3}$|^_", re.I)
_GRAIN = [
    ("month", re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|month|mtd|mom)\b", re.I)),
    ("quarter", re.compile(r"\b(q[1-4]|quarter|qtd|qoq)\b", re.I)),
    ("total", re.compile(r"\b(total|full year|fy total|cumulative|all)\b", re.I)),
    ("year", re.compile(r"\b(fy\d{2,4}|cy\d{2,4}|year\s*\d|yr\s*\d|year|annual|yoy|ytd|\d{4})\b", re.I)),
]


# ----------------------------------------------------------------------------- the slots
@dataclass
class Slot:
    """One filled slot. `source` says who filled it, which is what a follow-up needs to know."""
    value: str | None = None
    p: float = 0.0
    source: str = ""              # jev | local | head | inherited | derived | ""

    def __bool__(self) -> bool:
        return bool(self.value)


@dataclass
class Ask:
    """A spoken line, typed. Every value is a string the index itself produced."""
    metric: Slot = field(default_factory=Slot)
    period: Slot = field(default_factory=Slot)
    segment: Slot = field(default_factory=Slot)
    compare_to: Slot = field(default_factory=Slot)
    basis: Slot = field(default_factory=Slot)
    grain: str | None = None
    line: str = ""
    file: str | None = None
    # the gate's own three, when they rode in the same request
    kind: str | None = None
    needs_lookup: bool | None = None
    complete: bool | None = None
    latency_ms: int = 0

    SLOTS = ("metric", "period", "segment", "compare_to", "basis")

    def slots(self) -> dict[str, str | None]:
        return {n: getattr(self, n).value for n in self.SLOTS}

    def probs(self) -> dict[str, float]:
        return {n: getattr(self, n).p for n in self.SLOTS}

    def inherited(self) -> tuple[str, ...]:
        return tuple(n for n in self.SLOTS if getattr(self, n).source == "inherited")

    def filled(self) -> int:
        return sum(1 for n in self.SLOTS if getattr(self, n).value)


# ----------------------------------------------------------------------------- the vocabulary
@dataclass
class Vocab:
    """What the folder lets a question name, read off the index once per build.

    `metrics`, `periods` and `segments` are option lists; the three maps beside them take an
    option back to where it lives, which is what `evidence_targets` walks.
    """
    metrics: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    metric_at: dict[str, list[dict]] = field(default_factory=dict)
    period_at: dict[str, list[dict]] = field(default_factory=dict)
    segment_at: dict[str, list[dict]] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    folder: str = ""

    def sizes(self) -> dict[str, int]:
        return {"metrics": len(self.metrics), "periods": len(self.periods),
                "segments": len(self.segments)}


def _rows(index: Any, sql: str, limit: int = 4000) -> list[dict]:
    try:
        return [r for r in index.sql(sql, limit=limit) if "error" not in r]
    except Exception as e:                                   # noqa: BLE001
        log.debug("vocab query failed: %s", e)
        return []


def _clean(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _is_num(s: str) -> bool:
    try:
        float(str(s).replace(",", "").replace("%", "").replace("$", ""))
        return True
    except (TypeError, ValueError):
        return False


def build_vocab(index: Any, file: str | None = None) -> Vocab:
    """The option lists, from the index and nothing else.

    The rules are the index's own, not new ones. `presage.ingest._topic_sources` already decides
    what a row label is (the first column of a table with at most `MAX_ROW_LABEL_ROWS` rows) and
    what a filter value is (a text column with at most `MAX_DISTINCT_VALUES` distinct values), and
    this reuses both constants, so a slot can only ever be a string the topic index also carries.

    The one distinction added here is grid against record. A table of at most
    `MAX_ROW_LABEL_ROWS` rows is a **grid**: its rows are metrics and its numeric columns are
    periods. A longer table is a **record table**, one row per customer or per cell: its rows are
    not metrics at all and its numeric columns (`arr_usd`, `csat`) are. That is why a 200-row
    export does not put two hundred customer ids in front of a choice question.
    """
    from presage.ingest import MAX_DISTINCT_VALUES, MAX_ROW_LABEL_ROWS

    v = Vocab(folder=str(getattr(index, "folder", "") or ""))
    tables = [r for r in _rows(index, "SELECT name, file, sheet, title, n_rows FROM _tables "
                                      "ORDER BY file, sheet", limit=400)
              if not file or r.get("file") == file]

    def keep(bag: dict[str, list[dict]], seen: dict[str, str], label: Any, where: dict) -> None:
        label = _clean(label)
        if len(label) < 2 or len(label) > 64 or _DEAD_COL.match(label) or _is_num(label):
            return
        canon = seen.setdefault(label.lower(), label)
        bag.setdefault(canon, []).append(where)

    m_seen: dict[str, str] = {}
    p_seen: dict[str, str] = {}
    s_seen: dict[str, str] = {}
    for t in tables:
        cols = _table_rows(index, t["name"])
        if not cols:
            continue
        first, grid = cols[0], int(t.get("n_rows") or 0) <= MAX_ROW_LABEL_ROWS
        where = {"table": t["name"], "file": t.get("file"), "sheet": t.get("sheet")}
        keep(v.segment_at, s_seen, t.get("sheet"), {"kind": "table", **where})
        keep(v.segment_at, s_seen, str(t["name"]).replace("_", " "), {"kind": "table", **where})
        if grid and first["type"] == "TEXT":
            for r in _rows(index, f'SELECT "{first["name"]}" AS v, _row FROM "{t["name"]}" '
                                  f'WHERE "{first["name"]}" IS NOT NULL', limit=200):
                keep(v.metric_at, m_seen, r["v"], {"row": r["_row"], "letter": first["letter"], **where})
        for c in cols:
            if c["type"] == "REAL":
                bag = v.period_at if grid else v.metric_at
                seen = p_seen if grid else m_seen
                keep(bag, seen, c["label"] or c["name"],
                     {"column": c["name"], "letter": c["letter"], **where})
            elif c["type"] == "TEXT" and not (grid and c["ord"] == first["ord"]):
                vals = _rows(index, f'SELECT DISTINCT "{c["name"]}" AS v FROM "{t["name"]}" '
                                    f'WHERE "{c["name"]}" IS NOT NULL',
                             limit=MAX_DISTINCT_VALUES + 1)
                if not vals or len(vals) > MAX_DISTINCT_VALUES:
                    continue
                for r in vals:
                    keep(v.segment_at, s_seen, r["v"],
                         {"kind": "value", "column": c["name"], "value": _clean(r["v"]), **where})

    # the header the shaper wrote above a numeric cell, which is the column as the sheet prints it
    grids = {t["name"] for t in tables if int(t.get("n_rows") or 0) <= MAX_ROW_LABEL_ROWS}
    sheets = {(t.get("file"), t.get("sheet")) for t in tables if t["name"] in grids}
    for r in _rows(index, "SELECT header, file, sheet, COUNT(*) AS n FROM cells "
                          "WHERE header IS NOT NULL AND value_num IS NOT NULL "
                          "GROUP BY header, file, sheet"):
        if (r.get("file"), r.get("sheet")) in sheets and (not file or r.get("file") == file):
            keep(v.period_at, p_seen, r["header"],
                 {"file": r.get("file"), "sheet": r.get("sheet"), "header": _clean(r["header"])})
    # a definition names a metric the sheet computes but never labels
    for r in _rows(index, "SELECT name, file, sheet, cell_ref FROM definitions", limit=400):
        keep(v.metric_at, m_seen, r.get("name"),
             {"file": r.get("file"), "sheet": r.get("sheet"), "cell_ref": r.get("cell_ref"),
              "definition": True})

    v.metrics = sorted(v.metric_at, key=lambda s: (s.lower(), s))
    v.periods = sorted(v.period_at, key=lambda s: (s.lower(), s))
    v.segments = sorted(v.segment_at, key=lambda s: (s.lower(), s))
    v.files = sorted({t.get("file") for t in tables if t.get("file")})
    return v


def grain_of(period: str | None, vocab: Vocab | None = None) -> str | None:
    """month / quarter / year / total, read off the chosen column. Never asked of a model."""
    if not period:
        return None
    for name, rx in _GRAIN:
        if rx.search(period):
            return name
    return None


# ----------------------------------------------------------------------------- shortlisting
def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9%]+", (s or "").lower()) if len(w) > 1}


def shortlist(options: list[str], line: str, k: int, embed=None) -> list[str]:
    """The k options closest to the line, for a vocabulary over Jev's 255-option ceiling.

    Dense first (`embed` is `Brain._embed`'s contract: `list[str] -> (n, dim)` unit rows), word
    overlap second. E7 measured that a *word matcher* shortlist caps the two-stage path at 78.3%
    because its top 20 misses the answer on 19% of lines; k here is in the hundreds, not 20, and
    the fallback only has to beat dropping the tail alphabetically.
    """
    if len(options) <= k:
        return list(options)
    if embed is not None:
        try:
            m = embed([line] + list(options))
            if m is not None:
                sims = m[1:] @ m[0]
                order = sorted(range(len(options)), key=lambda i: -float(sims[i]))
                return [options[i] for i in order[:k]]
        except Exception as e:                                # noqa: BLE001
            log.debug("shortlist embed failed: %s", e)
    toks = _words(line)
    scored = sorted(options, key=lambda o: (-len(_words(o) & toks), o.lower()))
    return scored[:k]


# ----------------------------------------------------------------------------- local tier
def _norm(text: str) -> list[str]:
    """`presage.ingest.topics.normalize`: the canonicaliser that makes "third quarter" and "Q3"
    the same token on both sides. Falling back to bare words costs the alias table."""
    try:
        from presage.ingest.topics import normalize
        return normalize(text)
    except Exception:                                         # noqa: BLE001
        return [w for w in re.findall(r"[a-z0-9%]+", (text or "").lower()) if len(w) > 1]


def _rarity(options: list[str]) -> tuple[dict[str, float], list[set[str]]]:
    toks = [set(_norm(o)) for o in options]
    df: dict[str, int] = {}
    for t in toks:
        for w in t:
            df[w] = df.get(w, 0) + 1
    import math
    return {w: 1.0 / math.log(2.0 + c) for w, c in df.items()}, toks


def local_slot(line: str, options: list[str], embed=None, floor: float = 0.5) -> Slot:
    """One slot, no model call: rarity-weighted word overlap over the canonicalised option, with
    a static-embedding cosine as the tie-break. This is `Brain._lever_matches` pointed at row
    labels and column headers instead of levers, plus the alias table."""
    if not options:
        return Slot()
    weight, toks = _rarity(options)
    want = set(_norm(line))
    best, second, pick = 0.0, 0.0, None
    for opt, t in zip(options, toks):
        total = sum(weight[w] for w in t) or 1.0
        frac = sum(weight[w] for w in (t & want)) / total
        if frac > best:
            best, second, pick = frac, best, opt
        elif frac > second:
            second = frac
    if embed is not None and pick is not None and best - second < 0.15:
        try:
            m = embed([line] + options)
            if m is not None:
                sims = m[1:] @ m[0]
                i = int(sims.argmax())
                if float(sims[i]) >= 0.45:
                    pick = options[i]
        except Exception as e:                                # noqa: BLE001
            log.debug("local slot embed failed: %s", e)
    if best < floor:
        return Slot(None, round(best, 3), "local")
    return Slot(pick, round(best, 3), "local")


_PLAN = re.compile(r"\b(plan|budget|forecast|planned|budgeted)\b", re.I)
_SCEN = re.compile(r"\b(downside|upside|base case|worst case|best case|scenario)\b", re.I)
_UNIT = re.compile(r"\bper (subscriber|customer|head|user|seller|account|deal|partner|channel)\b", re.I)
_RATE = re.compile(r"\b(percent|percentage|margin|rate|as a share|basis points|bps)\b", re.I)
_CMP = [("plan", re.compile(r"\b(against|versus|vs|compared to|relative to)\s+(the\s+)?(plan|budget|forecast)\b", re.I)),
        ("prior", re.compile(r"\b(last (year|quarter|month)|the year before|prior (year|quarter)|year on year|yoy)\b", re.I)),
        ("benchmark", re.compile(r"\b(benchmark|the market|industry|competitor|peers?)\b", re.I)),
        ("other_segment", re.compile(r"\b(against|versus|vs|compared to)\b", re.I))]


def local_ask(line: str, vocab: Vocab, embed=None, prior: dict | None = None) -> Ask:
    """The whole ask with no model call. The 0 ms tier, and the floor the typed call has to beat."""
    t0 = time.time()
    a = Ask(line=line)
    a.metric = local_slot(line, vocab.metrics, embed)
    a.period = local_slot(line, vocab.periods, embed)
    a.segment = local_slot(line, vocab.segments, embed)
    basis = ("rate" if _RATE.search(line) else "per_unit" if _UNIT.search(line) else
             "plan" if _PLAN.search(line) else "scenario" if _SCEN.search(line) else None)
    a.basis = Slot(basis, 0.5, "local")
    a.compare_to = Slot(next((k for k, rx in _CMP if rx.search(line)), None), 0.5, "local")
    a.grain = grain_of(a.period.value)
    a.latency_ms = int((time.time() - t0) * 1000)
    if prior:
        a = inherit(a, prior)
    return a


# ----------------------------------------------------------------------------- the questions
METRIC_Q = ("Which figure on the sheets is the line asking for? Pick the row it names, in the "
            "words a person says out loud rather than the words on the sheet: \"top line\" is "
            "revenue, \"headcount\" is the staff row, \"what EMEA did\" is the EMEA row. Pick the "
            "thing being asked for, not the thing it is being compared against. Pick the last "
            "option when the line names no figure on the sheets at all.")
PERIOD_Q = ("Which column of figures does the line mean? Pick the period or the case it names -- "
            "a quarter, a month, a year, the total, the plan, a scenario. \"Last year\" and \"for "
            "the year\" mean the full-year column, \"the plan\" and \"budget\" mean the plan "
            "column. Pick the last option when the line names no column, which is most of the "
            "time: do not guess a period from the metric.")
SEGMENT_Q = ("Which sheet, table or slice of the data holds the figure? A region, a segment, a "
             "department, a named sheet or a named table. Pick the last option when the line "
             "names none.")
COMPARE_Q = ("What is the figure being held against, if anything? Pick the last option when the "
             "line just asks for a number, which is most of the time.")
BASIS_Q = ("On what basis is the figure being asked for? Pick what the line asks for, not what "
           "the sheet happens to hold.")


def ask_questions(vocab: Vocab, line: str = "", prior: dict | None = None, embed=None,
                  slots: tuple[str, ...] = Ask.SLOTS) -> list[Q]:
    """The choice questions for one line. Add them to the gate's own three and send one request."""
    qs: list[Q] = []
    if "metric" in slots and vocab.metrics:
        opts = shortlist(vocab.metrics, line, MAX_CHOICES, embed)
        qs.append(Q("metric", "choice", METRIC_Q, options=opts + [NONE]))
    if "period" in slots and vocab.periods:
        opts = shortlist(vocab.periods, line, MAX_CHOICES, embed)
        qs.append(Q("period", "choice", PERIOD_Q, options=opts + [NONE]))
    if "segment" in slots and vocab.segments:
        opts = shortlist(vocab.segments, line, MAX_CHOICES, embed)
        qs.append(Q("segment", "choice", SEGMENT_Q, options=opts + [NONE]))
    if "compare_to" in slots:
        qs.append(Q("compare_to", "choice", COMPARE_Q, options=COMPARES + [NONE]))
    if "basis" in slots:
        qs.append(Q("basis", "choice", BASIS_Q, options=BASES + [NONE]))
    return qs


GATE_KIND_Q = ("What is the last line? question = asks for information; what-if = asks what a "
               "number would become under a different assumption; request = asks to show, walk "
               "through or send something; objection = pushes back on a number, price, claim or "
               "plan; none = small talk, statements, agreement, filler.")
GATE_NEEDS_Q = ("Answering the last line well needs a looked-up number, date, name or fact from "
                "the decks, models, documents or public sources.")
GATE_COMPLETE_Q = "The last line is a complete thought (not cut off mid-sentence)."
GATE_KINDS = ["question", "what-if", "request", "objection", "none"]


def gate_questions() -> list[Q]:
    """`Brain._gate_decider`'s own three, so the slots ride in the batch that already runs."""
    return [Q("kind", "choice", GATE_KIND_Q, options=GATE_KINDS),
            Q("needs_lookup", "noul", GATE_NEEDS_Q),
            Q("complete", "noul", GATE_COMPLETE_Q)]


def prior_state(prior: dict | None) -> str:
    """The two paragraphs that tell the model what a follow-up would be following.

    Split out of `state_for` so `Brain._gate_decider` can append them to the gate's own state --
    which carries the mode and the speaker and is what `pregate.parse_gate_state` reads -- instead
    of rebuilding a second one that could drift from this.
    """
    if not prior:
        return ""
    was = ", ".join(f"{k}={v}" for k, v in (prior.get("slots") or {}).items() if v)
    return "\n\n".join([
        f'The PREVIOUS ask was: "{prior.get("line", "")}"' + (f"\nIt resolved to: {was}" if was else ""),
        "The line below may be a follow-up that changes ONE of those and leaves the "
        "rest standing. Answer for the line's OWN words, never for the previous "
        "ask's: where the line names something, that is the answer even when the "
        "previous ask named a different one, and where the line is silent the "
        "answer is the last option. Do not repeat a previous value back."])


def state_for(line: str, prior: dict | None = None, context: str = "") -> str:
    """The shared state string. `prior` carries the ask before this one, which is what makes a
    follow-up resolvable without a rewriter: the model sees what "and for Q2?" is a follow-up to."""
    head = ("A consultant is on a call with a client, with the client's own spreadsheets and decks "
            "open. Read the last line and say what figure it asks for.")
    parts = [head]
    if prior:
        parts.append(prior_state(prior))
    if context:
        parts.append(context)
    parts.append(f'THE LINE: "{line}"')
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- extraction
def _pick(out: dict, name: str, options_present: bool = True) -> Slot:
    d = out.get(name)
    if d is None:
        return Slot()
    value = str(d.value)
    if value == NONE or not options_present:
        return Slot(None, float(d.p), "jev")
    return Slot(value, float(d.p), "jev")


def _keyed(value: str | None, options: list[str], keys: list[str]) -> str | None:
    if not value:
        return None
    try:
        return keys[options.index(value)]
    except ValueError:
        return value


def read_ask(out: dict, line: str, vocab: Vocab | None = None, prior: dict | None = None,
             gate: bool = False, extra: dict[str, Slot] | None = None) -> Ask:
    """One decider response into an `Ask`: the answers that are there, and nothing invented.

    `extra` carries slots answered somewhere other than in the response -- the logistic heads for
    `basis` and `compare_to` -- so a slot that rode on a head is filled exactly as a slot that
    rode in the request would be. `Brain._gate_decider` shares this with `extract_ask`, so the
    batched path and the experiment's path read an answer the same way.
    """
    ask = Ask(line=line)
    if gate:
        ask.kind = str(out["kind"].value) if "kind" in out else None
        ask.needs_lookup = bool(out["needs_lookup"].value) if "needs_lookup" in out else None
        ask.complete = bool(out["complete"].value) if "complete" in out else None
    ask.metric = _pick(out, "metric")
    ask.period = _pick(out, "period")
    ask.segment = _pick(out, "segment")
    cmp_slot = _pick(out, "compare_to")
    ask.compare_to = Slot(_keyed(cmp_slot.value, COMPARES, COMPARE_KEYS), cmp_slot.p, cmp_slot.source)
    basis_slot = _pick(out, "basis")
    ask.basis = Slot(_keyed(basis_slot.value, BASES, BASIS_KEYS), basis_slot.p, basis_slot.source)
    for name, slot in (extra or {}).items():
        if name in Ask.SLOTS and slot is not None:
            setattr(ask, name, slot)
    ask.grain = grain_of(ask.period.value, vocab)
    if prior:
        ask = inherit(ask, prior)
    return ask


async def extract_ask(decider: Decider, line: str, vocab: Vocab, prior: dict | None = None,
                      gate: bool = True, embed=None, context: str = "") -> Ask:
    """One request: the five slots, and optionally the gate's own three alongside them.

    `prior` is `{"line": str, "slots": {...}}` for the ask before this one. It is used twice: the
    state tells the model what a follow-up would be following, and `inherit` fills whatever the
    line left unsaid once the answer is back.
    """
    if not (line or "").strip():
        return Ask(line=line)
    qs = ask_questions(vocab, line, prior, embed)
    if gate:
        qs = gate_questions() + qs
    t0 = time.time()
    out = await decider.decide(state_for(line, prior, context), qs)
    ms = int((time.time() - t0) * 1000)
    ask = read_ask(out, line, vocab, prior, gate=gate)
    ask.latency_ms = ms
    return ask


def inherit(ask: Ask, prior: dict | None, slots: tuple[str, ...] = ("metric", "period", "segment", "basis")) -> Ask:
    """Fill the slots the line did not restate from the ask before it. 0 ms, no model call.

    This is the whole of E2 proposal 6. "And for Q2?" restates `period` and nothing else; "same for
    EMEA" restates `segment`; "what about the plan?" restates `period` and `basis`. Everything the
    follow-up leaves unsaid is still true, and it is still typed, so there is nothing to rewrite.

    `compare_to` is deliberately not inherited: a comparison is asked once and does not carry.
    """
    if not prior:
        return ask
    was = prior.get("slots") or {}
    for name in slots:
        cur: Slot = getattr(ask, name)
        if cur.value or not was.get(name):
            continue
        setattr(ask, name, Slot(was[name], float(prior.get("p", {}).get(name, 0.0)), "inherited"))
    ask.grain = grain_of(ask.period.value)
    return ask


# ----------------------------------------------------------------------------- the targets
def _table_rows(index: Any, table: str) -> list[dict]:
    return [r for r in index.sql(f'SELECT name, label, letter, ord, type FROM _columns '
                                 f"WHERE table_name = '{table}' ORDER BY ord", limit=200)
            if "error" not in r]


def _q(s: str) -> str:
    return str(s).replace("'", "''")


def evidence_targets(ask: Ask, index: Any, limit: int = 6) -> list[dict]:
    """Where the answer lives, as targets `Brain._evidence_for_target` already renders.

    Order is the order the pre-pack should pack them in:

      1. `row`    the metric's row inside the table the segment names -- the figure and its own
                  period neighbours, which is the smallest block that can hold the answer
      2. `column` the period's column -- the same period across every metric, which is what an
                  objection ("that EMEA number feels too high") needs to argue with
      3. `value`  the segment as a filter, when the segment is a value in a text column
      4. `table`  the named table, when the metric did not resolve to a row

    Nothing is invented: every target names a table, a row or a column the index holds, so a
    target that cannot be built is simply not returned.
    """
    out: list[dict] = []
    seen: set[str] = set()

    def add(t: dict) -> None:
        key = repr(sorted(t.items(), key=lambda kv: kv[0]))
        if key not in seen and len(out) < limit:
            seen.add(key)
            out.append(t)

    metric, period, segment = ask.metric.value, ask.period.value, ask.segment.value
    tables = [r for r in index.sql("SELECT name, file, sheet, title FROM _tables", limit=200)
              if "error" not in r]
    by_name = {t["name"]: t for t in tables}

    # which tables the segment allows; empty means "any"
    allowed: set[str] = set()
    seg_values: list[dict] = []
    if segment:
        low = segment.lower()
        for t in tables:
            if low in (str(t.get("sheet") or "").lower(),
                       str(t.get("name") or "").replace("_", " ").lower()):
                allowed.add(t["name"])
        if not allowed:
            for t in tables:
                cols = [c for c in _table_rows(index, t["name"]) if c["type"] == "TEXT"]
                for c in cols:
                    hit = [r for r in index.sql(
                        f'SELECT _row FROM "{t["name"]}" WHERE lower("{c["name"]}") = \'{_q(low)}\'',
                        limit=4) if "error" not in r]
                    if hit:
                        seg_values.append({"kind": "value", "table": t["name"], "column": c["name"],
                                           "value": segment, "file": t.get("file"),
                                           "sheet": t.get("sheet")})
                        allowed.add(t["name"])
                        break

    # 1. the metric's row, preferring a table the segment allows
    rows: list[dict] = []
    if metric:
        for t in tables:
            cols = _table_rows(index, t["name"])
            if not cols:
                continue
            first = cols[0]["name"]
            hit = [r for r in index.sql(
                f'SELECT _row FROM "{t["name"]}" WHERE lower("{first}") = \'{_q(metric.lower())}\'',
                limit=4) if "error" not in r]
            for r in hit:
                rows.append({"kind": "row", "table": t["name"], "row": r["_row"],
                             "label": metric, "file": t.get("file"), "sheet": t.get("sheet")})
    rows.sort(key=lambda r: (r["table"] not in allowed, r["table"]))
    for r in rows[:2 if allowed else 3]:
        add(r)

    # 2. the period's column, in the same tables
    if period:
        low = period.lower()
        want = {r["table"] for r in rows} or allowed or {t["name"] for t in tables}
        for name in sorted(want):
            for c in _table_rows(index, name):
                if str(c.get("label") or "").lower() == low or str(c.get("name") or "").lower() == low:
                    t = by_name.get(name, {})
                    add({"kind": "column", "table": name, "column": c["name"],
                         "label": c.get("label"), "letter": c.get("letter"),
                         "file": t.get("file"), "sheet": t.get("sheet")})
                    break

    # 3. the segment as a filter value
    for t in seg_values:
        add(t)

    # 4. the named table, when nothing finer resolved
    if not out:
        for name in sorted(allowed):
            t = by_name.get(name, {})
            add({"kind": "table", "table": name, "file": t.get("file"), "sheet": t.get("sheet")})
    return out


def cell_for(ask: Ask, index: Any) -> str | None:
    """The one cell the ask names, as `file › sheet › ref`, or None.

    `Brain._evidence_for_target` has no `cell` kind, so this is not a target today; it is what a
    `cell` kind would render, and the experiment measures what adding one would buy.
    """
    for t in evidence_targets(ask, index, limit=4):
        if t.get("kind") != "row" or not ask.period.value:
            continue
        cols = _table_rows(index, t["table"])
        low = ask.period.value.lower()
        for c in cols:
            if str(c.get("label") or "").lower() == low or str(c.get("name") or "").lower() == low:
                return f'{t.get("file")} › {t.get("sheet") or t["table"]} › {c.get("letter")}{t["row"]}'
    return None


__all__ = ["Ask", "Slot", "Vocab", "build_vocab", "grain_of", "shortlist", "ask_questions",
           "local_ask", "local_slot",
           "gate_questions", "state_for", "prior_state", "extract_ask", "read_ask", "inherit",
           "evidence_targets", "cell_for", "NONE", "BASES", "BASIS_KEYS", "COMPARES",
           "COMPARE_KEYS"]
