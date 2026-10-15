"""Pre-run: read a spoken ask into an Index call, with no model anywhere in the loop.

`parse_ask(text, levers, prior=None)` turns one line of a call into an `Ask`: which cells to set,
what kind of run answers it (`whatif`, `goalseek`, `compare`, `sweep`, `explain`), which output
cell it is about, and how sure the parse is. Nothing here touches the workbook, the network or a
model. The caller runs the `Ask` through `Index` (`run_ask` does it in one line) and renders the
result as evidence (`evidence_block`).

Why it exists. E4 measured that the Index can compute 100% of 156 labelled what-ifs but that a
client only gets an answer with no model call on 17% of them, because `Brain._prerun_whatif`
needs a digit in the utterance and sets exactly one cell. Everything else a client says -- a
scenario by name, two levers in one breath, "what fee gets us to $10M", "premium versus market",
"from 5 to 15%", "why did costs move" -- falls outside those two rules. This module is the seven
pieces that close that gap:

  1. several levers in one sentence, values paired to them by position and by format
     ("1% fee and 6% churn" puts the rate on the fee and the churn on the churn);
  2. a scenario name is a value ("run the downside", "price at market", "add Fabrikam");
  3. goal-seek phrasing -> a target cell, a wanted value and the lever to solve;
  4. compare phrasing -> two cases, which is also what "X versus Y" means when the two values
     land on the same lever;
  5. sweeps -> one lever at several values ("at 0.5, 1 and 1.5", "from 5 to 15% churn");
  6. explain phrasing -> the output cell the question is about;
  7. a follow-up rides on `prior`, so "and at 1.25?" resolves to the lever of the last ask.

The lever table is the record `presage.ingest.scratch` builds (`Index.levers()`): label, header,
sheet, cell, value, fmt, choices, aliases, and the labelled outputs each lever moves. That last
column is the only map this module has from a spoken metric ("year five EBITDA") to a cell, and
it is enough, because a goal seek or an explain is always about something a lever moves.

Five things decide most of the accuracy, and all five came out of reading the misses on E4's 156
labelled asks rather than out of a design:

  * **Position.** "two thousand sellers and fifteen thousand buyers" is two levers and two
    numbers, and the only thing that says which is which is that a value sits beside its noun.
    `spoken_numbers` keeps the token span of every number so the pairing can be greedy on the
    word gap; `brain._asked_values` returns bare floats and cannot.
  * **Format.** "one percent fee" names no lever the word matcher will accept ("fee" is in
    thirty labels), but only one *percentage* lever has "fee" in its name. A weak name match
    plus a compatible format is evidence; either alone is not.
  * **Magnitude.** "design 300" fits `Design rate, per hour` (250) and `Design hours per
    designer per week` (37.5) equally on words. The rate is 1.2x away and the hours 8x, and the
    client meant the rate.
  * **What the words are already doing.** Once "downside" has been read as a value for the
    market scenario, it is not also evidence for `Planning downside, revenue multiplier`.
  * **A change has to be asked for.** "what is revenue per subscriber" names `Per-Subscriber Access
    (S3)` as surely as "switch the retail partners to per-subscriber access" does. A scenario only switches
    when the line asks for a change or already moves another lever.

Pure code has a ceiling: a line that gives a value but never names its lever ("what if we are at
1.25 percent") cannot be resolved from words at all. `augment_with_jev` adds one typed choice
over the lever table (E7: 95.7% top-1 at 252 ms) for exactly those lines, and is measured
separately.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

KINDS = ("whatif", "goalseek", "compare", "sweep", "explain")
MAX_SETS = 4                 # cells one spoken line may move
WEAK_GAP = 3                 # how near a value a weakly-named lever has to be to take it
YEAR_GAP = 5                 # ... and how near a year index has to be to its own input
# Every row of a block answers to the block's title, so the title names a lever only when
# nothing on the row itself matched. Letting it name one at full weight put "four retail partners" on a
# build-cost row under "Build-Up: Data and Retail partner Integrations" and made "added Fabrikam" mean every
# channel toggle in "Channel and Market Data"; adding it as a bonus on top of a row that already
# matched cost a question either way, so there is no bonus (E36).
SECTION_W = 0.5              # the block title as a name of last resort


# --------------------------------------------------------------------------- #
# the Ask
# --------------------------------------------------------------------------- #
@dataclass
class Ask:
    """One parsed ask, ready to run. `kind` None means nothing can be run with no model.

    sets     cell -> value, for a `whatif` (and the case the other kinds start from)
    target   the output cell a `goalseek` solves for or an `explain` is about
    cases    `compare`: [{"label": str, "sets": {cell: value}}], case 0 is the reference
    values   `sweep`: the values to run the lever at
    cell     `sweep`: the lever cell
    goal     `goalseek`: the value the target should reach
    lever    `goalseek`: the lever cell to solve
    watch    output cells worth reporting, best first (never required by the Index)
    """
    sets: dict[str, Any] = field(default_factory=dict)
    kind: str | None = None
    target: str | None = None
    cases: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    why: str = ""
    values: list = field(default_factory=list)
    cell: str | None = None
    goal: float | None = None
    lever: str | None = None
    watch: list[str] = field(default_factory=list)
    levers: list[dict] = field(default_factory=list, repr=False)
    ms: float = 0.0

    def __bool__(self) -> bool:
        return self.kind is not None

    def line(self) -> str:
        """One line for a log: what would run."""
        if self.kind == "whatif":
            return "whatif " + ", ".join(f"{k}={v}" for k, v in self.sets.items())
        if self.kind == "goalseek":
            return f"goalseek {self.lever} -> {self.target}={self.goal}"
        if self.kind == "compare":
            return "compare " + " | ".join(
                ", ".join(f"{k}={v}" for k, v in (c.get("sets") or {}).items()) or "as it stands"
                for c in self.cases)
        if self.kind == "sweep":
            return f"sweep {self.cell} at {self.values}"
        if self.kind == "explain":
            return f"explain {self.target}" + (f" from {list(self.sets)}" if self.sets else "")
        return "nothing to run"


# --------------------------------------------------------------------------- #
# words
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[A-Za-z]+|\d[\d,]*(?:\.\d+)?|%|\$")

_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "at", "by", "with", "as", "is",
    "are", "were", "was", "be", "been", "that", "this", "it", "we", "our", "us", "if", "what",
    "how", "would", "will", "do", "does", "did", "go", "goes", "went", "get", "gets", "got",
    "so", "just", "only", "about", "then", "than", "there", "their", "them", "they", "you",
    "look", "looks", "like", "say", "says", "me", "my", "i", "s", "t", "have", "has", "had",
}

# Words that never identify a lever on their own: every workbook has dozens of them.
_WEAK = {"year", "total", "annual", "per", "rate", "cost", "value", "number", "share", "base",
         "model", "input", "case", "amount", "fee", "price", "revenue", "month", "week", "day"}

_NUMWORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fourty", "fifty", "sixty", "seventy", "eighty",
    "ninety", "hundred", "thousand", "million", "billion", "half", "quarter", "third",
}

_IRREGULAR = {"paid": "pay", "pays": "pay", "paying": "pay", "spent": "spend", "built": "build",
              "took": "take", "taken": "take", "ran": "run", "run": "run", "gave": "give",
              "sold": "sell", "bought": "buy", "held": "hold", "kept": "keep", "left": "leave",
              "grew": "grow", "fell": "fall", "rose": "rise", "lost": "lose", "made": "make"}


def _stem(w: str) -> str:
    """Plurals and the commonest verb endings folded, so "signed" finds "signs"."""
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


def _akin(a: str, b: str) -> bool:
    """Two stems that are the same word ("plac" from placed, "place" from place)."""
    return a == b or a + "e" == b or a == b + "e"


@dataclass
class Tok:
    raw: str
    word: str          # lower case
    stem: str
    i: int             # token index
    start: int         # character offset, for citing the span back
    end: int
    broke: bool = False   # punctuation before this token: a list item, not the same number


def tokens(text: str) -> list[Tok]:
    out: list[Tok] = []
    text = text or ""
    prev_end = 0
    for i, m in enumerate(_TOKEN.finditer(text)):
        raw = m.group(0)
        low = raw.lower()
        gap = text[prev_end:m.start()]
        out.append(Tok(raw, low, _stem(low), i, m.start(), m.end(),
                       broke=bool(gap.strip())))
        prev_end = m.end()
    return out


def _content(toks: list[Tok]) -> dict[str, list[int]]:
    """Stem -> the token positions it appears at; stop words, digits and number words dropped
    (a label that says "two rounds" must not be found by "two thousand sellers")."""
    out: dict[str, list[int]] = {}
    for t in toks:
        if t.word in _STOP or len(t.word) < 2 or t.word.isdigit() or t.word in _NUMWORDS:
            continue
        out.setdefault(t.stem, []).append(t.i)
    return out


def _name_words(name: str) -> list[str]:
    """The content stems of a label, alias or choice, in order."""
    out = []
    for w in re.findall(r"[a-z0-9]+", str(name or "").lower()):
        if w in _STOP or len(w) < 2 or w.isdigit() or w in _NUMWORDS:
            continue
        out.append(_stem(w))
    return out


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE = {"hundred": 100, "thousand": 1_000, "k": 1_000, "million": 1_000_000, "m": 1_000_000,
          "mil": 1_000_000, "billion": 1_000_000_000, "bn": 1_000_000_000}
_FRACTION = {"half": 0.5, "quarter": 0.25, "third": 1.0 / 3.0}
_PCT_WORDS = {"percent", "percentage", "%", "pct"}
_MULT_WORDS = {"times", "x", "multiple"}
_UNIT_AFTER = set(_SCALE) | _PCT_WORDS | _MULT_WORDS | {"basis", "bps", "bp", "per"}


@dataclass
class Num:
    """A number the asker said, with where they said it."""
    value: float
    is_pct: bool = False
    is_year: bool = False           # "in year five": a column, not a value
    is_mult: bool = False           # "eight times"
    scaled: bool = False            # a scale word was spoken ("thousand", "million")
    aside: bool = False             # inside "instead of ...", so it is the value being replaced
    i0: int = 0                     # first token index
    i1: int = 0                     # last token index (inclusive)
    raw: str = ""


def _digits(word: str) -> float | None:
    try:
        return float(word.replace(",", ""))
    except ValueError:
        return None


def spoken_numbers(text: str, toks: list[Tok] | None = None) -> list[Num]:
    """Every number in the line, digits or words, with its unit and its token span.

    Handles what a transcriber actually writes: "one and a half million", "twenty five hundred",
    "seventy five basis points", "half a percent", "a thousand", "1.25 percent", "$4.2M",
    "225 an hour". Punctuation ends a number, so "one, one and a quarter" is two of them and not
    2.25. A bare number in a list takes the unit of the number it is listed with ("fifteen and
    thirty thousand" is 15,000 and 30,000), which is how a range is spoken.
    """
    toks = toks if toks is not None else tokens(text)
    nums: list[Num] = []
    i, n = 0, len(toks)
    while i < n:
        start = i
        total = cur = 0.0
        seen = False
        if toks[i].word == "$" and i + 1 < n and _digits(toks[i + 1].word) is not None:
            i += 1
        val = _digits(toks[i].word)
        if val is not None:
            cur, seen = val, True
            i += 1
        else:
            while i < n and not (i > start and toks[i].broke):
                w = toks[i].word
                nxt = toks[i + 1].word if i + 1 < n else ""
                if w in _ONES:
                    cur += _ONES[w]
                    seen = True
                elif w in _TENS:
                    cur += _TENS[w]
                    seen = True
                elif w == "hundred" and seen:
                    cur = (cur or 1) * 100
                elif w in ("thousand", "million", "billion") and seen:
                    total += (cur or 1) * _SCALE[w]
                    cur = 0.0
                elif w in _FRACTION:
                    cur += _FRACTION[w]
                    seen = True
                elif w in ("and", "a", "an") and (nxt in _FRACTION or (
                        nxt in ("a", "an") and i + 2 < n and toks[i + 2].word in _FRACTION)):
                    pass                          # "one and a half": the fraction is part of it
                elif w in ("a", "an") and not seen and nxt in _SCALE:
                    cur, seen = 1.0, True         # "a thousand"
                elif w == "point" and seen and nxt in _ONES:
                    frac = ""
                    j = i + 1
                    while j < n and toks[j].word in _ONES and _ONES[toks[j].word] < 10:
                        frac += str(_ONES[toks[j].word])
                        j += 1
                    cur = float(f"{int(cur)}.{frac}")
                    i = j
                    continue
                else:
                    break
                i += 1
        if not seen:
            i = start + 1
            continue
        value = total + cur
        last = i - 1
        is_pct = is_mult = False
        scaled = any(t.word in _SCALE and t.word != "hundred" for t in toks[start:i])
        while i < n and not toks[i].broke:
            w = toks[i].word
            if w in ("a", "an") and i + 1 < n and toks[i + 1].word in _UNIT_AFTER:
                i += 1                             # "half a percent"
                continue
            if w in _SCALE and w != "hundred":
                value *= _SCALE[w]
                scaled = True
                last = i
                i += 1
                continue
            if w in _PCT_WORDS or (w == "per" and i + 1 < n and toks[i + 1].word == "cent"):
                is_pct = True
                last = i + 1 if w == "per" else i
                break
            if w == "basis" and i + 1 < n and toks[i + 1].word.startswith("point"):
                value, is_pct = value / 100.0, True
                last = i + 1
                break
            if w in ("bps", "bp"):
                value, is_pct = value / 100.0, True
                last = i
                break
            if w in _MULT_WORDS:
                is_mult = True
                last = i
                break
            break
        if is_pct:
            value = value / 100.0
        i = max(i, last + 1)
        is_year = bool(start and toks[start - 1].stem == "year") or (
            last + 1 < n and toks[last + 1].stem == "year" and value <= 20 and not is_pct)
        nums.append(Num(value, is_pct, is_year, is_mult, scaled,
                        i0=start, i1=last, raw=text[toks[start].start:toks[last].end]))
    periods = period_spans(text)
    if periods:
        for num in nums:
            a, b = toks[num.i0].start, toks[num.i1].end
            if any(pa <= a and b <= pb for pa, pb in periods):
                num.is_year = True                # "Q3", "third quarter", "FY26": a column
    _spread_units(nums, toks)
    _mark_asides(nums, toks)
    return nums


def _spread_units(nums: list[Num], toks: list[Tok]) -> None:
    """"fifteen and thirty thousand", "six to twelve percent": a unit spoken once covers the
    list. Only when nothing but a conjunction separates the two, so "eight percent penetration,
    four retail partners" keeps the four a count."""
    for a, b in reversed(list(zip(nums, nums[1:]))):
        between = [t.word for t in toks[a.i1 + 1:b.i0]]
        if len(between) > 1 or any(w not in ("and", "to", "or", "then") for w in between):
            continue
        if b.is_pct and not a.is_pct and not a.scaled:
            a.value, a.is_pct = a.value / 100.0, True
        elif b.scaled and not a.scaled and not a.is_pct and not b.is_pct:
            scale = 1.0
            for t in toks[b.i0:b.i1 + 1]:
                if t.word in _SCALE and t.word != "hundred":
                    scale *= _SCALE[t.word]
            if scale > 1 and a.value < scale:
                a.value *= scale
                a.scaled = True


def _mark_asides(nums: list[Num], toks: list[Tok]) -> None:
    """"1% instead of the 0.85": the second number is where the model already is, not an ask."""
    for num in nums:
        for t in toks[max(0, num.i0 - 4):num.i0]:
            if t.word in ("instead", "rather", "currently", "today"):
                num.aside = True


# --------------------------------------------------------------------------- #
# levers
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    """A lever the line names, with where in the line it names it."""
    lever: dict
    cell: str
    score: float
    anchors: list[int] = field(default_factory=list)   # token positions of the naming words
    choices: list[tuple] = field(default_factory=list)  # (choice, anchors, share), best first
    rank: int = 0                                      # position in the lever table
    name: str = ""
    weak: bool = False                                 # named only by a word half the table uses
    solid: list[int] = field(default_factory=list)     # anchors that are not common words

    @property
    def fmt(self) -> str:
        return str(self.lever.get("fmt") or "number")

    @property
    def choice(self) -> str | None:
        return self.choices[0][0] if self.choices else None


def cell_of(lever: dict) -> str:
    return f'{lever.get("sheet")}!{lever.get("cell")}'


def _choice_words(choice: str) -> list[str]:
    """"Flat Dues (S1)" -> [flat, due]; "C-corp (New York City)" -> [c, corp, new, york, city]."""
    text = re.sub(r"\(s\d+\)", " ", str(choice or "").lower())
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text) if w not in _STOP and len(w) > 1]


_YES = {"yes", "true", "y", "on", "in"}
_NO = {"no", "false", "n", "off", "out"}
_ADD = {"add", "added", "adding", "include", "included", "including", "scope", "too",
        "also", "bring", "brought", "counted", "opt in"}
_DROP = {"drop", "dropped", "dropping", "remove", "removed", "exclude", "excluded", "without",
         "lose", "lost", "cut", "off", "out", "no"}

# A line only switches a scenario when it asks for a change. "what is revenue per subscriber" names
# `Per-Subscriber Access (S3)` exactly as well as "switch the retail partners to per-subscriber access" does.
_CHANGE = re.compile(
    r"\b(what\s+if|whatif|suppose|supposing|instead|rather\s+than|hypothetical|"
    r"switch(?:ed|ing)?|move[sd]?\s+to|went\s+with|go\s+with|going\s+with|using|used|use\s+the|"
    r"run(?:ning)?\s|ran\s|show\s+me|priced?\s+at|pricing|add(?:ed|ing)?\b|drop(?:ped|ping)?|"
    r"set\s+up\s+as|look(?:s|ed)?\s+like|what\s+about|how\s+about|say\s+on|in\s+scope|"
    r"scenario|case\b|versus|\bvs\b|against|compared|if\s+we|if\s+they|if\s+the|if\s+it|"
    r"if\s+only|were\s+in\b|with\s+no\b)", re.I)


def change_cue(text: str) -> bool:
    return bool(_CHANGE.search(text or ""))


_BLOCK_CACHE: dict[str, frozenset[str]] = {}


# A period said out loud is a column, never a value: "Q3 revenue", "revenue in Q 3", "third
# quarter", "quarter three", "FY26", "H1", "the second half", "by 2030". E70: "remind me what Q3
# revenue came in at" became a what-if that set revenue to 3,000 because "Q3" tokenised into
# "Q" and "3" and the bare 3 landed on the revenue lever.
_ORD = r"(?:first|second|third|fourth|1st|2nd|3rd|4th)"
_PERIOD = re.compile(
    r"\b(?:q\s?(?:[1-4]|one|two|three|four)|h\s?(?:[12]|one|two)|(?:fy|cy)\s?\d{2,4}"
    r"|" + _ORD + r"\s+(?:quarter|half)|quarter\s+(?:[1-4]|one|two|three|four)"
    r"|(?<!to\s)(?<!at\s)(?:19|20)\d{2})(?![\d,.%]|\s*(?:percent|k\b|m\b|million|thousand|billion))",
    re.I)


def period_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the periods the line names (quarters, halves, fiscal years, years)."""
    return [(m.start(), m.end()) for m in _PERIOD.finditer(text or "")]


def blank_periods(text: str) -> str:
    """The line with its period expressions replaced by spaces, same length, for number
    extractors that work on text rather than tokens (brain._asked_values)."""
    out = list(text or "")
    for a, b in period_spans(text):
        for k in range(a, b):
            out[k] = " "
    return "".join(out)


# A line that asks what a figure was is a lookup, not a what-if, unless it also asks for a change
# ("suppose churn came in at five instead of eight" keeps its cue).
_LOOKUP = re.compile(
    r"\b(?:remind\s+me|what\s+(?:was|were|did)\b|how\s+much\s+(?:was|were|did)\b|"
    r"came\s+in\s+at|ended\s+(?:up|at)|actuals?\b|last\s+(?:quarter|year|month|time)|"
    r"again\s*\??\s*$)", re.I)


def lookup_cue(text: str) -> bool:
    return bool(_LOOKUP.search(text or ""))


def _block_names(lever: dict) -> frozenset[str]:
    """The names a lever inherits from its block title rather than off its own row.

    They are in `aliases` because the deep pass, the option text and the grid's questions all
    want them; here they are scored as the block, not as the row.
    """
    section = str(lever.get("section") or "")
    if not section:
        return frozenset()
    got = _BLOCK_CACHE.get(section)
    if got is None:
        try:
            from presage.ingest.scratch import section_aliases
            got = frozenset(section_aliases(section))
        except Exception:  # noqa: BLE001  the parse must not need the ingest package
            got = frozenset()
        _BLOCK_CACHE[section] = got
    return got


def match_levers(text: str, levers: list[dict], toks: list[Tok] | None = None,
                 floor: float = 0.24) -> list[Hit]:
    """Every lever the line plausibly names, best first, each with the words that named it.

    Scored like `Brain._lever_matches` -- the share of a label's weight the line covers, each
    word weighted by how rare it is across the table -- and then five things that matcher does
    not do: a text lever also matches on one of its own values ("downside", "flat dues"); the
    matched word positions are kept, so a value can be paired to the lever nearest it; a rare
    word counts for more than its share of a long label ("Finro private *fintech*"); the block
    title the row sits under counts too, at a discount, so "1.25% of GMV" reaches a row labelled
    "Median platform fee on accounts over $1M" under `Structure 2: Percentage of GMV`; and a
    match too weak to stand on its own is kept and flagged, because "one percent fee" has only
    one percentage lever with "fee" in its name and that is the whole of the evidence.
    """
    if not levers:
        return []
    toks = toks if toks is not None else tokens(text)
    seen = _content(toks)
    if not seen:
        return []
    df: dict[str, int] = {}
    names_of: list[list[tuple[str, list[str]]]] = []
    for lv in levers:
        block = _block_names(lv)
        names = [("label", _name_words(lv.get("label")))]
        if lv.get("fmt") != "text":
            names += [("section" if a in block else "alias", _name_words(a))
                      for a in (lv.get("aliases") or []) if a]
        names += [("section", _name_words(lv.get("section")))]
        names = [(k, ws) for k, ws in names if ws]
        names_of.append(names)
        # Block titles are counted out of the frequencies. Every row under a heading carries
        # its words, so counting them said "engineering" was common and lifted a partial match
        # on "Blended engineering rate" over the strong threshold on a line about the design
        # rate (E36). The section is extra evidence, not a vote on how rare a word is.
        for w in {w for k, ws in names if k != "section" for w in ws}:
            df[w] = df.get(w, 0) + 1
    weight = {w: 1.0 / math.log(2.0 + c) for w, c in df.items()}

    def cover(words: list[str]) -> tuple[float, list[int], int, list[int]]:
        total = sum(weight.get(w, 1.0) for w in words) or 1.0
        got, anchors, rare, missed = 0.0, [], 0, 0
        solid: list[int] = []
        for w in words:
            pos = seen.get(w) or next((p for s, p in seen.items() if _akin(s, w)), None)
            if not pos:
                missed += df.get(w, 99) <= 3 and w not in _WEAK
                continue
            got += weight.get(w, 1.0)
            anchors += pos
            if w not in _WEAK:
                solid += pos
            if df.get(w, 99) <= 3 and w not in _WEAK:
                rare += 1
        return got / total, anchors, (rare if solid else 0) - min(2, missed), solid

    hits: list[Hit] = []
    for rank, (lv, names) in enumerate(zip(levers, names_of)):
        best, anchors, rare, label, solid = 0.0, [], 0, "", []
        sec, sec_anchors = 0.0, []
        for kind, words in names:
            cov, pos, r, sol = cover(words)
            if kind == "section":
                if cov > sec:
                    sec, sec_anchors = cov, pos
                continue
            # the tie has to be a tie to a tolerance: two names both fully covered come out
            # 1.0 and 0.9999999999999999 depending on how many weights were added up, and the
            # label losing that by 1e-16 took "annual fee per club" down to "annual fee",
            # which is two _WEAK words and therefore a weak hit (E36)
            if cov > best + 1e-9 or (cov > best - 1e-9 and kind == "label" and pos):
                best, anchors, rare, label, solid = cov, pos, r, " ".join(words), sol
        own = bool(anchors)            # the row's own words, not the ones its whole block shares
        choices = (_named_choices(lv, seen, toks, named=own)
                   if lv.get("fmt") == "text" else [])
        if not own and not choices and sec:
            best, anchors = sec * SECTION_W, sec_anchors      # the block, for want of anything
        if not anchors and not choices:
            continue
        head = _header_named(lv, seen)
        score = best + 0.08 * max(-2, min(2, rare)) + (0.15 if head else 0.0)
        if choices:
            score = max(score, 0.5 + 0.3 * choices[0][2]) + 0.02 * len(set(anchors))
        if lv.get("active") is False:
            score -= 0.03                  # a tie-break only: an unselected option is still one
        if not lv.get("value") and lv.get("fmt") != "text":
            score -= 0.30                  # a hard-coded zero is a placeholder, not a lever
        strong = bool(choices) or (
            score >= 0.33 and ((rare > 0 and best >= 0.33)
                               or (best >= 0.5 and any(w not in _WEAK for w in label.split()))))
        if not strong and (best < floor or score <= 0):
            continue
        if head:
            anchors = anchors + [p for w in _name_words(lv.get("header")) for p in seen.get(w, [])]
        anchors += [a for c in choices[:1] for a in c[1]]
        hits.append(Hit(lv, cell_of(lv), round(score, 3), sorted(set(anchors)), choices, rank,
                        label, weak=not strong, solid=sorted(set(solid))))
    hits.sort(key=lambda h: (h.weak, -h.score, h.rank))
    return _dedupe(hits)


def _dedupe(hits: list[Hit]) -> list[Hit]:
    """One cell once; one input once when the second is the same label under another scenario
    column; and one setting once when two switches answer to the same words ("Fabrikam" is both the
    channel and the buyers toggle, and "added Fabrikam" means the channel)."""
    out: list[Hit] = []
    by_cell: set[str] = set()
    by_label: dict[tuple, Hit] = {}
    for h in hits:
        if h.cell in by_cell:
            continue
        key = (str(h.lever.get("sheet")), str(h.lever.get("label")).lower(), h.fmt)
        twin = by_label.get(key)
        if twin is not None and abs(twin.score - h.score) < 0.05:
            continue
        if h.fmt == "text" and h.anchors and any(
                o.fmt == "text" and set(h.anchors) <= set(o.anchors) and o.score >= h.score - 0.05
                for o in out):
            continue
        by_cell.add(h.cell)
        by_label.setdefault(key, h)
        out.append(h)
    return out


def _named_choices(lever: dict, seen: dict[str, list[int]], toks: list[Tok],
                   named: bool = True) -> list[tuple]:
    """Which of a text lever's values the line names, best first, with where.

    A one-word value ("Market") has to be said as that word: "ongoing marketing" is not the
    market scenario. A value of several words may be two thirds there ("per subscriber" for
    `Per-Subscriber Access (S3)`), but its words have to sit together, or "flat dues versus the GMV
    fee" reads as `Flat Partner Fee (S4)` as well.
    """
    words_at = {t.word: [] for t in toks}
    for t in toks:
        words_at.setdefault(t.word, []).append(t.i)
    out: list[tuple] = []
    cur = str(lever.get("value") or "").strip().lower()
    choices = [str(c) for c in (lever.get("choices") or [])]
    if {c.lower() for c in choices} <= (_YES | _NO):
        if not named:
            return []                       # "add Fabrikam" is about the channel it names, not every one
        words = {t.word for t in toks} | {t.stem for t in toks}
        add, drop = words & _ADD, words & _DROP
        if add or drop:
            want = "no" if (drop and not add) else "yes"
            for c in choices:
                if c.strip().lower() == want:
                    return [(c, [], 0.8)]
        return []
    for c in choices:
        words = _choice_words(c)
        if not words:
            continue
        anchors, got = [], 0
        for w in words:
            pos = (words_at.get(w) or seen.get(w)
                   or next((p for s, p in seen.items() if _akin(s, w)), None))
            if len(words) == 1 and w not in words_at and _stem(w) not in (
                    {t.stem for t in toks} if w.endswith("s") else set()):
                pos = None                      # a one-word value has to be that word
            if pos:
                got += 1
                anchors += pos
        share = got / len(words)
        if share < 0.6 or not anchors:
            continue
        if max(anchors) - min(anchors) > len(words) + 1:
            continue                            # its words have to be said together
        out.append((c, sorted(set(anchors)), share))
    out.sort(key=lambda t: (-t[2], str(t[0]).strip().lower() == cur, t[1][0] if t[1] else 0))
    return out


def _header_named(lever: dict, seen: dict[str, list[int]]) -> bool:
    """"churn in the downside case" picks the Downside column of a two-column input."""
    words = [w for w in _name_words(lever.get("header")) if w not in _WEAK]
    return bool(words) and all(w in seen for w in words)


# --------------------------------------------------------------------------- #
# outputs: the metric a goal seek or an explain is about
# --------------------------------------------------------------------------- #
def output_table(levers: list[dict]) -> list[dict]:
    """Every labelled output cell the lever table knows, deduped, richest first.

    `cited` is how many levers move it, which is what makes a cell part of the headline block:
    the rows every input reaches are the rows a client quotes.
    """
    out: dict[str, dict] = {}
    for rank, lv in enumerate(levers):
        for o in lv.get("outputs") or []:
            cell = str(o.get("cell") or "")
            if not cell:
                continue
            if cell in out:
                out[cell]["cited"] += 1
                continue
            out[cell] = {"cell": cell, "label": o.get("label"), "fmt": o.get("fmt"),
                         "value": o.get("value"), "rank": rank, "cited": 1}
    return list(out.values())


def headline(outputs: list[dict], limit: int = 12) -> list[str]:
    """The cells most of the levers move: the summary block, without knowing which sheet it is."""
    ranked = sorted(outputs, key=lambda o: (-o.get("cited", 0), o.get("rank", 0)))
    return [o["cell"] for o in ranked[:limit]]


def match_output(text: str, outputs: list[dict], toks: list[Tok] | None = None,
                 near: list[int] | None = None, floor: float = 0.4) -> tuple[dict | None, float]:
    """The output cell a phrase names. Coverage decides, and the weight actually matched breaks
    the ties, so "the cost to serve a subscriber" is `Cost to serve subscribers` and not `Subscribers`.

    `near` is the token window that names the metric (the words after "gets us to", or the words
    that did not name the lever), so "what fee gets us to ten million of subscriber revenue" does
    not answer with the fee.
    """
    toks = toks if toks is not None else tokens(text)
    if near:
        keep = set(near)
        window = [t for t in toks if t.i in keep]
    else:
        window = toks
    seen = _content(window)
    if not seen or not outputs:
        return None, 0.0
    df: dict[str, int] = {}
    words_of = []
    for o in outputs:
        ws = _name_words(o.get("label"))
        words_of.append(ws)
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    weight = {w: 1.0 / math.log(2.0 + c) for w, c in df.items()}
    best, score, key = None, 0.0, (0.0, 0.0, -10**6)
    for o, ws in zip(outputs, words_of):
        if not ws:
            continue
        total = sum(weight.get(w, 1.0) for w in ws) or 1.0
        hit = [w for w in ws if w in seen or any(_akin(s, w) for s in seen)]
        if not any(not _TIMEY.fullmatch(w) for w in hit):
            continue                      # "year" alone does not name a metric
        got = sum(weight.get(w, 1.0) for w in hit)
        cov = got / total
        if cov < floor:
            continue
        here = (round(got, 3), round(cov, 3), -o.get("rank", 0))
        if here > key:
            best, score, key = o, cov, here
    return best, round(max(score, 0.0), 3)


# --------------------------------------------------------------------------- #
# what kind of ask it is
# --------------------------------------------------------------------------- #
_GOAL = re.compile(
    r"\b(?:what|which|how\s+much|how\s+many|how\s+big|how\s+high|how\s+low)\b[^.?]{0,70}?\b("
    r"gets?\s+(?:\w+\s+){0,4}?(?:to|down\s+to|up\s+to|under|over)\b|to\s+get\b|getting\s+to\b|"
    r"need(?:s|ed)?\b|require[ds]?\b|(?:have|has|had)\s+to\s+be\b|would\s+it\s+take\b|"
    r"puts?\s+(?:\w+\s+){0,4}?(?:at|to)\b|holds?\s+(?:\w+\s+){0,4}?(?:at|to)\b|"
    r"keeps?\s+(?:\w+\s+){0,4}?(?:at|to)\b|brings?\s+(?:\w+\s+){0,4}?(?:to|down)\b|"
    r"to\s+hit\b|to\s+reach\b|break\s+even\b)", re.I)
_GOAL_CUE = re.compile(
    r"\b(gets?|getting|need(?:s|ed)?|require[ds]?|(?:have|has|had)\s+to\s+be|puts?|holds?|"
    r"keeps?|brings?|hit|reach|break\s+even|take)\b", re.I)
_COMPARE = re.compile(
    r"\b(versus|vs\.?|against|compared\s+(?:to|with)|difference\s+between|gap\s+between|"
    r"side\s+by\s+side|how\s+much\s+(?:bigger|smaller|more|less|better|worse)|"
    r"how\s+does\s+that\s+compare|which\s+is\s+better)\b", re.I)
_SWEEP = re.compile(
    r"\b(range|anywhere\s+from|across\s+(?:all|the|every)|sensitivity|each\s+of|"
    r"from\s+(?:[\w.$]+\s+){1,3}?(?:to|through)\s|"
    r"between\s+(?:[\w.$]+\s+){1,3}?and\s)", re.I)
_EXPLAIN = re.compile(
    r"\b(why\s+(?:did|does|do|is|are|has|have|would|didn't|doesn't)|what\s+(?:is|are|s)\s+driving|"
    r"what\s+drives|what\s+makes\s+up|what\s+goes\s+into|what\s+sits\s+between|"
    r"walk\s+me\s+through|talk\s+me\s+through|how\s+(?:does|do|did)\s+.{0,40}?"
    r"(?:feed|flow|get|turn|roll|end\s+up|affect|drive)|where\s+does\s+.{0,40}?come\s+from|"
    r"what\s+(?:is|s)\s+behind|how\s+is\s+.{0,30}?(?:built|calculated|computed|derived))", re.I)
_FOLLOW = re.compile(r"^\s*(?:and|what about|how about|ok|okay)\b|\b(?:that|those|the same|it)\b",
                     re.I)


def _follow_up(text: str) -> bool:
    t = (text or "").strip()
    return len(t.split()) <= 12 and bool(_FOLLOW.search(t))


# --------------------------------------------------------------------------- #
# pairing values to levers
# --------------------------------------------------------------------------- #
def _compatible(num: Num, hit: Hit) -> bool:
    """Can this number be what that lever is being set to?"""
    fmt = hit.fmt
    if fmt == "text":
        return False
    if num.is_pct:
        return fmt == "pct"
    if num.is_year:
        cur = hit.lever.get("value")
        return (bool(_TIMEY.search(str(hit.lever.get("label", "")))) and fmt in ("count", "number")
                and isinstance(cur, (int, float)) and abs(cur) <= 24)
    if num.is_mult:
        return fmt in ("number", "count")
    if fmt == "pct":
        return num.value <= 100.0
    return True


def _as_value(num: Num, hit: Hit) -> float:
    """The number as the cell would hold it: a rate out of a bare "twelve", money out of "140"
    when the cell holds 153,000 and the client dropped the thousands."""
    v = float(num.value)
    if hit.fmt == "pct" and not num.is_pct and v > 1.0:
        v = v / 100.0
    cur = hit.lever.get("value")
    if (hit.fmt in ("money", "count") and not num.is_pct and not num.scaled
            and isinstance(cur, (int, float)) and cur and v):
        if abs(cur) / abs(v) >= 50 and abs(abs(cur) / (v * 1000.0) - 1) < 9:
            v *= 1000.0
    return v


def _off_by(value: float, cur) -> int:
    """How far a spoken value is from where the cell sits, in bands. A client moves an input,
    they do not usually replace it with something a hundred times bigger."""
    if not isinstance(cur, (int, float)) or isinstance(cur, bool) or not cur or not value:
        return 0
    r = abs(float(value) / float(cur))
    r = r if r >= 1 else 1 / r
    return 0 if r <= 4 else (1 if r <= 25 else 2)


def _pair(nums: list[Num], hits: list[Hit], text: str = "", *, keep_current: bool = False,
          limit: int = MAX_SETS) -> tuple[dict[str, Any], list[Hit], list[Num]]:
    """Values onto levers: nearest in the sentence first, one value per lever.

    The distance is counted in words, because a value sits beside the thing it is a value for --
    "two thousand sellers and fifteen thousand buyers" -- and greedy assignment on that distance
    is what stops the fee landing on the churn cell. A word that two candidates share is a worse
    anchor than one only this lever has, a value that is 8x away from where the cell sits is a
    worse fit than one 1.2x away, and a lever named only by a common word has to be right beside
    its value to take it at all.

    A named scenario value is settled first and its words are reserved, because "downside" is a
    value for the market scenario before it is a weak name match for `Planning downside, revenue
    multiplier`. It is only committed when the line asks for a change, or when something else in
    the line already moved: "what is revenue per subscriber" names `Per-Subscriber Access (S3)` too.
    """
    used_n: set[int] = set()
    used_h: set[str] = set()
    spoken: set[int] = set()                  # tokens already doing a job
    sets: dict[str, Any] = {}
    taken: list[Hit] = []
    shared = {a for i, h in enumerate(hits) for a in h.anchors
              if any(a in o.anchors for j, o in enumerate(hits) if j != i)}
    picks: list[tuple[Hit, tuple]] = []
    for h in hits:
        if not h.choices:
            continue
        cur = str(h.lever.get("value") or "").strip().lower()
        pick = h.choices[0] if keep_current else next(
            (c for c in h.choices if str(c[0]).strip().lower() != cur), None)
        if pick is not None:
            picks.append((h, pick))
    reserved = {a for _h, p in picks for a in p[1]}

    def commit(h: Hit, value, said: list[int] | None = None) -> None:
        sets[h.cell] = value
        used_h.add(h.cell)
        spoken.update(h.anchors if said is None else said)
        taken.append(h)

    pairs: list[tuple] = []
    for k, num in enumerate(nums):
        if num.aside:
            continue
        for h in hits:
            if not _compatible(num, h):
                continue
            if h.anchors and set(h.anchors) <= reserved and h.fmt != "text":
                continue                      # those words are a scenario's name
            near = h.anchors
            if num.is_year:
                near = [a for a in h.solid if not (num.i0 - 1 <= a <= num.i1 + 1)]
                if not near:
                    continue              # "in year five" is the number's own word
            uniq = [a for a in near if a not in shared]
            gap = min([abs(a - num.i0) if a < num.i0 else abs(a - num.i1)
                       for a in (uniq or near)] or [99])
            cost = gap + (0 if uniq else 2)   # only words other candidates share point here
            value = _as_value(num, h)
            cur = h.lever.get("value")
            if not keep_current and isinstance(cur, (int, float)) and _same(value, cur):
                continue
            if h.weak and (gap > WEAK_GAP or not h.lever.get("value")):
                continue
            if num.is_year and gap > YEAR_GAP:
                continue
            beside = -sum(1 for a in (h.solid or h.anchors)
                          if num.i0 - 3 <= a <= num.i1 + 3)
            pairs.append((cost + 2 * _off_by(value, cur) + 4 * int(h.weak), beside, -h.score,
                          h.rank, k, h, value))
    pairs.sort(key=lambda p: p[:5])
    for _cost, _b, _s, _r, k, h, value in pairs:
        if k in used_n or h.cell in used_h or len(sets) >= limit:
            continue
        if h.anchors and set(h.anchors) <= spoken:
            continue                          # every word naming it is already doing another job
        commit(h, value)
        used_n.add(k)
    asked = change_cue(text)
    for h, pick in picks:
        if h.cell in used_h or len(sets) >= limit:
            continue
        if not asked and not (sets or pick[2] >= 1.0):
            continue                          # nobody asked for a switch
        if pick[1] and set(pick[1]) & spoken:
            continue
        if any(n.i0 <= a <= n.i1 for a in pick[1] for n in nums):
            continue                          # those words are a number's unit
        commit(h, pick[0], pick[1])
    spare = [n for k, n in enumerate(nums) if k not in used_n and not n.aside and not n.is_year]
    return sets, taken, spare


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9 + 0.005 * max(abs(float(a)), abs(float(b)))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# relative moves: "a year later", "double that"
# --------------------------------------------------------------------------- #
_LATER = re.compile(r"\b(later|slip(?:s|ped|ping)?|delay(?:s|ed)?|push(?:ed)?\s+back|"
                    r"further\s+out|behind)\b", re.I)
_EARLIER = re.compile(r"\b(earlier|sooner|ahead\s+of|pull(?:ed)?\s+(?:forward|in)|"
                      r"bring(?:s|ing)?\s+forward)\b", re.I)
_DOUBLE = re.compile(r"\b(double|twice|two\s+times)\b", re.I)
_HALVE = re.compile(r"\b(halve[ds]?|half\s+(?:that|of\s+that|as\s+much))\b", re.I)
_TIMEY = re.compile(r"\b(year|month|week|quarter|cycle|period|date|timing|day)s?\b", re.I)


def _relative(text: str, hit: Hit, nums: list[Num]) -> float | None:
    """A move said against where the model already is, with no absolute value spoken."""
    cur = hit.lever.get("value")
    if not isinstance(cur, (int, float)) or isinstance(cur, bool):
        return None
    step = 1.0
    for num in nums:
        if num.is_year and num.value <= 5:
            step = float(num.value)
            break
    if _LATER.search(text):
        return cur + step
    if _EARLIER.search(text):
        return cur - step
    if _DOUBLE.search(text):
        return cur * 2
    if _HALVE.search(text):
        return cur / 2
    return None


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #
def parse_ask(text: str, levers: list[dict], prior: dict | None = None,
              outputs: list[dict] | None = None) -> Ask:
    """One spoken line -> the Index call that answers it, or an empty Ask.

    `levers` is `Index.levers()`. `prior` is the previous ask, `{"question", "sets", "lever"}`,
    which is what a follow-up ("and at 1.25?") and a "why did that move" resolve against.
    `outputs` is `output_table(levers)`, passed in when it is already built (it is pure and
    cacheable per workbook).
    """
    t0 = time.perf_counter()
    text = text or ""
    if not text.strip() or not levers:
        return Ask(why="nothing to parse")
    toks = tokens(text)
    nums = spoken_numbers(text, toks)
    outs = outputs if outputs is not None else output_table(levers)
    hits = match_levers(text, levers, toks)
    prior_sets = dict((prior or {}).get("sets") or {})
    prior_lever = (prior or {}).get("lever")
    if prior_lever and _follow_up(text) and not any(
            h.cell == cell_of(prior_lever) for h in hits):
        hits = [Hit(prior_lever, cell_of(prior_lever), 0.6, [t.i for t in toks[:1]], [], 0,
                    str(prior_lever.get("label") or ""))] + hits

    kind = _kind(text, hits, nums, toks)
    if kind == "explain":
        ask = _explain(text, toks, nums, hits, outs, prior_sets, prior_lever)
    elif kind == "goalseek":
        ask = _goalseek(text, toks, nums, hits, outs)
    elif kind == "compare":
        ask = _compare(text, toks, nums, hits)
    elif kind == "sweep":
        ask = _sweep(text, toks, nums, hits)
    elif lookup_cue(text) and not change_cue(text):
        ask = Ask(why="a lookup, not a what-if")   # "remind me what Q3 revenue came in at"
    else:
        ask = _whatif(text, toks, nums, hits)
    ask.levers = [h.lever for h in hits[:6]]
    if ask.kind and not ask.watch:
        named, _ = match_output(text, outs, toks)
        ask.watch = _watch(ask, hits, named, headline(outs))
    ask.ms = round((time.perf_counter() - t0) * 1000, 3)
    return ask


def _kind(text: str, hits: list[Hit], nums: list[Num], toks: list[Tok]) -> str | None:
    """Which run answers it. Order matters: an explain and a goal seek both look like what-ifs
    with a number in them, and a compare and a sweep are one sentence with a different verb."""
    if _EXPLAIN.search(text):
        return "explain"
    live = [n for n in nums if not n.is_year]
    if _GOAL.search(text) and live:
        return "goalseek"
    if _COMPARE.search(text):
        return "compare"
    if _SWEEP.search(text) and (len(live) >= 2 or _sweepable_text(hits)):
        return "sweep"
    if _listed(live, toks) >= 3:
        return "sweep"                     # "at 0.5, 1 and 1.5 percent"
    return None


def _listed(nums: list[Num], toks: list[Tok]) -> int:
    """How many numbers were said as one list: "at one, one and a quarter and one and a half"."""
    best = run = 1
    for a, b in zip(nums, nums[1:]):
        between = [t.word for t in toks[a.i1 + 1:b.i0]]
        run = run + 1 if (len(between) <= 1 and all(
            w in ("and", "to", "or", "then") for w in between)) else 1
        best = max(best, run)
    return best if len(nums) > 1 else 0


def _sweepable_text(hits: list[Hit]) -> bool:
    return any(h.fmt == "text" and len(h.lever.get("choices") or []) > 1 for h in hits)


def _whatif(text: str, toks: list[Tok], nums: list[Num], hits: list[Hit]) -> Ask:
    sets, taken, spare = _pair(nums, hits, text)
    rel = _relatives(text, nums, hits, sets)
    if rel:
        sets.update(rel[0])
        taken += rel[1]
    if not sets:
        return Ask(why="no lever took a value")
    why = ", ".join(f"{h.name or h.cell} ({h.score:.2f})" for h in taken)
    return Ask(sets=sets, kind="whatif", confidence=_confidence(taken, spare),
               why=f"levers named: {why}" + (f"; {len(spare)} value(s) unplaced" if spare else ""))


def _relatives(text: str, nums: list[Num], hits: list[Hit],
               sets: dict) -> tuple[dict, list[Hit]] | None:
    """"Northwind and Contoso both sign a year later": every lever the line names moves, and the
    move is said against where each one sits."""
    if not (_LATER.search(text) or _EARLIER.search(text) or _DOUBLE.search(text)
            or _HALVE.search(text)):
        return None
    timing = bool(_LATER.search(text) or _EARLIER.search(text))
    out, taken = {}, []
    for h in hits[:3]:
        if h.cell in sets or h.weak or len(sets) + len(out) >= MAX_SETS:
            continue
        if timing and not _TIMEY.search(str(h.lever.get("label") or "")):
            continue                        # "Contoso slipped a year" moves a year, not a roster
        value = _relative(text, h, nums)
        if value is None or _same(value, h.lever.get("value")):
            continue
        out[h.cell] = value
        taken.append(h)
    return (out, taken) if out else None


def _goalseek(text: str, toks: list[Tok], nums: list[Num], hits: list[Hit],
              outs: list[dict]) -> Ask:
    """"what fee gets us to ten million of subscriber revenue": the lever is named before the verb,
    the target after it, and the number in between belongs to the target."""
    m = _GOAL_CUE.search(text)
    cut = len([t for t in toks if t.start < (m.start() if m else 0)])
    named = [h for h in hits if h.anchors and min(h.anchors) < max(cut, 1)
             and h.fmt != "text" and h.lever.get("value")]
    if not named:
        return Ask(why="goal seek with no lever to solve")
    goal = None
    for num in nums:
        if num.is_year or num.aside:
            continue
        if goal is None or num.i0 >= cut:
            goal = num
    if goal is None:
        return Ask(why="goal seek with no target value")
    after = [t.i for t in toks if t.i > goal.i1 or (cut <= t.i < goal.i0)] or None
    # The lever and the target are chosen together, and the target counts for more: "what fee"
    # names a dozen fee inputs and only one of them moves subscriber revenue, so the pair that agrees
    # is the pair that was meant.
    best: tuple = (0.0, None, None, 0.0)
    for h in named[:8]:
        table = [{"cell": o["cell"], "label": o.get("label"), "fmt": o.get("fmt"),
                  "value": o.get("value"), "rank": i}
                 for i, o in enumerate(h.lever.get("outputs") or [])]
        t, s = match_output(text, table, toks, near=after)
        if t is not None and not _target_fits(t, goal):
            t, s = None, 0.0
        joint = min(h.score, 1.0) + 1.5 * s + (0.25 if t is not None else 0.0)
        if joint > best[0]:
            best = (joint, h, t, s)
    lever, target, score = best[1], best[2], best[3]
    if lever is None:
        lever = named[0]
    if target is None or not _target_fits(target, goal):
        target, score = match_output(text, outs, toks, near=after)
    if target is None or not _target_fits(target, goal):
        return Ask(why="goal seek: nothing the lever moves is named")
    value = goal.value
    if target.get("fmt") == "pct" and not goal.is_pct and value > 1:
        value = value / 100.0
    return Ask(kind="goalseek", lever=lever.cell, target=target["cell"], goal=value,
               confidence=round(min(1.0, 0.4 + 0.3 * min(lever.score, 1.0) + 0.3 * score), 2),
               watch=[target["cell"]],
               why=f"solve {lever.name or lever.cell} for {target.get('label')} = {goal.raw}")


def _target_fits(target: dict, goal: Num) -> bool:
    """A number in the millions is not a target for a cell that holds a percentage."""
    if target.get("fmt") == "pct" and not goal.is_pct and goal.value > 100:
        return False
    return True


def _compare(text: str, toks: list[Tok], nums: list[Num], hits: list[Hit]) -> Ask:
    """Two cases. The commonest shape by far is two values for one lever -- "premium versus
    market", "three retail partners versus four", "eight and twelve percent churn" -- and the next is one
    value against the model as it stands ("how much bigger is the ask in the downside")."""
    split = _COMPARE.search(text)
    at = len([t for t in toks if t.start < split.start()]) if split else len(toks) // 2
    shared, taken, spare = _pair(nums, hits, text, keep_current=True)
    a, b = _two_sided(text, toks, nums, hits, at, shared)
    if a is None:
        return Ask(why="compare with only one case")
    labels = [_case_label(a), _case_label(b)]
    return Ask(kind="compare", cases=[{"label": labels[0], "sets": a},
                                      {"label": labels[1], "sets": b}],
               sets=dict(a), confidence=_confidence(taken, spare),
               why=f"compare {labels[0]} against {labels[1]}")


def _two_sided(text: str, toks: list[Tok], nums: list[Num], hits: list[Hit], at: int,
               shared: dict) -> tuple[dict | None, dict]:
    """Split the line at the comparison word and read a case off each side. A lever named on one
    side carries over: "three retail partners versus four" says "retail partners" once and means it twice."""
    left = [h for h in (_clip(h, 0, at) for h in hits) if h]
    right = [h for h in (_clip(h, at, len(toks) + 1) for h in hits) if h]
    left_n = [n for n in nums if n.i1 < at]
    right_n = [n for n in nums if n.i0 >= at]
    a, _, _ = _pair(left_n, left or (right if left_n else []), text, keep_current=True, limit=3)
    b, _, _ = _pair(right_n, right or (left if right_n else []), text, keep_current=True, limit=3)
    rest = {k: v for k, v in shared.items() if k not in a and k not in b}
    if a and b:
        return {**rest, **a}, {**rest, **b}
    one = a or b
    if one:
        base = {c: next((h.lever.get("value") for h in hits if h.cell == c), None) for c in one}
        rest = {k: v for k, v in shared.items() if k not in one}
        return {**rest, **one}, {**rest, **base}
    if shared:
        cell = next(iter(shared))
        lever = next((h.lever for h in hits if h.cell == cell), {})
        return dict(shared), {cell: lever.get("value")}
    return None, {}


def _clip(hit: Hit, lo: int, hi: int) -> Hit | None:
    """The same lever as named inside one window of the line, choices and all."""
    anchors = [a for a in hit.anchors if lo <= a < hi]
    choices = [c for c in hit.choices if c[1] and all(lo <= a < hi for a in c[1])]
    if hit.choices and not hit.choices[0][1]:
        choices = hit.choices                      # a Yes/No toggle names no words
    if not anchors and not choices:
        return None
    return Hit(hit.lever, hit.cell, hit.score, anchors or hit.anchors, choices, hit.rank,
               hit.name, hit.weak, [a for a in hit.solid if lo <= a < hi])


def _case_label(sets: dict) -> str:
    return ", ".join(f"{v}" for v in sets.values()) or "as it stands"


def _sweep(text: str, toks: list[Tok], nums: list[Num], hits: list[Hit]) -> Ask:
    """One lever, several values: a list said in one breath, or the ends of a range filled in."""
    live = [n for n in nums if not n.is_year]
    if not hits:
        return Ask(why="sweep with no lever")
    lever, best = None, ()
    for h in hits:
        if h.fmt == "text" and len(h.lever.get("choices") or []) > 1 and change_cue(text):
            lever = h
            break
        fits = sum(1 for n in live if _compatible(n, h))
        if not fits or h.weak:
            continue
        here = (fits, -int(h.weak), min(h.score, 1.0))
        if not best or here > best:
            lever, best = h, here
    if lever is None:
        return Ask(why="sweep with no compatible lever")
    if lever.fmt == "text":
        values = [str(c) for c in (lever.lever.get("choices") or [])][:6]
        return Ask(kind="sweep", cell=lever.cell, values=values, confidence=0.6,
                   why=f"sweep {lever.name or lever.cell} over its {len(values)} values")
    vals = sorted({round(_as_value(n, lever), 10) for n in live if _compatible(n, lever)})
    if len(vals) < 2:
        return Ask(why="sweep with fewer than two values")
    if len(vals) == 2 and re.search(r"\b(from|between|range|anywhere)\b", text, re.I):
        lo, hi = vals
        vals = [round(lo + (hi - lo) * i / 3.0, 10) for i in range(4)] if hi > lo else vals
    return Ask(kind="sweep", cell=lever.cell, values=vals[:8],
               confidence=round(min(1.0, 0.35 + 0.5 * min(lever.score, 1.0)), 2),
               why=f"sweep {lever.name or lever.cell} at {len(vals)} values")


def _explain(text: str, toks: list[Tok], nums: list[Num], hits: list[Hit], outs: list[dict],
             prior_sets: dict, prior_lever: dict | None = None) -> Ask:
    """"why did costs move", "what drives year five EBITDA": the cell, and the change it moved
    under when there was one. The change comes from the line when the line carries one, and from
    the previous ask when it does not -- which is the live case, since "why did that move" only
    ever follows a what-if."""
    sets, taken, _ = _pair(nums, hits, text, limit=2)
    if not sets and prior_sets:
        named = {h.cell for h in hits}
        sets = {k: v for k, v in prior_sets.items() if k in named} or dict(prior_sets)
    table = []
    for h in hits[:4]:
        if h.weak:
            continue
        for i, o in enumerate(h.lever.get("outputs") or []):
            table.append({"cell": o["cell"], "label": o.get("label"), "fmt": o.get("fmt"),
                          "value": o.get("value"), "rank": i})
    target, score = match_output(text, table, toks, floor=0.3)
    alt, alt_score = match_output(text, outs, toks)
    if target is None or (alt is not None and alt_score > score + 0.25):
        target, score = alt, alt_score
    if target is None and prior_lever:
        # "why did that move" names no metric: it means the one the last ask moved
        first = (prior_lever.get("outputs") or [{}])[0]
        target = {"cell": first.get("cell"), "label": first.get("label")} if first.get("cell") \
            else None
        score = 0.5
    if target is None:
        return Ask(why="explain with no output cell")
    return Ask(kind="explain", target=target["cell"], sets=sets, watch=[target["cell"]],
               confidence=round(min(1.0, 0.4 + 0.5 * score), 2),
               why=f"explain {target.get('label')}"
                   + (f" under {', '.join(sets)}" if sets else " as it stands"))


def _confidence(taken: list[Hit], spare: list[Num]) -> float:
    """How much of what was said the parse accounted for. Act at 0.5."""
    if not taken:
        return 0.0
    worst = min(min(h.score, 1.0) for h in taken)
    conf = 0.35 + 0.55 * worst + 0.1 * (len(taken) > 1)
    conf -= 0.25 * len(spare) + 0.1 * sum(1 for h in taken if h.weak)
    return round(max(0.0, min(1.0, conf)), 2)


def _watch(ask: Ask, hits: list[Hit], named: dict | None = None,
           heads: list[str] | None = None) -> list[str]:
    """The outputs worth reporting: the metric the line named, then the ones the levers that
    moved are known to move, then the headline block. A client who names no metric means the
    headline, and a switch's own output list is not it."""
    out: list[str] = [named["cell"]] if named and named.get("cell") else []
    cells = set(ask.sets) | ({ask.cell} if ask.cell else set())
    cells |= {c for case in ask.cases for c in (case.get("sets") or {})}
    for h in hits:
        if h.cell not in cells:
            continue
        for o in (h.lever.get("outputs") or [])[:8]:
            if o.get("cell") and o["cell"] not in out:
                out.append(o["cell"])
    for cell in heads or []:
        if cell not in out:
            out.append(cell)
    return out[:20]


# --------------------------------------------------------------------------- #
# run it
# --------------------------------------------------------------------------- #
def run_ask(index: Any, ask: Ask, file: str | None = None) -> dict | None:
    """The Index call the Ask describes. Sync and CPU-bound: call it through a thread."""
    if not ask.kind:
        return None
    if ask.kind == "whatif":
        return index.whatif(file, dict(ask.sets), ask.watch or None)
    if ask.kind == "goalseek":
        return index.goalseek(file, ask.target, ask.goal, ask.lever)
    if ask.kind == "compare":
        return index.compare(file, [dict(c.get("sets") or {}) for c in ask.cases],
                             ask.watch or None, [c.get("label") for c in ask.cases])
    if ask.kind == "sweep":
        return index.sweep(file, ask.cell, list(ask.values), ask.watch or None)
    if ask.kind == "explain":
        return index.explain(file, ask.target, dict(ask.sets) or None)
    return None


# --------------------------------------------------------------------------- #
# render it as evidence
# --------------------------------------------------------------------------- #
def evidence_block(ask: Ask, out: dict) -> str:
    """The block the answer call reads instead of making a tool call.

    Every kind says the same three things in the same order: that the run already happened, what
    was changed, and what moved, each with a citation. The header names `answered_from` and the
    `working` string the card shows, so the model has nothing left to decide but the wording.
    """
    if not isinstance(out, dict) or not out:
        return ""
    if out.get("error") and ask.kind != "goalseek":
        return ""
    fn = {"whatif": _ev_whatif, "goalseek": _ev_goalseek, "compare": _ev_compare,
          "sweep": _ev_sweep, "explain": _ev_explain}.get(ask.kind or "")
    return fn(ask, out) if fn else ""


def _head(what: str, working: str) -> str:
    return (f"{what} (the workbook was recalculated here; answer from this, no tool call needed; "
            f'answered_from="derived"; working="{working}")')


def _ev_whatif(ask: Ask, out: dict) -> str:
    changes = list(out.get("changes") or [])
    if not changes:
        return ""
    working = ", ".join(
        f"{c.get('cell')} {c.get('was_s', c.get('was'))} → {c.get('now_s', c.get('now'))}"
        for c in changes[:3]) + ", model recalculated"
    lines = [_head("WHAT-IF ALREADY RUN with the asker's values", working)]
    for c in changes:
        lines.append(f"  changed: {c.get('label')} ({out.get('file')} › {c.get('cell')}) "
                     f"{c.get('was_s', c.get('was'))} → {c.get('now_s', c.get('now'))}")
    for o in (out.get("outputs") or [])[:8]:
        pct = o.get("pct")
        pct_s = f", {pct * 100:+.1f}%" if isinstance(pct, (int, float)) else ""
        lines.append(f"  {o.get('label')} ({o.get('cell')}): {o.get('was_s', o.get('was'))} → "
                     f"{o.get('now_s', o.get('now'))}{pct_s}")
    if out.get("partial"):
        lines.append(f"  partial recalc: {', '.join(map(str, out['partial'][:5]))}")
    return "\n".join(lines)


def _ev_goalseek(ask: Ask, out: dict) -> str:
    if out.get("error"):
        near = out.get("nearest")
        return _head("GOAL SEEK ALREADY RUN and the target is out of range",
                     f"{out.get('label')} cannot reach {out.get('target_label')}") + \
            f"\n  {out.get('error')}" + (f"\n  nearest reached: {near}" if near else "")
    working = (f"{out.get('label')} {out.get('was_s')} → {out.get('value_s')} puts "
               f"{out.get('target_label')} at {out.get('achieved_s')}")
    return "\n".join([
        _head("GOAL SEEK ALREADY RUN on the workbook (bisection, not a linear guess)", working),
        f"  solve for: {out.get('label')} ({out.get('file')} › {out.get('lever')})",
        f"  answer: {out.get('value_s')} (from {out.get('was_s')})",
        f"  which puts {out.get('target_label')} ({out.get('target')}) at "
        f"{out.get('achieved_s')}, against the {out.get('want_s')} asked for",
        f"  {out.get('tries')} recalculations, monotone={out.get('monotone')}"])


def _ev_compare(ask: Ask, out: dict) -> str:
    cases = out.get("cases") or []
    names = [c.get("label") for c in cases]
    lines = [_head("COMPARISON ALREADY RUN, every case on the same workbook",
                   " vs ".join(str(n) for n in names))]
    for c in cases:
        lines.append(f"  case: {c.get('label')} — " +
                     (", ".join(f"{k}={v}" for k, v in (c.get("sets") or {}).items())
                      or "as the model stands"))
    for row in (out.get("rows") or [])[:6]:
        vals = " | ".join(f"{n}: {s}" for n, s in zip(names, row.get("values_s") or []))
        spread = row.get("spread")
        gap = f"  (gap {spread:,.0f})" if isinstance(spread, (int, float)) else ""
        lines.append(f"  {row.get('label')} ({row.get('cell')}): {vals}{gap}")
    return "\n".join(lines)


def _ev_sweep(ask: Ask, out: dict) -> str:
    points = out.get("points") or []
    lines = [_head("RANGE ALREADY RUN on the workbook",
                   f"{out.get('label')} at {len(points)} values"),
             f"  swept: {out.get('label')} ({out.get('file')} › {out.get('cell')}) at "
             + ", ".join(str(p.get("value_s")) for p in points)]
    for row in (out.get("rows") or [])[:4]:
        cells = " | ".join(f"{p}: {s}" for p, s in
                           zip([p.get("value_s") for p in points], row.get("values_s") or []))
        lines.append(f"  {row.get('label')} ({row.get('cell')}): {cells}")
    return "\n".join(lines)


def _ev_explain(ask: Ask, out: dict) -> str:
    if out.get("reaches") is False:
        return "\n".join([
            _head("MECHANISM ALREADY TRACED and the input does not reach that cell",
                  str(out.get("note") or "")),
            f"  {out.get('note')}",
            "  say that it does not move, and what does move, rather than inventing a route"])
    working = " → ".join(str(h.get("cell")) for h in (out.get("chain") or [])[:4])
    lines = [_head("MECHANISM ALREADY TRACED through the workbook's own formulas", working)]
    lines.append(f"  {out.get('label')} ({out.get('file')} › {out.get('cell')}): "
                 f"{out.get('was')} → {out.get('now')}" if out.get("moved") else
                 f"  {out.get('label')} ({out.get('file')} › {out.get('cell')}) is built out of:")
    for hop in (out.get("chain") or [])[:6]:
        move = (f"{hop.get('was_s')} → {hop.get('now_s')}" if hop.get("delta")
                else str(hop.get("now_s")))
        lines.append(f"  {hop.get('label')} ({hop.get('cell')}): {move}   {hop.get('formula')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the one model call that pure code cannot replace
# --------------------------------------------------------------------------- #
JEV_FLOOR = 0.3              # act on the typed choice at this probability or better


async def augment_with_jev(ask: Ask, text: str, levers: list[dict], decider: Any,
                           prior: dict | None = None, outputs: list[dict] | None = None,
                           floor: float = JEV_FLOOR) -> Ask:
    """Fill the hole pure code has: a value whose lever is never named.

    "what if we are at 1.25 percent" names no input of the model, so no amount of word matching
    finds the fee. One typed choice over the whole lever table does (E7: 95.7% top-1 at 252 ms).
    It is only worth making when the parse came back short -- no lever, a value with nowhere to
    go, or no kind at all -- and it is asked about the part of the line that came back short,
    not about the whole line: "what if we charged one percent and churn were six percent" has
    already placed the churn, and asking about the whole line gets the churn back again.

    One request either way. When the fragment resolves, its value is added to the sets the pure
    parse already made; when nothing was placed at all, the parse is redone with the chosen
    lever at the head of the table, so position and format still decide what it takes.
    """
    from presage.llm import intent                                 # local: optional dependency
    toks = tokens(text)
    nums = spoken_numbers(text, toks)
    hits = match_levers(text, levers, toks)
    spare = [n for n in nums if not n.is_year and not n.aside]
    if ask.kind and not _short(ask, hits, spare):
        return ask
    unplaced = _unplaced(ask, nums, hits)
    if ask.kind == "whatif" and ask.sets and unplaced:
        frag = _fragment(text, toks, unplaced[0], ask, hits)
        pick = await intent.pick_lever(decider, frag, levers)
        if pick and pick.lever is not None and pick.p >= floor and pick.cell not in ask.sets:
            hit = Hit(pick.lever, pick.cell, float(pick.p), [unplaced[0].i0], [], 0,
                      str(pick.lever.get("label") or ""))
            if _compatible(unplaced[0], hit):
                ask.sets[pick.cell] = _as_value(unplaced[0], hit)
                ask.why += f"; {pick.cell} from a typed choice on \"{frag}\" (p={pick.p:.2f})"
                ask.confidence = round(min(1.0, ask.confidence + 0.3), 2)
                ask.ms += pick.latency_ms
                ask.watch = []
        return ask
    pick = await intent.pick_lever(decider, text, levers, prior=prior)
    if pick is None or pick.lever is None or pick.p < floor:
        return ask
    if any(h.cell == pick.cell and not h.weak for h in hits):
        return ask
    forced = [pick.lever] + [lv for lv in levers if cell_of(lv) != pick.cell]
    out = _parse_with(text, forced, pick, prior, outputs, levers)
    out.ms = ask.ms + pick.latency_ms
    out.why = (out.why + f"; lever from a typed choice ({pick.cell}, p={pick.p:.2f})").strip("; ")
    return out if out.kind else ask


def _unplaced(ask: Ask, nums: list[Num], hits: list[Hit]) -> list[Num]:
    """Values the parse said out loud and could not put anywhere."""
    placed = [v for v in ask.sets.values() if isinstance(v, (int, float))]
    out = []
    for num in nums:
        if num.is_year or num.aside:
            continue
        if any(_same(num.value, v) or _same(num.value / 100.0, v) or _same(num.value * 1000.0, v)
               for v in placed):
            continue
        out.append(num)
    return out


def _fragment(text: str, toks: list[Tok], num: Num, ask: Ask, hits: list[Hit]) -> str:
    """The part of the line around an unplaced value, cut at the levers already settled."""
    used = sorted(a for h in hits if h.cell in ask.sets for a in h.anchors)
    lo = max([a for a in used if a < num.i0] or [-1]) + 1
    hi = min([a for a in used if a > num.i1] or [len(toks)])
    frag = text[toks[lo].start:toks[min(hi, len(toks) - 1)].start].strip(" ,;.")
    return frag or text


def _short(ask: Ask, hits: list[Hit], spare: list[Num]) -> bool:
    """Did the parse leave something on the table that a lever choice would pick up?"""
    if not hits:
        return True
    placed = len(ask.sets) + len(ask.values) + (1 if ask.lever else 0)
    return placed < max(1, len(spare)) or ask.confidence < 0.45


def _parse_with(text: str, forced: list[dict], pick: Any, prior: dict | None,
                outputs: list[dict] | None, levers: list[dict]) -> Ask:
    """Re-parse with the chosen lever at the head of the table and a floor under its score."""
    toks = tokens(text)
    nums = spoken_numbers(text, toks)
    hits = match_levers(text, forced, toks)
    hits = [h for h in hits if h.cell != pick.cell or not h.weak]
    if not any(h.cell == pick.cell for h in hits):
        anchors = sorted({n.i0 for n in nums} | {n.i1 for n in nums}) or [0]
        hits = [Hit(pick.lever, pick.cell, float(pick.p), anchors, [], 0,
                    str(pick.lever.get("label") or pick.cell))] + hits
    outs = outputs if outputs is not None else output_table(levers)
    kind = _kind(text, hits, nums, toks)
    if kind == "explain":
        return _explain(text, toks, nums, hits, outs, dict((prior or {}).get("sets") or {}),
                        (prior or {}).get("lever"))
    if kind == "goalseek":
        return _goalseek(text, toks, nums, hits, outs)
    if kind == "compare":
        return _compare(text, toks, nums, hits)
    if kind == "sweep":
        return _sweep(text, toks, nums, hits)
    return _whatif(text, toks, nums, hits)


__all__ = ["Ask", "parse_ask", "run_ask", "evidence_block", "augment_with_jev", "output_table",
           "match_levers", "match_output", "spoken_numbers", "tokens", "cell_of", "change_cue",
           "KINDS"]
