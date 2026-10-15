"""The spoken-figure guard: a figure the consultant says aloud, against the cell it names.

E9b, the other half of E9. `presage/verify.py` (E23) checks a number **the app prints**; this
checks a number **the consultant says**. "So that's about nine million in fees" while the model
holds $11.98M is the failure mode: nothing on the page is wrong, the card was right, and the
figure that went into the client's ear was not.

The guard runs on a mic final, which is the one line brain.py deliberately drops today:

    presage/brain.py, _on_final
        # the consultant's own lines never get a card outside interview mode
        if u.source == "mic" and not prior_aid:
            return

Four steps, all local, no network, no model call:

1. **Extract** every figure the line says, in the forms an STT actually writes them.
   Parakeet writes "9 million" and "nine million" for the same audio, writes "nine point six"
   rather than 9.6, drops the currency entirely (a `$` never survives text-to-speech), and
   turns "two-fifty k" into three tokens. Dates, years, ordinals, slide numbers, page
   numbers, head counts and durations are dropped here, before anything is looked up: they
   are the commonest thing a person says aloud and none of them is a model figure.
2. **Refer**: which cell is the line talking about. The E38 vocabulary and `ask.local_slot`
   name the metric, period and segment off the index's own strings; `prerun.match_levers`
   names an input row; the deep Q&A cache is the last tier. All three are word matching over
   cached tables, so the whole step is a few milliseconds and never leaves the process.
3. **Compare**, with a band that follows how the figure was *said*. "Nine million" is one
   significant digit and means 8.5 to 9.5; "nine point six million" is two and means 9.55 to
   9.65; "about forty percent" widens again because the hedge is a licence to round. This is
   the same inversion `verify._window` does, with the scales cut down (see `_SCALES`).
4. **Verdict**: `silent`, `matches` or `contradicts`. Only the third produces a card, and it
   carries the cell, the value and one line of working so the consultant can correct
   themselves mid-sentence without reading a paragraph.

Four things buy silence, and each one costs a catch:

* the figure is a rounding of the cell (step 3's band, which is the whole point);
* the figure is a neighbour of the cell the line named: the same row in another period, or
  the same period in another row. A consultant reading down a column and naming the row
  loosely is under-specifying, not misstating, and `ask.local_slot` gets the period right
  67.5% of the time (E38), so a period mismatch is more often ours than theirs;
* the figure is a what-if the consultant just ran, which is in no cell by construction:
  `brain._last_whatif` and `brain.scratch` are passed in and their values are admitted;
* nothing resolved well enough to be a referent, which is most lines.

What it cannot do: catch a figure said about a file the folder does not hold, catch a wrong
operation between two right numbers, or tell a misstatement from a deliberate round number
the client already has in their head. Measured in docs/experiments/E9b-spoken-figures.md.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from presage.verify import _sig_of

log = logging.getLogger("presage.spoken")

# ----------------------------------------------------------------------------- tuning
# Every default here is the value the sweep in data/spoken/bench.py chose; the doc carries the
# table it chose from. They are module constants rather than settings because the guard has to
# behave the same in the bench and on the call.

REL = 0.02            # relative band under the printed precision, for a figure said plainly
HEDGE_MULT = 2.5      # how much wider a hedged figure gets: "about nine million" is a claim
                      # about the first digit, not about the number
FLOOR_METRIC = 0.55   # `local_slot` coverage under which the metric did not resolve
FLOOR_LEVER = 0.60    # `match_levers` score under which the lever did not resolve
FLOOR_ONE = 0.85      # a metric named by a SINGLE word has to be almost all of that label.
                      # `prerun.match_levers` calls the alternative `weak`: "about four
                      # hundred thousand in fees" covers most of "Subscriber fees" and is not
                      # about it. "Subscribers" and "Revenue" are one word and survive at 1.0.
FLOOR_QA = 0.62       # deep-Q&A cosine under which the cached question is not this line
MAX_CANDS = 12        # distinct values a referent may resolve to; more than this is a
                      # table, not a row, and the guard has nothing precise to argue with
NEIGHBOURS = False    # also admit the rest of the *column* the referent sits in. Off: the
                      # row is already the pool, and a column of a big-model sheet holds enough
                      # values that almost any figure lands on one. The sweep prices it.
MAX_FIGURES = 6       # figures checked per line; a line with more is a list, not a claim

# Scales a source value may be multiplied by to reach what was said. Deliberately narrower
# than `verify._SCALES`: a printed card cites its own cell, so a million-fold hop is evidence
# the card handed over, while a spoken line cites nothing and a hop that wide turns "nine
# million" into a match for a cell holding the integer 9. A workbook kept in $ thousands is
# the only convention left in.
_SCALES = {
    "pct": (1.0, 100.0, 0.01),
    "money": (1.0, 1e3, 1e-3),
    # a bare number gets the thousands hop too: no `$` survives a transcription, so "revenue
    # was fifty-five million" against a workbook kept in $ thousands is the ordinary case and
    # not a special one.
    "plain": (1.0, 1e3, 1e-3),
}
_SCALES_SOFT = (1.0, 1e3, 1e-3, 1e6, 1e-6, 100.0, 0.01)
MAX_GAP = 5.0         # a cell further than this multiple from what was said is a different
                      # thing, not a misstatement of this one, and the guard stays quiet


# ----------------------------------------------------------------------------- the words
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
# "grand" and "bucks" are how a figure is said out loud and never how a card prints one, so
# they are here and not in verify.
_SCALE_WORD = {"hundred": 1e2, "thousand": 1e3, "grand": 1e3, "k": 1e3, "million": 1e6,
               "mil": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
               "trillion": 1e12}
_MONEY_SCALE = {"grand", "k", "bucks", "dollars"}
_FRACTION = {"half": 0.5, "quarter": 0.25, "third": 1 / 3.0, "three-quarters": 0.75}

_PCT_WORD = {"percent", "percentage", "pct", "%", "points"}
_BPS_WORD = {"bps", "bips"}
# What makes a bare number money when no "$" survived the transcription. The STT never writes
# a currency sign, so the unit has to come out of the sentence.
_MONEY_CUE = re.compile(
    r"\b(dollar|dollars|bucks|usd|cash|fee|fees|cost|costs|revenue|revenues|spend|spending|"
    r"raise|raising|budget|price|priced|pricing|arr|mrr|ebitda|margin|profit|loss|income|"
    r"burn|runway|valuation|value|worth|salary|salaries|payroll|capex|opex|tco|"
    r"funding|invest|investment|payback|topline|top-line|bottom-line)\b", re.I)

_HEDGE = re.compile(
    r"\b(about|around|roughly|approximately|approx|ballpark|circa|call\s+it|somewhere\s+"
    r"(?:around|near|in)|near|nearly|almost|close\s+to|like|order\s+of|in\s+the\s+region\s+of|"
    r"give\s+or\s+take|or\s+so|ish|thereabouts|plus\s+or\s+minus|a\s+bit\s+(?:over|under)|"
    r"just\s+(?:over|under|about)|more\s+or\s+less|say)\b", re.I)
_OVER = re.compile(r"\b(north\s+of|over|more\s+than|at\s+least|upwards\s+of|in\s+excess\s+of|"
                   r"better\s+than|above|plus)\b$", re.I)
_UNDER = re.compile(r"\b(south\s+of|under|less\s+than|below|at\s+most|no\s+more\s+than|"
                    r"shy\s+of|short\s+of)\b$", re.I)

# Things said aloud that are never a model figure. Checked against the words either side of
# the span, which is cheaper and far more reliable than trying to type the number itself.
_LABEL_BEFORE = re.compile(
    r"\b(slide|page|pages|tab|sheet|row|rows|line|column|col|section|chapter|step|version|v|"
    r"figure|fig|exhibit|appendix|item|question|note|footnote|part|phase|option|scenario\s*#?|"
    r"year|yr|quarter|q|fy|cy|month|week|day|tier|round|stage|number|no\.|#)\s*$", re.I)
_TIME_AFTER = re.compile(
    r"^\s*(second|seconds|minute|minutes|hour|hours|day|days|week|weeks|month|months|"
    r"quarter|quarters|year|years|yr|yrs|fy|q[1-4]|am|pm|o'clock|oclock)\b", re.I)
_PEOPLE_AFTER = re.compile(
    r"^\s*(people|person|persons|folks|head|heads|headcount|hires|hire|fte|ftes|staff|"
    r"employee|employees|bodies|seats|engineers?|analysts?|consultants?|advisors?)\b", re.I)
_DATE_WORD = re.compile(
    r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
    r"sep|sept|september|oct|october|nov|november|dec|december|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)\b", re.I)
# A percent claim may only be held against a cell that could hold a percentage. Without this
# "their gross margin is around sixty five percent" reaches a cell holding 8,060 through the
# 0.01 scale hop and the guard corrects a figure that is not in the files at all.
_PCTISH = re.compile(r"%|\b(rate|percent|percentage|share|penetration|churn|growth|retention|"
                     r"yield|cagr|nrr|discount|multiple|uplift|margin\s*%)\b", re.I)
# A month makes a number a date only when it is right next to it ("March the third"), not
# anywhere in the sentence: "headcount in June was about 180" is a figure, not a date.
_DATE_NEXT = re.compile(
    r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
    r"sep|sept|september|oct|october|nov|november|dec|december|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)\s*(the\s+)?$", re.I)
_ORDINAL = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
    r"\d+(?:st|nd|rd|th))\b", re.I)

# A digit run glued to a letter or a dot is a label, not a figure: FY25, Q3, B48, V2.1, H1.
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-\.]*|\d[\d,]*(?:\.\d+)?|%|\S")
_GLUED = re.compile(r"(?<![A-Za-z0-9.])(?:fy|cy|q|h|v|b|c|d|e|f|g)\d", re.I)


_DIGIT_WORD = {**{w: str(v) for w, v in _ONES.items()},
               **{w: str(v) for w, v in _TENS.items()}}


_HEAD_KEY = re.compile(r"^(q[1-4]|h[12]|fy\d{2,4}|cy\d{2,4}|\d{1,2})$")


def _head_keys(header: str) -> set[str]:
    """The tokens of a column label that pin it to one period: `FY25 actual Q3` -> fy25, q3.

    A consultant says "in Q3", not "in FY25 actual Q3", so requiring the whole header to
    appear in the line never fires. One pinning token is enough, and a token that pins
    nothing ("actual", "plan") is not one.
    """
    return {t for t in _canon(header).split() if _HEAD_KEY.match(t)}


def _canon(text: str) -> str:
    """The line with its number words as digits, for matching a column label literally.

    `topics.normalize` drops bare digits, so "year two" and "Year 0" reduce to the same single
    token and the period slot cannot tell them apart. A literal match on the canonicalised
    line can, and it is the difference between quoting back the cell the consultant pointed
    at and quoting back a different year of the same row.
    """
    return " ".join(_DIGIT_WORD.get(w, w) for w in re.findall(r"[a-z0-9]+", text.lower()))


def _normalise(text: str) -> str:
    """What the STT hands over, in the one shape the parser reads.

    Only three things change: a hyphen inside a spoken number is a space ("two-fifty k",
    "twenty-eight"), a `$` becomes a word so the currency survives a typed line, and a bare
    `%` is detached from its digits.
    """
    t = re.sub(r"[‘’]", "'", str(text or ""))
    t = t.replace("$", " dollars ")
    t = re.sub(r"(\d)\s*%", r"\1 percent", t)
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t, flags=re.I)
    # Parakeet glues a period label to the word after it ("Q3EBITDA", "FY26plan"). Splitting
    # on a digit followed by two or more letters is safe as long as the unit suffixes a
    # figure really uses are spared.
    t = re.sub(r"(?<=\d)(?=[A-Za-z]{2,})(?!bn\b|mm\b|mn\b|bp|pct\b|mil\b|st\b|nd\b|rd\b|th\b)",
               " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass(frozen=True)
class Said:
    """One figure as a line said it: what it denotes, and how loosely it was said."""
    value: float
    kind: str           # "pct" | "money" | "plain"
    ulp: float          # half a unit in the last *significant* place, scaled like the value
    text: str           # the words it came from, as spoken
    sig: int = 1
    hedged: bool = False
    direction: int = 0  # +1 "north of", -1 "under", 0 a plain claim
    start: int = 0
    end: int = 0

    def band(self, rel: float = REL, mult: float = HEDGE_MULT) -> tuple[float, float]:
        """The values this figure could honestly have been said about.

        The hedge widens the *relative* half of the tolerance and not the ULP, because a round
        number and a hedge carry the same information twice. "Nine million" already claims
        only the first digit; "about nine million" claims no less. Multiplying the two is what
        makes a guard that never speaks: "around 40 percent" would cover 27.5% to 52.5%.
        """
        rel = rel * (mult if self.hedged else 1.0)
        tol = max(self.ulp, abs(self.value) * rel, 1e-9)
        lo, hi = self.value - tol, self.value + tol
        if self.direction > 0:            # "north of nine million": anything above is fine
            hi = float("inf")
        elif self.direction < 0:
            lo = float("-inf")
        return lo, hi


# ----------------------------------------------------------------------------- extraction
ULP_CAP = 0.10        # a ULP may never claim more than this share of the figure itself


def _ulp_of(digits: str, places: int, spelled: bool = False) -> float:
    """Half a unit in the last place the figure was actually said to.

    "Nine point six million" claims 9.55 to 9.65; "eleven million" claims 10.5 to 11.5, which
    is what makes it a rounding of $14.85M and not a contradiction of it.

    Two or more trailing zeros is a round number and widens the claim: "1,700 sellers" means
    1,650 to 1,750. **One** trailing zero does not, because "10" and "ten" are the same word
    and "ten million" is not a claim about the range 5 to 15 million. A spelled-out mantissa
    never gets the trailing-zero rule at all: its round hundreds are already folded into the
    scale by `_Reader._words`.
    """
    if places:
        return 0.5 * (10.0 ** -places)
    bare = digits.replace(",", "").split(".")[0]
    trail = len(bare) - len(bare.rstrip("0"))
    return 0.5 * (10.0 ** trail) if trail >= 2 and not spelled else 0.5


def _sig_digits(raw: str) -> int:
    d = raw.replace(",", "").replace(".", "").lstrip("0")
    return len(d.rstrip("0") if "." not in raw else d) or 1


class _Reader:
    """A single pass over the words of a line, collecting numbers as they are spoken.

    Written as a walker rather than a regular expression because the spoken forms nest:
    "a million and a half", "nine and a half million", "two fifty k", "a hundred and fifty
    grand". Each of those is one figure and a regex that catches all four also catches
    "twenty twenty six" as 2,026.
    """

    def __init__(self, text: str):
        self.raw = text
        self.toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]
        self.low = [t[0].lower().strip(".,;:!?") for t in self.toks]

    # -- helpers over the token stream -------------------------------------------------
    def word(self, i: int) -> str:
        return self.low[i] if 0 <= i < len(self.low) else ""

    def _digits(self, i: int) -> tuple[float, str, int, float] | None:
        w = self.toks[i][0]
        if not w[:1].isdigit():
            return None
        if _GLUED.search(self.raw[max(0, self.toks[i][1] - 2):self.toks[i][2]]):
            return None                                   # FY25, Q3, V2.1: a label
        try:
            return float(w.replace(",", "")), w, i + 1, 1.0
        except ValueError:
            return None

    def _words(self, i: int) -> tuple[float, str, int, float] | None:
        """A spelled-out number starting at token i, or None. Returns (value, text, next).

        The scale word is left for the caller, so this only ever reads a mantissa: "nine",
        "forty", "two fifty", "a hundred and fifty", "nine point six".
        """
        cur, seen, frac = 0.0, False, None
        hundreds = 1.0                    # folded into the scale when the number ends on it
        pieces: list[str] = []
        while i < len(self.low):
            w = self.low[i]
            if w in ("and", "a", "an") and seen:
                # "and a half" closes the number; "a hundred and fifty" continues it
                if self.word(i + 1) in _FRACTION or self.word(i + 2) in _FRACTION:
                    break
                if self.word(i + 1) in _ONES or self.word(i + 1) in _TENS:
                    pieces.append(w)
                    i += 1
                    continue
                break
            if w in ("a", "an") and not seen and self.word(i + 1) in _SCALE_WORD:
                cur, seen = 1.0, True                      # "a million", "a hundred"
                pieces.append(w)
                i += 1
                continue
            if w == "point" and seen:
                digits = ""
                j = i + 1
                while j < len(self.low) and self.low[j] in _ONES and _ONES[self.low[j]] < 10:
                    digits += str(_ONES[self.low[j]])
                    j += 1
                if not digits:
                    break
                frac = digits
                pieces += ["point"] + self.low[i + 1:j]
                i = j
                break
            if w in _ONES:
                if cur and _ONES[w] >= 10:
                    break                                  # "twelve twelve" is not a number
                cur += _ONES[w]
                hundreds = 1.0
                seen = True
            elif w in _TENS:
                # "two fifty" is 250, not 52: a unit said straight into a ten is a spoken
                # three-digit number, which is how "two-fifty k" reaches the transcript.
                # "a hundred and fifty" adds. Two tens in a row is a year or a phone number
                # ("twenty twenty six") and ends the figure.
                if cur and cur < 10 and cur == int(cur):
                    cur = cur * 100 + _TENS[w]
                elif cur >= 100 and cur % 100 == 0:
                    cur += _TENS[w]
                    hundreds = 1.0
                elif cur:
                    break
                else:
                    cur += _TENS[w]
                seen = True
            elif w == "hundred":
                cur = (cur or 1.0) * 100.0
                hundreds = 100.0
                seen = True
            else:
                break                     # a scale word: the caller applies it
            pieces.append(w)
            i += 1
        if not seen:
            return None
        val = float(f"{int(cur)}.{frac}") if frac is not None else cur
        if hundreds > 1.0 and frac is None:
            val /= hundreds            # "two hundred thousand" is 2 at a scale of 1e5
        return val, " ".join(pieces), i, hundreds

    def _scale(self, i: int) -> tuple[float, str, int, bool] | None:
        """A scale word at token i: (factor, text, next, is_money)."""
        w = self.low[i] if i < len(self.low) else ""
        if w in _SCALE_WORD and w != "hundred":
            return _SCALE_WORD[w], w, i + 1, w in _MONEY_SCALE
        return None

    def _half(self, i: int) -> tuple[float, int] | None:
        """"and a half" / "and a quarter" after a number, as a fraction of what follows."""
        if self.word(i) == "and" and self.word(i + 1) in ("a", "an") and \
                self.word(i + 2) in _FRACTION:
            return _FRACTION[self.word(i + 2)], i + 3
        if self.word(i) in ("and", "point") and self.word(i + 1) in _FRACTION:
            return _FRACTION[self.word(i + 1)], i + 2
        return None

    # -- the pass ----------------------------------------------------------------------
    def read(self) -> list[Said]:
        out: list[Said] = []
        i = 0
        while i < len(self.toks):
            got = self._read_one(i)
            if got is None:
                i += 1
                continue
            said, i = got
            if said is not None:
                out.append(said)
        return out

    def _read_one(self, i: int) -> tuple[Said | None, int] | None:
        start_tok = i
        # "half a million", "a quarter of a million": the fraction leads and the scale follows
        lead_frac = None
        if self.low[i] in _FRACTION and self.word(i + 1) in ("a", "an", "of"):
            j = i + 1 + (1 if self.word(i + 1) == "of" else 0)
            if self.word(j) in ("a", "an") and self.word(j + 1) in _SCALE_WORD:
                lead_frac, i = _FRACTION[self.low[i]], j + 1
        if lead_frac is None:
            head = self._digits(i) or self._words(i)
            if head is None:
                return None
            val, text, i, hundreds = head
            spelled = not text[:1].isdigit()
            # the mantissa as it was said, which is the whole of the precision claim
            mant = text if not spelled else f"{val:g}"
        else:
            val, mant, hundreds, spelled = 1.0, "0.5", 1.0, True

        # a fraction rides on the mantissa: "nine and a half million"
        frac = self._half(i)
        if frac is not None and float(val).is_integer():
            val, i = val + frac[0], frac[1]
            mant = f"{val:g}"

        kind = "plain"
        scale = self._scale(i)
        factor = hundreds
        if scale is not None:
            f2, _unit, i, money = scale
            factor *= f2
            if lead_frac is not None:
                val = lead_frac
            kind = "money" if money else "plain"
            # "a million and a half": the fraction lands after the scale word
            frac = self._half(i)
            if frac is not None:
                val, i = val + frac[0], frac[1]
                mant = f"{val:g}"
        elif lead_frac is not None:
            return None                                    # "half a" with nothing to halve

        places = len(mant.split(".")[1]) if "." in mant else 0
        sig = _sig_digits(mant)
        val = val * factor
        # the floor is the last place the figure was said to; the cap stops a round mantissa
        # claiming a band wider than a tenth of itself ("ten million" is 9.5 to 10.5, not 5
        # to 15), and the relative term in `Said.band` takes it from there.
        ulp = _ulp_of(mant, places, spelled) * factor
        ulp = max(min(ulp, ULP_CAP * abs(val) or ulp), 0.5 * (10.0 ** -places) * factor)

        # the unit, from the words on either side. No currency sign survives a transcription,
        # so money is decided by the sentence and not by the number.
        nxt = self.word(i)
        if nxt in _PCT_WORD:
            kind, i = "pct", i + 1
        elif nxt in _BPS_WORD or (nxt == "basis" and self.word(i + 1) in ("points", "point")):
            step = 1 if nxt in _BPS_WORD else 2
            kind, val, ulp, i = "pct", val / 100.0, ulp / 100.0, i + step
        elif kind != "money" and (scale is not None or places or abs(val) >= 1000):
            # "income" in the sentence makes "6.6 million" money; it must not make the "three"
            # of "year three" money, or every period label becomes a figure.
            tail = self.raw[self.toks[i - 1][2]:self.toks[i - 1][2] + 44] if i else ""
            before = self.raw[max(0, self.toks[start_tok][1] - 30):self.toks[start_tok][1]]
            if _MONEY_CUE.search(before) or _MONEY_CUE.search(tail):
                kind = "money"

        end_tok = i
        span_a, span_b = self.toks[start_tok][1], self.toks[max(0, end_tok - 1)][2]
        spoken = self.raw[span_a:span_b]
        if self._drop(start_tok, end_tok, val, kind, sig):
            return None, i
        before = self.raw[max(0, span_a - 32):span_a]
        hedged = bool(_HEDGE.search(before))
        direction = 1 if _OVER.search(before.rstrip()) else -1 if _UNDER.search(before.rstrip()) else 0
        after = self.raw[span_b:span_b + 14]
        if re.match(r"^\s*(or\s+so|ish\b|,?\s*give\s+or\s+take)", after, re.I):
            hedged = True
        return Said(float(val), kind, float(ulp), spoken.strip(), sig, hedged, direction,
                    span_a, span_b), i

    def _drop(self, a: int, b: int, val: float, kind: str, sig: int) -> bool:
        """True when this run of words is not a figure about the model.

        Everything here is a thing people say constantly and no guard should ever open its
        mouth about: a slide number, a date, a duration, a head count, a version, an ordinal,
        a bare year, and any single digit standing on its own.
        """
        before = self.raw[max(0, self.toks[a][1] - 24):self.toks[a][1]]
        after = self.raw[self.toks[b - 1][2]:self.toks[b - 1][2] + 24] if b else ""
        if _LABEL_BEFORE.search(before):
            return True
        if _TIME_AFTER.match(after) or _PEOPLE_AFTER.match(after):
            return True
        if _ORDINAL.match(self.toks[a][0]):
            return True
        if kind == "plain":
            if abs(val) < 10 and float(val).is_integer():
                return True                                # "three regions", "two teams"
            if float(val).is_integer() and 1900 <= val <= 2100 and sig >= 4:
                return True                                # a year
            if _DATE_NEXT.search(before) or _DATE_WORD.match(after.strip()):
                return True
        return False


def figures(line: str) -> list[Said]:
    """Every figure a spoken line claims, in order. The extractor on its own, for the bench."""
    return _Reader(_normalise(line)).read()[:MAX_FIGURES]


# ----------------------------------------------------------------------------- matching
def _agrees(said: Said, value: float, rel: float = REL, mult: float = HEDGE_MULT,
            soft: bool = False) -> bool:
    """True when `value`, at a scale the workbook actually uses, is what was said.

    A scale hop has to preserve significant digits, exactly as `verify._in` requires: without
    it "two point nine nine seven million" matches an integer 3 sitting in a cell that counts
    retail partners.
    """
    lo, hi = said.band(rel, mult)
    for v in (value, -value):
        for f in (_SCALES_SOFT if soft else _SCALES.get(said.kind, (1.0,))):
            scaled = v * f
            if lo <= scaled <= hi and (f == 1.0 or said.sig <= _sig_of(value)):
                return True
    return False


# ----------------------------------------------------------------------------- the referent
@dataclass
class Cand:
    """One cell the line might be talking about."""
    value: float
    cell: str                 # "file › sheet › ref"
    label: str = ""
    header: str = ""
    score: float = 0.0
    how: str = ""             # metric | lever | qa | neighbour | whatif | scratch

    def where(self) -> str:
        return self.cell


@dataclass
class Spoken:
    """What the guard decided about one Me line.

    `verdict` is the only field the wiring branches on. `silent` and `matches` both mean
    nothing goes on the page; they are kept apart because the bench has to know the
    difference between "the figure was right" and "there was nothing to check it against".
    """
    verdict: str = "silent"          # "silent" | "matches" | "contradicts"
    reason: str = ""
    said: str = ""
    said_value: float | None = None
    actual: float | None = None
    actual_s: str = ""
    cell: str = ""
    label: str = ""
    working: str = ""
    confidence: float = 0.0
    ms: float = 0.0
    figures: int = 0
    checked: int = 0
    how: str = ""

    def card(self, aid: str = "") -> dict:
        """The frame the page draws. Deliberately the same field names as an answer card so
        the existing renderer needs one new `type`, not a new component."""
        return {"type": "spoken", "id": aid, "verdict": self.verdict,
                "said": self.said, "figure": self.actual_s, "label": self.label,
                "citations": [{"file": self.cell.split(" › ")[0],
                               "loc": " › ".join(self.cell.split(" › ")[1:])}] if self.cell else [],
                "working": self.working, "reason": self.reason,
                "confidence": round(self.confidence, 3), "ms": round(self.ms, 2)}

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason, "said": self.said,
                "said_value": self.said_value, "actual": self.actual, "actual_s": self.actual_s,
                "cell": self.cell, "label": self.label, "working": self.working,
                "confidence": round(self.confidence, 3), "ms": round(self.ms, 3),
                "figures": self.figures, "checked": self.checked, "how": self.how}


def _fmt(value: float, kind: str) -> str:
    """A figure the way a card prints one, so the correction reads like the app."""
    if kind == "pct":
        v = value * 100.0 if abs(value) <= 1.5 else value
        return f"{v:,.1f}%".replace(".0%", "%")
    a = abs(value)
    sign = "-" if value < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.2f}bn"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.2f}M"
    if a >= 1e4:
        return f"{sign}${a:,.0f}"
    if a == int(a):
        return f"{sign}{int(a):,}"
    return f"{sign}{a:,.2f}"


class Guard:
    """Everything the check needs that does not change between lines, built once per index.

    `ask.local_slot` re-tokenises its whole option list on every call, which is 3.2 ms against
    the big test model's 812 metrics. The token sets and the rarity weights are the same on every
    line of a call, so they are built here and the per-line work is set intersections.
    """

    def __init__(self, index: Any, vocab: Any = None):
        from presage.llm import ask as askmod

        self.index = index
        self.version = getattr(index, "version", 0)
        self.vocab = vocab if vocab is not None else askmod.build_vocab(index)
        self._rar: dict[str, tuple[dict, list]] = {}
        self._nums: dict[str, list[set[str]]] = {}
        for name in ("metrics", "periods", "segments"):
            opts = getattr(self.vocab, name, []) or []
            self._rar[name] = askmod._rarity(opts) if opts else ({}, [])
            # the bare digits in an option's own name: "Subscribers, Year 5" carries a 5 and
            # `topics.normalize` throws it away, which is what makes that row and "Subscribers"
            # look like the same answer to "subscribers in year two"
            self._nums[name] = [{w for w in re.findall(r"[a-z0-9]+", o.lower()) if w.isdigit()}
                                for o in opts]
        try:
            self.levers = list(index.levers() or [])
        except Exception:                                  # noqa: BLE001
            self.levers = []
        self._norm = askmod._norm

    # -- the cached half of ask.local_slot ---------------------------------------------
    def slot(self, line: str, name: str, floor: float, k: int = 3) -> list[tuple[str, float]]:
        """The options the line names, best first, ties kept.

        `ask.local_slot` returns one option and ranks by coverage alone, which is why E38
        measured it at 60% reach: "Cost" and "Startup Costs" are both fully covered by "the
        startup cost", and the shorter one wins on alphabetical order. Two changes here.
        The **weight** the option matched breaks a tie in coverage, so the more specific label
        wins; and near-ties are all returned, because a guard that cannot tell "Subscribers" from
        "How we get to Year 5 subscribers" should check the figure against both rows and stay
        quiet if either one holds it.
        """
        opts = getattr(self.vocab, name, []) or []
        weight, toks = self._rar[name]
        want = set(self._norm(line))
        if not opts or not want:
            return []
        digits = {w for w in _canon(line).split() if w.isdigit()}
        scored: list[tuple[float, float, str]] = []
        for opt, t, own in zip(opts, toks, self._nums[name]):
            hit = t & want
            if not hit or (own and not own <= digits):
                continue                    # "Subscribers, Year 5" is not "subscribers in year two"
            got = sum(weight[w] for w in hit)
            total = sum(weight[w] for w in t) or 1.0
            frac = got / total
            if len(hit) == 1 and frac < FLOOR_ONE:
                continue                    # named by one word of a longer label
            if frac >= floor:
                scored.append((round(frac, 6), round(got, 6), opt))
        if not scored:
            return []
        scored.sort(key=lambda r: (-r[0], -r[1], r[2].lower()))
        top_frac, top_got = scored[0][0], scored[0][1]
        return [(o, f) for f, g, o in scored
                if f >= top_frac - 1e-9 and g >= top_got * 0.9][:k]

    # -- resolving -----------------------------------------------------------------------
    def _cells_for_label(self, label: str, limit: int = 60) -> list[Cand]:
        con = getattr(self.index, "con", None)
        if con is None or not label:
            return []
        rows = con.execute(
            "SELECT file, sheet, cell_ref, row_label, header, value_num FROM cells "
            "WHERE lower(row_label)=? AND value_num IS NOT NULL LIMIT ?",
            (label.lower(), limit)).fetchall()
        return [Cand(float(r[5]), f"{r[0]} › {r[1]} › {r[2]}", str(r[3] or ""),
                     str(r[4] or ""), 0.0, "metric") for r in rows]

    def _cells_for_column(self, label: str, limit: int = 60) -> list[Cand]:
        """A metric that is a *column* rather than a row: `arr_usd`, `csat`, `Customers`."""
        con = getattr(self.index, "con", None)
        if con is None or not label:
            return []
        rows = con.execute(
            "SELECT file, sheet, cell_ref, row_label, header, value_num FROM cells "
            "WHERE lower(header)=? AND value_num IS NOT NULL LIMIT ?",
            (label.lower(), limit)).fetchall()
        return [Cand(float(r[5]), f"{r[0]} › {r[1]} › {r[2]}", str(r[3] or ""),
                     str(r[4] or ""), 0.0, "metric") for r in rows]

    def _from_levers(self, line: str) -> list[Cand]:
        if not self.levers:
            return []
        try:
            from presage import prerun
            hits = prerun.match_levers(line, self.levers)
        except Exception as e:                             # noqa: BLE001
            log.debug("spoken: lever match failed: %s", e)
            return []
        out: list[Cand] = []
        for h in hits[:3]:
            if h.score < FLOOR_LEVER or h.weak:
                continue
            lv = h.lever
            v = lv.get("value")
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            out.append(Cand(float(v), f'{lv.get("file")} › {lv.get("sheet")} › {lv.get("cell")}',
                            str(lv.get("label") or ""), str(lv.get("header") or ""),
                            float(h.score), "lever"))
        return out

    def _from_qa(self, line: str) -> list[Cand]:
        """The deep Q&A cache: an anticipated question whose figure is already resolved."""
        try:
            hits = self.index.qa_matches(line, k=2)
        except Exception as e:                             # noqa: BLE001
            log.debug("spoken: qa_matches failed: %s", e)
            return []
        out: list[Cand] = []
        for q in hits:
            if float(getattr(q, "score", 0.0)) < FLOOR_QA:
                continue
            for c in (getattr(q, "citations", None) or [])[:2]:
                loc = str(c.get("loc") or "") if isinstance(c, dict) else str(c)
                f = str(c.get("file") or "") if isinstance(c, dict) else ""
                for cand in self._cell_value(f, loc):
                    cand.score = float(getattr(q, "score", 0.0))
                    cand.how = "qa"
                    out.append(cand)
        return out

    def _cell_value(self, file_: str, loc: str) -> list[Cand]:
        con = getattr(self.index, "con", None)
        parts = [p.strip() for p in re.split(r"\s*(?:›|>|»)\s*", loc) if p.strip()]
        if con is None or len(parts) < 2:
            return []
        sheet, ref = parts[-2], parts[-1].upper()
        if not re.match(r"^[A-Z]{1,3}\d{1,7}$", ref):
            return []
        rows = con.execute(
            "SELECT file, sheet, cell_ref, row_label, header, value_num FROM cells "
            "WHERE file=? AND sheet=? AND cell_ref=? AND value_num IS NOT NULL",
            (file_, sheet, ref)).fetchall()
        return [Cand(float(r[5]), f"{r[0]} › {r[1]} › {r[2]}", str(r[3] or ""),
                     str(r[4] or ""), 0.0, "qa") for r in rows]

    def _neighbours(self, cands: list[Cand], limit: int = 120) -> list[Cand]:
        """The rest of the row and the rest of the column the referent sits in.

        A consultant reading down a column names the row and lets the period ride on the
        pointer. `ask.local_slot` gets the period right 67.5% of the time (E38), so a figure
        that is the right metric in the wrong column is far more often our slot than their
        mistake. This is the guard's largest single source of silence, and the sweep in the
        E9b doc shows what turning it off costs.
        """
        con = getattr(self.index, "con", None)
        if con is None or not cands:
            return []
        out: list[Cand] = []
        for c in cands[:3]:
            parts = c.cell.split(" › ")
            if len(parts) < 3:
                continue
            m = re.match(r"^([A-Z]{1,3})(\d+)$", parts[2])
            if not m:
                continue
            for sql, args in (
                ("SELECT file, sheet, cell_ref, row_label, header, value_num FROM cells "
                 "WHERE file=? AND sheet=? AND row_label=? AND value_num IS NOT NULL LIMIT ?",
                 (parts[0], parts[1], c.label, limit)),
                ("SELECT file, sheet, cell_ref, row_label, header, value_num FROM cells "
                 "WHERE file=? AND sheet=? AND header=? AND value_num IS NOT NULL LIMIT ?",
                 (parts[0], parts[1], c.header, limit)),
            ):
                try:
                    out += [Cand(float(r[5]), f"{r[0]} › {r[1]} › {r[2]}", str(r[3] or ""),
                                 str(r[4] or ""), 0.0, "neighbour")
                            for r in con.execute(sql, args).fetchall()]
                except Exception:                          # noqa: BLE001
                    continue
        return out

    # -- the check -----------------------------------------------------------------------
    def check(self, line: str, whatif: Any = None, scratch: Any = None,
              prior: dict | None = None, rel: float = REL, mult: float = HEDGE_MULT,
              neighbours: bool = NEIGHBOURS) -> Spoken:
        t0 = time.perf_counter()
        try:
            out = self._check(line, whatif, scratch, prior, rel, mult, neighbours)
        except Exception as e:                             # noqa: BLE001
            log.debug("spoken check failed: %s", e, exc_info=True)
            out = Spoken("silent", f"check failed ({type(e).__name__})")
        out.ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _check(self, line: str, whatif: Any, scratch: Any, prior: dict | None,
               rel: float, mult: float, neighbours: bool) -> Spoken:
        # Everything downstream works on the normalised line, because a figure's span is an
        # offset into it: `_normalise` turns "$" into a word and splits "Q3EBITDA", so an
        # offset taken there does not point at the same characters in the raw line.
        norm = _normalise(line)
        said = _Reader(norm).read()[:MAX_FIGURES]
        if not said:
            return Spoken("silent", "no figure said")

        # the metric is named by the words that are *not* the figure: "nine million" would
        # otherwise reach a row called "Year 9" on its own digits
        stripped = norm
        for s in sorted(said, key=lambda x: -x.start):
            stripped = stripped[:s.start] + " " + stripped[s.end:]
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if prior and not self._norm(stripped):
            stripped = str(prior.get("line") or "")      # "and that is about nine million"

        metrics = self.slot(stripped, "metrics", FLOOR_METRIC)
        if not metrics and prior and (prior.get("slots") or {}).get("metric"):
            metrics = [(str(prior["slots"]["metric"]), FLOOR_METRIC)]
        periods = self.slot(stripped, "periods", FLOOR_METRIC)
        segments = self.slot(stripped, "segments", FLOOR_METRIC)

        # The pool is the whole row, not the one cell the slots point at. `ask.local_slot`
        # gets the period right 67.5% of the time (E38) and "year two" normalises to the same
        # token as "Year 0", so narrowing to a column before deciding turns our slot error
        # into their misstatement. The period is used to choose which cell to quote back, and
        # never to license a card.
        pool: list[Cand] = []
        how = ""
        for name, p in metrics:
            got = self._cells_for_label(name) or self._cells_for_column(name)
            for c in got:
                c.score = p
            pool += got
            how = "metric"
        if not pool:
            pool = self._from_levers(stripped)
            how = "lever" if pool else how
        if not pool:
            pool = self._from_qa(norm)
            how = "qa" if pool else how
        if not pool:
            return Spoken("silent", "nothing in the files answers to what the line named",
                          figures=len(said))
        seen: set[str] = set()
        pool = [c for c in pool if not (c.cell in seen or seen.add(c.cell))][:200]

        # A column the line names *literally* narrows the pool, unlike the period slot, which
        # only chooses what to quote. "In year two" and the `Year 2` header are the same
        # string once the number words are digits again; that is not a guess about what the
        # consultant meant, it is what they said.
        if any(f.kind == "pct" for f in said):
            keep = [c for c in pool if abs(c.value) <= 1.5
                    or _PCTISH.search(f"{c.label} {c.header}")]
            if not keep and all(f.kind == "pct" for f in said):
                return Spoken("silent", "nothing the line named could hold a percentage",
                              figures=len(said), how=how)
            pool = keep or pool
        canon = _canon(stripped)
        ctoks = set(canon.split())
        named = [c for c in pool if c.header and len(c.header) > 2
                 and (_canon(c.header) in canon or (_head_keys(c.header) & ctoks))]
        pool = named or pool
        values = {round(c.value, 6) for c in pool}
        if len(values) > MAX_CANDS:
            return Spoken("silent", f"the line named {len(values)} different figures, not one",
                          figures=len(said), how=how)

        extra: list[Cand] = []
        if neighbours:
            extra += self._neighbours(pool)
        extra += _whatif_cands(whatif) + _scratch_cands(scratch)

        # which of the row's cells to quote, when the row disagrees with what was said
        focus = pool
        if not named:
            for name, _p in periods[:1]:
                hit = [c for c in focus if c.header.lower() == name.lower()]
                focus = hit or focus
        for name, _p in segments[:1]:
            hit = [c for c in focus
                   if name.lower() in c.cell.lower() or name.lower() == c.label.lower()]
            focus = hit or focus

        checked = 0
        for s in said:
            checked += 1
            hit = next((c for c in pool if _agrees(s, c.value, rel, mult)), None)
            if hit is not None:
                where = "the cell it names" if hit in focus else "another period of that row"
                return Spoken("matches", f"{s.text.strip()} is {hit.label or hit.cell} "
                                         f"({where})",
                              s.text.strip(), s.value, hit.value, _fmt(hit.value, s.kind),
                              hit.cell, hit.label, "", hit.score or 1.0,
                              figures=len(said), checked=checked, how=how)
            soft = next((c for c in extra if _agrees(s, c.value, rel, mult, soft=True)), None)
            if soft is not None:
                why = {"neighbour": "another cell in the same column",
                       "whatif": "the what-if just run",
                       "scratch": "a figure on the scratch pad"}.get(soft.how, soft.how)
                return Spoken("silent", f"{s.text.strip()} is {why}", s.text.strip(),
                              s.value, soft.value, _fmt(soft.value, s.kind), soft.cell,
                              soft.label, "", 0.0, figures=len(said), checked=checked,
                              how=soft.how)
            # nothing in the row holds it: quote the cell the line pointed at
            best = min(focus, key=lambda c: _gap(s, c.value))
            if _gap(s, best.value) > MAX_GAP:
                continue                       # a different order of thing, not a misstatement
            actual = _fmt(best.value, s.kind)
            return Spoken("contradicts",
                          f"said {s.text.strip()}; {best.label or 'the cell'} is {actual}",
                          s.text.strip(), s.value, best.value, actual, best.cell, best.label,
                          _working(s, best), best.score, figures=len(said),
                          checked=checked, how=how)
        return Spoken("silent", "no figure is close enough to a cell to argue with",
                      figures=len(said), checked=checked, how=how)


def _gap(said: Said, value: float) -> float:
    """How far a cell is from what was said, as a multiple, at the kindest scale.

    A ratio and not a difference, because the question it answers is "are these two the same
    quantity said differently". A cell holding 37 subscribers is 243,000 times a claim of nine
    million and is not a misstatement of it; a cell holding $11.98M is 1.3 times it and is.
    Used to pick which cell to quote back, and to refuse to quote one that is not in the
    running (`MAX_GAP`).
    """
    a = abs(said.value)
    if not a:
        return float("inf")
    best = float("inf")
    for f in _SCALES.get(said.kind, (1.0,)):
        v = abs(value * f)
        if not v:
            continue
        best = min(best, max(v / a, a / v))
    return best


def _working(said: Said, cand: Cand) -> str:
    where = cand.cell.split(" › ")
    tail = f"{where[1]} {where[2]}" if len(where) >= 3 else cand.cell
    col = f", {cand.header}" if cand.header and not _DEADHEAD.match(cand.header) else ""
    return (f"said {said.text.strip()}; {cand.label or 'the cell'}{col} is "
            f"{_fmt(cand.value, said.kind)} in {tail}")


_DEADHEAD = re.compile(r"^col\s+[A-Z]{1,3}$", re.I)


def _whatif_cands(whatif: Any) -> list[Cand]:
    """A recalculated figure is in no cell by construction, so the caller has to hand it over.

    `brain._last_whatif` is exactly the shape `verify` already reads, and this is the same
    read: every output's before and after is a number the consultant is entitled to say."""
    out: list[Cand] = []
    if not isinstance(whatif, dict):
        return out
    for row in list(whatif.get("outputs") or []) + list(whatif.get("changes") or []):
        if not isinstance(row, dict):
            continue
        for k in ("now", "was"):
            v = row.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(Cand(float(v), str(row.get("cell") or ""),
                                str(row.get("label") or ""), "", 0.0, "whatif"))
    return out


def _scratch_cands(scratch: Any) -> list[Cand]:
    """`brain.scratch`: the last five what-ifs as the page shows them, values as strings."""
    out: list[Cand] = []
    for row in list(scratch or [])[:8]:
        if not isinstance(row, dict):
            continue
        for key in ("value", "output"):
            for s in _Reader(_normalise(str(row.get(key) or ""))).read():
                out.append(Cand(s.value, "", str(row.get("label") or ""), "", 0.0, "scratch"))
        for o in list(row.get("outputs") or [])[:6]:
            if isinstance(o, dict):
                for s in _Reader(_normalise(str(o.get("now") or ""))).read():
                    out.append(Cand(s.value, "", str(o.get("label") or ""), "", 0.0, "scratch"))
    return out


# ----------------------------------------------------------------------------- entry point
def guard_for(index: Any, vocab: Any = None) -> Guard | None:
    """The guard for this index build, cached on the index like `verify._verify_cache`.

    Rebuilt when `index.version` moves, which is what a re-index bumps. Costs one
    `ask.build_vocab` (1 ms on sample_context, 30 ms on the big test model) plus the rarity tables,
    so it must not be built on the transcript thread: call it from prewarm or housekeeping.
    """
    got = getattr(index, "_spoken_guard", None)
    if got is not None and got.version == getattr(index, "version", 0) and vocab is None:
        return got
    try:
        g = Guard(index, vocab)
    except Exception as e:                                 # noqa: BLE001
        log.debug("spoken: guard unavailable: %s", e)
        return None
    try:
        index._spoken_guard = g
    except Exception:                                      # noqa: BLE001
        pass
    return g


def check(line: str, index: Any, whatif: Any = None, scratch: Any = None,
          prior: dict | None = None, vocab: Any = None, **kw) -> Spoken:
    """Check one Me line. No model call, no network, no thread.

        v = check(u.text, self.index, whatif=self._last_whatif, scratch=self.scratch,
                  prior=self._ask_prior())
        if v.verdict == "contradicts":
            await self.emit(v.card(aid=new_id()))
    """
    g = guard_for(index, vocab)
    if g is None:
        return Spoken("silent", "no index")
    return g.check(line, whatif=whatif, scratch=scratch, prior=prior, **kw)


__all__ = ["Spoken", "Said", "Guard", "check", "figures", "guard_for",
           "REL", "HEDGE_MULT", "FLOOR_METRIC", "FLOOR_LEVER", "FLOOR_QA", "NEIGHBOURS"]
