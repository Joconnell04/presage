"""Post-STT repair: the workbook's own words, put back into the line before the brain reads it.

E46. Parakeet is a general recogniser with no idea what is on the consultant's screen, so the
words it gets wrong are exactly the words that matter: **ARR** comes back as "air", **AUM** as
"arm" or "orm", **APAC** as "epic" or "apic", **LATAM** as "latum", **UX** as "ucks", "Low case"
as "lowcase", "SaaS churn" as "sauce journ". E9b measured what that costs: a mangled *figure* is
a wrong figure the guard catches, but a mangled *label* is a referent nothing can recover, and
both of E9b's false alarms are one word ("AUM bracket" heard as "ORM bracket") resolving to the
wrong row.

The repair is not a language model and not a second decode. It is a lookup:

1. **Terms.** Every string the index already carries -- row labels, column headers, sheet and
   table names, filter values, definition names, lever labels and their aliases, document
   headings, and the capitalised names in the documents -- plus the distinctive single words and
   pairs inside them. Built once per index version, off the transcript thread.
2. **A phonetic key**, twice. A compact consonant-class key ("apac" and "epic" are both `APK`),
   and a second one with post-vocalic r dropped, which is the only way "arm" and "aum" ever meet
   (`AM`). A span matches a term when one of its two keys is exactly one of the term's. One edit
   apart is available (`PRESAGE_LEXICON_MAX_D=1`, through a deletion index) and is off: it finds
   nothing the exact match misses on 597 measured lines, and it turns "opex" into "capex".
3. **A frequency list** (`data/lexicon/common.txt`, the WordPiece vocabulary of
   `bert-base-uncased` in its own frequency order). A word Parakeet wrote that is ordinary
   English is what the consultant probably said, so it is left alone. The one exception, and it
   is the exception that buys most of the repair, is an **acronym**: "air" is a common word and
   ARR is a term in the workbook. That exception is narrow on purpose. The acronym has to come
   from the workbook rather than from prose, the two have to be within two letters and one letter
   of length ("air" for ARR, never "guess" for COGS), and something else within two words has to
   be a workbook term of its own, which is the difference between "the churned air" and "good to
   connect". The top of the list, the function words, is never rewritten at all.
4. **Numbers**, separately and conservatively. "twenty-five" to "25", "22 000" to "22,000",
   "9bn" to "9 billion", "q3ebitda" to "Q3 EBITDA", "year zero" to "year 0", "two-fifty" to
   "250". A spelled mantissa becomes digits; the *scale word* stays a word, because
   `presage/spoken.py` reads precision off how the figure was said and "9 million" claims one
   digit while "9000000" claims seven. A phrase that ends on "hundred" is left spelled for the
   same reason: `spoken._Reader._words` folds "two hundred thousand" into a mantissa of 2 at 1e5
   on purpose.

`Correction` keeps the original beside the repair, so the page can show what was heard.

Everything here is pure CPU and no allocation of consequence: `correct()` is a few dict lookups
per word over a memo table that a call fills in the first minute. Measured in
docs/experiments/E46-lexicon.md.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("presage.stt.lexicon")

ROOT = Path(__file__).resolve().parents[2]
COMMON_FILE = ROOT / "data" / "lexicon" / "common.txt"


def _flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() not in (
        "0", "false", "no", "off", "")


ON = _flag("PRESAGE_LEXICON")
TERMS_ON = _flag("PRESAGE_LEXICON_TERMS")
NUMBERS_ON = _flag("PRESAGE_LEXICON_NUMBERS")

# A word this common is what the consultant said, not what the recogniser invented. Ranks are
# positions in data/lexicon/common.txt: `air` 228, `arm` 740, `case` 479, `epic` 5736,
# `sauce` 9297, and `apac` / `churn` / `ebitda` / `tranche` are not in it at all.
PROTECT = int(os.environ.get("PRESAGE_LEXICON_PROTECT", "1500"))
# The function words, never rewritten whatever matches them: "are" is phonetically ARR.
CORE = int(os.environ.get("PRESAGE_LEXICON_CORE", "150"))
# A term this far down the list is a domain word rather than an English one, so putting it in
# place of another English word is a repair rather than a coin toss.
DISTINCT = int(os.environ.get("PRESAGE_LEXICON_DISTINCT", "3000"))
# An acronym may take a common word off it (air -> ARR). Off makes the corrector obey
# "never rewrite a common word" literally; the cost is in the doc.
ACRONYMS = _flag("PRESAGE_LEXICON_ACRONYMS")
MAX_SPAN = int(os.environ.get("PRESAGE_LEXICON_SPAN", "3"))       # words per replaceable span
MAX_TERMS = int(os.environ.get("PRESAGE_LEXICON_MAX_TERMS", "20000"))
ACRO_EDIT = int(os.environ.get("PRESAGE_LEXICON_ACRO_EDIT", "2"))   # letters, not sounds
# How far apart two phonetic keys may be. 0 is exact-on-either-key, which is what ships:
# one edit finds nothing the exact match misses on 597 measured lines, and it turns "opex"
# into "capex", which is a different row of the same sheet.
MAX_D = int(os.environ.get("PRESAGE_LEXICON_MAX_D", "0"))
MIN_KEY = 2                       # a one-symbol key collides with everything
MEMO_CAP = 20000


# --------------------------------------------------------------------------- the phonetic key

_VOWEL = frozenset("aeiou")
# One symbol per consonant class. j / ch / sh share X on purpose: "journ" is how Parakeet
# writes "churn", and "sauce" is how it writes "SaaS".
_SINGLE = {"b": "P", "p": "P", "f": "F", "v": "F", "d": "T", "t": "T",
           "k": "K", "q": "K", "j": "X", "z": "S", "s": "S",
           "l": "L", "m": "M", "n": "N", "r": "R", "g": "K", "c": "K"}
_SOFT = frozenset("eiy")          # c and g go soft in front of these


def phones(word: str) -> tuple[str, str]:
    """(rhotic, non-rhotic) keys for one word. "arm" -> ("ARM", "AM"), "aum" -> ("AM", "AM")."""
    s = "".join(ch for ch in word.lower() if ch.isalnum())
    if not s:
        return "", ""
    out: list[str] = []
    drop_r: list[bool] = []                    # per emitted symbol: is it a post-vocalic r?
    i, n = 0, len(s)
    while i < n:
        c, nxt = s[i], s[i + 1] if i + 1 < n else ""
        sym, step = "", 1
        if c.isdigit():
            sym, step = c, 1                   # FY25 and FY26 do not sound alike
        elif c in _VOWEL:
            # the first sound, not the first letter: "honestly" opens on a vowel
            sym, step = ("A" if not out else ""), 1
        elif c == "h":
            sym, step = "", 1                  # a bare h is air; the digraphs are handled below
        elif c == "w":
            if nxt == "h":
                step = 2
            sym = "W" if i == 0 else ""
        elif c == "y":
            sym = "Y" if i == 0 else ""
        elif c == "p" and nxt == "h":
            sym, step = "F", 2
        elif c == "g" and nxt == "h":
            sym, step = "", 2                  # "through", "night": silent
        elif c == "c" and nxt == "h":
            sym, step = "X", 2
        elif c == "s" and nxt == "h":
            sym, step = "X", 2
        elif c == "t" and nxt == "h":
            sym, step = "T", 2
        elif c == "c" and nxt == "k":
            sym, step = "K", 2
        elif c == "x":
            sym, step = "KS", 1
        elif c in ("c", "g") and nxt in _SOFT:
            sym, step = ("S" if c == "c" else "X"), 1
        else:
            sym, step = _SINGLE.get(c, ""), 1
        if sym:
            # post-vocalic r, the one an English speaker barely says: "arm" and "aum" are the
            # same sound. A doubled letter is one sound, so "carries" keeps its r.
            j = i + 1
            while j < n and s[j] == c:
                j += 1
            post_r = (c == "r" and i > 0 and s[i - 1] in _VOWEL
                      and (j >= n or s[j] not in _VOWEL))
            for ch in sym:
                if out and out[-1] == ch:      # "arr", "ss": one symbol
                    continue
                out.append(ch)
                drop_r.append(post_r)
        i += step
    rho = "".join(out)
    nonrho = "".join(ch for ch, d in zip(out, drop_r) if not d)
    # collapse again: dropping the r can leave a double ("barb" -> P P)
    if nonrho != rho:
        nonrho = re.sub(r"(.)\1+", r"\1", nonrho)
    return rho, nonrho


def _letters(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _edit(a: str, b: str, cap: int = 4) -> int:
    """Levenshtein, given up on at `cap`. Only ever called on two short strings."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            best = min(best, v)
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _dist1(a: str, b: str) -> bool:
    """Is the Levenshtein distance between two keys at most 1? Cheap, never builds a matrix."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) <= 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


# --------------------------------------------------------------------------- numbers

_ONES = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE = {"hundred", "thousand", "grand", "k", "million", "mil", "m", "billion", "bn", "b",
          "trillion"}
_UNIT_AFTER = _SCALE | {"percent", "percentage", "pct", "%", "bps", "bips", "points", "point",
                        "dollars", "bucks", "x", "basis"}
# Where the workbook itself writes a digit: "Year 0", "Tranche 1", "Bracket 2", "Scenario 3".
_ORDINAL_BEFORE = {"year", "years", "quarter", "q", "month", "week", "day", "tranche",
                   "bracket", "phase", "stage", "scenario", "case", "option", "tier", "round",
                   "version", "step", "part", "row", "line", "slide", "page", "fy", "cy",
                   "band", "level", "cohort", "segment", "wave", "sheet", "tab", "column"}

_HYPHEN_NUM = re.compile(
    r"\b((?:%s))-(?=(?:%s)\b)" % ("|".join(sorted(_ONES) + sorted(_TENS)),
                                  "|".join(sorted(_ONES) + sorted(_TENS) + ["and", "a"])), re.I)
# "22 000" and "1 500 000": a space where a thousands separator belongs.
_SPACE_THOUSANDS = re.compile(r"(?<![\d,.])(\d{1,3})((?:\s\d{3})+)(?![\d.])")
# a label glued to the digits in front of it: "q3ebitda". Unit suffixes are spared, and so is
# anything the sheet writes that way (FY25, V2, Q3) because the letters come first there.
_GLUED_LABEL = re.compile(r"(\d+)([A-Za-z]{2,})\b")
_GLUED_KEEP = frozenset(("bn", "bps", "bips", "st", "nd", "rd", "th", "pt", "pts", "pp",
                         "mm", "bb", "kg", "km", "am", "pm", "mn", "tn", "yr", "yrs", "hr",
                         "hrs", "ish"))
_BN = re.compile(r"(?<=\d)\s*bn\b|\b(?<=\d\s)bn\b", re.I)
_PERCENT_WORDS = re.compile(r"\bper\s+cent\b", re.I)
# A token keeps an inner dot ("V2.1", "9.6") and drops the one that ends a sentence.
_WORD = re.compile(r"[A-Za-z0-9'’&/+_-]+(?:\.[A-Za-z0-9][A-Za-z0-9'’&/+_-]*)*")


_INSIDE = frozenset(set(_ONES) | set(_TENS) | {"hundred", "and", "a", "an", "point"})


def _mantissa(words: list[str], i: int) -> tuple[str, int] | None:
    """The spelled number starting at words[i] as digits, and where it ends.

    Returns None when there is nothing to gain or something to lose: a bare "one", a phrase with
    "hundred" in it (`spoken._Reader._words` folds "two hundred thousand" into a mantissa of 2 at
    1e5 on purpose, and digits would claim a precision the consultant did not), a year-shaped
    pair, or a start in the middle of a number that began a word earlier.
    """
    if i and words[i - 1] in _INSIDE:
        return None
    cur, seen, frac, ends_hundred = 0, False, "", False
    j, n = i, len(words)
    while j < n:
        w = words[j]
        if w in ("and", "a", "an") and seen and j + 1 < n and (
                words[j + 1] in _ONES or words[j + 1] in _TENS):
            j += 1
            continue
        if w == "point" and seen:
            k, digits = j + 1, ""
            while k < n and words[k] in _ONES and _ONES[words[k]] < 10:
                digits += str(_ONES[words[k]])
                k += 1
            if not digits:
                break
            frac, j = digits, k
            break
        if w in _ONES:
            if cur and _ONES[w] >= 10:
                break                               # "twelve twelve" is not one number
            if cur and cur % 100 == 0 and cur >= 100:
                cur += _ONES[w]
            elif cur and cur < 10:
                break                               # "two three" is two numbers
            else:
                cur += _ONES[w]
            ends_hundred = False
        elif w in _TENS:
            if cur and cur < 10:
                cur = cur * 100 + _TENS[w]          # "two fifty" is 250
            elif cur and cur >= 100 and cur % 100 == 0:
                cur += _TENS[w]
            elif cur:
                break
            else:
                cur += _TENS[w]
            ends_hundred = False
        elif w == "hundred":
            ends_hundred = True
            break
        else:
            break
        seen = True
        j += 1
    if not seen or ends_hundred:
        return None
    out = f"{cur}.{frac}" if frac else str(cur)
    return out, j


def normalise_numbers(text: str, before_digit: Iterable[str] = ()) -> str:
    """The spoken forms of a figure, written the way the sheet writes them.

    Conservative on purpose: the mantissa becomes digits and the scale word stays a word, so a
    figure still claims exactly the precision it was said with.
    """
    if not text:
        return text
    t = _PERCENT_WORDS.sub("percent", text)
    t = _HYPHEN_NUM.sub(r"\1 ", t)                                   # twenty-five -> twenty five
    t = _SPACE_THOUSANDS.sub(lambda m: m.group(1) + m.group(2).replace(" ", ","), t)
    t = _BN.sub(" billion", t)
    t = _GLUED_LABEL.sub(
        lambda m: f"{m.group(1)} {m.group(2)}" if m.group(2).lower() not in _GLUED_KEEP
        else m.group(0), t)

    spans = [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(t)]
    low = [s[2].lower().strip(".,;:!?") for s in spans]
    starts = set(_ORDINAL_BEFORE) | {w.lower() for w in before_digit}
    out, i, cut = [], 0, 0
    while i < len(low):
        got = _mantissa(low, i)
        if got is None:
            i += 1
            continue
        digits, j = got
        multi = j - i > 1
        after = low[j] if j < len(low) else ""
        prev = low[i - 1] if i else ""
        # one bare number word is only a digit where the sheet writes one: "year zero",
        # "tranche one", "about 9 million". Everywhere else it is a determiner.
        if not (multi or after in _UNIT_AFTER or prev in starts):
            i = j
            continue
        out.append(t[cut:spans[i][0]])
        out.append(digits)
        cut = spans[j - 1][1]
        i = j
    out.append(t[cut:])
    return "".join(out)


# --------------------------------------------------------------------------- the term table

@dataclass(slots=True)
class Term:
    text: str                  # the surface the sheet uses: "APAC", "AUM Bracket (S2)"
    words: int
    key: str
    nkey: str
    acronym: bool
    distinct: bool = False     # a domain word, not an English one: safe to put over a word
    whole: bool = False        # the index's own complete string, not a piece one was cut from
    novel: bool = False        # holds a word that is not in the English list at all
    src: str = "vocab"         # "vocab": a label, header, value or lever. "doc": prose.


@dataclass(slots=True)
class Correction:
    text: str                  # what the brain should read
    heard: str                 # what the recogniser actually wrote
    changed: list[tuple[str, str]] = field(default_factory=list)
    ms: float = 0.0

    @property
    def touched(self) -> bool:
        return self.text != self.heard


_NUMWORD = frozenset(set(_ONES) | set(_TENS) | _SCALE | {"percent", "half"})
_ACRONYM = re.compile(r"^[A-Za-z][A-Za-z&.]{0,6}$")
_SPLIT = re.compile(r"[^A-Za-z0-9'’%]+")
_DEAD = frozenset(("the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "at", "by",
                   "with", "as", "is", "are", "was", "were", "be", "per", "from", "vs", "no",
                   "yes", "total", "year", "month", "value", "name", "type", "date", "item",
                   "unnamed", "column", "row", "sheet", "table", "notes", "other", "all"))


def _common() -> dict[str, int]:
    try:
        words = COMMON_FILE.read_text(encoding="utf-8").split()
    except OSError:
        # The list is data, not code: without it nothing but the function words is protected,
        # which is the wrong way round, so fall back to protecting everything that is a word.
        log.warning("no %s: term repair falls back to the built-in stop list", COMMON_FILE)
        words = sorted(_DEAD)
    return {w: i for i, w in enumerate(words)}


_COMMON: dict[str, int] | None = None
_SUFFIX = ("s", "es", "ed", "ing", "ly", "er", "ers", "est", "'s", "’s", "y", "ies")


def common_ranks() -> dict[str, int]:
    global _COMMON
    if _COMMON is None:
        _COMMON = _common()
    return _COMMON


def rank_of(word: str, table: dict[str, int]) -> int:
    """The word's frequency rank, -1 when it is not English at all.

    A WordPiece vocabulary keeps the lemma and throws the inflections away, so "swings",
    "audits" and "hopping" are missing from a list that has "swing", "audit" and "hop". Without
    the backoff every one of them reads as a token Parakeet invented, and the corrector feels
    free to overwrite it. The rank comes back off the stem, so an inflection is as protected as
    the word it is an inflection of.
    """
    r = table.get(word, -1)
    if r >= 0 or len(word) < 4:
        return r
    for suf in _SUFFIX:
        if not word.endswith(suf) or len(word) - len(suf) < 3:
            continue
        stem = word[: len(word) - len(suf)]
        for cand in (stem, stem + "e", stem[:-1] if stem[-1:] == stem[-2:-1] else ""):
            if cand:
                r = table.get(cand, -1)
                if r >= 0:
                    return r
    return -1


class Lexicon:
    """The term table for one index version, with the two phonetic indexes over it."""

    def __init__(self, folder: str = "", version: Any = None):
        self.folder = folder
        self.version = version
        self.terms: dict[str, Term] = {}
        self.by_key: dict[str, list[Term]] = {}
        self.by_nkey: dict[str, list[Term]] = {}
        self.by_del: dict[str, list[Term]] = {}
        self.before_digit: set[str] = set()
        self.common = common_ranks()
        self._memo: dict[str, tuple[str, str]] = {}
        self._rank: dict[str, int] = {}
        self.build_ms = 0.0
        self.stats = {"lines": 0, "changed": 0, "terms": 0}

    # -- building ---------------------------------------------------------------------
    def add(self, surface: Any, *, split: bool = True, src: str = "vocab") -> None:
        s = str(surface or "").strip()
        if not s or len(s) > 64 or len(self.terms) >= MAX_TERMS:
            return
        self._add_one(s, whole=True, src=src)
        if not split:
            return
        parts = [p for p in _SPLIT.split(s) if p]
        if len(parts) > 1:
            for k in range(len(parts)):
                self._add_one(parts[k], src=src)              # "Design and UX" -> "UX"
                if k + 1 < len(parts):
                    self._add_one(" ".join(parts[k:k + 2]), src=src)
                if k + 2 < len(parts):
                    self._add_one(" ".join(parts[k:k + 3]), src=src)

    def rank(self, word: str) -> int:
        """Frequency rank with the inflection backoff, memoised: it is read per word per line."""
        r = self._rank.get(word)
        if r is None:
            r = self._rank[word] = rank_of(word, self.common)
        return r

    def _anchor(self, word: str) -> bool:
        """Is this neighbour one of the workbook's own words, rather than plain English?"""
        t = self.terms.get(word)
        if t is None:
            return False
        r = self.rank(word)
        return t.acronym or r < 0 or r >= PROTECT

    def _distinct(self, word: str) -> bool:
        r = self.rank(word.lower())
        return word.lower() not in _DEAD and (r < 0 or r >= DISTINCT)

    def _add_one(self, s: str, whole: bool = False, src: str = "vocab") -> None:
        words = [w for w in _SPLIT.split(s) if w]
        if not words or len(words) > MAX_SPAN or any(len(w) < 2 for w in words):
            return
        low = " ".join(w.lower() for w in words)
        if low in self.terms:
            if whole:
                self.terms[low].whole = True   # first seen as a fragment, now in its own right
            return
        if len(low) < 2:
            return
        if len(words) == 1:
            w = words[0]
            if low in _DEAD or low.isdigit() or low in _NUMWORD:
                return                                        # "zero" is a number, not a label
            acro = bool(_ACRONYM.match(w)) and sum(c.isupper() for c in w) >= max(
                2, int(0.6 * sum(c.isalpha() for c in w)))
            # a snake_case column gives up its parts, and a short one is an acronym the
            # sheet happened to write in lower case: arr_usd -> ARR, USD
            if not acro and len(w) <= 4 and w.islower() and low not in self.common:
                w, acro = w.upper(), True
            rho, non = phones(low)
            if len(rho) < MIN_KEY:
                return
            t = Term(w if acro else s, 1, rho, non, acro, acro or self._distinct(low),
                     whole, acro or self.rank(low) < 0, src)
        else:
            if all(w.lower() in _DEAD or 0 <= self.rank(w.lower()) < CORE for w in words):
                return                                        # "for the" is not a term
            rho, non = "", ""
            for w in words:
                a, b = phones(w)
                rho, non = rho + a, non + b
            if len(rho) < MIN_KEY + 1:
                return
            t = Term(s, len(words), rho, non, False,
                     any(self._distinct(w) for w in words), whole,
                     any(self.rank(w.lower()) < 0 for w in words), src)
        self.terms[low] = t
        self.by_key.setdefault(t.key, []).append(t)
        if t.nkey != t.key:
            self.by_nkey.setdefault(t.nkey, []).append(t)
        if MAX_D > 0:            # the deletion index only exists to find a one-edit neighbour
            self.by_del.setdefault(t.key, []).append(t)
            if len(t.key) <= 12:
                for i in range(len(t.key)):
                    self.by_del.setdefault(t.key[:i] + t.key[i + 1:], []).append(t)
        if len(words) > 1 and words[-1].isdigit():
            self.before_digit.add(words[-2].lower())

    def finish(self) -> None:
        self.stats["terms"] = len(self.terms)

    # -- correcting -------------------------------------------------------------------
    def _keys(self, raw: str) -> tuple[str, str]:
        got = self._memo.get(raw)
        if got is None:
            rho = non = ""
            for w in raw.split():
                a, b = phones(w)
                rho, non = rho + a, non + b
            if len(self._memo) >= MEMO_CAP:
                self._memo.clear()
            got = self._memo[raw] = (rho, non)
        return got

    def _tier(self, words: list[str]) -> str:
        """How much of this span is ordinary English: core | common | rare | noise.

        A contraction is looked up on its stem, so "what's" is the function word "what" and
        not an unknown token the corrector is free to overwrite.
        """
        best = 10 ** 9
        for w in words:
            r = self.rank(w)
            if r < 0 and ("'" in w or "’" in w):
                r = self.rank(re.split(r"['’]", w)[0])
            if 0 <= r < best:
                best = r
        if best < CORE:
            return "core"
        if best < PROTECT:
            return "common"
        return "rare" if best < 10 ** 9 else "noise"

    def pick(self, raw: str, near: Iterable[str] = ()) -> Term | None:
        """The term this span should have been, or None to leave it alone.

        `near` is the couple of words on either side. It only matters for the acronym rule:
        an acronym takes a common English word ("air" for ARR) only when something else in
        the sentence is already one of the workbook's own domain words, which is the
        difference between "the churned air" and "good to connect".
        """
        if raw in self.terms:
            return None                                       # already the sheet's own word
        words = raw.split()
        tier = self._tier(words)
        if tier == "core":
            return None
        rho, non = self._keys(raw)
        if len(rho) < MIN_KEY:
            return None
        anchored = None
        raw_letters = _letters(raw)
        ceiling = max(2, int(0.34 * len(raw_letters)))
        best: Term | None = None
        best_score = ()
        seen: set[int] = set()

        def is_anchored() -> bool:
            nonlocal anchored
            if anchored is None:
                anchored = any(self._anchor(w) for w in near)
            return anchored

        def offer(t: Term, d: int) -> None:
            nonlocal best, best_score
            if id(t) in seen or t.text.lower() == raw:
                return
            seen.add(id(t))
            if not self._accept(tier, t, d, words):
                return
            if tier == "common" and not is_anchored():
                return
            # two terms can sound the same; the one that is also spelled like what Parakeet
            # wrote is the one it meant.
            spelled = _edit(raw_letters, _letters(t.text))
            term_letters = _letters(t.text)
            if abs(len(term_letters) - len(raw_letters)) > 1 and (
                    term_letters in raw_letters or raw_letters in term_letters):
                return       # "pre-arbitration" holds "arbitration": a prefix, not a mishearing
            if t.acronym and tier != "noise":
                # An acronym is a two- or three-symbol key that collides with half the language,
                # so to take a real English word it has to look like it too: "air" for ARR and
                # "arm" for AUM, never "guess" for COGS or "dig" for TCO. A token that is not a
                # word at all ("ucks" for UX) is exempt: there is nothing to protect.
                if spelled > ACRO_EDIT or abs(len(term_letters) - len(raw_letters)) > 1:
                    return
            elif not t.acronym and spelled > ceiling:
                return
            score = (d, 0 if t.whole else 1, spelled, abs(t.words - len(words)),
                     abs(len(t.key) - len(rho)), 0 if t.acronym else 1, len(t.text))
            if best is None or score < best_score:
                best, best_score = t, score

        for key in (rho, non):
            for bag in (self.by_key, self.by_nkey):
                for t in bag.get(key, ()):
                    offer(t, 0)
        if best is None and MAX_D > 0:
            cand = list(self.by_del.get(rho, ()))
            if len(rho) <= 12:
                for i in range(len(rho)):
                    cand += self.by_del.get(rho[:i] + rho[i + 1:], ())
            for t in cand:
                if _dist1(t.key, rho):
                    offer(t, 1)
        return best

    def _accept(self, tier: str, t: Term, d: int, words: list[str]) -> bool:
        """Is this rewrite worth the risk? The whole policy is these eight lines.

        It gets stricter as the words Parakeet wrote get more like English. A token that is
        not a word at all is simply wrong and almost anything beats it; a word off the bottom
        of the frequency list is replaced only by a domain word that sounds exactly like it; a
        word a consultant says every day is replaced only by an acronym, which is the case
        ("air" for ARR, "arm" for AUM) that the whole exercise is about; and a function word is
        never replaced by anything.
        """
        if t.words == 1 and len(words) == 1 and t.text.lower().rstrip("s") == words[0].rstrip("s"):
            return False                                      # a plural is not an error
        if tier == "common":
            return bool(ACRONYMS and t.acronym and d == 0 and t.src == "vocab")
        if tier == "rare":
            return d == 0 and ((t.acronym and t.src == "vocab") or (t.whole and t.novel))
        if t.src != "vocab" and not t.whole:
            return False                                      # a piece of a sentence in a memo
        # noise: not a word at all, so almost anything beats it
        return d == 0 or (d <= MAX_D and t.words == len(words))

    def correct(self, text: str) -> Correction:
        t0 = time.perf_counter()
        heard = text or ""
        out = heard
        changed: list[tuple[str, str]] = []
        if out and NUMBERS_ON:
            out = normalise_numbers(out, self.before_digit)
        if out and TERMS_ON and self.by_key:
            out, changed = self._terms(out)
        self.stats["lines"] += 1
        if out != heard:
            self.stats["changed"] += 1
        return Correction(out, heard, changed, (time.perf_counter() - t0) * 1000.0)

    def _terms(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        spans = [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(text)]
        if not spans:
            return text, []
        low = [s[2].lower().strip(".,;:!?'’") for s in spans]
        pieces: list[str] = []
        changed: list[tuple[str, str]] = []
        i, cut, n = 0, 0, len(spans)
        while i < n:
            hit = None
            for k in range(min(MAX_SPAN, n - i), 0, -1):
                if k > 1:
                    gap = text[spans[i + k - 2][1]:spans[i + k - 1][0]]
                    if gap.strip(" -"):                       # a comma ends the phrase
                        continue
                raw = " ".join(low[i:i + k])
                if not raw or raw[0].isdigit():
                    continue
                t = self.pick(raw, low[max(0, i - 2):i] + low[i + k:i + k + 2])
                if t is not None:
                    hit = (k, raw, t)
                    break
            if hit is None:
                i += 1
                continue
            k, raw, t = hit
            pieces.append(text[cut:spans[i][0]])
            pieces.append(t.text)
            changed.append((text[spans[i][0]:spans[i + k - 1][1]], t.text))
            cut = spans[i + k - 1][1]
            i += k
        if not changed:
            return text, []
        pieces.append(text[cut:])
        return "".join(pieces), changed


# --------------------------------------------------------------------------- building one

def _rows(con, query: str, limit: int = 4000) -> list[tuple]:
    try:
        return list(con.execute(query + f" LIMIT {limit}"))
    except Exception as e:                                    # noqa: BLE001
        log.debug("lexicon: %s", e)
        return []


def from_db(folder: Path | str) -> Lexicon:
    """The term table from the index's own SQLite file. No Index object, no chunks, no model."""
    from presage.ingest import store as _store

    lx = Lexicon(str(folder))
    t0 = time.perf_counter()
    p = _store.db_path(Path(folder))
    if not p.exists():
        lx.finish()
        return lx
    con = _store.connect_ro(Path(folder))
    try:
        for q, split in (
            ("SELECT DISTINCT row_label FROM cells WHERE row_label IS NOT NULL", True),
            ("SELECT DISTINCT header FROM cells WHERE header IS NOT NULL", True),
            ("SELECT DISTINCT sheet FROM _tables WHERE sheet IS NOT NULL", True),
            ("SELECT DISTINCT title FROM _tables WHERE title IS NOT NULL", True),
            ("SELECT DISTINCT label FROM _columns WHERE label IS NOT NULL", True),
            ("SELECT DISTINCT name FROM _columns WHERE name IS NOT NULL", True),
            ("SELECT DISTINCT name FROM definitions WHERE name IS NOT NULL", True),
            ("SELECT DISTINCT label FROM links WHERE label IS NOT NULL", True),
            ("SELECT DISTINCT value_text FROM cells WHERE value_text IS NOT NULL "
             "AND length(value_text) <= 32", True),
        ):
            for r in _rows(con, q):
                lx.add(r[0], split=split)
    finally:
        try:
            con.close()
        except Exception:                                     # noqa: BLE001
            pass
    lx.finish()
    lx.build_ms = (time.perf_counter() - t0) * 1000.0
    return lx


_CAPS = re.compile(r"\b([A-Z][A-Za-z0-9&.'’-]+(?:\s+[A-Z][A-Za-z0-9&.'’-]+){0,2})")
_ALLCAPS = re.compile(r"\b([A-Z][A-Z&.]{1,5})\b")


def build(index: Any, version: Any = None, levers: bool = False) -> Lexicon:
    """Everything `from_db` finds, plus the ask vocabulary, the levers and the documents."""
    folder = str(getattr(index, "folder", "") or "")
    t0 = time.perf_counter()
    lx = from_db(folder) if folder else Lexicon("")
    lx.version = getattr(index, "version", None) if version is None else version

    try:
        from presage.llm import ask as _ask
        v = _ask.build_vocab(index)
        for s in list(v.metrics) + list(v.periods) + list(v.segments):
            lx.add(s)
    except Exception as e:                                    # noqa: BLE001
        log.debug("lexicon: no ask vocabulary (%s)", e)

    try:
        table = index.levers() if levers else [
            lv for rec in getattr(index, "_scratch", {}).values()
            for lv in (rec or {}).get("levers", [])]
        for lv in table:
            lx.add(lv.get("label"))
            lx.add(lv.get("section"))
            for a in (lv.get("aliases") or [])[:8]:
                lx.add(a)
    except Exception as e:                                    # noqa: BLE001
        log.debug("lexicon: no levers (%s)", e)

    # the documents: headings are deck and memo titles, and a name that recurs in the prose is a
    # client or a product. Once is a turn of phrase; twice is a name.
    seen: dict[str, int] = {}
    for c in getattr(index, "chunks", [])[:4000]:
        if getattr(c, "heading", ""):
            lx.add(c.heading)
        text = getattr(c, "text", "") or ""
        for m in _ALLCAPS.finditer(text):
            seen[m.group(1)] = seen.get(m.group(1), 0) + 2
        for m in _CAPS.finditer(text):
            s = m.group(1)
            if len(s) > 2:
                seen[s] = seen.get(s, 0) + 1
    for s, n in seen.items():
        if n >= 2:
            lx.add(s, split=False, src="doc")

    lx.finish()
    lx.build_ms = (time.perf_counter() - t0) * 1000.0
    log.info("lexicon: %d terms for %s in %.0f ms", len(lx.terms), folder or "-", lx.build_ms)
    return lx


# --------------------------------------------------------------------------- the live one

_CACHE: dict[tuple, Lexicon] = {}
_ATTACHED: Any = None


def attach(index: Any) -> None:
    """Hand the STT stage the live index, so levers and documents join the table."""
    global _ATTACHED
    _ATTACHED = index


def for_index(index: Any, version: Any = None, levers: bool = False) -> Lexicon:
    key = (str(getattr(index, "folder", "") or id(index)),
           getattr(index, "version", None) if version is None else version, levers)
    got = _CACHE.get(key)
    if got is None:
        if len(_CACHE) > 8:
            _CACHE.clear()
        got = _CACHE[key] = build(index, version=key[1], levers=levers)
    return got


def _live_index() -> Any:
    """The running server's index, if this process has one. Never imports it: a module that is
    not already loaded is a process that has no server, and the folder alone is enough."""
    if _ATTACHED is not None:
        return _ATTACHED
    mod = sys.modules.get("presage.server")
    hub = getattr(mod, "hub", None) if mod is not None else None
    ix = getattr(hub, "index", None) if hub is not None else None
    return ix if getattr(ix, "folder", None) else None


def current() -> Lexicon | None:
    """The lexicon for the folder in play. Built once, rebuilt when the index moves.

    Cheap enough to call per utterance: a hit is two dict lookups. The build behind a miss is
    50-100 ms, so callers that are on the transcript thread run this in a worker first.
    """
    if not ON:
        return None
    ix = _live_index()
    if ix is not None:
        try:
            return for_index(ix)
        except Exception as e:                                # noqa: BLE001
            log.warning("lexicon: build failed (%s)", e)
            return None
    try:
        from presage.settings import prefs
        folder = Path(prefs.context_dir())
    except Exception:                                         # noqa: BLE001
        return None
    from presage.ingest import store as _store
    p = _store.db_path(folder)
    try:
        stamp = p.stat().st_mtime_ns
    except OSError:
        return None
    key = (str(folder), stamp, False)
    got = _CACHE.get(key)
    if got is None:
        if len(_CACHE) > 8:
            _CACHE.clear()
        got = _CACHE[key] = from_db(folder)
    return got


def correct(text: str) -> Correction:
    """Repair one line against whatever lexicon is in play. Never raises."""
    if not text or not ON:
        return Correction(text or "", text or "")
    try:
        lx = current()
    except Exception as e:                                    # noqa: BLE001
        log.debug("lexicon: %s", e)
        return Correction(text, text)
    if lx is None:
        # numbers need no workbook, so they still run
        out = normalise_numbers(text) if NUMBERS_ON else text
        return Correction(out, text)
    try:
        return lx.correct(text)
    except Exception as e:                                    # noqa: BLE001
        log.warning("lexicon: correct failed (%s)", e)
        return Correction(text, text)


# --------------------------------------------------------------------------- cli

def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="repair a line against a folder's own words")
    ap.add_argument("--folder", default="")
    ap.add_argument("--levers", action="store_true")
    ap.add_argument("--terms", type=int, default=0, help="print this many terms and stop")
    ap.add_argument("text", nargs="*")
    a = ap.parse_args()
    if a.folder:
        from presage.ingest import Index
        lx = build(Index(a.folder), levers=a.levers)
    else:
        lx = current() or Lexicon("")
    print(f"{len(lx.terms)} terms, built in {lx.build_ms:.0f} ms")
    if a.terms:
        for k in sorted(lx.terms)[:a.terms]:
            t = lx.terms[k]
            print(f"  {t.text!r:42s} key={t.key:12s} nkey={t.nkey:12s} "
                  f"{'acronym' if t.acronym else ''}")
        return 0
    for line in (a.text or []):
        c = lx.correct(line)
        print(f"  heard : {c.heard}\n  fixed : {c.text}\n  {c.changed}  {c.ms:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
