"""Validate a card before it renders: a cite-set check, then lexical entailment.

E23. The model is told never to state a number that is not in the files, and mostly obeys.
This checks it mechanically, with no second model call and no network: resolve every citation
on the card to the cells or document numbers it names, pull every number out of `figure`,
`say` and `answer`, and require each one to match something that resolved, within the
tolerance its own printed precision implies. Plain arithmetic over resolved values (a ratio,
a difference, per 12) counts too, because a card is allowed to divide.

Two rules keep the false-flag rate down, and both cost coverage:

* A number the asker said themselves ("at your 1%") is admitted as theirs, not ours.
* A card is decisive only when **every** citation resolved. The model can cite
  "customers.csv › column arr_usd (rows with churned=1)", which names a SQL aggregate and not
  a cell; nothing in the index can confirm or deny that sum, so the verdict is `unverified`
  rather than a rejection.

Usage from the answer path (wiring in docs/experiments/E23-verify.md):

    v = verify(answer, index, whatif=brain._last_whatif)
    if not v.ok:                      # a number contradicts the cells the card itself cites
        ...
    if v.level == "unverified":       # nothing to check it against; badge it
        ...
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

log = logging.getLogger(__name__)

# How strong the evidence was, strongest first. A card's level is the weakest one it leaned on.
LEVELS = ("cell", "doc", "derived", "web", "unverified")
_RANK = {name: i for i, name in enumerate(LEVELS)}

# Tolerances. REL_DIRECT is the floor under the printed precision: "$9.59M" against a cell
# holding 9,586,292 passes on precision alone, but a card that rounds 13.648% to 13.7% needs
# a little more, and a two-step derivation needs more again.
REL_DIRECT = 0.002
REL_DERIVED = 0.01
MAX_POOL = 60          # resolved values kept for matching
MAX_PAIRS = 30         # resolved values entered into the pairwise arithmetic pass
MAX_RANGE = 400        # cells a cited range may expand to
MAX_DERIVED_CLAIMS = 4  # unmatched numbers that get the arithmetic pass


# --------------------------------------------------------------------------- numbers

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALE = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMWORD = re.compile(r"\b((?:(?:%s)(?:[\s-]+|$))+(?:point(?:\s+(?:%s))+\s*)?)" % (
    "|".join(list(_ONES) + list(_TENS) + list(_SCALE) + ["and", "a"]), "|".join(_ONES)), re.I)

# A number as a card writes it. The lookbehind is what keeps "FY25", "Q3", "B48" and "V2.1"
# out: a digit run glued to a letter or a dot is a label, not a figure.
_NUM = re.compile(
    r"(?<![A-Za-z0-9.])(?P<cur>[$£€])?\s?(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|pct|bps|bp|k|m|bn|b|million|thousand|billion|x)?(?![A-Za-z])", re.I)

_SUFFIX = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "b": 1e9,
           "billion": 1e9}
_PCT = {"%", "percent", "pct"}
_BPS = {"bps", "bp"}


def _words_to_digits(text: str) -> str:
    """"twelve percent" -> "12 percent", "two million" -> "2000000".

    Cards and source prose both spell numbers out ("Sixteen customers churned ... out of two
    hundred"), so the same pass runs over the claim and over the text it is checked against.
    Copied from brain.py rather than imported: importing brain pulls in asyncio and the LLM
    clients, and this module has to run in a bench with neither.
    """
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
            return m.group(0)
        val = total + cur
        out = f"{val}.{frac}" if frac else str(val)
        return out + (" " if m.group(0).endswith((" ", "-")) else "")
    return _NUMWORD.sub(convert, text or "")


@dataclass(frozen=True)
class Num:
    """One number as a card prints it: what it denotes, and how precisely it was said."""
    value: float
    kind: str        # "pct" | "money" | "plain"
    ulp: float       # half a unit in the last printed place, scaled like the value
    text: str
    sig: int = 1     # significant digits as printed: "$14.0M" is three, "$3M" is one

    def tol(self, rel: float) -> float:
        return max(self.ulp, abs(self.value) * rel, 1e-9)


def _sig_of(val: float) -> int:
    """Significant digits a source value carries. 3 -> 1, 2,740 -> 3, 0.136 -> 3."""
    s = f"{abs(float(val)):.10g}"
    if "e" in s or "E" in s:
        s = re.split("[eE]", s)[0]
    s = s.replace(".", "").lstrip("0").rstrip("0")
    return len(s) or 1


def numbers_in(text: str) -> list[Num]:
    """Every figure in a line of card text, in order, deduplicated.

    Skipped deliberately: single-digit runs ("3 regions", "one of two"), which carry no
    information and would flag every card, and bare four-digit years, which are labels.
    """
    out: list[Num] = []
    seen: set[tuple[float, str]] = set()
    for m in _NUM.finditer(_words_to_digits(text or "")):
        raw, unit = m.group("num"), (m.group("unit") or "").lower()
        if len(raw.replace(",", "").replace(".", "").lstrip("0")) < 2:
            continue
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        places = len(raw.split(".")[1]) if "." in raw else 0
        ulp = 0.5 * (10.0 ** -places)
        # "thirty million" is 30,000,000 printed to one significant digit, and the cell holds
        # 30,184: the tolerance has to follow the last *significant* place, not the last
        # decimal one. Capped at 2% so a round count ("200 customers") stays a count.
        if places == 0:
            bare = raw.replace(",", "")
            trail = len(bare) - len(bare.rstrip("0"))
            if trail:
                ulp = max(ulp, min(0.5 * (10.0 ** trail), 0.02 * abs(float(bare))))
        if unit in _PCT:
            kind = "pct"
        elif unit in _BPS:
            kind, val, ulp = "pct", val / 100.0, ulp / 100.0
        elif unit in _SUFFIX:
            kind = "money" if m.group("cur") else "plain"
            val, ulp = val * _SUFFIX[unit], ulp * _SUFFIX[unit]
        elif m.group("cur"):
            kind = "money"
        else:
            kind = "plain"
            if places == 0 and "," not in raw and 1900 <= val <= 2100:
                continue                      # a year, not a figure
        key = (round(val, 6), kind)
        if key in seen:
            continue
        seen.add(key)
        digits = raw.replace(",", "").replace(".", "").lstrip("0")
        if "." not in raw:
            digits = digits.rstrip("0")      # "$14.0M" is three digits, "$300K" is one
        out.append(Num(val, kind, ulp, m.group(0).strip(), len(digits) or 1))
    return out


# Factors a SOURCE value may be multiplied by to reach the claim. A percent cell holds 0.136
# and the card says "13.6%"; a workbook in $ thousands holds 30,184 and the card says
# "$30.18M". Factors of ten are not on the list: "1.5" and "15" are different numbers.
_SCALES = {
    "pct": (1.0, 100.0, 0.01),
    "money": (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9),
    "plain": (1.0, 1e3, 1e-3, 1e6, 1e-6),
}
# Candidates the arithmetic pass produces are already in the pool's own units, so the only
# ambiguity left is the workbook's $ thousands convention. Keeping 1e6 and 1e9 here is what
# let an invented "$499.4M" land on the product of two unrelated cells scaled down.
_SCALES_DERIVED = {
    "pct": (1.0, 100.0, 0.01),
    "money": (1.0, 1e3, 1e-3),
    "plain": (1.0, 1e3, 1e-3),
}


def _window(claim: Num, rel: float = REL_DIRECT,
            derived: bool = False) -> tuple[tuple[float, float, float, int], ...]:
    """The bands a source value may fall in to be this claim, one per allowed scale.

    Inverting the comparison once per claim rather than per candidate is what keeps the
    arithmetic pass inside its millisecond: the inner loop becomes two float compares.
    """
    tol = claim.tol(rel)
    scales = (_SCALES_DERIVED if derived else _SCALES)[claim.kind]
    bands = []
    # both signs: a model in $ holds costs and outflows as negatives, and the card says the
    # magnitude ("$3.86M of burn"). The cost is that a sign error is not catchable.
    for v in (claim.value, -claim.value):
        lo, hi = v - tol, v + tol
        bands += [((lo / f, hi / f) if f > 0 else (hi / f, lo / f)) + (f, 0 if f == 1.0 else claim.sig)
                  for f in scales]
    return tuple(bands)


def _in(window: tuple[tuple[float, float, float, int], ...], val: float) -> bool:
    """A scale hop has to preserve significant digits.

    Without that, "$2.997M" matches the integer 3 sitting in a cell that counts retail partner
    partners, because three million is three at a scale the check allows. A hop is only
    evidence when the source itself carries the digits the claim prints.
    """
    for lo, hi, _f, need in window:
        if lo <= val <= hi and (not need or need <= _sig_of(val)):
            return True
    return False


def _agrees(claim: Num, src: float, rel: float = REL_DIRECT) -> bool:
    """True when `src`, at one of the scales a workbook actually uses, is the claim."""
    return _in(_window(claim, rel), src)


# --------------------------------------------------------------------------- citations

_SEP = re.compile(r"\s*(?:›|>|»|\|)\s*")
_A1 = re.compile(r"^\$?([A-Z]{1,3})\$?(\d{1,7})$")
_PAGE = re.compile(r"^(page|slide|sheet|section|row|tab)\s+(.+)$", re.I)
_ROWNUM = re.compile(r"^(?:row|line)\s*(\d{1,7})$", re.I)
_ROWWORD = re.compile(r"^(?:cells?|columns?)\s+", re.I)


@dataclass
class Cite:
    """One citation, parsed. `kind` is how it resolves, not what the model meant by it."""
    file: str = ""       # the file as the index knows it, "" when the folder has no such file
    raw_file: str = ""
    loc: str = ""
    kind: str = ""       # "cells" | "rowlabel" | "doc" | "unresolvable"
    sheet: str = ""
    refs: list[str] = field(default_factory=list)
    label: str = ""
    problem: str = ""

    def text(self) -> str:
        return f"{self.raw_file} › {self.loc}" if self.loc else (self.raw_file or "(no file)")


def _cite_pairs(answer: Any) -> list[tuple[str, str]]:
    """`citations` as (file, loc), tolerating every shape the wire actually carries."""
    out: list[tuple[str, str]] = []
    for c in _get(answer, "citations") or []:
        if isinstance(c, dict):
            f, loc = str(c.get("file") or ""), str(c.get("loc") or "")
            if not f and loc:
                parts = _SEP.split(loc)
                f, loc = parts[0], " › ".join(parts[1:])
        elif isinstance(c, str):
            parts = _SEP.split(c.strip())
            f, loc = parts[0], " › ".join(parts[1:])
        else:
            continue
        if f or loc:
            out.append((f.strip(), loc.strip()))
    return out


def _col(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n


def _letters(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _split_cells(text: str) -> list[str] | None:
    """"D6", "D8, F8, E8", "D6:E9" -> the cell refs, or None when this is not a cell list."""
    refs: list[str] = []
    for part in re.split(r"[,;/]| and ", text):
        part = part.strip().upper().replace(" ", "")
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            ma, mb = _A1.match(a), _A1.match(b)
            if not (ma and mb):
                return None
            c1, c2 = sorted((_col(ma.group(1)), _col(mb.group(1))))
            r1, r2 = sorted((int(ma.group(2)), int(mb.group(2))))
            if (c2 - c1 + 1) * (r2 - r1 + 1) > MAX_RANGE:
                return None
            refs += [f"{_letters(c)}{r}" for c in range(c1, c2 + 1) for r in range(r1, r2 + 1)]
            continue
        m = _A1.match(part)
        if not m:
            return None
        refs.append(f"{m.group(1)}{int(m.group(2))}")
    return refs or None


def _parse_cite(file_: str, loc: str, cache: dict) -> Cite:
    """Decide how a citation resolves, and name the first thing wrong with it."""
    c = Cite(raw_file=file_, loc=loc)
    known = cache["files"]
    key = file_.strip().lower()
    c.file = known.get(key) or known.get(key.rsplit("/", 1)[-1]) or ""
    if not c.file:
        c.kind, c.problem = "unresolvable", f"cites a file the folder does not hold: {file_ or '?'}"
        return c

    text = (loc or "").strip()
    if not text:
        c.kind = "unresolvable"
        return c

    # a document: "page 3", "slide 4", "section Churn and retention"
    docs = cache["docs"]
    if c.file in docs:
        m = _PAGE.match(text)
        unit, rest = (m.group(1).lower(), m.group(2).strip()) if m else ("", text)
        c.kind, c.label = "doc", text
        head = rest.split()[0] if rest.split() else ""
        if unit in ("page", "slide") and head.isdigit():
            n, total = int(head), docs[c.file]
            if total and n > total:
                c.kind = "unresolvable"
                c.problem = f"cites {unit} {n} of {c.file}, which has {total}"
        return c

    # a workbook: "<sheet>!<refs>", "<sheet> › <refs>", "<sheet> › <row label>", "<refs>"
    sheets = cache["sheets"].get(c.file, {})
    sheet, rest = "", text
    if "!" in text:
        sheet, rest = text.split("!", 1)
    else:
        parts = _SEP.split(text)
        if len(parts) > 1:
            sheet, rest = parts[0], " › ".join(parts[1:])
    sheet, rest = sheet.strip().strip("'\""), rest.strip()
    if not sheet and len(sheets) == 1:
        sheet = next(iter(sheets))
    lower = {s.lower(): s for s in sheets}
    if sheet and sheet.lower() not in lower:
        squashed = {s.lower().replace(" ", ""): s for s in sheets}
        hit = squashed.get(sheet.lower().replace(" ", ""))
        if not hit:
            c.kind = "unresolvable"
            c.problem = f"cites a sheet {c.file} does not have: {sheet}"
            return c
        sheet = hit
    c.sheet = lower.get(sheet.lower(), sheet)
    m = _ROWNUM.match(rest)
    if m:                                    # "Regions › row 13": a sheet row, not a cell
        c.kind, c.label = "rownum", m.group(1)
        return c
    rest = _ROWWORD.sub("", rest).strip()
    refs = _split_cells(rest)
    if refs:
        c.kind, c.refs = "cells", refs
    else:
        c.kind, c.label = "rowlabel", rest
    return c


# --------------------------------------------------------------------------- the index

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """An answer field, whether the card arrived as the dataclass or as its wire dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _con(index: Any):
    con = getattr(index, "con", None)
    if con is None:
        raise RuntimeError("index is not open")
    return con


def _cache(index: Any) -> dict:
    """File names, sheet names, document lengths and chunk text, built once per index build.

    Everything in it is small (names and page counts), and it is thrown away when
    `index.version` moves, which is what a re-index bumps.
    """
    got = getattr(index, "_verify_cache", None)
    if got is not None and got.get("version") == getattr(index, "version", 0):
        return got
    con = _con(index)
    files: dict[str, str] = {}
    for (path,) in con.execute("SELECT path FROM _files"):
        files[path.lower()] = path
        files[path.rsplit("/", 1)[-1].lower()] = path
    sheets: dict[str, dict[str, int]] = {}
    for f, s in con.execute("SELECT DISTINCT file, sheet FROM cells"):
        sheets.setdefault(f, {})[s or ""] = 1
    docs = {f: (n or 0) for f, n in con.execute("SELECT file, n_units FROM _documents")}
    # chunk text per (file, loc). The numbers table holds only what the deep pass extracted,
    # and a memo that writes "Sixteen customers ... out of two hundred" puts nothing in it at
    # all. Normalising the spelled-out numbers is deferred: a folder can hold thousands of
    # chunks and one card touches two or three.
    text: dict[tuple[str, str], str] = {}
    for ch in getattr(index, "chunks", []) or []:
        k = (ch.file, (ch.loc or "").strip().lower())
        text[k] = text.get(k, "") + " " + (ch.text or "")
    cache = {"version": getattr(index, "version", 0), "files": files, "sheets": sheets,
             "docs": docs, "text": text, "norm": {}, "wide": {}}
    try:
        index._verify_cache = cache
    except Exception:  # noqa: BLE001  (a duck-typed index may refuse new attributes)
        pass
    return cache


def _resolve(index: Any, c: Cite, cache: dict) -> list[tuple[float, str, str]]:
    """The numbers a citation points at, as (value, where it was, level)."""
    con = _con(index)
    if c.kind == "cells":
        marks = ",".join("?" * len(c.refs))
        rows = con.execute(
            f"SELECT cell_ref, value_num, value_text FROM cells "
            f"WHERE file=? AND sheet=? AND cell_ref IN ({marks})",
            (c.file, c.sheet, *c.refs)).fetchall()
        out: list[tuple[float, str, str]] = []
        found = set()
        for ref, num, txt in rows:
            found.add(ref)
            where = f"{c.file} › {c.sheet} › {ref}" if c.sheet else f"{c.file} › {ref}"
            if num is not None:
                out.append((float(num), where, "cell"))
            elif txt:                        # a label cell, or a number typed as text
                out += [(n.value, where, "cell") for n in numbers_in(str(txt))]
        missing = [r for r in c.refs if r not in found]
        if missing and len(missing) == len(c.refs):
            c.problem = (f"cites a cell that is empty in {c.file}: "
                         f"{c.sheet} › {', '.join(missing[:4])}")
        return out

    if c.kind == "rownum":
        rows = con.execute(
            "SELECT cell_ref, value_num FROM cells WHERE file=? AND sheet=? AND row=? "
            "AND value_num IS NOT NULL LIMIT 200", (c.file, c.sheet, int(c.label))).fetchall()
        if not rows:
            c.kind = "unresolvable"
            return []
        return [(float(v), f"{c.file} › {c.sheet} › {r}", "cell") for r, v in rows]

    if c.kind == "rowlabel":
        rows = con.execute(
            "SELECT cell_ref, value_num FROM cells WHERE file=? AND sheet=? "
            "AND lower(row_label)=? AND value_num IS NOT NULL LIMIT 200",
            (c.file, c.sheet, c.label.lower())).fetchall()
        if not rows:
            rows = con.execute(
                "SELECT cell_ref, value_num FROM cells WHERE file=? AND sheet=? "
                "AND lower(row_label) LIKE ? AND value_num IS NOT NULL LIMIT 200",
                (c.file, c.sheet, f"%{c.label.lower()}%")).fetchall()
        if not rows:
            c.kind = "unresolvable"          # a column description, a SQL predicate, a phrase
            return []
        return [(float(v), f"{c.file} › {c.sheet} › {r}", "cell") for r, v in rows]

    if c.kind == "doc":
        loc = c.label.strip().lower()
        rows = con.execute(
            "SELECT loc, value_num FROM numbers WHERE file=? AND "
            "(lower(loc)=? OR lower(loc) LIKE ?) AND value_num IS NOT NULL LIMIT 400",
            (c.file, loc, f"{loc}%")).fetchall()
        out = [(float(v), f"{c.file} › {l}", "doc") for l, v in rows]
        # the numbers table is a deep-pass product and is thin, so the chunk's own digits
        # join the pool: that is the lexical half of the check.
        for (f, l), txt in cache["text"].items():
            if f == c.file and (l == loc or l.startswith(loc)):
                out += [(n.value, f"{c.file} › {l}", "doc")
                        for n in numbers_in(_normed(cache, f, l, txt))]
        if not out:
            c.kind = "unresolvable"
        return out
    return []


def _normed(cache: dict, file_: str, loc: str, txt: str) -> str:
    key = (file_, loc)
    got = cache["norm"].get(key)
    if got is None:
        got = cache["norm"][key] = _words_to_digits(txt)
    return got


def _file_wide(index: Any, cache: dict, file_: str) -> list[tuple[float, str, str]]:
    """Every number anywhere in one cited file, cached for the life of the index build.

    The second tier of the check. A card that says "$30,184k across 96 customers" while
    citing only `Regions!D6:E9` has under-cited, not invented: 96 is in F6, two columns over.
    Dropping that card would be wrong, so the loose pass asks the weaker question — is this
    number anywhere in a file the card cites — and `Verdict.strict` records that the exact
    citation did not cover it.
    """
    got = cache["wide"].get(file_)
    if got is not None:
        return got
    out: list[tuple[float, str, str]] = []
    try:
        con = _con(index)
        for sheet, ref, num in con.execute(
                "SELECT sheet, cell_ref, value_num FROM cells WHERE file=? "
                "AND value_num IS NOT NULL LIMIT 200000", (file_,)):
            out.append((float(num), f"{file_} › {sheet or ''} › {ref}".replace(" ›  › ", " › "),
                        "cell"))
        for loc, num in con.execute(
                "SELECT loc, value_num FROM numbers WHERE file=? AND value_num IS NOT NULL "
                "LIMIT 20000", (file_,)):
            out.append((float(num), f"{file_} › {loc}", "doc"))
    except Exception as e:  # noqa: BLE001
        log.debug("verify: file-wide pool failed for %s: %s", file_, e)
    for (f, l), txt in cache["text"].items():
        if f == file_:
            out += [(n.value, f"{file_} › {l}", "doc")
                    for n in numbers_in(_normed(cache, f, l, txt))]
    cache["wide"][file_] = out
    return out


def _file_totals(index: Any, cache: dict, file_: str) -> list[tuple[float, str]]:
    """Every column and row total in a cited file, as candidate denominators.

    "Mid-market is 30% of revenue" cites the mid-market cell and nothing else: the total it
    divides by is a column the card never names. Without this the guard rejects a true card
    for the crime of under-citing its denominator.
    """
    got = cache.setdefault("totals", {}).get(file_)
    if got is not None:
        return got
    sums = _group_sums(_file_wide(index, cache, file_))
    out = sorted(((v, k) for k, v in sums.items() if v), key=lambda kv: -abs(kv[0]))[:200]
    cache["totals"][file_] = out
    return out


# --------------------------------------------------------------------------- arithmetic

# The operations a card is allowed to do out loud on numbers it cited. Yielded lazily and in
# rough order of likelihood, so a card that derives its figure stops the sweep early.
_SINGLE = (("per 12", 1 / 12.0), ("× 12", 12.0), ("per 4", 0.25), ("× 4", 4.0),
           ("halved", 0.5), ("doubled", 2.0))


def _derived(vals: list[tuple[float, str, str]]) -> Iterator[tuple[float, str]]:
    v = vals[:MAX_PAIRS]
    for a, sa, _ in v:
        for name, f in _SINGLE:
            yield a * f, f"{sa} {name}"
    for i, (a, sa, _) in enumerate(v):
        for b, sb, _ in v[i + 1:]:
            yield a - b, f"{sa} − {sb}"
            yield b - a, f"{sb} − {sa}"
            yield a + b, f"{sa} + {sb}"
            if b:
                yield a / b, f"{sa} ÷ {sb}"
                yield a / b * 100.0, f"{sa} ÷ {sb} as %"
                yield (a - b) / b * 100.0, f"change {sb} → {sa} as %"
                yield (b - a) / b * 100.0, f"the {sb} − {sa} gap as % of {sb}"
            if a:
                yield b / a, f"{sb} ÷ {sa}"
                yield b / a * 100.0, f"{sb} ÷ {sa} as %"
                yield (b - a) / a * 100.0, f"change {sa} → {sb} as %"
                yield (a - b) / a * 100.0, f"the {sa} − {sb} gap as % of {sa}"
            if abs(a) <= 1000 or abs(b) <= 1000:
                yield a * b, f"{sa} × {sb}"   # a rate or a count times a figure, not two figures
    for key, total in _group_sums(v).items():
        yield total, f"sum of {key}"


_REF_TAIL = re.compile(r"^([A-Z]{1,3})(\d{1,7})$")


def _group_sums(vals: list[tuple[float, str, str]]) -> dict[str, float]:
    """Column and row totals over cited cells. "Enterprise is 60% of revenue" divides by the
    column it sits in, and the card almost never cites the total row."""
    out: dict[str, float] = {}
    for a, src, _lv in vals:
        parts = src.split(" › ")
        m = _REF_TAIL.match(parts[-1].strip().upper()) if len(parts) > 1 else None
        if not m:
            continue
        stem = " › ".join(parts[:-1])
        for key in (f"{stem} column {m.group(1)}", f"{stem} row {m.group(2)}"):
            out[key] = out.get(key, 0.0) + a
    return out


# --------------------------------------------------------------------------- the verdict

@dataclass
class Verdict:
    """`ok` is "no number on this card is missing from the files it cites".

    `strict` is the harder question: was every number in the exact cell or page named. A card
    can be ok and not strict, which is an under-citation and not an invention; the wiring
    note in the E23 doc keeps those two apart on purpose.
    """
    ok: bool
    level: str
    problems: list[str] = field(default_factory=list)
    resolved: list[dict] = field(default_factory=list)
    reason: str = ""
    ms: float = 0.0
    checked: int = 0
    matched: int = 0
    strict: bool = True
    loose_only: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "level": self.level, "problems": self.problems,
                "resolved": self.resolved, "reason": self.reason, "ms": round(self.ms, 3),
                "checked": self.checked, "matched": self.matched, "strict": self.strict,
                "loose_only": self.loose_only}


#: what gets read out loud. `answer` is the detail line, and a number only in there is worth
#: a note rather than a rejection: see the scope table in docs/experiments/E23-verify.md.
SPOKEN = ("figure", "say")
ALL_FIELDS = ("figure", "say", "answer")


def verify(answer: Any, index: Any, whatif: Any = None,
           fields: tuple[str, ...] = ALL_FIELDS) -> Verdict:
    """Check a card against the cells and pages it cites. No model call, no network.

    `answer` is an `Answer` or its wire dict, `index` a built `presage.ingest.Index`, and
    `whatif` the brain's last what-if result, whose outputs are the only evidence a
    recalculated figure can have: by definition it is in no cell. `fields` narrows what
    counts as a claim; `SPOKEN` checks only the two the consultant reads off the screen.
    """
    t0 = time.perf_counter()
    try:
        v = _verify(answer, index, whatif, fields)
    except Exception as e:  # noqa: BLE001  a guard must never be the thing that drops a card
        log.debug("verify failed: %s", e, exc_info=True)
        v = Verdict(True, "unverified", [], [], f"check failed ({type(e).__name__})")
    v.ms = (time.perf_counter() - t0) * 1000.0
    return v


def _verify(answer: Any, index: Any, whatif: Any, fields: tuple[str, ...]) -> Verdict:
    from_ = str(_get(answer, "answered_from") or "files").lower()
    claims: list[Num] = []
    seen: set[tuple[float, str]] = set()
    for name in fields:
        for n in numbers_in(str(_get(answer, name) or "")):
            key = (round(n.value, 6), n.kind)
            if key not in seen:
                seen.add(key)
                claims.append(n)

    if from_ == "web":
        return Verdict(True, "web", [], [], "web answer: the citations are pages, not cells",
                       checked=len(claims))
    if from_ == "model":
        return Verdict(True, "unverified", [], [],
                       "model advisory: nothing in the files to check it against",
                       checked=len(claims))
    try:
        cache = _cache(index)
    except Exception as e:  # noqa: BLE001
        log.debug("verify: index unavailable: %s", e)
        return Verdict(True, "unverified", [], [], f"index unavailable ({type(e).__name__})")

    # ---- 1. resolve the citations
    cites = [_parse_cite(f, loc, cache) for f, loc in _cite_pairs(answer)]
    pool: list[tuple[float, str, str]] = []
    problems: list[str] = []
    for c in cites:
        if c.kind != "unresolvable":
            try:
                pool += _resolve(index, c, cache)
            except Exception as e:  # noqa: BLE001
                log.debug("verify: resolve failed for %s: %s", c.text(), e)
                c.kind = "unresolvable"
        if c.problem:
            problems.append(c.problem)
    got: set[tuple[float, str]] = set()
    pool = [p for p in pool if not ((p[0], p[1]) in got or got.add((p[0], p[1])))][:MAX_POOL]
    decisive = bool(cites) and all(c.kind != "unresolvable" for c in cites) and bool(pool)

    # ---- 2. what a card may lean on that is not a cell
    extra: list[tuple[float, str, str]] = []
    for n in numbers_in(str(_get(answer, "question") or "")):
        extra.append((n.value, "asked", "derived"))
    if isinstance(whatif, dict):
        for row in list(whatif.get("outputs") or []) + list(whatif.get("changes") or []):
            for k in ("now", "was"):
                if isinstance(row.get(k), (int, float)) and not isinstance(row.get(k), bool):
                    extra.append((float(row[k]),
                                  f"what-if {row.get('label') or row.get('cell')}", "derived"))
    # `working` is the card's own arithmetic, so it is evidence only once it touches a cell
    # the card cites: one grounded input anchors the line and the rest of its numbers, the
    # stated result included, count as derived. An unanchored working line counts for
    # nothing, which is what stops a card justifying an invented figure with invented
    # arithmetic. What it cannot catch is a wrong operation between real inputs.
    wnums = numbers_in(str(_get(answer, "working") or ""))
    anchor = next((n for n in wnums for val, _s, _l in pool if _agrees(n, val)), None)
    if anchor is not None:
        extra += [(n.value, f"working ({anchor.text} is cited)", "derived") for n in wnums]
    # A recalculated figure is in no cell by construction: "EBITDA goes from 6,772 to 3,672
    # when Revenue (Q1) moves to 9,300" is a pycel run, and 3,672 is nowhere in the workbook.
    # An anchored working line still feeds the pool, but it must not license a rejection, so
    # a derived card is only decisive when the caller hands over the what-if result it came
    # from. Refusing to decide is the honest verdict; deciding would be a guess.
    if from_ == "derived" and not isinstance(whatif, dict):
        decisive = False

    # ---- 3. every number on the card has to land somewhere
    everything = pool + extra
    files = [c.file for c in cites if c.file]
    resolved: list[dict] = []
    levels: list[str] = []
    unmatched: list[Num] = []
    loose_only: list[str] = []
    tried = 0
    for n in claims:
        win = _window(n)
        hit = next(((src, val, lv) for val, src, lv in everything if _in(win, val)), None)
        if hit is None and everything and tried < MAX_DERIVED_CLAIMS:
            tried += 1
            wide = _window(n, REL_DERIVED, derived=True)
            hit = next(((src, val, "derived") for val, src in _derived(everything)
                        if _in(wide, val)), None)
        if hit is None:                       # second tier: anywhere in a file it cites
            for f in files:
                hit = next(((src, val, lv) for val, src, lv in _file_wide(index, cache, f)
                            if _in(win, val)), None)
                if hit:
                    loose_only.append(f"{n.text} is at {hit[0]}, which the card did not cite")
                    break
        if hit is None and n.kind == "pct" and pool:
            wide = _window(n, REL_DERIVED, derived=True)
            for f in files:
                hit = next(((f"{src} of {tot}", a / t * 100.0, "derived")
                            for a, src, _l in pool[:MAX_PAIRS]
                            for t, tot in _file_totals(index, cache, f) if t
                            and _in(wide, a / t * 100.0)), None)
                if hit:
                    loose_only.append(f"{n.text} is {hit[0]}, a total the card did not cite")
                    break
        if hit is None:
            unmatched.append(n)
            continue
        src, val, lv = hit
        resolved.append({"citation": src, "value": val, "claim": n.text, "how": lv})
        levels.append(lv)

    # ---- 4. the verdict, one line of reason
    strict = not loose_only and not unmatched
    if unmatched and decisive:
        where = "; ".join(c.text() for c in cites[:2])
        problems.append(f"{unmatched[0].text} is in nothing the card cites ({where})")
    if problems:
        return Verdict(False, "unverified", problems, resolved, problems[0],
                       checked=len(claims), matched=len(resolved), strict=False,
                       loose_only=loose_only)
    if not claims:
        return Verdict(True, "unverified", [], resolved, "no figure to check")
    if unmatched:
        return Verdict(True, "unverified", [], resolved,
                       f"{len(resolved)} of {len(claims)} numbers matched; "
                       f"{unmatched[0].text} has no citation that resolves",
                       checked=len(claims), matched=len(resolved), strict=False,
                       loose_only=loose_only)
    level = _weakest(levels)
    kind = "cited cells" if any(c.kind in ("cells", "rowlabel", "rownum") for c in cites) else "cited pages"
    reason = f"{level}: {len(resolved)} of {len(claims)} numbers matched the {kind}"
    if loose_only:
        reason = f"{level}: every number is in the cited files, but {loose_only[0]}"
    return Verdict(True, level, [], resolved, reason, checked=len(claims),
                   matched=len(resolved), strict=strict, loose_only=loose_only)


def _weakest(levels: list[str]) -> str:
    return max(levels, key=lambda l: _RANK.get(l, 9)) if levels else "unverified"


__all__ = ["Verdict", "verify", "numbers_in", "Num", "LEVELS", "SPOKEN", "ALL_FIELDS",
           "REL_DIRECT", "REL_DERIVED"]
