"""The slide on screen against the model behind it.  (experiment E63)

Slide 4 says $9.6M. The model's `Subscriber Fee Model!B58` says 11,982,000. Somebody
re-ran the model after the deck was built and nobody re-exported the deck, and the
presenter is about to read the old number out loud.

E16's `links` table already knows which cell every printed figure came from, and it
was built when the figure and the cell still agreed: `resolve` only ever writes a row
whose value matched. So drift is not a search, it is a re-read. For the page on
screen, take its link rows, read each cited cell's value **now**, and compare it with
the figure **as the slide prints it**.

Three things keep this quiet, and each one costs coverage on purpose:

* **The slide's own precision is the tolerance.** `$9.6M` against 9,590,000 agrees,
  because a deck that prints one decimal is claiming nothing tighter than ±50,000.
  This is `presage/verify.py`'s comparison, unchanged and reused: the same windows,
  the same $-thousands and percent-vs-fraction scale hops, the same significant-digit
  guard that stops `$2.997M` landing on a bare 3.
* **The cell has to have moved.** A link was written because the figure and the cell
  agreed; if the cell still holds the value the link recorded, a disagreement now is
  this module and `link.py` reading a tolerance differently, not the model moving.
  Those are dropped as `unmoved` and counted, because they are a bug in here and not
  news for the presenter. (Measured: 0 of 74 links on the two folders.)
* **A cell with no value is not a changed cell.** A workbook saved by a tool that
  drops cached formula results makes every formula-derived cell read NULL at once;
  E16 measured 23 of 28 lost links as exactly that. `null-cell` rows are dropped.

Usage:

    rep = drift.check(index, {"file": "Strategy_Deck.pptx", "loc": "slide 4"})
    for d in rep.drifts:
        print(d.line)      # "Slide 4 says $9.6M; the model's Subscriber Fee Model!B58 is $11.98M"

CPU only: a handful of indexed SELECTs, no model call, no network. The brain runs it
off the transcript thread on a screen page change (`Brain.on_screen_change`) and shows
at most one card per page per session; `POST /api/drift` returns the whole list for
whatever is on screen now.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from presage import verify as _verify

log = logging.getLogger(__name__)

# How many link rows one page may be checked against. A slide carries eight figures at
# E16's cap and a dense memo page about twenty; 240 is the `links_for` read, filtered
# down to the page's own rows below.
MAX_ROWS = 240
PAGE_LINKS = 40           # rows kept after the prefix filter, best link first
SIG = 4                   # significant digits the model's value is printed to
EPS = 1e-9

# A figure whose number sat in more than this many other cells when it linked is not judged.
# Off, on evidence: E63 moved all 73 checkable cells on the two folders one at a time and got
# 72 real cards and 1 false one, and the false one is the single link E16 hand-checked as wrong
# (the big test memo's "$11.3M" resolved to `Funding Scenarios!D100`, "Cumulative capital", instead of
# `Summary Recommendation!B33`). `max_alt=4` removes it and costs 3 of the 72 real cards; a link
# score floor cannot do it at all, because the wrong link scores 0.60 and the right one 0.50.
# Set it if a folder ever turns out noisier than these two.
MAX_ALT: int | None = None

_REF = re.compile(r"^([A-Z]{1,3})(\d{1,7})$")
# The shape of the figure as the document printed it, so the model's value can be
# printed the same way: "$9.6M" -> "$11.98M", "13.6%" -> "15.23%", "30,184" -> "37,730".
_SHAPE = re.compile(
    r"^(?P<open>\()?\s*(?P<sign>[-−])?\s*(?P<cur>[$£€])?\s*(?P<num>[\d,]*\.?\d+)(?P<gap>\s*)"
    r"(?P<suf>%|percent|pct|bps|bn|billion|million|mm|mn|thousand|[kmbx]|ppt|pp)?",
    re.I)


# --------------------------------------------------------------------------- #
# One disagreement
# --------------------------------------------------------------------------- #
@dataclass
class Drift:
    """One printed figure that no longer matches the cell it came from."""
    file: str                 # the deck or memo
    loc: str                  # the chunk's own citation: "slide 4 'FY26 plan by region'"
    head: str                 # the page as the screen names it: "slide 4"
    kind: str                 # the figure's kind: title | body | table | chart | notes
    value_text: str           # exactly as the slide prints it: "$9.6M"
    cell_file: str
    cell_sheet: str
    cell_ref: str
    label: str                # the cell's own words, from the links table
    was: float                # the cell's value when the link was built
    now: float                # the cell's value now
    now_text: str             # `now`, printed in the slide's own units: "$11.98M"
    rel: float                # |now - was| / |was|, the size of the move
    score: float              # the link's score (E16: accept is 0.50)
    n_alt: int                # other cells that held the same number when it linked
    method: str

    @property
    def cell(self) -> str:
        return f"{self.cell_sheet}!{self.cell_ref}" if self.cell_sheet else self.cell_ref

    @property
    def line(self) -> str:
        """The card. 'Slide 4 says $9.6M; the model's Subscriber Fee Model!B58 is $11.98M'."""
        where = self.head[:1].upper() + self.head[1:] if self.head else self.file
        return (f"{where} says {self.value_text}; the model's {self.cell} "
                f"is {self.now_text}")

    @property
    def say(self) -> str:
        """Four to eight words the consultant can say instead of reading the slide."""
        return f"Model says {self.now_text}, not {self.value_text}"

    @property
    def working(self) -> str:
        """Everything behind the flag on one line: the cell, both values, the link."""
        bits = [f"{self.file} › {self.head} prints {self.value_text}",
                f"{self.cell_file} › {self.cell} ({self.label or 'no label'}) held "
                f"{_plain(self.was)} when the deck was linked and holds {_plain(self.now)} now"]
        tail = f"link score {self.score:.2f}"
        if self.n_alt:
            tail += f"; {self.n_alt} other cell{'s' if self.n_alt > 1 else ''} held the same number"
        bits.append(tail)
        return "; ".join(bits)

    def as_dict(self) -> dict:
        return {"file": self.file, "loc": self.loc, "head": self.head, "kind": self.kind,
                "value_text": self.value_text, "cell_file": self.cell_file,
                "cell": self.cell, "cell_sheet": self.cell_sheet, "cell_ref": self.cell_ref,
                "label": self.label, "was": self.was, "now": self.now,
                "now_text": self.now_text, "rel": round(self.rel, 6), "score": self.score,
                "n_alt": self.n_alt, "method": self.method,
                "line": self.line, "say": self.say, "working": self.working}


@dataclass
class Report:
    """What the check found on one page, and what it refused to judge."""
    file: str = ""
    head: str = ""
    links: int = 0                                  # rows on this page
    checked: int = 0                                # rows actually compared
    drifts: list = field(default_factory=list)      # list[Drift], strongest link first
    skipped: dict = field(default_factory=dict)     # reason -> count
    ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.drifts

    def as_dict(self) -> dict:
        return {"file": self.file, "head": self.head, "links": self.links,
                "checked": self.checked, "skipped": self.skipped,
                "ms": round(self.ms, 3),
                "drifts": [d.as_dict() for d in self.drifts]}


# --------------------------------------------------------------------------- #
# Printing a number the way the document printed its own
# --------------------------------------------------------------------------- #
def _plain(v: float) -> str:
    """A raw cell value, readable: 9586292.44 -> '9,586,292.44'."""
    s = f"{v:,.2f}"
    return s[:-3] if s.endswith(".00") else s


def _sig_fmt(v: float, sig: int = SIG, min_dec: int = 0) -> str:
    """`v` to `sig` significant digits, never fewer than `min_dec` decimals.

    11.982 -> '11.98', 15.2337 -> '15.23', 37730.4 -> '37,730', 0.15234 -> '0.1523'.
    """
    a = abs(v)
    exp = math.floor(math.log10(a)) if a > EPS else 0
    dec = max(min_dec, min(6, sig - 1 - exp))
    dec = max(0, dec)
    s = f"{v:,.{dec}f}"
    if "." in s and dec > min_dec:
        s = s.rstrip("0")
        keep = len(s.split(".")[1])
        if keep < min_dec:
            s += "0" * (min_dec - keep)
        s = s.rstrip(".")
    return s


def printed_num(value_text: str) -> float | None:
    """The numeral the document actually typed: "$9.6M" -> 9.6, "30,184" -> 30184."""
    m = _SHAPE.match(str(value_text or "").strip())
    if not m or not m.group("num"):
        return None
    try:
        return float(m.group("num").replace(",", ""))
    except ValueError:
        return None


def print_like(value_text: str, shown: float) -> str:
    """`shown`, in the units and shape the document used for its own figure.

    `shown` is already in the document's own printed units, which is what
    `shown_value` computes: the slide wrote "$9.6M" and the model's 11,982,000 arrives
    here as 11.982, to come out as "$11.98M".
    """
    m = _SHAPE.match(str(value_text or "").strip())
    cur, suf, num, gap = "", "", "", ""
    if m:
        cur = m.group("cur") or ""
        suf = m.group("suf") or ""
        num = m.group("num") or ""
        gap = m.group("gap") or ""      # "$34,500 thousand" keeps its space, "$9.6M" does not
    dec = len(num.split(".")[1]) if "." in num else 0
    body = _sig_fmt(abs(shown), min_dec=min(dec, 4))
    sign = "-" if shown < 0 else ""
    return f"{sign}{cur}{body}{gap if suf else ''}{suf}"


# --------------------------------------------------------------------------- #
# The cell, now
# --------------------------------------------------------------------------- #
def cell_now(index: Any, cell_file: str, sheet: str, ref: str) -> float | None:
    """This cell's value in the index as it stands, or None when it holds no number.

    A `row_sum` link cites a range (`D6:H6`), which is E16's synthetic total for a
    labelled row; it is re-summed rather than read, exactly as `link.load_cells` built
    it, so it moves when any cell in the row moves. That is also how it can move for no
    reason: a workbook saved without cached formula results drops a cell out of the sum
    and the total falls by whatever that cell held. So E16's NULL guard is applied to
    the whole row rather than to one cell: a row with an empty formula in it has no
    trustworthy total, and no total is better than a wrong one.
    """
    read = getattr(index, "_read", None)
    if read is None:
        return None
    if ":" in str(ref):
        a = _REF.match(str(ref).split(":")[0].strip())
        if not a:
            return None
        rows = read("SELECT value_num, is_formula, value_text FROM cells "
                    "WHERE file=? AND sheet=? AND row=?", (cell_file, sheet, int(a.group(2))))
        vals = [r for r in rows if r.get("value_num") is not None]
        if not vals:
            return None
        if any(r.get("is_formula") and r.get("value_num") is None and not r.get("value_text")
               for r in rows):
            return None
        return float(sum(r["value_num"] for r in vals))
    rows = read("SELECT value_num FROM cells WHERE file=? AND sheet=? AND cell_ref=?",
                (cell_file, sheet, ref))
    if not rows or rows[0].get("value_num") is None:
        return None
    return float(rows[0]["value_num"])


def scale_of(claim: Any, src: float) -> float | None:
    """The factor that turns a cell's units into the document's, or None if none does.

    `verify._window` inverts the comparison once per claim into bands of source values,
    one band per scale a workbook actually uses; this asks which band `src` fell in, so
    the model's new value can be printed at the scale the old one was read at. `src * f`
    is the claim's own value, so a sheet kept in $ thousands answers 1e3 and a percent
    held as the fraction 0.136 answers 100.
    """
    for lo, hi, f, need in _verify._window(claim):
        if lo <= src <= hi and (not need or need <= _verify._sig_of(src)):
            return f
    return None


def shown_value(value_text: str, claim: Any, was: float, now: float) -> float:
    """`now`, in the units the document printed its own figure in.

    Two hops. `scale_of` says what the workbook's units were worth against the claim
    (1e3 for a sheet in $ thousands, 100 for a fraction printed as a percent), and the
    ratio between the numeral on the page and the claim's value says what the page's
    own suffix was worth ("$9.6M" typed 9.6 for a claim of 9,600,000, so 1e-6).

    When `was` sits in no band at all the two hops collapse into one: apply the move the
    cell made to the numeral the page printed. That cannot happen for a link that E16
    accepted, and it is here so a retuned tolerance degrades into a sane number rather
    than a wrong scale.
    """
    shown = printed_num(value_text)
    f = scale_of(claim, was)
    if f is None or shown is None or abs(claim.value) <= EPS:
        return now * (shown / was) if (shown is not None and abs(was) > EPS) else now
    return now * f * (shown / claim.value)


# --------------------------------------------------------------------------- #
# The page's links
# --------------------------------------------------------------------------- #
def head_of(loc: str | None) -> str:
    """The page a citation belongs to. `slide 4 'FY26 plan by region'` -> `slide 4`.

    The screen poller only ever reports `slide N` or `page N` (`presage/screen.py`
    builds `loc` from the page index and nothing else), so that is the shape this
    normalizes. Any other citation is returned as it stands, because a caller that is
    not the poller can name a docx section (`section Asks`, which is the only page
    `Board_Memo.docx` has) or a chunk's full citation. Empty means no page, and no page
    means no check: "a number moved somewhere in this deck" is not a card.
    """
    s = str(loc or "").strip()
    if not s:
        return ""
    m = re.match(r"^(slide|page)\s+(\d+)", s, re.I)
    return f"{m.group(1).lower()} {m.group(2)}" if m else s


def page_links(index: Any, file: str, loc: str | None, limit: int = PAGE_LINKS) -> list[dict]:
    """Every link row on one page, best first, with the `slide 4` / `slide 40` trap closed.

    `Index.links_for` runs `loc = ? OR loc LIKE ? || '%'` and the screen's own `loc` is
    the bare head, so on a forty-slide deck `"slide 4"` also matches slides 40 to 49
    (E43a measured 11 rows where 8 belong). The head plus a space is what separates
    them, and it needs no chunk lookup: `slide 4 'FY26 plan'` starts with `slide 4 `,
    `slide 40 'Appendix'` does not.
    """
    head = head_of(loc)
    if not head:
        return []
    fn = getattr(index, "links_for", None)
    if fn is None:
        return []
    try:
        rows = fn(file, head, limit=MAX_ROWS) or []
    except Exception as e:  # noqa: BLE001
        log.debug("links_for failed: %s", e)
        return []
    out = []
    for lk in rows:
        if not isinstance(lk, dict) or lk.get("error") or not lk.get("cell_ref"):
            continue
        cloc = str(lk.get("loc") or "")
        if not (cloc == head or cloc.startswith(head + " ")):
            continue
        out.append(lk)
    out.sort(key=lambda lk: -float(lk.get("score") or 0.0))
    return out[:limit]


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #
def check(index: Any, state: Any, limit: int = PAGE_LINKS,
          max_alt: int | None = MAX_ALT) -> Report:
    """Compare every figure on the page on screen with the cell behind it.

    `state` is the screen poller's dict (`{"file": ..., "loc": "slide 4", ...}`) or a
    `(file, loc)` pair. A reading with no page is not checked: "somewhere in this deck
    a number moved" is not a card anybody can act on.
    """
    t0 = time.perf_counter()
    if isinstance(state, dict):
        file, loc = str(state.get("file") or ""), state.get("loc")
    else:
        file, loc = str(state[0] or ""), state[1]
    head = head_of(loc)
    rep = Report(file=file, head=head)
    if not file or not head:
        rep.skipped["no-page"] = 1
        rep.ms = (time.perf_counter() - t0) * 1000
        return rep

    rows = page_links(index, file, head, limit=limit)
    rep.links = len(rows)
    for lk in rows:
        text = str(lk.get("value_text") or "")
        nums = _verify.numbers_in(text)
        if not nums:
            # one significant digit ("$3M", "5%") carries no precision to disagree with:
            # `verify.numbers_in` drops it, and so does this.
            rep.skipped["no-claim"] = rep.skipped.get("no-claim", 0) + 1
            continue
        claim = nums[0]
        was = lk.get("cell_num")
        now = cell_now(index, lk.get("cell_file") or "", lk.get("cell_sheet") or "",
                       str(lk.get("cell_ref") or ""))
        if now is None:
            rep.skipped["null-cell"] = rep.skipped.get("null-cell", 0) + 1
            continue
        rep.checked += 1
        if _verify._agrees(claim, now):
            continue
        if was is None or abs(now - float(was)) <= max(EPS, abs(float(was)) * 1e-9):
            # the cell never moved, so the disagreement is a tolerance this module and
            # link.py read differently. That is a bug in here, not news for the room.
            rep.skipped["unmoved"] = rep.skipped.get("unmoved", 0) + 1
            continue
        if max_alt is not None and int(lk.get("n_alt") or 0) > max_alt:
            rep.skipped["ambiguous"] = rep.skipped.get("ambiguous", 0) + 1
            continue
        shown = shown_value(text, claim, float(was), now)
        rep.drifts.append(Drift(
            file=file, loc=str(lk.get("loc") or head), head=head,
            kind=str(lk.get("kind") or ""), value_text=text,
            cell_file=str(lk.get("cell_file") or ""),
            cell_sheet=str(lk.get("cell_sheet") or ""),
            cell_ref=str(lk.get("cell_ref") or ""), label=str(lk.get("label") or ""),
            was=float(was), now=now, now_text=print_like(text, shown),
            rel=abs(now - float(was)) / max(EPS, abs(float(was))),
            score=float(lk.get("score") or 0.0), n_alt=int(lk.get("n_alt") or 0),
            method=str(lk.get("method") or "")))
    # the most trustworthy link first, and the biggest move inside a tie: a card is
    # worth showing in proportion to how sure E16 was about the cell, not to the size
    # of the number.
    rep.drifts.sort(key=lambda d: (-d.score, -d.rel))
    rep.ms = (time.perf_counter() - t0) * 1000
    return rep


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """The printing, the scale hop and the guards, with no folder and no index."""
    from presage.verify import numbers_in

    fails: list[str] = []

    def eq(got, want, what):
        if got != want:
            fails.append(f"{what}: got {got!r}, want {want!r}")

    # print_like: the model's value in the slide's own shape
    eq(print_like("$9.6M", 11.982), "$11.98M", "print $M")
    eq(print_like("$9.6M", 12.0), "$12.0M", "print $M keeps the slide's decimal")
    eq(print_like("13.6%", 15.2337), "15.23%", "print %")
    eq(print_like("30,184", 37730.4), "37,730", "print bare")
    eq(print_like("$34,500 thousand", 36000), "$36,000 thousand", "print spelled scale")
    eq(print_like("1.35x", 1.6421), "1.642x", "print multiple")
    eq(print_like("$9.6M", -11.982), "-$11.98M", "print negative")

    # the tolerance is the slide's own precision
    c = numbers_in("$9.6M")[0]
    if not _verify._agrees(c, 9_590_000):
        fails.append("tolerance: $9.6M should agree with 9,590,000")
    if _verify._agrees(c, 11_982_000):
        fails.append("tolerance: $9.6M should not agree with 11,982,000")
    # the same figure against a sheet kept in $ thousands
    if not _verify._agrees(c, 9_590):
        fails.append("tolerance: $9.6M should agree with 9,590 in a $k sheet")
    eq(scale_of(c, 9_586_292.44), 1.0, "scale_of dollars")
    eq(scale_of(c, 9_590.0), 1e3, "scale_of a sheet kept in $ thousands")
    p = numbers_in("13.6%")[0]
    eq(scale_of(p, 0.136), 100.0, "scale_of fraction -> %")
    # the two hops together: the model's value in the slide's own units
    eq(round(shown_value("$9.6M", c, 9_586_292.44, 11_982_000.0), 4), 11.982, "shown $ -> $M")
    eq(round(shown_value("$9.6M", c, 9_590.0, 11_982.0), 4), 11.982, "shown $k -> $M")
    eq(round(shown_value("13.6%", p, 0.136, 0.152337), 4), 15.2337, "shown fraction -> %")

    # a single significant digit has nothing to disagree with
    if numbers_in("$3M"):
        fails.append("no-claim: $3M should not parse as a checkable claim")

    # the head, and the slide 4 / slide 40 trap
    eq(head_of("slide 4 'FY26 plan by region'"), "slide 4", "head_of slide")
    eq(head_of("page 2"), "page 2", "head_of page")
    eq(head_of("section Asks"), "section Asks", "head_of a docx section is its own page")
    eq(head_of(None), "", "head_of None")

    class _Ix:
        """Two link rows on slide 4 and one on slide 40, and a cells table that moved."""
        def links_for(self, file, loc=None, limit=60):
            return [
                {"file": file, "loc": "slide 4 'FY26 plan by region'", "kind": "table",
                 "value_text": "$9.6M", "value_num": 9.6, "unit": "$m",
                 "cell_file": "M.xlsx", "cell_sheet": "Subscriber Fee Model", "cell_ref": "B58",
                 "cell_num": 9_586_292.44, "label": "Subscriber Fee Model Total subscriber fees",
                 "score": 0.71, "method": "cell.unit", "n_alt": 4},
                {"file": file, "loc": "slide 4 'FY26 plan by region'", "kind": "table",
                 "value_text": "30,184", "value_num": 30184.0, "unit": "",
                 "cell_file": "M.xlsx", "cell_sheet": "Regions", "cell_ref": "D6",
                 "cell_num": 30184.0, "label": "Regions North America FY25 revenue",
                 "score": 0.74, "method": "cell", "n_alt": 0},
                {"file": file, "loc": "slide 40 'Appendix'", "kind": "table",
                 "value_text": "$1.20M", "value_num": 1.2, "unit": "$m",
                 "cell_file": "M.xlsx", "cell_sheet": "Retail Partner Fee Model", "cell_ref": "B93",
                 "cell_num": 1_197_500.0, "label": "Retail Partner Fee Model total",
                 "score": 0.80, "method": "cell.unit", "n_alt": 1},
            ][:limit]

        def _read(self, q, params=()):
            live = {("Subscriber Fee Model", "B58"): 11_982_000.0,   # moved
                    ("Regions", "D6"): 30_184.0,                 # unchanged
                    ("Retail Partner Fee Model", "B93"): 9_000_000.0}      # moved, but on slide 40
            v = live.get((params[1], params[2]))
            return [{"value_num": v}] if v is not None else []

    rep = check(_Ix(), {"file": "Strategy_Deck.pptx", "loc": "slide 4"})
    eq(rep.links, 2, "page_links keeps slide 4 off slide 40")
    eq(rep.checked, 2, "both slide-4 rows were compared")
    eq(len(rep.drifts), 1, "one figure drifted")
    if rep.drifts:
        d = rep.drifts[0]
        eq(d.line, "Slide 4 says $9.6M; the model's Subscriber Fee Model!B58 is $11.98M", "the card")
        eq(d.say, "Model says $11.98M, not $9.6M", "the say line")
        eq(d.n_alt, 4, "n_alt carried")
        if "4 other cells held the same number" not in d.working:
            fails.append(f"working: n_alt missing from {d.working!r}")

    # the guards: an unmoved cell and a NULL cell are not drift
    class _Unmoved(_Ix):
        def _read(self, q, params=()):
            return [{"value_num": 9_586_292.44}] if params[2] == "B58" else [{"value_num": 30184.0}]

    class _Null(_Ix):
        def _read(self, q, params=()):
            return [{"value_num": None}] if params[2] == "B58" else [{"value_num": 30184.0}]

    r2 = check(_Unmoved(), {"file": "D.pptx", "loc": "slide 4"})
    eq(len(r2.drifts), 0, "an unmoved cell is not drift")
    r3 = check(_Null(), {"file": "D.pptx", "loc": "slide 4"})
    eq(len(r3.drifts), 0, "a NULL cell is not drift")
    eq(r3.skipped.get("null-cell"), 1, "the NULL cell is counted")

    # a rounding is not drift: 9,590,000 is still "$9.6M"
    class _Rounded(_Ix):
        def _read(self, q, params=()):
            return [{"value_num": 9_590_000.0}] if params[2] == "B58" else [{"value_num": 30184.0}]

    eq(len(check(_Rounded(), {"file": "D.pptx", "loc": "slide 4"}).drifts), 0,
       "a rounding is not drift")

    # a workbook on screen has no page, so nothing is checked
    r4 = check(_Ix(), {"file": "M.xlsx", "loc": None, "sheet": "Regions"})
    eq(r4.skipped.get("no-page"), 1, "a sheet is not a page")

    n = 27
    for f in fails:
        print("FAIL", f)
    print(f"drift selftest: {n - len(fails)}/{n} checks passed")
    return 1 if fails else 0


def _cli() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="E63: the slide on screen against the model")
    ap.add_argument("folder", nargs="?", help="a context folder to check every page of")
    ap.add_argument("--page", help="one page only, e.g. 'slide 4'")
    ap.add_argument("--file", help="one document only")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.folder:
        return _selftest()

    from pathlib import Path

    from presage.ingest import Index

    ix = Index()
    ix.build(Path(a.folder))
    if not ix._read("SELECT 1 FROM links LIMIT 1"):
        print("no links table: run Index.link_build() on this folder first")
        return 2
    pages = sorted({(r["file"], head_of(r["loc"]))
                    for r in ix._read("SELECT file, loc FROM links")} - {("", "")})
    out = []
    for file, head in pages:
        if not head or (a.file and file != a.file) or (a.page and head != a.page):
            continue
        rep = check(ix, {"file": file, "loc": head})
        out.append(rep.as_dict())
        if not a.json:
            mark = "DRIFT" if rep.drifts else "  ok "
            print(f"{mark} {file} › {head}: {rep.links} links, {rep.checked} checked, "
                  f"{len(rep.drifts)} drift, {rep.ms:.2f} ms {rep.skipped or ''}")
            for d in rep.drifts:
                print(f"       {d.line}")
                print(f"       {d.working}")
    if a.json:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
