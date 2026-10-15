"""Figures in a deck or a memo -> the model cells they came from.  (experiment E16)

The presenter's deck is not the model. Slide 12 says "$9.6M"; the model says
9,586,292.44 in `Summary Scenarios!G57`. When the room asks "where does that come
from", the answer has to be in the slide's framing, not the workbook's, and it has
to name the cell.

This module builds that map once, at index time, with no model call:

  1. `figures_in(path, rel)` pulls every number out of a document **with its
     location and the label text around it** - slide title, bullet, table row
     label and column header, chart series and category, PDF line and the heading
     above it. pptx is read from the XML (`ppt/slides`, `ppt/charts/chart*.xml`,
     `ppt/embeddings/*.xlsx`, and the `ppt/tags` blobs think-cell hides its
     datasheets in), because python-pptx cannot see a chart's cached numbers.
  2. `resolve` matches each figure against the `cells` table by value, allowing
     for display rounding ("9.6M" against 9,586,292.44) and for the unit the
     slide chose ("13.6%" against 0.136, "$34.5m" against 34,500 in a $k sheet),
     plus row totals. Ties are broken by label similarity (model2vec, the same
     static embedding the chunk search uses), which is the signal that does the
     work: without it 49 links on the sample folder become 5 (E16 doc).
  3. `build_links` writes the `links` table. `Index.links_for(file, loc)` reads it
     back for one slide or page; `Index.cell_sources(sheet, cell)` reads it
     backwards, from a cell to every slide that quotes it.

Everything here is CPU-only and runs in well under a second on a deck-plus-model
folder, so it belongs in the fast build next to the chunkers.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
import re
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from . import docs as _docs
from . import store as _store

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tunables. Every one of these was set on sample_context and left alone for the big model.
# --------------------------------------------------------------------------- #
W_VALUE = 0.45           # how well the number fits the cell
W_LABEL = 0.40           # how well the slide's words fit the cell's row/header
W_LOCAL = 0.0            # the rest of this slide came off that sheet: measured as a
                         # no-op on both corpora and wrong once, so it is off. The pass
                         # is kept behind the constant; a chart-heavy deck may revive it.
W_SHEET = 0.06           # the text names the sheet ("Channel partners" -> Channel Fee Model)
TIE = 0.02               # locality only reorders candidates this close on the evidence
ACCEPT = 0.50            # below this a figure stays unresolved
STRONG = 0.62            # a link confident enough to vote in the locality pass
MAX_CAND = 240           # candidates per probe, nearest first
MIN_ABS = 1e-9           # a figure of 0 matches everything; never link it
ROUND_FLOOR = 0.5        # tolerance for an integer written without decimals

_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_C = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


# --------------------------------------------------------------------------- #
# 1. Numbers, with the unit the writer chose and the precision they showed
# --------------------------------------------------------------------------- #
_NUM = re.compile(
    r"(?<![\w.])"
    r"(?P<open>\()?\s*"
    r"(?P<sign>[-−])?\s*"
    r"(?P<cur>[$£€])?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<suf>%|percent|pct|bps|bn|billion|million|mm|mn|thousand|[kmbx]|ppt|pp)?"
    r"(?!\.\d)(?!\w)",
    re.I,
)
_DATE = re.compile(r"\d{1,4}[/-]\d{1,2}[/-]\d{2,4}")
_SCALE = {"bn": 1e9, "b": 1e9, "billion": 1e9,
          "million": 1e6, "mm": 1e6, "mn": 1e6, "m": 1e6,
          "thousand": 1e3, "k": 1e3}
_PCT = {"%", "percent", "pct"}
_BULLET = "•▪◦●○·-–—*​  \t"


@dataclass
class Figure:
    """One number as a document shows it, with everything around it that names it."""
    file: str
    unit_no: int                 # slide / page / section number
    loc: str = ""                # the citable location, identical to the chunk's
    kind: str = "body"           # title | body | table | chart | notes | tc-cache
    value_text: str = ""         # exactly as written: "$16.4M", "13.6%"
    value_num: float = 0.0       # as written, in the writer's units: 13.2
    unit: str = ""               # "$m", "%", "x", "" ...
    label: str = ""              # the words that name it
    scale: float = 1.0           # 1e6 for "M"
    decimals: int = 0            # digits shown after the point
    sig: int = 1                 # significant digits shown: "24,000" is 2, "30,184" is 5
    is_pct: bool = False
    model_like: bool = True      # False for "Ask 1", "Year 3", a bare page number

    def probes(self) -> list[tuple[float, float, str]]:
        """-> [(value to look for in a cell, tolerance, route)] in likelihood order.

        A slide says "$34.5m" and the sheet holds 34,500 because it is kept in $k.
        A slide says "13.6%" and the cell holds 0.136. Neither is a different
        number, only a different unit, so every reading gets a probe. `route` says
        how much guessing it took: "unit" converted a scale the writer stated,
        "guess" invented one, and the scoring charges for the difference.
        """
        v, dec = self.value_num, self.decimals
        half = 0.5 * (10.0 ** -dec) if dec else ROUND_FLOOR
        out: list[tuple[float, float, str]] = []

        def add(val: float, tol: float, route: str) -> None:
            if abs(val) > MIN_ABS:
                out.append((val, max(tol, abs(val) * 1e-9), route))

        if self.is_pct:
            add(v / 100.0, half / 100.0, "pct")      # 13.6% -> 0.136
            add(v, half, "pct-raw")                  # ... or the sheet holds 13.6
        elif self.unit in ("x", "bps", "ppt"):
            if self.unit == "bps":
                add(v / 10000.0, half / 10000.0, "as-written")
                add(v, half, "as-written")
            else:
                add(v, half, "as-written")
        else:
            base, stated = v * self.scale, self.scale > 1
            conv = "unit" if stated else "guess"
            add(base, half * self.scale, "as-written")
            # the same money in a sheet kept in thousands, or in millions
            add(base / 1e3, half * self.scale / 1e3, conv)
            add(base / 1e6, half * self.scale / 1e6, conv)
        seen, uniq = set(), []
        for val, tol, route in out:
            key = round(val, 12)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((val, tol, route))
        return uniq


def sig_digits(raw: str) -> int:
    """Digits the writer actually committed to. "24,000" is 2, "30,184" is 5."""
    s = raw.replace(",", "")
    if "." in s:
        return max(1, len(s.replace(".", "").lstrip("0")))
    return max(1, len(s.lstrip("0").rstrip("0")) or 1)


def parse_figures(text: str, label: str, file: str, unit_no: int,
                  kind: str = "body") -> list[Figure]:
    """Every number in one line of text, labelled by `label`."""
    out: list[Figure] = []
    if not text:
        return out
    blanked = _DATE.sub(lambda m: " " * len(m.group(0)), text)   # 9/17/2026 is not a figure
    for m in _NUM.finditer(blanked):
        raw_num = m.group("num")
        try:
            v = float(raw_num.replace(",", ""))
        except ValueError:
            continue
        if text[m.end():m.end() + 1] == "+":
            continue                  # "$1M+ portfolios" is a threshold, not a figure
        suf = (m.group("suf") or "").lower()
        cur = m.group("cur") or ""
        neg = bool(m.group("sign")) or (m.group("open") and ")" in text[m.end():m.end() + 2])
        dec = len(raw_num.split(".")[1]) if "." in raw_num else 0
        scale, is_pct, unit = 1.0, False, ""
        if suf in _PCT:
            is_pct, unit = True, "%"
        elif suf == "bps":
            unit = "bps"
        elif suf in ("ppt", "pp"):
            unit = "ppt"
        elif suf == "x":
            unit = "x"
        elif suf in _SCALE:
            scale = _SCALE[suf]
            unit = {1e9: "bn", 1e6: "m", 1e3: "k"}[scale]
            unit = ("$" + unit) if cur else unit
        elif cur:
            unit = "$"
        # "Ask 1", "Year 3", "Years 1-5", "1." in a numbered list: a bare single-digit
        # integer names a thing, not a value, and in a 5,000-cell model it will always
        # find a cell to agree with. Anything with a currency, a unit, a separator or a
        # decimal point is a figure, and so is a bare 10 or more ("44 logos").
        model_like = bool(cur or suf or "," in raw_num or "." in raw_num or abs(v) >= 10)
        if 1900 <= v <= 2100 and not suf and not cur and dec == 0:
            model_like = False                       # a year
        shown = m.group(0).strip()
        if shown.startswith("(") and not neg:        # "(0.85% of assets)" -> "0.85%"
            shown = shown[1:].strip()
        out.append(Figure(
            file=file, unit_no=unit_no, kind=kind,
            value_text=shown, value_num=-v if neg else v,
            unit=unit, label=label.strip()[:300], scale=scale, decimals=dec,
            sig=sig_digits(raw_num), is_pct=is_pct, model_like=model_like,
        ))
    return out


# --------------------------------------------------------------------------- #
# 2a. pptx: slides, tables, chart datasheets, embedded books, think-cell blobs
# --------------------------------------------------------------------------- #
def _rels(z: zipfile.ZipFile, part: str) -> dict[str, str]:
    """rId -> resolved part name, for one part's .rels file."""
    p = Path(part)
    rel_name = f"{p.parent.as_posix()}/_rels/{p.name}.rels"
    if rel_name not in z.namelist():
        return {}
    out: dict[str, str] = {}
    for rel in ET.fromstring(z.read(rel_name)):
        tgt = rel.get("Target", "")
        if rel.get("TargetMode") == "External" or not tgt:
            continue
        if tgt.startswith("/"):
            out[rel.get("Id", "")] = tgt.lstrip("/")
        else:
            out[rel.get("Id", "")] = (p.parent / tgt).resolve().as_posix().lstrip("/")
            # resolve() anchors at cwd on posix; redo it by hand to stay inside the zip
            parts: list[str] = []
            for seg in (p.parent.as_posix() + "/" + tgt).split("/"):
                if seg in ("", "."):
                    continue
                if seg == "..":
                    if parts:
                        parts.pop()
                else:
                    parts.append(seg)
            out[rel.get("Id", "")] = "/".join(parts)
    return out


def _slide_parts(z: zipfile.ZipFile) -> list[str]:
    """Slide part names in presentation order."""
    names = z.namelist()
    if "ppt/presentation.xml" not in names:
        return []
    rels = _rels(z, "ppt/presentation.xml")
    root = ET.fromstring(z.read("ppt/presentation.xml"))
    out = []
    for sld in root.findall(f"{_P}sldIdLst/{_P}sldId"):
        tgt = rels.get(sld.get(f"{_R}id", ""))
        if tgt and tgt in names:
            out.append(tgt)
    if out:
        return out
    return sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                  key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))


def _para_texts(el: ET.Element) -> list[str]:
    """One string per <a:p>, runs joined, empty paragraphs dropped."""
    out = []
    for p in el.iter(f"{_A}p"):
        txt = "".join(t.text or "" for t in p.iter(f"{_A}t")).strip()
        if txt:
            out.append(txt)
    return out


def _shape_title(root: ET.Element) -> str:
    for sp in root.iter(f"{_P}sp"):
        ph = sp.find(f".//{_P}nvSpPr/{_P}nvPr/{_P}ph")
        if ph is not None and (ph.get("type") or "") in ("title", "ctrTitle"):
            texts = _para_texts(sp)
            if texts:
                return texts[0]
    return ""


def _table_figures(tbl: ET.Element, file: str, n: int, title: str) -> list[Figure]:
    grid: list[list[str]] = []
    for tr in tbl.findall(f"{_A}tr"):
        row = ["".join(t.text or "" for t in tc.iter(f"{_A}t")).strip()
               for tc in tr.findall(f"{_A}tc")]
        grid.append(row)
    if not grid:
        return []
    head = grid[0]
    out: list[Figure] = []
    for r, row in enumerate(grid):
        stub = row[0] if row else ""
        for c, cell in enumerate(row):
            if not cell or (r == 0 and c == 0):
                continue
            col = head[c] if c < len(head) else ""
            bits = [title, stub if r else "", col if c else ""]
            label = " | ".join(b for b in bits if b)
            out.extend(parse_figures(cell, label or cell, file, n, "table"))
    return out


def _chart_figures(z: zipfile.ZipFile, part: str, file: str, n: int,
                   title: str) -> list[Figure]:
    """The numbers a chart carries in its own datasheet cache.

    A think-cell or native PowerPoint chart draws from `c:numCache`, which is the
    datasheet as it stood when the deck was last saved. That cache is the thing on
    the slide, so it is the thing to link; the embedded workbook is only read when
    the cache is missing.
    """
    try:
        root = ET.fromstring(z.read(part))
    except (KeyError, ET.ParseError):
        return []
    ctitle = " ".join(t.text or "" for t in root.iter(f"{_A}t") if t.text)
    ctitle = ctitle.strip()[:80]
    out: list[Figure] = []
    for ser in root.iter(f"{_C}ser"):
        name = ""
        tx = ser.find(f"{_C}tx")
        if tx is not None:
            name = " ".join(v.text or "" for v in tx.iter(f"{_C}v")).strip()
        cats: dict[int, str] = {}
        cat = ser.find(f"{_C}cat")
        if cat is not None:
            for pt in cat.iter(f"{_C}pt"):
                v = pt.find(f"{_C}v")
                cats[int(pt.get("idx", "0"))] = (v.text or "").strip() if v is not None else ""
        for tag in (f"{_C}val", f"{_C}yVal", f"{_C}bubbleSize"):
            holder = ser.find(tag)
            if holder is None:
                continue
            for pt in holder.iter(f"{_C}pt"):
                v = pt.find(f"{_C}v")
                if v is None or not (v.text or "").strip():
                    continue
                idx = int(pt.get("idx", "0"))
                bits = [title, ctitle, name, cats.get(idx, "")]
                label = " | ".join(b for b in bits if b)
                figs = parse_figures(v.text.strip(), label or ctitle, file, n, "chart")
                if not figs:                          # "1.2E7" and friends
                    try:
                        val = float(v.text)
                    except ValueError:
                        continue
                    figs = [Figure(file=file, unit_no=n, kind="chart",
                                   value_text=v.text.strip(), value_num=val,
                                   label=label or ctitle)]
                out.extend(figs)
    if not out:
        out.extend(_embedded_book_figures(z, part, file, n, title or ctitle))
    return out


def _embedded_book_figures(z: zipfile.ZipFile, chart_part: str, file: str, n: int,
                           title: str) -> list[Figure]:
    """The chart's own little workbook, when the cache is empty."""
    import io

    figs: list[Figure] = []
    for tgt in _rels(z, chart_part).values():
        if not tgt.lower().endswith((".xlsx", ".xlsm")):
            continue
        try:
            from openpyxl import load_workbook
            wb = load_workbook(io.BytesIO(z.read(tgt)), data_only=True, read_only=True)
        except Exception as e:                        # noqa: BLE001 - never fail a build
            log.debug("link: embedded book %s: %s", tgt, e)
            continue
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)][:200]
            head = [str(x) if x is not None else "" for x in (rows[0] if rows else [])]
            for r, row in enumerate(rows):
                stub = str(row[0]) if row and row[0] is not None else ""
                for c, val in enumerate(row):
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        continue
                    col = head[c] if c < len(head) else ""
                    bits = [title, stub if r else "", col if c else ""]
                    figs.append(Figure(file=file, unit_no=n, kind="chart",
                                       value_text=f"{val:g}", value_num=float(val),
                                       label=" | ".join(b for b in bits if b) or title))
        wb.close()
    return figs


_TC_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")
_TC_WORD = re.compile(r"[A-Za-z][A-Za-z0-9 &%/'\-]{2,40}")


def _decode_blob(val: str) -> str:
    """think-cell keeps its datasheet as a base64 (often deflated) blob in a p:tag.

    Return whatever printable text comes out, or "". No deck in the repo carries a
    think-cell payload, so this path is exercised by a built fixture in
    link_smoke.py rather than by a real deck; treat its numbers as advisory.
    """
    s = (val or "").strip()
    if len(s) < 48:
        return ""
    raw = b""
    try:
        raw = base64.b64decode(s + "=" * (-len(s) % 4), validate=False)
    except (binascii.Error, ValueError):
        return ""
    for attempt in (lambda b: zlib.decompress(b),
                    lambda b: zlib.decompress(b, -15),
                    lambda b: b):
        try:
            out = attempt(raw)
        except zlib.error:
            continue
        try:
            text = out.decode("utf-8", "ignore")
        except Exception:                             # noqa: BLE001
            continue
        printable = sum(c.isprintable() or c in "\n\t" for c in text)
        if text and printable / max(1, len(text)) > 0.6:
            return text
    return ""


def _thinkcell_figures(z: zipfile.ZipFile, root: ET.Element, part: str,
                       file: str, n: int, title: str) -> list[Figure]:
    """Numbers out of the tag blobs think-cell (and friends) attach to a slide."""
    rels = _rels(z, part)
    parts = {t for t in rels.values() if "/tags/" in t}
    for cust in root.iter(f"{_P}tags"):
        tgt = rels.get(cust.get(f"{_R}id", ""))
        if tgt:
            parts.add(tgt)
    out: list[Figure] = []
    for tp in sorted(parts):
        try:
            tag_root = ET.fromstring(z.read(tp))
        except (KeyError, ET.ParseError):
            continue
        for tag in tag_root.iter(f"{_P}tag"):
            text = _decode_blob(tag.get("val", "")) or ""
            if not text:
                continue
            name = (tag.get("name") or "").lower()
            if not ("think" in name or "tc" in name or "chart" in name or "data" in name):
                if not any(ch.isdigit() for ch in text):
                    continue
            for m in _TC_NUM.finditer(text):
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                window = text[max(0, m.start() - 90):m.start()]
                words = _TC_WORD.findall(window)
                label = " | ".join([title] + words[-2:]) if title else " | ".join(words[-2:])
                out.append(Figure(file=file, unit_no=n, kind="tc-cache",
                                  value_text=m.group(1), value_num=val,
                                  label=label.strip() or title,
                                  decimals=len(m.group(1).split(".")[1])
                                  if "." in m.group(1) else 0))
    return out


def figures_from_pptx(path: Path, rel: str) -> list[Figure]:
    out: list[Figure] = []
    with zipfile.ZipFile(path) as z:
        for n, part in enumerate(_slide_parts(z), start=1):
            try:
                root = ET.fromstring(z.read(part))
            except (KeyError, ET.ParseError):
                continue
            title = _shape_title(root)
            paras = []
            for sp in root.iter(f"{_P}sp"):
                paras.extend(_para_texts(sp))
            if not title and paras:
                title = paras[0][:80]
            for i, txt in enumerate(paras):
                kind = "title" if (title and txt == title and i == 0) else "body"
                label = txt if kind == "title" else f"{title} | {txt}" if title else txt
                out.extend(parse_figures(txt, label, rel, n, kind))
            for tbl in root.iter(f"{_A}tbl"):
                out.extend(_table_figures(tbl, rel, n, title))
            for tgt in _rels(z, part).values():
                if "/charts/chart" in tgt and tgt.endswith(".xml"):
                    out.extend(_chart_figures(z, tgt, rel, n, title))
            out.extend(_thinkcell_figures(z, root, part, rel, n, title))
    return out


# --------------------------------------------------------------------------- #
# 2b. pdf / docx / md: the text layer, line by line, under its heading
# --------------------------------------------------------------------------- #
_END = ".!?:;"


def _row_pairs(seg: str) -> list[tuple[str, str]] | None:
    """"Quarter: Q1; Revenue: 12,400; COGS: 4,340" -> [(header, value), ...].

    That shape is how the docx and pptx chunkers flatten a table row, and keeping
    the pairs apart is the difference between labelling 12,400 "Revenue" and
    labelling it with every other line item on the row.
    """
    if ";" not in seg or ":" not in seg:
        return None
    parts = [p.strip() for p in seg.split(";") if p.strip()]
    pairs = [tuple(p.split(":", 1)) for p in parts if ":" in p]
    if len(pairs) < 2 or len(pairs) < len(parts) - 1:
        return None
    return [(h.strip(), v.strip()) for h, v in pairs]


def figures_from_lines(text: str, file: str, unit_no: int, heading: str,
                       kind: str = "body") -> list[Figure]:
    """Label a number with its own sentence, plus the heading above it.

    A chunk's text keeps the document's line breaks where the reader saw them (a
    PDF bullet, a markdown list item) and joins sentences with a space where it
    did not, so both get split back out: the tightest true label for a figure is
    the sentence it sits in, not the 300-word chunk around it.
    """
    segs: list[str] = []
    for raw in (text or "").splitlines():
        ln = raw.strip(_BULLET).strip()
        if not ln:
            continue
        for s in _docs.split_sentences(ln):
            s = s.strip(_BULLET).strip()
            if s:
                segs.append(s)
    out: list[Figure] = []
    head = heading
    for i, seg in enumerate(segs):
        if not any(ch.isdigit() for ch in seg):
            if 3 <= len(seg) <= 70:                   # a short wordy line is a heading
                head = seg
            continue
        pairs = _row_pairs(seg)
        if pairs:
            stub = pairs[0][1] or pairs[0][0]
            for h, v in pairs:
                label = " | ".join(x for x in (head, stub, h) if x)
                out.extend(parse_figures(v, label or seg, file, unit_no, "table"))
            continue
        whole = seg
        nxt = segs[i + 1] if i + 1 < len(segs) else ""
        if nxt and seg[-1] not in _END and (nxt[:1].islower() or len(nxt) < 30):
            whole = f"{seg} {nxt}"
        label = f"{head} | {whole}" if head and head != whole else whole
        out.extend(parse_figures(seg, label, file, unit_no, kind))
    return out


# --------------------------------------------------------------------------- #
def figures_from_docx(path: Path, rel: str) -> list[Figure]:
    """Paragraphs and real table cells, sectioned exactly the way docs.read_docx is.

    Going through the chunk text instead would work for prose and ruin tables: the
    chunker flattens a row to "Line: Revenue; Q1: 12,400; ..." and the sentence
    splitter then cuts it at every colon, so 12,400 loses the word "Revenue".
    """
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    heading, parts = "", []
    sections: list[tuple[str, list]] = []

    def flush() -> None:
        if parts:
            sections.append((heading, list(parts)))

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            txt = p.text.strip()
            if not txt:
                continue
            style = (p.style.name or "") if p.style is not None else ""
            if style.startswith("Heading") or style == "Title":
                flush()
                heading, parts = txt, []
            else:
                parts.append(("p", txt))
        elif tag == "tbl":
            rows = [[c.text.strip() for c in r.cells] for r in Table(child, doc).rows]
            if rows:
                parts.append(("t", rows))
    flush()

    out: list[Figure] = []
    for i, (head, items) in enumerate(sections, start=1):
        for kindtag, payload in items:
            if kindtag == "p":
                for sent in _docs.split_sentences(payload) or [payload]:
                    label = f"{head} | {sent}" if head else sent
                    out.extend(parse_figures(sent, label, rel, i))
                continue
            grid = payload
            cols = grid[0]
            for r, row in enumerate(grid):
                stub = row[0] if row else ""
                for c, cell in enumerate(row):
                    if not cell or (r == 0):
                        continue
                    col = cols[c] if c < len(cols) else ""
                    label = " | ".join(x for x in (head, stub if c else "", col if c else "") if x)
                    out.extend(parse_figures(cell, label or cell, rel, i, "table"))
    return out


def figures_in(path: Path, rel: str) -> list[Figure]:
    """Every figure in one document, with `loc` set to the chunker's own citation."""
    res = _docs.read_document(path, rel, 0)
    loc_of: dict[int, str] = {}
    notes_of: dict[int, str] = {}
    for c in res.chunks:
        if c.kind == "pptx-notes":
            notes_of.setdefault(c.unit, c.loc)
            continue
        loc_of.setdefault(c.unit, c.loc)

    figs: list[Figure]
    ext = path.suffix.lower()
    if ext in _docs.PPTX_EXT:
        figs = figures_from_pptx(path, rel)
        for c in res.chunks:
            if c.kind == "pptx-notes":
                figs.extend(figures_from_lines(c.text, rel, c.unit, c.heading, "notes"))
    elif ext in _docs.DOCX_EXT:
        figs = figures_from_docx(path, rel)
    else:
        figs = []
        for c in res.chunks:
            figs.extend(figures_from_lines(c.text, rel, c.unit, c.heading))

    seen: set[tuple] = set()
    out: list[Figure] = []
    for f in figs:
        f.loc = (notes_of.get(f.unit_no, "") if f.kind == "notes"
                 else loc_of.get(f.unit_no, "")) or f"unit {f.unit_no}"
        key = (f.loc, f.kind, f.value_text, f.label[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# 3. The cells side
# --------------------------------------------------------------------------- #
@dataclass
class CellRow:
    file: str
    sheet: str
    ref: str
    value: float
    label: str
    is_formula: int = 0
    method: str = "cell"
    has_label: bool = False


_STOP = {"model", "sheet", "total", "year", "fee", "data", "summary", "detail", "tab"}


def _stems(text: str) -> set[str]:
    """Five-character stems of the words in a string: 'Channels' and 'Channel' agree."""
    return {w[:5] for w in re.split(r"[^a-z0-9]+", (text or "").lower())
            if len(w) >= 4 and w not in _STOP}


@dataclass
class CellBook:
    rows: list[CellRow] = field(default_factory=list)
    order: list[int] = field(default_factory=list)      # row indexes sorted by value
    values: list[float] = field(default_factory=list)   # the sorted values
    labels: list[str] = field(default_factory=list)     # distinct label strings
    label_of: list[int] = field(default_factory=list)   # row index -> labels index
    vecs: object = None                                 # (len(labels), d) float32
    _sheet_stems: dict = field(default_factory=dict)

    def sheet_stems(self, sheet: str) -> set[str]:
        got = self._sheet_stems.get(sheet)
        if got is None:
            got = _stems(sheet)
            self._sheet_stems[sheet] = got
        return got

    def near(self, value: float, tol: float, cap: int = MAX_CAND) -> tuple[list[int], int]:
        """-> (row indexes inside the tolerance, how many DISTINCT values were there).

        The count is of distinct values, not of cells. A model repeats its total
        ask down five sheets; those are one number wearing five cell references,
        not five competing answers, and only the label can say which reference to
        cite. Counting cells instead would punish exactly the figures a summary
        slide quotes most.
        """
        lo = bisect_left(self.values, value - tol)
        hi = bisect_right(self.values, value + tol)
        idx = self.order[lo:hi]
        crowd = len({round(self.values[i], 6) for i in range(lo, hi)})
        if len(idx) > cap:
            idx = sorted(idx, key=lambda i: abs(self.rows[i].value - value))[:cap]
        return idx, crowd


def load_cells(con, files: list[str] | None = None) -> CellBook:
    """Every numeric cell, plus one synthetic row total per labelled table row."""
    q = ("SELECT file, sheet, cell_ref, row, col, value_num, row_label, header, is_formula "
         "FROM cells WHERE value_num IS NOT NULL")
    params: tuple = ()
    if files:
        q += " AND file IN (%s)" % ",".join("?" * len(files))
        params = tuple(files)
    book = CellBook()
    sums: dict[tuple, list] = {}
    for r in con.execute(q, params):
        lab = " ".join(x for x in ((r["row_label"] or "").strip(),
                                   (r["header"] or "").strip()) if x)
        book.rows.append(CellRow(
            file=r["file"], sheet=r["sheet"] or "", ref=r["cell_ref"],
            value=float(r["value_num"]),
            label=f"{r['sheet'] or ''} {lab}".strip(),
            is_formula=int(r["is_formula"] or 0), has_label=bool(lab),
        ))
        if r["row_label"]:
            key = (r["file"], r["sheet"], r["row"])
            rec = sums.setdefault(key, [0.0, 0, None, None, (r["row_label"] or "").strip()])
            rec[0] += float(r["value_num"])
            rec[1] += 1
            rec[2] = r["cell_ref"] if rec[2] is None else rec[2]
            rec[3] = r["cell_ref"]
    for (file, sheet, _row), (total, n, first, last, lab) in sums.items():
        # a row of shares or multiples has no meaningful total; only money and counts do
        if n >= 2 and abs(total) >= 10:
            book.rows.append(CellRow(
                file=file, sheet=sheet or "", ref=f"{first}:{last}", value=total,
                label=f"{sheet or ''} {lab} total".strip(), method="row_sum",
                has_label=True))

    book.order = sorted(range(len(book.rows)), key=lambda i: book.rows[i].value)
    book.values = [book.rows[i].value for i in book.order]
    uniq: dict[str, int] = {}
    for row in book.rows:
        key = row.label.lower()[:120]
        if key not in uniq:
            uniq[key] = len(book.labels)
            book.labels.append(row.label[:120])
        book.label_of.append(uniq[key])
    return book


# --------------------------------------------------------------------------- #
# 4. Label similarity
# --------------------------------------------------------------------------- #
class Labels:
    """model2vec cosine over label strings, with a token-overlap fallback."""

    def __init__(self, enabled: bool = True):
        from . import search as _search
        self.np = None
        self.model = None
        if enabled and _search.Dense.available():
            try:
                import numpy as np
                self.model = _search.Dense().model()
                self.np = np
            except Exception as e:                    # noqa: BLE001
                log.debug("link: model2vec unavailable (%s); falling back to tokens", e)
                self.model = None

    def encode(self, texts: list[str]):
        if self.model is None or not texts:
            return None
        v = self.model.encode(texts).astype("float32")
        n = self.np.linalg.norm(v, axis=1, keepdims=True)
        return v / self.np.where(n == 0, 1, n)

    @staticmethod
    def tokens(s: str) -> set[str]:
        return {w for w in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(w) > 2}

    def overlap(self, a: str, b: str) -> float:
        ta, tb = self.tokens(a), self.tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / math.sqrt(len(ta) * len(tb))


# --------------------------------------------------------------------------- #
# 5. Resolution
# --------------------------------------------------------------------------- #
@dataclass
class Link:
    file: str
    loc: str
    kind: str
    value_text: str
    value_num: float
    unit: str
    cell_file: str
    cell_sheet: str
    cell_ref: str
    cell_num: float
    label: str
    score: float
    method: str
    n_alt: int = 0

    def cite(self) -> str:
        return (f"{self.loc} › {self.value_text} ← "
                f"{self.cell_sheet}!{self.cell_ref}" if self.cell_sheet
                else f"{self.loc} › {self.value_text} ← {self.cell_ref}")


_ROUTE_COST = {"as-written": 0.0, "pct": 0.0, "unit": 0.08, "guess": 0.28, "pct-raw": 0.30}


def _value_fit(cell_value: float, probe: float, tol: float, route: str,
               crowd: int = 1, sig: int = 5) -> float:
    """How much evidence this cell's value is, between 0 and 1.

    Four things weaken it. The writer stated a scale and we converted it ("unit",
    cheap) or we invented one ("guess", dear). A percentage was matched against a
    bare integer: "50% of revenue" is not the 50 in a headcount column. The value
    is so common in the workbook that hitting it says nothing, which is what stops
    "Ask 1" resolving. And the figure shows few significant digits: "31" agreeing
    with a cell is a coincidence until the words agree too, where "30,184" is not.
    """
    delta = abs(cell_value - probe)
    if delta > tol:
        return 0.0
    # Inside the rounding window is inside: a cell that rounds to "$5.85M" can sit
    # anywhere in it, so distance is only a mild tie-break, not most of the score.
    exact = 1.0 - (delta / tol if tol > 0 else 0.0)
    fit = 0.80 + 0.20 * exact
    fit -= _ROUTE_COST.get(route, 0.10)
    if crowd > 1:
        fit -= min(0.45, 0.07 * math.log2(crowd))
    fit *= 0.55 + 0.45 * min(1.0, (sig - 1) / 4.0)
    return max(0.0, fit)


def resolve(figs: list[Figure], book: CellBook, labels: Labels | None = None,
            accept: float = ACCEPT) -> list[Link]:
    """Score every figure against every value-compatible cell; keep the winner."""
    labels = labels or Labels()
    if not figs or not book.rows:
        return []
    fig_vecs = labels.encode([f.label[:120] or f.value_text for f in figs])
    if book.vecs is None and labels.model is not None:
        book.vecs = labels.encode(book.labels)

    np = labels.np

    def label_sim(fi: int, row_i: int) -> float:
        if fig_vecs is not None and book.vecs is not None:
            return float(max(0.0, fig_vecs[fi] @ book.vecs[book.label_of[row_i]]))
        return labels.overlap(figs[fi].label, book.rows[row_i].label)

    # pass 1: value + label only
    ranked: list[list[tuple[float, int, str, float]]] = []
    for fi, f in enumerate(figs):
        best: dict[int, tuple[float, str, float]] = {}
        if abs(f.value_num) <= MIN_ABS or not f.model_like:
            ranked.append([])
            continue
        for probe, tol, route in f.probes():
            near, crowd = book.near(probe, tol)
            loose = tol > 0.02 * abs(probe)
            for ri in near:
                row = book.rows[ri]
                if row.method == "row_sum" and loose:
                    continue      # nobody wrote a row total down; it needs a tight match
                fit = _value_fit(row.value, probe, tol, route, crowd, f.sig)
                if fit <= 0:
                    continue
                prev = best.get(ri)
                if prev is None or fit > prev[0]:
                    meth = row.method
                    if route in ("pct", "pct-raw"):
                        meth += ".pct"
                    elif route in ("unit", "guess"):
                        meth += "." + route
                    best[ri] = (fit, meth, row.value)
        ftok = _stems(f.label)
        scored = []
        for ri, (fit, meth, cval) in best.items():
            row = book.rows[ri]
            if row.method == "row_sum" and (f.is_pct or f.unit in ("x", "bps", "ppt")):
                continue                      # a share is never the sum of a row
            sim = label_sim(fi, ri)
            bonus = (0.03 if row.is_formula else 0.0) + (0.03 if row.has_label else 0.0)
            if ftok & book.sheet_stems(row.sheet):
                bonus += W_SHEET             # the text names the sheet: direct evidence
            scored.append((W_VALUE * fit + W_LABEL * sim + bonus, ri, meth, cval))
        scored.sort(key=lambda t: -t[0])
        ranked.append(scored[:12])

    # pass 2: sheet locality. A slide is about one part of the model; once a few of
    # its figures agree on a sheet, the ambiguous ones on that slide follow them.
    # It only ever reorders candidates that were already close (TIE), because a memo
    # page can cover four sheets at once and locality must not outvote the words.
    votes: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for f, scored in zip(figs, ranked) if W_LOCAL else ():
        if scored and scored[0][0] >= STRONG:
            row = book.rows[scored[0][1]]
            votes.setdefault((f.file, f.loc), {}).setdefault((row.file, row.sheet), 0)
            votes[(f.file, f.loc)][(row.file, row.sheet)] += 1

    out: list[Link] = []
    for f, scored in zip(figs, ranked):
        if not scored:
            continue
        local = votes.get((f.file, f.loc), {})
        total = sum(local.values()) or 1
        top = scored[0][0]
        final = []
        for base, ri, meth, cval in scored:
            row = book.rows[ri]
            share = local.get((row.file, row.sheet), 0) / total
            bump = W_LOCAL * share if base >= top - TIE else 0.0
            final.append((base + bump, ri, meth, cval))
        final.sort(key=lambda t: -t[0])
        score, ri, meth, cval = final[0]
        if score < accept:
            continue
        row = book.rows[ri]
        n_alt = sum(1 for s, r2, _m, _c in final[1:]
                    if s >= score - 0.05 and (book.rows[r2].sheet, book.rows[r2].ref)
                    != (row.sheet, row.ref))
        out.append(Link(
            file=f.file, loc=f.loc, kind=f.kind, value_text=f.value_text,
            value_num=f.value_num, unit=f.unit, cell_file=row.file,
            cell_sheet=row.sheet, cell_ref=row.ref, cell_num=cval,
            label=row.label, score=round(score, 4), method=meth, n_alt=n_alt,
        ))
    return out


# --------------------------------------------------------------------------- #
# 6. The build
# --------------------------------------------------------------------------- #
def document_files(con) -> list[tuple[str, str]]:
    return [(r["file"], r["kind"]) for r in con.execute(
        "SELECT file, kind FROM _documents ORDER BY file")]


def build_links(folder: Path, con_rw=None, accept: float = ACCEPT) -> dict:
    """Extract, resolve and store. -> {figures, model_like, links, files, ms}."""
    t0 = time.perf_counter()
    folder = Path(folder)
    own = con_rw is None
    con = con_rw or _store.connect_rw(folder)
    try:
        con.executescript(_store.LINKS_SCHEMA)
        t_cells = time.perf_counter()
        book = load_cells(con)
        labels = Labels()
        cells_ms = (time.perf_counter() - t_cells) * 1000
        figs: list[Figure] = []
        per_file: dict[str, dict] = {}
        for file, _kind in document_files(con):
            path = folder / file
            if not path.exists():
                continue
            t1 = time.perf_counter()
            try:
                got = figures_in(path, file)
            except Exception as e:                    # noqa: BLE001 - one bad deck, not a dead build
                log.warning("link: %s: %s", file, e)
                continue
            figs.extend(got)
            per_file[file] = {"figures": len(got),
                              "model_like": sum(1 for f in got if f.model_like),
                              "extract_ms": (time.perf_counter() - t1) * 1000}
        t_res = time.perf_counter()
        links = resolve(figs, book, labels, accept)
        resolve_ms = (time.perf_counter() - t_res) * 1000
        con.execute("DELETE FROM links")
        con.executemany(
            "INSERT INTO links VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(li.file, li.loc, li.kind, li.value_text, li.value_num, li.unit,
              li.cell_file, li.cell_sheet, li.cell_ref, li.cell_num, li.label,
              li.score, li.method, li.n_alt) for li in links])
        con.commit()
        for li in links:
            per_file.setdefault(li.file, {}).setdefault("links", 0)
            per_file[li.file]["links"] = per_file[li.file].get("links", 0) + 1
        return {"figures": len(figs),
                "model_like": sum(1 for f in figs if f.model_like),
                "links": len(links), "cells": len(book.rows),
                "files": per_file, "cells_ms": cells_ms, "resolve_ms": resolve_ms,
                "extract_ms": sum(d.get("extract_ms", 0.0) for d in per_file.values()),
                "ms": (time.perf_counter() - t0) * 1000}
    finally:
        if own:
            con.close()
