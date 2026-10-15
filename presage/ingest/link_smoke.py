"""Smoke test for the deck -> model cell linker (presage/ingest/link.py, E16).

    uv run python -m presage.ingest.link_smoke [folder]        # default sample_context

Three groups of cases, all offline and all under a second:

  A. parsing      one number at a time: the unit, the precision, the things that
                  look like numbers and are not ("Ask 1", "9/17/2026", "$1M+").
  B. extraction   a pptx read from its XML: bullets, table row/column labels, a
                  chart's datasheet cache, the same chart with the cache stripped
                  so the embedded workbook has to answer, and a think-cell-style
                  base64+deflate blob in a `ppt/tags` part. The last three are
                  built here rather than committed, so there is no binary fixture.
  C. resolution   the real sample folder: the figures that must land on a named
                  cell, the figures that must NOT resolve because the model does
                  not hold them, and the two read paths (`links_for`,
                  `cell_sources`).

Exit code is the number of failed cases.
"""

from __future__ import annotations

import base64
import re
import shutil
import sys
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

from presage.ingest import Index, link

PASS, FAIL = [], []


def check(name: str, got, want=None, ok: bool | None = None) -> None:
    good = ok if ok is not None else (got == want)
    (PASS if good else FAIL).append(name)
    mark = "ok  " if good else "FAIL"
    extra = "" if good else f"   got {got!r}, want {want!r}"
    print(f"  {mark} {name}{extra}")


# --------------------------------------------------------------------------- #
def case_parsing() -> None:
    print("\nA. parsing")

    def one(text: str):
        figs = link.parse_figures(text, text, "f", 1)
        return figs[0] if figs else None

    f = one("FY26 plan holds opex at $34,500k, 50% of revenue.")
    check("trailing period does not eat the suffix", f.value_text, "$34,500k")
    check("  ... keeps the k", (f.unit, f.scale), ("$k", 1000.0))
    check("  ... 34,500 shows 3 significant digits", (f.sig, f.decimals), (3, 0))

    f = one("the big model requires total capital of $16.4M to build out.")
    check("$16.4M value and scale", (f.value_num, f.scale), (16.4, 1000000.0))
    probes = dict((round(v, 4), r) for v, _t, r in f.probes())
    check("  ... probes the $k sheet", probes.get(16400.0), "unit")
    check("  ... and the raw 16.4", probes.get(16.4), "unit")
    tol = dict((round(v, 4), t) for v, t, _r in f.probes())[13200000.0]
    check("  ... tolerance is half the last shown digit", tol, 50000.0)

    f = one("Enterprise churn ran at 13.6 percent.")
    check("percent becomes a fraction probe", round(f.probes()[0][0], 5), 0.136)
    check("  ... within 0.0005", round(f.probes()[0][1], 6), 0.0005)

    check("significant digits: 24,000", link.sig_digits("24,000"), 2)
    check("significant digits: 30,184", link.sig_digits("30,184"), 5)
    check("significant digits: 0.85", link.sig_digits("0.85"), 2)

    check("a bare 1 is not a figure", one("Ask 1: approve the envelope.").model_like, False)
    check("a bare 44 is a figure", one("Enterprise is 44 logos.").model_like, True)
    check("a year is not a figure", one("Based on the model dated 2026.").model_like, False)
    check("a date is skipped", one("Based on Plan Model V2.1 (9/17/2026)."), None)
    check("a threshold is skipped",
          one("the median platform fee on $1M+ accounts"), None)
    check("a version is skipped", link.parse_figures("Plan Model V2.1", "x", "f", 1), [])

    figs = link.parse_figures("Revenue of $54,880k, up 23% year over year", "lbl", "f", 2)
    check("two numbers in one bullet", [g.value_text for g in figs], ["$54,880k", "23%"])


# --------------------------------------------------------------------------- #
def _chart_deck(path: Path) -> None:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "FY26 plan by region"
    data = CategoryChartData()
    data.categories = ["North America", "EMEA", "APAC", "LATAM"]
    data.add_series("FY26 plan ($k)", (37675, 20550, 7535, 2740))
    slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(2),
                           Inches(6), Inches(4), data)
    prs.save(str(path))


def case_pptx(tmp: Path) -> None:
    print("\nB. pptx extraction")
    deck = tmp / "chartdeck.pptx"
    _chart_deck(deck)

    figs = link.figures_from_pptx(deck, "chartdeck.pptx")
    chart = [f for f in figs if f.kind == "chart"]
    check("chart datasheet cache is read", len(chart), 4)
    check("  ... labelled title | series | category",
          chart[0].label, "FY26 plan by region | FY26 plan ($k) | North America")
    check("  ... values are the datasheet's",
          sorted(f.value_num for f in chart), [2740.0, 7535.0, 20550.0, 37675.0])

    # the same deck with the cache stripped: the embedded workbook has to answer
    nocache = tmp / "nocache.pptx"
    zin = zipfile.ZipFile(deck)
    with zipfile.ZipFile(nocache, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "ppt/charts/chart1.xml":
                data = re.sub(rb"<c:pt idx=\"\d+\"><c:v>[^<]*</c:v></c:pt>", b"", data)
            zo.writestr(item, data)
    zin.close()
    got = [f for f in link.figures_from_pptx(nocache, "nocache.pptx") if f.kind == "chart"]
    check("empty cache falls back to ppt/embeddings",
          sorted(f.value_num for f in got), [2740.0, 7535.0, 20550.0, 37675.0])

    # a think-cell style datasheet: base64 over deflate, inside a ppt/tags part
    tc = tmp / "tcdeck.pptx"
    payload = ('{"chart":"Plan capital","rows":[["Platform build",5019000],'
               '["Team and overhead",827033.78],["Total ask",13189248.24]]}')
    blob = base64.b64encode(zlib.compress(payload.encode())).decode()
    tag = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<p:tagLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
           f'<p:tag name="THINK-CELL-DATA" val="{blob}"/></p:tagLst>')
    zin = zipfile.ZipFile(deck)
    with zipfile.ZipFile(tc, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "ppt/slides/_rels/slide1.xml.rels":
                data = data.replace(b"</Relationships>", (
                    b'<Relationship Id="rIdTC" Type="http://schemas.openxmlformats.org'
                    b'/officeDocument/2006/relationships/tags" '
                    b'Target="../tags/tag1.xml"/></Relationships>'))
            zo.writestr(item, data)
        zo.writestr("ppt/tags/tag1.xml", tag)
    zin.close()
    cache = [f for f in link.figures_from_pptx(tc, "tcdeck.pptx") if f.kind == "tc-cache"]
    check("think-cell blob is decoded",
          sorted(f.value_num for f in cache), [827033.78, 5019000.0, 13189248.24])
    check("  ... each number keeps the words before it",
          "Platform build" in cache[0].label, True)
    check("a deck with no tags yields no tc-cache rows",
          [f for f in figs if f.kind == "tc-cache"], [])


def case_sample_deck(folder: Path) -> None:
    print("\nB2. the sample deck")
    deck = folder / "Strategy_Deck.pptx"
    if not deck.exists():
        check("sample deck present", str(deck), ok=False)
        return
    figs = link.figures_in(deck, "Strategy_Deck.pptx")
    by = {(f.loc, f.value_text): f for f in figs}
    loc4 = next(k[0] for k in by if k[0].startswith("slide 4"))
    check("loc matches the chunker's citation", loc4, "slide 4 'FY26 plan by region'")
    f = by[(loc4, "30,184")]
    check("a table cell carries row label and column header",
          f.label, "FY26 plan by region | North America | FY25 revenue ($k)")
    check("  ... and is tagged as a table figure", f.kind, "table")
    check("speaker notes are read too",
          any(f.kind == "notes" for f in figs), True)


# --------------------------------------------------------------------------- #
WANT = [
    # (file, loc head, figure, cell)
    ("Strategy_Deck.pptx", "slide 2", "$54,880k", "P&L!F6"),
    ("Strategy_Deck.pptx", "slide 2", "$35,672k", "P&L!F8"),
    ("Strategy_Deck.pptx", "slide 4", "30,184", "Regions!D6"),
    ("Strategy_Deck.pptx", "slide 4", "37,675", "Regions!E6"),
    ("Strategy_Deck.pptx", "slide 4", "2,196", "Regions!D9"),
    ("Strategy_Deck.pptx", "slide 5", "152", "Headcount!G9"),
    ("Strategy_Deck.pptx", "slide 6", "$34.5m", "P&L!G13"),       # $m against a $k sheet
    ("Board_Memo.docx", "section Quarterly", "1,773", "P&L!D14"),
    ("Board_Memo.docx", "section Regional", "96", "Regions!F6"),
    ("notes.md", "section Open questions", "$34,500k", "P&L!G13"),
]
# the model does not hold these, so a link would be an invented citation
WANT_NONE = [
    ("Strategy_Deck.pptx", "slide 2", "12.3%"),    # a formula with no cached value
    ("Strategy_Deck.pptx", "slide 4", "25%"),      # the deck computed it
    ("Strategy_Deck.pptx", "slide 3", "118%"),     # NRR is nowhere in the workbook
    ("Strategy_Deck.pptx", "slide 6", "1"),        # "Ask 1"
    ("Market_Report.pdf", "page 1", "14.2bn"),     # another company's market
]


def case_resolution(folder: Path) -> None:
    print("\nC. resolution on the sample folder")
    ix = Index()
    ix.build(folder)
    t0 = time.perf_counter()
    stats = ix.link_build()
    first_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    stats = ix.link_build()
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"       {stats['figures']} figures ({stats['model_like']} model-like) -> "
          f"{stats['links']} links over {stats['cells']} cell rows; "
          f"{first_ms:.0f} ms cold, {warm_ms:.0f} ms warm "
          f"(extract {stats['extract_ms']:.0f}, cells {stats['cells_ms']:.0f}, "
          f"resolve {stats['resolve_ms']:.0f})")

    rows = {}
    for r in ix.sql("SELECT * FROM links", limit=500):
        rows[(r["file"], r["loc"], r["value_text"])] = f"{r['cell_sheet']}!{r['cell_ref']}"

    def find(file, head, value):
        for (f, loc, v), cell in rows.items():
            if f == file and loc.startswith(head) and v == value:
                return cell
        return None

    for file, head, value, want in WANT:
        check(f"{head} {value} -> {want}", find(file, head, value), want)
    for file, head, value in WANT_NONE:
        check(f"{head} {value} stays unresolved", find(file, head, value), None)

    got = ix.links_for("Strategy_Deck.pptx", "slide 4")
    check("links_for takes the head of a loc", len(got), 8)
    check("  ... best first", got[0]["score"] >= got[-1]["score"], True)
    check("links_for with the full loc agrees",
          len(ix.links_for("Strategy_Deck.pptx", "slide 4 'FY26 plan by region'")), 8)

    src = ix.cell_sources("P&L", "G13")
    check("cell_sources finds every quote of a cell", len(src) >= 4, True)
    check("  ... 'Sheet!Ref' form works",
          len(ix.cell_sources("P&L!G13")), len(src))
    check("  ... and names the documents",
          sorted({r["file"] for r in src}),
          ["Board_Memo.docx", "Strategy_Deck.pptx", "notes.md"])
    check("cell_sources on an unquoted cell is empty", ix.cell_sources("P&L", "A1"), [])
    check("every stored link keeps the model's own value",
          all(r["cell_num"] is not None for r in ix.sql("SELECT * FROM links", limit=500)),
          True)


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    folder = Path(argv[1] if len(argv) > 1 else "sample_context").resolve()
    print(f"link_smoke: {folder}")
    case_parsing()
    tmp = Path(tempfile.mkdtemp(prefix="presage-link-"))
    try:
        case_pptx(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    case_sample_deck(folder)
    case_resolution(folder)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return len(FAIL)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
