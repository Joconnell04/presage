"""Generate the sample_context/ folder: realistic, messy, internally consistent files.

    python -m presage.ingest.gen_samples sample_context

Every number below is chosen so cross-file questions have exactly one right answer
(regional revenue sums to P&L revenue, customer count sums to 200, etc.).
The answer key is written to ANSWER_KEY.md, which the indexer skips.
"""

from __future__ import annotations

import csv
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

# --------------------------------------------------------------------------- #
# The single source of truth for every number in the sample folder.
# $ thousands unless stated.
# --------------------------------------------------------------------------- #
QTRS = ["Q1", "Q2", "Q3", "Q4"]

PNL = {
    # line item -> (Q1, Q2, Q3, Q4, FY26 plan)   FY25 total is a formula
    "Revenue": (12400, 13150, 14020, 15310, 68500),
    "COGS": (4340, 4602, 4907, 5359, 23290),
    "Sales & marketing": (3100, 3250, 3400, 3600, 16000),
    "Research & development": (2480, 2560, 2640, 2720, 12400),
    "General & administrative": (1240, 1270, 1300, 1340, 6100),
}
GM = tuple(PNL["Revenue"][i] - PNL["COGS"][i] for i in range(5))          # 8060 ... 45210
OPEX = tuple(
    PNL["Sales & marketing"][i] + PNL["Research & development"][i] + PNL["General & administrative"][i]
    for i in range(5)
)                                                                         # 6820 ... 34500
EBITDA = tuple(GM[i] - OPEX[i] for i in range(5))                         # 1240 ... 10710
FY25 = {k: sum(v[:4]) for k, v in PNL.items()}
FY25["Gross margin"] = sum(GM[:4])
FY25["Total opex"] = sum(OPEX[:4])
FY25["EBITDA"] = sum(EBITDA[:4])

HEADCOUNT = {
    "Engineering": [48, 50, 52, 55, 57, 60],
    "Sales": [32, 34, 36, 38, 40, 42],
    "Marketing": [12, 12, 13, 14, 14, 15],
    "Customer Success": [18, 19, 20, 21, 22, 24],
    "G&A": [9, 9, 10, 10, 11, 11],
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
HC_TOTAL = [sum(v[i] for v in HEADCOUNT.values()) for i in range(6)]      # 119 ... 152

REGIONS = {
    # region -> (FY25 revenue, FY26 plan, customers)
    "North America": (30184, 37675, 96),
    "EMEA": (16464, 20550, 58),
    "APAC": (6036, 7535, 31),
    "LATAM": (2196, 2740, 15),
}
SEGMENTS = {
    # segment -> (FY25 revenue, logos)
    "Enterprise": (32928, 44),
    "Mid-market": (16464, 71),
    "SMB": (5488, 85),
}
CHURNED_PER_SEGMENT = {"Enterprise": 6, "Mid-market": 5, "SMB": 5}


# --------------------------------------------------------------------------- #
# Excel: openpyxl writes formulas with no cached value. Real workbooks saved by
# Excel carry one, so we inject <v> into the sheet XML after saving. A couple of
# cells are deliberately left with no cached value to exercise the NULL path.
# --------------------------------------------------------------------------- #
def inject_cached_values(path: Path, cache: dict[str, dict[str, float]], sheetnames: list[str]) -> None:
    tmp = path.with_suffix(".tmp.xlsx")
    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                data = zin.read(name)
                m = re.fullmatch(r"xl/worksheets/sheet(\d+)\.xml", name)
                if m:
                    idx = int(m.group(1)) - 1
                    sheet = sheetnames[idx]
                    if cache.get(sheet):
                        data = _patch_sheet(data, cache[sheet])
                zout.writestr(name, data)
    shutil.move(str(tmp), str(path))


def _patch_sheet(data: bytes, cells: dict[str, float]) -> bytes:
    ET.register_namespace("", MAIN_NS)
    root = ET.fromstring(data)
    for c in root.iter(f"{{{MAIN_NS}}}c"):
        ref = c.get("r")
        if ref not in cells:
            continue
        f = c.find(f"{{{MAIN_NS}}}f")
        if f is None:
            continue
        c.attrib.pop("t", None)          # numeric is the default type
        for old in c.findall(f"{{{MAIN_NS}}}v"):
            c.remove(old)
        v = ET.SubElement(c, f"{{{MAIN_NS}}}v")
        v.text = repr(float(cells[ref])) if cells[ref] % 1 else str(int(cells[ref]))
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def build_workbook(out: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    cache: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ P&L
    ws = wb.active
    ws.title = "P&L"
    cache["P&L"] = {}
    ws["A1"] = "Acme Corporation — Management accounts"
    ws.merge_cells("A1:G1")
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = "$ thousands, unaudited. Fiscal year ends 31 March."
    ws["A2"].font = Font(italic=True, size=9)
    # row 3 blank; two-row header band on rows 4-5, merged cells and all
    ws["B4"] = "FY25 actual"
    ws.merge_cells("B4:F4")
    ws["B4"].alignment = Alignment(horizontal="center")
    ws["G4"] = "FY26 plan"
    for col, label in zip("ABCDEFG", ["Line item", "Q1", "Q2", "Q3", "Q4", "Total", "Plan"]):
        ws[f"{col}5"] = label
        ws[f"{col}5"].font = Font(bold=True)

    def row(r: int, label: str, quarters, plan, total_formula: str, cached_total: float):
        ws[f"A{r}"] = label
        for col, val in zip("BCDE", quarters):
            ws[f"{col}{r}"] = val
        ws[f"F{r}"] = total_formula
        cache["P&L"][f"F{r}"] = cached_total
        ws[f"G{r}"] = plan

    row(6, "Revenue", PNL["Revenue"][:4], PNL["Revenue"][4], "=SUM(B6:E6)", FY25["Revenue"])
    row(7, "COGS", PNL["COGS"][:4], PNL["COGS"][4], "=SUM(B7:E7)", FY25["COGS"])

    ws["A8"] = "Gross margin"
    for i, col in enumerate("BCDE"):
        ws[f"{col}8"] = f"={col}6-{col}7"
        cache["P&L"][f"{col}8"] = GM[i]
    ws["F8"] = "=F6-F7"
    cache["P&L"]["F8"] = FY25["Gross margin"]
    ws["G8"] = "=G6-G7"
    cache["P&L"]["G8"] = GM[4]

    ws["A9"] = "Gross margin %"                      # deliberately NO cached values
    for col in "BCDEFG":
        ws[f"{col}9"] = f"={col}8/{col}6"

    row(10, "Sales & marketing", PNL["Sales & marketing"][:4], PNL["Sales & marketing"][4],
        "=SUM(B10:E10)", FY25["Sales & marketing"])
    row(11, "Research & development", PNL["Research & development"][:4], PNL["Research & development"][4],
        "=SUM(B11:E11)", FY25["Research & development"])
    row(12, "General & administrative", PNL["General & administrative"][:4], PNL["General & administrative"][4],
        "=SUM(B12:E12)", FY25["General & administrative"])

    ws["A13"] = "Total opex"                         # the totals row
    ws["A13"].font = Font(bold=True)
    for i, col in enumerate("BCDE"):
        ws[f"{col}13"] = f"=SUM({col}10:{col}12)"
        cache["P&L"][f"{col}13"] = OPEX[i]
    ws["F13"] = "=SUM(F10:F12)"
    cache["P&L"]["F13"] = FY25["Total opex"]
    ws["G13"] = "=SUM(G10:G12)"
    cache["P&L"]["G13"] = OPEX[4]

    ws["A14"] = "EBITDA"
    ws["A14"].font = Font(bold=True)
    for i, col in enumerate("BCDE"):
        ws[f"{col}14"] = f"={col}8-{col}13"
        cache["P&L"][f"{col}14"] = EBITDA[i]
    ws["F14"] = "=F8-F13"
    cache["P&L"]["F14"] = FY25["EBITDA"]
    ws["G14"] = "=G8-G13"
    cache["P&L"]["G14"] = EBITDA[4]

    ws["A15"] = "EBITDA margin %"                    # deliberately NO cached values
    for col in "BCDEFG":
        ws[f"{col}15"] = f"={col}14/{col}6"
    ws.column_dimensions["A"].width = 26

    # ------------------------------------------------------------ Headcount
    hc = wb.create_sheet("Headcount")
    cache["Headcount"] = {}
    hc["A1"] = "Headcount plan — FY26 H1 (FTE at month end)"
    hc["A1"].font = Font(bold=True, size=12)
    # row 2 blank
    hc["A3"] = "Department"
    for col, m in zip("BCDEFG", MONTHS):
        hc[f"{col}3"] = m
        hc[f"{col}3"].font = Font(bold=True)
    hc["A3"].font = Font(bold=True)
    r = 4
    for dept, vals in HEADCOUNT.items():
        hc[f"A{r}"] = dept
        for col, v in zip("BCDEFG", vals):
            hc[f"{col}{r}"] = v
        r += 1
    hc[f"A{r}"] = "Total"
    hc[f"A{r}"].font = Font(bold=True)
    for i, col in enumerate("BCDEFG"):
        hc[f"{col}{r}"] = f"=SUM({col}4:{col}{r-1})"
        cache["Headcount"][f"{col}{r}"] = HC_TOTAL[i]
    hc.column_dimensions["A"].width = 20

    # -------------------------------------------------------------- Regions
    rg = wb.create_sheet("Regions")
    cache["Regions"] = {}
    rg["C3"] = "Revenue split — supporting detail"
    rg["C3"].font = Font(bold=True)
    for col, label in zip("CDEF", ["Region", "FY25 revenue", "FY26 plan", "Customers"]):
        rg[f"{col}5"] = label
        rg[f"{col}5"].font = Font(bold=True)
    r = 6
    for region, (fy25, fy26, cust) in REGIONS.items():
        rg[f"C{r}"] = region
        rg[f"D{r}"] = fy25
        rg[f"E{r}"] = fy26
        rg[f"F{r}"] = cust
        r += 1
    # rows 10 and 11 left blank -> block splitter starts a second table
    for col, label in zip("CDE", ["Segment", "FY25 revenue", "Logos"]):
        rg[f"{col}12"] = label
        rg[f"{col}12"].font = Font(bold=True)
    r = 13
    for seg, (fy25, logos) in SEGMENTS.items():
        rg[f"C{r}"] = seg
        rg[f"D{r}"] = fy25
        rg[f"E{r}"] = logos
        r += 1
    rg.column_dimensions["C"].width = 18

    wb.save(out)
    inject_cached_values(out, cache, wb.sheetnames)


# --------------------------------------------------------------------------- #
def build_customers(out: Path) -> dict:
    rng = random.Random(20260318)
    first = ["Northwind", "Cobalt", "Ridgeline", "Halcyon", "Tessera", "Brightpath", "Vantage",
             "Ironwood", "Solstice", "Meridian", "Kestrel", "Anvil", "Lumen", "Thornbury",
             "Cedarwood", "Palisade", "Quarry", "Fairlane", "Orchard", "Beacon", "Granite",
             "Mainsail", "Lakeshore", "Foxglove", "Harbour"]
    second = ["Systems", "Logistics", "Health", "Capital", "Retail", "Foods", "Energy",
              "Analytics", "Manufacturing", "Media", "Partners", "Group", "Labs", "Holdings"]

    region_pool = [r for r, (_, _, n) in REGIONS.items() for _ in range(n)]
    rng.shuffle(region_pool)

    rows = []
    cid = 1000
    for seg, (target_k, n) in SEGMENTS.items():
        weights = [rng.uniform(0.6, 1.6) for _ in range(n)]
        scale = target_k * 1000 / sum(weights)
        arrs = [int(round(w * scale, -2)) for w in weights]
        arrs[-1] += target_k * 1000 - sum(arrs)          # absorb rounding into the last row
        churn_idx = set(rng.sample(range(n), CHURNED_PER_SEGMENT[seg]))
        for i in range(n):
            cid += 1
            rows.append({
                "customer_id": f"C{cid}",
                "customer_name": f"{rng.choice(first)} {rng.choice(second)}",
                "segment": seg,
                "region": region_pool.pop(),
                "arr_usd": arrs[i],
                "churned": 1 if i in churn_idx else 0,
                "renewal_month": rng.choice(MONTHS),
                "csat": round(rng.uniform(3.1, 4.9), 1),
            })
    rng.shuffle(rows)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    facts = {"total_arr": sum(r["arr_usd"] for r in rows),
             "churned_arr": sum(r["arr_usd"] for r in rows if r["churned"]),
             "by_segment": {}}
    for seg in SEGMENTS:
        sub = [r for r in rows if r["segment"] == seg]
        ch = [r for r in sub if r["churned"]]
        facts["by_segment"][seg] = {
            "n": len(sub), "arr": sum(r["arr_usd"] for r in sub),
            "churned_n": len(ch), "churned_arr": sum(r["arr_usd"] for r in ch),
            "churn_rate": round(100 * len(ch) / len(sub), 1),
        }
    return facts


# --------------------------------------------------------------------------- #
def build_deck(out: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    blank, title_layout, bullet_layout = prs.slide_layouts[6], prs.slide_layouts[0], prs.slide_layouts[1]

    s = prs.slides.add_slide(title_layout)
    s.shapes.title.text = "Acme FY26 Strategy Review"
    s.placeholders[1].text = "Board session, prepared by the strategy team"
    s.notes_slide.notes_text_frame.text = (
        "Opening. Keep this to ninety seconds. The board already has the pack; the job today is the "
        "FY26 plan gap and the two asks on slide six."
    )

    def bullets(title, items, notes):
        sl = prs.slides.add_slide(bullet_layout)
        sl.shapes.title.text = title
        tf = sl.placeholders[1].text_frame
        tf.text = items[0]
        for it in items[1:]:
            p = tf.add_paragraph()
            p.text = it
            p.level = 0
        sl.notes_slide.notes_text_frame.text = notes
        return sl

    bullets(
        "FY25 in review: growth held, margin did not",
        [f"Revenue of ${FY25['Revenue']:,}k, up 23% year over year",
         f"Gross margin of ${FY25['Gross margin']:,}k, 65% of revenue",
         f"EBITDA of ${FY25['EBITDA']:,}k, a 12.3% margin",
         f"Q4 was the strongest quarter at ${PNL['Revenue'][3]:,}k of revenue",
         "Opex grew faster than revenue in every quarter except Q4"],
        "The one number to land: EBITDA margin was 12.3 percent against a 15 percent commitment. "
        "Do not get drawn into the quarterly bridge here, that is the board memo.",
    )

    bullets(
        "Growth engine: enterprise land and expand",
        ["Enterprise is 44 logos but 60% of ARR",
         "Net revenue retention of 118% in enterprise, 94% in SMB",
         "Average enterprise ARR is roughly 12x average SMB ARR",
         "Churn is concentrated in enterprise accounts that never got past one use case",
         "FY26 plan assumes 14 net new enterprise logos"],
        "If asked about enterprise churn, the honest answer is that six enterprise accounts churned "
        "in FY25 against five in mid-market, off a much smaller base. It is getting worse, not better.",
    )

    sl = prs.slides.add_slide(blank)
    tb = sl.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(8.8), Inches(0.8))
    tb.text_frame.text = "FY26 plan by region"
    tb.text_frame.paragraphs[0].runs[0].font.size = Pt(28)
    tb.text_frame.paragraphs[0].runs[0].font.bold = True
    rows = len(REGIONS) + 1
    tbl = sl.shapes.add_table(rows, 4, Inches(0.6), Inches(1.4), Inches(8.8), Inches(0.4 * rows)).table
    for j, h in enumerate(["Region", "FY25 revenue ($k)", "FY26 plan ($k)", "Growth"]):
        tbl.cell(0, j).text = h
    for i, (region, (fy25, fy26, _)) in enumerate(REGIONS.items(), start=1):
        tbl.cell(i, 0).text = region
        tbl.cell(i, 1).text = f"{fy25:,}"
        tbl.cell(i, 2).text = f"{fy26:,}"
        tbl.cell(i, 3).text = f"{100 * (fy26 / fy25 - 1):.0f}%"
    sl.notes_slide.notes_text_frame.text = (
        "Every region is planned at the same 25 percent growth rate. That is a placeholder, not a "
        "plan, and the board will spot it. APAC cannot carry 25 percent without two more sellers."
    )

    bullets(
        "Cost discipline: hold opex at 50% of revenue",
        [f"FY25 opex was ${FY25['Total opex']:,}k, 53% of revenue",
         f"FY26 plan holds opex at ${OPEX[4]:,}k, 50% of revenue",
         "Headcount grows from 119 in January to 152 by June",
         "Engineering absorbs half of the net adds",
         "No change to the sales quota model in FY26"],
        "The headcount numbers come off the Headcount tab. If the board asks for the June total it "
        "is 152.",
    )

    bullets(
        "Risks and asks",
        ["Risk: enterprise churn is running ahead of mid-market",
         "Risk: APAC plan is not supported by a hiring plan",
         "Ask 1: approve the FY26 opex envelope of $34.5m",
         "Ask 2: approve two APAC sales hires in Q1",
         "Decision needed this session, not offline"],
        "Close on the two asks. Do not reopen the pricing debate.",
    )
    prs.save(out)


# --------------------------------------------------------------------------- #
def build_memo(out: Path) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading("Board memo: FY25 close and FY26 plan", level=0)
    p = doc.add_paragraph("Prepared for the Acme board. Confidential.")
    p.runs[0].font.size = Pt(9)

    doc.add_heading("Summary", level=1)
    doc.add_paragraph(
        f"FY25 closed with revenue of ${FY25['Revenue']:,} thousand and EBITDA of "
        f"${FY25['EBITDA']:,} thousand, a margin of 12.3 percent against a commitment of 15 percent. "
        "The shortfall is entirely an operating expense story: gross margin held at 65 percent in "
        "every quarter, while operating expenses grew 9 percent from Q1 to Q4."
    )
    doc.add_paragraph(
        "The FY26 plan carries revenue to $68,500 thousand, a 25 percent increase, and holds "
        "operating expenses to $34,500 thousand. That combination produces EBITDA of $10,710 "
        "thousand, a 15.6 percent margin, which clears the commitment for the first time."
    )

    doc.add_heading("Quarterly EBITDA bridge", level=1)
    doc.add_paragraph("All figures in $ thousands. Source: management accounts, P&L tab.")
    t = doc.add_table(rows=1, cols=5)
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for j, h in enumerate(["Line", "Q1", "Q2", "Q3", "Q4"]):
        hdr[j].text = h
    for label, vals in [("Revenue", PNL["Revenue"][:4]), ("Gross margin", GM[:4]),
                        ("Total opex", OPEX[:4]), ("EBITDA", EBITDA[:4])]:
        cells = t.add_row().cells
        cells[0].text = label
        for j, v in enumerate(vals, start=1):
            cells[j].text = f"{v:,}"

    doc.add_heading("Churn and retention", level=1)
    doc.add_paragraph(
        "Sixteen customers churned in FY25 out of two hundred. The headline rate of 8 percent is "
        "not the problem; the mix is. Enterprise churn ran at 13.6 percent against 7.0 percent in "
        "mid-market and 5.9 percent in SMB. Enterprise accounts are the ones we cannot replace "
        "inside a year, and every one of the six that left had a single production use case."
    )
    doc.add_paragraph(
        "Net revenue retention of 118 percent in enterprise masks this. Expansion inside the "
        "accounts that stay is covering the ones that leave, and that will not hold if the logo "
        "count keeps falling."
    )

    doc.add_heading("Regional detail", level=1)
    doc.add_paragraph(
        "North America contributes $30,184 thousand of FY25 revenue across 96 customers. EMEA "
        "contributes $16,464 thousand across 58. APAC is $6,036 thousand across 31 customers and "
        "is the only region without a dedicated sales leader. LATAM remains a retail partner motion at "
        "$2,196 thousand."
    )

    doc.add_heading("Asks", level=1)
    doc.add_paragraph("Approve the FY26 operating expense envelope of $34,500 thousand.", style="List Number")
    doc.add_paragraph("Approve two APAC sales hires in the first quarter.", style="List Number")
    doc.add_paragraph("Note the change in churn reporting to a segment view.", style="List Number")
    doc.save(out)


# --------------------------------------------------------------------------- #
def build_pdf(out: Path) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    st = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=LETTER, title="Vertical SaaS Market Report 2026")
    flow = []

    def page(title, paras):
        flow.append(Paragraph(title, st["Heading1"]))
        flow.append(Spacer(1, 10))
        for p in paras:
            flow.append(Paragraph(p, st["BodyText"]))
            flow.append(Spacer(1, 7))

    page("Vertical SaaS market report 2026 &mdash; sizing", [
        "The addressable market for vertical workflow software in the segments Acme serves was "
        "$14.2 billion in 2025 and is forecast to reach $21.6 billion by 2028, a compound annual "
        "growth rate of 15 percent.",
        "North America accounts for 54 percent of spend, EMEA 28 percent, APAC 13 percent and "
        "Latin America 5 percent. APAC is the fastest growing region at 22 percent a year, roughly "
        "half again the global rate, and it is the region where incumbent penetration is lowest.",
        "Growth is concentrated in the enterprise tier. Buyers with more than 2,000 employees "
        "represent 31 percent of logos but 64 percent of spend, and their contracts renew at "
        "materially higher rates than the mid-market.",
        "Median annual contract value across the sample was $61,000. The enterprise median was "
        "$248,000 and the small business median was $9,400.",
    ])
    flow.append(PageBreak())
    page("Competitive landscape", [
        "Four vendors hold 46 percent of the market between them. The remainder is fragmented "
        "across more than 120 regional and niche providers, most with fewer than fifty customers.",
        "The largest incumbent, Veritas Workflow, holds an estimated 19 percent share and competes "
        "on breadth rather than depth. Its renewal rate has fallen for three consecutive years as "
        "buyers unbundle.",
        "Two venture-backed challengers entered the enterprise tier in 2025 with aggressive "
        "discounting, typically 30 to 40 percent below list on a three year commitment. Win rates "
        "against them improve sharply when the evaluation includes a production pilot.",
        "Displacement cycles average eleven months in enterprise and four months in the mid-market. "
        "Vendors that land a second production use case inside the first year retain at roughly "
        "twice the rate of single use case deployments.",
    ])
    flow.append(PageBreak())
    page("Pricing benchmarks and retention", [
        "List pricing has converged on per-seat models with a platform fee. The median platform fee "
        "is $24,000 a year and the median per-seat price is $38 a month, before discount.",
        "Realized discount off list averages 22 percent overall and 34 percent in competitive "
        "enterprise deals. Multi-year commitments carry a further 8 point discount on average.",
        "Median net revenue retention across the panel was 104 percent. Upper quartile vendors "
        "reached 121 percent, and every one of them reported more than one production use case per "
        "account as their leading indicator.",
        "Gross retention tells a harsher story: the panel median was 88 percent, and vendors "
        "serving predominantly small business customers reported 79 percent. Logo churn in the "
        "enterprise tier is rare but expensive, with the median lost enterprise account "
        "representing 4 percent of a vendor's annual recurring revenue.",
    ])
    doc.build(flow)


# --------------------------------------------------------------------------- #
def build_notes(out: Path) -> None:
    out.write_text(
        """# Acme working notes

Running notes from the FY26 planning work. Not a deliverable.

## Open questions for the CFO

- The FY25 total column on the P&L tab is formula-driven. Confirm the workbook was
  recalculated before it went to the board; the gross margin percent and EBITDA margin
  percent rows came through blank in the copy we were sent.
- Is the 25 percent regional growth rate a plan or a placeholder? Every region carries the
  same rate, which never happens in a real plan.
- Headcount runs to 152 by June against an opex envelope of $34,500k. Does that envelope
  include the two APAC sellers or not?

## Churn

Enterprise churn is the thing nobody wants to say out loud. Six enterprise logos left in
FY25 against five in mid-market, off a base of 44 versus 71. That is 13.6 percent against
7.0 percent. The common factor in all six is a single production use case that never
expanded past its original team.

Net revenue retention of 118 percent in enterprise is real but it is masking the logo
losses. Expansion in the accounts that stay is covering the ones that leave.

## Pricing

Market report puts median realized discount at 22 percent, and 34 percent in competitive
enterprise deals. We are discounting at 29 percent blended, which is not obviously wrong,
but we do not track it by competitor.

## Things to prefetch on the call

Likely questions: Q3 revenue, FY26 plan EBITDA, headcount total by June, churned ARR,
enterprise churn rate, APAC revenue, opex as a percent of revenue.
"""
    , encoding="utf-8")


# --------------------------------------------------------------------------- #
def write_answer_key(out: Path, facts: dict) -> None:
    bs = facts["by_segment"]
    lines = [
        "# ANSWER_KEY",
        "",
        "Ground truth for the sample folder. The indexer skips any file whose name starts with",
        "`ANSWER_KEY` or `.`, so this file is never retrievable and cannot leak into an answer.",
        "All workbook figures are $ thousands unless the unit says otherwise.",
        "",
        "## Acme_Financials.xlsx › P&L",
        "",
        "Two title rows, a merged `FY25 actual` band on row 4 over a `Q1..Q4 / Total` header on",
        "row 5, data from row 6. Column F (`Total`) and the gross margin / total opex / EBITDA",
        "rows are formulas that carry cached values. Rows 9 and 15 (the two percent rows) are",
        "formulas with **no cached value** and must come back NULL.",
        "",
        "| Question | Answer | Citation |",
        "| --- | --- | --- |",
        f"| Q3 FY25 revenue | {PNL['Revenue'][2]:,} | Acme_Financials.xlsx › P&L › D6 |",
        f"| Q3 FY25 EBITDA | {EBITDA[2]:,} | Acme_Financials.xlsx › P&L › D14 |",
        f"| FY25 total revenue | {FY25['Revenue']:,} | Acme_Financials.xlsx › P&L › F6 |",
        f"| FY25 EBITDA | {FY25['EBITDA']:,} | Acme_Financials.xlsx › P&L › F14 |",
        f"| FY26 plan revenue | {PNL['Revenue'][4]:,} | Acme_Financials.xlsx › P&L › G6 |",
        f"| FY26 plan EBITDA | {EBITDA[4]:,} | Acme_Financials.xlsx › P&L › G14 |",
        f"| FY26 plan total opex | {OPEX[4]:,} | Acme_Financials.xlsx › P&L › G13 |",
        f"| FY25 total opex | {FY25['Total opex']:,} | Acme_Financials.xlsx › P&L › F13 |",
        f"| FY25 gross margin | {FY25['Gross margin']:,} | Acme_Financials.xlsx › P&L › F8 |",
        "| Gross margin % (any quarter) | NULL — formula with no cached value | Acme_Financials.xlsx › P&L › B9:G9 |",
        "| EBITDA margin % (any quarter) | NULL — formula with no cached value | Acme_Financials.xlsx › P&L › B15:G15 |",
        "",
        "Quarterly series (Q1, Q2, Q3, Q4):",
        "",
        f"- Revenue: {', '.join(f'{v:,}' for v in PNL['Revenue'][:4])}",
        f"- COGS: {', '.join(f'{v:,}' for v in PNL['COGS'][:4])}",
        f"- Gross margin: {', '.join(f'{v:,}' for v in GM[:4])}",
        f"- Total opex: {', '.join(f'{v:,}' for v in OPEX[:4])}",
        f"- EBITDA: {', '.join(f'{v:,}' for v in EBITDA[:4])}",
        "",
        "## Acme_Financials.xlsx › Headcount",
        "",
        "Title on row 1, header on row 3, departments rows 4-8, `Total` row 9 (formulas, cached).",
        "",
        "| Question | Answer | Citation |",
        "| --- | --- | --- |",
        f"| Total headcount in January | {HC_TOTAL[0]} | Acme_Financials.xlsx › Headcount › B9 |",
        f"| Total headcount in June | {HC_TOTAL[5]} | Acme_Financials.xlsx › Headcount › G9 |",
        f"| Engineering headcount in June | {HEADCOUNT['Engineering'][5]} | Acme_Financials.xlsx › Headcount › G4 |",
        f"| Net adds Jan to Jun | {HC_TOTAL[5] - HC_TOTAL[0]} | derived from B9 and G9 |",
        "",
        f"Monthly totals ({', '.join(MONTHS)}): {', '.join(str(v) for v in HC_TOTAL)}",
        "",
        "## Acme_Financials.xlsx › Regions",
        "",
        "Two tables on one sheet. The first starts at C5 (columns A and B are empty), the second",
        "at C12 after two blank rows.",
        "",
        "| Question | Answer | Citation |",
        "| --- | --- | --- |",
    ]
    letters = {"North America": 6, "EMEA": 7, "APAC": 8, "LATAM": 9}
    for region, (fy25, fy26, cust) in REGIONS.items():
        r = letters[region]
        lines.append(f"| {region} FY25 revenue | {fy25:,} | Acme_Financials.xlsx › Regions › D{r} |")
    for region, (fy25, fy26, cust) in REGIONS.items():
        r = letters[region]
        lines.append(f"| {region} customers | {cust} | Acme_Financials.xlsx › Regions › F{r} |")
    seg_rows = {"Enterprise": 13, "Mid-market": 14, "SMB": 15}
    for seg, (fy25, logos) in SEGMENTS.items():
        lines.append(f"| {seg} FY25 revenue | {fy25:,} | Acme_Financials.xlsx › Regions › D{seg_rows[seg]} |")
    lines += [
        "",
        "Consistency checks that must hold:",
        "",
        f"- Regions FY25 revenue sums to {sum(v[0] for v in REGIONS.values()):,} = P&L F6.",
        f"- Regions FY26 plan sums to {sum(v[1] for v in REGIONS.values()):,} = P&L G6.",
        f"- Region customers sum to {sum(v[2] for v in REGIONS.values())} = rows in customers.csv.",
        f"- Segment FY25 revenue sums to {sum(v[0] for v in SEGMENTS.values()):,} = P&L F6.",
        f"- Segment logos sum to {sum(v[1] for v in SEGMENTS.values())} = rows in customers.csv.",
        "",
        "## customers.csv",
        "",
        "200 rows. Columns: customer_id, customer_name, segment, region, arr_usd, churned,",
        "renewal_month, csat. ARR is in dollars; segment ARR totals match the Regions sheet",
        "segment revenue in thousands exactly.",
        "",
        "| Question | Answer |",
        "| --- | --- |",
        f"| Total ARR | ${facts['total_arr']:,} |",
        f"| Churned ARR (all segments) | ${facts['churned_arr']:,} |",
        f"| Churned customer count | {sum(CHURNED_PER_SEGMENT.values())} of 200 |",
    ]
    for seg in SEGMENTS:
        s = bs[seg]
        lines.append(f"| {seg}: customers / ARR / churned / churned ARR / churn rate | "
                     f"{s['n']} / ${s['arr']:,} / {s['churned_n']} / ${s['churned_arr']:,} / {s['churn_rate']}% |")
    lines += [
        "",
        "Enterprise churn (13.6%) is deliberately worse than mid-market (7.0%) and SMB (5.9%), so",
        "\"is churn getting worse in enterprise\" has a real answer that agrees with Board_Memo.docx,",
        "notes.md and the Strategy_Deck speaker notes.",
        "",
        "## Documents",
        "",
        "| File | What it says | Where |",
        "| --- | --- | --- |",
        "| Strategy_Deck.pptx | FY25 revenue, gross margin, EBITDA and the 12.3% margin | slide 2 |",
        "| Strategy_Deck.pptx | Enterprise is 44 logos but 60% of ARR; NRR 118% vs 94% SMB | slide 3 |",
        "| Strategy_Deck.pptx | Region table, every region planned at 25% growth | slide 4 |",
        "| Strategy_Deck.pptx | Headcount 119 in January to 152 by June | slide 5 |",
        "| Strategy_Deck.pptx | Speaker notes admit APAC cannot carry 25% growth | slide 4 notes |",
        "| Board_Memo.docx | 12.3% actual vs 15% commitment; opex is the whole story | Summary |",
        "| Board_Memo.docx | Quarterly EBITDA bridge table | Quarterly EBITDA bridge |",
        "| Board_Memo.docx | 16 of 200 churned; 13.6% / 7.0% / 5.9% by segment | Churn and retention |",
        "| Market_Report.pdf | TAM $14.2bn in 2025 to $21.6bn by 2028, 15% CAGR | page 1 |",
        "| Market_Report.pdf | Four vendors hold 46% of the market | page 2 |",
        "| Market_Report.pdf | Median NRR 104%, upper quartile 121%; gross retention 88% | page 3 |",
        "| notes.md | The two percent rows came through blank | Open questions for the CFO |",
        "| notes.md | Blended discount 29% vs market 22% | Pricing |",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    folder = Path(argv[1] if len(argv) > 1 else "sample_context").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    build_workbook(folder / "Acme_Financials.xlsx")
    facts = build_customers(folder / "customers.csv")
    build_deck(folder / "Strategy_Deck.pptx")
    build_memo(folder / "Board_Memo.docx")
    build_pdf(folder / "Market_Report.pdf")
    build_notes(folder / "notes.md")
    write_answer_key(folder / "ANSWER_KEY.md", facts)
    for p in sorted(folder.iterdir()):
        print(f"  {p.name:26s} {p.stat().st_size:>9,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
