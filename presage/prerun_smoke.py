"""Pre-run smoke test:  uv run python -m presage.prerun_smoke [--workbook PATH]

Checks `presage.prerun` the way the brain uses it: a spoken line goes in, an `Ask` comes out,
and the Ask maps onto exactly one Index call with the right cells and values. Most of it runs
against a hand-written lever table shaped like the one `presage.ingest.scratch` builds (a rate,
a churn, a headcount, a salary, a year and two scenario switches), so it is hermetic and takes
milliseconds. If the big test workbook is on this machine the last section runs four lines through
the real Index and checks the numbers that come back.

Every check prints PASS or FAIL and the run ends with "prerun smoke: N passed, M failed".
No model is called anywhere, with or without a key.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from presage import prerun as P

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def head(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def near(a, b, tol: float = 1e-9) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# a lever table shaped like the real one
# --------------------------------------------------------------------------- #
def out(cell: str, label: str, fmt: str = "money", value: float = 1.0) -> dict:
    return {"cell": cell, "label": label, "fmt": fmt, "value": value}


REVENUE = out("Summary!G19", "Subscriber fee revenue", "money", 8_148_348.58)
SUBSCRIBERS = out("Summary!G16", "Subscribers", "count", 368)
OPEX = out("Summary!G27", "Total operating cost", "money", 15_755_977.35)
EBITDA = out("Summary!G28", "EBITDA", "money", 12_079_871.22)

LEVERS = [
    {"file": "m.xlsx", "sheet": "Subscriber Fee Model", "cell": "B48", "fmt": "pct", "value": 0.0085,
     "label": "Median platform fee on accounts over $1M", "header": "Year 1", "choices": [],
     "aliases": ["platform fee", "gmv"], "active": True, "outputs": [REVENUE, EBITDA, SUBSCRIBERS]},
    {"file": "m.xlsx", "sheet": "Assumptions", "cell": "C39", "fmt": "pct", "value": 0.08,
     "label": "Annual subscriber churn", "header": "Base", "choices": [], "aliases": ["subscriber churn"],
     "active": True, "outputs": [OPEX, EBITDA, SUBSCRIBERS]},
    {"file": "m.xlsx", "sheet": "Assumptions", "cell": "C34", "fmt": "count", "value": 3.0,
     "label": "Retail partners paying, Year 5", "header": "Base", "choices": [], "aliases": [],
     "active": True, "outputs": [out("Summary!G20", "Retail partner revenue", "money", 3_592_500)]},
    {"file": "m.xlsx", "sheet": "Operating Costs", "cell": "B30", "fmt": "money", "value": 153_000,
     "label": "Chief executive", "header": "Base salary", "choices": [], "aliases": [],
     "active": True, "outputs": [OPEX, EBITDA]},
    {"file": "m.xlsx", "sheet": "Operating Costs", "cell": "B31", "fmt": "money", "value": 175_000,
     "label": "Head of product", "header": "Base salary", "choices": [], "aliases": [],
     "active": True, "outputs": [OPEX, EBITDA]},
    {"file": "m.xlsx", "sheet": "Channel Fee Model", "cell": "C19", "fmt": "count", "value": 3.0,
     "label": "Year Northwind signs", "header": "Base", "choices": [], "aliases": ["northwind"],
     "active": True, "outputs": [out("Summary!G21", "Channel partner funding", "money", 16e6)]},
    {"file": "m.xlsx", "sheet": "Summary", "cell": "B4", "fmt": "text", "value": "Base",
     "label": "Market Scenario", "header": "Year 0", "aliases": ["downside", "base", "upside"],
     "choices": ["Downside", "Base", "Upside"], "active": True, "outputs": [REVENUE, EBITDA]},
    {"file": "m.xlsx", "sheet": "Summary", "cell": "B8", "fmt": "text", "value": "GMV Fee (S2)",
     "label": "Subscriber Fee setting", "header": "Year 0", "aliases": ["flat dues", "gmv fee"],
     "choices": ["Flat Dues (S1)", "GMV Fee (S2)", "Volume Tiers (S3)"], "active": True,
     "outputs": [REVENUE]},
]
OUTS = P.output_table(LEVERS)
FEE, CHURN, RETAIL = "Subscriber Fee Model!B48", "Assumptions!C39", "Assumptions!C34"
CEO, CPO, NORTHWIND = "Operating Costs!B30", "Operating Costs!B31", "Channel Fee Model!C19"
SCENARIO, STRUCTURE = "Summary!B4", "Summary!B8"


def ask(line: str, prior: dict | None = None) -> P.Ask:
    return P.parse_ask(line, LEVERS, prior=prior, outputs=OUTS)


# --------------------------------------------------------------------------- #
class FakeIndex:
    """Records the one call the Ask maps onto, so the wiring is checked without a workbook."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def whatif(self, file, sets, watch=None):
        self.calls.append(("whatif", file, dict(sets), watch))
        return {"file": "m.xlsx", "changes": [
            {"cell": c, "label": "x", "was": 0, "now": v, "was_s": "0", "now_s": str(v)}
            for c, v in sets.items()], "outputs": [
            {"cell": "Summary!G19", "label": "Subscriber fees", "was": 1.0, "now": 2.0,
             "was_s": "$1", "now_s": "$2", "pct": 1.0}], "partial": [], "ms": 1}

    def goalseek(self, file, target, value, lever, lo=None, hi=None):
        self.calls.append(("goalseek", file, target, value, lever))
        return {"file": "m.xlsx", "lever": lever, "label": "fee", "target": target,
                "target_label": "Subscriber fees", "was": 0.0085, "was_s": "0.85%", "value": 0.0104,
                "value_s": "1.04%", "achieved": 1e7, "achieved_s": "$10.0M", "want_s": "$10.0M",
                "monotone": True, "tries": 21, "ms": 25}

    def compare(self, file, cases, watch=None, labels=None):
        self.calls.append(("compare", file, [dict(c) for c in cases], labels))
        return {"file": "m.xlsx", "cases": [{"label": (labels or [None, None])[i],
                                             "sets": dict(c), "changes": [], "outputs": {},
                                             "partial": []} for i, c in enumerate(cases)],
                "rows": [], "ms": 2}

    def sweep(self, file, cell, values, watch=None):
        self.calls.append(("sweep", file, cell, list(values)))
        return {"file": "m.xlsx", "cell": cell, "label": "fee", "fmt": "pct",
                "points": [{"value": v, "value_s": str(v), "outputs": {}, "outputs_s": {},
                            "partial": []} for v in values], "rows": [], "ms": 3}

    def explain(self, file, cell, sets=None, hops=6):
        self.calls.append(("explain", file, cell, dict(sets or {})))
        return {"file": "m.xlsx", "cell": cell, "label": "EBITDA", "was": 1.0, "now": 2.0,
                "moved": True, "hops": 3, "reaches": True, "chain": [
                    {"cell": cell, "label": "EBITDA", "formula": "=A1", "was": 1, "now": 2,
                     "delta": 1, "was_s": "1", "now_s": "2", "loc": "m.xlsx › X"}], "ms": 4}


# --------------------------------------------------------------------------- #
def numbers() -> None:
    head("what the client said, as a number")
    cases = [
        ("one percent", 0.01, True), ("1.25 percent", 0.0125, True),
        ("75 basis points", 0.0075, True), ("half a percent", 0.005, True),
        ("twelve percent", 0.12, True), ("thirty thousand", 30_000.0, False),
        ("twenty five hundred", 2500.0, False), ("four hundred thousand", 400_000.0, False),
        ("one and a half million", 1_500_000.0, False), ("220 thousand", 220_000.0, False),
        ("a thousand", 1000.0, False), ("a half", 0.5, False), ("eight times", 8.0, False),
        ("15.8 million", 15_800_000.0, False), ("one and a quarter", 1.25, False),
    ]
    for text, want, is_pct in cases:
        got = P.spoken_numbers(f"what if it were {text} then")
        ok = bool(got) and near(got[0].value, want, 1e-6) and got[0].is_pct == is_pct
        check(f'"{text}" is {want:g}', ok, "" if ok else f"got {[g.value for g in got]}")
    seq = P.spoken_numbers("at one, one and a quarter and one and a half percent")
    check("a list keeps its items apart and shares the unit",
          [round(n.value, 6) for n in seq] == [0.01, 0.0125, 0.015], str([n.value for n in seq]))
    both = P.spoken_numbers("dues between fifteen and thirty thousand")
    check("a bare endpoint takes the scale of the one beside it",
          [n.value for n in both] == [15_000.0, 30_000.0], str([n.value for n in both]))
    aside = P.spoken_numbers("one percent of GMV instead of the 0.85")
    check("the value being replaced is marked as an aside",
          len(aside) == 2 and not aside[0].aside and aside[1].aside)
    for line, want in [("remind me what Q3 revenue came in at again?", [True]),
                       ("third quarter revenue was what again", [True]),
                       ("revenue in Q 3", [True]), ("FY26 revenue at 3 million", [True, False]),
                       ("take the fee to 1 percent in Q4", [False, True]),
                       ("what if it were 2,000 by 2030", [False, True])]:
        got = P.spoken_numbers(line)
        check(f'periods are columns: "{line}"', [n.is_year for n in got] == want,
              str([(n.raw, n.is_year) for n in got]))                                  # E70
    look = ask("remind me what the platform fee came in at last quarter?")
    check("a lookup with a period in it is not a what-if", look.kind is None, f"{look.kind} {look.sets}")
    still = ask("suppose the platform fee came in at one percent in Q3")
    check("a change cue keeps a what-if even with 'came in at'",
          still.kind == "whatif" and still.sets.get(FEE) == 0.01, f"{still.kind} {still.sets}")
    year = P.spoken_numbers("four retail partners by year five")
    check("a year is a column, not a value",
          len(year) == 2 and not year[0].is_year and year[1].is_year)


def one_lever() -> None:
    head("one lever, one value")
    a = ask("what if we charged one percent of GMV instead of the 0.85")
    check("the rate goes on the fee", a.kind == "whatif" and near(a.sets.get(FEE), 0.01), a.line())
    check("it does not also set the value it was told to replace", len(a.sets) == 1)
    b = ask("what if subscriber churn were twelve percent")
    check("churn by its spoken name", near(b.sets.get(CHURN), 0.12), b.line())
    c = ask("what if we signed four retail partners by year five instead of three")
    check("a count, and the year is not one", c.sets == {RETAIL: 4.0}, c.line())
    d = ask("what if the chief executive took two hundred thousand")
    check("money at its own scale", near(d.sets.get(CEO), 200_000.0), d.line())
    e = ask("what if Northwind signed a year later")
    check("a move said against where the model is", near(e.sets.get(NORTHWIND), 4.0), e.line())
    f = ask("what is the fee right now")
    check("a lookup moves nothing", f.kind is None, f.line())


def several_levers() -> None:
    head("several levers in one sentence")
    a = ask("what if we charged one percent and churn were six percent")
    check("the churn takes the churn number", near(a.sets.get(CHURN), 0.06), a.line())
    b = ask("one percent GMV fee and eight percent penetration of churn")
    check("two values, paired by position", near(b.sets.get(FEE), 0.01), b.line())
    c = ask("what if we cut the chief executive to 140 and the head of product to 150")
    check("two salaries, each beside its own name",
          near(c.sets.get(CEO), 140_000.0) and near(c.sets.get(CPO), 150_000.0), c.line())
    d = ask("what if we are at 1.25 percent with four retail partners")
    check("the lever nobody named is left alone rather than guessed",
          d.sets.get(RETAIL) == 4.0 and FEE not in d.sets, d.line())


def scenarios() -> None:
    head("a scenario name is a value")
    a = ask("what if we ran the downside case")
    check("a named case switches the selector", a.sets == {SCENARIO: "Downside"}, a.line())
    b = ask("what if we used flat dues instead of the GMV fee")
    check("the structure it asks for, not the one it is replacing",
          b.sets == {STRUCTURE: "Flat Dues (S1)"}, b.line())
    c = ask("what does downside look like at a one percent fee")
    check("a case and a rate together",
          c.sets.get(SCENARIO) == "Downside" and near(c.sets.get(FEE), 0.01), c.line())
    d = ask("what is the fee per subscriber in the model")
    check("a lookup does not switch a scenario", d.kind is None, d.line())
    e = ask("show me the upside scenario")
    check("a scenario named with no what-if still switches",
          e.sets == {SCENARIO: "Upside"}, e.line())


def other_kinds() -> None:
    head("goal seek, compare, sweep, explain")
    a = ask("what fee would get us to ten million of subscriber revenue in year five")
    check("goal seek: the lever, the target and the number",
          a.kind == "goalseek" and a.lever == FEE and near(a.goal, 1e7)
          and a.target == "Summary!G19", a.line())
    b = ask("how many retail partners do we need for five million of retail partner revenue")
    check("goal seek on a count", b.kind == "goalseek" and b.lever == RETAIL, b.line())
    c = ask("upside versus downside on year five EBITDA")
    check("compare: two cases on one switch",
          c.kind == "compare" and [x["sets"] for x in c.cases]
          == [{SCENARIO: "Upside"}, {SCENARIO: "Downside"}], c.line())
    d = ask("what is the gap between eight and twelve percent churn")
    check("compare: two values on one lever",
          d.kind == "compare" and all(CHURN in x["sets"] for x in d.cases), d.line())
    e = ask("show me churn from six to twelve percent")
    check("sweep: the ends of a range filled in",
          e.kind == "sweep" and e.cell == CHURN and near(min(e.values), 0.06)
          and near(max(e.values), 0.12) and len(e.values) >= 3, e.line())
    f = ask("give me the GMV fee at one, one and a quarter and one and a half percent")
    check("sweep: a list said in one breath",
          f.kind == "sweep" and [round(v, 6) for v in f.values] == [0.01, 0.0125, 0.015], f.line())
    g = ask("why did the EBITDA move when we raised churn")
    check("explain: the cell it is about",
          g.kind == "explain" and g.target == "Summary!G28", g.line())
    h = ask("what is driving EBITDA")
    check("explain with nothing changed",
          h.kind == "explain" and h.target == "Summary!G28" and not h.sets, h.line())


def follow_ups() -> None:
    head("follow-ups and confidence")
    prior = {"question": "what if the fee were 1%", "lever": LEVERS[0],
             "sets": {FEE: 0.01}}
    a = ask("and at 1.25?", prior=prior)
    check("a follow-up lands on the previous lever", near(a.sets.get(FEE), 0.0125), a.line())
    b = ask("why did that move", prior=prior)
    check("an explain carries the previous change",
          b.kind == "explain" and b.sets == {FEE: 0.01}, b.line())
    c = ask("what if churn were six percent")
    check("a clean parse is confident", c.confidence >= 0.5, f"confidence {c.confidence}")
    d = ask("what if we are at 1.25 percent with four retail partners")
    check("a parse that left a value unplaced is not", d.confidence < 0.5,
          f"confidence {d.confidence}")


def wiring() -> None:
    head("the Ask maps onto one Index call")
    ix = FakeIndex()
    a = ask("what if churn were six percent")
    out_ = P.run_ask(ix, a)
    check("whatif reaches Index.whatif with the cell and the value",
          ix.calls[-1][0] == "whatif" and near(ix.calls[-1][2].get(CHURN), 0.06), str(ix.calls[-1]))
    block = P.evidence_block(a, out_)
    check("the evidence block says the run already happened",
          "WHAT-IF ALREADY RUN" in block and 'answered_from="derived"' in block,
          block.splitlines()[0][:60] if block else "(empty)")
    b = ask("what fee would get us to ten million of subscriber revenue in year five")
    gs = P.run_ask(ix, b)
    check("goalseek reaches Index.goalseek with target, value and lever",
          ix.calls[-1][0] == "goalseek" and ix.calls[-1][4] == FEE, str(ix.calls[-1])[:70])
    check("its evidence names the answer and what it puts the target at",
          "GOAL SEEK ALREADY RUN" in P.evidence_block(b, gs)
          and "1.04%" in P.evidence_block(b, gs))
    c = ask("upside versus downside on year five EBITDA")
    cp = P.run_ask(ix, c)
    check("compare reaches Index.compare with both cases",
          ix.calls[-1][0] == "compare" and len(ix.calls[-1][2]) == 2, str(ix.calls[-1])[:70])
    check("its evidence lists every case", "COMPARISON ALREADY RUN" in P.evidence_block(c, cp))
    d = ask("show me churn from six to twelve percent")
    sw = P.run_ask(ix, d)
    check("sweep reaches Index.sweep with the values",
          ix.calls[-1][0] == "sweep" and len(ix.calls[-1][3]) >= 3, str(ix.calls[-1])[:70])
    check("its evidence lists every point", "RANGE ALREADY RUN" in P.evidence_block(d, sw))
    e = ask("why did the EBITDA move when we raised churn")
    ex = P.run_ask(ix, e)
    check("explain reaches Index.explain with the cell",
          ix.calls[-1][0] == "explain" and ix.calls[-1][2] == "Summary!G28", str(ix.calls[-1])[:70])
    check("its evidence is the chain", "MECHANISM ALREADY TRACED" in P.evidence_block(e, ex))
    check("an empty Ask runs nothing", P.run_ask(ix, P.Ask()) is None)


def speed() -> None:
    head("cost of the parse")
    lines = ["what if we charged one percent and churn were six percent",
             "what fee would get us to ten million of subscriber revenue",
             "upside versus downside on year five EBITDA",
             "show me churn from six to twelve percent",
             "why did the EBITDA move when we raised churn"]
    t0 = time.perf_counter()
    for _ in range(40):
        for line in lines:
            P.parse_ask(line, LEVERS, outputs=OUTS)
    ms = (time.perf_counter() - t0) * 1000 / (40 * len(lines))
    check("a line parses in under 5 ms", ms < 5.0, f"{ms:.2f} ms a line")


# --------------------------------------------------------------------------- #
# Set PRESAGE_BIG_MODEL to a large scenario workbook of your own to run this section.
SOURCES = [p for p in [os.environ.get("PRESAGE_BIG_MODEL", "")] if p]


def real_workbook(given: str | None) -> None:
    """Four lines end to end on the real model, if it is on this machine."""
    head("the real workbook, end to end")
    path = next((p for p in (Path(c).expanduser() for c in ([given] if given else []) + SOURCES)
                 if p.exists()), None)
    if path is None:
        print("  skipped: no PRESAGE_BIG_MODEL workbook on this machine")
        return
    from presage.ingest import Index
    ctx = Path(tempfile.gettempdir()) / "presage-e4-ctx"
    ctx.mkdir(parents=True, exist_ok=True)
    target = ctx / path.name
    if not target.exists() or target.stat().st_size != path.stat().st_size:
        shutil.copy2(path, target)
    ix = Index()
    ix.build(ctx)
    levers = ix.levers()
    outs = P.output_table(levers)
    check("the lever table is there", len(levers) > 50, f"{len(levers)} levers")

    a = P.parse_ask("what if we charged one percent of GMV instead of the 0.85", levers,
                    outputs=outs)
    got = P.run_ask(ix, a)
    check("1% of GMV puts subscriber fees at $9.59M",
          any(abs((o.get("now") or 0) - 9_586_292.44) < 1 for o in (got or {}).get("outputs", [])),
          a.line())
    b = P.parse_ask("what if we ran the downside case", levers, outputs=outs)
    got = P.run_ask(ix, b)
    check("the downside case recalculates the whole model",
          bool((got or {}).get("changes")) and len((got or {}).get("outputs") or []) > 1, b.line())
    c = P.parse_ask("what fee would get us to ten million of subscriber revenue in year five",
                    levers, outputs=outs)
    got = P.run_ask(ix, c) or {}
    check("the goal seek solves the fee at about 1.04%",
          not got.get("error") and abs((got.get("value") or 0) - 0.01043) < 5e-4,
          f'{c.line()} -> {got.get("value_s")}')
    d = P.parse_ask("premium versus market pricing", levers, outputs=outs)
    got = P.run_ask(ix, d) or {}
    check("the comparison runs both cases on the same workbook",
          len(got.get("cases") or []) == 2, d.line())


def main(argv: list[str]) -> int:
    given = None
    if "--workbook" in argv:
        given = argv[argv.index("--workbook") + 1]
    numbers()
    one_lever()
    several_levers()
    scenarios()
    other_kinds()
    follow_ups()
    wiring()
    speed()
    if "--no-workbook" not in argv:
        try:
            real_workbook(given)
        except Exception as e:  # noqa: BLE001  the hermetic checks are the ones that must pass
            check("the real workbook section", False, f"{type(e).__name__}: {e}")
    head(f"prerun smoke: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
