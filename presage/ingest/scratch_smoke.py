"""Scratch-paper smoke test:  uv run python -m presage.ingest.scratch_smoke

    uv run python -m presage.ingest.scratch_smoke ["~/Downloads/Plan_Model.xlsx"]

Against a real consultant model (Plan_Model.xlsx, 22 sheets, 3,468 formulas) it checks
that the workbook compiles, that the lever table finds the inputs a client asks about, that
a what-if reproduces Excel to the cent, that the QUARTILE plugin is actually being used,
that `calc` computes and refuses, and that the grid lands as derived Q&A on the instant path.

The workbook is copied into a scratch folder (`PRESAGE_SCRATCH_CTX`, else a temp dir) and
the original is never touched. Every check prints PASS or FAIL and the run ends with
"scratch smoke: N passed, M failed". The deep pass runs with a canned model client so the
check costs nothing; if ANTHROPIC_API_KEY is set, the spoken-alias call is exercised once
for real at the end.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from presage.ingest import Index
from presage.ingest import scratch as _scratch

SOURCE = Path("~/Downloads/Plan_Model.xlsx").expanduser()
FILE = "Plan_Model.xlsx"

# SOURCE values, read off the workbook itself (Excel's own cached results).
LEVER = ("Subscriber Fee Model", "B48", 0.0085)
CHURN = ("Assumptions", "C39", 0.08)
EXPECT = {
    "Summary Scenarios!G57": (8148348.575, 9586292.44),
    "Subscriber Fee Model!B58": (22142.25156, 26049.71),
}
QUARTILE_CELL = ("Revenue Multiple", "B30", 5.125)
PLUGIN_CELLS = {("Revenue Multiple", "B30"): 5.125, ("Revenue Multiple", "D30"): 7.675,
                ("Revenue Multiple", "C30"): 6.3, ("Subscriber Fee Model", "E8"): 1}
GRID_BUDGET_S = 15.0

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


def close(a, b, tol: float = 0.02) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol


def head(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def workdir(source: Path) -> Path:
    folder = Path(os.environ.get("PRESAGE_SCRATCH_CTX")
                  or (Path(tempfile.gettempdir()) / "presage-scratch-ctx"))
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / source.name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)          # a copy: the original is never written to
    return folder


def dry_llm_json():
    """A model client that answers nothing, so the lever table must stand on its own."""
    calls = {"n": 0}

    async def llm_json(prompt: str, schema: dict, model: str | None = None) -> dict:
        calls["n"] += 1
        props = schema.get("properties", {})
        for key in ("contexts", "qa", "definitions", "levers"):
            if key in props:
                return {key: []}
        return {"file": "(dry run)", "sections": []}

    return llm_json, calls


def real_llm_json():
    """The same shape on the real API, for the one aliases call at the end."""
    from presage.llm import make_llm

    llm = make_llm("anthropic")
    model = os.environ.get("PRESAGE_DEEP_FAST", "claude-haiku-4-5-20251001")

    async def llm_json(prompt: str, schema: dict, model_name: str | None = None) -> dict:
        resp = await llm.client.messages.create(
            model=model_name or model,
            max_tokens=1000,          # the ceiling classify() gives the deep pass
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}})
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    return llm_json


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    source = Path(argv[1]).expanduser() if len(argv) > 1 else SOURCE
    if not source.exists():
        print(f"no test workbook at {source}")
        return 1
    folder = workdir(source)

    head(f"BUILD  {folder}")
    t0 = time.perf_counter()
    ix = Index()
    st = ix.build(folder)
    print(f"  {st}")
    print(f"  wall {(time.perf_counter() - t0) * 1000:.0f} ms")

    # ------------------------------------------------------------- the graph
    head("COMPILE  (pycel + presage.ingest.xlfuncs)")
    graph = ix._graph_for(FILE)
    t0 = time.perf_counter()
    graph.compile()
    compile_s = time.perf_counter() - t0
    print(f"  compiled in {compile_s:.2f} s; {len(graph._snapshot):,} cells in the graph")
    check("every formula cell evaluates (nothing falls back to its cached value)",
          not graph.broken, f"{len(graph.broken)} unevaluable: {sorted(graph.broken)[:4]}")
    if graph.mismatched:
        print(f"  note: {len(graph.mismatched)} cells recompute differently from Excel's "
              f"saved value: {sorted(graph.mismatched)}")

    head("QUARTILE PLUGIN")
    from pycel import ExcelCompiler
    ex = ExcelCompiler(str(folder / FILE), plugins=[_scratch.PLUGIN])
    sheet, cell, want = QUARTILE_CELL
    addr = _scratch.canon(sheet, cell)
    ex.evaluate(addr)                          # loads the cell with Excel's cached value
    ex.cell_map[addr].value = None             # force pycel to actually run the formula
    got = ex.evaluate(addr)
    check(f"ExcelCompiler(plugins=[...]) evaluates {addr} numerically",
          isinstance(got, (int, float)) and not isinstance(got, bool), f"got {got!r}")
    check(f"{addr} == the value Excel cached", close(got, want, 1e-9), f"{got} vs {want}")
    for (psheet, pcell), expect in PLUGIN_CELLS.items():
        paddr = _scratch.canon(psheet, pcell)
        node = ex.cell_map.get(paddr) or (ex.evaluate(paddr), ex.cell_map[paddr])[1]
        node.value = None
        val = ex.evaluate(paddr)
        check(f"{paddr} recomputes to Excel's value "
              f"({str(node.formula.base_formula)[:28]})", close(val, expect, 1e-6),
              f"{val} vs {expect}")

    # ------------------------------------------------------------- the levers
    head("LEVERS + GRID")
    from presage.ingest import store as _store
    con = _store.connect_ro(folder)             # built from scratch, never from the cache
    try:
        t0 = time.perf_counter()
        record, _ = _scratch.build(con, folder / FILE, FILE, ix._file_key(FILE), graph)
    finally:
        con.close()
    build_s = time.perf_counter() - t0
    ix._scratch[FILE] = record
    levers, grid = record["levers"], record["grid"]
    print(f"  {len(levers)} levers, {len(grid)} grid points: {compile_s:.2f} s to compile "
          f"+ {build_s:.2f} s to discover and run the grid")
    check(f"the whole table and grid build in under {GRID_BUDGET_S:.0f} s",
          compile_s + build_s < GRID_BUDGET_S, f"{compile_s + build_s:.2f} s")

    by_cell = {(lv["sheet"], lv["cell"]): lv for lv in levers}
    sheet, cell, value = LEVER
    lever = by_cell.get((sheet, cell))
    check(f"{sheet}!{cell} is a lever", lever is not None)
    if lever:
        check(f"{sheet}!{cell} reads as a percentage at {value}",
              lever["fmt"] == "pct" and close(lever["value"], value, 1e-9),
              f'fmt={lever["fmt"]} value={lever["value"]}')
        check(f"{sheet}!{cell} moves a labelled cell on Summary Scenarios",
              any(o["cell"].startswith("Summary Scenarios!") and o["label"]
                  for o in lever["outputs"]),
              ", ".join(o["cell"] for o in lever["outputs"][:3]))
    sheet, cell, value = CHURN
    churn = by_cell.get((sheet, cell))
    check(f"{sheet}!{cell} (churn) is a lever", churn is not None,
          f'{churn["label"]} = {churn["value"]}' if churn else "")
    if churn:
        check(f"{sheet}!{cell} is {value}", close(churn["value"], value, 1e-9),
              str(churn["value"]))

    block = ix.levers_block(6)
    check("levers_block prints the prompt table", block.startswith("== LEVERS"),
          f"{len(ix.levers_block(40))} chars at limit 40")
    print("\n".join("    " + line for line in block.splitlines()[:4]))

    # ------------------------------------------------------------- the what-if
    head("WHAT-IF   Subscriber Fee Model!B48 = 1%")
    got = ix.whatif(None, {"Subscriber Fee Model!B48": "1%"},
                    watch=list(EXPECT))
    print(f"  {got['ms']:.1f} ms, {len(got['outputs'])} outputs")
    rows = {o["cell"]: o for o in got["outputs"]}
    for ref, (was, now) in EXPECT.items():
        row = rows.get(ref)
        check(f"{ref} was {was:,.2f}", row is not None and close(row["was"], was),
              str(row and row["was"]))
        check(f"{ref} becomes {now:,.2f}", row is not None and close(row["now"], now),
              str(row and row["now"]))
    check("the what-if reports no partial recalc", not got["partial"], str(got["partial"]))
    change = got["changes"][0]
    check("the change is reported as 0.85% -> 1%",
          change["was_s"] == "0.85%" and change["now_s"] == "1%",
          f'{change["was_s"]} -> {change["now_s"]}')
    check("the model is restored afterwards",
          close(ix._graph_for(FILE).value("Summary Scenarios", "G57"), 8148348.575),
          str(ix._graph_for(FILE).value("Summary Scenarios", "G57")))
    for bad, why in (("Summary Scenarios!G57", "a formula cell"),
                     ("Subscriber Fee Model!ZZ999", "an unknown cell")):
        try:
            ix.whatif(None, {bad: 1})
            check(f"whatif refuses {why}", False, "no error raised")
        except ValueError as e:
            check(f"whatif refuses {why}", True, str(e)[:70])

    head("PARTIAL RECALC  (simulating a formula pycel cannot run)")
    import networkx as nx

    live = ix._graph_for(FILE)
    was_broken, was_tainted = dict(live.broken), set(live._tainted)
    lever_node = live._ex.cell_map["Subscriber Fee Model!B48"]
    out_node = live._ex.cell_map[lever["outputs"][0]["cell"]]
    path = nx.shortest_path(live._ex.dep_graph, lever_node, out_node)
    blind = path[max(1, len(path) // 2)]
    blind_addr = blind.address.address
    live.broken[blind_addr] = "simulated: pretend this function is not implemented"
    live._tainted = (was_tainted | {blind_addr}
                     | {c.address.address for c in nx.descendants(live._ex.dep_graph, blind)})
    got = ix.whatif(None, {"Subscriber Fee Model!B48": "1%"})
    rows = {o["cell"]: o for o in got["outputs"]}
    check(f"an unevaluable cell ({blind_addr}) puts its dependants in partial",
          bool(got["partial"]), f'{len(got["partial"])} of {len(got["outputs"])} outputs')
    check("the other outputs still come back with numbers",
          any(isinstance(o["now"], (int, float)) for o in got["outputs"]))
    live.broken, live._tainted = was_broken, was_tainted
    check("the model still reads the original values afterwards",
          close(live.value("Summary Scenarios", "G57"), 8148348.575))

    head("CONCURRENCY + GRAPH LIFETIME")
    import threading
    errors: list[str] = []

    def hammer(rate: float) -> None:
        try:
            got = ix.whatif(None, {"Subscriber Fee Model!B48": rate},
                            watch=["Summary Scenarios!G57"])
            row = got["outputs"][0]
            if not close(row["was"], 8148348.575):
                errors.append(f'{rate}: was {row["was"]}')
        except Exception as e:  # noqa: BLE001
            errors.append(f"{rate}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=hammer, args=(r,))
               for r in (0.005, 0.0085, 0.01, 0.02, 0.03, 0.05)]
    t0 = time.perf_counter()
    [t.start() for t in threads]
    [t.join() for t in threads]
    check("six concurrent what-ifs all see the original model", not errors,
          f"{(time.perf_counter() - t0) * 1000:.0f} ms" + ("; " + "; ".join(errors)
                                                           if errors else ""))

    live = ix._graph_for(FILE)
    live.key = "not-the-hash-any-more"
    fresh = ix._graph_for(FILE)
    check("a changed file hash drops the compiled model",
          fresh is not live and not fresh.ready)
    ix._graphs[FILE] = live                      # put the compiled one back
    live.key = ix._file_key(FILE)

    # ------------------------------------------------------------------- calc
    head("CALC")
    out = ix.calc("assets * rate", {"assets": "Subscriber Fee Model!B57", "rate": 0.01})
    check("calc('assets * rate') == 26,049.71", close(out["value"], 26049.71),
          f'{out["value"]}  <- {out["inputs"][0]["ref"]}')
    out2 = ix.calc("pct(a, b)", {"a": "Subscriber Fee Model!B58", "b": "Subscriber Fee Model!B57"})
    check("calc('pct(a, b)') == 0.85%", close(out2["value"], 0.0085, 1e-6), str(out2["value"]))
    for expr, inputs in (("__import__('os')", {}), ("rate.real", {"rate": 0.01}),
                         ("rate * missing", {"rate": 0.01}), ("2 ** 9999", {})):
        try:
            ix.calc(expr, inputs)
            check(f"calc refuses {expr!r}", False, "no error raised")
        except ValueError as e:
            check(f"calc refuses {expr!r}", True, str(e)[:66])

    # --------------------------------------------------- scenario switches (text)
    head("SCENARIO SWITCHES   (a text cell the formulas compare against)")
    switch = by_cell.get(("Summary Scenarios", "B5"))
    check("Summary Scenarios!B5 (Pricing Positioning) is a lever", switch is not None)
    if switch:
        check("it reads as text at its current value",
              switch["fmt"] == "text" and switch["value"] == "Premium",
              f'fmt={switch["fmt"]} value={switch["value"]!r}')
        check("its choices are the ones the workbook allows",
              switch["choices"] == ["Premium", "Market"], str(switch["choices"]))
        check("a client saying the case by name finds the cell",
              "market" in [a.lower() for a in switch["aliases"]], str(switch["aliases"]))
    scen = by_cell.get(("Summary Scenarios", "B4"))
    check("Summary Scenarios!B4 (Market Scenario) carries all three cases",
          bool(scen) and scen["choices"] == ["Downside", "Base", "Upside"],
          str(scen and scen["choices"]))

    got = ix.whatif(None, {"Summary Scenarios!B5": "Market"},
                    watch=["Summary Scenarios!C362"])
    row = {o["cell"]: o for o in got["outputs"]}.get("Summary Scenarios!C362")
    check("whatif on a text lever recalculates the model",
          row is not None and close(row["now"], 2875887.732),
          f'{row and row["was"]} -> {row and row["now"]}')
    check("the change reads as Premium -> Market",
          got["changes"][0]["was_s"] == "Premium" and got["changes"][0]["now_s"] == "Market",
          f'{got["changes"][0]["was_s"]} -> {got["changes"][0]["now_s"]}')
    loose = ix.whatif(None, {"Summary Scenarios!B4": "downside"},
                      watch=["Summary Scenarios!G57"])
    check("a value said loosely resolves to the workbook's spelling",
          loose["changes"][0]["now"] == "Downside", str(loose["changes"][0]["now"]))
    g57 = {o["cell"]: o for o in loose["outputs"]}.get("Summary Scenarios!G57")
    check("the downside case is the model's own downside",
          g57 is not None and close(g57["now"], 4893437.595), str(g57))
    try:
        ix.whatif(None, {"Summary Scenarios!B5": "Sideways"})
        check("whatif refuses a value the cell does not take", False, "no error raised")
    except ValueError as e:
        check("whatif refuses a value the cell does not take",
              "Premium" in str(e) and "Market" in str(e), str(e)[:70])
    check("the model is restored after a text what-if",
          ix._graph_for(FILE).value("Summary Scenarios", "B5") == "Premium")

    head("TWO LEVERS AT ONCE   (a number and a scenario in one call)")
    both = ix.whatif(None, {"Subscriber Fee Model!B48": "1%",
                            "Summary Scenarios!B4": "Upside"},
                     watch=["Summary Scenarios!G57"])
    check("both changes are reported", len(both["changes"]) == 2,
          ", ".join(f'{c["cell"]} {c["now_s"]}' for c in both["changes"]))
    g57 = {o["cell"]: o for o in both["outputs"]}.get("Summary Scenarios!G57")
    check("the outputs follow both",
          g57 is not None and close(g57["now"], 13415599.48, 1.0), str(g57 and g57["now"]))
    combos = [g for g in grid if g.get("cells")]
    check("the grid carries pairwise combinations of the top levers", len(combos) >= 20,
          f"{len(combos)} of {len(grid)} points")

    head("GOAL SEEK   Summary Scenarios!G57 = $10M")
    goal = ix.goalseek(None, "Summary Scenarios!G57", 10_000_000, "Subscriber Fee Model!B48")
    print(f'  {goal.get("value_s")} in {goal.get("tries")} tries, {goal.get("ms")} ms')
    check("goalseek reaches the target", not goal.get("error")
          and close(goal.get("achieved"), 10_000_000, 10_000.0), str(goal)[:90])
    check("the fee it names is between 1% and 1.1%",
          0.010 < (goal.get("value") or 0) < 0.011, str(goal.get("value")))
    back = ix.whatif(None, {"Subscriber Fee Model!B48": goal["value"]},
                     watch=["Summary Scenarios!G57"])
    check("running the model at that fee reproduces the target",
          close(back["outputs"][0]["now"], goal["achieved"], 1.0),
          f'{back["outputs"][0]["now"]} vs {goal["achieved"]}')
    far = ix.goalseek(None, "Summary Scenarios!G57", 10_000_000_000, "Subscriber Fee Model!B48")
    check("an unreachable target says so, with the nearest it got",
          bool(far.get("error")) and "nearest" in far, str(far.get("error"))[:80])

    head("EXPLAIN   why Summary Scenarios!G57 moved")
    why = ix.explain(None, "Summary Scenarios!G57", sets={"Subscriber Fee Model!B48": "1%"})
    for hop in why["chain"]:
        print(f'    {hop["cell"]:<28} {hop["label"][:26]:<26} {hop["was_s"]:>12} -> {hop["now_s"]}')
    check("the chain runs from the lever to the cell",
          why["chain"] and why["chain"][0]["cell"] == "Subscriber Fee Model!B48"
          and why["chain"][-1]["cell"] == "Summary Scenarios!G57", f'{why["hops"]} hops')
    check("it is three to six hops", 3 <= why["hops"] <= 6, str(why["hops"]))
    check("every hop carries a formula and both values",
          all(h["formula"] or h is why["chain"][0] for h in why["chain"])
          and all(h["now"] is not None for h in why["chain"])) 
    check("the last hop is the move the client saw",
          close(why["chain"][-1]["now"], 9586292.44), str(why["chain"][-1]["now"]))
    made = ix.explain(None, "Summary Scenarios!G57")
    check("with no what-if it is what the number is made of",
          made["hops"] >= 3 and made["chain"][-1]["cell"] == "Summary Scenarios!G57",
          " <- ".join(h["cell"] for h in made["chain"]))

    head("COMPARE + SWEEP")
    cmp_ = ix.compare(None, [{"Summary Scenarios!B5": "Premium"},
                             {"Summary Scenarios!B5": "Market"}],
                      watch=["Summary Scenarios!C362"])
    row = next(r for r in cmp_["rows"] if r["cell"] == "Summary Scenarios!C362")
    check("compare runs both cases in one call",
          close(row["values"][0], 8148348.575) and close(row["values"][1], 2875887.732),
          f'{row["values_s"]}')
    check("it reports the spread", close(row["spread"], 5272460.843, 1.0),
          str(row["spread"]))
    sw = ix.sweep(None, "Subscriber Fee Model!B48", ["1%", "1.25%", "1.5%"],
                  watch=["Summary Scenarios!G57"])
    srow = next(r for r in sw["rows"] if r["cell"] == "Summary Scenarios!G57")
    check("sweep runs one lever at three values", len(sw["points"]) == 3
          and close(srow["values"][0], 9586292.44)
          and close(srow["values"][2], 14379438.66, 1.0), str(srow["values_s"]))

    # ----------------------------------------------------------- the deep pass
    head("DEEP PASS  (canned model client: the table must work without aliases)")
    (ix._dir / "deep" / "scratch.json").unlink(missing_ok=True)   # make the unit run again
    llm_json, calls = dry_llm_json()
    t0 = time.perf_counter()
    deep = asyncio.run(ix.deep_build(llm_json, model_fast="dry", model_good="dry"))
    print(f"  {deep.line()}")
    print(f"  {calls['n']} model calls, {time.perf_counter() - t0:.1f} s")
    check("deep stats carry the lever and grid counts",
          deep.levers == len(levers) and deep.grid == len(grid),
          f"levers={deep.levers} grid={deep.grid}")
    cached = json.loads((ix._dir / "deep" / "scratch.json").read_text())
    check("scratch.json is cached by file hash and table schema",
          len(cached) == 1
          and list(cached)[0] == f"{ix._file_key(FILE)}:{_scratch.SCHEMA}", str(list(cached)))

    derived = [q for q in ix._qa if q.derived]
    # one grid point can make several entries (one per spoken name for the lever), so the
    # count is a coverage check: every point with a number worth saying is in there once
    by_ref = {_scratch.canon(lv["sheet"], lv["cell"]): lv for lv in levers}

    def sayable(point: dict) -> bool:
        if point.get("cells"):
            outs = list({o["cell"]: o for c in point["cells"]
                         for o in (by_ref.get(c) or {}).get("outputs", [])}.values())
            top, move = _scratch._headline(outs, point["outputs"])
            return top is not None and move >= 0.001
        lever = by_ref.get(point["cell"])
        return bool(lever and lever["outputs"]
                    and _scratch._headline(lever["outputs"], point["outputs"])[0])

    want = sum(1 for g in grid if sayable(g))
    check("every grid point with something to say became a derived Q&A entry",
          len({q.working for q in derived}) == want and len(derived) >= want,
          f"{len(derived)} entries over {want} of {len(grid)} points")
    if derived:
        entry = next((q for q in derived
                      if q.working.startswith("Subscriber Fee Model!B48")), derived[0])
        print(f"    Q: {entry.question}")
        print(f"    A: {entry.answer}")
        print(f"    say: {entry.say!r}  figure: {entry.figure!r}")
        print(f"    working: {entry.working}")
        print(f"    cite: {entry.citations}")
        check("a derived entry carries working and two citations",
              bool(entry.working) and len(entry.citations) == 2)
        check("a derived answer is at most 25 words", len(entry.answer.split()) <= 25,
              f"{len(entry.answer.split())} words")
        check("a derived say-line is 4 to 8 words", 3 <= len(entry.say.split()) <= 8,
              f"{len(entry.say.split())} words")

    reopened = Index(folder)
    again = [q for q in reopened._qa if q.derived]
    check("a reopened index still has the derived entries", len(again) == len(derived),
          f"{len(again)} entries")
    if again:
        hits = reopened.qa_matches(again[0].question, 3)
        if not hits:
            print("    qa_matches skipped: the embedding model is not available here")
        else:
            print(f"    qa_matches -> {hits[0].question!r} ({hits[0].score:.3f}, "
                  f"derived={hits[0].derived}, working={hits[0].working!r})")
            check("qa_matches returns the derived entry with its working line",
                  hits[0].derived and bool(hits[0].working), hits[0].question)
        spoken = reopened.qa_matches("what if we charged one percent of GMV", 3)
        for h in spoken[:2]:
            print(f"    spoken   -> {h.score:.3f} {h.question!r} derived={h.derived}")

    t0 = time.perf_counter()
    deep2 = asyncio.run(reopened.deep_build(llm_json, model_fast="dry", model_good="dry"))
    check("a second deep pass finds everything cached", deep2.total == 0,
          f"{deep2.total} units to do, {(time.perf_counter() - t0) * 1000:.0f} ms")

    # ------------------------------------------------------- the aliases call
    head("SPOKEN ALIASES  (one fast-model call per workbook)")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  skipped: ANTHROPIC_API_KEY is not set (the lever table works without it)")
    else:
        try:
            raw = asyncio.run(real_llm_json()(
                _scratch.aliases_prompt(FILE, levers), _scratch.ALIASES_SCHEMA, None))
            n = _scratch.apply_aliases(levers, raw)
            check("the aliases call names levers the client would say", n > 0,
                  f"{n} of {len(levers)} levers named")
            for lever in [lv for lv in levers if lv["aliases"]][:4]:
                print(f'    {lever["sheet"]}!{lever["cell"]:5s} {lever["label"][:40]:<42} '
                      f'{lever["aliases"]}')
        except Exception as e:  # noqa: BLE001
            text = str(e).lower()
            if "credit balance" in text or "rate_limit" in text or "quota" in text:
                print(f"  skipped: the account cannot call the API right now ({type(e).__name__})")
            else:
                check("the aliases call", False, f"{type(e).__name__}: {e}")

    head(f"scratch smoke: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
