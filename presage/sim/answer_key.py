"""Write an answer key for a workbook folder from the workbook itself.

    uv run python -m presage.sim.answer_key docs/replay/sim/big

The sample folder ships with a hand-written `ANSWER_KEY.md` (it was generated with the
data). A real client folder has none, and typing one is how a wrong figure gets into a
benchmark, so this builds it from the index and checks every number twice:

  1. the `cells` table gives the cached value, its row label, its column header and its
     formula (presage's own fast index, the same one the server answers from);
  2. pycel recompiles the workbook and re-evaluates the cell; a figure whose recomputed
     value differs from the cached one is dropped, not reported.

Only unambiguous cells survive: a label that carries two different values in the workbook
would make a spoken question that has two right answers, so it is dropped as well.

Writes `<folder>/ANSWER_KEY.md` (human) and `<folder>/ANSWER_KEY.json` (what
`presage.sim.scripts` reads). Both are skipped by the indexer, so neither can leak into an
answer: `presage/ingest/__init__.py` skips any file whose name starts with `ANSWER_KEY`.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from presage.ingest import Index
from presage.ingest.scratch import ModelGraph, Sheet, fmt_kind, fmt_value, number_formats

SKIP_SHEETS = re.compile(r"sources|appendix|read\s*me|glossary|notes", re.I)
SKIP_HEADERS = re.compile(r"^(rank|notes?|line|source)$", re.I)
MAX_PER_SHEET = 5
MAX_ENTRIES = 70
MAX_LABEL_WORDS = 9
REL_TOL = 1e-6           # pycel vs the cached value

# a client asks about the outcome sheets first; the cap is spread over sheets in this order
# so a 22-sheet workbook does not spend the whole key on whatever sorts first
SHEET_ORDER = ("p&l", "summary recommendation", "summary scenarios", "valuation synthesis",
               "funding scenarios", "unit economics", "cost to build & tco",
               "cash flow & balance sheet", "subscription ramp", "consolidated revenue",
               "market ceiling", "operating costs & headcount", "assumptions")


def _sheet_rank(sheet: str) -> int:
    s = (sheet or "").strip().lower()
    return SHEET_ORDER.index(s) if s in SHEET_ORDER else len(SHEET_ORDER)


def _sig(x: float, digits: int = 6) -> float:
    if not x or not math.isfinite(x):
        return 0.0
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def _plain(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().rstrip(":")).strip()


def candidates(ix: Index, file: str) -> list[dict]:
    """Every labelled numeric cell that could carry a spoken question, ranked by sheet."""
    book = Sheet(ix.con, file)
    rows = [r for r in book.rows
            if r["value_num"] is not None and (r["row_label"] or "").strip()
            and not SKIP_SHEETS.search(r["sheet"] or "")]

    # A label+header pair that holds several different values in the workbook is a question
    # with several answers. Most are the same row copied onto a summary sheet, with one or
    # two unselected permutation blocks carrying the other value; so the pair survives only
    # when one value dominates, and only that value is keyed.
    values: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        header = book.local_header(r["sheet"], r["row"], r["col"])
        key = (_plain(r["row_label"]).lower(), header.lower())
        values.setdefault(key, []).append(_sig(float(r["value_num"])))
    dominant: dict[tuple[str, str], float | None] = {}
    for key, vs in values.items():
        top = max(set(vs), key=vs.count)
        dominant[key] = top if vs.count(top) / len(vs) >= 0.6 else None

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    per_sheet: dict[str, int] = {}
    for r in rows:
        label = _plain(r["row_label"])
        if not label or len(label) > 60 or len(label.split()) > MAX_LABEL_WORDS:
            continue
        value = float(r["value_num"])
        if value == 0:
            continue
        header = book.local_header(r["sheet"], r["row"], r["col"])
        if SKIP_HEADERS.match(header):
            continue
        key = (label.lower(), header.lower())
        if key in seen or dominant.get(key) is None or dominant[key] != _sig(value):
            continue
        if per_sheet.get(r["sheet"], 0) >= MAX_PER_SHEET:
            continue
        seen.add(key)
        per_sheet[r["sheet"]] = per_sheet.get(r["sheet"], 0) + 1
        out.append({"sheet": r["sheet"], "cell_ref": r["cell_ref"], "row_label": label,
                    "header": header, "value": value,
                    "is_formula": bool(r["is_formula"]), "formula": r["formula"] or ""})
    out.sort(key=lambda p: (_sheet_rank(p["sheet"]), p["sheet"], p["cell_ref"]))
    return out


def verify(path: Path, file: str, picks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Re-evaluate every pick with pycel. Returns (kept, dropped)."""
    graph = ModelGraph(path, file=file).compile()
    kept, dropped = [], []
    for p in picks:
        try:
            got = graph.value(p["sheet"], p["cell_ref"])
        except Exception as e:  # noqa: BLE001  a cell pycel cannot reach is not a key entry
            p["drop"] = f"pycel: {type(e).__name__}"
            dropped.append(p)
            continue
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            p["drop"] = f"pycel returned {got!r}"
            dropped.append(p)
            continue
        want = p["value"]
        if abs(float(got) - want) > REL_TOL * max(1.0, abs(want)):
            p["drop"] = f"pycel {got} != cached {want}"
            dropped.append(p)
            continue
        p["verified"] = "pycel" if p["is_formula"] else "literal+pycel"
        p["unstable"] = bool(graph.unstable(p["sheet"], p["cell_ref"]))
        kept.append(p)
    return [k for k in kept if not k["unstable"]], dropped + [k for k in kept if k["unstable"]]


def build(folder: Path) -> dict:
    ix = Index()
    stats = ix.build(folder)
    con = ix.con
    files = [r["file"] for r in con.execute(
        "SELECT DISTINCT file FROM cells ORDER BY file").fetchall()]
    if not files:
        raise SystemExit(f"no workbook cells indexed under {folder}")

    entries: list[dict] = []
    for file in files:
        picks = candidates(ix, file)
        formats = number_formats(folder / file, {(p["sheet"], p["cell_ref"]) for p in picks})
        kept, dropped = verify(folder / file, file, picks)
        print(f"{file}: {len(kept)} verified, {len(dropped)} dropped", file=sys.stderr)
        for p in kept:
            kind = fmt_kind(formats.get((p["sheet"], p["cell_ref"]), "General"), p["value"])
            ask = f"{p['row_label']} ({p['header']})" if p["header"] else p["row_label"]
            entries.append({
                "id": f"K{len(entries) + 1:02d}",
                "ask": ask,
                "figure": fmt_value(p["value"], kind),
                "value": p["value"],
                "unit": kind,
                "file": file,
                "loc": f"{p['sheet']} › {p['cell_ref']}",
                "citation": f"{file} › {p['sheet']} › {p['cell_ref']}",
                "formula": p["formula"][:120],
                "verified": p["verified"],
            })
        if len(entries) >= MAX_ENTRIES:
            break

    docs = [dict(r) for r in con.execute("SELECT file, kind, n_units FROM _documents").fetchall()]
    return {"folder": str(folder), "files": files, "documents": docs,
            "index": {"tables": stats.tables, "cells": stats.cells, "chunks": stats.chunks},
            "entries": entries[:MAX_ENTRIES]}


def write(folder: Path, key: dict) -> None:
    (folder / "ANSWER_KEY.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    lines = [
        "# ANSWER_KEY",
        "",
        f"Ground truth for `{folder.name}`, built by `presage.sim.answer_key` from the workbook,",
        "never typed by hand. Every figure is the cached cell value **and** pycel's own",
        "re-evaluation of that cell; a cell where the two disagree is not in this table. A label",
        "that carries two different values in the workbook is dropped too, so every question here",
        "has one right answer. The indexer skips any file whose name starts with `ANSWER_KEY`, so",
        "this file is never retrievable and cannot leak into an answer.",
        "",
        f"Index: {key['index']['tables']} tables, {key['index']['cells']} cells, "
        f"{key['index']['chunks']} document chunks.",
        "",
        "| id | ask | figure | citation | verified |",
        "| --- | --- | --- | --- | --- |",
    ]
    for e in key["entries"]:
        lines.append(f"| {e['id']} | {e['ask']} | {e['figure']} | {e['citation']} | {e['verified']} |")
    lines.append("")
    (folder / "ANSWER_KEY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    a = ap.parse_args()
    folder = a.folder.expanduser().resolve()
    key = build(folder)
    write(folder, key)
    print(f"{len(key['entries'])} entries -> {folder / 'ANSWER_KEY.md'}")


if __name__ == "__main__":
    main()
