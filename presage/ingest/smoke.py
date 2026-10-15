"""Smoke test:  python -m presage.ingest.smoke sample_context [--rebuild]

Builds the index, prints the schema summary the model would be given, then runs
the four SQL queries, three text searches and six topic matches the real product
does on a call, with timings. Nothing here is a unit test; it is the one command
that says "ingest still works".
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from . import Index

SQL_QUERIES = [
    ("Q3 FY25 revenue",
     "SELECT line_item, fy25_actual_q3 FROM p_and_l WHERE line_item = 'Revenue'"),
    ("FY26 plan EBITDA",
     "SELECT line_item, fy26_plan FROM p_and_l WHERE line_item = 'EBITDA'"),
    ("Headcount total by month",
     "SELECT department, jan, jun FROM headcount WHERE _is_total = 1"),
    ("Churned ARR by segment",
     "SELECT segment, COUNT(*) AS logos, SUM(arr_usd) AS churned_arr "
     "FROM customers WHERE churned = 1 GROUP BY segment ORDER BY churned_arr DESC"),
]

FALLBACK_QUERIES = [
    ("cells fallback: Q3 revenue with a citation",
     "SELECT file, sheet, cell_ref, value_num FROM cells "
     "WHERE row_label LIKE '%revenue%' AND header LIKE '%Q3%' AND value_num IS NOT NULL"),
    ("cells fallback: uncached formulas",
     "SELECT sheet, cell_ref, row_label, formula FROM cells "
     "WHERE is_formula = 1 AND value_num IS NULL AND value_text IS NULL LIMIT 3"),
]

GUARD_QUERIES = [
    "DROP TABLE p_and_l",
    "SELECT 1; DELETE FROM cells",
    "SELECT * FROM no_such_table",
]

SEARCHES = [
    "why is enterprise churn getting worse",
    "what does the market report say about net revenue retention",
    "what are we asking the board to approve",
]

UTTERANCES = [
    "so what did revenue look like in q3",
    "walk me through the headcount plan",
    "is churn getting worse in enterprise",
    "remind me what the fy26 plan has for ebitda",
    "how much of that is apac",
    "what was cost of goods sold last year",
]


def timed(fn, *a, repeat=25, **kw):
    fn(*a, **kw)                                   # warm
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn(*a, **kw)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return out, statistics.median(ts), ts[min(len(ts) - 1, int(0.95 * len(ts)))]


def fmt(v):
    if isinstance(v, float):
        return f"{int(v):,}" if v == int(v) else f"{v:,.2f}"
    return str(v)


def main(argv: list[str]) -> int:
    folder = Path(argv[1] if len(argv) > 1 else "sample_context").resolve()
    rebuild = "--rebuild" in argv
    if not folder.exists():
        print(f"no such folder: {folder}")
        return 1

    print("=" * 78)
    print(f"BUILD  {folder}")
    print("=" * 78)
    t = time.perf_counter()
    ix = Index()
    st = ix.build(folder, force=rebuild)
    print(f"  {st}")
    print(f"  rebuilt: {', '.join(st.rebuilt) or '(nothing changed)'}")
    if st.skipped:
        print(f"  skipped (not indexed): {', '.join(st.skipped)}")
    if st.errors:
        print(f"  ERRORS: {st.errors}")
    print(f"  wall: {(time.perf_counter() - t) * 1000:.0f} ms")

    t = time.perf_counter()
    ix2 = Index()
    st2 = ix2.build(folder)
    print(f"  incremental no-op rebuild: {st2.build_ms:.0f} ms "
          f"(wall {(time.perf_counter() - t) * 1000:.0f} ms)")

    print()
    print("=" * 78)
    print("SCHEMA SUMMARY  (this text goes into the cached system prompt)")
    print("=" * 78)
    summary, med, p95 = timed(ix.schema_summary, repeat=5)
    print(summary)
    print()
    print(f"  -> {len(summary):,} chars, ~{len(summary) // 4:,} tokens, "
          f"built in {med:.1f} ms")

    print()
    print("=" * 78)
    print("SQL  (read-only; numbers only ever come from here)")
    print("=" * 78)
    for label, q in SQL_QUERIES + FALLBACK_QUERIES:
        rows, med, p95 = timed(ix.sql, q)
        print(f"\n  {label}   [{med:.2f} ms median, {p95:.2f} ms p95]")
        print(f"    {q}")
        for r in rows[:6]:
            print("      " + "  ".join(f"{k}={fmt(v)}" for k, v in r.items()))

    print("\n  read-only guard:")
    for q in GUARD_QUERIES:
        print(f"    {q!r:42s} -> {ix.sql(q)[0]}")

    print()
    print("=" * 78)
    print(f"TEXT SEARCH  ({len(ix.chunks)} chunks, dense={'on' if st.dense else 'off'})")
    print("=" * 78)
    for q in SEARCHES:
        hits, med, p95 = timed(ix.search_text, q, 4)
        print(f"\n  {q!r}   [{med:.2f} ms median, {p95:.2f} ms p95]")
        for c in hits:
            print(f"    {c.score:5.3f}  {c.cite()}")
            print(f"           {c.text[:110].replace(chr(10), ' ')}...")

    print()
    print("=" * 78)
    print(f"TOPIC MATCH  ({st.topics} topics; runs on every transcript line)")
    print("=" * 78)
    alltimes = []
    for u in UTTERANCES:
        hits, med, p95 = timed(ix.match_topics, u, repeat=200)
        alltimes.append(med)
        print(f"\n  {u!r}   [{med:.3f} ms median, {p95:.3f} ms p95]")
        for h in hits[:5]:
            tg = h.targets[0]
            if tg["kind"] == "column":
                what = f'{tg["table"]}.{tg["column"]} (col {tg["letter"]})'
            elif tg["kind"] == "row":
                what = f'{tg["table"]} row {tg["row"]} @ {tg["cell_ref"]}'
            elif tg["kind"] == "value":
                what = f'{tg["table"]}.{tg["column"]} = {tg["value"]!r}'
            elif tg["kind"] == "chunk":
                what = f'{tg["file"]} chunks {tg["chunk_ids"]}'
            else:
                what = f'table {tg["table"]}'
            extra = f" +{len(h.targets) - 1} more" if len(h.targets) > 1 else ""
            print(f"    {h.score:5.2f}  {h.kind:<9} {h.topic[:40]:<42} -> {what}{extra}")
    print(f"\n  worst median across the six utterances: {max(alltimes):.3f} ms (target < 5 ms)")

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
