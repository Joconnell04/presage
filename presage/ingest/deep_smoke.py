"""Deep-index smoke test against the real API.

    uv run python -m presage.ingest.deep_smoke sample_context [--force] [--dry]

Builds the fast index, runs `deep_build`, then prints deep_stats, the summaries block, the
definitions block, `find_numbers` and `qa_matches` for spoken-style questions, and the cold
vs cached wall time. `--dry` uses a canned llm_json so the shape can be checked for free.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from presage.ingest import Index

SPOKEN = [
    "so what did q3 look like",
    "how many heads by june",
    "is churn worse in enterprise",
    "what's the market size",
    "what are we asking the board for",
]


def make_llm_json(model_fast: str, model_good: str):
    """An LLMJson on the Anthropic client, mirroring llm.classify but returning `_usage`.

    The server passes `brain.llm.classify`, which has the same signature and no usage; the
    deep pass reads `_usage` when present and falls back to its own estimate when absent.
    """
    from presage.llm import make_llm

    llm = make_llm("anthropic")
    totals = {"calls": 0, "in": 0, "out": 0}

    async def llm_json(prompt: str, schema: dict, model: str | None = None) -> dict:
        resp = await llm.client.messages.create(
            model=model or model_fast,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}
        u = resp.usage
        totals["calls"] += 1
        totals["in"] += u.input_tokens
        totals["out"] += u.output_tokens
        data["_usage"] = {"in": u.input_tokens, "out": u.output_tokens}
        return data

    return llm_json, totals, llm


def dry_llm_json():
    async def llm_json(prompt: str, schema: dict, model: str | None = None) -> dict:
        props = schema.get("properties", {})
        if "contexts" in props:
            return {"contexts": []}
        if "qa" in props:
            return {"qa": []}
        if "definitions" in props:
            return {"definitions": []}
        return {"file": "(dry run)", "sections": []}
    return llm_json, {"calls": 0, "in": 0, "out": 0}, None


async def main(argv: list[str]) -> int:
    folder = Path(argv[1] if len(argv) > 1 else "sample_context").resolve()
    force = "--force" in argv
    dry = "--dry" in argv
    model_fast = os.environ.get("PRESAGE_DEEP_FAST", "claude-haiku-4-5-20251001")
    model_good = os.environ.get("PRESAGE_DEEP_GOOD", "claude-sonnet-5")

    print("=" * 78)
    print(f"FAST BUILD  {folder}")
    print("=" * 78)
    ix = Index()
    st = ix.build(folder)
    print(f"  {st}")

    if dry:
        llm_json, totals, llm = dry_llm_json()
    else:
        from dotenv import load_dotenv
        load_dotenv(folder.parent / ".env" if (folder.parent / ".env").exists() else ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  ANTHROPIC_API_KEY not set; rerun with --dry")
            return 1
        llm_json, totals, llm = make_llm_json(model_fast, model_good)

    print()
    print("=" * 78)
    print(f"DEEP BUILD   fast={model_fast}  good={model_good}  force={force}")
    print("=" * 78)
    seen = {"n": 0, "planned": False}

    def on_progress(d: dict) -> None:
        if d["state"] == "running" and not seen["planned"]:
            seen["planned"] = True
            print(f"  plan: {d['total']} units over {d['files']} files, "
                  f"est {d['est_in_tokens']:,} in + {d['est_out_tokens']:,} out tokens")
            for f, t in sorted(d["est_by_file"].items(), key=lambda kv: -kv[1]):
                print(f"      {f:<24} ~{t:,} tokens")
        if d["state"] == "running" and d["done"] > seen["n"]:
            seen["n"] = d["done"]
            if d["total"] and (d["done"] % 5 == 0 or d["done"] == d["total"]):
                print(f"  {d['done']:>3}/{d['total']} units")

    t0 = time.perf_counter()
    deep = await ix.deep_build(llm_json, on_progress=on_progress,
                               model_fast=model_fast, model_good=model_good, force=force)
    cold = time.perf_counter() - t0

    d = ix.deep_stats()
    print()
    print("  deep_stats():")
    for k in ("state", "done", "total", "qa", "summaries", "chunks_contextualized",
              "numbers", "definitions", "error", "started", "finished"):
        print(f"    {k:<24} {d[k]}")
    print(f"    {'llm_calls':<24} {d['llm_calls']}")
    print(f"    {'qa_rejected':<24} {d['qa_rejected']}  (figures that failed the digit check)")
    print(f"  cold wall: {cold:.1f} s")
    if totals["calls"]:
        cost = totals["in"] / 1e6 * 1.0 + totals["out"] / 1e6 * 5.0
        print(f"  actual usage: {totals['calls']} calls, {totals['in']:,} in + "
              f"{totals['out']:,} out tokens (est was {d['est_in_tokens']:,} + "
              f"{d['est_out_tokens']:,})")
        print(f"  rough cost at Haiku/Sonnet blended list: ${cost:.3f}")

    # cached run
    t0 = time.perf_counter()
    ix2 = Index()
    fast2 = ix2.build(folder)
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    deep2 = await ix2.deep_build(llm_json, model_fast=model_fast, model_good=model_good)
    cached = time.perf_counter() - t0
    print(f"  cached deep_build: {cached * 1000:.0f} ms ({deep2.total} units to do, "
          f"{deep2.llm_calls} model calls)")
    print(f"  fast build() with the deep cache present: {fast2.build_ms:.0f} ms "
          f"(wall {t_fast * 1000:.0f} ms)")
    t0 = time.perf_counter()
    Index(folder)
    print(f"  Index(folder) reopen: {(time.perf_counter() - t0) * 1000:.0f} ms")

    print()
    print("=" * 78)
    print(f"summaries()   {len(ix.summaries())} chars, ~{len(ix.summaries()) // 4} tokens")
    print("=" * 78)
    print(ix.summaries())

    print()
    print("=" * 78)
    print("DEFINITIONS  (inside schema_summary)")
    print("=" * 78)
    full = ix.schema_summary()
    i = full.find("== DEFINITIONS")
    print(full[i:] if i >= 0 else "  (none found)")
    print(f"\n  schema_summary(): {len(full)} chars, ~{len(full) // 4} tokens")

    print()
    print("=" * 78)
    print("numbers / find_numbers")
    print("=" * 78)
    n = ix.sql("SELECT COUNT(*) AS n FROM numbers")[0].get("n")
    print(f"  {n} numbers extracted from the documents")
    for pat in ("year over year", "retention", "churn"):
        rows = ix.find_numbers(pat, limit=3)
        print(f"\n  find_numbers({pat!r}) -> {len(rows)}")
        for r in rows:
            if "error" in r:
                print(f"    {r}")
                continue
            print(f"    {r['value_text']:>12} {r['unit']:<4} {r['file']} › {r['loc']}")
            print(f"                      {r['sentence'][:88]}")

    print()
    print("=" * 78)
    print(f"qa_matches   ({len(ix._qa)} cached questions)")
    print("=" * 78)
    times = []
    for q in SPOKEN:
        ix.qa_matches(q, 3)
        t0 = time.perf_counter()
        hits = ix.qa_matches(q, 3)
        times.append((time.perf_counter() - t0) * 1000)
        print(f"\n  {q!r}   [{times[-1]:.3f} ms]")
        for h in hits:
            cites = "; ".join(f'{c["file"]} › {c["loc"]}' for c in h.citations)
            print(f"    {h.score:.3f}  Q: {h.question}")
            print(f"           A: {h.answer}")
            print(f"           say: {h.say!r}  figure: {h.figure!r}")
            print(f"           cite: {cites}")
    if times:
        print(f"\n  qa_matches worst: {max(times):.3f} ms (target < 1 ms)")

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
