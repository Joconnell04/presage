"""Accuracy for every route, scored with `presage/llm/gate_eval.py`'s own metrics.

    python -m presage.llm.laya_fast.evaluate --route coreml --split data/gate/test.jsonl
    python -m presage.llm.laya_fast.evaluate --route coreml --lines

Two tests, both from `data/gate/RESULTS.md`:

  the split   the labelled rows, scored by the same `report()` the reference gate is scored by, so
              the numbers sit in the same table as the ones already published.
  the lines   the ten hand-written transcript lines in `docs/research-laya.md` section 5. Nothing
              derived from them ever reached an optimiser, which makes them the honest test.

A route that changes only the runtime (Core ML, ONNX) has to match the torch numbers to the
decimal; a route that changes the weights (the distilled student) has to stay within a point.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# a JSON list of gate cases; not shipped with the repo (it was built from real calls)
CASES = Path(os.environ.get("PRESAGE_LAYA_CASES", "~/.cache/presage/laya-cases.json")).expanduser()


def make(route: str, model_dir: str | None, packages: str | None, units: str):
    from presage.llm.laya import LayaDecider
    from presage.llm.laya_fast.decider import FastLayaDecider
    if route == "torch":
        return LayaDecider(model_dir)
    if route == "coreml":
        return FastLayaDecider(model_dir, packages, units)
    if route == "mlx":
        from presage.llm.laya_fast.mlxrt import MlxDecider
        return MlxDecider(model_dir)
    raise SystemExit(f"unknown route {route!r}")


async def lines(dec, path: Path) -> dict:
    """The ten held-out lines. Returns the three agreement counts."""
    from presage.llm.jev import Q
    d = json.loads(path.read_text())
    qs = [Q(q["name"], q["kind"], q["instructions"], q.get("options")) for q in d["questions"]]
    hits = {"kind": 0, "needs_lookup": 0, "complete": 0}
    rows = []
    for c in d["cases"]:
        out = await dec.decide(c["state"], qs)
        got = {"kind": str(out["kind"].value), "needs_lookup": bool(out["needs_lookup"].value),
               "complete": bool(out["complete"].value)}
        for k in hits:
            hits[k] += int(got[k] == c["expect"][k])
        rows.append({"id": c["id"], "line": c["last_line"][:60], "expect": c["expect"], "got": got,
                     "p": {k: round(float(out[k].p), 3) for k in hits}})
    for r in rows:
        bad = [k for k in hits if r["got"][k] != r["expect"][k]]
        print(f"  {r['id']:22s} {r['line'][:52]:54s} "
              + "  ".join(f"{k}={r['got'][k]}" + ("*" if k in bad else "") for k in hits))
    print(f"  agreement  kind {hits['kind']}/10   needs_lookup {hits['needs_lookup']}/10   "
          f"complete {hits['complete']}/10")
    return {"hits": hits, "rows": rows}


async def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m presage.llm.laya_fast.evaluate")
    ap.add_argument("--route", default="coreml", help="torch | coreml | mlx")
    ap.add_argument("--split", default=None, help="path to a jsonl split")
    ap.add_argument("--lines", action="store_true", help="the ten held-out lines")
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--packages", default=None)
    ap.add_argument("--units", default="CPU_AND_GPU")
    ap.add_argument("--summary", type=Path, default=None)
    a = ap.parse_args()

    from presage.llm import gate_eval as G

    dec = make(a.route, a.model_dir, a.packages, a.units)
    tag = f"{a.route}" + (f"/{a.units}" if a.route == "coreml" else "")
    out: dict = {"route": tag, "model_dir": str(getattr(dec, "model_dir", ""))}

    if a.lines:
        print(f"\nten held-out lines, {tag}")
        out["lines"] = (await lines(dec, a.cases))["hits"]

    if a.split:
        path = Path(a.split)
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        if a.limit:
            rows = rows[:a.limit]
        print(f"\n{tag} over {len(rows)} rows of {path.name}", flush=True)
        t0 = time.time()
        res, errors = await G.run(dec, rows, 1)
        res.sort(key=lambda r: str(r["id"]))
        out["split"] = G.report(tag, path.stem, res, errors, time.time() - t0)
        out["fallbacks"] = getattr(dec, "fallbacks", 0)

    close = getattr(dec, "aclose", None)
    if close:
        await close()
    if a.summary:
        a.summary.write_text(json.dumps(out, indent=1, default=float))
        print(f"\n-> {a.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
