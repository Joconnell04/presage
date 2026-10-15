"""The E1 table: every route, measured the same way.

    python -m presage.llm.laya_fast.bench                       # the whole table
    python -m presage.llm.laya_fast.bench --routes torch coreml-gpu
    python -m presage.llm.laya_fast.bench --one coreml-gpu      # one route, one JSON line

One measurement, everywhere: `presage.llm.laya.GATE_QUESTIONS` (the three questions `brain.py`
asks) against `SAMPLES[0]` (a four-line transcript), 50 runs after 10 warm-ups, p50 and p95 of the
wall clock around the whole call including tokenizing and the softmax.

Every route runs in its own child process. That is the only way load time and peak RSS mean
anything: a second route in the same process inherits the first one's imports and its allocator,
and would report a load of nothing and an RSS that belongs to its predecessor.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROUTES = ("torch-mps-fp16", "torch-mps-bf16", "torch-cpu-fp32", "coreml-gpu", "coreml-ane",
          "coreml-all", "coreml-cpu", "onnx-cpu", "onnx-cpu-int8", "onnx-coreml", "mlx")
CACHE = Path.home() / ".cache" / "presage" / "laya"


# ----------------------------------------------------------------------------- one route
def run_one(route: str, model_dir: str | None, packages: str | None, onnx_dir: str | None,
            runs: int, warmup: int) -> dict:
    from presage.llm.laya import GATE_QUESTIONS, SAMPLES
    from presage.llm.laya_fast.runtime import percentiles, rss_mb

    state = SAMPLES[0][1]
    rss0 = rss_mb()
    t0 = time.perf_counter()
    call = None
    note = ""

    if route.startswith("torch"):
        from presage.llm.laya import LayaDecider
        _, dev, dt = route.split("-")
        dec = LayaDecider(model_dir, device=dev, dtype=dt).load()
        load_s = time.perf_counter() - t0
        tokens = int(dec.system_one(state, GATE_QUESTIONS)["usage"]["input_tokens"])

        def call():
            dec.decide_sync(state, GATE_QUESTIONS)

    elif route.startswith("coreml"):
        from presage.llm.laya_fast.decider import FastLayaDecider
        units = {"gpu": "CPU_AND_GPU", "ane": "CPU_AND_NE", "all": "ALL", "cpu": "CPU_ONLY"}[route.split("-")[1]]
        dec = FastLayaDecider(model_dir, packages, units)
        why = dec.why_not()
        if why:
            return {"route": route, "error": why}
        dec.load()
        dec.decide_sync(state, GATE_QUESTIONS)            # compile the plan before the clock stops
        load_s = time.perf_counter() - t0
        tokens = int(dec.system_one(state, GATE_QUESTIONS)["usage"]["input_tokens"])
        note = f"bucket {dec.last_bucket}, fallbacks {dec.fallbacks}"

        def call():
            dec.decide_sync(state, GATE_QUESTIONS)

    elif route.startswith("onnx"):
        from presage.llm.laya import LayaDecider
        from presage.llm.laya_fast.onnx_rt import CACHE as OC, OnnxGate
        from presage.llm.laya_fast.runtime import gate_inputs
        prov = "CoreMLExecutionProvider" if route.endswith("coreml") else "CPUExecutionProvider"
        tag = "-int8" if route.endswith("int8") else ""
        g = OnnxGate(Path(onnx_dir or OC), prov, tag)
        if not g.available():
            return {"route": route, "error": f"no onnx models in {onnx_dir or OC}"}
        ref = LayaDecider(model_dir, device="cpu", dtype="fp32").load()      # tokenizer + type table
        L = g.buckets[0]
        ids, bias, tv, mm, ks = gate_inputs(ref, state, L)
        g.run(ids, bias, tv, mm)
        load_s = time.perf_counter() - t0
        tokens = int(bias.shape[0] * bias.shape[-1])
        note = f"bucket {L}, forward only"

        def call():
            g.run(ids, bias, tv, mm)

    elif route == "mlx":
        try:
            import laya_mlx  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return {"route": route, "error": f"laya-mlx not installed: {type(e).__name__}"}
        from presage.llm.laya_fast.mlxrt import MlxGate
        g = MlxGate(model_dir)
        g.load()
        tokens = g.warm(state, GATE_QUESTIONS)   # mlx maps the weights lazily; the first call
        load_s = time.perf_counter() - t0        # is where they are actually read and cast

        def call():
            g.gate(state, GATE_QUESTIONS)
        note = g.note
    else:
        return {"route": route, "error": "unknown route"}

    for _ in range(warmup):
        call()
    took = []
    for _ in range(runs):
        t = time.perf_counter()
        call()
        took.append((time.perf_counter() - t) * 1e3)
    p50, p95 = percentiles(took)
    return {"route": route, "p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
            "min_ms": round(min(took), 1), "load_s": round(load_s, 1),
            "rss_mb": round(rss_mb(), 0), "rss_delta_mb": round(rss_mb() - rss0, 0),
            "tokens": tokens, "note": note}


# ----------------------------------------------------------------------------- the table
def child(route: str, a) -> dict:
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONWARNINGS="ignore")
    cmd = [sys.executable, "-u", "-m", "presage.llm.laya_fast.bench", "--one", route,
           "--runs", str(a.runs), "--warmup", str(a.warmup)]
    for flag, val in (("--model-dir", a.model_dir), ("--packages", a.packages), ("--onnx-dir", a.onnx_dir)):
        if val:
            cmd += [flag, str(val)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           cwd=str(Path(__file__).resolve().parents[3]), timeout=a.timeout)
    except subprocess.TimeoutExpired:
        return {"route": route, "error": f"timed out after {a.timeout:.0f} s"}
    for line in reversed(p.stdout.splitlines()):
        at = line.find('{"route"')          # the Core ML runtime prints without a trailing newline
        if at >= 0:
            return json.loads(line[at:])
    return {"route": route, "error": (p.stderr or p.stdout).strip().splitlines()[-1][:160]
            if (p.stderr or p.stdout).strip() else f"no output (rc {p.returncode})"}


def table(rows: list[dict]) -> str:
    head = f"{'route':16s} {'p50 ms':>8} {'p95 ms':>8} {'load s':>7} {'RSS MB':>7}  note"
    out = [head, "-" * len(head)]
    for r in rows:
        if r.get("error"):
            out.append(f"{r['route']:16s} {'-':>8} {'-':>8} {'-':>7} {'-':>7}  {r['error'][:70]}")
        else:
            out.append(f"{r['route']:16s} {r['p50_ms']:8.1f} {r['p95_ms']:8.1f} {r['load_s']:7.1f} "
                       f"{r['rss_mb']:7.0f}  {r['note']}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m presage.llm.laya_fast.bench")
    ap.add_argument("--routes", nargs="*", default=list(ROUTES))
    ap.add_argument("--one", default=None, help="run a single route here and print one JSON line")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--packages", default=None, help="Core ML package directory")
    ap.add_argument("--onnx-dir", default=None)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    if a.one:
        try:
            print(json.dumps(run_one(a.one, a.model_dir, a.packages, a.onnx_dir, a.runs, a.warmup)))
        except Exception as e:  # noqa: BLE001  a broken route is a row, not a crash
            print(json.dumps({"route": a.one, "error": f"{type(e).__name__}: {str(e)[:160]}"}))
        return

    rows = []
    for route in a.routes:
        print(f"... {route}", file=sys.stderr, flush=True)
        rows.append(child(route, a))
        print(json.dumps(rows[-1]), file=sys.stderr, flush=True)
    print("\n" + table(rows))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
