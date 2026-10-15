"""Core ML export of the gate, one compiled package per sequence-length bucket.

    python -m presage.llm.laya_fast.coreml --export            # writes ~/.cache/presage/laya/fast/
    python -m presage.llm.laya_fast.coreml --bench --units CPU_AND_GPU

**Fixed shapes, not flexible ones, and this is the whole finding of the route.** One package
carrying every length as `EnumeratedShapes` is the tidy answer -- one copy of the weights on disk
instead of one per length -- and it is 16x slower: 776 ms p50 against 47 ms for the same weights at
a fixed 224, with the runtime printing `tensor_buffer has known strides while the model has
FlexibleShapeInfo` on every call. `RangeDim` is the same story. So each bucket is its own package
and its own 843 MB, and the bucket list is a disk-space decision: three of them is 2.5 GB.

`coremltools` is a build *and* run dependency here (`pyproject.toml`, extra `laya-fast`), but a
much smaller one than the path it replaces: about 60 MB against torch and transformers' 610.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BUCKETS = (224, 384, 640)
CACHE = Path.home() / ".cache" / "presage" / "laya" / "fast"


def package_for(out_dir: Path, L: int | None = None) -> Path:
    """One package per weight set, not one per length: a separate `.mlpackage` per bucket is a
    separate 843 MB copy of the encoder, and five of those is 4 GB on disk for one model."""
    return out_dir / (f"gate_b3_L{L}.mlpackage" if L else "gate_b3.mlpackage")


def compiled_for(pkg: Path) -> Path:
    """The package compiled once, beside it, and recompiled only when a re-export is newer.

    Loading the `.mlpackage` itself compiles a fresh copy into $TMPDIR on every load and never
    removes it, and every new path also gets its own GPU bundle in
    ~/Library/Caches/<python>/com.apple.e5rt.e5bundlecache. Two days of server restarts and benches
    left 164 compiled copies and 878 bundles behind: 557 GB. A fixed path compiles once and lets
    the runtime reuse its bundle."""
    import os
    import shutil

    import coremltools as ct

    out = pkg.with_suffix(".mlmodelc")
    if out.exists() and out.stat().st_mtime >= (pkg / "Manifest.json").stat().st_mtime:
        return out
    tmp = pkg.with_name(f"{pkg.stem}.{os.getpid()}.mlmodelc")
    shutil.rmtree(tmp, ignore_errors=True)
    ct.utils.compile_model(str(pkg), str(tmp))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


# ----------------------------------------------------------------------------- build
def export(model_dir: str | Path, buckets: tuple[int, ...] | list[int], out_dir: Path,
           batch: int = 3, k: int = 5, precision: str = "fp16") -> list[Path]:
    """One fixed-shape package per bucket. See the module docstring for why not one flexible one."""
    import coremltools as ct
    import torch

    from presage.llm.laya_fast.encoder import load_gate, tables

    buckets = sorted(buckets)
    g, cfg, _ = load_gate(model_dir, torch.float32, batch)
    d, names = g.scorer[1].in_features, g.table_names
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for L in buckets:
        tabs = tables(g.c, L, torch.float32)
        args = (torch.zeros(batch, L, dtype=torch.long),
                torch.zeros(batch, 1, 1, L, dtype=torch.float32),
                torch.zeros(batch, d, dtype=torch.float32),
                torch.zeros(batch, k, L, dtype=torch.float32),
                *(tabs[n] for n in names))
        with torch.no_grad():
            traced = torch.jit.trace(g, args, check_trace=False)
        ins = [ct.TensorType(name="input_ids", shape=(batch, L), dtype=np.int32),
               ct.TensorType(name="pad_bias", shape=(batch, 1, 1, L), dtype=np.float16),
               ct.TensorType(name="type_vec", shape=(batch, d), dtype=np.float16),
               ct.TensorType(name="marker_map", shape=(batch, k, L), dtype=np.float16)]
        ins += [ct.TensorType(name=n, shape=tuple(tabs[n].shape), dtype=np.float16) for n in names]
        m = ct.convert(
            traced, inputs=ins,
            outputs=[ct.TensorType(name="logits", dtype=np.float32)],
            convert_to="mlprogram",
            compute_precision={"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[precision],
            minimum_deployment_target=ct.target.macOS15,
        )
        m.save(str(package_for(out_dir, L)))
        made.append(package_for(out_dir, L))
        (out_dir / "meta.json").write_text(json.dumps(
            {"model_dir": str(model_dir), "buckets": buckets, "batch": batch, "k": k,
             "precision": precision, "hidden": d, "tables": names,
             "encoder": json.loads((Path(model_dir).expanduser() / "encoder" / "config.json").read_text()),
             "head_max_len": cfg["head_max_len"], "max_len": cfg["max_len"]}, indent=1))
    return made


# ----------------------------------------------------------------------------- run
class CoreMLGate:
    """One compiled package, every bucket. `run` picks the smallest bucket the tokens fit in."""

    def __init__(self, out_dir: Path = CACHE, units: str = "ALL") -> None:
        from presage.llm.laya_fast.runtime import tables_np
        self.dir = Path(out_dir).expanduser()
        self.units = units
        self.meta = json.loads((self.dir / "meta.json").read_text()) if (self.dir / "meta.json").exists() else {}
        self.buckets = tuple(self.meta.get("buckets") or ())
        self.names: list[str] = list(self.meta.get("tables") or [])
        self._tables = tables_np
        self.tabs: dict[int, dict[str, np.ndarray]] = {}
        self.models: dict[int, Any] = {}
        self.load_ms: dict[str, float] = {}

    def available(self) -> bool:
        return bool(self.buckets) and all(package_for(self.dir, L).exists() for L in self.buckets)

    def bucket(self, n: int) -> int:
        for b in self.buckets:
            if n <= b:
                return b
        raise ValueError(f"{n} tokens is past the largest exported bucket {self.buckets[-1]}")

    def tables(self, L: int) -> dict[str, np.ndarray]:
        t = self.tabs.get(L)
        if t is None:
            t = self._tables(self.meta["encoder"], L, np.float16)
            self.tabs[L] = t
        return t

    def load(self, L: int):
        m = self.models.get(L)
        if m is None:
            import coremltools as ct
            t = time.perf_counter()
            m = ct.models.CompiledMLModel(str(compiled_for(package_for(self.dir, L))),
                                          compute_units=getattr(ct.ComputeUnit, self.units))
            self.load_ms[f"L{L}"] = (time.perf_counter() - t) * 1e3
            self.models[L] = m
        return m

    def warm(self, L: int) -> None:
        b, k, d = self.meta.get("batch", 3), self.meta.get("k", 5), self.meta.get("hidden", 1024)
        self.run(np.zeros((b, L), np.int32), np.zeros((b, 1, 1, L), np.float16),
                 np.zeros((b, d), np.float16), np.zeros((b, k, L), np.float16))

    def run(self, input_ids: np.ndarray, pad_bias: np.ndarray, type_vec: np.ndarray,
            marker_map: np.ndarray) -> np.ndarray:
        L = input_ids.shape[1]
        feed = {"input_ids": input_ids.astype(np.int32), "pad_bias": pad_bias.astype(np.float16),
                "type_vec": type_vec.astype(np.float16), "marker_map": marker_map.astype(np.float16)}
        feed.update(self.tables(L))
        return np.asarray(self.load(L).predict(feed)["logits"], dtype=np.float32)


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from presage.llm.laya import weights_dir

    ap = argparse.ArgumentParser(prog="python -m presage.llm.laya_fast.coreml")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", type=Path, default=CACHE)
    ap.add_argument("--buckets", type=int, nargs="*", default=list(BUCKETS))
    ap.add_argument("--units", default="ALL", help="ALL | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    src = Path(a.model_dir).expanduser() if a.model_dir else weights_dir()
    if a.export:
        t = time.perf_counter()
        for pkg in export(src, a.buckets, a.out):
            sz = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1e6
            print(f"{pkg.name:24s} {sz:7.0f} MB  {time.perf_counter() - t:6.1f} s cumulative", flush=True)
    if a.bench:
        from presage.llm.laya_fast.runtime import gate_inputs
        import statistics
        g = CoreMLGate(a.out, a.units)
        print(f"buckets {g.buckets}")
        from presage.llm.laya import SAMPLES, LayaDecider
        dec = LayaDecider(device="cpu", dtype="fp32")
        dec.load()
        for L in g.buckets:
            ids, bias, tv, mm, ks = gate_inputs(dec, SAMPLES[0][1], L)
            t = time.perf_counter()
            g.warm(L)
            first = time.perf_counter() - t
            for _ in range(a.warmup):
                g.run(ids, bias, tv, mm)
            took = []
            for _ in range(a.runs):
                t = time.perf_counter()
                g.run(ids, bias, tv, mm)
                took.append((time.perf_counter() - t) * 1e3)
            print(f"  L={L:4d} units={a.units:11s} p50 {statistics.median(took):7.2f}  "
                  f"p95 {sorted(took)[int(.95 * len(took))]:7.2f}  "
                  f"load {g.load_ms.get('mlmodel', 0) / 1e3:.1f}s + first {first:.1f}s", flush=True)


if __name__ == "__main__":
    main()
