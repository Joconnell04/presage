"""ONNX Runtime: the same graph, the CoreML and CPU execution providers, and an INT8 CPU build.

    python -m presage.llm.laya_fast.onnx_rt --export --buckets 224
    python -m presage.llm.laya_fast.onnx_rt --bench --provider CPUExecutionProvider

presage already ships `onnxruntime` for the STT path, so this route costs no new dependency at run
time. `onnx` and `onnxscript` are needed only to write the file.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "presage" / "laya" / "fast-onnx"


def model_path(out_dir: Path, L: int, tag: str = "") -> Path:
    return out_dir / f"gate_b3_L{L}{tag}.onnx"


# ----------------------------------------------------------------------------- build
def export(model_dir: str | Path, L: int, out_dir: Path, batch: int = 3, k: int = 5,
           fp16: bool = False) -> Path:
    import torch

    from presage.llm.laya_fast.encoder import load_gate, tables

    g, cfg, _ = load_gate(model_dir, torch.float32, batch)
    d, names = g.scorer[1].in_features, g.table_names
    tabs = tables(g.c, L, torch.float32)
    args = (torch.zeros(batch, L, dtype=torch.long),
            torch.zeros(batch, 1, 1, L, dtype=torch.float32),
            torch.zeros(batch, d, dtype=torch.float32),
            torch.zeros(batch, k, L, dtype=torch.float32),
            *(tabs[n] for n in names))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = model_path(out_dir, L, "-fp16" if fp16 else "")
    torch.onnx.export(g, args, str(p),
                      input_names=["input_ids", "pad_bias", "type_vec", "marker_map", *names],
                      output_names=["logits"], opset_version=17, dynamo=False)
    (out_dir / "meta.json").write_text(json.dumps(
        {"model_dir": str(model_dir), "batch": batch, "k": k, "hidden": d, "tables": names,
         "encoder": json.loads((Path(model_dir).expanduser() / "encoder" / "config.json").read_text()),
         "buckets": sorted({L, *(int(q.stem.split("_L")[1].split("-")[0])
                                 for q in out_dir.glob("gate_b3_L*.onnx"))})}, indent=1))
    return p


def quantize(src: Path) -> Path:
    """Dynamic INT8 over the weights. Only the CPU provider does anything with it."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst = src.with_name(src.stem + "-int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


# ----------------------------------------------------------------------------- run
class OnnxGate:
    def __init__(self, out_dir: Path = CACHE, provider: str = "CPUExecutionProvider",
                 tag: str = "", threads: int = 0) -> None:
        self.dir, self.provider, self.tag, self.threads = Path(out_dir).expanduser(), provider, tag, threads
        self.meta = json.loads((self.dir / "meta.json").read_text()) if (self.dir / "meta.json").exists() else {}
        self.buckets = tuple(self.meta.get("buckets") or ())
        self.sessions: dict[int, object] = {}
        self.load_ms: dict[int, float] = {}
        self.tabs: dict[int, dict] = {}

    def available(self) -> bool:
        return bool(self.buckets)

    def bucket(self, n: int) -> int:
        for b in self.buckets:
            if n <= b:
                return b
        raise ValueError(f"{n} tokens is past the largest bucket {self.buckets[-1]}")

    def session(self, L: int):
        s = self.sessions.get(L)
        if s is None:
            import onnxruntime as ort
            o = ort.SessionOptions()
            o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if self.threads:
                o.intra_op_num_threads = self.threads
            t = time.perf_counter()
            s = ort.InferenceSession(str(model_path(self.dir, L, self.tag)), o, providers=[self.provider])
            self.load_ms[L] = (time.perf_counter() - t) * 1e3
            self.sessions[L] = s
        return s

    def tables(self, L: int) -> dict:
        t = self.tabs.get(L)
        if t is None:
            from presage.llm.laya_fast.runtime import tables_np
            t = tables_np(self.meta["encoder"], L, np.float32)
            self.tabs[L] = t
        return t

    def run(self, ids, pad_bias, type_vec, marker_map) -> np.ndarray:
        L = ids.shape[1]
        feed = {"input_ids": ids.astype(np.int64), "pad_bias": pad_bias.astype(np.float32),
                "type_vec": type_vec.astype(np.float32), "marker_map": marker_map.astype(np.float32)}
        feed.update(self.tables(L))
        return np.asarray(self.session(L).run(["logits"], feed)[0], dtype=np.float32)


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    from presage.llm.laya import SAMPLES, LayaDecider, weights_dir
    from presage.llm.laya_fast.runtime import gate_inputs, percentiles

    ap = argparse.ArgumentParser(prog="python -m presage.llm.laya_fast.onnx_rt")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", type=Path, default=CACHE)
    ap.add_argument("--buckets", type=int, nargs="*", default=[224])
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--tag", default="")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    src = Path(a.model_dir).expanduser() if a.model_dir else weights_dir()
    if a.export:
        for L in a.buckets:
            t = time.perf_counter()
            p = export(src, L, a.out)
            print(f"L={L:4d}  {p.stat().st_size / 1e6:7.0f} MB  {time.perf_counter() - t:5.1f} s  {p}",
                  flush=True)
    if a.quantize:
        for L in a.buckets:
            t = time.perf_counter()
            p = quantize(model_path(a.out, L))
            print(f"int8 L={L:4d}  {p.stat().st_size / 1e6:7.0f} MB  {time.perf_counter() - t:5.1f} s",
                  flush=True)
    if a.bench:
        dec = LayaDecider(src, device="cpu", dtype="fp32").load()
        g = OnnxGate(a.out, a.provider, a.tag)
        for L in (a.buckets or g.buckets):
            ids, bias, tv, mm, ks = gate_inputs(dec, SAMPLES[0][1], L)
            for _ in range(a.warmup):
                g.run(ids, bias, tv, mm)
            took = []
            for _ in range(a.runs):
                t = time.perf_counter()
                g.run(ids, bias, tv, mm)
                took.append((time.perf_counter() - t) * 1e3)
            p50, p95 = percentiles(took)
            print(f"  L={L:4d} {a.provider:26s}{a.tag:6s} p50 {p50:8.2f}  p95 {p95:8.2f}  "
                  f"load {g.load_ms[L] / 1e3:.1f}s", flush=True)


if __name__ == "__main__":
    main()
