"""MLX route: the community port, pointed at presage's own fine-tuned weights.

    uv pip install laya-mlx
    python -m presage.llm.laya_fast.mlxrt --bench

`laya-mlx` (github.com/mizorewww/laya-mlx, PyPI `laya-mlx`) is a ModernBERT + decision-head port in
MLX that loads the published `typed-decisions` layout, which is the layout `data/gate/train_laya.py`
writes. So the fine-tuned gate loads into it with no conversion: `laya_mlx.load(weights_dir())`.
Its `system_one(state, questions)` returns Jev's wire body, the same shape `presage/llm/laya.py`
returns, so `jev.parse_answers` reads it unchanged.

E1 measured this and it is not the fast route on this Mac. MLX's fp16 matmul tops out at the same
throughput as PyTorch's MPS backend on the shapes this model uses; see the write-up.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

from presage.llm.jev import Decision, DecisionError, Q, parse_answers


class MlxGate:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        from presage.llm.laya import weights_dir
        self.model_dir = Path(model_dir).expanduser() if model_dir else weights_dir()
        self.agent: Any = None
        self.note = ""

    def load(self) -> "MlxGate":
        import laya_mlx
        self.agent = laya_mlx.load(str(self.model_dir))
        self.note = type(self.agent).__name__
        return self

    def body(self, state: str, questions: list[Q]) -> dict:
        """laya-mlx takes Jev's request shape: a mapping from question id to its definition."""
        qs: dict[str, dict] = {}
        for q in questions:
            d: dict = {"type": q.kind, "instructions": q.instructions}
            if q.kind == "choice":
                d["criteria"] = {o: None for o in (q.options or [])}
            elif q.kind == "score":
                d["criteria"] = q.levels()
            qs[q.name] = d
        return self.agent.system_one(state, qs)

    def gate(self, state: str, questions: list[Q]) -> dict:
        return parse_answers(questions, self.body(state, questions), 0)

    def warm(self, state: str, questions: list[Q]) -> int:
        b = self.body(state, questions)
        return int((b.get("usage") or {}).get("input_tokens") or 0)


class MlxDecider:
    """`LayaDecider`'s contract over `laya-mlx`. No shape envelope and no fallback: the MLX port
    takes any question list and any length, exactly as the torch path does."""

    provider = "laya"
    kind = "laya"

    def __init__(self, model_dir: str | Path | None = None, preload: bool = False) -> None:
        self.gate = MlxGate(model_dir)
        self.model_dir = self.gate.model_dir
        self.warm_ms = 0
        self._load_lock = __import__("threading").Lock()
        self._alock = None
        self._loader = None
        if preload:
            self.preload()

    def __repr__(self) -> str:
        return f"MlxDecider({self.model_dir}, {'loaded' if self.gate.agent else 'not loaded'})"

    def why_not(self) -> str | None:
        if not (Path(self.model_dir) / "model.safetensors").is_file():
            return f"no weights in {self.model_dir}"
        try:
            import laya_mlx  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return f"laya-mlx not installed ({type(e).__name__}): uv pip install laya-mlx"
        return None

    def preload(self) -> None:
        import threading
        if self.gate.agent is not None or (self._loader is not None and self._loader.is_alive()):
            return
        self._loader = threading.Thread(target=self._load_quiet, name="laya-mlx-load", daemon=True)
        self._loader.start()

    def _load_quiet(self) -> None:
        import logging
        try:
            self.load()
        except Exception as e:  # noqa: BLE001
            logging.getLogger("presage.llm.laya_fast").warning("mlx preload: %s", e)

    def load(self) -> "MlxDecider":
        with self._load_lock:
            if self.gate.agent is None:
                why = self.why_not()
                if why:
                    raise DecisionError(why)
                self.gate.load()
        return self

    def decide_sync(self, state: str, questions: list[Q]) -> dict[str, Decision]:
        if self.gate.agent is None:
            self.load()
        t0 = time.perf_counter()
        body = self.gate.body(state, questions)
        return parse_answers(questions, body, int((time.perf_counter() - t0) * 1e3))

    def _lock(self):
        import asyncio
        if self._alock is None:
            self._alock = asyncio.Lock()
        return self._alock

    async def decide(self, state: str, questions: list[Q]) -> dict[str, Decision]:
        import asyncio
        async with self._lock():
            return await asyncio.to_thread(self.decide_sync, state, questions)

    async def validate_key(self) -> tuple[bool, str]:
        import asyncio
        why = self.why_not()
        if why:
            return False, why[:140]
        async with self._lock():
            try:
                self.warm_ms = await asyncio.to_thread(self._warm_gate)
            except DecisionError as e:
                return False, str(e)[:140]
            except Exception as e:  # noqa: BLE001
                return False, f"{type(e).__name__}: {e}"[:140]
        return True, f"ok ({self.warm_ms} ms)"

    def _warm_gate(self) -> int:
        from presage.llm.laya import GATE_QUESTIONS, SAMPLES
        self.decide_sync(SAMPLES[0][1], GATE_QUESTIONS)
        return self.decide_sync(SAMPLES[0][1], GATE_QUESTIONS)["kind"].latency_ms

    async def warm(self) -> float | None:
        ok, _ = await self.validate_key()
        return float(self.warm_ms) if ok else None

    async def aclose(self) -> None:
        self.gate.agent = None


def main() -> None:
    from presage.llm.laya import GATE_QUESTIONS, SAMPLES
    from presage.llm.laya_fast.runtime import percentiles

    ap = argparse.ArgumentParser(prog="python -m presage.llm.laya_fast.mlxrt")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    t0 = time.perf_counter()
    g = MlxGate(a.model_dir).load()
    load_s = time.perf_counter() - t0
    state = SAMPLES[0][1]
    print(g.gate(state, GATE_QUESTIONS))
    for _ in range(a.warmup):
        g.gate(state, GATE_QUESTIONS)
    took = []
    for _ in range(a.runs):
        t = time.perf_counter()
        g.gate(state, GATE_QUESTIONS)
        took.append((time.perf_counter() - t) * 1e3)
    p50, p95 = percentiles(took)
    print(f"mlx  p50 {p50:.1f}  p95 {p95:.1f}  min {min(took):.1f} ms  load {load_s:.1f} s "
          f"(median of {len(took)}, {statistics.mean(took):.1f} mean)")


if __name__ == "__main__":
    main()
