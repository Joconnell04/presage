"""`FastLayaDecider`: `LayaDecider`'s contract, Core ML underneath.

    from presage.llm.laya_fast import FastLayaDecider
    dec = FastLayaDecider(preload=True)
    out = await dec.decide(state, questions)      # {"kind": Decision(...), ...}

Same `decide` / `decide_sync` / `validate_key` / `aclose` / `kind = "laya"` as
`presage.llm.laya.LayaDecider`, so `make_decider` can hand this back instead with no other change.

What it is doing differently:

  runtime     the exported Core ML package rather than torch on MPS. Measured on this M5, the same
              weights and the same answers come back in 47 ms instead of 75 (E1).
  load        no torch, no transformers, no `AutoModel.from_config`: a tokenizer, one tensor read
              out of the safetensors file, and a compiled model. Seconds, not twenty seconds.
  shapes      Core ML wants static shapes, so the export is batch 3, five option slots, and a
              handful of sequence lengths. Anything else -- a different question count, a longer
              state, a `score` question with six levels -- falls back to `LayaDecider`, which is
              loaded lazily and only if something actually asks for it.

The fallback is the reason this is safe to wire in: the fast path answers the gate presage actually
asks, and every other shape still reaches the reference implementation.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from presage.llm.jev import Decision, DecisionError, Q, parse_answers
from presage.llm.laya import QTYPES, load_config, weights_dir
from presage.llm.laya_fast import runtime as R
from presage.llm.laya_fast.coreml import CACHE, CoreMLGate

log = logging.getLogger("presage.llm.laya_fast")


def type_table(model_dir: Path) -> np.ndarray:
    """`type_emb.weight` (3, d) without reading the other 842 MB."""
    from safetensors import safe_open
    with safe_open(str(model_dir / "model.safetensors"), framework="np") as f:
        return np.asarray(f.get_tensor("type_emb.weight"), dtype=np.float32)


class FastLayaDecider:
    provider = "laya"
    kind = "laya"

    def __init__(self, model_dir: str | Path | None = None, packages: str | Path | None = None,
                 units: str = "CPU_AND_GPU", preload: bool = False, calib: str | None = None) -> None:
        self.model_dir = Path(model_dir).expanduser() if model_dir else weights_dir()
        self.packages = Path(packages).expanduser() if packages else CACHE
        self.units = units
        self.want_calib = calib                 # None reads PRESAGE_CALIB; "" forces the shipped one
        self.cfg: dict = {}
        self.tok = None
        self.types: np.ndarray | None = None
        self.gate: CoreMLGate | None = None
        self.prefixes: R.Prefixes | None = None
        self.load_ms: dict[str, float] = {}
        self.warm_ms = 0
        self.fallbacks = 0
        self.last_bucket = 0
        self._slow: Any = None
        self._load_lock = threading.Lock()
        self._alock: asyncio.Lock | None = None
        self._loader: threading.Thread | None = None
        if preload:
            self.preload()

    def __repr__(self) -> str:
        return (f"FastLayaDecider({R._short(self.model_dir) if hasattr(R, '_short') else self.model_dir}, "
                f"coreml/{self.units}, {'loaded' if self.gate else 'not loaded'})")

    # ---- availability ----
    def why_not(self) -> str | None:
        if not (self.model_dir / "model.safetensors").is_file():
            return f"no weights in {self.model_dir}"
        g = CoreMLGate(self.packages, self.units)
        if not g.available():
            return (f"no Core ML packages in {self.packages}: "
                    f"uv run python -m presage.llm.laya_fast.coreml --export")
        return None

    def available(self) -> bool:
        return self.why_not() is None

    # ---- load ----
    def preload(self) -> None:
        if self.gate is not None or (self._loader is not None and self._loader.is_alive()):
            return
        self._loader = threading.Thread(target=self._load_quiet, name="laya-fast-load", daemon=True)
        self._loader.start()

    def _load_quiet(self) -> None:
        try:
            self.load()
        except Exception as e:  # noqa: BLE001  preloading must never take the process down
            log.warning("laya_fast preload: %s", e)

    def load(self, warm_bucket: int | None = None) -> "FastLayaDecider":
        with self._load_lock:
            if self.gate is None:
                self._load(warm_bucket)
        return self

    def _load(self, warm_bucket: int | None = None) -> None:
        why = self.why_not()
        if why:
            raise DecisionError(why)
        ms = self.load_ms
        t = time.perf_counter()
        # through laya.load_config, so PRESAGE_CALIB picks the same temperatures here as on the
        # torch path: `runtime.wire_body` divides the marker logits by whatever this cfg carries
        self.cfg = load_config(self.model_dir, self.want_calib)
        self.tok = R.load_tokenizer(self.model_dir)
        self.types = type_table(self.model_dir)
        self.prefixes = R.Prefixes(self.tok, self.cfg)
        ms["tokenizer"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        g = CoreMLGate(self.packages, self.units)
        # every bucket, smallest first: Core ML's first prediction on a bucket costs 2.5 to 3.2 s
        # (E17, ten simulator runs), and on a call that lands on the first question the pre-gate
        # will not decide. Warming here, in the preload thread, moves that cost to startup.
        order = [warm_bucket] if warm_bucket else []
        order += [L for L in g.buckets if L not in order]
        for L in order:
            tb = time.perf_counter()
            g.warm(L)
            ms[f"warm_L{L}"] = (time.perf_counter() - tb) * 1e3
        ms["coreml"] = (time.perf_counter() - t) * 1e3
        ms["total"] = ms["tokenizer"] + ms["coreml"]     # the warm_L* entries are inside coreml
        self.gate = g
        log.info("laya_fast: %d buckets ready in %.1f s on coreml/%s",
                 len(g.buckets), ms["total"] / 1e3, self.units)

    # ---- the slow path ----
    def slow(self):
        if self._slow is None:
            from presage.llm.laya import LayaDecider
            self._slow = LayaDecider(self.model_dir, calib=self.want_calib).load()
        return self._slow

    # ---- the forward pass ----
    def system_one(self, state: Any, questions: list[Q]) -> dict:
        if self.gate is None:
            self.load()
        if not questions:
            raise DecisionError("no questions")
        names = [q.name for q in questions]
        if len(set(names)) != len(names):
            raise DecisionError(f"duplicate question names: {names}")

        rows, markers, qtypes, shapes = [], [], [], []
        for q in questions:
            ids, mk, qi = R.build_ids(self.tok, self.prefixes, state, q, self.cfg)
            if len(mk) != len(R.render_options(qi)):
                raise DecisionError(f"question {q.name!r}: options do not fit in "
                                    f"head_max_len={self.cfg['head_max_len']} tokens")
            rows.append(ids)
            markers.append(mk)
            qtypes.append(QTYPES[qi["t"]])
            shapes.append(qi)

        K = max(len(m) for m in markers)
        n = max(len(r) for r in rows)
        meta = self.gate.meta
        if len(rows) != meta.get("batch", 3) or K > meta.get("k", 5) or n > self.gate.buckets[-1]:
            self.fallbacks += 1
            return self.slow().system_one(state, questions)

        L = self.last_bucket = self.gate.bucket(n)
        b = R.pack(rows, markers, qtypes, self.types, self.tok.pad_token_id, L, meta.get("k", 5))
        logits = self.gate.run(b["input_ids"], b["pad_bias"], b["type_vec"], b["marker_map"])
        return R.wire_body(questions, shapes, [len(m) for m in markers], logits, self.cfg,
                           sum(len(r) for r in rows), self.cfg.get("model_name", "laya") + "-coreml")

    # ---- the Decider contract ----
    def decide_sync(self, state: str, questions: list[Q]) -> dict[str, Decision]:
        t0 = time.perf_counter()
        body = self.system_one(state, questions)
        return parse_answers(questions, body, int((time.perf_counter() - t0) * 1e3))

    def _lock(self) -> asyncio.Lock:
        if self._alock is None:
            self._alock = asyncio.Lock()
        return self._alock

    async def decide(self, state: str, questions: list[Q]) -> dict[str, Decision]:
        async with self._lock():
            return await asyncio.to_thread(self.decide_sync, state, questions)

    async def validate_key(self) -> tuple[bool, str]:
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
        ok, msg = await self.validate_key()
        if not ok:
            log.info("laya_fast warm failed: %s", msg)
            return None
        return float(self.warm_ms)

    async def aclose(self) -> None:
        self.gate = None
        if self._slow is not None:
            await self._slow.aclose()
            self._slow = None
