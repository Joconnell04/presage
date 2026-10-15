"""Typed decisions: many small yes/no, pick-one and rate-it judgements in one round trip.

The brain's question gate runs on a Haiku call that costs 0.8-1.8 s (docs/research-llm.md). Jev,
TypeSafe AI's "System One" model, answers the same shape of question as a typed decision with a
calibrated probability instead of text, in one request that carries every question at once.

Two backends behind one `Decider`:

  JevDecider    POST https://api.typesafe.ai/v1/systemone, raw httpx, 2 s timeout, warm connection.
  HaikuDecider  the same questions as one JSON call through the existing LLM provider, for when
                there is no Jev key or the machine must stay local.

The wire format here was read out of the `langchain-typesafe` SDK source (0.0.1a3) rather than
guessed, then checked against the live endpoint; `docs/research-llm.md` records both, and which
fields the API actually returns versus which this module derives.

Nothing here generates text: Jev cannot write an answer, only judge one.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import httpx

log = logging.getLogger("presage.llm.jev")

BASE_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
PATH = "/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-latest")
TIMEOUT_S = float(os.environ.get("PRESAGE_JEV_TIMEOUT", "2.0"))
MAX_OPTIONS = 255          # documented ceiling for one choice question
SCORE_LEVELS = 5           # levels synthesized for a score question with no explicit rubric
USER_AGENT = "presage/0.1 (+typed-decisions)"

Kind = Literal["noul", "choice", "score"]


@dataclass
class Q:
    """One question. `name` is the key the answer comes back under."""
    name: str
    kind: Kind
    instructions: str
    options: list[str] | None = None      # choice: the labels; score: the ordered rubric
    lo: float = 0.0                       # score: value returned for the lowest level
    hi: float = 1.0                       # score: value returned for the highest level

    def levels(self) -> list[str]:
        """The ordered rubric for a score question, synthesized when none was given."""
        if self.options:
            return list(self.options)
        step = (self.hi - self.lo) / (SCORE_LEVELS - 1)
        return [f"{self.lo + i * step:g}" for i in range(SCORE_LEVELS)]


@dataclass
class Decision:
    """One answer. `value` is what to act on, `p` how strongly, `confidence` how sure Jev is."""
    value: bool | str | float
    p: float
    confidence: float
    probs: dict[str, float] | None = None
    latency_ms: int = 0
    raw: dict | None = field(default=None, repr=False)


class DecisionError(RuntimeError):
    """Any failure to get typed decisions. The brain falls back to heuristics on this."""

    def __init__(self, message: str, status: int | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.request_id = request_id


@runtime_checkable
class Decider(Protocol):
    async def decide(self, state: str, questions: list[Q]) -> dict[str, Decision]: ...
    async def validate_key(self) -> tuple[bool, str]: ...


# ----------------------------------------------------------------------------- request shaping

def question_payload(q: Q) -> dict:
    """One question in Jev's wire shape (SDK: Noul / Choice / Score, `exclude_none`)."""
    if q.kind == "noul":
        return {"type": "noul", "instructions": q.instructions}
    if q.kind == "choice":
        opts = list(q.options or [])
        if not opts:
            raise DecisionError(f"choice question {q.name!r} has no options")
        if len(opts) > MAX_OPTIONS:
            raise DecisionError(f"choice question {q.name!r} has {len(opts)} options, max {MAX_OPTIONS}")
        # criteria maps every label to its description; we only have the label itself
        return {"type": "choice", "instructions": q.instructions, "criteria": {o: o for o in opts}}
    if q.kind == "score":
        levels = q.levels()
        if len(levels) < 2:
            raise DecisionError(f"score question {q.name!r} needs at least 2 levels")
        return {"type": "score", "instructions": q.instructions, "criteria": levels}
    raise DecisionError(f"unknown question kind {q.kind!r}")


def build_payload(state: str, questions: list[Q], model: str = MODEL) -> dict:
    if not questions:
        raise DecisionError("no questions")
    names = [q.name for q in questions]
    if len(set(names)) != len(names):
        raise DecisionError(f"duplicate question names: {names}")
    return {"state": state, "model": model,
            "questions": {q.name: question_payload(q) for q in questions}}


# ----------------------------------------------------------------------------- response shaping

def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_answer(q: Q, answer: dict, ms: int) -> Decision:
    """One answer object into a Decision.

    Only `choice` and `score` carry `confidence` on the wire; a `noul` answer is the single
    probability, so its confidence is derived as the distance from a coin flip.
    """
    kind = answer.get("type") or q.kind
    if kind == "noul":
        p = _as_float(answer.get("noul"))
        return Decision(value=p >= 0.5, p=p, confidence=abs(p - 0.5) * 2,
                        probs={"true": p, "false": 1.0 - p}, latency_ms=ms, raw=answer)
    if kind == "choice":
        probs = {str(k): _as_float(v) for k, v in (answer.get("probabilities") or {}).items()}
        choice = answer.get("choice")
        if choice is None and probs:
            choice = max(probs, key=probs.get)
        return Decision(value=str(choice), p=probs.get(str(choice), 0.0),
                        confidence=_as_float(answer.get("confidence")), probs=probs or None,
                        latency_ms=ms, raw=answer)
    if kind == "score":
        levels = q.levels()
        raw_probs = {str(k): _as_float(v) for k, v in (answer.get("probabilities") or {}).items()}
        score = _as_float(answer.get("score"))          # expected value over 0..n-1, may be fractional
        span = (len(levels) - 1) or 1
        value = q.lo + (score / span) * (q.hi - q.lo)
        conf = _as_float(answer.get("confidence"))
        return Decision(value=value, p=conf, confidence=conf, probs=raw_probs or None,
                        latency_ms=ms, raw=answer)
    raise DecisionError(f"unknown answer type {kind!r} for {q.name!r}")


def parse_answers(questions: list[Q], body: dict, ms: int) -> dict[str, Decision]:
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise DecisionError("response has no answers object")
    out: dict[str, Decision] = {}
    for q in questions:
        a = answers.get(q.name)
        if not isinstance(a, dict):
            raise DecisionError(f"no answer for question {q.name!r}")
        out[q.name] = parse_answer(q, a, ms)
    return out


# ----------------------------------------------------------------------------- Jev

class JevDecider:
    """Hosted typed decisions. One warm HTTP/2 connection, every question in one request."""

    provider = "jev"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout: float = TIMEOUT_S, base_url: str | None = None,
                 transport: Any = None) -> None:
        self.api_key = api_key or os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        self.model = model or MODEL
        self.timeout = timeout
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self.http2 = False

    @property
    def url(self) -> str:
        return self.base_url + PATH

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            kw: dict[str, Any] = {
                "timeout": httpx.Timeout(self.timeout, connect=min(self.timeout, 1.0)),
                "limits": httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0),
                "headers": {"content-type": "application/json", "user-agent": USER_AGENT},
            }
            if self._transport is not None:
                kw["transport"] = self._transport
            else:
                try:
                    self._client = httpx.AsyncClient(http2=True, **kw)
                    self.http2 = True
                    return self._client
                except ImportError:
                    log.info("jev: h2 not installed, using HTTP/1.1 keep-alive")
            self._client = httpx.AsyncClient(**kw)
        return self._client

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise DecisionError("JEV_API_KEY (or TYPESAFE_API_KEY) is not set")
        return {"authorization": f"Bearer {self.api_key}"}

    async def decide(self, state: str, questions: list[Q]) -> dict[str, Decision]:
        payload = build_payload(state, questions, self.model)
        headers = self._headers()
        t0 = time.time()
        try:
            resp = await self.client.post(self.url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            raise DecisionError(f"jev timed out after {self.timeout}s") from e
        except httpx.HTTPError as e:
            raise DecisionError(f"cannot reach {self.base_url}: {type(e).__name__}") from e
        ms = int((time.time() - t0) * 1000)
        rid = resp.headers.get("x-typesafe-request-id")
        if resp.status_code >= 400:
            raise DecisionError(f"jev {resp.status_code}: {self._error_text(resp)}",
                                status=resp.status_code, request_id=rid)
        try:
            body = resp.json()
        except ValueError as e:
            raise DecisionError("jev returned a non-JSON body", status=resp.status_code, request_id=rid) from e
        out = parse_answers(questions, body, ms)
        log.debug("jev %s %d questions %d ms", body.get("model"), len(questions), ms)
        return out

    @staticmethod
    def _error_text(resp: httpx.Response) -> str:
        """The server's message, never the request (which carries the transcript).

        Measured shapes: `{"detail": "Choice question must have at least one choice: a"}`,
        `{"detail": {"error_type": "authentication_error", "message": "..."}}`, and FastAPI's
        `{"detail": [{"loc": [...], "msg": "..."}]}` on 422.
        """
        try:
            body = resp.json()
        except ValueError:
            return (resp.text or "")[:200]
        if isinstance(body, dict):
            for key in ("detail", "error", "message"):
                value = body.get(key)
                if isinstance(value, str) and value:
                    return value[:200]
                if isinstance(value, dict):
                    return str(value.get("message") or value)[:200]
                if isinstance(value, list) and value:
                    first = value[0]
                    if isinstance(first, dict):
                        loc = ".".join(str(x) for x in (first.get("loc") or []))
                        return f"{loc}: {first.get('msg')}"[:200] if loc else str(first.get("msg"))[:200]
                    return str(first)[:200]
        return json.dumps(body, default=str)[:200]

    async def warm(self) -> float | None:
        """One tiny decision so the TLS connection and the key check are already paid for.
        Returns the round trip in ms, or None if it failed (never raises)."""
        try:
            d = await self.decide("ready", [Q(name="ok", kind="noul", instructions="Is this text non-empty?")])
            return float(d["ok"].latency_ms)
        except Exception as e:  # noqa: BLE001
            log.info("jev warm failed: %s", e)
            return None

    async def validate_key(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "JEV_API_KEY (or TYPESAFE_API_KEY) is not set"
        try:
            d = await self.decide("ready", [Q(name="ok", kind="noul", instructions="Is this text non-empty?")])
        except DecisionError as e:
            if e.status in (401, 403):
                return False, f"JEV_API_KEY rejected ({e.status})"
            if e.status == 429:
                return False, "Jev rate limit reached (429)"
            return False, str(e)[:140]
        return True, f"ok ({d['ok'].latency_ms} ms)"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ----------------------------------------------------------------------------- Haiku fallback

_HAIKU_SYSTEM = (
    "Answer every question about the state below. Reply with the JSON object only.\n"
    "For each question give `answer` and a calibrated `confidence` from 0 to 1, where 0.5 means "
    "a coin flip and 0.99 means certain. Judge only what the state says."
)


class HaikuDecider:
    """The same questions as one structured call on the existing provider. Slower, no key needed
    beyond the one presage already has, and the state never leaves the model you already trust."""

    provider = "haiku"

    def __init__(self, llm: Any, model: str | None = None) -> None:
        self.llm = llm
        self.model = model          # None means the provider's classifier model

    @staticmethod
    def schema(questions: list[Q]) -> dict:
        props: dict[str, Any] = {}
        for q in questions:
            if q.kind == "noul":
                answer: dict[str, Any] = {"type": "boolean"}
            elif q.kind == "choice":
                answer = {"type": "string", "enum": list(q.options or [])}
            else:
                answer = {"type": "number", "minimum": q.lo, "maximum": q.hi}
            props[q.name] = {
                "type": "object",
                "properties": {"answer": answer, "confidence": {"type": "number"}},
                "required": ["answer", "confidence"],
                "additionalProperties": False,
            }
        return {"type": "object", "properties": props, "required": list(props),
                "additionalProperties": False}

    @staticmethod
    def prompt(state: str, questions: list[Q]) -> str:
        lines = [_HAIKU_SYSTEM, "", "State:", state, "", "Questions:"]
        for q in questions:
            if q.kind == "choice":
                lines.append(f"- {q.name}: {q.instructions} One of: {', '.join(q.options or [])}")
            elif q.kind == "score":
                lines.append(f"- {q.name}: {q.instructions} A number from {q.lo:g} to {q.hi:g}.")
            else:
                lines.append(f"- {q.name}: {q.instructions} true or false.")
        return "\n".join(lines)

    async def decide(self, state: str, questions: list[Q]) -> dict[str, Decision]:
        if not questions:
            raise DecisionError("no questions")
        t0 = time.time()
        try:
            data = await self.llm.classify(self.prompt(state, questions), self.schema(questions),
                                           model=self.model)
        except Exception as e:  # noqa: BLE001
            raise DecisionError(f"{type(e).__name__}: {str(e)[:140]}") from e
        ms = int((time.time() - t0) * 1000)
        if not isinstance(data, dict) or not data:
            raise DecisionError("classifier returned no answers")
        out: dict[str, Decision] = {}
        for q in questions:
            got = data.get(q.name)
            if not isinstance(got, dict) or "answer" not in got:
                raise DecisionError(f"no answer for question {q.name!r}")
            conf = min(max(_as_float(got.get("confidence"), 0.5), 0.0), 1.0)
            answer = got["answer"]
            if q.kind == "noul":
                value = bool(answer)
                p = conf if value else 1.0 - conf
                out[q.name] = Decision(value=value, p=p, confidence=abs(p - 0.5) * 2,
                                       probs={"true": p, "false": 1.0 - p}, latency_ms=ms, raw=got)
            elif q.kind == "choice":
                opts = list(q.options or [])
                rest = (1.0 - conf) / max(len(opts) - 1, 1)
                probs = {o: (conf if o == str(answer) else rest) for o in opts}
                out[q.name] = Decision(value=str(answer), p=conf, confidence=conf,
                                       probs=probs or None, latency_ms=ms, raw=got)
            else:
                value = min(max(_as_float(answer, q.lo), q.lo), q.hi)
                out[q.name] = Decision(value=value, p=conf, confidence=conf, probs=None,
                                       latency_ms=ms, raw=got)
        return out

    async def validate_key(self) -> tuple[bool, str]:
        return await self.llm.validate_key()


# ----------------------------------------------------------------------------- factory

def jev_key() -> str | None:
    return os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or None


_LAYA: dict[str, Any] = {}


def shared_laya(model_dir: Any = None, packages: Any = None, runtime: str | None = None,
                preload: bool = True, units: str | None = None,
                device: str | None = None) -> Decider:
    """One local decider per (weights directory, runtime, units) for the whole process.

    The brain rebuilds its decider on every provider or settings change; three fresh
    421M-parameter loads at startup once starved the transcriber for 40 s (replay bench,
    2026-09-22), so the loaded model is reused. E21a added a second caller -- the typed-intent
    checkpoint (`presage/llm/intent_local.py`) -- which is why the key carries the directory: two
    different checkpoints are two models, the same checkpoint asked for twice is one.

    `runtime` is "coreml", "torch", or None for "Core ML if its packages are exported, else
    torch". `packages` points at a Core ML package directory other than the gate's, which is what
    the intent decider's batch-1 lever package needs.

    `units` is the Core ML compute units for *this* decider (ALL | CPU_AND_NE | CPU_AND_GPU |
    CPU_ONLY) and `device` the torch device, both None for the runtime's own default. E21c keeps
    them per-decider rather than per-process: the gate wants the GPU, and the intent package
    beside it on the same GPU is what stalled the transcriber (E41), so the two packages in one
    server can now be sent to different hardware.
    """
    from presage.llm import laya as _laya
    d = Path(model_dir).expanduser() if model_dir else _laya.weights_dir()
    key = f"{d}|{runtime or 'auto'}|{packages or ''}|{units or ''}|{device or ''}"
    dec = _LAYA.get(key)
    if dec is not None:
        return dec
    if runtime != "torch":
        try:
            # the Core ML export of the same weights (E1): 47.6 ms and a 3.9 s load against 75 ms
            # and 22.7 s on torch MPS, identical answers; falls back to torch when not exported
            from presage.llm.laya_fast import FastLayaDecider
            from presage.llm.laya_fast.coreml import CoreMLGate
            kw = {"packages": packages} if packages else {}
            if units:
                kw["units"] = units
            fast = FastLayaDecider(model_dir=d, preload=False, **kw)
            # a Core ML package carries the encoder weights it was traced from; only the tokenizer
            # and `type_emb` come from `model_dir`. Handing it a different checkpoint's directory
            # would answer with the traced weights under the other one's name, so the package is
            # used only when it was exported from exactly this directory.
            traced = str(CoreMLGate(fast.packages, fast.units).meta.get("model_dir") or "")
            why = fast.why_not()
            if not why and traced and Path(traced).expanduser() != d:
                why = f"package in {fast.packages} was traced from {traced}"
            if not why:
                if preload:
                    fast.preload()
                _LAYA[key] = fast
                return fast
            if runtime == "coreml":
                raise DecisionError(why)
            log.info("laya: Core ML not usable for %s (%s); using torch", d.name, why)
        except DecisionError:
            raise
        except Exception as e:  # noqa: BLE001
            log.debug("laya_fast unavailable: %s", e)
    # the 20 s load runs once, in the background
    dec = _laya.LayaDecider(model_dir=d, device=device, preload=preload)
    _LAYA[key] = dec
    return dec


def _shared_laya() -> Decider:
    """The shared local gate, on whatever `laya.weights_dir()` resolves to."""
    return shared_laya()


def _with_pregate(slow: Decider) -> Decider:
    """The 0.2 ms local pre-gate in front of the real decider (E22: decides 85-95% of lines alone,
    cascade 98.7% vs Jev 94.8% on real lines). Off with PRESAGE_PREGATE=0 or when its weights are
    not built (data/gate/pregate/train.py)."""
    if os.environ.get("PRESAGE_PREGATE", "1").lower() in ("0", "false", "off"):
        return slow
    try:
        from presage.llm.pregate import cascade
        return cascade(None, slow)
    except Exception as e:  # noqa: BLE001
        log.debug("pregate unavailable: %s", e)
        return slow


def _fast_available() -> bool:
    """The Core ML export of the tuned gate can run here (coremltools installed, packages on disk).
    It needs no torch, so the gate must not be skipped just because the torch extra is missing."""
    try:
        from presage.llm.laya_fast import FastLayaDecider
        import coremltools  # noqa: F401  the package check below does not import it
        return bool(FastLayaDecider(preload=False).available())
    except Exception:  # noqa: BLE001
        return False


def make_decider(llm: Any = None, prefer: str | None = None) -> Decider:
    """The local fine-tuned gate when its weights are present, else Jev when a key is set, else
    the provider's own small model (E17 set that order).

    `prefer="haiku"` forces the provider path even with a key and weights; `prefer="jev"` sends
    the gate to the hosted decider, so the transcript leaves the machine on that path.
    """
    want = (prefer or os.environ.get("PRESAGE_DECIDER", "")).strip().lower()
    if want in ("haiku", "llm", "off"):
        return _with_pregate(HaikuDecider(llm))
    if want == "laya":
        return _with_pregate(_shared_laya())
    if want in ("", "auto"):
        # Local first (E17, docs/experiments/E17-local-vs-jev.md): over 20 simulator runs the Core ML
        # v2 gate put 30 wrong cards on screen against Jev's 43 of 122 client turns, missed 5
        # against 6, cost 0 ms per turn on first card, and stalled transcription zero times. The
        # 40 s stall that once kept it second (docs/experiments/LOG.md, 2026-09-22) was the torch
        # MPS load; the Core ML export does not reproduce it. The transcript also stays on the
        # machine. The stock checkpoint is never picked automatically; it measured 42% and has to
        # be asked for by name.
        try:
            from presage.llm import laya as _laya
            if _laya.tuned_present() and (_fast_available() or _laya.available()):
                return _with_pregate(_shared_laya())
            log.info("local gate not picked: tuned weights %s, Core ML %s, torch %s",
                     _laya.tuned_present(), _fast_available(), _laya.available())
        except Exception as e:  # noqa: BLE001  the local gate is optional
            log.debug("laya unavailable: %s", e)
    if want in ("jev", "typesafe") or (want in ("", "auto") and jev_key()):
        return _with_pregate(JevDecider())        # by name, or auto with a key and no local weights
    return _with_pregate(HaikuDecider(llm))


async def _selftest() -> None:  # pragma: no cover - exercised by presage.llm.smoke --jev
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    d = JevDecider()
    print("validate_key:", await d.validate_key())
    qs = [Q("is_question", "noul", "Is the last line a question aimed at the speaker?"),
          Q("kind", "choice", "What is the last line?", options=["question", "statement", "objection"]),
          Q("urgency", "score", "How urgent does the last line sound?")]
    out = await d.decide("Them: So on the revenue side, what did Q3 look like versus plan?", qs)
    for k, v in out.items():
        print(f"  {k}: value={v.value!r} p={v.p:.3f} conf={v.confidence:.3f} {v.latency_ms} ms")
    await d.aclose()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_selftest())
