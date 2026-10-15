"""presage/feedback.py: the intervention log (experiment E12).

Every card presage shows is an intervention: an interruption the consultant either used, let go
by, or had to discard because it was wrong. The literature says the judge ("should I show this")
is learnable and the policy is not (ProactiveBench: 98% recall at 48% precision), and nobody
publishes how many wrong cards a user tolerates. That number has to come from presage's own log,
so this module writes one.

**Where.** Append-only JSONL, one file per local date:

    ~/.cache/presage/log/2026-09-22.jsonl        (PRESAGE_LOG_DIR overrides the folder)

Append-only means nothing is ever rewritten: feedback on an older card is a new row that refers
back to it by id, and `presage.feedback_report` folds the rows together when it reads them. A
crashed process loses at most the row it was writing.

**What.** Three event kinds, all carrying `ts`, `iso` (local time with its UTC offset) and
`session` (one id per server process, so two runs in a day stay apart):

  `card`      a card was broadcast: what the brain knew (id, question, the judged line, source,
              answered_from, derived, cached, figure, citations, latency_ms, mode, topic) and
              what the gate knew (its verdict, its confidence, the tier that answered it:
              pregate / jev / laya / haiku, and the gate's own latency).
  `replaced`  an earlier card was pushed off the top of the stack by a later one, with how long
              it had been up. A card replaced with no feedback on it is the log's weakest form
              of "ignored"; an explicit mark is the strong form.
  `feedback`  the user's verdict on a card: used | ignored | wrong | good.

**What is never written.** API keys (every string is scrubbed on the way out, see `scrub`), and
the transcript beyond the one line the gate judged and the question that line became. No context
lines, no prior turns, no mic side. The file is local and never leaves the machine.

**Use.**

    from presage import feedback
    feedback.record({"kind": "card", "id": a["id"], ...})    # brain or server, sync, ~0.1 ms
    feedback.mark("a4", "wrong")                             # from POST /api/feedback

`install_gate_watch()` attaches a log handler to `presage.brain` that reads `_gate_decider`'s
line ("gate CascadeDecider kind=question p=0.93 needs=True complete=True 9ms") and the cascade's
own stats, so the gate verdict and the tier reach a card event without brain.py knowing this
module exists.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("presage.feedback")

VERDICTS = ("used", "ignored", "wrong", "good")
EVENTS = ("card", "replaced", "feedback")

LOG_DIR = Path(os.environ.get("PRESAGE_LOG_DIR") or (Path.home() / ".cache" / "presage" / "log"))

GATE_MAX_AGE_S = 5.0       # the gate logs just before the answer starts; older than this is not it
RECENT_CARDS = 400         # ids kept in memory, for mark() and for replacement timing
MAX_TEXT = 400             # question / answer / line are truncated to this before writing

# Anything that looks like a credential is replaced before it reaches the file. The log holds
# card text, and card text comes from a model, so this is belt and braces rather than a real
# expectation: no field here is ever handed a key.
_SECRETS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bjev_[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|authorization|bearer|secret|token)\b\s*[=:]\s*\S+"),
]


def scrub(value: Any) -> Any:
    """Replace anything credential-shaped, at any depth. Called on every row before it is written."""
    if isinstance(value, str):
        for rx in _SECRETS:
            value = rx.sub("[redacted]", value)
        return value
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


def clip(text: Any, n: int = MAX_TEXT) -> str:
    s = "" if text is None else str(text)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ----------------------------------------------------------------------------- the log


class FeedbackLog:
    """One append-only JSONL file per local date, plus enough memory to relate rows to each other."""

    def __init__(self, folder: Path | str | None = None, session: str | None = None) -> None:
        self.folder = Path(folder) if folder else LOG_DIR
        self.session = session or uuid.uuid4().hex[:8]
        self.lock = threading.Lock()
        self.cards: dict[str, dict] = {}                     # id -> {ts, question, marked}
        self.order: deque[str] = deque(maxlen=RECENT_CARDS)
        self.top: str | None = None                          # the card currently at the top
        self.counts = {"card": 0, "replaced": 0, "feedback": 0, "errors": 0}

    # ------------------------------------------------------------------ paths

    def path(self, ts: float | None = None) -> Path:
        day = datetime.fromtimestamp(ts if ts is not None else time.time()).strftime("%Y-%m-%d")
        return self.folder / f"{day}.jsonl"

    # ------------------------------------------------------------------ writing

    def append(self, row: dict) -> dict:
        """Write one row. Never raises: a log that fails must not take a call down with it."""
        ts = float(row.get("ts") or time.time())
        row = {"ts": round(ts, 3),
               "iso": datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds"),
               "session": self.session, **row}
        row = scrub(row)
        try:
            with self.lock:
                self.folder.mkdir(parents=True, exist_ok=True)
                with self.path(ts).open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self.counts[str(row.get("event"))] = self.counts.get(str(row.get("event")), 0) + 1
        except Exception as e:  # noqa: BLE001
            self.counts["errors"] += 1
            log.warning("intervention log write failed: %s", e)
        return row

    def record(self, event: dict) -> dict | None:
        """Log one card. `event` is an `answer` frame (or anything with the same field names);
        everything the gate knew is attached here rather than by the caller.

        Re-sending a card id is an update, not a new intervention: the row says `update: true`.
        A streaming answer does this on every chunk, and so does the web escalation, so the count
        of card rows is much larger than the count of interruptions. The report folds them.
        """
        if not isinstance(event, dict):
            return None
        cid = str(event.get("id") or "").strip()
        if not cid:
            return None
        ts = float(event.get("logged_ts") or time.time())
        update = cid in self.cards
        if self.top and self.top != cid:
            # any frame with a different id takes the top of the stack, so whatever was there is
            # off it now, whether this one is new or a re-send of an older card
            self._replace(self.top, cid, ts)

        gate = event.get("gate")
        if gate is None and event.get("source") == "live":
            # only a live card can have come through the gate, and the gate runs just before the
            # answer does, so the anchor is when the question landed, not when the card went out
            ms = event.get("latency_ms")
            gate = gate_for(ts - (float(ms) / 1000 if isinstance(ms, (int, float)) else 0.0))
        row = {
            "event": "card", "id": cid, "ts": ts, "update": update,
            "question": clip(event.get("question")),
            "utt": event.get("utt"),
            "utt_text": clip(event.get("utt_text") or event.get("line")),
            "source": event.get("source"),
            "answered_from": event.get("answered_from"),
            "derived": bool(event.get("derived")),
            "cached": bool(event.get("cached")),
            "in_files": event.get("in_files"),
            "figure": clip(event.get("figure"), 40) or None,
            "citations": _citations(event.get("citations")),
            "web_citations": _web(event.get("web_citations")),
            "latency_ms": event.get("latency_ms"),
            "mode": event.get("mode"),
            "model": event.get("model"),
            "topic": clip(event.get("topic"), 120) or None,
            "say": clip(event.get("say"), 160),
            "answer": clip(event.get("answer")),
            "working": clip(event.get("working"), 200) or None,
            "gate": gate,
        }
        topics = [clip(t, 120) for t in (event.get("topics") or [])][:6]
        if topics:                      # only a caller using record() directly passes these
            row["topics"] = topics
        written = self.append(row)
        self.cards.setdefault(cid, {"ts": ts, "question": row["question"], "marked": None})
        if cid not in self.order:
            if len(self.order) == self.order.maxlen:
                self.cards.pop(self.order[0], None)      # the deque drops it; don't leak the dict
            self.order.append(cid)
        self.top = cid
        return written

    def _replace(self, old: str, by: str, ts: float) -> None:
        was = self.cards.get(old) or {}
        self.append({"event": "replaced", "id": old, "by": by, "ts": ts,
                     "after_s": round(ts - float(was.get("ts") or ts), 2),
                     "marked": was.get("marked")})

    def mark(self, card_id: str, verdict: str, note: str | None = None) -> dict:
        """The user's verdict on a card. Returns the row written; raises ValueError on a bad verdict."""
        verdict = str(verdict or "").strip().lower()
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
        cid = str(card_id or "").strip()
        if not cid:
            raise ValueError("no card id")
        known = self.cards.get(cid)
        ts = time.time()
        row = {"event": "feedback", "id": cid, "verdict": verdict, "ts": ts,
               "known": known is not None,
               "after_s": round(ts - float(known["ts"]), 2) if known else None,
               "question": (known or {}).get("question", ""),
               "was": (known or {}).get("marked")}
        if note:
            row["note"] = clip(note, 200)
        written = self.append(row)
        if known is not None:
            known["marked"] = verdict
        return written

    def stats(self) -> dict:
        return {"file": str(self.path()), "session": self.session, "cards": len(self.cards), **self.counts}


def _citations(cites: Any) -> list[str]:
    """`[{"file","loc"}]` -> `["Acme.xlsx › P&L › B12"]`. Locations are cell addresses, not content."""
    out = []
    for c in (cites or [])[:6]:
        if isinstance(c, dict):
            out.append(clip(f"{c.get('file', '')} › {c.get('loc', '')}".strip(" ›"), 120))
        else:
            out.append(clip(c, 120))
    return [c for c in out if c]


def _web(cites: Any) -> list[dict]:
    out = []
    for c in (cites or [])[:6]:
        if isinstance(c, dict):
            out.append({"title": clip(c.get("title"), 120), "url": clip(c.get("url"), 200)})
    return out


# ----------------------------------------------------------------------------- the gate watcher

_gate_seen: deque[dict] = deque(maxlen=32)
_gate_stats: dict[str, int] = {}
_gate_brain: Callable[[], Any] | None = None
GATE_RE = re.compile(r"^gate (?P<decider>\S+) kind=(?P<kind>\S+) p=(?P<p>[\d.]+) "
                     r"needs=(?P<needs>\S+) complete=(?P<complete>\S+) (?P<ms>\d+)ms$")


class _GateWatch(logging.Handler):
    """Reads `Brain._gate_decider`'s own log line, so brain.py needs no hook for the gate."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            m = GATE_RE.match(record.getMessage())
        except Exception:  # noqa: BLE001
            return
        if not m:
            return
        _gate_seen.append({"verdict": m["kind"],
                           "confidence": round(float(m["p"]), 4),
                           "needs_files": m["needs"] == "True",
                           "complete": None if m["complete"] == "None" else m["complete"] == "True",
                           "decider": m["decider"],
                           "tier": _tier(m["decider"]),
                           "ms": int(m["ms"]),
                           "ts": record.created})


def _tier(decider_class: str) -> str:
    """Which tier actually answered. A cascade counts its own calls, so the tier is the counter
    that moved since the last gate line: `fast` is the local pre-gate, `slow` is jev / laya /
    haiku. A plain decider names itself."""
    brain = _gate_brain() if _gate_brain else None
    d = getattr(brain, "decider", None)
    if d is None:
        return decider_class.replace("Decider", "").lower() or "unknown"
    stats = getattr(d, "stats", None)
    if isinstance(stats, dict) and "fast" in stats and "slow" in stats:
        fast, slow = int(stats.get("fast", 0)), int(stats.get("slow", 0))
        moved_fast = fast - _gate_stats.get("fast", 0)
        _gate_stats["fast"], _gate_stats["slow"] = fast, slow
        if moved_fast > 0:
            return "pregate"
        d = getattr(d, "slow", d)
    return str(getattr(d, "kind", None) or getattr(d, "provider", None)
               or type(d).__name__.replace("Decider", "").lower())


def gate_for(landed: float, max_age: float = GATE_MAX_AGE_S) -> dict | None:
    """The gate verdict this card came from: the newest one that fired around the moment the
    question landed (card time minus the answer's own latency), not the newest one before the
    card went out. A slow answer is easily overtaken by the gate call for the next line, and
    `landed` is what keeps that verdict off this card.

    It is still an association by time rather than an identity, so the row keeps `age_s` and the
    report can drop a stale one. `null` is a real answer, not missing data: a prefetched card, a
    typed ask, and the heuristic fast path (which cancels its own gate task before it runs) never
    had a gate verdict at all."""
    best = None
    for g in _gate_seen:
        if g["ts"] <= landed + 0.5 and (landed - g["ts"]) <= max_age:
            if best is None or g["ts"] > best["ts"]:
                best = g
    if best is None:
        return None
    out = {k: v for k, v in best.items() if k != "ts"}
    out["age_s"] = round(landed - best["ts"], 2)
    return out


def install_gate_watch(brain: Callable[[], Any] | None = None, logger: str = "presage.brain") -> None:
    """Attach the watcher once. `brain` is a zero-arg callable returning the live Brain.

    The line it reads is an INFO one, so `PRESAGE_LOG=WARNING` leaves every card's `gate` null.
    That is a logging setting deciding what the log knows, which is worth knowing about; the
    server's default is INFO."""
    global _gate_brain
    if brain is not None:
        _gate_brain = brain
    lg = logging.getLogger(logger)
    if any(isinstance(h, _GateWatch) for h in lg.handlers):
        return
    h = _GateWatch(level=logging.INFO)
    lg.addHandler(h)
    log.debug("gate watch installed on %s", logger)


# ----------------------------------------------------------------------------- module API

_log: FeedbackLog | None = None


def default_log() -> FeedbackLog:
    global _log
    if _log is None:
        _log = FeedbackLog()
    return _log


def record(event: dict) -> dict | None:
    return default_log().record(event)


def mark(card_id: str, verdict: str, note: str | None = None) -> dict:
    return default_log().mark(card_id, verdict, note)


def stats() -> dict:
    return default_log().stats()


def read(path: Path | str) -> list[dict]:
    """Every well-formed row in one file, oldest first. A half-written last line is skipped."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("event") in EVENTS:
            rows.append(row)
    return rows


def days(folder: Path | str | None = None) -> list[Path]:
    """Every day file, oldest first."""
    f = Path(folder) if folder else LOG_DIR
    return sorted(f.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].jsonl")) if f.exists() else []


if __name__ == "__main__":       # `uv run python -m presage.feedback` prints today's file
    fl = default_log()
    print(f"intervention log  {fl.path()}")
    rows = read(fl.path())
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["event"]] = kinds.get(r["event"], 0) + 1
    print(f"  {len(rows)} rows: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
          if rows else "  empty")
