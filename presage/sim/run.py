"""Run a script against a real server and score what the page would have shown.

    uv run python -m presage.sim.run sample-01-quarterly
    uv run python -m presage.sim.run --all
    uv run python -m presage.sim.run --score-only docs/experiments/sim/sample-01-quarterly

One server at a time, started the way the README starts one in replay mode, with both voices
when the platform's mic replay channel is there:

    PRESAGE_CONTEXT=<the script's folder> PRESAGE_REPLAY=<the Them wav> \\
        PRESAGE_REPLAY_MIC=<the Me wav> uv run python -m presage.server

The run records every WebSocket frame with the time it arrived at the client, waits for the
clip plus 20 s, stops the server, and scores against the script's `expect` blocks and the
clip's timeline. Everything lands in `docs/experiments/sim/<run-id>/`:

    frames.jsonl   every frame, with `_ts` (client clock) added
    server.log     the server's own log, which is where the gate lines and the llm cost
                   records are read from
    run.json       what was run, and the audio clock estimate the scoring depends on
    score.json     per turn and totals
    score.md       the same, readable, with the failure list

**Scorer v2 is the default, and every new run is written with it** (E45,
`docs/experiments/E45-scorer.md`). A hit is a card that answers the turn -- its figure matches
the expected figure, or the answer-key entry behind a `request` expectation that carries none --
that lands within `HIT_WINDOW_S` (8 s) of the second the question ended in the clip, and that was
not already on screen from an earlier turn unless the figure on it is the one this turn asked
for. A right card past the window is a **late hit**: reported on its own line and counted as a
miss in the headline. Wrong cards, noise and redraws keep E5's definitions but are counted per
card shown rather than per turn. v1, the scorer E5, E34 and E17 were published with, stays
callable for comparisons: `score(..., version=1)` or `--score-version 1`.

E68 adds to v2 without moving any number it already reported (re-scoring the 43 committed runs
gives the same totals and turn verdicts). A **no-answer card** (`answer.no_answer` set: the brain's
"Couldn't answer" / "Not in the files" card for a registered question with nothing on screen) is
labelled `no-answer` on a turn that wanted a card: it answers nothing, so it is never a hit, and it
misleads nobody, so it is not a wrong card; on a turn that wanted none it is noise like any other
card. **Registration** is read per turn: a `question` frame for the line, or, for runs recorded
before E68 had the frame, an answer frame carrying the line's utterance, which is the same decision
seen from the card side. A consultant's card (`who: "Me"`) is charged to the consultant's own turn
through its `line` id.

The one clock that matters is the audio clock: a card is measured against the second the
question ENDED IN THE CLIP, not against the transcript. The server does not publish when
its replay started, so it is bracketed: not before the "audio started" line in the log, not
after the first transcript frame could have arrived, not after the first HTTP poll that saw
the session running, and consistent with the "replay finished" line a clip later. `anchor`
is the middle of that bracket and `anchor_uncertainty_ms` is its half width, which every
first-card number carries.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import statistics
import contextlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime
from pathlib import Path

import websockets

from presage.sim import render as R
from presage.sim import scripts as S

RUN_DIR = S.ROOT / "docs" / "experiments" / "sim"
SETTINGS = S.ROOT / "settings.json"
BACKUP = S.ROOT / "settings.json.simbak"
PORT = int(os.environ.get("PRESAGE_PORT", "8765"))
TAIL_S = 20.0
READY_TIMEOUT_S = 180.0
POLL_S = 0.025
FIG_TOL = 0.005          # 0.5% relative: "$27.84M" is the same figure as 27,835,848.58

# E45. v2 is the scorer every new run is written with; v1 stays callable so the committed E5,
# E34 and E17 tables can still be reproduced frame for frame.
SCORE_VERSION = 2
HIT_WINDOW_S = 8.0       # a card later than this answers a question the call has moved off

# $/1M tokens, SOURCE: the claude-api skill's model table (cached 2026-06-24). Cache reads are
# ~0.1x input and cache writes ~1.25x input. claude_code answers bill the subscription, not the
# API, so they are counted in tokens and left out of the dollar total.
PRICES = {"claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
          "claude-haiku-4-5-20251001": (1.0, 5.0), "claude-opus-5": (5.0, 25.0)}

GATE_RE = re.compile(
    r"^(\d\d:\d\d:\d\d) \w+ presage\.brain: gate (\S+) kind=(\S+) p=([\d.]+) needs=(\S+) "
    r"complete=(\S+) (\d+)ms")
LLM_RE = re.compile(r"^(\d\d:\d\d:\d\d) \w+ presage\.brain: llm (\{.*\})\s*$")
SKIP_RE = re.compile(r"^(\d\d:\d\d:\d\d) \w+ presage\.brain: (skip \w+ [^\n]*)$")
STAMP_RE = re.compile(r"^(\d\d:\d\d:\d\d) ")
STARTED_RE = re.compile(r"^(\d\d:\d\d:\d\d) \w+ presage\.audio_session: audio started")
FINISHED_RE = re.compile(r"^(\d\d:\d\d:\d\d) \w+ presage\.audio_session: replay finished")


# ----------------------------------------------------------------------------- figures

_FIG = re.compile(r"(-)?\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(%|bps|k|m|bn|b|thousand|million|billion)?",
                  re.I)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6,
          "bn": 1e9, "b": 1e9, "billion": 1e9}


def candidates(text: str) -> list[float]:
    """Every number a figure string could mean, so "$27.84M" and 27,835,848.58 can meet.

    A figure with no magnitude on it is ambiguous by unit (a workbook in $ thousands says
    14,020 for $14.02M), so the plain reading and the x1,000 / x1,000,000 readings are all
    candidates; a figure that carries its own magnitude ("$27.84M", "13.6%") is not.
    """
    m = _FIG.search(str(text or ""))
    if not m:
        return []
    value = float(m.group(2).replace(",", ""))
    if m.group(1):
        value = -value
    suffix = (m.group(3) or "").lower()
    if suffix == "%":
        return [value, value / 100.0]
    if suffix == "bps":
        return [value, value / 10000.0]
    if suffix in _SCALE:
        return [value * _SCALE[suffix]]
    return [value, value * 1e3, value * 1e6]


def figure_match(expected: str, got: str) -> bool:
    want, have = candidates(expected), candidates(got)
    if not want or not have:
        return bool(expected) and expected.strip().lower() in str(got or "").strip().lower()
    for a in want:
        for b in have:
            if abs(a - b) <= FIG_TOL * max(abs(a), abs(b), 1e-9):
                return True
    return False


# ----------------------------------------------------------------------------- the server

def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) != 0


def kill_listeners(port: int) -> list[str]:
    """Whatever is still listening on the port after the server is gone.

    The answer provider spawns `claude` children that inherit the server's listening socket, so
    the port can stay bound for a minute after the server itself is dead.
    """
    killed = []
    pids = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                          capture_output=True, text=True).stdout.split()
    for pid in pids:
        cmd = subprocess.run(["ps", "-p", pid, "-o", "command="],
                             capture_output=True, text=True).stdout.strip()
        if any(k in cmd for k in ("presage", "claude", "python")):
            with contextlib.suppress(ProcessLookupError, PermissionError, ValueError):
                os.kill(int(pid), signal.SIGKILL)
                killed.append(f"{pid} {cmd[:60]}")
    return killed


def free_port(port: int, patience_s: int = 90) -> bool:
    for i in range(patience_s):
        if port_free(port):
            return True
        if i in (5, 25, 60):
            for k in kill_listeners(port):
                print(f"  killed {k}, it was still holding port {port}", file=sys.stderr)
        time.sleep(1)
    return port_free(port)


def get_state(port: int, timeout: float = 1.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


class Poller(threading.Thread):
    """Polls /api/state until the replay session is running, and keeps the last state.

    The replay pump starts inside the server's own startup, before the HTTP port answers, so
    this only ever gives an upper bound on when audio started; the log gives the lower one.
    """

    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.stop = threading.Event()
        self.first_ok: float | None = None       # first /api/state that answered at all
        self.first_running: float | None = None
        self.last_not_running: float | None = None
        self.state: dict | None = None

    def run(self) -> None:
        while not self.stop.is_set():
            t = time.time()
            st = get_state(self.port, timeout=0.8)
            if st is not None:
                self.state = st
                if self.first_ok is None:
                    self.first_ok = t
                audio = (st.get("audio") or {})
                if audio.get("state") == "running":
                    if self.first_running is None:
                        self.first_running = time.time()
                    return
                self.last_not_running = time.time()
            self.stop.wait(POLL_S)


async def record_ws(port: int, path: Path, stop: asyncio.Event) -> int:
    """Every frame to disk as it arrives, one JSON object a line. Checkpointed: a killed run
    still has everything up to the moment it died."""
    n = 0
    url = f"ws://127.0.0.1:{port}/ws"
    with path.open("w", encoding="utf-8") as fh:
        while not stop.is_set():
            try:
                async with websockets.connect(url, max_size=None, open_timeout=5) as ws:
                    while not stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
                        msg = json.loads(raw)
                        msg["_ts"] = time.time()
                        fh.write(json.dumps(msg) + "\n")
                        fh.flush()
                        n += 1
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  the socket is up only once the server is
                await asyncio.sleep(0.25)
    return n


def head() -> dict:
    """Which build answered, so two runs can be told apart. The server reads the working tree,
    not the commit, so a dirty file under presage/ is part of the answer."""
    def git(*args: str) -> str:
        p = subprocess.run(["git", *args], cwd=str(S.ROOT), capture_output=True, text=True)
        return p.stdout.strip()
    dirty = [l[3:] for l in git("status", "--porcelain", "presage").splitlines()]
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": dirty}


def start_server(context: Path, clip: Path, port: int, log: Path,
                 live: str = "same", mic_clip: Path | None = None,
                 screen_script: str | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop("PRESAGE_OPEN_BROWSER", None)
    env.update(PRESAGE_CONTEXT=str(context), PRESAGE_REPLAY=str(clip),
               PRESAGE_PORT=str(port), PRESAGE_LOG="INFO", PYTHONUNBUFFERED="1",
               # belt and braces with the settings.json pin: one provider for the whole call
               PRESAGE_LIVE_PROVIDER=live)
    if mic_clip is not None and mic_clip.is_file():
        env["PRESAGE_REPLAY_MIC"] = str(mic_clip)     # the consultant's own lines, same timeline
    else:
        env.pop("PRESAGE_REPLAY_MIC", None)
        print(f"  no Me wav ({mic_clip}): the mic side of the call will be silent. "
              f"`python -m presage.sim.render <id>` rebuilds it", file=sys.stderr)
    # E65. The screen poller reads the real Mac: with it on, whatever the person running the
    # bench happens to have open becomes evidence, and a page of it going stale can raise a card
    # in the middle of someone else's script. A run either drives the screen from the script
    # (`presage.sim.serve`) or has no screen at all.
    if screen_script:
        env["PRESAGE_SIM_SCRIPT"] = screen_script
        env.pop("PRESAGE_SCREEN", None)
        module = "presage.sim.serve"
    else:
        env.pop("PRESAGE_SIM_SCRIPT", None)
        env["PRESAGE_SCREEN"] = "0"
        module = "presage.server"
    fh = log.open("w", encoding="utf-8")
    return subprocess.Popen([sys.executable, "-m", module], cwd=str(S.ROOT), env=env,
                            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)


def stop_server(proc: subprocess.Popen, port: int = PORT) -> None:
    if proc.poll() is not None:
        return
    with_group = os.getpgid(proc.pid)
    os.killpg(with_group, signal.SIGTERM)
    for _ in range(80):
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    if proc.poll() is None:
        os.killpg(with_group, signal.SIGKILL)
        proc.wait(timeout=5)
    subprocess.run(["pkill", "-f", "presage.server"], capture_output=True)
    subprocess.run(["pkill", "-f", "audiotap"], capture_output=True)
    kill_listeners(port)


def pin_live_provider(choice: str) -> dict | None:
    """Pin where live answers run for the duration of the run, with the old file backed up.

    `live_choice: auto` sends the live answer to the Anthropic API whenever a key exists,
    which is a different provider from the one the drafts run on and a different bill; a
    benchmark wants one provider for the whole call. The original file is restored by
    `restore_settings`, and a leftover backup is restored on the next run.
    """
    if not SETTINGS.exists():
        return None
    raw = SETTINGS.read_text(encoding="utf-8")
    data = json.loads(raw)
    if choice in (None, "", "keep") or data.get("live_choice") == choice:
        return None
    BACKUP.write_text(raw, encoding="utf-8")
    data["live_choice"] = choice
    SETTINGS.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def restore_settings() -> bool:
    if not BACKUP.exists():
        return False
    SETTINGS.write_text(BACKUP.read_text(encoding="utf-8"), encoding="utf-8")
    BACKUP.unlink(missing_ok=True)
    return True


# ----------------------------------------------------------------------------- the clock

def log_seconds(line: str, day: datetime) -> float | None:
    m = STAMP_RE.match(line)
    if not m:
        return None
    h, mi, s = (int(x) for x in m.group(1).split(":"))
    return day.replace(hour=h, minute=mi, second=s, microsecond=0).timestamp()


def estimate_anchor(log_text: str, frames: list[dict], poller_first_running: float | None,
                    clip_s: float, day: datetime) -> dict:
    """When audio second 0 was fed, as a bracket. See the module docstring."""
    lo: list[tuple[float, str]] = []
    hi: list[tuple[float, str]] = []
    for line in log_text.splitlines():
        t = log_seconds(line, day)
        if t is None:
            continue
        if STARTED_RE.match(line):
            lo.append((t, "log: audio started"))          # the log stamp floors to the second
            hi.append((t + 1.0, "log: audio started +1s"))
        elif FINISHED_RE.match(line):
            lo.append((t - clip_s, "log: replay finished"))
            hi.append((t + 1.0 - clip_s, "log: replay finished +1s"))
    if poller_first_running:
        hi.append((poller_first_running, "first /api/state running"))
    deltas = [f["_ts"] - float(f.get("t1") or 0.0) for f in frames
              if f.get("type") == "transcript" and f.get("source") == "system"
              and f.get("t1") is not None]
    if deltas:
        hi.append((min(deltas), "first transcript frame"))

    low, low_why = max(lo, default=(0.0, "none"))
    high, high_why = min(hi, default=(0.0, "none"))
    if not lo or not hi or high < low:                   # bracket collapsed: trust the log line
        anchor = low or high
        return {"anchor": anchor, "lo": low, "hi": high, "uncertainty_ms": 500,
                "lo_from": low_why, "hi_from": high_why, "bracket_ok": False}
    return {"anchor": (low + high) / 2, "lo": low, "hi": high,
            "uncertainty_ms": round((high - low) * 500),
            "lo_from": low_why, "hi_from": high_why, "bracket_ok": True}


# ----------------------------------------------------------------------------- scoring

def read_frames(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def utterance_turns(frames: list[dict], turns: list[dict]) -> dict[int, int]:
    """system utterance number -> turn index, by overlap on the audio clock."""
    spans: dict[int, list[float]] = {}
    for f in frames:
        if f.get("type") != "transcript" or f.get("source") != "system":
            continue
        u = f.get("utt")
        if u is None:
            continue
        t0, t1 = float(f.get("t0") or 0.0), float(f.get("t1") or 0.0)
        cur = spans.setdefault(int(u), [t0, t1])
        cur[0], cur[1] = min(cur[0], t0), max(cur[1], t1)
    out: dict[int, int] = {}
    for u, (t0, t1) in spans.items():
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(t1, t["end_s"]) - max(t0, t["start_s"])
            if ov > best_ov:
                best, best_ov = t["i"], ov
        if best is not None:
            out[u] = best
    return out


def speaker_turns(frames: list[dict], turns: list[dict], source: str = "mic",
                  speaker: str = "me") -> dict[int, int]:
    """`utterance_turns` for one source: utterance number -> turn index, by overlap on the clip
    clock. E68 needs the consultant's side, which `utterance_turns` never mapped."""
    spans: dict[int, list[float]] = {}
    for f in frames:
        if f.get("type") != "transcript" or f.get("source") != source or f.get("utt") is None:
            continue
        t0, t1 = float(f.get("t0") or 0.0), float(f.get("t1") or 0.0)
        cur = spans.setdefault(int(f["utt"]), [t0, t1])
        cur[0], cur[1] = min(cur[0], t0), max(cur[1], t1)
    out: dict[int, int] = {}
    mine = [t for t in turns if t["speaker"] == speaker]
    for u, (t0, t1) in spans.items():
        best, best_ov = None, 0.0
        for t in mine:
            ov = min(t1, t["end_s"]) - max(t0, t["start_s"])
            if ov > best_ov:
                best, best_ov = t["i"], ov
        if best is not None:
            out[u] = best
    return out


def _line_utt(line: object) -> int | None:
    """The utterance number out of a transcript line id ("tm3" -> 3), or None."""
    m = re.match(r"^t[ms](\d+)$", str(line or ""))
    return int(m.group(1)) if m else None


def attribute(card: dict, turns: list[dict], anchor: float) -> int | None:
    """Which turn a card with no utterance number belongs to: the question it repeats if the
    card carries one, otherwise the client turn it landed inside or just after."""
    them = [t for t in turns if t["speaker"] == "them"]
    q = str(card.get("question") or "").lower()
    if q:
        words = set(re.findall(r"[a-z']{3,}", q))
        best, score_ = None, 0.0
        for t in them:
            tw = set(re.findall(r"[a-z']{3,}", t["text"].lower()))
            if tw:
                ov = len(words & tw) / max(1, len(tw))
                if ov > score_:
                    best, score_ = t["i"], ov
        if best is not None and score_ >= 0.5:
            return best
    at = card["_ts"] - anchor
    inside = [t for t in them if t["start_s"] - 2.0 <= at <= t["end_s"] + 10.0]
    if inside:
        return inside[-1]["i"]
    prior = [t for t in them if t["end_s"] <= at]
    return prior[-1]["i"] if prior else None


def gate_lines(log_text: str, day: datetime, anchor: float) -> list[dict]:
    out = []
    for line in log_text.splitlines():
        m = GATE_RE.match(line)
        if not m:
            continue
        t = log_seconds(line, day)
        out.append({"at_s": round((t or 0.0) + 0.5 - anchor, 2), "decider": m.group(2),
                    "kind": m.group(3), "p": float(m.group(4)), "needs": m.group(5) == "True",
                    "complete": m.group(6), "ms": int(m.group(7))})
    return out


def skip_lines(log_text: str, day: datetime, anchor: float) -> list[dict]:
    """The brain's own "skip ..." decisions, which is how a question gets no card without the
    gate ever being asked."""
    out = []
    for line in log_text.splitlines():
        m = SKIP_RE.match(line)
        if m:
            t = log_seconds(line, day)
            out.append({"at_s": round((t or 0.0) + 0.5 - anchor, 2), "why": m.group(2)})
    return out


def providers_seen(log_text: str) -> list[str]:
    """Which providers actually answered, from the log's own records. The page's control API is
    open on localhost, so a server can be switched under a run; this is how that shows up."""
    out = []
    for line in log_text.splitlines():
        m = LLM_RE.match(line)
        if m:
            try:
                rec = json.loads(m.group(2))
            except json.JSONDecodeError:
                continue
            tag = f"{rec.get('provider')}/{rec.get('model')}"
            if tag not in out:
                out.append(tag)
    return out


def llm_cost(log_text: str) -> dict:
    by_model: dict[str, dict] = {}
    for line in log_text.splitlines():
        m = LLM_RE.match(line)
        if not m:
            continue
        try:
            rec = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        key = f"{rec.get('provider')}/{rec.get('model')}"
        acc = by_model.setdefault(key, {"calls": 0, "in": 0, "out": 0, "cache_read": 0,
                                        "cache_write": 0, "total_s": 0.0})
        acc["calls"] += 1
        for f in ("in", "out", "cache_read", "cache_write"):
            acc[f] += int(rec.get(f) or 0)
        acc["total_s"] += float(rec.get("total_s") or 0.0)
    usd = sub = 0.0
    for line in log_text.splitlines():
        m = LLM_RE.match(line)
        if m:
            try:
                sub += float(json.loads(m.group(2)).get("cost_usd_list") or 0.0)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    for key, acc in by_model.items():
        provider, _, model = key.partition("/")
        price = PRICES.get(model)
        acc["usd"] = None
        if provider == "anthropic" and price:
            acc["usd"] = round((acc["in"] + acc["cache_read"] * 0.1 + acc["cache_write"] * 1.25)
                               * price[0] / 1e6 + acc["out"] * price[1] / 1e6, 4)
            usd += acc["usd"]
        acc["total_s"] = round(acc["total_s"], 1)
    return {"by_model": by_model, "usd": round(usd, 4),
            "subscription_list_usd": round(sub, 4),
            "note": "usd is what the API was billed at list price (SOURCE: the claude-api model "
                    "table); subscription_list_usd is what the claude_code answers would have "
                    "cost on the API, as the Agent SDK reports it, and is drawn from the plan"}


def card_text(c: dict) -> str:
    return " ".join(str(c.get(k) or "") for k in ("figure", "say", "answer"))


def score(script: dict, timeline: dict, frames: list[dict], log_text: str, anchor: float,
          day: datetime, version: int = SCORE_VERSION) -> dict:
    """The scorer. `version=2` (the default) is E45's; `version=1` is what E5, E34 and E17
    were published with and is kept so those tables can be reproduced."""
    fn = {1: score_v1, 2: score_v2}.get(int(version))
    if fn is None:
        raise ValueError(f"no scorer version {version}")
    sc = fn(script, timeline, frames, log_text, anchor, day)
    sc["score_version"] = int(version)
    return sc


def turn_table(script: dict, timeline: dict) -> list[dict]:
    """The clip's turns with the script's expectation and trap on each."""
    by_i = {t["i"]: t for t in script["turns"]}
    out = []
    for t in timeline["turns"]:
        turn = dict(t)
        turn["expect"] = by_i[t["i"]]["expect"]
        turn["trap"] = by_i[t["i"]]["trap"]
        out.append(turn)
    return out


def score_v1(script: dict, timeline: dict, frames: list[dict], log_text: str, anchor: float,
             day: datetime) -> dict:
    turns = turn_table(script, timeline)
    them = [t for t in turns if t["speaker"] == "them"]
    u2t = utterance_turns(frames, turns)

    cards = [f for f in frames if f.get("type") == "answer"]
    per_turn: dict[int, list[dict]] = {t["i"]: [] for t in turns}
    orphans: list[dict] = []
    for c in cards:
        u = c.get("utt")
        i = u2t.get(int(u)) if u is not None else None
        if i is None:
            i = attribute(c, turns, anchor)
        (per_turn[i] if i is not None else orphans).append(c)

    finals_by_turn: dict[int, float] = {}
    for f in frames:
        if (f.get("type") == "transcript" and f.get("source") == "system" and f.get("final")
                and f.get("utt") is not None):
            i = u2t.get(int(f["utt"]))
            if i is not None:
                finals_by_turn[i] = max(finals_by_turn.get(i, 0.0), f["_ts"])

    gates = gate_lines(log_text, day, anchor)
    skips = skip_lines(log_text, day, anchor)
    seen_cards: dict[str, set] = {}      # card id -> every figure it has carried so far
    rows, failures = [], []
    for t in turns:
        mine = sorted(per_turn.get(t["i"], []), key=lambda c: c["_ts"])
        withtext = [c for c in mine if (c.get("answer") or c.get("figure"))]
        finals = [c for c in mine if not c.get("partial")]
        first, last = (withtext[0] if withtext else None), (finals[-1] if finals else None)
        e = t["expect"]
        row = {"i": t["i"], "speaker": t["speaker"], "text": t["text"], "trap": t["trap"],
               "end_s": t["end_s"], "expect": e, "cards": len(mine), "card_id": None,
               "first_card_ms": None, "final_card_ms": None, "figure": None, "say": None,
               "answered_from": None, "cached": None, "card_source": None, "verdict": "",
               "transcript_final_ms": (round((finals_by_turn[t["i"]] - (anchor + t["end_s"])) * 1000)
                                       if t["i"] in finals_by_turn else None),
               "gate": next((g for g in gates
                             if t["end_s"] - 0.2 <= g["at_s"] <= t["end_s"] + 6.0), None),
               "skip": next((s for s in skips
                             if t["end_s"] - 0.2 <= s["at_s"] <= t["end_s"] + 8.0), None)}
        if first is not None:
            row["first_card_ms"] = round((first["_ts"] - (anchor + t["end_s"])) * 1000)
            row["first_card_kind"] = ("partial" if first.get("partial")
                                      else ("instant" if first.get("cached") else
                                            first.get("source", "?")))
        if last is not None:
            row["final_card_ms"] = round((last["_ts"] - (anchor + t["end_s"])) * 1000)
            row["figure"] = last.get("figure")
            row["say"] = last.get("say")
            row["answered_from"] = last.get("answered_from")
            row["cached"] = bool(last.get("cached"))
            row["card_source"] = last.get("source")

        got_card = last is not None or first is not None
        card = last or first
        row["card_id"] = (card or {}).get("id")
        # the page replaces a card in place when the id comes back, so the same id carrying the
        # same figure on the next turn is the card that is already on screen, not a new one
        redraw = bool(card) and str(card.get("figure")) in seen_cards.get(str(card.get("id")), set())
        if card is not None:
            for c in mine:
                seen_cards.setdefault(str(c.get("id")), set()).add(str(c.get("figure")))
        row["redraw"] = redraw
        if t["speaker"] == "me" or not e["card"]:
            row["verdict"] = "quiet" if not got_card else ("redraw" if redraw else "noise")
        elif not got_card:
            row["verdict"] = "miss"
        elif e["source"] == "none":
            # "not in files" is the right card, and so is a cited web page: with web on auto, a
            # public fact the files cannot hold is exactly what the escalation is for
            from_ = (card or {}).get("answered_from")
            row["verdict"] = "hit" if from_ in ("none", "web") else "wrong"
            if from_ == "web":
                row["note"] = "answered from the web instead of saying not in files"
        elif e["source"] == "web":
            # a public fact is only answered when the web escalation fired; "not in files" is
            # the failure this trap is here to catch
            row["verdict"] = "hit" if (card or {}).get("answered_from") == "web" else "wrong"
        elif e["source"] == "derived":
            # the right value for a spoken what-if is not in any key, so this asks for a card
            # with a real number on it; `answered_from` in the row says whether it recalculated
            row["verdict"] = "hit" if (card or {}).get("answered_from") not in ("none", None) else "wrong"
        elif not e["figure"]:
            row["verdict"] = "hit"
        else:
            row["verdict"] = "hit" if figure_match(e["figure"], card_text(card or {})) else "wrong"
        if row["verdict"] in ("wrong", "miss", "noise"):
            failures.append({"i": t["i"], "speaker": t["speaker"], "text": t["text"],
                             "trap": t["trap"], "verdict": row["verdict"], "expect": e,
                             "gate": row["gate"], "skip": row["skip"],
                             "transcript_final_ms": row["transcript_final_ms"],
                             "card": None if card is None else
                             {"figure": card.get("figure"), "say": card.get("say"),
                              "answer": card.get("answer"), "source": card.get("source"),
                              "answered_from": card.get("answered_from"),
                              "cached": bool(card.get("cached")),
                              "at_ms": row["first_card_ms"]}})
        rows.append(row)

    asked = [r for r in rows if r["speaker"] == "them" and r["expect"]["card"]]
    hits = [r for r in asked if r["verdict"] == "hit"]
    firsts = [max(0, r["first_card_ms"]) for r in hits if r["first_card_ms"] is not None]
    finals_ms = [max(0, r["final_card_ms"]) for r in hits if r["final_card_ms"] is not None]
    lags = [r["transcript_final_ms"] for r in rows
            if r["speaker"] == "them" and r["transcript_final_ms"] is not None]
    wrong = [r for r in rows if r["verdict"] in ("wrong", "noise")]
    redraws = [r for r in rows if r["verdict"] == "redraw"]
    missed = [r for r in asked if r["verdict"] == "miss"]
    prefetched = [r for r in hits if (r.get("first_card_kind") in ("prefetch", "instant"))
                  or (r["first_card_ms"] or 0) <= 0]
    return {
        "script": script["id"],
        "context": script["context"],
        "clip_s": timeline["duration_s"],
        "turns": len(turns),
        "them_turns": len(them),
        "questions": len(asked),
        "totals": {
            "hits": len(hits),
            "median_first_card_ms": round(statistics.median(firsts)) if firsts else None,
            "median_final_card_ms": round(statistics.median(finals_ms)) if finals_ms else None,
            "first_card_ms": sorted(firsts),
            "wrong_cards": len(wrong),
            "wrong_card_rate": round(len(wrong) / max(1, len(them)), 3),
            "redraws": len(redraws),
            "missed": len(missed),
            "miss_rate": round(len(missed) / max(1, len(asked)), 3),
            "prefetched_or_instant": len(prefetched),
            "orphan_cards": len(orphans),
            "median_transcript_final_ms": round(statistics.median(lags)) if lags else None,
            "gate_calls": len(gates),
            "gate_ms_median": round(statistics.median([g["ms"] for g in gates])) if gates else None,
            "skips": len(skips),
        },
        "skip_lines": skips,
        "cost": llm_cost(log_text),
        "providers_seen": providers_seen(log_text),
        "gates": gates,
        "rows": rows,
        "failures": failures,
    }


# ------------------------------------------------------------------------- scorer v2 (E45)

_KEY_CACHE: dict[str, dict] = {}


def key_entries(script: dict) -> dict[str, dict]:
    """The context folder's answer key, by id, so an expectation that carries no figure can
    still be checked against the entry the script picked (`expect.key_ref`)."""
    ctx = str(script.get("context") or "")
    if ctx not in _KEY_CACHE:
        try:
            _KEY_CACHE[ctx] = {e["id"]: e for e in S.load_entries((S.ROOT / ctx).resolve())}
        except (SystemExit, OSError, ValueError):
            _KEY_CACHE[ctx] = {}
    return _KEY_CACHE[ctx]


def wanted_figure(expect: dict, keys: dict[str, dict]) -> str:
    """The figure a card has to carry to answer this turn.

    `scripts.py` blanks `expect.figure` on a `request` turn ("walk me through the security and
    compliance line") while keeping the key entry it was written from, and v1 then counted any
    card at all as the answer. The key entry still holds one right number, so v2 uses it.
    """
    fig = str(expect.get("figure") or "")
    if fig and fig != "not in files":
        return fig
    if not fig and expect.get("source") == "files" and expect.get("key_ref"):
        return str((keys.get(expect["key_ref"]) or {}).get("figure") or "")
    return ""


UNPROMPTED = ("correction", "drift")     # E65: cards the platform raises with no question
UNPROMPTED_LEAD_S = 2.0                  # a drift card fires on the page turn, mid-line


def unprompted_turn(card: dict, turns: list[dict], anchor: float) -> int | None:
    """Which turn a card that answers no question belongs to, or None if it answers one.

    E9c's correction and E63's drift card are raised by something other than a client question
    -- the consultant misstating a figure, the page on screen going stale -- and neither carries
    an utterance number (`Brain._correction` and `Brain._drift_card` leave `utt` unset). The
    turn they belong to is the one whose expectation asked for them: `expect.source` is
    `correction` or `drift`, and those expectations sit on the consultant's own lines, which
    `attribute` never returns. Without this they land on whichever client turn ended last and
    are scored there as a wrong card.
    """
    from_ = card.get("answered_from")
    if from_ not in UNPROMPTED:
        return None
    at = card["_ts"] - anchor
    want = [t for t in turns
            if t["expect"].get("card") and t["expect"].get("source") == from_
            and t["start_s"] - UNPROMPTED_LEAD_S <= at <= t["end_s"] + HIT_WINDOW_S + 4.0]
    if not want:
        return None
    return min(want, key=lambda t: abs(at - t["end_s"]))["i"]


def card_states(frames: list[dict], turns: list[dict], u2t: dict[int, int],
                anchor: float, me2t: dict[int, int] | None = None) -> tuple[dict[int, list[dict]], list[dict]]:
    """Every answer frame collapsed into the card states a consultant would have seen.

    One state is one thing on screen: a card id showing one figure. The partials streaming
    toward a figure are that same card appearing, so they collapse into it and are timed at the
    first frame that carried any text; a card id that later shows a *different* figure is a
    second state, because the number in front of the consultant changed. `first_turn` is the
    turn a state was first seen on, which is what makes a carry-over visible.
    """
    per: dict[int, list[dict]] = {t["i"]: [] for t in turns}
    orphans: list[dict] = []
    for c in frames:
        if c.get("type") != "answer":
            continue
        i = unprompted_turn(c, turns, anchor)      # E65: a card nobody asked for
        if i is not None:
            per[i].append(c)
            continue
        if c.get("who") == "Me" and me2t is not None:   # E68: the consultant's own question
            i = me2t.get(_line_utt(c.get("line")) if _line_utt(c.get("line")) is not None else -1)
            if i is not None:
                per[i].append(c)
                continue
        u = c.get("utt")
        i = u2t.get(int(u)) if u is not None else None
        if i is None:
            i = attribute(c, turns, anchor)
        (per[i] if i is not None else orphans).append(c)

    out: dict[int, list[dict]] = {}
    first_turn: dict[tuple, int] = {}
    for t in sorted(turns, key=lambda t: t["i"]):
        mine = sorted(per.get(t["i"], []), key=lambda c: c["_ts"])
        by_id: dict[str, list[dict]] = {}
        for c in mine:
            by_id.setdefault(str(c.get("id")), []).append(c)
        states: list[dict] = []
        for cid, cs in by_id.items():
            # one generation = the partials streaming into a slot plus the frame that settles
            # it, which is one thing the consultant saw; the next non-partial frame under the
            # same id is the answer that replaced it
            gens: list[list[dict]] = [[]]
            for c in cs:
                gens[-1].append(c)
                if not c.get("partial"):
                    gens.append([])
            for gen in [g for g in gens if g]:
                texted = [c for c in gen if (c.get("answer") or c.get("figure"))]
                settled = [c for c in gen if not c.get("partial")]
                rep = (settled or texted or gen)[-1]
                prev = states[-1] if states and states[-1]["id"] == cid else None
                fig = str(rep.get("figure") or "")
                # E68: a no-answer card and the answer that replaces it are two things on screen
                # even when neither carries a figure ("Couldn't answer" then "Not in files")
                if prev is not None and prev["figure"] == fig \
                        and bool(prev["frame"].get("no_answer")) == bool(rep.get("no_answer")):
                    # the same number re-sent or its wording refined: one card, not two. It keeps
                    # the moment it appeared and takes the wording it settled on.
                    prev["frame"] = rep
                    prev["partial"] = bool(rep.get("partial"))
                    continue
                states.append({"id": cid, "figure": fig, "frame": rep,
                               "partial": bool(rep.get("partial")),
                               "_ts": (texted or gen)[0]["_ts"]})
        states.sort(key=lambda s: s["_ts"])
        for s in states:
            key = (s["id"], s["figure"] or str(s["frame"].get("answer") or "")[:60])
            s["first_turn"] = first_turn.setdefault(key, t["i"])
            s["carried"] = s["first_turn"] != t["i"]
            s["at_ms"] = round((s["_ts"] - (anchor + t["end_s"])) * 1000)
        out[t["i"]] = states
    return out, orphans


def answers_question(expect: dict, want: str, frame: dict) -> tuple[bool, bool]:
    """Does this card answer the turn? Returns (answers, the figure itself matched).

    The second flag is what (c) turns on: a card that was already up for an earlier turn still
    counts when the number on it is the number this turn asked for.
    """
    src, from_ = expect.get("source"), frame.get("answered_from")
    if src == "none":        # "not in files" is right, and so is a cited web page
        return from_ in ("none", "web"), False
    if src == "web":         # a public fact is only answered when the escalation fired
        return from_ == "web", False
    if src in UNPROMPTED:    # E65: the correction (E9c) and the drift card (E63)
        ok = from_ == src and (not want or figure_match(want, card_text(frame)))
        return ok, bool(ok and want)
    if src == "derived":
        # a spoken what-if has no right value in any key, so a card with a real number on it is
        # all E5 could ask for. A goal seek does have one: `presage.sim.e65` solves it with the
        # platform's own `prerun` while the script is written and puts it in `expect.figure`,
        # and then it is checked like any other figure (E45: "what this does not fix").
        real = from_ not in ("none", None, *UNPROMPTED)
        if want:
            ok = real and figure_match(want, card_text(frame))
            return ok, ok
        return real, False
    if want:
        ok = figure_match(want, card_text(frame))
        return ok, ok
    return True, False       # a card was expected and no figure can be checked


def score_v2(script: dict, timeline: dict, frames: list[dict], log_text: str, anchor: float,
             day: datetime) -> dict:
    """E45's scorer. Three rules v1 did not have, and cards counted one by one.

    A hit is a card that (a) answers the turn -- its figure matches the expected figure, or the
    key entry behind a `request` expectation that carries none -- and (b) lands within
    `HIT_WINDOW_S` of the second the question ended in the clip, and (c) was not already up for
    an earlier turn, unless the figure on it is the one this turn asked for. A card that
    answers the turn outside the window is a **late hit**, reported on its own line and counted
    as a miss in the headline. Wrong cards, noise and redraws keep v1's definitions and are
    counted per card shown rather than per turn.
    """
    turns = turn_table(script, timeline)
    them = [t for t in turns if t["speaker"] == "them"]
    u2t = utterance_turns(frames, turns)
    me2t = speaker_turns(frames, turns, "mic", "me")
    keys = key_entries(script)
    states, orphans = card_states(frames, turns, u2t, anchor, me2t)
    # E68: which turns the brain registered as a question. The `question` frame when the run has
    # one, and for a run recorded before it existed, a card carrying the line's utterance.
    registered: set[int] = set()
    for f in frames:
        if f.get("type") == "question":
            umap = me2t if f.get("source") == "mic" else u2t
            utts = f.get("utts") or ([f["utt"]] if f.get("utt") is not None else [])
            idx = [umap.get(int(x)) for x in utts]
        elif f.get("type") == "answer" and f.get("utt") is not None and f.get("who") != "Me":
            idx = [u2t.get(int(f["utt"]))]
        else:
            continue
        registered.update(i for i in idx if i is not None)

    finals_by_turn: dict[int, float] = {}
    for f in frames:
        if (f.get("type") == "transcript" and f.get("source") == "system" and f.get("final")
                and f.get("utt") is not None):
            i = u2t.get(int(f["utt"]))
            if i is not None:
                finals_by_turn[i] = max(finals_by_turn.get(i, 0.0), f["_ts"])

    gates, skips = gate_lines(log_text, day, anchor), skip_lines(log_text, day, anchor)
    window_ms = round(HIT_WINDOW_S * 1000)
    rows, failures, cards_seen = [], [], []
    for t in turns:
        e = t["expect"]
        want = wanted_figure(e, keys) if e["card"] else ""
        mine = states.get(t["i"], [])
        # E65: the expectation is the authority, not the speaker. `scripts.py` blanks every card
        # expectation on a Me line, so this is the same rule for every script written before it;
        # a hand-written script can put one there, which is the only way to score a card the
        # platform raises off the consultant's own voice (E9c) or off the page (E63).
        wants_card = bool(e["card"])
        labelled = []
        for s in mine:
            na = bool(s["frame"].get("no_answer"))       # E68: "Couldn't answer" answers nothing
            ok, fig_ok = (answers_question(e, want, s["frame"]) if wants_card and not na else (False, False))
            allowed = ok and (not s["carried"] or fig_ok)
            if s["carried"]:
                label = "redraw"
            elif not wants_card:
                label = "noise"
            elif na:
                label = "no-answer"
            elif allowed:
                label = "right" if s["at_ms"] <= window_ms else "late"
            else:
                label = "wrong"
            labelled.append(dict(s, label=label, answers=allowed, figure_matched=fig_ok))
        good = [s for s in labelled if s["answers"]]
        inside = [s for s in good if s["at_ms"] <= window_ms]
        first = min(inside, key=lambda s: s["at_ms"]) if inside else None
        late = min(good, key=lambda s: s["at_ms"]) if good and not inside else None
        settled = [s for s in labelled if not s["partial"]]
        last = (settled or labelled)[-1] if labelled else None

        if not wants_card:
            verdict = ("quiet" if not labelled else
                       "redraw" if all(s["label"] == "redraw" for s in labelled) else "noise")
        elif first is not None:
            verdict = "hit"
        elif late is not None:
            verdict = "late hit"
        else:
            verdict = "miss"
        row = {
            "i": t["i"], "speaker": t["speaker"], "text": t["text"], "trap": t["trap"],
            "end_s": t["end_s"], "expect": e, "want_figure": want, "verdict": verdict,
            "cards": len(labelled),
            "card_labels": [s["label"] for s in labelled],
            "card_id": (first or late or last or {}).get("id"),
            "first_card_ms": first["at_ms"] if first is not None else None,
            "late_card_ms": late["at_ms"] if late is not None else None,
            "any_card_ms": labelled[0]["at_ms"] if labelled else None,
            "final_card_ms": last["at_ms"] if last is not None else None,
            "figure": (first or late or last or {}).get("figure"),
            "say": (first or late or last or {}).get("frame", {}).get("say"),
            "answered_from": (first or late or last or {}).get("frame", {}).get("answered_from"),
            "cached": bool((first or late or last or {}).get("frame", {}).get("cached")),
            "card_source": (first or late or last or {}).get("frame", {}).get("source"),
            "carried_cards": sum(1 for s in labelled if s["carried"]),
            "wrong_cards": sum(1 for s in labelled if s["label"] == "wrong"),
            "noise_cards": sum(1 for s in labelled if s["label"] == "noise"),
            "redraw_cards": sum(1 for s in labelled if s["label"] == "redraw"),
            "late_cards": sum(1 for s in labelled if s["label"] == "late"),
            "no_answer_cards": sum(1 for s in labelled if s["label"] == "no-answer"),
            "registered": t["i"] in registered,
            "transcript_final_ms": (round((finals_by_turn[t["i"]] - (anchor + t["end_s"])) * 1000)
                                    if t["i"] in finals_by_turn else None),
            "gate": next((g for g in gates
                          if t["end_s"] - 0.2 <= g["at_s"] <= t["end_s"] + 6.0), None),
            "skip": next((s for s in skips
                          if t["end_s"] - 0.2 <= s["at_s"] <= t["end_s"] + 8.0), None),
        }
        if first is not None:
            fr = first["frame"]
            row["first_card_kind"] = ("partial" if fr.get("partial")
                                      else ("instant" if fr.get("cached") else fr.get("source", "?")))
        wrong_before = 0
        for s in labelled:
            if s["label"] == "wrong":
                wrong_before += 1
            cards_seen.append({"i": t["i"], "speaker": t["speaker"], "label": s["label"],
                               "at_ms": s["at_ms"], "figure": s["figure"], "id": s["id"],
                               "carried": s["carried"], "partial": s["partial"],
                               "first_turn": s["first_turn"],
                               "source": s["frame"].get("source"),
                               "answered_from": s["frame"].get("answered_from"),
                               "over_wrong": s["label"] == "redraw" and wrong_before > 0})
        if verdict in ("miss", "late hit", "noise") or row["wrong_cards"]:
            worst = (late if verdict == "late hit" else
                     next((s for s in labelled if s["label"] in ("wrong", "noise")), last))
            fr = (worst or {}).get("frame") or {}
            failures.append({"i": t["i"], "speaker": t["speaker"], "text": t["text"],
                             "trap": t["trap"], "verdict": verdict, "expect": e,
                             "want_figure": want, "gate": row["gate"], "skip": row["skip"],
                             "transcript_final_ms": row["transcript_final_ms"],
                             "card": None if not worst else
                             {"figure": worst.get("figure"), "say": fr.get("say"),
                              "answer": fr.get("answer"), "source": fr.get("source"),
                              "answered_from": fr.get("answered_from"),
                              "cached": bool(fr.get("cached")), "carried": worst.get("carried"),
                              "at_ms": worst.get("at_ms")}})
        rows.append(row)

    asked = [r for r in rows if r["speaker"] == "them" and r["expect"]["card"]]
    # E65: the cards nobody asked for are counted beside the questions, never inside them, so
    # `questions`, `hits` and the miss rate stay on E5's and E45's scale.
    unasked = [r for r in rows if r["speaker"] == "me" and r["expect"]["card"]]
    hits = [r for r in asked if r["verdict"] == "hit"]
    late_hits = [r for r in asked if r["verdict"] == "late hit"]
    blank = [r for r in asked if r["verdict"] == "miss"]
    firsts = [max(0, r["first_card_ms"]) for r in hits if r["first_card_ms"] is not None]
    late_ms = sorted(r["late_card_ms"] for r in late_hits if r["late_card_ms"] is not None)
    finals_ms = [max(0, r["final_card_ms"]) for r in hits if r["final_card_ms"] is not None]
    lags = [r["transcript_final_ms"] for r in rows
            if r["speaker"] == "them" and r["transcript_final_ms"] is not None]
    wrong_n = sum(1 for c in cards_seen if c["label"] == "wrong")
    noise_n = sum(1 for c in cards_seen if c["label"] == "noise")
    redraw_n = sum(1 for c in cards_seen if c["label"] == "redraw")
    over_wrong = sum(1 for c in cards_seen if c["over_wrong"])
    prefetched = [r for r in hits if (r.get("first_card_kind") in ("prefetch", "instant"))
                  or (r["first_card_ms"] or 0) <= 0]
    # E68: registration, and the consultant's side. A consultant question turn is one the script
    # wrote as a question to the client (the `me-question` trap); every other consultant turn is
    # a line nothing should register.
    me_rows = [r for r in rows if r["speaker"] == "me"]
    me_q = [r for r in me_rows if r["trap"] == "me-question" or r["expect"].get("kind") == "question"]
    them_quiet = [r for r in rows if r["speaker"] == "them" and not r["expect"]["card"]]
    no_answer_n = sum(1 for c in cards_seen if c["label"] == "no-answer")
    return {
        "script": script["id"],
        "context": script["context"],
        "clip_s": timeline["duration_s"],
        "turns": len(turns),
        "them_turns": len(them),
        "questions": len(asked),
        "hit_window_s": HIT_WINDOW_S,
        "totals": {
            "hits": len(hits),
            "late_hits": len(late_hits),
            "median_first_card_ms": round(statistics.median(firsts)) if firsts else None,
            "median_final_card_ms": round(statistics.median(finals_ms)) if finals_ms else None,
            "first_card_ms": sorted(firsts),
            "late_hit_ms": late_ms,
            "wrong_cards": wrong_n + noise_n,
            "wrong_on_questions": wrong_n,
            "noise_cards": noise_n,
            "wrong_card_rate": round((wrong_n + noise_n) / max(1, len(them)), 3),
            "redraws": redraw_n,
            "redraws_over_a_wrong_card": over_wrong,
            "cards_shown": len(cards_seen),
            "missed": len(late_hits) + len(blank),
            "missed_no_card": len(blank),
            "miss_rate": round((len(late_hits) + len(blank)) / max(1, len(asked)), 3),
            "prefetched_or_instant": len(prefetched),
            "unprompted": len(unasked),
            "registered": sum(1 for r in asked if r["registered"]),
            "registered_false": sum(1 for r in them_quiet if r["registered"]),
            "me_question_turns": len(me_q),
            "me_registered": sum(1 for r in me_q if r["registered"]),
            "me_registered_false": sum(1 for r in me_rows if r["registered"] and r not in me_q),
            "no_answer_cards": no_answer_n,
            "silent_questions": sum(1 for r in asked if not r["cards"]),
            "unprompted_hits": sum(1 for r in unasked if r["verdict"] == "hit"),
            "unprompted_kinds": sorted(str(r["expect"].get("source")) for r in unasked),
            "orphan_cards": len(orphans),
            "median_transcript_final_ms": round(statistics.median(lags)) if lags else None,
            "gate_calls": len(gates),
            "gate_ms_median": round(statistics.median([g["ms"] for g in gates])) if gates else None,
            "skips": len(skips),
        },
        "cards": cards_seen,
        "skip_lines": skips,
        "cost": llm_cost(log_text),
        "providers_seen": providers_seen(log_text),
        "gates": gates,
        "rows": rows,
        "failures": failures,
    }


# ----------------------------------------------------------------------------- reporting

def score_md(sc: dict, meta: dict) -> str:
    t = sc["totals"]
    out = [f"# {sc['script']}", "",
           f"Context `{sc['context']}`, clip {sc['clip_s']:.0f}s, {sc['them_turns']} client "
           f"turns, {sc['questions']} of them expecting a card.",
           f"Audio clock +/- {meta['clock']['uncertainty_ms']} ms "
           f"({meta['clock']['lo_from']} .. {meta['clock']['hi_from']}). "
           f"Provider {meta.get('provider')}, gate {meta.get('decider')}, build "
           f"{(meta.get('build') or {}).get('commit')}"
           f"{' (dirty)' if (meta.get('build') or {}).get('dirty') else ''}, "
           f"load {(meta.get('load') or {}).get('before', ['?'])[0]}"
           f"{'' if meta.get('complete', True) else ', RUN INCOMPLETE: the server died mid-clip'}."
           + ("" if len(sc.get("providers_seen") or []) <= 1 else
              f" ANSWERED BY {', '.join(sc['providers_seen'])}: the provider changed under the run."),
           "",
           f"Scored under **v{sc.get('score_version', 1)}**"
           + (f", hit window {sc['hit_window_s']:.0f} s (E45)." if "hit_window_s" in sc
              else " (E5's rules, kept for comparison)."),
           "",
           "| metric | value |", "| --- | --- |",
           f"| median first card | {t['median_first_card_ms']} ms |",
           f"| median final card | {t['median_final_card_ms']} ms |",
           f"| median final transcript | {t['median_transcript_final_ms']} ms |",
           f"| hits | {t['hits']} / {sc['questions']} |"]
    if "late_hits" in t:
        out += [f"| late hits (right card, past the window) | {t['late_hits']} |",
                f"| cards shown | {t['cards_shown']} |"]
    per_turn = ("noise_cards" in t)      # v2 counts cards, so the rate is cards per client turn
    out += [f"| wrong cards | {t['wrong_cards']}"
            + (f" ({t['wrong_on_questions']} on a question, {t['noise_cards']} on a line that "
               f"wanted none), {t['wrong_card_rate']:.2f} per client turn |" if per_turn else
               f" ({t['wrong_card_rate'] * 100:.0f}% of client turns) |"),
            f"| redraws of a card already on screen | {t['redraws']}"
            + (f" ({t['redraws_over_a_wrong_card']} of them replacing a wrong card)"
               if t.get("redraws_over_a_wrong_card") else "") + " |",
            f"| missed | {t['missed']} ({t['miss_rate'] * 100:.0f}% of questions"
            + (f", {t['late_hits']} of them a late hit" if t.get("late_hits") else "") + ") |",
            f"| before the question ended | {t['prefetched_or_instant']} |"]
    if "registered" in t:      # E68
        out += [f"| questions registered | {t['registered']} / {sc['questions']}, "
                f"{t['registered_false']} client lines that were not questions |",
                f"| consultant questions registered | {t['me_registered']} / {t['me_question_turns']}, "
                f"{t['me_registered_false']} other consultant lines |",
                f"| no-answer cards | {t['no_answer_cards']}; questions with nothing on screen at all "
                f"{t['silent_questions']} |"]
    out += [
            f"| gate calls | {t['gate_calls']} (median {t['gate_ms_median']} ms), "
            f"{t['skips']} skips |",
            f"| cost | ${sc['cost']['usd']:.4f} on the API, "
            f"${sc['cost']['subscription_list_usd']:.2f} list on the subscription |", "",
            "## Turns", "",
            "| # | who | end s | expects | verdict | final transcript | first card | final card "
            "| figure |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sc["rows"]:
        e = r["expect"]
        fig = r.get("want_figure") or e["figure"]
        want = (f"{e['kind']}/{e['source']}" + (f" {fig}" if fig else "")
                if e["card"] else "no card")
        at = r["first_card_ms"] if r["first_card_ms"] is not None else r.get("late_card_ms")
        out.append(f"| {r['i']} | {r['speaker']} | {r['end_s']:.1f} | {want} | {r['verdict']} | "
                   f"{r['transcript_final_ms'] if r['transcript_final_ms'] is not None else ''} | "
                   f"{at if at is not None else ''} | "
                   f"{r['final_card_ms'] if r['final_card_ms'] is not None else ''} | "
                   f"{(r['figure'] or '')[:18]} |")
    out += ["", "## Failures", ""]
    if not sc["failures"]:
        out.append("None.")
    for f in sc["failures"]:
        g = f["gate"]
        gate = (f"gate {g['kind']} p={g['p']:.2f} needs={g['needs']} complete={g['complete']} "
                f"({g['ms']} ms)" if g else "no gate line: the heuristic path decided it")
        if f.get("skip"):
            gate += f"; brain said \"{f['skip']['why']}\""
        card = f["card"]
        said = ("no card" if not card else
                f"{card['source']}/{card['answered_from']} figure={card['figure']!r} "
                f"say={card['say']!r} at {card['at_ms']} ms"
                + (" (already on screen from an earlier turn)" if card.get("carried") else ""))
        e = f["expect"]
        fig = f.get("want_figure") or e["figure"]
        want = (f"a card, {e['kind']} from {e['source']}"
                + (f", figure {fig}" if fig else "")) if e["card"] else "no card"
        out += [f"**{f['verdict']} · turn {f['i']} ({f['speaker']})** {f['text']}", "",
                f"- expected: {want}" + (f" (trap: {f['trap']})" if f["trap"] else ""),
                f"- {gate}; final transcript {f['transcript_final_ms']} ms after the turn ended",
                f"- card: {said}", ""]
    return "\n".join(out) + "\n"


def summary_md(runs: list[dict]) -> str:
    late = any("late_hits" in sc["totals"] for sc in runs)
    head = "| script | questions | hits | median first card | wrong cards | misses |"
    if late:
        head = ("| script | questions | hits | late hits | median first card | wrong cards "
                "| misses |")
    out = [head, "| --- |" * (7 if late else 6) + ""]
    out[1] = "| " + " | ".join(["---"] * (7 if late else 6)) + " |"
    firsts: list[float] = []
    q = w = m = them = h = lh = 0
    for sc in runs:
        t = sc["totals"]
        flag = "" if sc.get("meta", {}).get("complete", True) else " (incomplete)"
        cells = [f"{sc['script']}{flag}", str(sc["questions"]), str(t["hits"])]
        if late:
            cells.append(str(t.get("late_hits", "")))
        cells += [f"{t['median_first_card_ms']} ms",
                  f"{t['wrong_cards']}" + (f" ({t['wrong_card_rate']:.2f}/turn)" if late else
                                           f" ({t['wrong_card_rate'] * 100:.0f}%)"),
                  f"{t['missed']} ({t['miss_rate'] * 100:.0f}%)"]
        out.append("| " + " | ".join(cells) + " |")
        firsts += t["first_card_ms"]
        q += sc["questions"]
        w += t["wrong_cards"]
        m += t["missed"]
        h += t["hits"]
        lh += t.get("late_hits", 0)
        them += sc["them_turns"]
    cells = ["**all**", str(q), str(h)] + ([str(lh)] if late else []) + [
        f"{round(statistics.median(firsts)) if firsts else None} ms",
        f"{w}" + (f" ({w / max(1, them):.2f}/turn)" if late
                  else f" ({w / max(1, them) * 100:.0f}%)"),
        f"{m} ({m / max(1, q) * 100:.0f}%)"]
    out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- one run

def wait_for_load(max_load: float, patience_s: float = 600.0) -> list:
    """The transcriber is on this Mac's GPU and a busy machine starves it, which shows up as
    transcripts arriving whole seconds late and, on this machine, as servers that die. So a run
    waits for the load to come down rather than quietly measuring someone else's benchmark."""
    t0 = time.time()
    while max_load and os.getloadavg()[0] > max_load and time.time() - t0 < patience_s:
        print(f"  load {os.getloadavg()[0]:.1f} > {max_load}, waiting", file=sys.stderr)
        time.sleep(20)
    load = list(os.getloadavg())
    if max_load and load[0] > max_load:
        print(f"  load is still {load[0]:.1f}: running anyway, the run is flagged",
              file=sys.stderr)
    return [round(x, 2) for x in load]


async def one(script_id: str, tail: float, port: int, live: str, run_id: str | None,
              max_load: float = 4.0, version: int = SCORE_VERSION) -> dict:
    script = S.load(script_id)
    timeline = R.load_timeline(script_id)
    context = (S.ROOT / script["context"]).resolve()
    clip = R.CLIP_DIR / timeline["them_wav"]
    mic_clip = R.CLIP_DIR / timeline["me_wav"] if timeline.get("me_wav") else None
    for p in (context, clip):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    with wave.open(str(clip), "rb") as w:
        clip_s = w.getnframes() / w.getframerate()

    if restore_settings():
        print("restored a leftover settings.json backup", file=sys.stderr)
    if not free_port(port):
        raise SystemExit(f"port {port} is busy: stop it first (pkill -f presage.server)")
    load = wait_for_load(max_load)

    out = RUN_DIR / (run_id or script_id)
    out.mkdir(parents=True, exist_ok=True)
    frames_path, log_path = out / "frames.jsonl", out / "server.log"
    day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    patched = pin_live_provider(live)
    stop = asyncio.Event()
    recorder = asyncio.create_task(record_ws(port, frames_path, stop))
    poller = Poller(port)
    proc = None
    try:
        t_launch = time.time()
        screen_script = script_id if any(t.get("screen") for t in script["turns"]) else None
        proc = start_server(context, clip, port, log_path,
                            live if live != "keep" else "auto", mic_clip, screen_script)
        poller.start()
        state = None
        while time.time() - t_launch < READY_TIMEOUT_S:
            if poller.first_running:
                state = poller.state
                break
            if proc.poll() is not None:
                raise SystemExit(f"server exited early, see {log_path}")
            await asyncio.sleep(0.2)
        if state is None:
            raise SystemExit(f"server never started listening, see {log_path}")
        ready_s = time.time() - t_launch
        deep = ((state.get("status") or {}).get("index") or {}).get("deep") or {}
        print(f"  up in {ready_s:.1f}s; deep index {deep.get('state')}; playing {clip_s:.0f}s",
              file=sys.stderr)

        end = poller.first_running + clip_s + tail
        died_at = None
        while time.time() < end:
            await asyncio.sleep(2.0)
            if proc.poll() is not None:
                died_at = round(time.time() - poller.first_running, 1)
                print(f"  server died {died_at}s into a {clip_s:.0f}s clip", file=sys.stderr)
                break
        await asyncio.sleep(0.5)
    finally:
        poller.stop.set()
        stop.set()
        with_timeout = asyncio.wait_for(recorder, timeout=10)
        try:
            await with_timeout
        except (asyncio.TimeoutError, asyncio.CancelledError):
            recorder.cancel()
        if proc is not None:
            stop_server(proc, port)
        if patched:
            restore_settings()

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    frames = read_frames(frames_path)
    clock = estimate_anchor(log_text, frames, poller.first_running, clip_s, day)
    meta = {"script": script_id, "context": script["context"], "clip": str(clip),
            "clip_s": clip_s, "tail_s": tail, "port": port, "live_choice": live,
            "provider": ((poller.state or {}).get("llm") or {}).get("provider"),
            "decider": (((poller.state or {}).get("llm") or {}).get("decider") or {}).get("kind"),
            "answer_model": ((poller.state or {}).get("llm") or {}).get("answer_model"),
            "mic_replay": str(mic_clip) if mic_clip and mic_clip.is_file() else None,
            "screen": ("scripted" if screen_script else "off"),
            "load": {"before": load, "after": [round(x, 2) for x in os.getloadavg()],
                     "max_load": max_load, "quiet": load[0] <= max_load},
            "ready_s": round(ready_s, 1), "frames": len(frames), "clock": clock,
            "died_at_s": died_at, "complete": died_at is None or died_at >= clip_s,
            "build": head(),
            "started": datetime.now().isoformat(timespec="seconds")}
    (out / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    sc = score(script, timeline, frames, log_text, clock["anchor"], day, version=version)
    sc["meta"] = meta
    (out / "score.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
    (out / "score.md").write_text(score_md(sc, meta), encoding="utf-8")
    t = sc["totals"]
    print(f"{script_id}: first card {t['median_first_card_ms']} ms, "
          f"{t['hits']}/{sc['questions']} hits, {t['wrong_cards']} wrong, {t['missed']} missed "
          f"-> {out}")
    return sc


def rescore(run_dir: Path, version: int = SCORE_VERSION, write: bool = True) -> dict:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    script = S.load(meta["script"])
    timeline = R.load_timeline(meta["script"])
    frames = read_frames(run_dir / "frames.jsonl")
    log_text = (run_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    day = datetime.fromisoformat(meta["started"]).replace(hour=0, minute=0, second=0,
                                                          microsecond=0)
    clock = estimate_anchor(log_text, frames, None, meta["clip_s"], day)
    clock["anchor"] = meta["clock"]["anchor"] if meta["clock"].get("bracket_ok") else clock["anchor"]
    sc = score(script, timeline, frames, log_text, clock["anchor"], day, version=version)
    sc["meta"] = meta
    if write:
        (run_dir / "score.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
        (run_dir / "score.md").write_text(score_md(sc, meta), encoding="utf-8")
    t = sc["totals"]
    late = f", {t['late_hits']} late" if "late_hits" in t else ""
    print(f"{meta['script']}: first card {t['median_first_card_ms']} ms, "
          f"{t['hits']}/{sc['questions']} hits{late}, {t['wrong_cards']} wrong, "
          f"{t['missed']} missed (v{version})")
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", nargs="*", help="script id(s)")
    ap.add_argument("--all", action="store_true", help="every script in docs/replay/sim/scripts")
    ap.add_argument("--tail", type=float, default=TAIL_S, help="seconds to keep recording after the clip")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--live", default="same",
                    help="live_choice for the run: same (default), auto, a provider, or keep")
    ap.add_argument("--run-id", default=None, help="output folder name (one script only)")
    ap.add_argument("--score-only", type=Path, default=None, help="rescore an existing run folder")
    ap.add_argument("--score-version", type=int, default=SCORE_VERSION,
                    help="2 (default, E45) or 1 (what E5, E34 and E17 were published with)")
    ap.add_argument("--summary", type=Path, default=None, help="write a combined table here")
    ap.add_argument("--max-load", type=float, default=4.0,
                    help="wait for the 1-minute load average to drop below this first (0 = never)")
    ap.add_argument("--retries", type=int, default=1,
                    help="rerun a script whose server died before the clip ended")
    ap.add_argument("--dry", action="store_true",
                    help="check the script(s) without a server: every expected figure is in the "
                         "folder and every expectation scores (presage.sim.check)")
    a = ap.parse_args()

    if a.dry:
        from presage.sim import check as CHECK
        ids = a.script or ([p.stem for p in sorted(S.SCRIPT_DIR.glob("*.json"))]
                           if a.all else [])
        if not ids:
            raise SystemExit("name a script, or --all")
        out = [CHECK.check(sid) for sid in ids]
        bad = [r for r in out if not r["ok"]]
        print(f"\n{len(out) - len(bad)}/{len(out)} scripts pass")
        if bad:
            sys.exit(1)
        return

    if a.score_only:
        rescore(a.score_only, a.score_version)
        return
    ids = a.script or ([p.stem for p in sorted(S.SCRIPT_DIR.glob("*.json"))] if a.all else [])
    if not ids:
        raise SystemExit("name a script, or --all")
    runs = []
    for sid in ids:
        sc = None
        for attempt in range(a.retries + 1):
            if attempt:
                print(f"  {sid}: attempt {attempt + 1}", file=sys.stderr)
                time.sleep(5)
            try:
                sc = asyncio.run(one(sid, a.tail, a.port, a.live,
                                     a.run_id if len(ids) == 1 else None, a.max_load,
                                     a.score_version))
            except SystemExit as e:          # a bad run must not end the sweep
                print(f"  {sid}: {e}", file=sys.stderr)
                continue
            if sc["meta"]["complete"]:
                break
            print(f"  {sid}: the server did not survive the clip", file=sys.stderr)
        if sc is not None:
            runs.append(sc)
        time.sleep(3)           # let the port and the helper go before the next server
    if len(runs) > 1 or a.summary:
        path = a.summary or (RUN_DIR / "summary.md")
        path.write_text(summary_md(runs), encoding="utf-8")
        print(f"\n{summary_md(runs)}\n{path}")


if __name__ == "__main__":
    main()
