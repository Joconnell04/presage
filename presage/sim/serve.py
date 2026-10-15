"""The server, with the page on screen driven by the script instead of by the window list.

    PRESAGE_SIM_SCRIPT=big-03-deck uv run python -m presage.sim.serve

`presage.sim.run` starts this instead of `presage.server` when a script's turns carry `screen`
blocks, and it *is* the server: it imports `presage.server` and calls its `main()`, so the
process, the API and the brain are the ones a real call runs on. One thing is replaced, before
the server starts: `presage.screen`'s module singleton, which becomes a scripted reader that
answers "what is on screen now" from the clip's own clock.

That is the same seam E43a's bench uses (`Brain.screen_state` is a callable, and the server
wires it to `presage.screen.snapshot`), so nothing in `presage/brain.py` or `presage/server.py`
is touched:

  * `screen.snapshot()` returns the page the script says is up at this second in the clip,
    re-stamped every tick, so `Brain._screen()` sees a reading inside its 10 s TTL;
  * `screen.start()` hands `server.start_screen()` this reader rather than a real poller, which
    also wires `on_change` to the server's own dispatcher, so a scripted page turn fires
    `Brain.on_screen_change` exactly as a real page turn does -- which is what raises E63's
    drift card;
  * `screen.stop()` is a no-op, so the reader survives the audio session being stopped at the
    end of the run.

No window list, no Accessibility grant, no osascript helper and no 1 Hz poll of a real screen:
the reader is a dict and a timer, so a headless run costs nothing and prompts for nothing.

The clock starts when `server.start_screen()` is called, which is the line after
`hub.audio.start()` returns and therefore within milliseconds of the replay pump's first
sample. Page turns are written into the pauses between turns, so a few hundred milliseconds of
slack cannot move one onto the wrong line.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

from presage.sim import render as R
from presage.sim import scripts as S

LEAD_S = 0.4              # the page is up this long before the turn it belongs to starts
TICK_S = 1.0              # the real poller's period, so `age` looks the same to the brain


def schedule_for(script: dict, timeline: dict | None = None) -> list[dict]:
    """[{at_s, state}] from the script's `screen` blocks, on the clip's clock.

    A block takes effect just before the turn that carries it and stays up until the next one,
    which is what a consultant walking a deck actually does.
    """
    starts = {t["i"]: t["start_s"] for t in (timeline or {}).get("turns", [])}
    out: list[dict] = []
    for t in script["turns"]:
        block = t.get("screen")
        if not block:
            continue
        at = starts.get(t["i"])
        if at is None:                       # no clip yet: fall back to the scripted pauses
            at = sum(x["pause_after_s"] for x in script["turns"][:t["i"]]) + R.LEAD_IN_S
        out.append({"at_s": max(0.0, float(at) - LEAD_S), "turn": t["i"],
                    "state": dict(block)})
    out.sort(key=lambda s: s["at_s"])
    return out


class ScriptedScreen:
    """What `presage.screen`'s watcher looks like from the server's side, driven by a schedule.

    Only the four things the server and the brain touch are real: `snapshot()`, `on_change`,
    `interval` and `_thread` (`server.screen_block` reports whether the poller is alive).
    """

    def __init__(self, schedule: list[dict], index: Any = None, interval: float = TICK_S) -> None:
        self.schedule = schedule
        self.index = index
        self.interval = float(interval)
        self.on_change = None
        self.ticks = 0
        self.cpu_s = 0.0
        self.child_cpu_s = 0.0
        self.detail = True
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._at = -1                     # index of the schedule entry currently on screen
        self._state: dict = {"status": "unavailable", "reason": "not_started", "ts": 0.0}

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> "ScriptedScreen":
        if self._thread and self._thread.is_alive():
            return self
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="presage-sim-screen", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def state(self) -> Any:
        return self._state

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._state)
        ts = float(s.get("ts") or 0.0)
        s["age"] = round(max(0.0, time.time() - ts), 3) if ts else None
        return s

    # ------------------------------------------------------------------ the clock

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.time() - self._t0

    def _due(self, at: float) -> int:
        last = -1
        for i, s in enumerate(self.schedule):
            if s["at_s"] <= at:
                last = i
        return last

    def _run(self) -> None:
        while not self._stop.is_set():
            self.ticks += 1
            now, at = time.time(), self.elapsed()
            i = self._due(at)
            changed = i != self._at and i >= 0
            with self._lock:
                if changed:
                    self._at = i
                    st = dict(self.schedule[i]["state"])
                    st.setdefault("status", "ok")
                    st.setdefault("reason", None)
                    st.setdefault("via", "sim")
                    st.setdefault("focused", True)
                    st["ts"] = now
                    self._state = st
                    print(f"  screen -> {st.get('file')} › {st.get('loc')} at {at:.1f}s",
                          file=sys.stderr, flush=True)
                else:
                    self._state["ts"] = now       # a live reading, the way a 1 Hz poller keeps it
                fresh = dict(self._state)
            if changed and self.on_change is not None:
                fresh["age"] = 0.0
                try:
                    self.on_change(fresh)
                except Exception as e:  # noqa: BLE001  a callback must never kill the reader
                    print(f"  screen on_change: {type(e).__name__}: {e}", file=sys.stderr)
            self._stop.wait(self.interval)


def install(schedule: list[dict]) -> ScriptedScreen:
    """Put the scripted reader where the process's one screen poller would be."""
    from presage import screen as screenmod

    fake = ScriptedScreen(schedule)

    def start(index: Any = None, interval: float = TICK_S, detail: bool = True) -> ScriptedScreen:
        if index is not None:
            fake.index = index
        return fake.start()

    screenmod._watcher = fake            # noqa: SLF001  the module singleton, by design
    screenmod.start = start
    screenmod.stop = lambda: None        # the reader outlives the audio session in a run
    return fake


def main() -> None:
    sid = os.environ.get("PRESAGE_SIM_SCRIPT", "").strip()
    if not sid:
        raise SystemExit("set PRESAGE_SIM_SCRIPT to a script id")
    script = S.load(sid)
    try:
        timeline = R.load_timeline(sid)
    except (OSError, ValueError):
        timeline = None
    schedule = schedule_for(script, timeline)
    if not schedule:
        raise SystemExit(f"{sid} has no screen blocks: run `python -m presage.server` instead")
    install(schedule)
    print(f"  sim screen: {len(schedule)} page(s) from {sid}"
          + ("" if timeline else " (no clip yet: pauses only)"), file=sys.stderr)
    print("  " + json.dumps([{"at_s": s["at_s"], "loc": s["state"].get("loc")}
                             for s in schedule]), file=sys.stderr)

    from presage import server                       # after the install, never before
    server.main()


if __name__ == "__main__":
    main()
