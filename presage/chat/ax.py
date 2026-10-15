"""The live route: the chat panel read through the macOS Accessibility API.

Spawns the Swift helper at presage/bin/axchat, which finds the chat container in a Zoom or
Teams window and prints every text leaf under it, in visual order, whenever they change.
This module groups those leaves into messages (presage/chat/parse.py) and yields the ones
it has not seen.

    async for line in watch():
        ...   # {"t", "sender", "text", "app": "zoom"|"teams", "via": "ax", "when"}

What it needs from the user, once:

    System Settings > Privacy & Security > Accessibility > + > Presage

That is a different grant from Screen & System Audio Recording, which presage already
asks for, and macOS never prompts for it on its own. It attaches to the app that launches
the helper, so under ~/Applications/Presage.app the row says Presage; run from a terminal
and the row says the terminal. Re-running make_app.sh re-signs the bundle ad hoc, which
changes its code identity and silently invalidates the grant: the System Settings toggle
still looks on while AXIsProcessTrusted() returns false. Remove and re-add the row after
a rebuild.

What breaks it (all MEASURED or SOURCE, see docs/experiments/E37-zoom-chat.md):

  * the screen locked: macOS returns a stub tree where every entry of AXWindows is the
    application element. The helper reports {"event":"locked"} and reads nothing.
  * a window fully covered or minimised: Chromium stops updating its accessibility tree
    when it believes the window is occluded, which is Teams, and any window on another
    Space is treated the same way.
  * a fullscreen window: it does not appear in AXWindows at all.
  * the chat panel closed: the elements do not exist, so there is nothing to read. The
    panel popped out into its own window on the second screen is the configuration this
    is built for, and is the one the Zoom community already uses.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import sys
import time
from typing import AsyncIterator

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
BIN = REPO / "bin" / "axchat"
SRC = pathlib.Path(__file__).resolve().parent / "axchat.swift"
BUILD = pathlib.Path(__file__).resolve().parent / "build.sh"

from . import parse as _parse


def ensure_built() -> pathlib.Path:
    """Build the helper if it is missing or older than its source."""
    if not BIN.exists() or BIN.stat().st_mtime < SRC.stat().st_mtime:
        subprocess.run([str(BUILD)], check=True)
    return BIN


def trusted() -> bool:
    """Whether this process's launching app holds the Accessibility grant."""
    ensure_built()
    done = subprocess.run([str(BIN), "--list"], capture_output=True, text=True, timeout=30)
    return "\"event\":\"untrusted\"" not in done.stdout


def dump(app: str = "zoom", depth: int = 14) -> str:
    """The AX subtree of that app's windows, for tuning the container locator against a
    real meeting. `python -m presage.chat --dump zoom` prints it."""
    ensure_built()
    done = subprocess.run([str(BIN), "--dump", app, str(depth)],
                          capture_output=True, text=True, timeout=120)
    return done.stdout + (("\n" + done.stderr) if done.stderr else "")


async def watch(app: str = "auto", interval: float = 0.5, backfill: bool = False,
                on_event=None) -> AsyncIterator[dict]:
    """Yield each new chat message the panel shows.

    `backfill=True` also yields what was already on screen when the watch started; the
    default skips it, because a panel that has been open all meeting would otherwise
    replay the whole call into the brain at once.

    `on_event(dict)` receives the helper's own frames (target, locked, untrusted,
    no_target) so a caller can surface "chat panel not found" without parsing stdout.
    """
    ensure_built()
    proc = await asyncio.create_subprocess_exec(
        str(BIN), "--watch", "--app", app, "--interval", str(interval),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    seen: set[tuple] = set()
    primed = backfill
    try:
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            try:
                frame = json.loads(raw)
            except ValueError:
                continue
            if frame.get("event") != "leaves":
                if frame.get("event") in ("target", "no_target"):
                    # a new panel is a new conversation: re-prime instead of replaying it
                    primed = backfill
                    seen.clear()
                if on_event:
                    on_event(frame)
                continue
            if on_event:
                on_event({k: v for k, v in frame.items() if k != "items"})
            msgs = _parse.group(frame.get("items") or [], app=frame.get("app", "zoom"))
            for m in msgs:
                k = _parse.key(m)
                if k in seen:
                    continue
                seen.add(k)
                if not primed:
                    continue
                yield {"t": float(frame.get("t") or time.time()), "sender": m["sender"],
                       "text": m["text"], "app": m["app"], "via": "ax", "when": m["when"]}
            primed = True
            if len(seen) > 4000:
                seen = set(list(seen)[-2000:])
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                proc.kill()


if __name__ == "__main__":     # pragma: no cover - convenience only
    print(dump(sys.argv[1] if len(sys.argv) > 1 else "zoom"))
