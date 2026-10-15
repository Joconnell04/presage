"""Typed questions from the call's chat, treated like spoken ones.

Zoom hides the in-meeting chat from the presenter by design ("Chat can't be shared during
a screen share", Zoom Community) and has no meeting Q&A panel, so a question typed while
you present is invisible until someone reads it out. presage already owns the second
screen, which is where the Zoom community parks the chat window anyway.

    from presage.chat import watch
    async for line in watch():
        line   # {"t": 1758196800.1, "sender": "Walt De Kock",
               #  "text": "what was churn in Q3?", "app": "zoom", "via": "ax"}

Two sources, merged and de-duplicated:

  * `ax`   the chat panel read through the Accessibility API, about one second behind the
           message appearing (the helper's poll interval). Needs the Accessibility grant
           and needs the panel to be on screen. This is the live one.
  * `file` Zoom's own chat autosave file. No grant, survives a closed panel and a
           fullscreen screen share, but measured here at 7.8-54.8 s behind (median 33 s,
           n=10 meetings) because Zoom flushes it on a 60-second timer, and it only runs
           for the meeting host.

Neither reads anything the user cannot already see on their own screen, neither joins the
meeting, and nothing is sent anywhere.

Demo:  uv run python -m presage.chat --demo
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator

from . import ax as _ax
from . import parse as _parse
from . import zoomfile as _zoomfile

__all__ = ["watch", "ChatLine"]

ChatLine = dict            # {"t": float, "sender": str, "text": str, "app": str, "via": str}

_WS = re.compile(r"\s+")
_DEDUP_TTL_S = 900.0


def _dedup_key(line: ChatLine) -> tuple:
    """Same message from the AX panel and from the saved file must collide. Sender and
    text only: the two sources disagree on the timestamp by up to a minute."""
    return (_WS.sub(" ", str(line.get("sender", "")).strip().lower()),
            _WS.sub(" ", str(line.get("text", "")).strip().lower()))


async def watch(sources: tuple[str, ...] = ("ax", "file"), app: str = "auto",
                interval: float = 0.5, backfill: bool = False,
                on_event=None) -> AsyncIterator[ChatLine]:
    """Yield chat lines from every source that is available, newest last.

    A line seen on one source is not yielded again from the other for 15 minutes, so the
    file route only ever adds what the panel missed (the panel was closed, the window was
    fullscreen, the grant is not given). Sources that fail to start are skipped and
    reported through `on_event`, so a missing Accessibility grant degrades to the file
    route rather than raising.
    """
    queue: asyncio.Queue = asyncio.Queue()
    tasks = []

    async def pump(name: str, agen) -> None:
        try:
            async for line in agen:
                await queue.put(line)
        except asyncio.CancelledError:
            raise
        except Exception as e:                      # noqa: BLE001 - one dead source must not kill the other
            if on_event:
                on_event({"event": "source_failed", "source": name, "error": str(e)})
        finally:
            await queue.put(None)

    if "ax" in sources:
        tasks.append(asyncio.create_task(
            pump("ax", _ax.watch(app=app, interval=interval, backfill=backfill, on_event=on_event))))
    if "file" in sources:
        tasks.append(asyncio.create_task(
            pump("file", _zoomfile.watch(backfill=backfill))))
    if not tasks:
        return

    seen: dict[tuple, float] = {}
    live = len(tasks)
    try:
        while live:
            line = await queue.get()
            if line is None:
                live -= 1
                continue
            now = time.time()
            if len(seen) > 2000:
                seen = {k: t for k, t in seen.items() if now - t < _DEDUP_TTL_S}
            k = _dedup_key(line)
            if now - seen.get(k, -_DEDUP_TTL_S) < _DEDUP_TTL_S:
                continue
            seen[k] = now
            yield line
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
