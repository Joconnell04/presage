"""Zoom's own chat autosave file, tailed while the meeting runs.

With "Save chat automatically" on, the Zoom desktop client writes every in-meeting chat
message to

    ~/Documents/Zoom/<YYYY-MM-DD HH.MM.SS Meeting name>/meeting_saved_new_chat.txt

(`meeting_saved_chat.txt` on older clients; both names are in this machine's
zVideoAppPlugin binary, and every folder written here since 2026-07 uses the `_new_` one).

The file is not appended per message. Measured on the ten meetings in this Mac's
~/Documents/Zoom on 2026-09-22: the file is created with the meeting folder, and its
mtime is always an exact whole number of minutes after its birth time (+30:00, +41:00,
+1:00, +5:00, and +0:00 for the six meetings whose chat stopped inside the first minute).
Ten files, no exceptions. So Zoom flushes on a 60-second timer anchored at the meeting
start and only when there is something new, which puts a typed line on disk 0-60 s after
it was sent, ~30 s on average. That is a backstop, not a live feed; presage/chat/ax.py is
the live one. Two other limits, from Zoom's own support article: the autosave runs for the
host only, and the setting lives in the web portal where an admin can lock it.

    async for line in watch():
        ...   # {"t", "sender", "to", "text", "app": "zoom", "via": "file"}

`python -m presage.chat --file` runs it on its own.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import pathlib
import re
import time
from typing import AsyncIterator, Iterator

ZOOM_DIR = pathlib.Path.home() / "Documents" / "Zoom"
NAMES = ("meeting_saved_new_chat.txt", "meeting_saved_chat.txt")

# 2026-09-18 13:53:21 From Walt De Kock to Everyone:
#         That was under 10 - gotta revise the average down
HEADER = re.compile(r"^(?P<when>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+From\s+(?P<who>.+?)\s+to\s+(?P<to>.+?)\s*:\s*$")
# the pre-2022 layout: clock time only, and the message on the same line after a tab
LEGACY = re.compile(r"^(?P<when>\d{2}:\d{2}:\d{2})\s+From\s+(?P<who>.+?)\s+to\s+(?P<to>.+?)\s*:\s*(?P<text>.*)$")
FOLDER_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2})\.(\d{2})\.(\d{2}) ?(.*)$")


def parse(text: str, day: str | None = None) -> list[dict]:
    """Every message in one saved-chat file, oldest first.

    `day` (YYYY-MM-DD) dates the legacy layout, which only carries a clock time; it comes
    from the containing folder's name. Message bodies are tab-indented continuation lines,
    so a multi-line message keeps its newlines and a trailing tab-only line is a separator.
    """
    out: list[dict] = []
    cur: dict | None = None
    body: list[str] = []

    def close() -> None:
        nonlocal cur, body
        if cur is None:
            return
        joined = "\n".join(body).strip()
        if joined:
            cur["text"] = joined
            out.append(cur)
        cur, body = None, []

    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = HEADER.match(raw)
        if m:
            close()
            cur = {"t": _epoch(m["when"]), "sender": m["who"].strip(), "to": m["to"].strip()}
            continue
        m = LEGACY.match(raw)
        if m:
            close()
            when = f"{day} {m['when']}" if day else m["when"]
            out.append({"t": _epoch(when), "sender": m["who"].strip(),
                        "to": m["to"].strip(), "text": m["text"].strip()})
            continue
        if cur is not None and (raw.startswith("\t") or raw.startswith("    ")):
            body.append(raw.lstrip("\t").rstrip())
    close()
    for row in out:
        row["app"] = "zoom"
        row["via"] = "file"
    return out


def _epoch(when: str) -> float:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(when, fmt)
        except ValueError:
            continue
        if fmt == "%H:%M:%S":       # no date in the legacy layout and no folder to borrow one from
            today = dt.date.today()
            parsed = parsed.replace(year=today.year, month=today.month, day=today.day)
        return parsed.timestamp()
    return time.time()


def meetings(root: pathlib.Path = ZOOM_DIR) -> list[pathlib.Path]:
    """Every meeting folder that holds a saved chat file, newest first."""
    if not root.is_dir():
        return []
    found = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        for n in NAMES:
            f = d / n
            if f.is_file():
                found.append(f)
                break
    return sorted(found, key=lambda f: f.stat().st_mtime, reverse=True)


def newest(root: pathlib.Path = ZOOM_DIR, within_s: float = 6 * 3600) -> pathlib.Path | None:
    """The saved-chat file of the meeting that is most likely the live one: the newest
    folder whose name timestamp is inside `within_s`. Falls back to the newest file."""
    files = meetings(root)
    now = time.time()
    for f in files:
        m = FOLDER_TS.match(f.parent.name)
        if not m:
            continue
        stamp = f"{m[1]} {m[2]}:{m[3]}:{m[4]}"
        if 0 <= now - _epoch(stamp) <= within_s:
            return f
    return files[0] if files else None


def read(path: pathlib.Path) -> list[dict]:
    m = FOLDER_TS.match(path.parent.name)
    rows = parse(path.read_text(encoding="utf-8", errors="replace"), day=m[1] if m else None)
    for row in rows:
        row["meeting"] = (m[5] if m else path.parent.name) or path.parent.name
    return rows


def _key(row: dict) -> tuple:
    return (round(row["t"]), row["sender"], row["text"])


async def watch(root: pathlib.Path = ZOOM_DIR, interval: float = 1.0,
                backfill: bool = False) -> AsyncIterator[dict]:
    """Yield each new message as Zoom flushes it.

    Zoom rewrites the whole file rather than appending, so this re-reads on any mtime or
    size change and de-duplicates on (second, sender, text) instead of tracking an offset.
    A new meeting folder is picked up without a restart. `backfill=True` also yields what
    was already in the file when the watch started.
    """
    seen: set[tuple] = set()
    path: pathlib.Path | None = None
    stamp: tuple | None = None
    while True:
        live = newest(root)
        if live != path:
            path, stamp = live, None
            if not backfill and path is not None:
                try:
                    seen.update(_key(r) for r in read(path))
                except OSError:
                    pass
        if path is not None:
            try:
                st = path.stat()
                now = (st.st_mtime, st.st_size)
                if now != stamp:
                    stamp = now
                    for row in read(path):
                        k = _key(row)
                        if k not in seen:
                            seen.add(k)
                            yield row
            except OSError:
                pass
        await asyncio.sleep(interval)


def flush_evidence(root: pathlib.Path = ZOOM_DIR) -> Iterator[str]:
    """One line per saved-chat file: birth, mtime, mtime-birth, and the message times.
    This is the measurement behind the 60-second claim in this module's docstring."""
    for f in sorted(meetings(root), key=lambda p: p.parent.name):
        st = f.stat()
        birth = getattr(st, "st_birthtime", st.st_ctime)
        rows = read(f)
        delta = st.st_mtime - birth
        last = max((r["t"] for r in rows), default=birth)
        yield (f"{f.parent.name[:19]}  birth={_hms(birth)} mtime={_hms(st.st_mtime)} "
               f"delta={delta:7.1f}s ({delta / 60:.3f} min)  msgs={len(rows)} "
               f"last_msg={_hms(last)} flush_lag={st.st_mtime - last:6.1f}s")


def _hms(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")
