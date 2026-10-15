"""Chat reader smoke tests.

    uv run python -m presage.chat --demo            both sources, prints lines as they land
    uv run python -m presage.chat --demo --ax       the accessibility panel only
    uv run python -m presage.chat --demo --file     Zoom's saved-chat file only
    uv run python -m presage.chat --demo --raw      also print every text leaf axchat sees
    uv run python -m presage.chat --dump zoom       the AX tree of Zoom's windows
    uv run python -m presage.chat --check           grant, screen lock, what is readable
    uv run python -m presage.chat --flush-evidence  the 60-second flush measurement
    uv run python -m presage.chat --self-test       the parsers, against fixtures and real files
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import subprocess
import sys

from . import ax, parse, watch, zoomfile


def _clock(t: float) -> str:
    """Wall clock to the millisecond. The seconds alone were too coarse for the thing this
    demo exists to measure: line-to-reader latency is one poll interval, 0.5 s by default."""
    return dt.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]


async def demo(args) -> None:
    sources = tuple(s for s in ("ax", "file")
                    if (getattr(args, s) or not (args.ax or args.file)))
    print(f"watching {', '.join(sources)}; Ctrl-C to stop")

    def on_event(e: dict) -> None:
        kind = e.get("event")
        if kind == "leaves" and not args.raw:
            return
        if kind == "untrusted":
            print("  ! no Accessibility grant: System Settings > Privacy & Security > "
                  "Accessibility > + > Presage (or your terminal)")
        elif kind == "locked" and e.get("value"):
            print("  ! the screen is locked; macOS stubs every window tree until it is unlocked")
        elif kind == "target":
            print(f"  > chat panel: {e.get('app')} «{e.get('window')}» "
                  f"{e.get('leaves')} leaves at {e.get('path')}")
        elif kind == "no_target":
            print("  > no chat panel found (open the chat, or pop it out onto the second screen)")
        elif kind == "source_failed":
            print(f"  ! {e.get('source')} source failed: {e.get('error')}")
        elif kind == "leaves":
            print(f"  · {len(e.get('items', []))} leaves in {e.get('ms')} ms")

    async def raw_leaves() -> None:
        """--raw also shows what axchat sees, which is how the locator gets tuned."""
        proc = await asyncio.create_subprocess_exec(
            str(ax.ensure_built()), "--watch", "--app", args.app, "--interval", str(args.interval),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            try:
                frame = json.loads(raw)
            except ValueError:
                continue
            if frame.get("event") == "leaves":
                print(f"  --- {len(frame['items'])} leaves ({frame.get('ms')} ms) ---")
                for it in frame["items"]:
                    print(f"      {it['r']:<16} y={it['y']:<7.0f} {it['v'][:100]!r}")

    if args.raw and "ax" in sources:
        asyncio.create_task(raw_leaves())
    n = 0
    async for line in watch(sources=sources, app=args.app, interval=args.interval,
                            backfill=args.backfill, on_event=on_event):
        n += 1
        via = line.get("via", "?")
        print(f"[{_clock(line['t'])}] {line['app']}/{via}  {line['sender']}: {line['text']}")
    print(f"stopped after {n} lines")


def check() -> int:
    binary = ax.ensure_built()
    done = subprocess.run([str(binary), "--list"], capture_output=True, text=True, timeout=30)
    apps, locked, untrusted = [], False, False
    for row in done.stdout.splitlines():
        try:
            e = json.loads(row)
        except ValueError:
            continue
        if e.get("event") == "app":
            apps.append(e)
        elif e.get("event") == "locked":
            locked = bool(e.get("value"))
        elif e.get("event") == "untrusted":
            untrusted = True
    print(f"helper:        {binary}")
    print(f"accessibility: {'NOT GRANTED' if untrusted else 'granted'}")
    print(f"screen:        {'LOCKED - no window tree is readable until it is unlocked' if locked else 'unlocked'}")
    real = [a for a in apps if any(w.startswith("AXWindow:") for w in a["windows"])]
    print(f"windows:       {len(real)} of {len(apps)} apps expose a real AXWindow"
          f"{' (all stubbed: the screen lock does this)' if locked and not real else ''}")
    for a in apps:
        if any(k in a["bundle"] for k in ("zoom", "teams")):
            print(f"  {a['name']:<22} {a['bundle']:<28} {a['windows']}")
    live = zoomfile.newest()
    print(f"zoom autosave: {live if live else 'no saved-chat file in ~/Documents/Zoom'}")
    if live:
        rows = zoomfile.read(live)
        newest_at = dt.datetime.fromtimestamp(rows[-1]["t"]).strftime("%H:%M:%S") if rows else "-"
        print(f"               {len(rows)} messages, newest {newest_at}")
    return 0 if not untrusted else 1


def _leaf(v, y, x=40, r="AXStaticText"):
    return {"r": r, "v": v, "x": x, "y": y}


# The leaf shapes both clients render. These are fixtures, not captures: no live Zoom
# meeting was available to dump (see docs/experiments/E37-zoom-chat.md, "What is not
# measured"). Re-cut them from `--dump zoom` output against a real panel.
FIXTURE = [
    _leaf("Walt De Kock", 100), _leaf("14:05", 100, x=180),
    _leaf("what was enterprise churn in Q3?", 120),
    _leaf("Reply", 120, x=300, r="AXButton"),
    _leaf("14:06", 150, x=180), _leaf("and per month, not per quarter", 170),
    _leaf("Priya N  2:07 PM", 200), _leaf("can you show the bridge again", 220),
    _leaf("Type message here...", 400),
]

CASES: list[tuple[str, list[dict], list[tuple[str, str]]]] = [
    ("a run of messages from one sender keeps the sender", FIXTURE,
     [("Walt De Kock", "what was enterprise churn in Q3?"),
      ("Walt De Kock", "and per month, not per quarter"),
      ("Priya N", "can you show the bridge again")]),
    ("the same panel with no positions at all", [dict(it, x=0, y=0) for it in FIXTURE],
     [("Walt De Kock", "what was enterprise churn in Q3?"),
      ("Walt De Kock", "and per month, not per quarter"),
      ("Priya N", "can you show the bridge again")]),
    ("a new speaker's label does not land in the previous body",
     [_leaf("Walt De Kock", 100), _leaf("14:05", 100, x=180), _leaf("body A", 120),
      _leaf("Priya N", 150), _leaf("14:07", 150, x=180), _leaf("body C", 170)],
     [("Walt De Kock", "body A"), ("Priya N", "body C")]),
    ("a wrapped message keeps its lines",
     [_leaf("Walt", 100), _leaf("14:05", 100, x=180), _leaf("line one", 120), _leaf("line two", 140)],
     [("Walt", "line one\nline two")]),
    ("Teams draws the name and clock as one leaf",
     [_leaf("Priya N  2:07 PM", 200), _leaf("can you show the bridge again", 220)],
     [("Priya N", "can you show the bridge again")]),
    ("chrome is matched whole, so a message may say reply",
     [_leaf("Walt", 100), _leaf("14:05", 100, x=180), _leaf("please reply to the RFP", 120)],
     [("Walt", "please reply to the RFP")]),
    ("a panel with no clock time yields nothing rather than a guess",
     [_leaf("just some text", 0, x=0)], []),
]


def self_test() -> int:
    bad = 0
    print("ax leaves -> messages:")
    for name, leaves, want in CASES:
        got = [(m["sender"], m["text"]) for m in parse.group(leaves)]
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            print(f"       got  {got}\n       want {want}")

    files = zoomfile.meetings()
    print(f"\nzoom saved-chat files: {len(files)}")
    total = 0
    for f in files:
        rows = zoomfile.read(f)
        total += len(rows)
        if not rows:
            print(f"  FAIL: parsed 0 messages from {f}")
            bad += 1
        for r in rows:
            if not r["sender"] or not r["text"] or r["t"] <= 0:
                print(f"  FAIL: empty field in {r}")
                bad += 1
    print(f"  {total} messages parsed, every one with a sender, a body and a time")

    legacy = zoomfile.parse("14:05:31 From Walt De Kock to Everyone:\tolder client layout\n",
                            day="2026-09-18")
    if len(legacy) != 1 or legacy[0]["text"] != "older client layout":
        print(f"  FAIL: legacy meeting_saved_chat.txt layout: {legacy}")
        bad += 1
    else:
        print("  legacy single-line layout parses too")
    print("\nOK" if not bad else f"\n{bad} FAILURES")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m presage.chat", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="print chat lines as they appear")
    p.add_argument("--ax", action="store_true", help="accessibility panel only")
    p.add_argument("--file", action="store_true", help="Zoom's saved-chat file only")
    p.add_argument("--raw", action="store_true", help="also print every text leaf axchat reads")
    p.add_argument("--backfill", action="store_true", help="also emit what is already on screen")
    p.add_argument("--app", default="auto", choices=["auto", "zoom", "teams"])
    p.add_argument("--interval", type=float, default=0.5, help="AX poll seconds (default 0.5)")
    p.add_argument("--dump", metavar="APP", help="print that app's AX tree and exit")
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--check", action="store_true", help="grant, screen lock, what is readable")
    p.add_argument("--flush-evidence", action="store_true",
                   help="birth, mtime and message times of every saved-chat file")
    p.add_argument("--self-test", action="store_true", help="parsers, against fixtures and real files")
    args = p.parse_args()
    # The fixture runs `--demo --raw` into a log file while the chat is driven from another
    # terminal, and a redirected stdout is block-buffered, so nothing appears until 8 KB of
    # leaves have piled up. Line buffering costs nothing and keeps the log live.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):       # not a text stream
        pass

    if args.dump:
        print(ax.dump(args.dump, args.depth))
        return 0
    if args.check:
        return check()
    if args.flush_evidence:
        for line in zoomfile.flush_evidence():
            print(line)
        return 0
    if args.self_test:
        return self_test()
    if not args.demo:
        p.print_help()
        return 0
    try:
        asyncio.run(demo(args))
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
