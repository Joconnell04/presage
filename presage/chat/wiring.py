"""How a chat line would reach the brain. Written, not wired: nothing in presage imports
this module, and `presage/server.py` is unchanged.

To turn it on, three lines in `presage/server.py`:

    from presage.chat import wiring as chat_wiring          # next to the other imports
    ...
    chat_wiring.attach(hub)                                 # in startup(), after hub.brain exists
    ...
    chat_wiring.detach()                                    # in the shutdown path, beside audio.stop()

and one switch in Settings, because reading the chat panel needs an Accessibility grant
the user has to give by hand and should be able to take back.

The shape of the decision, and why it is this one:

  * **The UI frame carries `source: "chat"`.** `docs/ui-protocol.md` says the page ignores
    frames and fields it does not recognise, and it renders anything that is not `mic` on
    the Them side, so an unpatched page shows chat lines in the Them column today and a
    later patch can badge them without a protocol change. `speaker` carries the sender,
    which is the field the protocol already defines for "Them A".

  * **The brain is fed `source="system"`.** Not `"chat"`. Every gating rule in
    `brain._on_final` keys off `u.source == "system"`: the pre-gate veto, the full gate,
    the cue match, and the rule that a line from Me never gets a card. A new source string
    would silently route around all four and a typed line would get a card the same
    sentence spoken would not. A chat question is a question from Them; it should have to
    clear the same bar.

  * **Chat gets its own utterance numbers, offset by `UTT_BASE`.** The transcript id is
    `f"t{source[0]}{utt}"` and `brain._utt_answers` is keyed on `(source, utt)`, so chat
    sharing the system counter would collide with the transcriber's ids and overwrite
    cards. Starting at 1,000,000 keeps both unique without a second id scheme.

  * **Your own chat messages are dropped.** A line you typed is a line from Me, and Me
    lines are not questions for the copilot outside Interview mode. `me` is matched
    case-insensitively against the sender; the display name Zoom puts in the chat is the
    one in the saved file, so it can be read from there rather than configured.

  * **`final=True`, `question_forming=False`.** A typed line arrives complete. There is no
    partial to prefetch on, which costs the speculative head start a spoken question gets:
    a chat question is answered live, from the moment it is read.

  * **Nothing is fed while presage is not listening**, so a chat panel left open between
    calls cannot quietly drive the brain.
"""

from __future__ import annotations

import asyncio
import logging
import time

log = logging.getLogger("presage.chat")

UTT_BASE = 1_000_000          # chat utterance numbers start here; the transcriber's never reach it
_task: asyncio.Task | None = None
_seq = 0


async def on_chat(hub, line: dict, me: str = "") -> None:
    """One chat line -> the page, and the brain, exactly as a Them line."""
    from presage.brain import Utterance          # local: this module is importable without the brain

    text = (line.get("text") or "").strip()
    sender = (line.get("sender") or "").strip()
    if not text or (me and sender.lower() == me.strip().lower()):
        return
    if not hub.status.get("listening"):
        return

    global _seq
    _seq += 1
    utt = UTT_BASE + _seq
    tid = f"ts{utt}"                              # "system" + a number the transcriber never reaches
    now = time.time()
    t = float(line.get("t") or now)

    await hub.broadcast({
        "type": "transcript", "id": tid, "source": "chat", "text": text, "final": True,
        "t0": t, "t1": t, "ts": now, "utt": utt, "retracts": False,
        "speaker": sender or "?", "app": line.get("app", "zoom"), "via": line.get("via", "ax"),
    })
    if not hub.brain:
        return
    fields = set(getattr(Utterance, "__dataclass_fields__", {}))
    kw = {"id": tid, "source": "system", "text": text, "final": True, "ts": now}
    if "utt" in fields:
        kw["utt"] = utt
    if "retracts" in fields:
        kw["retracts"] = False
    if "question_forming" in fields:
        kw["question_forming"] = False
    asyncio.create_task(hub.brain.on_transcript(Utterance(**kw)))


def attach(hub, me: str = "", sources: tuple[str, ...] = ("ax", "file"),
           app: str = "auto") -> asyncio.Task:
    """Start the chat watcher. Returns the task; safe to call twice."""
    global _task
    if _task and not _task.done():
        return _task

    async def run() -> None:
        from presage.chat import watch

        def on_event(e: dict) -> None:
            if e.get("event") in ("untrusted", "no_target", "source_failed"):
                log.info("chat: %s", e)
            hub.status["chat"] = {**hub.status.get("chat", {}), **e}

        while True:
            try:
                async for line in watch(sources=sources, app=app, on_event=on_event):
                    try:
                        await on_chat(hub, line, me=me)
                    except Exception as e:      # noqa: BLE001 - one bad line must not stop the watch
                        log.warning("chat line failed: %s", e)
            except asyncio.CancelledError:
                raise
            except Exception as e:              # noqa: BLE001
                log.warning("chat watch died (%s); retrying in 10 s", e)
            await asyncio.sleep(10)

    _task = asyncio.create_task(run())
    return _task


def detach() -> None:
    global _task
    if _task and not _task.done():
        _task.cancel()
    _task = None
