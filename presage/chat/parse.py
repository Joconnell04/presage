"""Turn a chat panel's accessibility text leaves into messages.

`bin/axchat` does no parsing: it walks the chat container and prints every text leaf under
it in visual order. This module is the part that has to change when Zoom or Teams moves a
label, which is why it lives in Python next to a fixture instead of inside the Swift.

The shape both clients render is the same: a sender name, a clock time, then one or more
body lines, with the sender omitted on a run of consecutive messages from one person, and
UI chrome (Reply, To:, the composer placeholder) interleaved. So the rule is:

  * a leaf that is only a clock time starts a message;
  * a leaf that is "Name 3:42 PM" starts a message and names the sender;
  * the sender is the nearest preceding leaf that is not chrome and not a time;
  * everything after the time, up to the next time, is the body;
  * chrome is dropped by an exact-match stop list, never by a substring, so a message
    that happens to say "reply" survives.

`group()` returns the whole visible panel every time; the caller diffs. That is
deliberate: a chat panel scrolls, edits and deletes, and a diff over a stable key is the
only thing that survives all three.
"""

from __future__ import annotations

import re

# 14:05, 2:05 PM, 14:05:31, 2.05 pm
TIME = re.compile(r"^\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?$")
SENDER_TIME = re.compile(r"^(?P<who>.{1,60}?)\s{1,4}(?P<when>\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?)$")
# Zoom's saved-file phrasing sometimes reaches the panel too
FROM_TO = re.compile(r"^From\s+(?P<who>.+?)\s+to\s+(?P<to>.+?)\s*:?\s*$")

# Exact matches only. Anything here is chrome, not a message.
CHROME = {
    "", "reply", "replies", "to:", "to", "send", "more", "chat", "meeting chat", "everyone",
    "type message here...", "type message here", "send to", "who can see your messages?",
    "delete", "edit", "edited", "copy", "close", "file", "emoji", "screenshot", "new",
    "message", "messages", "you", "(you)", "direct message", "(direct message)", "private",
    "saved", "chat with everyone", "start a conversation", "no messages yet",
}
# A leaf that is only punctuation or a bullet is chrome too.
PUNCT = re.compile(r"^[\W_]{1,3}$")


def is_chrome(text: str) -> bool:
    t = text.strip()
    return t.lower() in CHROME or bool(PUNCT.match(t))


def group(items: list[dict], app: str = "zoom") -> list[dict]:
    """Messages, oldest first: [{"sender", "when", "text", "app"}].

    `items` are axchat's leaves, already in visual order: [{"r", "v", "x", "y"}].
    A panel with no clock time anywhere yields nothing rather than guessing, and the
    caller can show the raw leaves instead (`python -m presage.chat --demo --raw`).
    """
    leaves = [dict(it, v=str(it.get("v", "")).strip()) for it in items]
    leaves = [it for it in leaves if it["v"]]

    has_geometry = any(float(it.get("y") or 0) for it in leaves)

    starts: list[tuple[int, str | None, str]] = []      # index, sender or None, when
    for i, it in enumerate(leaves):
        v = it["v"]
        if TIME.match(v):
            starts.append((i, None, v))
            continue
        m = SENDER_TIME.match(v)
        if m and not is_chrome(m["who"]):
            starts.append((i, m["who"].strip(), m["when"].strip()))
            continue
        m = FROM_TO.match(v)
        if m:
            starts.append((i, m["who"].strip(), ""))
    if not starts:
        return []

    # resolve every sender first: the next message's name sits inside this message's span
    senders: list[str] = []
    last_sender = ""
    for i, sender, _ in starts:
        who = sender if sender is not None else (_sender_before(leaves, i, has_geometry) or last_sender)
        if who:
            last_sender = who
        senders.append(who or "?")

    out: list[dict] = []
    for n, (i, _, when) in enumerate(starts):
        sender = senders[n]
        end = starts[n + 1][0] if n + 1 < len(starts) else len(leaves)
        body = [leaves[j]["v"] for j in range(i + 1, end) if not is_chrome(leaves[j]["v"])]
        # a sender label that sits after its own timestamp is not body
        if body and body[0] == sender:
            body = body[1:]
        # nor is the next message's sender label, which sits inside this message's span
        if body and n + 1 < len(starts) and body[-1] == senders[n + 1] and starts[n + 1][1] is None:
            body = body[:-1]
        text = "\n".join(body).strip()
        if not text:
            continue
        out.append({"sender": sender, "when": when, "text": text, "app": app})
    return out


ROW_TOLERANCE = 6.0        # points; a sender label and its clock sit on one row


def _sender_before(leaves: list[dict], i: int, has_geometry: bool) -> str | None:
    """The name that belongs to a bare timestamp at `i`, or None when the message is a
    continuation from the same sender.

    A chat panel draws the sender label and the clock on one row and the body on the next,
    so the test is geometric: the nearest non-chrome leaf above the clock counts as the
    sender only if it shares the clock's row. Without that the previous message's last
    body line would be read as a name, which is what a run of messages from one person
    looks like.

    Some trees give no positions at all. Then the fallback is strict adjacency: the name
    must be the immediately preceding leaf, with no chrome between it and the clock.
    """
    row = float(leaves[i].get("y") or 0)
    for j in range(i - 1, -1, -1):
        v = leaves[j]["v"]
        if len(v) > 60 or "\n" in v:
            return None
        if is_chrome(v) or TIME.match(v):
            if has_geometry:
                continue
            return None                    # chrome between the name and the clock: not a name
        if has_geometry:
            return v if abs(float(leaves[j].get("y") or 0) - row) <= ROW_TOLERANCE else None
        return v if j == i - 1 else None
    return None


def key(msg: dict) -> tuple:
    """The identity a caller diffs on. Deliberately not the timestamp alone: two people
    can post in the same minute, and the same person can repeat a line."""
    return (msg.get("sender", ""), msg.get("when", ""), msg.get("text", ""))
