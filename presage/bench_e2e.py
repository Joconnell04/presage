"""End-to-end latency bench, measured over the WebSocket exactly as the page sees it.

    uv run python -m presage.bench_e2e [--url ws://localhost:8765/ws] [--seconds 60]

Run against a server started in replay mode, e.g.
    PRESAGE_CONTEXT=sample_context PRESAGE_REPLAY=docs/replay/call.wav ./run.sh

For every final "system" transcript line it reports: when the final landed, when the first card
(partial, instant/cached, prefetched or live) appeared, when the final answer for it landed, and
what kind it was. Timings are relative to the final transcript frame's arrival at the client.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets


async def main(url: str, seconds: float) -> None:
    finals: list[dict] = []           # {"ts", "text", "utt"}
    cards: list[dict] = []            # every answer / candidate frame with client ts
    t_start = time.time()
    async with websockets.connect(url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        print(f"connected; mode={hello.get('settings', {}).get('mode')} audio={hello.get('audio', {}).get('state')}")
        while time.time() - t_start < seconds:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            m = json.loads(raw)
            now = time.time()
            k = m.get("type")
            if k == "transcript" and m.get("final") and m.get("source") == "system":
                finals.append({"ts": now, "text": m["text"], "utt": m.get("utt"), "id": m.get("id")})
            elif k in ("answer", "candidate"):
                m["_ts"] = now
                cards.append(m)
    # attribute cards to the final question they answer: the latest final before the card
    print(f"\n{len(finals)} questions, {len(cards)} card frames\n")
    print(f"{'question':58} {'first card':>10} {'kind':>9} {'final ans':>10} {'source':>8}  cached")
    for i, f in enumerate(finals):
        nxt = finals[i + 1]["ts"] if i + 1 < len(finals) else float("inf")
        if f.get("utt") is not None and any(c.get("utt") == f["utt"] for c in cards):
            mine = [c for c in cards if c.get("type") == "answer" and c.get("utt") == f["utt"]]
        else:
            mine = [c for c in cards if f["ts"] - 5.0 <= c["_ts"] < nxt and c.get("type") == "answer"]
        # cards that arrived before the final (prefetched while the question was being spoken) count as 0 ms
        firsts = [c for c in mine if c.get("answer")]
        if not firsts:
            print(f"{f['text'][:58]:58} {'-':>10}")
            continue
        first = min(firsts, key=lambda c: c["_ts"])
        finals_for = [c for c in firsts if not c.get("partial")]
        last = max(finals_for, key=lambda c: c["_ts"]) if finals_for else first
        dt_first = max(0.0, first["_ts"] - f["ts"])
        dt_last = max(0.0, last["_ts"] - f["ts"])
        kind = "partial" if first.get("partial") else ("instant" if first.get("cached") else first.get("source", "?"))
        print(f"{f['text'][:58]:58} {dt_first*1000:8.0f}ms {kind:>9} {dt_last*1000:8.0f}ms {last.get('source','?'):>8}  {bool(last.get('cached'))}")
        print(f"    say={last.get('say')!r} figure={last.get('figure')!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765/ws")
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.seconds))
