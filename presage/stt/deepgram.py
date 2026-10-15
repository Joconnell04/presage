"""Deepgram Nova-3 streaming provider. UNTESTED — no API key exists on this machine.

Written against the current docs (developers.deepgram.com/reference/speech-to-text/
listen-streaming, fetched 2026-09-18). It has never made a network call. Treat the
first run as a debugging session.

One websocket per source. Deepgram supports multichannel=true with interleaved audio,
but presage captures mic and system as two independent streams that arrive at different
times, and interleaving them would mean buffering to align them — latency for nothing.

Uses the `websockets` library directly rather than deepgram-sdk: it is one dependency
instead of a stack, and the message shapes are three fields we care about.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator
from urllib.parse import urlencode

import numpy as np

from .base import SOURCES, STTProvider, Transcript

log = logging.getLogger("presage.stt.deepgram")

ENDPOINT = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-3")
KEEPALIVE_S = 5.0

DEFAULT_KEYTERMS = [
    "EBITDA", "ARR", "NRR", "CAC", "churn", "cohort", "headcount",
    "gross margin", "net revenue retention", "payback period", "runway",
]


def _params(keyterms: list[str]) -> str:
    p = [
        ("model", DEFAULT_MODEL),
        ("encoding", "linear16"),
        ("sample_rate", "16000"),
        ("channels", "1"),
        ("interim_results", "true"),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("endpointing", os.environ.get("DEEPGRAM_ENDPOINTING", "300")),
        ("utterance_end_ms", os.environ.get("DEEPGRAM_UTTERANCE_END_MS", "1000")),
        ("vad_events", "true"),
        ("language", os.environ.get("DEEPGRAM_LANGUAGE", "en")),
    ]
    # keyterm is Nova-3 only and is repeated once per term.
    p += [("keyterm", k) for k in keyterms if k]
    return urlencode(p)


class DeepgramProvider(STTProvider):
    name = "deepgram"

    def __init__(self, keyterms: list[str] | None = None):
        env = os.environ.get("DEEPGRAM_KEYTERMS", "")
        self.keyterms = keyterms or (
            [k.strip() for k in env.split(",") if k.strip()] or DEFAULT_KEYTERMS
        )
        self._ws: dict[str, object] = {}
        self._tasks: list[asyncio.Task] = []
        self._q: asyncio.Queue[Transcript | None] = asyncio.Queue()
        self._t_zero = 0.0
        self._utt_start: dict[str, float] = {s: 0.0 for s in SOURCES}
        # Same contract as the local provider: the server keys transcript ids on
        # (source, utt), so every provider has to number its utterances or every
        # question on a source collapses into one line.
        self._utt: dict[str, int] = {s: 0 for s in SOURCES}
        self._closing = False

    async def start(self) -> None:
        key = os.environ.get("DEEPGRAM_API_KEY")
        if not key:
            log.error(
                "DEEPGRAM_API_KEY is not set. Either export it or run with "
                "STT_PROVIDER=local (the default)."
            )
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        try:
            import websockets
        except ImportError as e:
            raise RuntimeError(
                "the `websockets` package is required for STT_PROVIDER=deepgram"
            ) from e

        url = f"{ENDPOINT}?{_params(self.keyterms)}"
        headers = {"Authorization": f"Token {key}"}
        self._t_zero = time.perf_counter()

        for source in SOURCES:
            # websockets >= 14 renamed extra_headers to additional_headers.
            try:
                ws = await websockets.connect(url, additional_headers=headers)
            except TypeError:
                ws = await websockets.connect(url, extra_headers=headers)
            self._ws[source] = ws
            self._tasks.append(asyncio.create_task(self._reader(source, ws)))
            self._tasks.append(asyncio.create_task(self._keepalive(ws)))
        log.info("deepgram %s connected, %d keyterms", DEFAULT_MODEL, len(self.keyterms))

    async def feed(self, source: str, pcm: np.ndarray) -> None:
        ws = self._ws.get(source)
        if ws is None:
            raise RuntimeError("DeepgramProvider.feed() before start()")
        if pcm.dtype != np.int16:
            raise TypeError(f"feed() wants int16, got {pcm.dtype}")
        await ws.send(pcm.tobytes())

    async def _keepalive(self, ws) -> None:
        # Deepgram closes an idle socket after ~10 s of no audio.
        try:
            while not self._closing:
                await asyncio.sleep(KEEPALIVE_S)
                await ws.send(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _reader(self, source: str, ws) -> None:
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")

                if kind == "SpeechStarted":
                    self._utt_start[source] = float(msg.get("timestamp", 0.0))

                elif kind == "Results":
                    alts = msg.get("channel", {}).get("alternatives", [])
                    text = (alts[0].get("transcript", "") if alts else "").strip()
                    if not text:
                        continue
                    start = float(msg.get("start", 0.0))
                    end = start + float(msg.get("duration", 0.0))
                    # speech_final means Deepgram's endpointer closed the utterance.
                    # is_final alone only means this chunk of words won't be revised.
                    final = bool(msg.get("speech_final", False))
                    await self._q.put(Transcript(
                        source=source, text=text, is_final=final,
                        t0=self._utt_start.get(source, start), t1=end,
                        provider=self.name, utt=self._utt[source],
                    ))
                    if final:
                        self._utt_start[source] = end
                        self._utt[source] += 1

                elif kind == "UtteranceEnd":
                    # Fallback boundary when endpointing never produced speech_final
                    # (happens with continuous background speech).
                    self._utt_start[source] = float(msg.get("last_word_end", 0.0))
                    self._utt[source] += 1

                elif kind == "Metadata":
                    log.debug("deepgram metadata %s: %s", source, msg.get("request_id"))

                elif kind == "Error":
                    log.error("deepgram error on %s: %s", source, msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closing:
                log.exception("deepgram reader died for %s", source)

    async def stop(self) -> None:
        self._closing = True
        for ws in self._ws.values():
            try:
                await ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
        await asyncio.sleep(0.2)  # let final Results arrive before we tear down
        for t in self._tasks:
            t.cancel()
        for ws in self._ws.values():
            try:
                await ws.close()
            except Exception:
                pass
        self._ws.clear()
        self._tasks.clear()
        await self._q.put(None)

    async def transcripts(self) -> AsyncIterator[Transcript]:
        while True:
            item = await self._q.get()
            if item is None:
                return
            yield item
