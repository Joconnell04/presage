"""One capture session: helper -> STT -> transcripts, started and stopped at runtime.

The server used to own this as three module-level tasks that could only be started
once, at boot. Here it is a small object the page can toggle:

    session = AudioSession(on_transcript=..., on_change=...)
    await session.start(mic_uid=None)     # idempotent
    session.state()                       # {"state": "running", "mic_state": "ok", ...}
    await session.stop()                  # helper and STT worker gone inside 2 s

States are "stopped", "starting", "running" and "error" (with `error` set). The
helper's own JSON events drive `mic_state` ("ok" | "silent" | "recovering" |
"fallback"), which is what the page's Mic dot reads.

PRESAGE_REPLAY=file.wav feeds a wav in as the system source instead of the helper
(PRESAGE_REPLAY_MIC=file.wav does the same for the mic source, so a scripted two-voice call replays),
through the same class, so the replay smoke test still works.

PRESAGE_RECORD_AUDIO=<dir> (E69b, off unless set) records a session for diagnosis: both streams as
they reach Python (16 kHz mono int16 wavs, exactly what the recogniser is fed, filler included),
every 20 ms frame's arrival time and level, every VAD probability, every segment boundary, eager
and retract decision and decode timing from the provider, every partial and final, and the helper's
device events. One folder per session, `<dir>/<YYYYmmdd-HHMMSS>/`. About 240 MB an hour. Nothing
is written anywhere unless the variable is set. `data/stt/mic/recording.py <folder>` reads it back.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import signal
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable

import json
import queue
import threading

import numpy as np

log = logging.getLogger("presage.audio_session")

STOP_PUMP_S = 0.8        # wait for the frame loop to notice the stop flag
STOP_STT_S = 1.0         # wait for the STT worker to flush and close
FILLER_GAP_S = 0.25      # system tap goes quiet between sounds; fill it so VAD closes
ME_SPEAKING_MAX_S = 25   # watchdog: a mic utterance that never closes must not hold cards
RECORD_DIR = os.environ.get("PRESAGE_RECORD_AUDIO") or None
# E69b. While the tap is idle the filler feeds 100 ms of zeros per 300 ms or so, so the system
# stream's clock runs at a third of real time: a Them utterance whose last word was also the
# last thing the call app played waits about 2.7 s, not 0.9 s, for its long-silence final, and
# system timestamps drift behind the mic's. With this on it feeds the silence that really
# elapsed (at most FILLER_MAX_S per tick). Off until a recording shows a call app going idle.
FILLER_REALTIME = os.environ.get("PRESAGE_FILLER_REALTIME", "0").lower() not in (
    "0", "false", "off", "no", "")
FILLER_MAX_S = 0.5


class _WavAppender:
    """16 kHz mono int16 wav that stays readable if the process dies mid-session: the RIFF sizes
    are rewritten every couple of seconds and on close, so at worst the last two seconds are
    outside the header, and a reader that trusts the file size gets them back."""

    def __init__(self, path: Path) -> None:
        self.f = open(path, "wb+")
        self.n = 0
        self._fixed_at = time.monotonic()
        self._header()

    def _header(self) -> None:
        data = self.n
        self.f.seek(0)
        self.f.write(b"RIFF" + (36 + data).to_bytes(4, "little") + b"WAVEfmt "
                     + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
                     + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
                     + (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
                     + (16).to_bytes(2, "little") + b"data" + data.to_bytes(4, "little"))
        self.f.seek(0, 2)

    def write(self, pcm: bytes) -> None:
        self.f.write(pcm)
        self.n += len(pcm)
        if time.monotonic() - self._fixed_at > 2.0:
            self._header()
            self.f.flush()
            self._fixed_at = time.monotonic()

    def close(self) -> None:
        self._header()
        self.f.close()


class _Recorder:
    """PRESAGE_RECORD_AUDIO: everything needed to replay a session and see which words went where.

    Every call is a non-blocking put onto one queue; a daemon thread does the writing. If the disk
    cannot keep up the queue drops data and counts it, rather than ever slowing the audio path.

        mic.wav, system.wav   the streams exactly as fed to the recogniser
        frames.csv            wall,t,source,sample,n,peak,dbfs,filler   one row per 20 ms frame
        vad.csv               source,sample,p                           one row per 32 ms frame
        stt.jsonl             provider trace (open, eager, retract, resume_new, close, decode
                              with queue wait and duration) and every partial and final
        events.jsonl          session start and stop, settings, the helper's device events
    """

    def __init__(self, root: str, meta: dict) -> None:
        self.dir = Path(root).expanduser() / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        self.samples = {"mic": 0, "system": 0}
        self.dropped = 0
        self._q: "queue.Queue[tuple | None]" = queue.Queue(maxsize=20000)
        self._wav = {s: _WavAppender(self.dir / f"{s}.wav") for s in ("mic", "system")}
        self._frames = open(self.dir / "frames.csv", "w", buffering=1 << 16)
        self._frames.write("wall,t,source,sample,n,peak,dbfs,filler\n")
        self._vad = open(self.dir / "vad.csv", "w", buffering=1 << 16)
        self._vad.write("source,sample,p\n")
        self._stt = open(self.dir / "stt.jsonl", "w", buffering=1 << 16)
        self._events = open(self.dir / "events.jsonl", "w", buffering=1 << 14)
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        self.event({"event": "record_start", "dir": str(self.dir), **meta})

    def _put(self, item: tuple) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def frame(self, source: str, pcm: np.ndarray, filler: bool = False) -> None:
        if source not in self.samples or not len(pcm):
            return
        at = self.samples[source]
        self.samples[source] += len(pcm)
        self._put(("pcm", source, at, time.time(), time.monotonic() - self.t0,
                   np.asarray(pcm, dtype=np.int16).tobytes(), filler))

    def vad(self, source: str, sample: int, p: float) -> None:
        self._put(("vad", source, sample, p))

    def trace(self, kind: str, fields: dict) -> None:
        self._put(("stt", {"kind": kind, "wall": round(time.time(), 3), **fields}))

    def transcript(self, t) -> None:
        d = {"kind": "final" if getattr(t, "is_final", False) else "partial",
             "wall": round(time.time(), 3)}
        for k in ("source", "utt", "text", "unstable", "t0", "t1", "eager", "retracts",
                  "question_forming", "complete_p", "heard"):
            v = getattr(t, k, None)
            if v is not None and v != "":
                d[k] = v
        self._put(("stt", d))

    def event(self, ev: dict) -> None:
        self._put(("ev", {"wall": round(time.time(), 3), **ev}))

    def _run(self) -> None:
        last_flush = time.monotonic()
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                item = ()
            if item is None:
                break
            try:
                if item:
                    kind = item[0]
                    if kind == "pcm":
                        _, src, at, wall, t, raw, filler = item
                        self._wav[src].write(raw)
                        a = np.frombuffer(raw, dtype=np.int16)
                        peak = int(np.abs(a.astype(np.int32)).max()) if a.size else 0
                        rms = float(np.sqrt(np.mean(a.astype(np.float32) ** 2))) if a.size else 0.0
                        db = round(20 * np.log10(rms / 32768), 1) if rms > 0 else -120.0
                        self._frames.write(f"{wall:.3f},{t:.4f},{src},{at},{a.size},{peak},"
                                           f"{db},{int(filler)}\n")
                    elif kind == "vad":
                        self._vad.write(f"{item[1]},{item[2]},{item[3]:.3f}\n")
                    elif kind == "stt":
                        self._stt.write(json.dumps(item[1], default=str) + "\n")
                    elif kind == "ev":
                        self._events.write(json.dumps(item[1], default=str) + "\n")
                if time.monotonic() - last_flush > 1.0:
                    for f in (self._frames, self._vad, self._stt, self._events):
                        f.flush()
                    last_flush = time.monotonic()
            except Exception:  # noqa: BLE001  a recorder must never take the session down
                log.debug("recorder write failed", exc_info=True)
        for w in self._wav.values():
            with contextlib.suppress(Exception):
                w.close()
        for f in (self._frames, self._vad, self._stt, self._events):
            with contextlib.suppress(Exception):
                f.close()

    def close(self) -> None:
        self.event({"event": "record_stop", "samples": dict(self.samples),
                    "dropped": self.dropped})
        with contextlib.suppress(Exception):
            self._q.put(None, timeout=2.0)
        self._thread.join(timeout=5.0)


def _record_meta() -> dict:
    """The knobs in force, minus anything that looks like a secret."""
    env = {k: v for k, v in os.environ.items()
           if (k.startswith("PRESAGE_") or k.startswith("STT_"))
           and not any(x in k for x in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    return {"env": env, "pid": os.getpid()}


class AudioSession:
    """Owns the STT provider, the audio pump and the helper's mic state."""

    def __init__(self,
                 on_transcript: Callable[[Any], Awaitable[None]],
                 on_change: Callable[[], None] | None = None,
                 levels: dict[str, float] | None = None) -> None:
        self.on_transcript = on_transcript
        self.on_change = on_change
        self.levels = levels if levels is not None else {"mic": 0.0, "system": 0.0}

        self.status = "stopped"
        self.error: str | None = None
        self.mic_state = "unknown"          # ok | silent | recovering | fallback | unknown
        self.mic_device: str | None = None
        self.mic_uid: str | None = None
        self.output_device: str | None = None
        self.mic = False
        self.system = False
        self.me_speaking = False        # a mic utterance is open: the page holds cards
        self._me_since = 0.0
        self.stt = None
        self.stt_name = (os.environ.get("STT_PROVIDER") or "local").lower()
        self.replay = os.environ.get("PRESAGE_REPLAY") or None
        self.replay_mic = os.environ.get("PRESAGE_REPLAY_MIC") or None   # a second wav fed as Me
        # speaker labels on the far side (experiment E3): a 30 MB ONNX embedder + online clustering,
        # 3.6% of one core, first label about half a second after a speaker change
        self.diarize = os.environ.get("PRESAGE_DIARIZE", "1").lower() not in ("0", "false", "off")
        self._diar = None
        self._diar_q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=4096)
        self._diar_thread: threading.Thread | None = None
        self._labels: list = []                  # SpeakerLabel, audio clock = seconds since start
        self._labels_lock = threading.Lock()
        self._me_pcm: list[np.ndarray] = []      # first 10 s of the mic, for enrollment
        self._me_samples = 0
        self._me_enrolled = False
        self.started_at: float | None = None

        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._tasks: dict[str, asyncio.Task] = {}
        self._last_frame: dict[str, float] = {"mic": 0.0, "system": 0.0}
        self._rec: _Recorder | None = None       # PRESAGE_RECORD_AUDIO, E69b

    # ------------------------------------------------------------------ public

    def state(self) -> dict:
        return {"state": self.status, "started_at": self.started_at,
                "mic_state": self.mic_state,
                "mic_device": self.mic_device,
                "mic_uid": self.mic_uid,
                "output_device": self.output_device,
                "error": self.error,
                "mic": self.mic,
                "system": self.system,
                "me_speaking": self.me_speaking,
                "stt": self.stt_name,
                "listening": self.status == "running",
                "replay": bool(self.replay),
                "recording": str(self._rec.dir) if self._rec is not None else None}

    async def start(self, mic_uid: str | None = None) -> dict:
        """Start capturing. A second call while running is a no-op unless the mic changed.

        mic_uid: a device uid, "" for the system default on purpose, None for whatever
        settings.json has (which is what auto start at boot passes).
        """
        async with self._lock:
            if self.status == "starting":
                return self.state()
            if self.status == "running":
                if mic_uid is None or (mic_uid or None) == self.mic_uid:
                    return self.state()
                await self._stop_locked()
            await self._start_locked(mic_uid)
            return self.state()

    async def stop(self) -> dict:
        async with self._lock:
            await self._stop_locked()
            return self.state()

    # ------------------------------------------------------------------ lifecycle

    async def _start_locked(self, mic_uid: str | None) -> None:
        from presage import settings

        if mic_uid is None:
            saved = settings.load().get("mic_uid")
            mic_uid = str(saved) if saved else None
        self.mic_uid = mic_uid or None
        self.status = "starting"
        self.error = None
        self.mic_state = "unknown"
        self.mic_device = None          # the helper names the device it actually opened
        self.mic = self.system = False
        self.me_speaking = False
        self._stop = asyncio.Event()
        self._changed()

        try:
            from presage.stt import make_provider
            stt = make_provider(os.environ.get("STT_PROVIDER"))
            await stt.start()
        except Exception as e:  # noqa: BLE001  a missing model must not kill the server
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            log.error("stt start failed: %s", self.error)
            self._changed()
            return

        self.stt = stt
        self.stt_name = getattr(stt, "name", type(stt).__name__)
        self.started_at = time.time()
        self._start_recorder(stt)
        self._start_diarizer()
        self._tasks = {"consume": asyncio.create_task(self._consume(stt))}
        if self.replay or self.replay_mic:
            if self.replay:
                self.system = True
                self._tasks["pump"] = asyncio.create_task(self._replay_pump(Path(self.replay).expanduser()))
            if self.replay_mic:
                self._tasks["pump_mic"] = asyncio.create_task(
                    self._replay_pump(Path(self.replay_mic).expanduser(), source="mic"))
        else:
            self._tasks["pump"] = asyncio.create_task(self._audio_pump())
            self._tasks["filler"] = asyncio.create_task(self._filler())
        self.status = "running"
        self._changed()
        log.info("audio started (stt=%s mic_uid=%s)", self.stt_name, self.mic_uid or "default")

    async def _stop_locked(self) -> None:
        self._stop_diarizer()
        if self.status == "stopped" and not self._tasks:
            return
        t0 = time.time()
        self._stop.set()

        pump = self._tasks.pop("pump", None)
        if pump is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(pump, STOP_PUMP_S)   # on timeout wait_for cancels it
            if not pump.done():
                pump.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(pump), 0.4)

        filler = self._tasks.pop("filler", None)
        if filler is not None:
            filler.cancel()

        stt, self.stt = self.stt, None
        if stt is not None:
            try:
                await asyncio.wait_for(stt.stop(), STOP_STT_S)
            except Exception as e:  # noqa: BLE001
                log.warning("stt stop: %s", e)

        consume = self._tasks.pop("consume", None)
        if consume is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(consume, 0.3)
            if not consume.done():
                consume.cancel()
        self._stop_recorder(stt)

        self._tasks.clear()
        await self._reap_helper()
        self.status = "stopped"
        self.error = None
        self.mic = self.system = False
        self.me_speaking = False
        self.mic_state = "unknown"
        self.levels["mic"] = self.levels["system"] = 0.0
        self.started_at = None
        self._changed()
        log.info("audio stopped in %.0f ms", (time.time() - t0) * 1000)

    async def _reap_helper(self) -> None:
        """Belt and braces: SIGTERM any audiotap still parented to this process."""
        def scan() -> list[int]:
            p = subprocess.run(["pgrep", "-P", str(os.getpid()), "audiotap"],
                               capture_output=True, text=True, timeout=1)
            return [int(x) for x in p.stdout.split() if x.strip().isdigit()]

        try:
            pids = await asyncio.wait_for(asyncio.to_thread(scan), 0.3)
        except Exception:  # noqa: BLE001  pgrep is a convenience, not a requirement
            return
        for pid in pids:
            log.warning("audiotap %s still running after stop, terminating", pid)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)

    # ------------------------------------------------------------------ pumps

    async def _audio_pump(self) -> None:
        """Helper frames -> STT, with level meters. Closes the helper on the way out."""
        from presage import audio as audio_mod

        kwargs: dict[str, Any] = {"on_event": self._on_event}
        if self.mic_uid and _takes(audio_mod.frames, "mic_uid"):
            kwargs["mic_uid"] = self.mic_uid
        elif self.mic_uid:
            log.warning("presage.audio.frames() has no mic_uid argument yet; using the default input")

        gen = audio_mod.frames(**kwargs)
        try:
            async for source, pcm in gen:
                if self._stop.is_set():
                    break
                self._last_frame[source] = time.time()
                if len(pcm):
                    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) / 32768.0)
                    self.levels[source] = max(rms, self.levels.get(source, 0.0) * 0.8)
                if self._rec is not None:
                    self._rec.frame(source, pcm)
                if self.stt is not None:
                    await self.stt.feed(source, pcm)
                self._diar_feed(source, pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if not self._stop.is_set():
                self.status, self.error = "error", f"{type(e).__name__}: {e}"
                log.error("audio pump: %s", self.error)
                self._changed()
        finally:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await gen.aclose()          # runs the helper's terminate/kill path
            self.mic = self.system = False
            if not self._stop.is_set() and self.status == "running":
                self.status = "error"
                self.error = self.error or "audio helper exited"
                log.error("audio helper exited")
                self._changed()

    async def _filler(self) -> None:
        """The system tap only fires while something is playing; silence closes the utterance."""
        zeros = np.zeros(1600, dtype=np.int16)   # 100 ms
        while not self._stop.is_set():
            await asyncio.sleep(0.1)
            now = time.time()
            if self.me_speaking and self._me_since and now - self._me_since > ME_SPEAKING_MAX_S:
                self.me_speaking = False      # a final that never came must not hold cards forever
                self._me_since = 0.0
                self._changed()
            last = self._last_frame["system"]
            if last and now - last > FILLER_GAP_S:
                if self.stt is None:
                    return
                pad = zeros
                if FILLER_REALTIME:
                    n = int(min(now - last, FILLER_MAX_S) * 16000) // 320 * 320
                    pad = np.zeros(max(320, n), dtype=np.int16)
                if self._rec is not None:
                    self._rec.frame("system", pad, filler=True)
                with contextlib.suppress(Exception):
                    await self.stt.feed("system", pad)
                # The speaker labeller counts its own samples to time its labels, and
                # speaker_for() compares those times with the recogniser's. Before E69b it never
                # saw the filler, so after every idle stretch its clock fell behind the
                # transcript's and labels landed on the wrong lines.
                self._diar_feed("system", pad)
                self._last_frame["system"] = now

    async def _replay_pump(self, path: Path, source: str = "system") -> None:
        """Feed a 16 kHz mono int16 wav as if it were live audio. For smoke tests."""
        with wave.open(str(path), "rb") as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, \
                "need 16 kHz mono int16 wav"
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if source == "system":
            self.system = True
        self.mic_state = "ok"
        self.mic_device = f"replay: {path.name}"
        self._changed()
        chunk = 320   # 20 ms
        t_start = time.time()
        for i in range(0, len(data), chunk):
            if self._stop.is_set():
                return
            pcm = data[i:i + chunk]
            self.levels[source] = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) / 32768.0)
            if self.stt is None:
                return
            if self._rec is not None:
                self._rec.frame(source, pcm)
            await self.stt.feed(source, pcm)
            self._diar_feed(source, pcm)
            delay = t_start + (i + chunk) / 16000 - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
        if self.stt is not None:
            tail = np.zeros(16000, dtype=np.int16)                          # closes the last utterance
            if self._rec is not None:
                self._rec.frame(source, tail)
            await self.stt.feed(source, tail)
        log.info("replay finished: %s", path)

    # ------------------------------------------------------------------ recording (E69b)

    def _start_recorder(self, stt) -> None:
        """PRESAGE_RECORD_AUDIO only. Never raises: a recorder that cannot start is a warning."""
        if not RECORD_DIR or self._rec is not None:
            return
        try:
            meta = _record_meta()
            meta.update(stt=self.stt_name, mic_uid=self.mic_uid, replay=self.replay,
                        replay_mic=self.replay_mic)
            self._rec = _Recorder(RECORD_DIR, meta)
        except Exception as e:  # noqa: BLE001
            log.warning("PRESAGE_RECORD_AUDIO set but the recorder did not start: %s", e)
            self._rec = None
            return
        if hasattr(stt, "trace"):
            stt.trace = self._rec.trace
        if hasattr(stt, "set_vad_probe"):
            stt.set_vad_probe(self._rec.vad)
        log.warning("PRESAGE_RECORD_AUDIO: recording both streams and the transcript to %s",
                    self._rec.dir)

    def _stop_recorder(self, stt) -> None:
        rec, self._rec = self._rec, None
        if rec is None:
            return
        if stt is not None:
            if hasattr(stt, "trace"):
                stt.trace = None
            if hasattr(stt, "set_vad_probe"):
                with contextlib.suppress(Exception):
                    stt.set_vad_probe(None)
            rec.event({"event": "stt_stats", "stats": dict(getattr(stt, "stats", {}) or {})})
        rec.close()
        log.info("recording closed: %s (%d samples mic, %d system, %d items dropped)",
                 rec.dir, rec.samples["mic"], rec.samples["system"], rec.dropped)

    # ------------------------------------------------------------------ speaker labels (E3)

    def _start_diarizer(self) -> None:
        if not self.diarize or self._diar_thread is not None:
            return
        with self._labels_lock:
            self._labels = []
        self._me_pcm, self._me_samples, self._me_enrolled = [], 0, False
        self._diar_q = queue.Queue(maxsize=4096)

        def run() -> None:
            try:
                from presage.diarize import OnlineDiarizer
                self._diar = OnlineDiarizer()
            except Exception as e:  # noqa: BLE001  labels are optional
                log.warning("speaker labels off: %s", e)
                self._diar = None
                return
            log.info("speaker labeller ready")
            while True:
                item = self._diar_q.get()
                if item is None:
                    break
                try:
                    kind, pcm = item
                    if kind == "enroll":
                        self._diar.enroll_me(pcm)
                        self._me_enrolled = True
                        log.info("speaker labeller: Me enrolled from %.1f s of mic audio", len(pcm) / 16000)
                        continue
                    labels = self._diar.feed(pcm)
                    if labels:
                        with self._labels_lock:
                            self._labels.extend(labels)
                            del self._labels[:-4000]
                except Exception as e:  # noqa: BLE001
                    log.debug("speaker labeller: %s", e)

        self._diar_thread = threading.Thread(target=run, name="diarize", daemon=True)
        self._diar_thread.start()

    def _diar_feed(self, source: str, pcm: np.ndarray) -> None:
        """Far-side frames go to the labeller; the first 10 s of the mic enroll Me."""
        if not self.diarize or self._diar_thread is None or not len(pcm):
            return
        if source == "mic":
            if not self._me_enrolled and self._me_samples < 16000 * 10:
                rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) / 32768.0)
                if rms > 0.01:                       # speech-ish frames only
                    self._me_pcm.append(np.asarray(pcm, dtype=np.int16))
                    self._me_samples += len(pcm)
                    if self._me_samples >= 16000 * 10:
                        with contextlib.suppress(queue.Full):
                            self._diar_q.put_nowait(("enroll", np.concatenate(self._me_pcm)))
                        self._me_pcm = []
            return
        if source != "system":
            return
        try:
            self._diar_q.put_nowait(("feed", np.asarray(pcm, dtype=np.int16)))
        except queue.Full:
            pass                                     # labels are best effort; never back up audio

    def _stop_diarizer(self) -> None:
        th, self._diar_thread = self._diar_thread, None
        if th is not None:
            with contextlib.suppress(queue.Full):
                self._diar_q.put_nowait(None)
        self._diar = None

    def speaker_for(self, source: str, t0: float, t1: float) -> str | None:
        """The far-side speaker who spoke most of [t0, t1] (audio clock), or None."""
        if source != "system" or not self.diarize:
            return None
        with self._labels_lock:
            labels = list(self._labels[-400:])
        if not labels:
            return None
        weight: dict[str, float] = {}
        for lb in labels:
            ov = min(t1, lb.t1) - max(t0, lb.t0)
            if ov > 0:
                weight[lb.speaker] = weight.get(lb.speaker, 0.0) + ov
        if not weight:
            return None
        best = max(weight, key=weight.get)
        return best if weight[best] >= 0.3 * max(0.2, t1 - t0) else None

    def speakers(self) -> list[str]:
        d = self._diar
        try:
            return list(d.speakers()) if d is not None else []
        except Exception:  # noqa: BLE001
            return []

    def _track_me(self, t) -> None:
        """Me is mid-sentence from a mic interim until its final. The page queues cards while
        this is true, so both edges are pushed at once instead of waiting for the 1 Hz tick."""
        if getattr(t, "source", None) != "mic":
            return
        speaking = not bool(getattr(t, "is_final", False))
        if speaking == self.me_speaking:
            return
        self.me_speaking = speaking
        self._me_since = time.time() if speaking else 0.0
        self._changed()

    async def _consume(self, stt) -> None:
        try:
            async for t in stt.transcripts():
                if self._rec is not None:
                    self._rec.transcript(t)
                self._track_me(t)
                await self.on_transcript(t)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("transcript consumer stopped")

    # ------------------------------------------------------------------ helper events

    def _on_event(self, ev: dict) -> None:
        if self._rec is not None:
            self._rec.event(ev)
        kind = ev.get("event")
        if kind == "ready":
            self.mic, self.system = bool(ev.get("mic")), bool(ev.get("system"))
            self.mic_device = _first(ev, "mic_device", "device", "input", "input_device") or self.mic_device
            self.mic_uid = _first(ev, "mic_uid", "uid") or self.mic_uid
            self.output_device = _first(ev, "output_device", "output") or self.output_device
            if self.mic_state == "unknown":
                self.mic_state = "ok" if self.mic else "silent"
            self._changed()
        elif kind == "mic_state":
            st = str(ev.get("state") or "").strip()
            if st:
                self.mic_state = st
            self.mic_device = _first(ev, "device", "name", "mic_device") or self.mic_device
            self.mic_uid = _first(ev, "uid", "mic_uid") or self.mic_uid
            log.info("mic %s: %s", self.mic_state, self.mic_device or "")
            self._changed()
        elif kind == "silent":
            if ev.get("reason") == "zero_samples":
                log.warning("%s stream is all zeros: missing permission? %s", ev.get("stream"), ev.get("hint"))
            if ev.get("stream") == "mic" and self.mic_state in ("unknown", "ok"):
                self.mic_state = "silent"
                self._changed()
        elif kind == "retuned":
            log.info("audiotap: %s", ev)
            if self.mic_state == "unknown":
                self.mic_state = "ok"
                self._changed()
        elif kind == "error":
            self.error = str(_first(ev, "message", "error") or ev)
            log.error("audiotap: %s", ev)
            self._changed()
        elif kind in ("backlog", "overflow"):
            # E69b: this process stopped reading the helper's pipe for a while. A backlog was
            # queued in the helper and arrives late; an overflow is audio that is gone.
            log.warning("audiotap %s: %s", kind, ev)
        else:
            log.info("audiotap: %s", ev)

    def _changed(self) -> None:
        if self.on_change is None:
            return
        try:
            self.on_change()
        except Exception:  # noqa: BLE001  a UI callback must never break capture
            log.exception("audio on_change failed")


def _first(d: dict, *keys: str):
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return None


def _takes(fn, name: str) -> bool:
    """True when fn accepts a keyword argument. Lets this run against either audio.py."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
