"""Streaming speaker labels: 20 ms int16 chunks in, (t, speaker) out, under a second.

The shape of the problem, from presage's side: the "system" source is one mixed
channel carrying every remote participant, and an in-person meeting on one microphone
carries Me as well. What is wanted is a label per stretch of speech, stable for the
length of a call, available soon enough that the card the brain builds can say who
asked.

How this does it, and why each piece is the cheap one:

  VAD          Silero, the same weights presage already runs for endpointing, in its
               own instance so nothing about the STT path changes. Speech runs are
               where the speaker can change; silence is not worth an embedding.
  EMBEDDING    a WeSpeaker / 3D-Speaker ONNX graph on onnxruntime (26-30 MB), fed
               Kaldi fbank computed in numpy (see fbank.py). No torch, no new package.
  CLUSTERING   online centroid matching on cosine similarity, with the decision split
               in two -- see below. Nothing is ever re-clustered, so cost is flat in
               call length and ids never renumber under the UI.
  ME           a centroid seeded from 10 s of the mic at call start. Any window that
               matches it is Me, wherever it turns up -- leaked into the system
               channel, or in a single-mic room mix.

THE SPLIT, WHICH IS THE WHOLE DESIGN. A 1.5 s window is enough to ask "which of the
people I already know does this sound most like". It is not enough to ask "is this
somebody new", because the cosine between two windows of one voice depends on the
voice: measured here with 3D-Speaker CAM++, two 1.5 s windows of a macOS `say` voice
sit at 0.72 and two windows of a real person at 0.41, against 0.34 and 0.13 for two
different speakers in the same two recordings. Any fixed window-level threshold is
wrong on one of them by more than the gap it is trying to measure, which is how a
diarizer scores 3% DER on synthetic clips and 45% on real voices.

At run level the ratio is stable even though the level is not: a whole speech run of
one real person matches another run of that person at 0.67 and a different person at
0.20; for `say` voices it is 0.77 against 0.38. So new speakers are born only when a
run closes, from the run's own embedding, and the bar is a fixed fraction of how alike
that speaker's own runs have been (`SELF_BETA`), which is a number that travels. Live,
a window only picks the nearest voice already known.

The price is one run: a person who has not spoken yet is labelled as whoever they
sound closest to until their first run ends, then every label in that run is corrected
(`revises=True`, the same contract as presage's STT retracting an eager final).
`bench.py` reports DER before and after those corrections, because a caller that
cannot revise pays the difference.

    d = OnlineDiarizer()
    d.enroll_me(mic_pcm)                  # optional, 10 s of int16
    for chunk in frames:                  # 20 ms, int16, 16 kHz, mono
        for lab in d.feed(chunk):
            print(lab.t0, lab.t1, lab.speaker)
    labels += d.flush()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .embed import Embedder, normalize

SAMPLE_RATE = 16000
VAD_FRAME = 512          # Silero at 16 kHz wants exactly this

RUN_EMBED_MAX_S = 10.0   # cap on the audio one run-level embedding reads
MIN_RUN_EMBED_S = 1.0    # below this a stretch is too short to be evidence of anyone
MIN_CHUNK_S = 1.2        # a shorter stretch is folded into its neighbour, never left
                         # standing on its own: the leading fragment of a run is the
                         # least like the rest of it, and left alone it invents people
SELF_PRIOR = 0.70        # assumed self-similarity of a voice we have only heard once
SELF_BETA = 0.72         # a run joins a speaker at this fraction of that speaker's own
                         # self-similarity. The band that works is 0.65-0.80: below it
                         # the hardest pair of voices merges, above it one voice splits.
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.65
EMA_BETA = 0.30          # how fast a speaker's self-similarity follows the call


@dataclass(frozen=True)
class SpeakerLabel:
    t0: float
    t1: float
    speaker: str          # "Me", "A", "B", ... stable for the life of the call
    score: float          # cosine to the centroid it was assigned to
    revises: bool = False
    """True when this supersedes an earlier label for the same interval."""

    emitted_at: float = 0.0
    """Audio-clock time at which the label became available. `emitted_at - t0` is the
    label delay; for a live stream the audio clock and the wall clock are the same."""


@dataclass
class _Centroid:
    name: str
    total: np.ndarray
    weight: float
    pinned: bool = False       # the enrolled Me: it informs, it does not drift far
    provisional: bool = False  # created live, not yet confirmed by a closed run
    self_sim: float | None = None
    """How alike two runs of THIS voice are, measured on this call. The gate for
    joining is a fraction of it, which is what makes the decision scale-free: a
    synthetic voice matches itself at 0.77 and a person on a laptop mic at 0.67, and
    a single number cannot be right for both."""

    last_run: int = -1
    """The run this centroid last absorbed. Two stretches of the SAME run overlap in
    audio and match each other at 0.97, which is not what one voice looks like across
    a conversation; learning self_sim from them drives the gate to the ceiling and the
    speaker's next real turn gets a new identity. Only a later run teaches anything."""

    @property
    def vector(self) -> np.ndarray:
        return normalize(self.total)

    def update(self, v: np.ndarray, w: float) -> None:
        if self.pinned:
            w = min(w, 0.25)      # let Me adapt to the room, not to whoever else spoke
        self.total = self.total + v * w
        self.weight += w

    def replace(self, v: np.ndarray, w: float) -> None:
        self.total = v * w
        self.weight = w
        self.provisional = False

    def note_join(self, score: float, run: int) -> None:
        if run == self.last_run:
            return                      # same run: the audio overlaps, so does the score
        self.last_run = run
        self.self_sim = score if self.self_sim is None else \
            (1 - EMA_BETA) * self.self_sim + EMA_BETA * score


@dataclass
class _Window:
    t0: float
    t1: float
    emb: np.ndarray
    live_speaker: str      # what was emitted at the time
    speaker: str           # what it should have been, once the run closed
    score: float
    emitted_at: float
    seconds: float


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


class OnlineDiarizer:
    """Speaker labels for one audio source. One instance per source; not thread-safe."""

    def __init__(self,
                 embedder: Embedder | str | None = None,
                 *,
                 window_s: float = _env_f("PRESAGE_DIAR_WINDOW_S", 3.0),
                 min_speech_s: float = _env_f("PRESAGE_DIAR_MIN_SPEECH_S", 0.40),
                 hop_s: float = _env_f("PRESAGE_DIAR_HOP_S", 0.40),
                 threshold: float | str = os.environ.get("PRESAGE_DIAR_THRESHOLD", "auto"),
                 sticky_margin: float = _env_f("PRESAGE_DIAR_STICKY", 0.04),
                 max_speakers: int = 8,
                 vad_threshold: float = _env_f("PRESAGE_DIAR_VAD", 0.5),
                 vad_neg_threshold: float = _env_f("PRESAGE_DIAR_VAD_NEG", 0.35),
                 close_ms: int = 280,
                 speech_pad_ms: int = 100,
                 revise: bool = True) -> None:
        from presage.stt.vad import SileroVAD        # read-only use of the STT weights

        self.embed = (embedder if isinstance(embedder, Embedder)
                      else Embedder(embedder or "campplus3"))
        self.window_n = int(window_s * SAMPLE_RATE)
        self.min_speech_n = int(min_speech_s * SAMPLE_RATE)
        self.hop_n = int(hop_s * SAMPLE_RATE)
        self.beta = SELF_BETA if threshold == "auto" else float(threshold)
        self.sticky_margin = sticky_margin
        self.max_speakers = max_speakers
        self.revise = revise

        self.vad = SileroVAD()
        self.vad_threshold = vad_threshold
        self.vad_neg_threshold = vad_neg_threshold
        self.close_frames = max(1, close_ms * SAMPLE_RATE // 1000 // VAD_FRAME)
        self.pad_n = speech_pad_ms * SAMPLE_RATE // 1000

        # Audio kept for windowing. 30 s is far more than window_s and costs 1 MB.
        self._ring = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
        self._written = 0
        self._pending = np.zeros(0, dtype=np.int16)

        self.centroids: list[_Centroid] = []
        self._next_name = 0
        self._in_run = False
        self._run_start = 0
        self._run_speech = 0
        self._silence_frames = 0
        self._last_emit = 0
        self._last_decision = 0
        self._windows: list[_Window] = []
        self._last_speaker: str | None = None
        self._run_index = 0

        self.stats = {"embed_calls": 0, "embed_s": 0.0, "vad_calls": 0, "vad_s": 0.0,
                      "runs": 0, "new_speakers": 0, "revised_windows": 0}

    # ------------------------------------------------------------------ enrollment

    def enroll_me(self, pcm: np.ndarray, seconds_weight: float = 10.0) -> np.ndarray:
        """Seed the Me centroid from the mic. 10 s of int16 is enough; 3 s works.

        Returns the embedding, so a caller can persist it between calls and skip the
        enrollment next time.
        """
        v = self.embed(_to_float(pcm))
        self.set_me_embedding(v, seconds_weight)
        return v

    def set_me_embedding(self, v: np.ndarray, seconds_weight: float = 10.0) -> None:
        """Restore a saved enrollment without re-recording."""
        v = normalize(np.asarray(v, dtype=np.float32))
        self.centroids = [c for c in self.centroids if c.name != "Me"]
        self.centroids.insert(0, _Centroid("Me", v * seconds_weight,
                                           seconds_weight, pinned=True))

    # ------------------------------------------------------------------ streaming

    def feed(self, pcm: np.ndarray) -> list[SpeakerLabel]:
        """Append samples (int16 or float32 in [-1, 1]); get whatever labels are ready."""
        x = np.asarray(pcm)
        if x.dtype != np.int16:
            x = np.clip(x * 32768.0, -32768, 32767).astype(np.int16)
        self._pending = np.concatenate([self._pending, x]) if self._pending.size else x

        out: list[SpeakerLabel] = []
        n_frames = self._pending.size // VAD_FRAME
        for i in range(n_frames):
            frame = self._pending[i * VAD_FRAME: (i + 1) * VAD_FRAME]
            self._store(frame)
            t0 = time.perf_counter()
            p = self.vad(frame)
            self.stats["vad_s"] += time.perf_counter() - t0
            self.stats["vad_calls"] += 1
            out.extend(self._on_frame(p))
        self._pending = self._pending[n_frames * VAD_FRAME:]
        return out

    def flush(self) -> list[SpeakerLabel]:
        """Close any open run at the end of the stream."""
        return self._close_run(self._written) if self._in_run else []

    # ------------------------------------------------------------------ ring buffer

    def _store(self, frame: np.ndarray) -> None:
        f = frame.astype(np.float32) / 32768.0
        size = self._ring.size
        start = self._written % size
        end = start + f.size
        if end <= size:
            self._ring[start:end] = f
        else:
            split = size - start
            self._ring[start:] = f[:split]
            self._ring[: end - size] = f[split:]
        self._written += f.size

    def _slice(self, a: int, b: int) -> np.ndarray:
        """Absolute sample range out of the ring."""
        size = self._ring.size
        a = max(a, self._written - size, 0)
        b = min(b, self._written)
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        i, j = a % size, b % size
        return self._ring[i:j].copy() if i < j else \
            np.concatenate([self._ring[i:], self._ring[:j]])

    # ------------------------------------------------------------------ run tracking

    def _on_frame(self, p: float) -> list[SpeakerLabel]:
        now = self._written
        if not self._in_run:
            if p >= self.vad_threshold:
                self._in_run = True
                self._run_start = max(0, now - VAD_FRAME - self.pad_n)
                self._run_speech = VAD_FRAME
                self._silence_frames = 0
                self._last_emit = self._run_start
                self._last_decision = self._run_start
                self._windows = []
                self._last_speaker = None
            return []

        if p >= self.vad_neg_threshold:
            self._run_speech += VAD_FRAME
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            if self._silence_frames >= self.close_frames:
                end = now - self._silence_frames * VAD_FRAME + self.pad_n
                return self._close_run(end)

        if (self._run_speech >= self.min_speech_n
                and now - self._last_decision >= self.hop_n):
            lab = self._decide(now)
            return [lab] if lab else []
        return []

    def _embed(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        emb = self.embed(audio)
        cost = time.perf_counter() - t0
        self.stats["embed_s"] += cost
        self.stats["embed_calls"] += 1
        return emb, cost

    def _decide(self, now: int) -> SpeakerLabel | None:
        """One live label: the nearest voice we already know."""
        a = max(self._run_start, now - self.window_n)
        audio = self._slice(a, now)
        if audio.size < int(0.25 * SAMPLE_RATE):
            return None
        emb, cost = self._embed(audio)
        name, score = self._nearest(emb, audio.size / SAMPLE_RATE)
        # emitted_at counts the inference too: live, the label lands that much after
        # the audio it was computed from, and the delay budget is end to end.
        lab = SpeakerLabel(t0=self._last_emit / SAMPLE_RATE,
                           t1=now / SAMPLE_RATE,
                           speaker=name, score=score,
                           emitted_at=now / SAMPLE_RATE + cost)
        self._windows.append(_Window(lab.t0, lab.t1, emb, name, name, score,
                                     lab.emitted_at, audio.size / SAMPLE_RATE))
        self._last_emit = now
        self._last_decision = now
        self._last_speaker = name
        return lab

    def _nearest(self, emb: np.ndarray, seconds: float) -> tuple[str, float]:
        """Nearest centroid. Creates a speaker only when there is no centroid at all."""
        if not self.centroids:
            return self._new_speaker(emb, seconds, provisional=True), 1.0
        sims = np.array([float(np.dot(emb, c.vector)) for c in self.centroids])
        best = int(np.argmax(sims))
        score = float(sims[best])

        # Stickiness: when two centroids are within sticky_margin, a speaker who was
        # already talking keeps the turn. Flapping mid-sentence is worse than a late
        # change, because a card that renames its asker looks broken.
        if self._last_speaker is not None:
            for i, c in enumerate(self.centroids):
                if c.name == self._last_speaker and score - float(sims[i]) <= self.sticky_margin:
                    best, score = i, float(sims[i])
                    break
        if not self.centroids[best].provisional and score >= self._gate(self.centroids[best]):
            self.centroids[best].update(emb, min(seconds, 2.0))
        return self.centroids[best].name, score

    def _new_speaker(self, emb: np.ndarray, seconds: float,
                     provisional: bool = False) -> str:
        name = chr(ord("A") + self._next_name) if self._next_name < 26 \
            else f"S{self._next_name}"
        self._next_name += 1
        w = min(seconds, 4.0)
        self.centroids.append(_Centroid(name, emb * w, w, provisional=provisional))
        self.stats["new_speakers"] += 1
        return name

    # ------------------------------------------------- run close: who that really was

    def _close_run(self, end: int) -> list[SpeakerLabel]:
        out: list[SpeakerLabel] = []
        if self._last_emit < end and self._run_speech >= int(0.25 * SAMPLE_RATE):
            lab = self._decide(end)
            if lab:
                out.append(lab)
        if self._windows:
            self._settle(end)
            if self.revise:
                out.extend(self._revisions(end))
        self.stats["runs"] += 1
        self._run_index += 1
        self._in_run = False
        self._run_speech = 0
        self._silence_frames = 0
        self._windows = []
        self._last_speaker = None
        return out

    def _settle(self, end: int) -> None:
        """Re-do the decision on the run, where the embedding is trustworthy.

        A run is not always one speaker -- somebody interrupts and the VAD never
        closes -- so it is first cut into stretches whose windows resemble each
        other, and each stretch gets its own join-or-create. The cut is relative:
        consecutive windows are compared with the run's own median, never with a
        threshold, because the absolute level does not travel between voices.
        """
        for chunk in self._chunks():
            a = int(chunk[0].t0 * SAMPLE_RATE)
            b = int(chunk[-1].t1 * SAMPLE_RATE)
            a = max(a, b - int(RUN_EMBED_MAX_S * SAMPLE_RATE))
            audio = self._slice(a, b)
            long_enough = audio.size >= int(MIN_RUN_EMBED_S * SAMPLE_RATE)
            if long_enough:
                emb, _ = self._embed(audio)
                seconds = audio.size / SAMPLE_RATE
            else:                      # too short to re-embed: the windows are all we have
                emb = normalize(np.stack([w.emb for w in chunk]).mean(axis=0))
                seconds = sum(w.t1 - w.t0 for w in chunk)
            # A stretch too short to embed is too short to be evidence of a new person.
            # Letting it open a speaker is where the phantom participants came from.
            name, score = self._join_or_create(emb, seconds, may_create=long_enough)
            for w in chunk:
                i = self._windows.index(w)
                self._windows[i] = _Window(w.t0, w.t1, w.emb, w.live_speaker, name,
                                           max(w.score, score), w.emitted_at, w.seconds)
            self._last_speaker = name

    def _chunks(self) -> list[list[_Window]]:
        """Split the run where consecutive windows stop sounding like each other."""
        ws = self._windows
        if len(ws) < 3:
            return [ws]
        embs = np.stack([w.emb for w in ws])
        adj = np.array([float(np.dot(embs[i], embs[i + 1])) for i in range(len(ws) - 1)])
        cut = float(np.median(adj)) - 0.25
        out, cur = [], [ws[0]]
        for i, sim in enumerate(adj):
            if sim < cut:
                out.append(cur)
                cur = []
            cur.append(ws[i + 1])
        out.append(cur)
        out = [c for c in out if c]
        return _merge_short(out)

    def _gate(self, c: _Centroid) -> float:
        """The cosine a run must reach to be this speaker: a fraction of how alike
        this speaker's own runs have been so far."""
        base = SELF_PRIOR if c.self_sim is None else c.self_sim
        return float(min(MAX_THRESHOLD, max(MIN_THRESHOLD, self.beta * base)))

    def _join_or_create(self, emb: np.ndarray, seconds: float,
                        may_create: bool = True) -> tuple[str, float]:
        known = [c for c in self.centroids if not c.provisional]
        # Each speaker sets their own bar. margin is how far over it the run is, so
        # the winner is the speaker it fits best, not the one with the loosest bar.
        margins = [float(np.dot(emb, c.vector)) - self._gate(c) for c in known]
        best = int(np.argmax(margins)) if margins else -1
        score = float(np.dot(emb, known[best].vector)) if best >= 0 else -1.0

        if best >= 0 and margins[best] >= 0:
            known[best].update(emb, min(seconds, 4.0))
            known[best].note_join(score, self._run_index)
            # A speaker invented live turned out to be somebody already known.
            self.centroids = [c for c in self.centroids if not c.provisional]
            return known[best].name, score

        prov = next((c for c in self.centroids if c.provisional), None)
        if prov is not None:
            prov.replace(emb, min(seconds, 4.0))       # promote, keeping the id
            return prov.name, score
        if may_create and len(self.centroids) < self.max_speakers:
            return self._new_speaker(emb, seconds), score
        return (known[best].name if best >= 0 else self.centroids[0].name), score


    def _revisions(self, end: int) -> list[SpeakerLabel]:
        out = []
        at = end / SAMPLE_RATE
        for w in self._windows:
            if w.speaker != w.live_speaker:
                self.stats["revised_windows"] += 1
                out.append(SpeakerLabel(w.t0, w.t1, w.speaker, w.score,
                                        revises=True, emitted_at=at))
        return out

    # ------------------------------------------------------------------ reporting

    def speakers(self) -> list[str]:
        return [c.name for c in self.centroids]

    def timings(self) -> dict:
        s = self.stats
        return {"embed_calls": s["embed_calls"],
                "embed_ms_mean": 1000 * s["embed_s"] / max(1, s["embed_calls"]),
                "vad_ms_mean": 1000 * s["vad_s"] / max(1, s["vad_calls"]),
                "runs": s["runs"], "new_speakers": s["new_speakers"],
                "revised_windows": s["revised_windows"],
                "gates": {c.name: round(self._gate(c), 3) for c in self.centroids},
                "self_sim": {c.name: None if c.self_sim is None else round(c.self_sim, 3)
                             for c in self.centroids},
                "compute_s": s["embed_s"] + s["vad_s"]}


def _merge_short(chunks: list[list[_Window]]) -> list[list[_Window]]:
    """Fold any stretch under MIN_CHUNK_S into whichever neighbour it sounds like."""
    while len(chunks) > 1:
        spans = [c[-1].t1 - c[0].t0 for c in chunks]
        i = int(np.argmin(spans))
        if spans[i] >= MIN_CHUNK_S:
            break
        if i == 0:
            j = 1
        elif i == len(chunks) - 1:
            j = i - 1
        else:
            here = normalize(np.stack([w.emb for w in chunks[i]]).mean(axis=0))
            left = normalize(np.stack([w.emb for w in chunks[i - 1]]).mean(axis=0))
            right = normalize(np.stack([w.emb for w in chunks[i + 1]]).mean(axis=0))
            j = i - 1 if float(np.dot(here, left)) >= float(np.dot(here, right)) else i + 1
        lo, hi = min(i, j), max(i, j)
        chunks[lo] = chunks[lo] + chunks[hi]
        del chunks[hi]
    return chunks


def _to_float(pcm: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm)
    return x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
