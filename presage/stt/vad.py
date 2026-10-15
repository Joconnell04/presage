"""Silero VAD on onnxruntime. No torch.

The pip package `silero-vad` drags in torch + torchaudio (~1 GB) just to run a 1.3 MB
model, so the ONNX weights from that package are vendored next to this file and run
directly. Same weights, same numbers, 30 MB of onnxruntime instead of a gigabyte.

Model: silero_vad_16k_op15.onnx (silero-vad 6.2.2). 16 kHz only, 512-sample frames.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import numpy as np

# Silero v5+ at 16 kHz wants exactly 512 samples (32 ms) per call.
FRAME_SAMPLES = 512
SAMPLE_RATE = 16000
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32
# The ONNX graph expects [last 64 samples of the previous frame | this 512-sample frame].
# Miss this and the model returns ~0.0 for obvious speech, silently.
CONTEXT_SAMPLES = 64
# The VAD-only gain (E69b) learns the talker's level from frames at or above this probability.
AGC_LEARN_P = 0.2

_DEFAULT_MODEL = Path(__file__).with_name("silero_vad_16k_op15.onnx")


class SileroVAD:
    """Stateful per-stream speech detector. One instance per audio source.

    Feed exactly FRAME_SAMPLES int16 samples at a time; get back P(speech) in [0, 1].
    The LSTM state is carried across calls, so instances are NOT thread-safe and must
    not be shared between "mic" and "system".
    """

    def __init__(self, model_path: str | os.PathLike | None = None, threads: int = 1):
        import onnxruntime as ort

        path = Path(model_path or os.environ.get("PRESAGE_VAD_MODEL") or _DEFAULT_MODEL)
        if not path.exists():
            raise FileNotFoundError(
                f"Silero VAD weights not found at {path}. They ship with this package; "
                f"if the file is missing, copy silero_vad_16k_op15.onnx from the "
                f"silero-vad pip package's silero_vad/data/ directory."
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = threads
        opts.intra_op_num_threads = threads
        # This model is tiny; graph optimisation costs more at load than it saves.
        opts.log_severity_level = 3  # silence the unused-initializer warnings
        self._sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Clear the LSTM state and the context window. Call between calls."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, frame_i16: np.ndarray) -> float:
        """P(speech) for one 512-sample int16 frame."""
        if frame_i16.shape[0] != FRAME_SAMPLES:
            raise ValueError(
                f"Silero wants exactly {FRAME_SAMPLES} samples, got {frame_i16.shape[0]}"
            )
        frame = (frame_i16.astype(np.float32) / 32768.0).reshape(1, FRAME_SAMPLES)
        x = np.concatenate([self._context, frame], axis=1)
        out, self._state = self._sess.run(
            ["output", "stateN"], {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(out[0, 0])


class UtteranceSegmenter:
    """Turns a stream of VAD probabilities into utterance open/close events.

    Owns its own frame-alignment buffer so callers can push arbitrary chunk sizes
    (20 ms from the capture path, whatever a wav file hands us) and this class slices
    them into the 512-sample frames Silero requires.

    push() returns a list of events, each ("open" | "speech" | "eager" | "close" |
    "drop", n_samples), where n_samples is the absolute sample index in this source's
    stream.

    Two-threshold endpointing: with `eager_silence_ms` set, a quiet run first crosses
    the eager mark and emits ("eager", n) once, then, if nobody finalises in the
    meantime, crosses `silence_ms` and emits ("close", n). The caller decides what to
    do in between -- here that means asking a turn detector whether the words look
    finished. Leave `eager_silence_ms` at None for the original one-threshold
    behaviour.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        silence_ms: int = 500,
        max_utterance_s: float = 15.0,
        speech_pad_ms: int = 200,
        min_utterance_ms: int = 300,
        eager_silence_ms: int | None = None,
        neg_threshold: float | None = None,
        vad: SileroVAD | None = None,
        split_at_dip: bool = False,
        split_lookback_ms: int = 1500,
        agc: bool = False,
        agc_target_dbfs: float = -26.0,
        agc_max_gain_db: float = 18.0,
    ):
        self.vad = vad or SileroVAD()
        self.threshold = threshold
        # Hysteresis, the way Silero's own examples and meetily do it: it takes a
        # decisively quiet frame to START counting silence, so a short dip inside a
        # word does not end the utterance. Open at 0.5, stay open down to 0.35.
        # Measured to matter most on quiet speech, where mid-word probabilities sag
        # toward the open threshold and a single-threshold VAD chops words in half.
        self.neg_threshold = (neg_threshold if neg_threshold is not None
                              else max(0.0, threshold - 0.15))
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.eager_frames = (max(1, eager_silence_ms // FRAME_MS)
                             if eager_silence_ms else None)
        self.max_samples = int(max_utterance_s * SAMPLE_RATE)
        self.pad_samples = int(speech_pad_ms * SAMPLE_RATE / 1000)
        self.min_samples = int(min_utterance_ms * SAMPLE_RATE / 1000)

        # E69b. Where a monologue that hits max_utterance_s is cut: at the quietest frame of
        # the last split_lookback_ms (a gap between words), not wherever the cap lands (inside
        # a word, which both halves then mangle). Off keeps the hard cut.
        self.split_at_dip = split_at_dip
        self._recent: deque[tuple[int, float]] = deque(
            maxlen=max(2, split_lookback_ms // FRAME_MS))
        # E69b. Gain on the VAD's copy of the audio only, never on what the recogniser hears.
        # Boosts a quiet mic toward agc_target_dbfs, measured on frames already judged speech,
        # never attenuates, and never by more than agc_max_gain_db. Off by default.
        self.agc = agc
        self._agc_target = 32768.0 * 10 ** (agc_target_dbfs / 20)
        self._agc_max = 10 ** (agc_max_gain_db / 20)
        self._agc_level = 0.0          # running RMS of speech frames, int16 units
        self.gain = 1.0
        # Called as probe(frame_start_sample, p) for every 32 ms frame when set. The E69b
        # recorder uses it; None costs one attribute check per frame.
        self.probe = None

        self._tail = np.zeros(0, dtype=np.int16)
        self._pos = 0              # absolute samples consumed
        self.triggered = False
        self.start_sample = 0
        self._quiet = 0
        self._eager_sent = False

    def rebase(self, sample: int) -> None:
        """Move the current utterance's start without closing it.

        Used when a finalised utterance is followed by speech that turns out to be a
        new turn rather than a continuation: the segmenter never closed, so the caller
        tells it where the new utterance actually began.
        """
        self.start_sample = max(0, sample)

    def push(self, pcm_i16: np.ndarray) -> list[tuple[str, int]]:
        events: list[tuple[str, int]] = []
        buf = np.concatenate([self._tail, pcm_i16]) if self._tail.size else pcm_i16

        n_full = buf.shape[0] // FRAME_SAMPLES
        for i in range(n_full):
            frame = buf[i * FRAME_SAMPLES: (i + 1) * FRAME_SAMPLES]
            p = self.vad(self._gained(frame) if self.agc else frame)
            self._pos += FRAME_SAMPLES
            frame_start = self._pos - FRAME_SAMPLES
            if self.agc:
                self._agc_update(frame, p)
            if self.probe is not None:
                self.probe(frame_start, p)
            if self.split_at_dip:
                self._recent.append((frame_start, p))

            if not self.triggered:
                if p >= self.threshold:
                    self.triggered = True
                    self._quiet = 0
                    self._eager_sent = False
                    # Rewind a little so we don't clip the first phoneme.
                    self.start_sample = max(0, frame_start - self.pad_samples)
                    events.append(("open", self.start_sample))
            else:
                if p < self.neg_threshold:
                    self._quiet += 1
                    end = min(self._pos, frame_start + self.pad_samples)
                    if (self.eager_frames is not None and not self._eager_sent
                            and self._quiet >= self.eager_frames
                            and self._quiet < self.silence_frames
                            and end - self.start_sample >= self.min_samples):
                        self._eager_sent = True
                        events.append(("eager", end))
                    if self._quiet >= self.silence_frames:
                        self.triggered = False
                        if end - self.start_sample >= self.min_samples:
                            events.append(("close", end))
                        else:
                            events.append(("drop", end))
                        self._quiet = 0
                        self._eager_sent = False
                else:
                    self._quiet = 0
                    self._eager_sent = False
                    events.append(("speech", self._pos))

                if self.triggered and self._pos - self.start_sample >= self.max_samples:
                    # Monologue guard: force a final so the prefetcher isn't starved.
                    cut = self._dip() if self.split_at_dip else None
                    if cut is not None:
                        # Close at the gap and carry straight on from it: the new
                        # utterance starts where the old one ended, no overlap, no hole.
                        events.append(("close", cut))
                        events.append(("open", cut))
                        self.start_sample = cut
                        self._eager_sent = False
                    else:
                        self.triggered = False
                        events.append(("close", self._pos))
                        self._quiet = 0

        self._tail = buf[n_full * FRAME_SAMPLES:].copy()
        return events

    def _dip(self) -> int | None:
        """Start of the quietest recent frame, if it leaves both halves a real length."""
        lo = self.start_sample + self.min_samples
        cands = [(p, fs) for fs, p in self._recent if fs > lo]
        if not cands:
            return None
        p, fs = min(cands, key=lambda c: (c[0], -c[1]))   # quietest, latest on a tie
        return fs + FRAME_SAMPLES // 2

    def _gained(self, frame: np.ndarray) -> np.ndarray:
        if self.gain <= 1.0001:
            return frame
        return np.clip(frame.astype(np.float32) * self.gain, -32768, 32767).astype(np.int16)

    def _agc_update(self, frame: np.ndarray, p: float) -> None:
        # Learn from anything that looks remotely like voice, not only frames already past the
        # open threshold: a talker who never reaches 0.5 is exactly the one who needs the gain.
        if p < AGC_LEARN_P:
            return
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        if rms <= 1.0:
            return
        # fast to follow a louder talker, slow to decide someone got quieter
        a = 0.3 if rms > self._agc_level else 0.05
        self._agc_level = rms if self._agc_level == 0 else (1 - a) * self._agc_level + a * rms
        self.gain = float(min(self._agc_max, max(1.0, self._agc_target / self._agc_level)))

    def flush(self) -> list[tuple[str, int]]:
        """Close whatever is open. Call on stop()."""
        if not self.triggered:
            return []
        self.triggered = False
        end = self._pos + self._tail.shape[0]
        return [("close", end)] if end - self.start_sample >= self.min_samples else []
