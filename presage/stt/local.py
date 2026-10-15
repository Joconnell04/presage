"""Local STT: NVIDIA Parakeet TDT 0.6B v3 via MLX, segmented by Silero VAD.

Design, and why it is not an incremental streaming decoder:

parakeet-mlx re-transcribes a whole 7-second utterance in ~130 ms on this machine
(M5, measured -- see docs/research-stt.md). That is so far inside our latency budget
that there is no reason to run an incremental decoder. So while an utterance is open
we simply re-transcribe the accumulated audio every PARTIAL_MS and emit is_final=False;
when VAD says the utterance closed we transcribe once more and emit is_final=True.

The win over a streaming decoder is that every partial is a clean full-context decode,
so text never drifts, and over Whisper that Parakeet does not hallucinate completions
on a truncated prefix. Whisper small.en turns a 0.7 s prefix into "Can you remind me of
the world of the world?"; Parakeet says "Can you remind me?". Prefetching on a
hallucinated word is worse than prefetching late.

parakeet-mlx does ship a real streaming decoder (model.transcribe_stream() ->
StreamingParakeet.add_audio() / .result). It was benchmarked and is not faster in wall
clock at this chunk size, and it forces the encoder into local attention, which costs
accuracy. Not used.

MLX is not thread-safe for concurrent evaluation, so all inference for both sources runs
on a single dedicated worker thread. The event loop never blocks.

Three additions after the first production runs, each behind an env flag, each ON:

1. STABLE PREFIX (PRESAGE_STT_STABLE_PREFIX). Partials carry only what two consecutive
   re-transcriptions agreed on; the rest rides along in `unstable` for the UI to dim.
   LocalAgreement-2, Whisper-Streaming, arXiv 2307.14743. Prefetching on a word the
   next decode rewrites fetches an answer to a question nobody asked.

2. TWO-THRESHOLD ENDPOINTING (PRESAGE_STT_SEMANTIC_EOT). At EAGER_SILENCE_MS of quiet
   a turn detector reads the text so far. If the words look finished, finalise now
   instead of at 500 ms. If they do not, wait all the way to LONG_SILENCE_MS, because
   a pause mid-question is a pause, not an ending. Deepgram's Flux prices exactly this
   split as `eager_eot_threshold`.

3. RETRACT (PRESAGE_STT_RETRACT). An eager final is a bet. If the speaker resumes
   within RETRACT_MS and the audio reads as one continuing sentence, the bet lost: we
   re-emit the same `utt` with the joined text and `retracts=True`, and the utterance
   carries on to a real final. Non-eager finals never retract, so anything that waited
   the full silence is safe to treat as done.

4. LEXICON (PRESAGE_LEXICON). Parakeet has never seen the workbook, so it writes "air"
   for ARR and "orm bracket" for GMV Bracket. `presage/stt/lexicon.py` puts the index's
   own words back before anything downstream reads the line, and normalises the spoken
   form of a number. It runs inside `_transcribe`, which is the single place a text is
   produced, so partials, eager finals, retract joins and finals are all corrected the
   same way and `verdict()` never compares a corrected string with a raw one. The table
   is built on this same worker thread at load, not on the first utterance. What the
   recogniser actually wrote rides along in `Transcript.heard`. E46.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from . import lexicon
from .base import SOURCES, STTProvider, Transcript
from .eot import QuestionForming, StablePrefix, load_detector
from .vad import SAMPLE_RATE, SileroVAD, UtteranceSegmenter

log = logging.getLogger("presage.stt.local")


def _flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() not in (
        "0", "false", "no", "off", "")


DEFAULT_MODEL = os.environ.get(
    "PRESAGE_STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3"
)
PARTIAL_MS = int(os.environ.get("PRESAGE_STT_PARTIAL_MS", "700"))
SILENCE_MS = int(os.environ.get("PRESAGE_STT_SILENCE_MS", "500"))
MAX_UTT_S = float(os.environ.get("PRESAGE_STT_MAX_UTT_S", "15"))
MIN_UTT_MS = int(os.environ.get("PRESAGE_STT_MIN_UTT_MS", "300"))
VAD_THRESHOLD = float(os.environ.get("PRESAGE_VAD_THRESHOLD", "0.5"))
# Stay-open threshold. Open at 0.5, keep the utterance open until the probability
# drops below this. See vad.py for why the gap matters on quiet speech.
VAD_NEG_THRESHOLD = float(os.environ.get("PRESAGE_VAD_NEG_THRESHOLD", "0.35"))

STABLE_PREFIX = _flag("PRESAGE_STT_STABLE_PREFIX")
SEMANTIC_EOT = _flag("PRESAGE_STT_SEMANTIC_EOT")
RETRACT = _flag("PRESAGE_STT_RETRACT")

# Two-threshold endpointing. EAGER is what we pay when the detector is confident;
# LONG is what we pay when it is not, and it is deliberately longer than the old
# single 500 ms, because "unfinished" is exactly when interrupting is worst.
EAGER_SILENCE_MS = int(os.environ.get("PRESAGE_STT_EAGER_SILENCE_MS", "220"))
LONG_SILENCE_MS = int(os.environ.get("PRESAGE_STT_LONG_SILENCE_MS", "900"))
# The mic is the user (Me). Those finals only feed transcript context and the number guard --
# nothing prefetches on them -- so there is no reason to pay latency for them, and
# every reason to keep those sentences whole. Longer thresholds on both ends mean fewer
# splits mid-thought. Set PRESAGE_STT_MIC_SAME=1 to treat both sources alike.
MIC_SAME = _flag("PRESAGE_STT_MIC_SAME", False)
MIC_EAGER_SILENCE_MS = int(os.environ.get("PRESAGE_STT_MIC_EAGER_SILENCE_MS", "480"))
MIC_LONG_SILENCE_MS = int(os.environ.get("PRESAGE_STT_MIC_LONG_SILENCE_MS", "1400"))
RETRACT_MS = int(os.environ.get("PRESAGE_STT_RETRACT_MS", "600"))
# How much of the resumed speech to hear before judging one sentence or two. Judging
# on the first frame reads the eager final's own punctuation as a sentence break:
# "...headcount plan?" + "For" looks like two sentences, "...headcount plan for next
# year?" does not. Wait for words, not frames.
RETRACT_MIN_NEW_MS = int(os.environ.get("PRESAGE_STT_RETRACT_MIN_NEW_MS", "150"))
RETRACT_SETTLE_MS = int(os.environ.get("PRESAGE_STT_RETRACT_SETTLE_MS", "650"))
# How often the worker thread re-reads the lexicon. A context change rebuilds the term
# table (about 100 ms on the big test model), and this is the thread that would pay for it, between
# two decodes rather than inside one.
LEXICON_REFRESH_S = float(os.environ.get("PRESAGE_LEXICON_REFRESH_S", "5"))
# Ping-pong guard: after this many retracts an utterance rides out the long silence.
MAX_RETRACTS = int(os.environ.get("PRESAGE_STT_MAX_RETRACTS", "2"))

# E69b. Speech that resumes after an eager final used to be thrown away whenever no retract
# was going to look at it: resumed more than RETRACT_MS after the final, while a retract check
# was undecided at RETRACT_SETTLE_MS, after MAX_RETRACTS, or with retract off. The segmenter
# was still open, so no final ever came for it. On the mic that is any pause between about
# 1.2 s and 1.4 s. With this on, such speech opens a new utterance instead; speech that
# resumed while the eager final was still being decided gets its retract check from where it
# really started, not from the first frame after the final landed; and an utterance that has
# used its retracts waits out the long silence, as the comment above always said.
RESUME_NEW = _flag("PRESAGE_STT_RESUME_NEW", True)
# E69b input-chain knobs. Pre-roll before the first frame Silero calls speech, per source.
VAD_PAD_MS = int(os.environ.get("PRESAGE_VAD_PAD_MS", "200"))
MIC_VAD_PAD_MS = int(os.environ.get("PRESAGE_VAD_MIC_PAD_MS", str(VAD_PAD_MS)))
# Mic-only VAD thresholds; they default to the shared ones.
MIC_VAD_THRESHOLD = float(os.environ.get("PRESAGE_VAD_MIC_THRESHOLD", str(VAD_THRESHOLD)))
MIC_VAD_NEG_THRESHOLD = float(os.environ.get("PRESAGE_VAD_MIC_NEG_THRESHOLD",
                                             str(VAD_NEG_THRESHOLD)))
# 0: the mic never finalises early; every Me utterance waits out MIC_LONG_SILENCE_MS.
MIC_EAGER = _flag("PRESAGE_STT_MIC_EAGER", True)
# Cut a monologue that reaches MAX_UTT_S at the quietest frame of its last 1.5 s, and start the
# next utterance exactly there. The hard cut closed mid-word and reopened 200 ms back, so the
# word at the cut came out wrong in both halves: on 48 minutes of AMI meeting speech the chain
# lost 104 words with the hard cut and 54 with this (mean of three runs, E69b).
SPLIT_AT_DIP = _flag("PRESAGE_STT_SPLIT_AT_DIP", True)
# Gain on the VAD's copy of the audio (never the recogniser's), toward -26 dBFS, <= +18 dB.
VAD_AGC = _flag("PRESAGE_VAD_AGC", False)
# MLX keeps every freed Metal buffer in its own cache, and re-decoding a growing utterance asks
# for a new buffer size on every partial, so with no limit a long call walks this process from
# 1.2 GB of weights to 5 GB and more (E69a measured 4.9 GB after 4 utterances). On 16 GB that
# is swap, and a swapping transcriber is one that stops keeping up. Cap the cache; -1 leaves
# MLX's default (no practical limit). The limit is process wide, which is what we want.
MLX_CACHE_MB = int(os.environ.get("PRESAGE_MLX_CACHE_MB", "512"))
# Also hand the cache back to the OS after every final. Off unless measured to be needed.
MLX_CLEAR_ON_FINAL = _flag("PRESAGE_MLX_CLEAR_ON_FINAL", False)
# Decode a final (and an eager check) only up to this long after the last frame the VAD called
# speech, instead of through the whole closing silence (1.4 s on the mic). Parakeet v3 has been
# reported to return nothing, or to lose the last words, on windows with long trailing silence
# (NVIDIA NeMo issue 15757, FluidAudio issue 747). -1 keeps the full window.
TRIM_TAIL_MS = int(os.environ.get("PRESAGE_STT_TRIM_TAIL_MS", "-1"))
# A final whose decode comes back empty although the utterance's partials had words used to be
# dropped without a trace. With this on it is decoded once more trimmed to the speech, and if
# that is empty too, the last partial's text is the final.
EMPTY_FINAL_FALLBACK = _flag("PRESAGE_STT_EMPTY_FINAL_FALLBACK", False)

# Ring buffer per source. 60 s is far more than max_utterance_s and costs 2 MB.
RING_SECONDS = 60

_WORD_RE = re.compile(r"[\w'\-]+")
_SENTENCE_END = re.compile(r"[.?!…]")


class Heard(str):
    """The corrected text, carrying what the recogniser actually wrote.

    A `str` subclass rather than a pair, because the corrected text flows through the stable
    prefix, the turn detector and `verdict()` as an ordinary string, and only `_mk` needs the
    original. Anything that slices or joins it gets a plain `str` back, which is correct: half
    a corrected line has no single thing it was heard as.
    """

    __slots__ = ("heard",)

    def __new__(cls, text: str, heard: str) -> "Heard":
        o = super().__new__(cls, text)
        o.heard = heard
        return o


@dataclass(slots=True, repr=False)
class HeardTranscript(Transcript):
    """A Transcript whose text the lexicon changed. `heard` is what Parakeet wrote (E46)."""

    heard: str = ""


class _Ring:
    """Fixed-size int16 ring keyed by absolute sample index.

    Absolute indices are what the segmenter reports, so the ring has to translate them
    into its own storage. Anything older than RING_SECONDS is gone; utterances are
    capped well below that.
    """

    def __init__(self, seconds: int = RING_SECONDS):
        self.size = seconds * SAMPLE_RATE
        self.buf = np.zeros(self.size, dtype=np.int16)
        self.written = 0  # absolute count of samples ever written

    def append(self, pcm: np.ndarray) -> None:
        n = pcm.shape[0]
        if n >= self.size:  # pathological chunk, keep the tail
            self.buf[:] = pcm[-self.size:]
            self.written += n
            return
        start = self.written % self.size
        end = start + n
        if end <= self.size:
            self.buf[start:end] = pcm
        else:
            split = self.size - start
            self.buf[start:] = pcm[:split]
            self.buf[: end - self.size] = pcm[split:]
        self.written += n

    def slice(self, a: int, b: int) -> np.ndarray:
        """Absolute [a, b) as a contiguous copy. Clamped to what's still resident."""
        a = max(a, self.written - self.size, 0)
        b = min(b, self.written)
        if b <= a:
            return np.zeros(0, dtype=np.int16)
        s, e = a % self.size, b % self.size
        if s < e:
            return self.buf[s:e].copy()
        return np.concatenate([self.buf[s:], self.buf[:e]])


# Words that cannot start a sentence, so a "sentence break" in front of one is the
# recogniser's punctuation not having caught up yet.
_CONNECTIVES = frozenset((
    "for", "and", "or", "of", "to", "in", "on", "at", "by", "with", "from",
    "than", "versus", "vs", "plus", "minus", "over", "under", "into", "about",
    "across", "per", "that", "which", "whom", "whose", "because", "since",
    "while", "unless", "though", "although",
))
# Deliberately NOT here: "so" and "but". Both are discourse markers that routinely
# open a spoken sentence ("So, what did revenue look like?"), so treating them as
# continuations would glue a greeting onto the question that follows it.


def _trimmed(start: int, end: int, speech_end: int, tail_ms: int | None = None) -> int:
    """End of the audio to decode: `end`, or TRIM_TAIL_MS past the last speech frame."""
    ms = TRIM_TAIL_MS if tail_ms is None else tail_ms
    if ms < 0 or speech_end <= start:
        return end
    return max(start + SAMPLE_RATE // 10, min(end, speech_end + ms * SAMPLE_RATE // 1000))


def verdict(old: str, joined: str) -> str:
    """One sentence, two sentences, or not enough heard yet? -> wait|retract|new

    `old` is the text of an eager final; `joined` is the re-transcription of the same
    audio plus whatever the speaker said next.

      "What did we assume?"  ->  "What did we assume for ARR growth?"
          same words up to here, no sentence break between them: the eager final cut a
          question in half.                                              -> retract

      "What's the churn?"    ->  "What's the churn? Is that blended?"
          the recogniser put a sentence break exactly where we finalised. -> new

      "What did we assume?"  ->  "What did we expect for ARR growth?"
          it revised words we had committed, so the final was wrong.      -> retract

      "...headcount plan?"   ->  "...headcount plan?"  (nothing new decoded yet)
                                                                          -> wait

    Two subtleties, both learned the hard way on `retract.wav`:

    * Parakeet keeps the question mark it already committed until it has heard enough
      to re-render the whole sentence. At 2.95 s the joined text is still
      "...headcount plan?"; at 3.05 s it is "...headcount plan? For next"; only at
      3.40 s does it become "...headcount plan for next year?". Deciding on the first
      resumed frame always reads the stale mark and always says "new".
    * So a break in front of a word that cannot begin a sentence ("For next") is not a
      break, it is punctuation lagging. That resolves the case at 3.05 s without
      waiting for the re-render.

    Punctuation is only ever read off `joined`, never off `old`: Parakeet punctuates
    every partial, so `old` always ends in something confident-looking.
    """
    old_words = _WORD_RE.findall(old.lower())
    if not old_words:
        return "new"
    spans = list(_WORD_RE.finditer(joined))
    new_words = [m.group(0).lower() for m in spans]
    n = len(old_words)
    if new_words[:n] != old_words:
        return "retract"                      # revised what we committed
    if len(new_words) <= n:
        return "wait"                         # nothing new decoded yet
    gap = joined[spans[n - 1].end(): spans[n].start()]
    if not _SENTENCE_END.search(gap):
        return "retract"                      # one sentence after all
    return "new" if new_words[n] not in _CONNECTIVES else "retract"


class _SourceState:
    def __init__(self, source: str):
        self.source = source
        self.ring = _Ring()
        # The mic override only applies with semantic endpointing on. With
        # PRESAGE_STT_SEMANTIC_EOT=0 both sources fall back to the single SILENCE_MS
        # threshold, which is the pre-Round-2 behaviour exactly, deliberately.
        mic = source == "mic" and not MIC_SAME
        eager_ms = MIC_EAGER_SILENCE_MS if mic else EAGER_SILENCE_MS
        long_ms = MIC_LONG_SILENCE_MS if mic else LONG_SILENCE_MS
        is_mic = source == "mic"
        self.seg = UtteranceSegmenter(
            threshold=MIC_VAD_THRESHOLD if is_mic else VAD_THRESHOLD,
            neg_threshold=MIC_VAD_NEG_THRESHOLD if is_mic else VAD_NEG_THRESHOLD,
            silence_ms=long_ms if SEMANTIC_EOT else SILENCE_MS,
            eager_silence_ms=(eager_ms if SEMANTIC_EOT and (MIC_EAGER or not is_mic)
                              else None),
            max_utterance_s=MAX_UTT_S,
            min_utterance_ms=MIN_UTT_MS,
            speech_pad_ms=MIC_VAD_PAD_MS if is_mic else VAD_PAD_MS,
            split_at_dip=SPLIT_AT_DIP,
            agc=VAD_AGC,
            vad=SileroVAD(),
        )
        self.stable = StablePrefix()
        self.qform = QuestionForming()
        self.open = False
        self.start_sample = 0
        self.last_partial_at = 0.0   # wall clock of the last partial we emitted
        self.inflight = False        # a partial for this source is already running
        self.last_text = ""
        self.utt = 0                 # monotonic per-source utterance counter

        # eager-final bookkeeping
        self.finalized = False       # an eager final is standing for this utt
        self.final_text = ""
        self.final_end = 0           # last sample the eager final covered
        self.final_at = 0.0          # perf_counter when it was emitted
        self.eager_busy = False      # an eager decision is in flight
        self.retract_busy = False
        self.n_retracts = 0
        self.last_speech = 0         # sample of the newest frame the VAD called speech
        self.eager_end = 0           # where the eager decision in flight would end the utt
        self.resume_at = None        # first speech after eager_end, while that decision runs

    def reset_utterance(self, start: int) -> None:
        self.start_sample = start
        self.stable.reset()
        self.qform.reset()
        self.last_text = ""
        self.finalized = False
        self.final_text = ""
        self.final_end = 0
        self.final_at = 0.0
        self.n_retracts = 0


class LocalProvider(STTProvider):
    name = "local"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self._model = None
        self._get_logmel = None
        self._mx = None
        self._eot = None
        self._sources = {s: _SourceState(s) for s in SOURCES}
        self._q: asyncio.Queue[Transcript | None] = asyncio.Queue()
        # One worker: MLX evaluation is not safe to run concurrently.
        self._pool = ExecutorHolder()
        # asyncio only holds weak references to tasks, so an untracked task can be
        # garbage collected mid-flight and the transcript silently never arrives.
        self._tasks: set[asyncio.Task] = set()
        self._t_zero = 0.0
        self._started = False
        # counters, read by smoke.py and worth logging in production
        self.stats = {"finals": 0, "eager_finals": 0, "retracts": 0,
                      "eot_calls": 0, "eot_ms": 0.0,
                      "lex_lines": 0, "lex_changed": 0, "lex_ms": 0.0, "lex_max_ms": 0.0,
                      "resumed_new": 0, "tail_rescued": 0}
        # E69b: trace(kind, fields) receives segment boundaries, eager and retract
        # decisions and every decode's queue wait and duration. PRESAGE_RECORD_AUDIO sets it
        # (presage/audio_session.py); None, the default, costs one check per event.
        self.trace = None
        # the term table, read on the worker thread only (E46)
        self._lex = None
        self._lex_at = -1e9

    def _trace(self, event: str, st: "_SourceState | None" = None, **kw) -> None:
        if self.trace is None:
            return
        try:
            kw["t"] = round(time.perf_counter() - self._t_zero, 4)
            if st is not None:
                kw["source"] = st.source
                kw.setdefault("utt", st.utt)
            self.trace(event, kw)
        except Exception:  # noqa: BLE001  a recorder must never break transcription
            log.debug("trace failed", exc_info=True)

    def set_vad_probe(self, fn) -> None:
        """fn(source, frame_start_sample, p) for every 32 ms VAD frame, or None to stop."""
        for s, st in self._sources.items():
            st.seg.probe = (None if fn is None
                            else (lambda fs, p, _s=s: fn(_s, fs, p)))

    async def _decode(self, st: "_SourceState", kind: str, audio: np.ndarray):
        """One transcription on the worker thread, timed when a recorder is listening."""
        loop = asyncio.get_running_loop()
        clear = MLX_CLEAR_ON_FINAL and kind == "final"
        if self.trace is None and not clear:
            return await loop.run_in_executor(self._pool.ex, self._transcribe, audio)
        box: dict[str, float] = {}
        t_sub = time.perf_counter()

        def work():
            box["t0"] = time.perf_counter()
            try:
                return self._transcribe(audio)
            finally:
                box["t1"] = time.perf_counter()
                if clear and self._mx is not None:
                    self._mx.clear_cache()      # on the worker thread, between decodes

        text = await loop.run_in_executor(self._pool.ex, work)
        if self.trace is None:
            return text
        self._trace("decode", st, what=kind, n=int(audio.shape[0]),
                    wait_ms=round((box.get("t0", t_sub) - t_sub) * 1000, 1),
                    ms=round((box.get("t1", t_sub) - box.get("t0", t_sub)) * 1000, 1),
                    text=str(text or ""))
        return text

    def _resume_as_new(self, st: "_SourceState", at: int, why: str) -> None:
        """Speech after an eager final that no retract will absorb: its own utterance."""
        start = max(st.final_end, at - st.seg.pad_samples)
        st.utt += 1
        st.seg.rebase(start)
        st.reset_utterance(st.seg.start_sample)
        st.open = True
        st.last_speech = at
        st.last_partial_at = time.perf_counter()
        self.stats["resumed_new"] += 1
        self._trace("resume_new", st, why=why, start=start, at=at)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._pool.open()
        t = time.perf_counter()
        await loop.run_in_executor(self._pool.ex, self._load)
        self._t_zero = time.perf_counter()
        self._started = True
        log.info("parakeet %s loaded in %.2fs [stable_prefix=%s semantic_eot=%s "
                 "retract=%s]", self.model_id, self._t_zero - t,
                 STABLE_PREFIX, SEMANTIC_EOT, RETRACT)

    def _load(self) -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        self._mx = mx
        self._get_logmel = get_logmel
        if MLX_CACHE_MB >= 0:
            mx.set_cache_limit(MLX_CACHE_MB * 1024 * 1024)
        self._model = from_pretrained(self.model_id)
        # Warm the Metal kernels so the first real utterance isn't the slow one.
        warm = np.zeros(SAMPLE_RATE, dtype=np.int16)
        self._transcribe(warm)

        if lexicon.ON:
            t = time.perf_counter()
            lx = self._lexicon()
            log.info("lexicon: %s terms in %.0f ms",
                     len(lx.terms) if lx else 0, (time.perf_counter() - t) * 1000)

        if SEMANTIC_EOT:
            t = time.perf_counter()
            self._eot = load_detector()
            self._eot.probability("warm the graph up please")  # first call is slow
            log.info("turn detector ready in %.2fs", time.perf_counter() - t)

    async def stop(self) -> None:
        for st in self._sources.values():
            for kind, end in st.seg.flush():
                if kind == "close" and st.open and not st.finalized:
                    await self._emit_final(st, st.start_sample, end, st.utt,
                                           speech_end=st.last_speech, fallback=st.last_text)
                elif (kind == "close" and st.open and RESUME_NEW
                      and st.last_speech > st.final_end
                      and end - st.final_end >= SAMPLE_RATE // 10):
                    await self._emit_final(st, st.final_end, end, st.utt + 1)
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        await self._q.put(None)
        self._pool.close()
        self._started = False

    # ------------------------------------------------------------------ audio in

    async def feed(self, source: str, pcm: np.ndarray) -> None:
        if not self._started:
            raise RuntimeError("LocalProvider.feed() before start()")
        st = self._sources[source]
        if pcm.dtype != np.int16:
            raise TypeError(f"feed() wants int16, got {pcm.dtype}")
        st.ring.append(pcm)

        for kind, at in st.seg.push(pcm):
            if kind == "open":
                st.open = True
                st.last_partial_at = time.perf_counter()
                st.reset_utterance(at)
                st.last_speech = at
                self._trace("open", st, at=at)

            elif kind == "speech":
                st.last_speech = at
                if (st.eager_busy and not st.finalized and st.resume_at is None
                        and at > st.eager_end):
                    st.resume_at = at        # resumed while the eager final is being decided
                # Speech after an eager final: either the speaker never stopped, or
                # this is a new turn. _maybe_retract decides, off the capture path.
                if (RETRACT and st.finalized and not st.retract_busy
                        and st.n_retracts < MAX_RETRACTS
                        and (time.perf_counter() - st.final_at) * 1000 <= RETRACT_MS):
                    st.retract_busy = True
                    self._spawn(self._maybe_retract(st, at))
                elif RESUME_NEW and st.finalized and not st.retract_busy:
                    # Nobody is going to look at this speech as a continuation, and the
                    # segmenter is still open, so without this it never gets a final.
                    self._resume_as_new(st, at, "late")

            elif kind == "eager":
                if (not st.finalized and not st.eager_busy and st.open
                        and not (RESUME_NEW and st.n_retracts >= MAX_RETRACTS)):
                    st.eager_busy = True
                    st.eager_end = at
                    st.resume_at = None
                    self._spawn(self._maybe_eager_final(st, st.start_sample, at,
                                                        st.utt, st.last_speech))

            elif kind == "close" and st.open:
                st.open = False
                self._trace("close", st, at=at, start=st.start_sample,
                            finalized=st.finalized)
                if st.finalized:
                    # The eager final already went out and nothing superseded it.
                    if (RESUME_NEW and st.last_speech > st.final_end
                            and at - st.final_end >= SAMPLE_RATE // 10):
                        # Speech after the eager final that nothing picked up (a retract
                        # check still deciding when the silence ran out): give it a final.
                        st.utt += 1
                        self.stats["tail_rescued"] += 1
                        self._trace("tail_rescue", st, start=st.final_end, at=at)
                        self._spawn(self._emit_final(st, st.final_end, at, st.utt,
                                                     speech_end=st.last_speech))
                    st.utt += 1
                else:
                    # Spawned, not awaited: feed() is on the capture path and must not
                    # stall for the ~100 ms the decode takes. The single-threaded
                    # executor keeps submission order.
                    self._spawn(self._emit_final(st, st.start_sample, at, st.utt,
                                                 speech_end=st.last_speech,
                                                 fallback=st.last_text))
                    st.utt += 1

            elif kind == "drop":
                st.open = False  # too short to be speech; no inference, no transcript
                self._trace("drop", st, at=at, start=st.start_sample)

        if st.open and not st.finalized:
            now = time.perf_counter()
            if not st.inflight and (now - st.last_partial_at) * 1000 >= PARTIAL_MS:
                st.last_partial_at = now
                st.inflight = True
                self._spawn(self._run_partial(st))

    # ------------------------------------------------------------------ inference

    async def _run_partial(self, st: _SourceState) -> None:
        try:
            start, end, utt = st.start_sample, st.ring.written, st.utt
            audio = st.ring.slice(start, end)
            if audio.shape[0] < SAMPLE_RATE // 10:
                return
            text = await self._decode(st, "partial", audio)
            if not st.open or st.start_sample != start or st.utt != utt:
                return  # utterance moved on while we decoded; the final supersedes
            if st.finalized:
                return  # an eager final landed; a partial after it would confuse
            if not text or text == st.last_text:
                return
            st.last_text = text
            stable, unstable = self._split(st, text)
            if not stable and not unstable:
                return
            # Score the COMMITTED text, not the full hypothesis: question_forming is
            # a promise the brain acts on, and the unstable tail can still vanish.
            p, forming = await self._score(st, stable or unstable)
            await self._q.put(self._mk(st, stable, start, end, utt,
                                       final=False, unstable=unstable,
                                       complete_p=p, question_forming=forming))
        except Exception:
            log.exception("partial transcribe failed for %s", st.source)
        finally:
            st.inflight = False

    async def _maybe_eager_final(self, st: _SourceState, start: int, end: int,
                                 utt: int, speech_end: int = 0) -> None:
        """EAGER_SILENCE_MS of quiet: do the words look like a finished turn?"""
        try:
            audio = st.ring.slice(start, _trimmed(start, end, speech_end))
            if audio.shape[0] < SAMPLE_RATE // 10:
                return
            loop = asyncio.get_running_loop()
            text = await self._decode(st, "eager", audio)
            if not text or st.utt != utt or st.finalized:
                return
            t = time.perf_counter()
            done, p, why = await loop.run_in_executor(
                self._pool.ex, self._eot.judge, text)
            self.stats["eot_calls"] += 1
            self.stats["eot_ms"] += (time.perf_counter() - t) * 1000
            log.debug("eot %s p=%.5f %s %r", st.source, p, why, text[-48:])
            self._trace("eager", st, done=bool(done), p=round(float(p), 5), why=why,
                        end=end, text=str(text))
            if not done:
                return  # unfinished: ride out the long silence
            self.stats["eager_" + why] = self.stats.get("eager_" + why, 0) + 1
            if st.utt != utt or st.finalized:
                return
            st.finalized = True
            st.final_text = text
            st.final_end = end
            st.final_at = time.perf_counter()
            st.last_text = text
            self.stats["finals"] += 1
            self.stats["eager_finals"] += 1
            await self._q.put(self._mk(st, text, start, end, utt,
                                       final=True, eager=True, complete_p=p))
            ra = st.resume_at
            if (RESUME_NEW and ra is not None and st.utt == utt and st.finalized
                    and not st.retract_busy):
                # The speaker started again while this final was being decided. Every
                # "speech" frame since arrived before the final existed, so nothing looked
                # at it; judge it now, from where it really began.
                self.stats["resumed_during_decision"] = (
                    self.stats.get("resumed_during_decision", 0) + 1)
                if RETRACT and st.n_retracts < MAX_RETRACTS:
                    st.retract_busy = True
                    self._spawn(self._maybe_retract(st, ra))
                else:
                    self._resume_as_new(st, ra, "during_decision")
        except Exception:
            log.exception("eager endpoint check failed for %s", st.source)
        finally:
            st.eager_busy = False

    async def _maybe_retract(self, st: _SourceState, at: int) -> None:
        """Speech resumed inside RETRACT_MS. One sentence, or two?"""
        try:
            start, utt, old = st.start_sample, st.utt, st.final_text
            loop = asyncio.get_running_loop()
            # Poll until the recogniser has decoded enough of the resumed speech to
            # say. verdict() returns "wait" while the joined text is still identical
            # to the eager final. Give up at the deadline and leave the final standing.
            want = at + RETRACT_MIN_NEW_MS * SAMPLE_RATE // 1000
            deadline = time.perf_counter() + RETRACT_SETTLE_MS / 1000
            end, joined, call = 0, "", "wait"
            while True:
                if st.utt != utt or not st.finalized:
                    return
                if st.ring.written >= want:
                    end = st.ring.written
                    audio = st.ring.slice(start, end)
                    joined = await self._decode(st, "retract", audio)
                    call = verdict(old, joined) if joined else "wait"
                    if call != "wait":
                        break
                if time.perf_counter() >= deadline:
                    # Undecidable in the window; the eager final stands. Before E69b the
                    # resumed speech then went nowhere, because the segmenter never closed.
                    self._trace("retract", st, call="undecided", old=old, joined=joined)
                    if (RESUME_NEW and st.utt == utt and st.finalized
                            and st.seg.triggered):
                        self._resume_as_new(st, at, "undecided")
                    return
                await asyncio.sleep(0.06)
            if st.utt != utt or not st.finalized:
                return
            self._trace("retract", st, call=call, old=old, joined=joined)
            if call == "new":
                # A new turn that happened to start fast. Re-base the open utterance
                # onto the resumed speech so its transcripts get a fresh utt.
                st.utt += 1
                pad = int(0.2 * SAMPLE_RATE)  # don't clip the new turn's first word
                st.seg.rebase(max(start, at - pad))
                st.reset_utterance(st.seg.start_sample)
                st.open = True
                st.last_partial_at = time.perf_counter()
                return
            st.n_retracts += 1
            st.finalized = False
            st.final_text = ""
            st.last_text = joined
            self.stats["retracts"] += 1
            stable, unstable = self._split(st, joined)
            p, forming = await self._score(st, stable or joined)
            await self._q.put(self._mk(st, stable or joined, start, end, utt,
                                       final=False, retracts=True,
                                       unstable=unstable if stable else "",
                                       complete_p=p, question_forming=forming))
        except Exception:
            log.exception("retract check failed for %s", st.source)
        finally:
            st.retract_busy = False

    async def _emit_final(self, st: _SourceState, start: int, end: int,
                          utt: int, speech_end: int = 0, fallback: str = "") -> None:
        cut = _trimmed(start, end, speech_end)
        audio = st.ring.slice(start, cut)
        if audio.shape[0] < SAMPLE_RATE // 10:
            return
        try:
            text = await self._decode(st, "final", audio)
        except Exception:
            log.exception("final transcribe failed for %s", st.source)
            return
        if not text and fallback:
            # Parakeet heard words in this utterance a moment ago and now returns nothing.
            self.stats["empty_final"] = self.stats.get("empty_final", 0) + 1
            self._trace("empty_final", st, utt_final=utt, fallback=str(fallback))
            if EMPTY_FINAL_FALLBACK:
                tight = _trimmed(start, end, speech_end, 300)
                if tight < cut:
                    try:
                        text = await self._decode(st, "final_retry",
                                                  st.ring.slice(start, tight))
                    except Exception:
                        log.exception("final retry failed for %s", st.source)
                if not text:
                    text = fallback
                self.stats["empty_final_rescued"] = (
                    self.stats.get("empty_final_rescued", 0) + 1)
        if text:
            self.stats["finals"] += 1
            p = 0.0
            if self._eot is not None:
                p, _ = await self._score(st, text)
            await self._q.put(self._mk(st, text, start, end, utt, final=True,
                                       complete_p=p))

    def _split(self, st: _SourceState, text: str) -> tuple[str, str]:
        """(committed, not-yet-agreed). Whole text as committed when the flag is off.

        The committed prefix carries `heard` only when it is the whole hypothesis: half a
        corrected line was never heard as any one thing, so there is nothing honest to show.
        """
        if not STABLE_PREFIX:
            return text, ""
        stable, unstable = st.stable.update(text)
        raw = getattr(text, "heard", "")
        if raw and stable == str(text):
            stable = Heard(stable, raw)
        return stable, unstable

    async def _score(self, st: _SourceState, text: str) -> tuple[float, bool]:
        """(P(complete), question_forming) for a partial. ~2-4 ms, worth it."""
        if self._eot is None or not text:
            return 0.0, False
        loop = asyncio.get_running_loop()
        t = time.perf_counter()
        done, p, _ = await loop.run_in_executor(self._pool.ex, self._eot.judge, text)
        self.stats["eot_calls"] += 1
        self.stats["eot_ms"] += (time.perf_counter() - t) * 1000
        return p, st.qform.update(text, p, done)

    def _lexicon(self):
        """The term table, re-read at most every LEXICON_REFRESH_S. Worker thread only."""
        now = time.perf_counter()
        if now - self._lex_at >= LEXICON_REFRESH_S:
            self._lex_at = now
            try:
                self._lex = lexicon.current()
            except Exception:                    # noqa: BLE001  a bad index must not stop STT
                log.exception("lexicon unavailable; transcripts pass through uncorrected")
                self._lex = None
        return self._lex

    def _transcribe(self, pcm_i16: np.ndarray) -> str:
        """Runs on the worker thread. Never call from the event loop.

        The lexicon runs here and nowhere else, so every text that leaves this provider has had
        the same repair: partials, eager finals, the retract join and finals. `verdict()`
        compares two corrected strings, and `StablePrefix` agrees on corrected words.
        """
        audio = self._mx.array(pcm_i16.astype(np.float32) / 32768.0)
        mel = self._get_logmel(audio, self._model.preprocessor_config)
        text = self._model.generate(mel)[0].text.strip()
        lx = self._lexicon()
        if lx is None or not text:
            return text
        t = time.perf_counter()
        try:
            c = lx.correct(text)
        except Exception:                        # noqa: BLE001
            log.exception("lexicon: correct failed")
            return text
        ms = (time.perf_counter() - t) * 1000
        self.stats["lex_lines"] += 1
        self.stats["lex_ms"] += ms
        self.stats["lex_max_ms"] = max(self.stats["lex_max_ms"], ms)
        if not c.touched:
            return text
        self.stats["lex_changed"] += 1
        log.debug("lexicon %s -> %s", c.changed, c.text[-48:])
        return Heard(c.text, text)

    def _mk(self, st: _SourceState, text: str, start: int, end: int, utt: int,
            *, final: bool, unstable: str = "", eager: bool = False,
            retracts: bool = False, complete_p: float = 0.0,
            question_forming: bool = False) -> Transcript:
        heard = getattr(text, "heard", "")
        cls = HeardTranscript if heard else Transcript
        kw = {"heard": heard} if heard else {}
        return cls(
            source=st.source,
            text=str(text),
            is_final=final,
            t0=start / SAMPLE_RATE,
            t1=end / SAMPLE_RATE,
            provider=self.name,
            utt=utt,
            unstable=unstable,
            eager=eager,
            retracts=retracts,
            question_forming=question_forming and not final,
            complete_p=round(complete_p, 5),
            **kw,
        )

    # ------------------------------------------------------------------ output

    async def transcripts(self) -> AsyncIterator[Transcript]:
        while True:
            item = await self._q.get()
            if item is None:
                return
            yield item


class ExecutorHolder:
    """A single-thread executor that can be opened and closed more than once."""

    def __init__(self):
        self.ex: ThreadPoolExecutor | None = None

    def open(self) -> None:
        if self.ex is None:
            self.ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    def close(self) -> None:
        if self.ex is not None:
            # cancel_futures matters: without it, decodes queued at teardown keep
            # running on the old thread, hold the model alive, and fight the next
            # provider for the GPU. In the benchmark that showed up as a clip with
            # zero partials and a final 8 s late, with the feed loop perfectly on
            # time -- every partial had been starved by the previous clip's backlog.
            self.ex.shutdown(wait=False, cancel_futures=True)
            self.ex = None
