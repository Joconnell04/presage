"""Other recognisers in presage's local pipeline, through mlx-audio. Opt-in; Parakeet stays default.

    from presage.stt.mlxaudio import MlxAudioProvider
    p = MlxAudioProvider("granite-nar")     # or another key below, or any mlx-audio STT repo id

Only the decode changes. VAD, the 700 ms partial loop, the stable prefix, eager finals, retract and
the lexicon are LocalProvider's own, inherited unchanged, so the transcripts carry the same fields
and timing contract as the Parakeet ones.

E69a (docs/experiments/E69a-stt-models.md) measured these on 40 minutes of real call audio
(Earnings-22, AMI headset and room mic, Monsoon phone calls) against Parakeet TDT 0.6B v3's 10.2%:

    granite-nar  IBM Granite Speech 4.1 2B NAR, 5-bit MLX, Apache-2.0. WER 8.1% (paired 95%
                 interval -3.1 to -1.3 points), final text 11 ms later, 4.0 GB against 2.6.
    granite      IBM Granite Speech 5.0 TurboCTC, 470M, Apache-2.0. WER 8.4% (-2.8 to -1.0),
                 a final decoded in 25 ms against 145 (medians), 1.8 GB.
    qwen3        Qwen3-ASR 0.6B, 8-bit. No significant WER change, finals about 90 ms slower.

Why neither Granite is the default:
  * Both were trained on the AMI and Earnings-22 training splits, the two sets they won on. On the
    held-out phone calls (Monsoon) the gain is nil (-0.1 and -0.5 points, both inside the noise),
    and on the Open ASR Leaderboard's held-out conversational sets both sit behind Parakeet v3
    (14.34 and 13.97 against 13.44). Whether they help on Jackson's calls is what a replay decides.
  * They write lowercase text; TurboCTC also drops all punctuation, expands contractions and writes
    figures as bare digits, and the NAR model's punctuation comes and goes with the audio (it
    copies each corpus's transcription style). `written()` puts back what it cheaply can: a
    capital to open each sentence and on "I", Q3/H1/FY26, and a bare figure of a million or more
    as "742 million", which is what E9b's spoken-figure guard tuned its tolerance on. It never
    invents punctuation, because an invented full stop would fire eager finals. Without it the
    turn detector's punctuation route cannot fire (worth 269 ms in the endpointing sweep of
    docs/research-stt.md) and retract's verdict() sees no sentence breaks.

mlx-audio is not one of presage's dependencies: `uv pip install "mlx-audio[stt]>=0.5.5"` into the
venv (and add it to an extra in pyproject.toml, or run.sh's `uv sync` removes it again).
"""

from __future__ import annotations

import logging
import os
import re
import time

import numpy as np

from . import lexicon
from .eot import load_detector
from .local import SEMANTIC_EOT, Heard, LocalProvider
from .vad import SAMPLE_RATE

log = logging.getLogger("presage.stt.mlxaudio")

MODELS: dict[str, tuple[str, dict]] = {
    "granite-nar": ("mlx-community/granite-speech-4.1-2b-nar-mlx-5bit", {}),
    "granite": ("ibm-granite/granite-speech-5.0-470m-turboctc", {}),
    # max_tokens bounds the repetition loops Qwen3-ASR's issue tracker reports (#129)
    "qwen3": ("mlx-community/Qwen3-ASR-0.6B-8bit", {"language": "English", "max_tokens": 256}),
}
MODELS["granite-turboctc"] = MODELS["granite"]
MODELS["qwen3-asr"] = MODELS["qwen3"]
_LOWERCASE = ("granite-speech-5.0", "granite-speech-4.1-2b-nar")   # repos that write no capitals

# The same cap and the same reason as local.py's PRESAGE_MLX_CACHE_MB (E69a): MLX keeps every
# freed Metal buffer, re-decoding a growing utterance asks for a new size every 700 ms, and
# uncapped that walks the process into swap within a call.
MLX_CACHE_MB = int(os.environ.get("PRESAGE_MLX_CACHE_MB", "512"))

_SCALES = ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"))
_BARE_FIGURE = re.compile(r"(?<![\d.,])(\d{7,15})(?![\d.,])")
_PERIOD = re.compile(r"\b([qh]) ([1-4])\b")
_FISCAL = re.compile(r"\bfy ?(\d{2,4})\b")
_I = re.compile(r"\bi(?=\b|')")
_SENTENCE = re.compile(r"([.?!]\s+)([a-z])")


def _figure(m: re.Match) -> str:
    """742000000 -> "742 million", 1040000000 -> "1.04 billion"; a figure that does not reduce to
    three decimals of a scale is left as written, because rounding it would invent precision."""
    n = int(m.group(1))
    for scale, word in _SCALES:
        if n >= scale:
            q = f"{n / scale:.3f}".rstrip("0").rstrip(".")
            return f"{q} {word}" if round(float(q) * scale) == n else m.group(1)
    return m.group(1)


def written(text: str) -> str:
    """A normalised recogniser line in the written style presage's text layers expect.

    >>> written("so the q 4 number was 742000000 and i think fy 26 is flat. is that right?")
    'So the Q4 number was 742 million and I think FY26 is flat. Is that right?'
    """
    t = _BARE_FIGURE.sub(_figure, text)
    t = _PERIOD.sub(lambda m: m.group(1).upper() + m.group(2), t)
    t = _FISCAL.sub(lambda m: "FY" + m.group(1), t)
    t = _I.sub("I", t)
    t = _SENTENCE.sub(lambda m: m.group(1) + m.group(2).upper(), t)
    return t[:1].upper() + t[1:]


class MlxAudioProvider(LocalProvider):
    """LocalProvider with the decode done by an mlx-audio model on the same single worker thread."""

    def __init__(self, model: str = "granite-nar"):
        repo, gen = MODELS.get(model, (model, {}))
        super().__init__(model_id=repo)
        self.name = model.split("/")[-1]
        self._gen = gen
        # only the lowercase models need the rewrite; Qwen3-ASR already writes prose
        self._format = written if any(k in repo for k in _LOWERCASE) else (lambda s: s)

    def _load(self) -> None:
        import mlx.core as mx
        try:
            from mlx_audio.stt import load
        except ImportError as e:
            raise RuntimeError(
                f'{self.model_id} needs mlx-audio: uv pip install "mlx-audio[stt]>=0.5.5"') from e
        self._mx = mx
        if MLX_CACHE_MB >= 0:
            mx.set_cache_limit(MLX_CACHE_MB * 1024 * 1024)
        self._model = load(self.model_id)
        self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.int16))   # warm the Metal kernels
        if lexicon.ON:
            t = time.perf_counter()
            lx = self._lexicon()
            log.info("lexicon: %s terms in %.0f ms",
                     len(lx.terms) if lx else 0, (time.perf_counter() - t) * 1000)
        if SEMANTIC_EOT:
            t = time.perf_counter()
            self._eot = load_detector()
            self._eot.probability("warm the graph up please")
            log.info("turn detector ready in %.2fs", time.perf_counter() - t)

    def _transcribe(self, pcm_i16: np.ndarray) -> str:
        """Runs on the worker thread, like LocalProvider._transcribe, and repairs the line the same
        way (E46 lexicon), after putting it in written form."""
        audio = self._mx.array(pcm_i16.astype(np.float32) / 32768.0)
        out = self._model.generate(audio, **self._gen)
        text = self._format(str(getattr(out, "text", out)).strip())
        lx = self._lexicon()
        if lx is None or not text:
            return text
        t = time.perf_counter()
        try:
            c = lx.correct(text)
        except Exception:                        # noqa: BLE001  a bad index must not stop STT
            log.exception("lexicon: correct failed")
            return text
        ms = (time.perf_counter() - t) * 1000
        self.stats["lex_lines"] += 1
        self.stats["lex_ms"] += ms
        self.stats["lex_max_ms"] = max(self.stats["lex_max_ms"], ms)
        if not c.touched:
            return text
        self.stats["lex_changed"] += 1
        return Heard(c.text, text)
