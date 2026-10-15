"""Semantic end-of-turn detection: is this text a finished question, or mid-sentence?

Silence alone cannot tell "What did we assume for..." (thinking) from "What did we
assume for ARR growth?" (done). A text model can. That lets `local.py` finalize after
~220 ms of silence when the words look complete, instead of always waiting 500 ms.

Model: livekit/turn-detector, revision v1.2.2-en. Qwen2.5-0.5B distilled and pruned,
INT8 ONNX, 66 MB, CPU. It scores P(<|im_end|> comes next) after the user message, so a
high number means "they stopped here". The graph bakes the softmax in and outputs that
probability directly as `prob`.

Measured on this machine (M5, 2 ONNX threads): load 0.31 s, 1.4 ms p50 per call,
4.1 ms at 68 tokens. Two orders of magnitude under the 60 ms budget, so it runs on
every silence tick without thought.

Why the deprecated English revision and not the recommended multilingual one: the
multilingual q8 is 396 MB against 66 MB and keeps the full 151 k vocab and LM head, so
it is far slower per call for a tool whose calls are all English. See
docs/research-stt.md for both sets of numbers.

THE TRAP: Parakeet punctuates every partial. A 1.5 s prefix comes out as
"So what did revenue look like?" -- with a question mark the speaker never uttered. Fed
raw, that "?" alone pushes the detector over the line: on real Parakeet partials it
called 18 of 26 complete, against 10 of 26 with the trailing punctuation stripped. So
`strip_tail()` removes trailing sentence punctuation before scoring. The ASR's guess
about where a sentence ends is not evidence about whether the speaker finished.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("presage.stt.eot")

MODEL_REPO = os.environ.get("PRESAGE_EOT_REPO", "livekit/turn-detector")
MODEL_REV = os.environ.get("PRESAGE_EOT_REV", "v1.2.2-en")
MAX_TOKENS = 128          # the model's documented context for this task
_EOU_TOKEN_ID = 151645    # <|im_end|> in the Qwen2.5 vocab
_TRAILING = " .?!,;:…"


def strip_tail(text: str) -> str:
    """Drop the ASR's guessed sentence punctuation. See THE TRAP above."""
    return text.rstrip(_TRAILING)


_QUESTION_WORDS = (
    "what", "how", "why", "when", "where", "who", "which", "whose", "can",
    "could", "would", "should", "do", "does", "did", "is", "are", "was", "were",
    "will", "have", "has", "walk", "tell", "show", "give", "remind",
)

# Cues that a question is being built. Multi-word ones count anywhere, because a
# consultant's question often starts mid-breath ("...and remind me how many").
_CUES_ANYWHERE = (
    "how many", "how much", "how did", "how do", "how does", "how's", "how is",
    "when did", "when do", "what's", "whats", "what did", "what was", "what were",
    "what do", "what does", "what are", "what is", "what i want",
    "can you", "could you", "would you", "should we", "do we", "do you", "does that",
    "did we", "did you", "is that", "is there", "are we", "are there", "any sense of",
    "was that", "were there", "walk me", "tell me", "show me", "remind me",
    "give me", "talk me through", "help me understand", "i want to know",
    "i'd like to know", "do we have", "is it", "are they", "which of",
)
# Bare question words only count as an opener. "That's what we assumed" is not a
# question; "What we assumed..." is on its way to being one.
_CUES_LEADING = (
    "what", "how", "why", "when", "where", "which", "who", "whose", "is", "are",
    "was", "were", "do", "does", "did", "can", "could", "would", "should", "will",
    "walk", "tell", "show", "remind", "give",
)
_DANGLING = (
    "and", "or", "but", "so", "the", "a", "an", "of", "for", "to", "in", "on",
    "at", "by", "with", "that", "this", "is", "was", "if", "then", "because",
)


def heuristic_complete(text: str) -> bool:
    """Does the recogniser's own punctuation say this is a finished sentence?

    Used two ways: as the PUNCT route in TurnDetector.judge(), and as the whole
    detector when the model will not load.

    A dangling function word beats any punctuation: Parakeet writes "Is that blended
    or" with no mark and "So what did revenue look like?" with one, and only the
    second is even a candidate.
    """
    t = text.strip()
    if not t:
        return False
    words = re.findall(r"[\w'\-]+", t.lower())
    if not words or words[-1] in _DANGLING:
        return False
    if t.endswith("?"):
        return len(words) >= 3 and (words[0] in _QUESTION_WORDS or len(words) >= 5)
    if t.endswith((".", "!")):
        # A short declarative is usually a fragment the speaker is still building:
        # "And remind me.", "Gross margin.", "Walk me through the." Five words is
        # where Parakeet's full stops start meaning something.
        return len(words) >= 5
    return False


def has_question_cue(text: str) -> bool:
    """Does this text open with, or contain, a phrase that starts a question?"""
    t = re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()
    if not t:
        return False
    if any(f" {c}" in f" {t}" for c in _CUES_ANYWHERE):
        return True
    first = re.findall(r"[\w'\-]+", t)
    # A discourse opener does not hide the question behind it: "So, what did..."
    while first and first[0] in ("so", "and", "but", "ok", "okay", "right", "um",
                                "uh", "well", "yeah", "now", "then", "also"):
        first = first[1:]
    return bool(first) and first[0] in _CUES_LEADING


class QuestionForming:
    """Per-utterance tracker for `Transcript.question_forming`.

    True when the committed text is a question under construction: the detector says
    it is NOT finished, and either the words carry a question cue or the completion
    probability is climbing toward one. Both halves matter.

    * The cue alone fires on "what" inside a statement.
    * The rising probability alone fires on any sentence approaching its end.
    * Requiring "not finished" is what makes it a *forming* signal rather than a
      finished one -- once the detector says finished, an eager final is coming and
      the brain should act on that instead.

    The rising test uses a small margin so ordinary decode jitter does not count, and
    the flag latches for the rest of the utterance once set: a question that looked
    like one at 2 s has not stopped being one at 3 s just because the tail dipped.
    """

    RISE = 0.004  # absolute increase in P(complete) that counts as "climbing"

    def __init__(self) -> None:
        self.prev_p = 0.0
        self.latched = False

    def reset(self) -> None:
        self.prev_p = 0.0
        self.latched = False

    def update(self, text: str, p: float, finished: bool) -> bool:
        rising = p > self.prev_p + self.RISE
        self.prev_p = p
        if has_question_cue(text) or rising:
            self.latched = True
        return self.latched and not finished


class TurnDetector:
    """Text-in, probability-out. Load once in start(); calls are ~1.4 ms.

    Not thread-safe by itself, but onnxruntime sessions are, and `local.py` only
    calls this from its single STT worker thread.
    """

    def __init__(self, threshold: float | None = None, threads: int = 2):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        d = Path(snapshot_download(
            MODEL_REPO, revision=MODEL_REV,
            allow_patterns=["onnx/model_q8.onnx", "languages.json", "tokenizer.json",
                            "tokenizer_config.json", "config.json",
                            "special_tokens_map.json", "added_tokens.json",
                            "merges.txt", "vocab.json"]))
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(d / "onnx" / "model_q8.onnx"), sess_options=opts,
            providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}

        vendor = 0.0289
        try:
            vendor = json.load(open(d / "languages.json"))["en"]["threshold"]
        except Exception:
            pass
        self.vendor_threshold = vendor
        self.threshold = (
            threshold if threshold is not None
            else float(os.environ.get("PRESAGE_EOT_THRESHOLD", str(vendor)))
        )
        self.punct_floor = float(os.environ.get("PRESAGE_EOT_PUNCT_FLOOR", "0.001"))
        log.info("turn-detector %s@%s loaded, threshold %.4f (vendor %.4f), "
                 "punct floor %.4f", MODEL_REPO, MODEL_REV, self.threshold, vendor,
                 self.punct_floor)

    def probability(self, text: str) -> float:
        """P(the speaker stopped here), 0-1."""
        body = strip_tail(text).strip()
        if not body:
            return 0.0
        ids = self._tok.encode(f"<|im_start|>user\n{body}",
                               add_special_tokens=False).ids[-MAX_TOKENS:]
        arr = np.array([ids], dtype=np.int64)
        feed = {"input_ids": arr}
        if "attention_mask" in self._inputs:
            feed["attention_mask"] = np.ones_like(arr)
        if "position_ids" in self._inputs:
            feed["position_ids"] = np.arange(arr.shape[1], dtype=np.int64)[None, :]
        out = self._sess.run(None, feed)[0]
        # Three shapes in the wild: v1.2.2-en exports one probability, v0.4.x-intl
        # exports a probability per position, and a raw LM export would give logits.
        if out.ndim == 1:
            return float(out[0])
        if out.ndim == 2:
            return float(out[0, -1])
        logits = out[0, -1, :].astype(np.float64)
        logits -= logits.max()
        e = np.exp(logits)
        return float(e[_EOU_TOKEN_ID] / e.sum())

    def judge(self, text: str) -> tuple[bool, float, str]:
        """(finished?, probability, why). Two routes to "finished":

        MODEL   the detector clears its threshold on punctuation-stripped text.
        PUNCT   the recogniser put a sentence-ending mark here AND the detector is not
                actively arguing against it (p >= punct_floor).

        The second route exists because the model systematically underrates trailing
        prepositional phrases: it scores "...in the headcount plan by June?" at 0.004,
        well under threshold, though a human hears a finished question. Measured on
        the replay set, adding the PUNCT route takes eager finals from 3 of 4 to 4 of
        4 and costs one extra premature fire, which retract undoes. The floor is what
        makes it safe: Parakeet punctuates every partial, so "...what did Q3
        actually?" also ends in a question mark, but the model scores it 0.00003 and
        vetoes it.
        """
        p = self.probability(text)
        if p >= self.threshold:
            return True, p, "model"
        if p >= self.punct_floor and heuristic_complete(text):
            return True, p, "punct"
        return False, p, "wait"

    def complete(self, text: str) -> bool:
        return self.judge(text)[0]


class HeuristicDetector:
    """Stand-in with the same shape, used when the model cannot load."""

    threshold = 0.5
    vendor_threshold = 0.5
    punct_floor = 0.0

    def probability(self, text: str) -> float:
        return 1.0 if heuristic_complete(text) else 0.0

    def judge(self, text: str) -> tuple[bool, float, str]:
        ok = heuristic_complete(text)
        return ok, 1.0 if ok else 0.0, "heuristic"

    def complete(self, text: str) -> bool:
        return heuristic_complete(text)


def load_detector() -> TurnDetector | HeuristicDetector:
    """Never raises. A broken model download degrades to the heuristic, loudly."""
    try:
        return TurnDetector()
    except Exception as e:  # noqa: BLE001
        log.warning("turn-detector unavailable (%s: %s); falling back to the "
                    "trailing-punctuation heuristic", type(e).__name__, e)
        return HeuristicDetector()


# ---------------------------------------------------------------- LocalAgreement-2

def common_prefix(a: list[str], b: list[str]) -> list[str]:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def _norm_word(w: str) -> str:
    """Compare words without punctuation or case.

    Parakeet repunctuates as it gets more context: "And remind me how many" becomes
    "And remind me, how many". Comparing raw tokens treats "me" and "me," as a
    disagreement, which un-commits a word we had already committed and renders the
    line twice. Agreement is about the words, not the commas.
    """
    return re.sub(r"[^\w'\-]", "", w.lower())


class StablePrefix:
    """LocalAgreement-2 from Whisper-Streaming (arXiv 2307.14743).

    Commit only what two consecutive re-transcriptions agree on. Presage prefetches on
    words, so committing a word the next decode will rewrite means an answer fetched
    for a question nobody asked. "Can you remind me what our net revenue?" becomes
    "...net revenue retention" one tick later; only "Can you remind me what our net"
    was ever safe.

    What is committed is a word COUNT, not a list of strings. The rendered text always
    comes from the newest hypothesis, so late repunctuation shows through, but the
    boundary between committed and unstable never moves backwards inside an utterance.
    """

    def __init__(self) -> None:
        self.prev: list[str] = []   # normalised words of the previous hypothesis
        self.n = 0                  # how many leading words are committed

    def reset(self) -> None:
        self.prev = []
        self.n = 0

    def update(self, text: str) -> tuple[str, str]:
        """Feed the newest full hypothesis; get (stable_text, unstable_tail)."""
        words = text.split()
        norm = [_norm_word(w) for w in words]
        agreed = len(common_prefix(self.prev, norm))
        self.prev = norm
        # Monotonic within an utterance, but never past the end of what we now have.
        self.n = min(max(self.n, agreed), len(words))
        return " ".join(words[: self.n]), " ".join(words[self.n:])
