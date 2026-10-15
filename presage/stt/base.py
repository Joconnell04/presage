"""Swappable STT provider interface.

Two audio sources, "mic" and "system", segmented independently. A provider emits
interim transcripts while an utterance is still open (is_final=False) and one final
transcript when it closes (is_final=True). The prefetcher downstream fires on words
in the interim text, so interims matter as much as finals.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator, Literal

import numpy as np

Source = Literal["mic", "system"]
SOURCES: tuple[str, ...] = ("mic", "system")


@dataclass(slots=True)
class Transcript:
    source: str          # "mic" | "system"
    text: str            # committed text: safe to act on, never revised (see `unstable`)
    is_final: bool       # False while the utterance is still open
    t0: float            # utterance start, seconds since provider.start()
    t1: float            # end of the audio this text covers, same clock
    provider: str        # "local" | "deepgram" | ...

    # --- added after first production use. All default, so old call sites still work.

    utt: int = 0
    """Per-source utterance counter, monotonic from 0. Two emissions with the same
    (source, utt) describe the SAME utterance, so the server keys transcript ids on
    that pair and the UI line updates in place instead of adding a row."""

    unstable: str = ""
    """The tail the recogniser has not agreed with itself about yet (LocalAgreement-2).
    `text` is the stable prefix; `text + " " + unstable` is the full current hypothesis.
    Render `unstable` dimmed; do not prefetch on it, it gets rewritten."""

    eager: bool = False
    """This final was committed early, on a short silence, because a turn detector
    judged the text complete. It may still be retracted; see `retracts`."""

    retracts: bool = False
    """This emission supersedes an earlier eager final with the same (source, utt):
    the speaker kept going. Drop whatever was built on that final and use this text."""

    question_forming: bool = False
    """The committed text reads like a question still being built: the turn detector
    says it is not finished, and either it opens with a question cue or its completion
    probability is climbing. The brain drafts on this rather than waiting for a final,
    in every mode. Always False on a final."""

    complete_p: float = 0.0
    """The turn detector's P(the speaker stopped here), 0-1, for the text in this
    emission. 0.0 when the detector is off or has not scored this text yet. Useful as
    a confidence dial downstream: a partial at 0.4 is nearly a question."""

    def __repr__(self) -> str:  # keep logs readable
        kind = "FINAL" if self.is_final else "part "
        flags = "".join(c for c, on in
                        (("E", self.eager), ("R", self.retracts),
                         ("Q", self.question_forming)) if on)
        flags = f"[{flags}]" if flags else ""
        tail = f" +{self.unstable!r}" if self.unstable else ""
        return (f"<{kind}{flags} {self.source}#{self.utt} "
                f"{self.t0:6.2f}-{self.t1:6.2f} p={self.complete_p:.3f} "
                f"{self.text!r}{tail}>")


class STTProvider(abc.ABC):
    """Feed PCM in, get Transcripts out.

    Lifecycle: start() -> feed()/transcripts() concurrently -> stop().
    feed() must never block the event loop; do inference in an executor.
    """

    name: str = "base"

    @abc.abstractmethod
    async def start(self) -> None:
        """Load models, open connections. Do the slow work here, not on first feed()."""

    @abc.abstractmethod
    async def feed(self, source: str, pcm: np.ndarray) -> None:
        """Append int16 mono 16 kHz samples for one source.

        pcm is np.ndarray dtype=int16, shape (n,). Typically 20 ms (320 samples).
        """

    @abc.abstractmethod
    def transcripts(self) -> AsyncIterator[Transcript]:
        """Interims and finals for every source, interleaved, in emission order."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Flush open utterances, close connections, release the model."""
