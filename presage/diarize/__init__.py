"""Speaker labels for presage's audio streams (experiment E3).

The "system" source mixes every remote participant, so a card cannot say who asked.
An in-person meeting captured by one microphone is worse: Me and Them share a channel.
This package turns a stream of 20 ms int16 frames into `(t, speaker)` labels with a
label delay under a second, and identifies Me from a short enrollment of the mic.

    from presage.diarize import OnlineDiarizer
    d = OnlineDiarizer()                       # loads the embedder, ~30 MB, onnxruntime
    d.enroll_me(mic_pcm_10s)                   # optional
    for label in d.feed(pcm_20ms):             # -> [SpeakerLabel(...), ...]
        ...

Nothing here imports or modifies presage.stt or presage.audio*; the diarizer runs its
own Silero VAD instance on the same frames the STT provider gets.
"""

from __future__ import annotations

from typing import Any

__all__ = ["OnlineDiarizer", "SpeakerLabel"]


def __getattr__(name: str) -> Any:
    """Lazy, so `python -m presage.diarize.clips` does not load onnxruntime."""
    if name in __all__:
        from . import online
        return getattr(online, name)
    raise AttributeError(name)
