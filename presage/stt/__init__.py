"""STT providers for presage.

    from presage.stt import make_provider
    p = make_provider()            # STT_PROVIDER env var, default "local"
    await p.start()
"""

from __future__ import annotations

import os

from .base import SOURCES, STTProvider, Transcript

__all__ = ["make_provider", "STTProvider", "Transcript", "SOURCES"]


def make_provider(name: str | None = None) -> STTProvider:
    """Build a provider. name overrides STT_PROVIDER, which defaults to "local"."""
    name = (name or os.environ.get("STT_PROVIDER") or "local").lower()

    if name == "local":
        from .local import LocalProvider
        return LocalProvider()

    if name == "deepgram":
        from .deepgram import DeepgramProvider
        return DeepgramProvider()

    if name in ("granite-nar", "granite", "qwen3"):      # E69a, opt-in: needs `uv sync --extra stt-alt`
        from .mlxaudio import MlxAudioProvider
        return MlxAudioProvider(name)

    raise ValueError(f"unknown STT provider {name!r} (have: local, deepgram, granite-nar, granite, qwen3)")
