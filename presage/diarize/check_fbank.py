"""Prove the numpy fbank matches kaldi-native-fbank, so the no-torch route is honest.

sherpa-onnx wraps the same ONNX graphs with C++ kaldi-native-fbank features. If this
file's embeddings and sherpa-onnx's embeddings agree to cosine ~1.0 on the same audio,
the numpy front end is correct and presage can keep its dependency list as it is.

    uv run --no-project --with sherpa-onnx --with onnxruntime --with numpy \
        python -m presage.diarize.check_fbank docs/replay/diar/far2.wav

Run it from the repo root; it puts the repo on sys.path itself.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from presage.diarize.embed import MODELS, Embedder, model_path, similarity  # noqa: E402


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0)


def main() -> None:
    import sherpa_onnx

    wav = sys.argv[1] if len(sys.argv) > 1 else "docs/replay/diar/far2.wav"
    samples = read_wav(wav)[: 16000 * 8]      # 8 s is plenty to expose a wrong filter
    print(f"{wav}: {len(samples)/16000:.1f}s\n")

    for name in MODELS:
        try:
            path = model_path(name, download=False)
        except FileNotFoundError:
            print(f"{name:10s} not downloaded, skipped")
            continue

        mine = Embedder(name)
        v_mine = mine(samples)

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(path), num_threads=1, debug=0, provider="cpu")
        ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        st = ext.create_stream()
        st.accept_waveform(sample_rate=16000, waveform=samples)
        st.input_finished()
        v_ref = np.asarray(ext.compute(st), dtype=np.float32)
        v_ref /= np.linalg.norm(v_ref)

        cos = similarity(v_mine, v_ref)
        flag = "OK " if cos > 0.9999 else "!! "
        print(f"{flag}{name:10s} dim={mine.dim:4d} {mine.size_mb:5.1f} MB  "
              f"cos(numpy, kaldi-native-fbank) = {cos:.6f}")


if __name__ == "__main__":
    main()
