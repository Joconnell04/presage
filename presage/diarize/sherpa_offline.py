"""The offline reference point: sherpa-onnx's whole-file diarization on the E3 clips.

This is not a candidate for presage -- it needs the entire call before it says
anything, which is the opposite of what a live copilot wants -- but it is the number
the online route has to be judged against: pyannote segmentation-3.0 plus the same
speaker embeddings plus spectral clustering over the complete recording.

    uv run --no-project --with sherpa-onnx --with numpy \
        python -m presage.diarize.sherpa_offline

Writes one RTTM per clip to /tmp/sherpaout and prints DER with the same scorer
`bench.py` uses. Needs the segmentation model:

    cd ~/.cache/presage/diar/models
    curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/\
speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
    tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from presage.diarize import der                                    # noqa: E402
from presage.diarize.embed import CACHE, MODELS, model_path        # noqa: E402

CLIPS = Path(__file__).resolve().parents[2] / "docs" / "replay" / "diar"
SEG = CACHE / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
OUT = Path("/tmp/sherpaout")


def read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(
            np.float32) / 32768.0


def main() -> None:
    import sherpa_onnx

    if not SEG.exists():
        raise SystemExit(f"segmentation model missing at {SEG}; see this file's docstring")
    OUT.mkdir(exist_ok=True)
    embed = sys.argv[1] if len(sys.argv) > 1 else "campplus3"
    emb_path = str(model_path(embed, download=False))
    print(f"segmentation: pyannote-3.0 ({SEG.stat().st_size/1e6:.1f} MB)   "
          f"embedding: {embed} ({MODELS[embed][1]} MB)\n")

    print(f"{'clip':14s} {'spk':>7s} {'DER@.25':>8s} {'DER@0':>7s} {'proc_s':>7s} {'RTF':>6s}")
    for clip in ("far2", "far3", "far3_overlap", "far2_me", "room3"):
        wav = CLIPS / f"{clip}.wav"
        samples = read(wav)

        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(SEG)),
                num_threads=1),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=emb_path, num_threads=1),
            clustering=sherpa_onnx.FastClusteringConfig(threshold=0.5),
            min_duration_on=0.2, min_duration_off=0.3)
        sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

        t0 = time.perf_counter()
        result = sd.process(samples).sort_by_start_time()
        proc = time.perf_counter() - t0

        hyp = [der.Segment(s.start, s.end, f"S{s.speaker}") for s in result]
        der.write_rttm(OUT / f"{clip}.rttm", hyp, clip)
        ref = der.read_rttm(CLIPS / f"{clip}.rttm")
        dur = len(samples) / 16000
        s25 = der.score(ref, hyp, collar=0.25, duration=dur)
        s00 = der.score(ref, hyp, collar=0.0, duration=dur)
        print(f"{clip:14s} {s25['hyp_speakers']}/{s25['ref_speakers']:<5d} "
              f"{100*s25['der']:7.2f}% {100*s00['der']:6.2f}% "
              f"{proc:7.2f} {proc/dur:6.3f}")


if __name__ == "__main__":
    main()
