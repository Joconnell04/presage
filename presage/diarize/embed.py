"""Speaker embeddings from an ONNX graph, on onnxruntime, with no new dependencies.

presage already ships `onnxruntime` (for Silero VAD) and `numpy`. A speaker-embedding
model is one more .onnx file on disk, so the whole diarization route costs a download
and nothing in the environment. That is the reason this exists instead of pyannote or
SpeechBrain, both of which want torch.

The models come from the sherpa-onnx model zoo, which carries its preprocessing recipe
in the ONNX metadata: `framework`, `normalize_samples`, `feature_normalize_type`,
`window_type`, `output_dim`. This class reads that metadata and does what it says, so
the same code drives WeSpeaker, 3D-Speaker and NeMo graphs.

    e = Embedder()                       # default: 3D-Speaker CAM++ en, 29.6 MB
    v = e(samples_float32)               # -> (dim,) float32, L2-normalised
    Embedder.similarity(v1, v2)          # cosine, in [-1, 1]

Models are cached in ~/.cache/presage/diar/models and downloaded on first use.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from .fbank import cmn, fbank, per_feature_norm

ZOO = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
       "speaker-recongition-models")   # the upstream tag really is spelled that way

# name -> (file, approximate MB). All Apache-2.0 / permissive, all English VoxCeleb.
MODELS: dict[str, tuple[str, float]] = {
    "resnet34":  ("wespeaker_en_voxceleb_resnet34_LM.onnx", 26.5),
    "campplus":  ("wespeaker_en_voxceleb_CAM++_LM.onnx", 29.3),
    "campplus3": ("3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx", 29.6),
    "eres2net":  ("3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx", 26.5),
    "titanet":   ("nemo_en_titanet_small.onnx", 40.3),
}
DEFAULT_MODEL = os.environ.get("PRESAGE_DIAR_EMBED", "campplus3")
CACHE = Path(os.environ.get("PRESAGE_DIAR_CACHE",
                            Path.home() / ".cache" / "presage" / "diar" / "models"))


def model_path(name: str = DEFAULT_MODEL, download: bool = True) -> Path:
    if name not in MODELS:
        raise ValueError(f"unknown embedder {name!r}; have {', '.join(MODELS)}")
    fname = MODELS[name][0]
    path = CACHE / fname
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    if not download:
        raise FileNotFoundError(f"{path} not present; run with download=True")
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(f"{ZOO}/{fname}", tmp)   # noqa: S310  a pinned GitHub release
    tmp.replace(path)
    return path


class Embedder:
    """One ONNX speaker-embedding graph. Not thread-safe; one instance per thread."""

    def __init__(self, name: str = DEFAULT_MODEL, threads: int = 1,
                 provider: str = "cpu") -> None:
        import onnxruntime as ort

        self.name = name
        self.path = model_path(name)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
        opts.log_severity_level = 3
        # CPU only unless asked: the CoreML provider shares Metal with the MLX transcriber and
        # killed the server on "commit an already committed command buffer" in 8 of 20 starts
        # (docs/experiments/E5-simulator.md); the CPU path is 3.6% of one core anyway
        providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                     if os.environ.get("PRESAGE_DIAR_COREML", "0").lower() in ("1", "true", "on")
                     else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(str(self.path), sess_options=opts,
                                         providers=providers)

        meta = dict(self.sess.get_modelmeta().custom_metadata_map)
        self.framework = meta.get("framework", "")
        self.scale_to_int16 = meta.get("normalize_samples", "1") == "0"
        self.feature_norm = meta.get("feature_normalize_type", "")
        self.window_type = meta.get("window_type", "povey")
        self.dim = int(meta.get("output_dim", 0) or 0)
        self.size_mb = self.path.stat().st_size / 1e6

        self.inputs = [i.name for i in self.sess.get_inputs()]
        # NeMo graphs are feature-major and take a length; the rest are (B, T, F).
        self.feat_major = self.framework == "nemo"
        outs = [o.name for o in self.sess.get_outputs()]
        self.out_name = ("embs" if "embs" in outs
                         else "embedding" if "embedding" in outs else outs[-1])

    # ------------------------------------------------------------------ inference

    def features(self, samples: np.ndarray) -> np.ndarray:
        """float32 samples in [-1, 1] -> the exact tensor this graph was trained on."""
        x = np.asarray(samples, dtype=np.float32)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if self.scale_to_int16:
            x = x * 32768.0
        feats = fbank(x, window_type=self.window_type)
        if self.feature_norm == "global-mean":
            feats = cmn(feats)
        elif self.feature_norm == "per_feature":
            feats = per_feature_norm(feats)
        return feats

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """One window of audio -> one L2-normalised embedding."""
        feats = self.features(samples)
        if feats.shape[0] == 0:
            return np.zeros(self.dim or 256, dtype=np.float32)
        if self.feat_major:
            tensor = feats.T[None, :, :].astype(np.float32)          # (1, 80, T)
            feed = {self.inputs[0]: tensor,
                    self.inputs[1]: np.array([feats.shape[0]], dtype=np.int64)}
        else:
            feed = {self.inputs[0]: feats[None, :, :].astype(np.float32)}  # (1, T, 80)
        emb = self.sess.run([self.out_name], feed)[0][0].astype(np.float32)
        return normalize(emb)

    def batch(self, windows: list[np.ndarray]) -> np.ndarray:
        """Several equal-length windows in one graph call. Used by bench.py, not live."""
        if not windows:
            return np.zeros((0, self.dim or 256), dtype=np.float32)
        feats = [self.features(w) for w in windows]
        n = min(f.shape[0] for f in feats)
        if n == 0:
            return np.stack([self(w) for w in windows])
        stacked = np.stack([f[:n] for f in feats]).astype(np.float32)
        if self.feat_major:
            feed = {self.inputs[0]: stacked.transpose(0, 2, 1),
                    self.inputs[1]: np.full(len(windows), n, dtype=np.int64)}
        else:
            feed = {self.inputs[0]: stacked}
        embs = self.sess.run([self.out_name], feed)[0].astype(np.float32)
        return np.stack([normalize(e) for e in embs])


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Both arguments are assumed L2-normalised already."""
    return float(np.dot(a, b))
