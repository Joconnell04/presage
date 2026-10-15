"""Kaldi-compatible log-mel filterbank in numpy. No torch, no torchaudio, no C extension.

The speaker-embedding ONNX graphs in the sherpa-onnx model zoo all want the same thing
on their input: 80-dimensional Kaldi fbank, 25 ms window, 10 ms shift, Povey window,
pre-emphasis 0.97, power spectrum, natural log. sherpa-onnx computes that with C++
`kaldi-native-fbank`; presage already ships onnxruntime and numpy and nothing else, so
the features are computed here instead and diarization adds no package to the venv.

Three settings are not the Kaldi defaults and all three matter. They are sherpa-onnx's
`FeatureExtractorConfig` defaults, which is what these graphs were fed at export time:

    high_freq = -400   ->  the mel bank stops at 7600 Hz, not at Nyquist
    snip_edges = false ->  frames are centred and the signal is mirrored at both ends
    dither = 0         ->  no noise, so digital silence stays digital silence

Getting high_freq wrong moves every filter and costs about 0.08 of cosine on an
embedding, which is enough to lose a speaker. Verified against kaldi-native-fbank on
far2.wav: mean |diff| 7.0e-06 over 5773 x 80 values (max 7.0e-02, all of it in digital
silence, where log(FLT_EPSILON) magnifies nothing into something), and the resulting
embeddings from all four WeSpeaker / 3D-Speaker graphs agree with sherpa-onnx's own to
cosine 1.000000 (`presage.diarize.check_fbank`).

NeMo graphs (TitaNet) are NOT matched: they want low_freq 0, no DC removal, a Hann
window and a librosa-style mel bank, which is a different filter construction. Cosine
against sherpa-onnx is 0.65, so TitaNet is excluded from the route rather than fudged.

    feats = fbank(samples_float32)        # -> (frames, 80) float32
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FRAME_LENGTH = 400        # 25 ms
FRAME_SHIFT = 160         # 10 ms
FFT_SIZE = 512            # next power of two above FRAME_LENGTH
PREEMPH = 0.97
LOW_FREQ = 20.0
HIGH_FREQ = -400.0        # negative means "Nyquist + this", i.e. 7600 Hz
EPS = float(np.finfo(np.float32).eps)   # Kaldi floors the mel energy at FLT_EPSILON


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_banks(num_bins: int = NUM_MEL_BINS,
               low_freq: float = LOW_FREQ,
               high_freq: float = HIGH_FREQ) -> np.ndarray:
    """Kaldi's triangular mel bank as a (num_bins, FFT_SIZE//2) matrix.

    Note FFT_SIZE//2, not //2 + 1: Kaldi builds its bins over `padded_window_size / 2`
    bins and never touches the Nyquist bin. Off by one there shifts every filter.
    """
    nyquist = 0.5 * SAMPLE_RATE
    high = high_freq if high_freq > 0 else nyquist + high_freq
    num_fft_bins = FFT_SIZE // 2
    fft_bin_width = SAMPLE_RATE / FFT_SIZE

    mel_low, mel_high = _mel(low_freq), _mel(high)
    delta = (mel_high - mel_low) / (num_bins + 1)

    mels = _mel(fft_bin_width * np.arange(num_fft_bins, dtype=np.float64))
    banks = np.zeros((num_bins, num_fft_bins), dtype=np.float64)
    for b in range(num_bins):
        left = mel_low + b * delta
        center, right = left + delta, left + 2 * delta
        w = np.minimum((mels - left) / delta, (right - mels) / delta)
        banks[b] = np.where((mels > left) & (mels < right), np.maximum(w, 0.0), 0.0)
    return banks.astype(np.float32)


def _window(kind: str = "povey") -> np.ndarray:
    n = np.arange(FRAME_LENGTH, dtype=np.float64)
    hann_like = 0.5 - 0.5 * np.cos(2 * np.pi * n / (FRAME_LENGTH - 1))
    if kind == "povey":
        return np.power(hann_like, 0.85).astype(np.float32)
    if kind in ("hanning", "hann"):
        return hann_like.astype(np.float32)
    if kind == "hamming":
        return (0.54 - 0.46 * np.cos(2 * np.pi * n / (FRAME_LENGTH - 1))).astype(np.float32)
    if kind == "rectangular":
        return np.ones(FRAME_LENGTH, dtype=np.float32)
    raise ValueError(f"unknown window {kind!r}")


_BANKS_CACHE: dict[tuple, np.ndarray] = {}
_WINDOWS: dict[str, np.ndarray] = {}


def _frames(x: np.ndarray, snip_edges: bool) -> np.ndarray:
    """Kaldi framing. snip_edges=False centres the frames and mirrors past the ends."""
    n = x.shape[0]
    if snip_edges:
        if n < FRAME_LENGTH:
            return np.zeros((0, FRAME_LENGTH), dtype=np.float32)
        num = 1 + (n - FRAME_LENGTH) // FRAME_SHIFT
        starts_pad, pre = 0, x
    else:
        num = (n + FRAME_SHIFT // 2) // FRAME_SHIFT
        if num <= 0:
            return np.zeros((0, FRAME_LENGTH), dtype=np.float32)
        # First sample of frame f is f*shift + shift/2 - length/2 == f*shift - 120.
        starts_pad = FRAME_LENGTH // 2 - FRAME_SHIFT // 2
        end = (num - 1) * FRAME_SHIFT - starts_pad + FRAME_LENGTH
        right = max(0, end - n)
        # 'symmetric' is Kaldi's mirroring exactly: index -1 -> 0, index n -> n-1.
        pre = np.pad(x, (starts_pad, right), mode="symmetric")
    return np.lib.stride_tricks.as_strided(
        pre, shape=(num, FRAME_LENGTH),
        strides=(pre.strides[0] * FRAME_SHIFT, pre.strides[0]),
    ).astype(np.float32, copy=True)


def fbank(samples: np.ndarray,
          window_type: str = "povey",
          snip_edges: bool = False,
          low_freq: float = LOW_FREQ,
          high_freq: float = HIGH_FREQ,
          remove_dc_offset: bool = True) -> np.ndarray:
    """80-dim log-mel fbank for float32 samples. Returns (frames, 80)."""
    x = np.ascontiguousarray(samples, dtype=np.float32)
    frames = _frames(x, snip_edges)
    if frames.shape[0] == 0:
        return np.zeros((0, NUM_MEL_BINS), dtype=np.float32)

    if window_type not in _WINDOWS:
        _WINDOWS[window_type] = _window(window_type)
    key = (low_freq, high_freq)
    if key not in _BANKS_CACHE:
        _BANKS_CACHE[key] = _mel_banks(low_freq=low_freq, high_freq=high_freq)
    banks = _BANKS_CACHE[key]

    if remove_dc_offset:
        frames -= frames.mean(axis=1, keepdims=True)
    # Pre-emphasis, Kaldi style: the first sample is emphasised against itself.
    frames -= PREEMPH * np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames *= _WINDOWS[window_type]

    spec = np.fft.rfft(frames, n=FFT_SIZE, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)[:, :FFT_SIZE // 2]
    mel = power @ banks.T
    np.maximum(mel, EPS, out=mel)
    return np.log(mel, out=mel)


def cmn(feats: np.ndarray) -> np.ndarray:
    """Subtract the per-dimension mean over time ("global-mean" in the model metadata)."""
    return feats - feats.mean(axis=0, keepdims=True)


def per_feature_norm(feats: np.ndarray) -> np.ndarray:
    """NeMo's normalisation: zero mean, unit variance per dimension over the utterance."""
    return (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-5)
