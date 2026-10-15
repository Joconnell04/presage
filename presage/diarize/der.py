"""Diarization error rate, the NIST md-eval definition, in numpy.

    DER = (missed + false alarm + confusion) / total reference speaker time

computed on a 10 ms grid after the hypothesis speakers have been mapped onto the
reference speakers by the mapping that maximises agreement. pyannote.metrics would do
this too, but it wants torch and pandas; this is 120 lines and has no dependencies,
which keeps the whole E3 route installable as "one .onnx file".

Two knobs, both standard, both reported because they change the number a lot:

    collar          seconds excluded either side of every reference boundary.
                    0.25 is the CALLHOME / DIHARD convention; 0.0 is unforgiving.
    skip_overlap    drop frames where the reference has two speakers talking.
                    CALLHOME numbers in the literature are usually WITH this on.

    score(ref_rttm, hyp_segments) -> {"der": .., "missed": .., "fa": .., "conf": ..}
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

FRAME_S = 0.010


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    speaker: str


def read_rttm(path: str | Path) -> list[Segment]:
    segs = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        t0, dur, spk = float(parts[3]), float(parts[4]), parts[7]
        segs.append(Segment(t0, t0 + dur, spk))
    return sorted(segs, key=lambda s: s.t0)


def write_rttm(path: str | Path, segs: list[Segment], uri: str = "clip") -> None:
    with Path(path).open("w") as f:
        for s in sorted(segs, key=lambda s: s.t0):
            f.write(f"SPEAKER {uri} 1 {s.t0:.3f} {s.t1 - s.t0:.3f} "
                    f"<NA> <NA> {s.speaker} <NA> <NA>\n")


def _grid(segs: list[Segment], names: list[str], n_frames: int) -> np.ndarray:
    """(n_speakers, n_frames) bool occupancy."""
    idx = {n: i for i, n in enumerate(names)}
    g = np.zeros((len(names), n_frames), dtype=bool)
    for s in segs:
        a = max(0, int(round(s.t0 / FRAME_S)))
        b = min(n_frames, int(round(s.t1 / FRAME_S)))
        if b > a:
            g[idx[s.speaker], a:b] = True
    return g


def _boundary_mask(segs: list[Segment], n_frames: int, collar: float) -> np.ndarray:
    """False on frames within `collar` of any reference segment edge."""
    keep = np.ones(n_frames, dtype=bool)
    if collar <= 0:
        return keep
    half = int(round(collar / FRAME_S))
    for s in segs:
        for edge in (s.t0, s.t1):
            c = int(round(edge / FRAME_S))
            keep[max(0, c - half): min(n_frames, c + half + 1)] = False
    return keep


def score(ref: list[Segment], hyp: list[Segment], *,
          collar: float = 0.25, skip_overlap: bool = False,
          duration: float | None = None) -> dict:
    """DER and its three parts. Speaker names in ref and hyp need not match."""
    if duration is None:
        duration = max([s.t1 for s in ref + hyp] or [0.0])
    n = int(np.ceil(duration / FRAME_S)) + 1

    ref_names = sorted({s.speaker for s in ref})
    hyp_names = sorted({s.speaker for s in hyp})
    R = _grid(ref, ref_names, n)
    H = _grid(hyp, hyp_names, n)

    keep = _boundary_mask(ref, n, collar)
    if skip_overlap:
        keep &= R.sum(axis=0) <= 1
    R, H = R[:, keep], H[:, keep]

    n_ref, n_hyp = R.sum(axis=0), H.sum(axis=0)
    total = float(n_ref.sum()) * FRAME_S
    if total <= 0:
        return {"der": 0.0, "missed": 0.0, "fa": 0.0, "conf": 0.0,
                "total_s": 0.0, "mapping": {}, "ref_speakers": len(ref_names),
                "hyp_speakers": len(hyp_names)}

    missed = float(np.maximum(n_ref - n_hyp, 0).sum()) * FRAME_S
    fa = float(np.maximum(n_hyp - n_ref, 0).sum()) * FRAME_S

    # Best mapping of hypothesis speakers onto reference speakers. Brute force is
    # exact and instant at call sizes; the greedy fallback only fires past 7 speakers.
    co = (R.astype(np.int32) @ H.astype(np.int32).T)      # (n_ref, n_hyp) frames agreed
    mapping, correct = _best_mapping(co, ref_names, hyp_names)
    correct_s = float(correct) * FRAME_S

    conf = float(np.minimum(n_ref, n_hyp).sum()) * FRAME_S - correct_s
    conf = max(conf, 0.0)
    return {"der": (missed + fa + conf) / total,
            "missed": missed / total,
            "fa": fa / total,
            "conf": conf / total,
            "total_s": total,
            "mapping": mapping,
            "ref_speakers": len(ref_names),
            "hyp_speakers": len(hyp_names)}


def _best_mapping(co: np.ndarray, ref_names: list[str],
                  hyp_names: list[str]) -> tuple[dict[str, str], int]:
    n_ref, n_hyp = co.shape
    if n_ref == 0 or n_hyp == 0:
        return {}, 0
    if max(n_ref, n_hyp) <= 7:
        best, best_map = -1, {}
        if n_hyp <= n_ref:
            for perm in permutations(range(n_ref), n_hyp):
                total = int(sum(co[perm[j], j] for j in range(n_hyp)))
                if total > best:
                    best = total
                    best_map = {hyp_names[j]: ref_names[perm[j]] for j in range(n_hyp)}
        else:
            for perm in permutations(range(n_hyp), n_ref):
                total = int(sum(co[i, perm[i]] for i in range(n_ref)))
                if total > best:
                    best = total
                    best_map = {hyp_names[perm[i]]: ref_names[i] for i in range(n_ref)}
        return best_map, best
    # Greedy: take the largest remaining cell each time.
    used_r, used_h, total, mapping = set(), set(), 0, {}
    order = np.dstack(np.unravel_index(np.argsort(-co, axis=None), co.shape))[0]
    for i, j in order:
        if i in used_r or j in used_h:
            continue
        used_r.add(int(i))
        used_h.add(int(j))
        mapping[hyp_names[j]] = ref_names[i]
        total += int(co[i, j])
    return mapping, total


def binary_accuracy(ref: list[Segment], hyp: list[Segment], target: str, *,
                    duration: float | None = None) -> dict:
    """How well the hypothesis finds one named speaker. Used for the Me test.

    Scored on reference speech frames only: silence is the VAD's problem, not the
    speaker identifier's. Returns precision / recall / frame accuracy for `target`.
    """
    if duration is None:
        duration = max([s.t1 for s in ref + hyp] or [0.0])
    n = int(np.ceil(duration / FRAME_S)) + 1
    ref_names = sorted({s.speaker for s in ref})
    R = _grid(ref, ref_names, n)
    speech = R.sum(axis=0) > 0
    ref_is = _grid([s for s in ref if s.speaker == target], [target], n)[0] if target in ref_names \
        else np.zeros(n, dtype=bool)
    hyp_is = _grid([s for s in hyp if s.speaker == target], [target], n)[0] \
        if any(s.speaker == target for s in hyp) else np.zeros(n, dtype=bool)

    r, h = ref_is[speech], hyp_is[speech]
    tp = float((r & h).sum())
    fp = float((~r & h).sum())
    fn = float((r & ~h).sum())
    tn = float((~r & ~h).sum())
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "accuracy": (tp + tn) / max(1.0, tp + tn + fp + fn),
            "target_s": (tp + fn) * FRAME_S,
            "speech_s": float(speech.sum()) * FRAME_S}
