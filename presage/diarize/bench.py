"""Measure the diarization routes on the E3 clips: DER, label delay, CPU.

    uv run python -m presage.diarize.bench                     # the default route
    uv run python -m presage.diarize.bench --models resnet34,campplus,campplus3,eres2net
    uv run python -m presage.diarize.bench --sweep             # threshold sweep
    uv run python -m presage.diarize.bench --me                # Me enrollment tests
    uv run python -m presage.diarize.bench --fluid /tmp/fluidout   # score FluidAudio JSON
    uv run python -m presage.diarize.bench --all               # all of the above

Audio is fed the way `presage.audio_session` feeds it: 20 ms int16 chunks, in order,
one call per chunk. Nothing is played.

The three numbers, and what each one means:

  DER       missed + false alarm + confusion over reference speech. Reported at
            collar 0.25 s (the CALLHOME convention) and at collar 0, which is the
            honest number for a live system because a card shown 200 ms late is
            still late. LIVE is the first label for each stretch; SETTLED applies
            the revisions that land when a speech run closes.
  DELAY     seconds from a speaker change in the reference to the first emitted
            label that names that speaker correctly. This is the budget: under 1 s.
  CPU       process CPU seconds per second of audio, as a percentage of one core.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import wave
from pathlib import Path

import numpy as np

from . import der
from .embed import MODELS, Embedder
from .online import OnlineDiarizer

REPO = Path(__file__).resolve().parents[2]
CLIPS = REPO / "docs" / "replay" / "diar"
CHUNK = 320          # 20 ms, what audio_session delivers
FAR_CLIPS = ["far2", "far3", "far3_overlap", "far2_me", "room3", "real2", "real3"]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


# --------------------------------------------------------------------------- running

def run_clip(clip: str, embedder: Embedder, *, enroll: np.ndarray | None = None,
             **kw) -> dict:
    """Stream one clip through OnlineDiarizer and collect labels + cost."""
    pcm = read_wav(CLIPS / f"{clip}.wav")
    d = OnlineDiarizer(embedder, **kw)
    if enroll is not None:
        d.enroll_me(enroll)

    labels = []
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for i in range(0, len(pcm), CHUNK):
        labels.extend(d.feed(pcm[i:i + CHUNK]))
    labels.extend(d.flush())
    cpu = time.process_time() - cpu0
    wall = time.perf_counter() - wall0

    audio_s = len(pcm) / 16000
    return {"clip": clip, "labels": labels, "audio_s": audio_s,
            "cpu_s": cpu, "wall_s": wall,
            "cpu_pct": 100 * cpu / audio_s, "rtf": wall / audio_s,
            "speakers": d.speakers(), "timings": d.timings()}


def timelines(labels) -> tuple[list[der.Segment], list[der.Segment]]:
    """LIVE (first label per interval) and SETTLED (revisions applied)."""
    live = [der.Segment(l.t0, l.t1, l.speaker) for l in labels if not l.revises]
    settled = {(round(l.t0, 3), round(l.t1, 3)): l.speaker
               for l in labels if not l.revises}
    for l in labels:
        if l.revises:
            settled[(round(l.t0, 3), round(l.t1, 3))] = l.speaker
    return live, [der.Segment(a, b, s) for (a, b), s in sorted(settled.items())]


def label_delays(ref: list[der.Segment], labels, mapping: dict[str, str]) -> dict:
    """Seconds from a speaker change to the first correct label naming that speaker."""
    turns = []
    prev = None
    for s in ref:
        if s.speaker != prev:
            turns.append(s)
        prev = s.speaker

    delays, misses = [], 0
    for s in turns:
        hit = None
        for l in labels:
            if l.revises or l.t1 <= s.t0:
                continue
            if mapping.get(l.speaker, l.speaker) == s.speaker:
                hit = l
                break
            if l.t0 >= s.t1:        # the whole turn went by under another name
                break
        if hit is None:
            misses += 1
        else:
            delays.append(max(0.0, hit.emitted_at - s.t0))
    delays.sort()
    return {"turns": len(turns), "missed_turns": misses,
            "p50": statistics.median(delays) if delays else float("nan"),
            "p90": delays[int(0.9 * (len(delays) - 1))] if delays else float("nan"),
            "max": delays[-1] if delays else float("nan")}


def score_clip(clip: str, run: dict) -> dict:
    ref = der.read_rttm(CLIPS / f"{clip}.rttm")
    dur = run["audio_s"]
    live, settled = timelines(run["labels"])
    s25 = der.score(ref, live, collar=0.25, duration=dur)
    s00 = der.score(ref, live, collar=0.0, duration=dur)
    t25 = der.score(ref, settled, collar=0.25, duration=dur)
    delay = label_delays(ref, run["labels"], s25["mapping"])
    return {"clip": clip,
            "der_live_25": s25["der"], "der_live_00": s00["der"],
            "der_settled_25": t25["der"],
            "missed": s00["missed"], "fa": s00["fa"], "conf": s00["conf"],
            "conf_25": s25["conf"],
            "ref_speakers": s25["ref_speakers"], "hyp_speakers": s25["hyp_speakers"],
            "delay_p50": delay["p50"], "delay_p90": delay["p90"],
            "delay_max": delay["max"], "missed_turns": delay["missed_turns"],
            "turns": delay["turns"],
            "cpu_pct": run["cpu_pct"], "rtf": run["rtf"],
            "embed_ms": run["timings"]["embed_ms_mean"],
            "mapping": s25["mapping"]}


# --------------------------------------------------------------------------- reports

def table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"{'clip':14s} {'spk':>7s} {'DER@.25':>8s} {'DER@0':>7s} {'settled':>8s} "
          f"{'conf':>6s} {'d50':>6s} {'d90':>6s} {'miss':>5s} {'CPU%':>6s} {'RTF':>6s}")
    for r in rows:
        print(f"{r['clip']:14s} {r['hyp_speakers']}/{r['ref_speakers']:<5d} "
              f"{100*r['der_live_25']:7.2f}% {100*r['der_live_00']:6.2f}% "
              f"{100*r['der_settled_25']:7.2f}% {100*r['conf_25']:5.2f}% "
              f"{r['delay_p50']:6.2f} {r['delay_p90']:6.2f} "
              f"{r['missed_turns']:5d} {r['cpu_pct']:5.1f}% {r['rtf']:6.3f}")
    if rows:
        print(f"{'MEAN':14s} {'':7s} "
              f"{100*statistics.mean(r['der_live_25'] for r in rows):7.2f}% "
              f"{100*statistics.mean(r['der_live_00'] for r in rows):6.2f}% "
              f"{100*statistics.mean(r['der_settled_25'] for r in rows):7.2f}% "
              f"{100*statistics.mean(r['conf_25'] for r in rows):5.2f}% "
              f"{statistics.mean(r['delay_p50'] for r in rows):6.2f} "
              f"{statistics.mean(r['delay_p90'] for r in rows):6.2f} "
              f"{sum(r['missed_turns'] for r in rows):5d} "
              f"{statistics.mean(r['cpu_pct'] for r in rows):5.1f}% "
              f"{statistics.mean(r['rtf'] for r in rows):6.3f}")


# --------------------------------------------------------------------------- suites

def bench_models(names: list[str], clips: list[str], **kw) -> dict:
    out = {}
    for name in names:
        emb = Embedder(name)
        rows = [score_clip(c, run_clip(c, emb, **kw)) for c in clips]
        table(rows, f"embedder = {name}  ({emb.size_mb:.1f} MB, dim {emb.dim})")
        out[name] = rows
    return out


def bench_sweep(name: str, clips: list[str]) -> dict:
    emb = Embedder(name)
    out = {}
    print(f"\nthreshold sweep, embedder = {name}")
    print(f"{'thr':>5s} {'DER@.25':>8s} {'DER@0':>7s} {'conf@.25':>9s} "
          f"{'spk err':>8s} {'d50':>6s}")
    for thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        rows = [score_clip(c, run_clip(c, emb, threshold=thr)) for c in clips]
        spk_err = sum(abs(r["hyp_speakers"] - r["ref_speakers"]) for r in rows)
        out[f"{thr:.2f}"] = rows
        print(f"{thr:5.2f} {100*statistics.mean(r['der_live_25'] for r in rows):7.2f}% "
              f"{100*statistics.mean(r['der_live_00'] for r in rows):6.2f}% "
              f"{100*statistics.mean(r['conf_25'] for r in rows):8.2f}% "
              f"{spk_err:8d} {statistics.mean(r['delay_p50'] for r in rows):6.2f}")
    return out


def bench_window(name: str, clips: list[str]) -> dict:
    """The latency/accuracy dial: how little speech is enough to name the speaker."""
    emb = Embedder(name)
    out = {}
    print(f"\nwindow sweep, embedder = {name}  (min_speech / window)")
    print(f"{'min_s':>6s} {'win_s':>6s} {'DER@.25':>8s} {'conf@.25':>9s} "
          f"{'d50':>6s} {'d90':>6s} {'CPU%':>6s}")
    for min_s, win_s in ((0.30, 1.5), (0.40, 1.5), (0.50, 1.5), (0.60, 1.5),
                         (0.80, 1.5), (0.60, 1.0), (0.60, 2.0), (0.60, 3.0)):
        rows = [score_clip(c, run_clip(c, emb, min_speech_s=min_s, window_s=win_s))
                for c in clips]
        out[f"{min_s:.2f}/{win_s:.1f}"] = rows
        print(f"{min_s:6.2f} {win_s:6.1f} "
              f"{100*statistics.mean(r['der_live_25'] for r in rows):7.2f}% "
              f"{100*statistics.mean(r['conf_25'] for r in rows):8.2f}% "
              f"{statistics.mean(r['delay_p50'] for r in rows):6.2f} "
              f"{statistics.mean(r['delay_p90'] for r in rows):6.2f} "
              f"{statistics.mean(r['cpu_pct'] for r in rows):5.1f}%")
    return out


def bench_me(names: list[str]) -> dict:
    """Can 10 s of the mic flag Me inside a mixed channel?"""
    enroll = read_wav(CLIPS / "me_enroll.wav")
    out = {}
    print("\nMe enrollment (10.4 s of the mic channel, then the far-side labeller)")
    print(f"{'embedder':10s} {'clip':12s} {'prec':>6s} {'rec':>6s} {'acc':>6s} "
          f"{'DER@.25':>8s} {'found':>6s}")
    for name in names:
        emb = Embedder(name)
        rows = []
        for clip in ("far2_me", "room3", "far2"):
            run = run_clip(clip, emb, enroll=enroll)
            ref = der.read_rttm(CLIPS / f"{clip}.rttm")
            hyp = [der.Segment(l.t0, l.t1, l.speaker)
                   for l in run["labels"] if not l.revises]
            acc = der.binary_accuracy(ref, hyp, "Me", duration=run["audio_s"])
            d = der.score(ref, hyp, collar=0.25, duration=run["audio_s"])
            found = sum(l.t1 - l.t0 for l in run["labels"]
                        if l.speaker == "Me" and not l.revises)
            rows.append({"clip": clip, **acc, "der": d["der"], "me_seconds": found})
            print(f"{name:10s} {clip:12s} {acc['precision']:6.3f} {acc['recall']:6.3f} "
                  f"{acc['accuracy']:6.3f} {100*d['der']:7.2f}% {found:5.1f}s")
        out[name] = rows
    return out


def bench_fluid(dirpath: Path, clips: list[str]) -> dict:
    """Score FluidAudio's own JSON output with the same DER, for a like-for-like row."""
    out = {}
    for mode in ("lseend", "sortformer", "process"):
        rows = []
        for clip in clips:
            p = dirpath / f"{mode}_{clip}.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            hyp = [der.Segment(s["startTimeSeconds"], s["endTimeSeconds"],
                               str(s.get("speaker") or s.get("speakerId")))
                   for s in data.get("segments", [])
                   if s["endTimeSeconds"] > s["startTimeSeconds"]]
            ref = der.read_rttm(CLIPS / f"{clip}.rttm")
            dur = data.get("durationSeconds") or max(s.t1 for s in ref)
            s25 = der.score(ref, hyp, collar=0.25, duration=dur)
            s00 = der.score(ref, hyp, collar=0.0, duration=dur)
            rows.append({"clip": clip, "der_live_25": s25["der"],
                         "der_live_00": s00["der"], "der_settled_25": s25["der"],
                         "conf_25": s25["conf"], "missed": s00["missed"],
                         "fa": s00["fa"], "conf": s00["conf"],
                         "ref_speakers": s25["ref_speakers"],
                         "hyp_speakers": s25["hyp_speakers"],
                         "delay_p50": float("nan"), "delay_p90": float("nan"),
                         "delay_max": float("nan"), "missed_turns": 0, "turns": 0,
                         "cpu_pct": float("nan"),
                         "rtf": 1.0 / data.get("rtfx", data.get("realTimeFactor", 1)),
                         "embed_ms": float("nan"), "mapping": s25["mapping"]})
        if rows:
            table(rows, f"FluidAudio {mode} (offline over the whole file)")
            out[mode] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="presage.diarize.bench")
    ap.add_argument("--models", default="resnet34")
    ap.add_argument("--clips", default=",".join(FAR_CLIPS))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--me", action="store_true")
    ap.add_argument("--fluid", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None, help="write the raw numbers as JSON")
    args = ap.parse_args()

    names = [n for n in args.models.split(",") if n and n in MODELS]
    clips = [c for c in args.clips.split(",") if c]
    results: dict = {}

    results["models"] = bench_models(names, clips)
    if args.sweep or args.all:
        results["sweep"] = bench_sweep(names[0], clips)
    if args.window or args.all:
        results["window"] = bench_window(names[0], clips)
    if args.me or args.all:
        results["me"] = bench_me(names)
    fluid = args.fluid or ("/tmp/fluidout" if args.all else None)
    if fluid and Path(fluid).is_dir():
        results["fluid"] = bench_fluid(Path(fluid), clips)

    if args.out:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items() if k != "labels"}
            if isinstance(o, list):
                return [clean(v) for v in o]
            if isinstance(o, float):
                return None if o != o else round(o, 6)
            return o
        Path(args.out).write_text(json.dumps(clean(results), indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
