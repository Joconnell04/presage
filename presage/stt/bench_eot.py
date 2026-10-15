"""Endpointing benchmark. Generates its own speech, runs the provider, prints a table.

    python -m presage.stt.bench_eot                    # full set, current defaults
    python -m presage.stt.bench_eot --kind hesitation  # one category
    python -m presage.stt.bench_eot --sweep            # grid over EAGER/LONG/threshold
    python -m presage.stt.bench_eot --quiet-noise      # the attenuated/noisy variants

Clips are built from fragments with measured silence between them, so the script knows
exactly where every question really ends and can tell a legitimate eager final from one
that fired mid-question. Audio is cached under PRESAGE_BENCH_DIR
(default ~/.cache/presage/bench) and only regenerated when missing.

Nothing is ever played: `say -o` writes a file, `afconvert` converts it.

What the columns mean:
    t_final   ms from the last speech sample of a question to the final that completes it
    eager     finals committed on the short silence
    long      finals that waited out the long silence
    falseE    eager finals that ended mid-question AND were never completed later
    retr      retracts emitted
    split     finals minus the number of real questions (positive = over-segmented)
    WER       words lost or invented across all finals, against the script
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000
CHUNK = SR * 20 // 1000
BENCH_DIR = Path(os.environ.get(
    "PRESAGE_BENCH_DIR", Path.home() / ".cache" / "presage" / "bench"))


# --------------------------------------------------------------------------- clips

@dataclass
class Clip:
    cid: str
    kind: str
    voice: str
    # (text, silence_ms_after). Fragments are rendered separately, so a fragment with
    # no terminal punctuation gets continuation intonation -- which is what a real
    # hesitation sounds like.
    parts: list[tuple[str, int]]
    # Index of the LAST fragment of each real question. Everything else is a pause
    # inside a question, and a final that ends there is a false eager.
    ends: list[int]
    rate: int = 180
    gain_db: float = 0.0
    noise_dbfs: float | None = None

    @property
    def questions(self) -> list[str]:
        out, start = [], 0
        for e in self.ends:
            out.append(" ".join(p[0] for p in self.parts[start:e + 1]))
            start = e + 1
        return out


def _c(cid, kind, voice, parts, ends, **kw) -> Clip:
    return Clip(cid, kind, voice, parts, ends, **kw)


CLIPS: list[Clip] = [
    # ---- clean questions: the target is < 400 ms time-to-final
    _c("clean1", "clean", "Samantha",
       [("So what did revenue look like in Q3 versus plan?", 0)], [0]),
    _c("clean2", "clean", "Daniel",
       [("Walk me through the EBITDA bridge.", 0)], [0]),
    _c("clean3", "clean", "Karen",
       [("What's the churn on the enterprise cohort?", 0)], [0]),
    _c("clean4", "clean", "Rishi",
       [("What did we assume for ARR growth next year?", 0)], [0]),

    # ---- hesitations: pauses INSIDE one question. Any final before the last
    #      fragment is a false eager unless retract recovers it.
    _c("hes1", "hesitation", "Samantha",
       [("So, um", 520), ("what did Q3", 620),
        ("actually look like versus plan?", 0)], [2]),
    _c("hes2", "hesitation", "Daniel",
       [("And the headcount plan", 380),
        ("for next year", 300), ("is that still forty heads?", 0)], [2]),
    _c("hes3", "hesitation", "Karen",
       [("What's the churn", 700), ("on the enterprise cohort?", 0)], [1]),
    _c("hes4", "hesitation", "Rishi",
       [("Gross margin by region", 450), ("is that in the deck?", 0)], [1]),
    _c("hes5", "hesitation", "Samantha",
       [("How many heads are in the headcount plan", 350),
        ("for next year?", 0)], [1]),

    # ---- multi-part: two real questions, one breath. Splitting is CORRECT here.
    _c("multi1", "multipart", "Daniel",
       [("Is churn getting worse in enterprise?", 260),
        ("What's the number?", 0)], [0, 1]),
    _c("multi2", "multipart", "Karen",
       [("Did we hit plan in Q3?", 300),
        ("And what was the variance?", 0)], [0, 1]),

    # ---- statement that ends like a question, and a tag question
    _c("stmt1", "statement", "Samantha",
       [("That's the number we had, right.", 0)], [0]),
    _c("stmt2", "statement", "Daniel",
       [("So revenue came in at forty two million.", 0)], [0]),
    _c("stmt3", "statement", "Rishi",
       [("We assumed thirty percent ARR growth, I think.", 0)], [0]),

    # ---- pleasantry + question in one breath: one utterance in the script,
    #      but splitting it is arguably right, so it is scored separately.
    _c("pleas1", "pleasantry", "Karen",
       [("Thanks for walking us through the deck.", 240),
        ("So on the revenue side, what did Q3 look like versus plan?", 0)], [0, 1]),
    _c("pleas2", "pleasantry", "Samantha",
       [("Great, that's helpful.", 220),
        ("What's the churn on the enterprise cohort?", 0)], [0, 1]),

    # ---- monologue with sentence pauses: three real sentences, the last is the
    #      question. Tests that we do not wait 15 s for a final.
    _c("mono1", "monologue", "Daniel",
       [("So the way we've been thinking about this is fairly simple.", 340),
        ("Revenue is tracking behind plan but margin is ahead.", 320),
        ("What I want to know is whether churn explains the gap.", 0)],
       [0, 1, 2]),
    _c("mono2", "monologue", "Rishi",
       [("The enterprise cohort has been the weak spot all year.", 300),
        ("We lost four logos in Q3 alone.", 300),
        ("Is that in the deck anywhere?", 0)], [0, 1, 2]),
]

# Quiet and noisy variants of the clean set, for the VAD threshold question.
for _src in ("clean1", "clean3"):
    _base = next(c for c in CLIPS if c.cid == _src)
    CLIPS.append(Clip(f"{_src}_quiet", "quiet", _base.voice, _base.parts,
                      _base.ends, gain_db=-18.0))
    CLIPS.append(Clip(f"{_src}_noisy", "noisy", _base.voice, _base.parts,
                      _base.ends, noise_dbfs=-40.0))
    CLIPS.append(Clip(f"{_src}_quietnoisy", "quietnoisy", _base.voice, _base.parts,
                      _base.ends, gain_db=-18.0, noise_dbfs=-40.0))


# ----------------------------------------------------------------------- rendering

def _say(text: str, voice: str, rate: int, out: Path) -> np.ndarray:
    aiff = out.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
                   check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(aiff), str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    aiff.unlink(missing_ok=True)
    with wave.open(str(out), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    out.unlink(missing_ok=True)
    return pcm


def render(clip: Clip, d: Path) -> dict:
    """Build the wav and return its manifest: where every question really ends."""
    wav = d / f"{clip.cid}.wav"
    man = d / f"{clip.cid}.json"
    if wav.exists() and man.exists():
        return json.loads(man.read_text())

    d.mkdir(parents=True, exist_ok=True)
    lead = np.zeros(int(0.25 * SR), dtype=np.int16)
    pieces: list[np.ndarray] = [lead]
    part_end_s: list[float] = []
    n = len(lead)
    for i, (text, gap_ms) in enumerate(clip.parts):
        pcm = _say(text, clip.voice, clip.rate, d / f"_tmp_{clip.cid}_{i}.wav")
        pieces.append(pcm)
        n += len(pcm)
        part_end_s.append(n / SR)          # last speech sample of this fragment
        if gap_ms:
            g = np.zeros(int(gap_ms / 1000 * SR), dtype=np.int16)
            pieces.append(g)
            n += len(g)
    pieces.append(np.zeros(int(2.5 * SR), dtype=np.int16))
    audio = np.concatenate(pieces).astype(np.float32)

    if clip.gain_db:
        audio *= 10 ** (clip.gain_db / 20)
    if clip.noise_dbfs is not None:
        rng = np.random.default_rng(7)
        amp = 32768 * 10 ** (clip.noise_dbfs / 20)
        audio = audio + rng.normal(0, amp, audio.shape).astype(np.float32)
    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.tobytes())

    info = {
        "cid": clip.cid, "kind": clip.kind, "voice": clip.voice,
        "part_end_s": part_end_s,
        "question_end_s": [part_end_s[e] for e in clip.ends],
        "questions": clip.questions,
        "peak": int(np.abs(audio).max()),
        "dur_s": round(len(audio) / SR, 2),
    }
    man.write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------------ scoring

_W = re.compile(r"[\w'\-]+")
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
         70: "seventy", 80: "eighty", 90: "ninety"}


def _spell(n: int) -> str:
    """Digits to words, so 'Q3' and 'Q three' are the same thing to the scorer."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = n // 10 * 10, n % 10
        return _TENS[t] + (" " + _ONES[r] if r else "")
    for div, name in ((1_000_000, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= div:
            head, rest = divmod(n, div)
            return f"{_spell(head)} {name}" + (f" {_spell(rest)}" if rest else "")
    return str(n)


def words(s: str) -> list[str]:
    """Normalise to spoken form. Parakeet writes '42 million' and 'Q3'; the script
    says 'forty two million' and 'Q three'. Those are not transcription errors, and
    counting them as such buries the ones that are (ARR -> 'air' still counts)."""
    s = s.lower().replace("%", " percent ").replace("$", " dollars ")
    s = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", s)   # q3 -> q 3
    s = re.sub(r"\d+", lambda m: " " + _spell(int(m.group())) + " ", s)
    return [w for w in _W.findall(s)]


def wer(ref: list[str], hyp: list[str]) -> float:
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[len(ref), len(hyp)] / max(1, len(ref))


def score(info: dict, events: list[dict]) -> dict:
    all_finals = [e for e in events if e["final"]]
    # A final that a later retract on the same utt superseded is not part of the
    # output: it was explicitly withdrawn. Counting it double-counts its words and
    # makes a working retract look like a WER regression.
    retract_at = {}
    for e in events:
        if e["retracts"]:
            retract_at.setdefault(e["utt"], []).append(e["at_ms"])
    finals = [f for f in all_finals
              if not any(t > f["at_ms"] for t in retract_at.get(f["utt"], []))]
    n_superseded = len(all_finals) - len(finals)
    q_words = [words(q) for q in info["questions"]]

    # A final COMPLETES question k if its text ends with question k's last 3 words.
    completed: dict[int, dict] = {}
    for f in finals:
        fw = words(f["text"])
        for k, qw in enumerate(q_words):
            tail = qw[-3:] if len(qw) >= 3 else qw
            if len(fw) >= len(tail) and fw[-len(tail):] == tail and k not in completed:
                completed[k] = f

    # A false eager is an eager final whose words are a STRICT prefix of a question
    # (it stopped mid-question) and that is not itself a completion.
    false_eager, recovered = [], 0
    for f in finals:
        if not f["eager"] or f in completed.values():
            continue
        fw = words(f["text"])
        for k, qw in enumerate(q_words):
            if len(fw) < len(qw) and qw[:len(fw)] == fw:
                false_eager.append({"text": f["text"], "q": k})
                if k in completed and completed[k]["at_ms"] > f["at_ms"]:
                    recovered += 1
                break

    lat = []
    for k, f in sorted(completed.items()):
        lat.append(f["at_ms"] - info["question_end_s"][k] * 1000)

    all_final_words = [w for f in finals for w in words(f["text"])]
    ref_words = [w for qw in q_words for w in qw]

    return {
        "n_q": len(q_words), "n_final": len(finals), "n_superseded": n_superseded,
        "n_eager": sum(1 for f in finals if f["eager"]),
        "n_long": sum(1 for f in finals if not f["eager"]),
        "n_retract": sum(1 for e in events if e["retracts"]),
        "n_partial": sum(1 for e in events if not e["final"]),
        "completed": len(completed),
        "false_eager": len(false_eager),
        "false_eager_recovered": recovered,
        "false_eager_unrecovered": len(false_eager) - recovered,
        "false_eager_texts": [f["text"] for f in false_eager],
        "split": len(finals) - len(q_words),
        "lat_ms": [round(x) for x in lat],
        "wer": round(wer(ref_words, all_final_words), 3),
        "texts": [f["text"] for f in finals],
        "n_qforming": sum(1 for e in events
                          if not e["final"] and e.get("question_forming")),
    }


# -------------------------------------------------------------------------- runner

async def run_clip(clip: Clip, info: dict, d: Path, source: str) -> dict:
    from . import make_provider
    with wave.open(str(d / f"{clip.cid}.wav"), "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    p = make_provider("local")
    await p.start()
    events: list[dict] = []

    async def drain():
        async for tr in p.transcripts():
            events.append({
                "at_ms": (time.perf_counter() - t0) * 1000,
                "utt": tr.utt, "final": tr.is_final, "eager": tr.eager,
                "retracts": tr.retracts, "text": tr.text, "unstable": tr.unstable,
                "question_forming": tr.question_forming,
                "complete_p": tr.complete_p, "t0": tr.t0, "t1": tr.t1,
            })

    t0 = time.perf_counter()
    reader = asyncio.create_task(drain())
    late = 0.0
    for i in range(0, len(audio), CHUNK):
        c = audio[i:i + CHUNK]
        if c.shape[0] < CHUNK:
            c = np.pad(c, (0, CHUNK - c.shape[0]))
        await p.feed(source, c)
        delay = t0 + (i + CHUNK) / SR - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            late = min(late, delay)
    await asyncio.sleep(2.4)
    await p.stop()
    await reader

    out = score(info, events)
    out.update(cid=clip.cid, kind=clip.kind, events=events,
               stats=dict(getattr(p, "stats", {})), feed_late_s=round(-late, 3))
    release(p)
    return out


def release(p) -> None:
    """Drop the 2.4 GB model between clips.

    Without this the harness holds one Parakeet per clip until the garbage collector
    gets round to it, and on a 16 GB machine that means swapping. Symptom when it was
    missing: one clip in eighteen came back with a single final at 16 s and no
    partials at all -- the event loop had been stalled for eight seconds. Every
    latency number in a sweep is worthless if this is not done.
    """
    import gc
    p._model = None
    p._get_logmel = None
    p._eot = None
    p._sources.clear()
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'clip':16s} {'kind':11s} {'q':>2s} {'fin':>3s} {'eag':>3s} {'lng':>3s} "
           f"{'falseE':>6s} {'rec':>3s} {'retr':>4s} {'spl':>3s} {'WER':>5s}  t_final ms")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lat = ",".join(str(x) for x in r["lat_ms"]) or "-"
        bad = "!" if r["false_eager_unrecovered"] else " "
        print(f"{r['cid']:16s} {r['kind']:11s} {r['n_q']:2d} {r['n_final']:3d} "
              f"{r['n_eager']:3d} {r['n_long']:3d} {r['false_eager']:6d}{bad}"
              f"{r['false_eager_recovered']:3d} {r['n_retract']:4d} "
              f"{r['split']:+3d} {r['wer']:5.2f}  {lat}")

    def agg(kinds):
        sel = [r for r in rows if r["kind"] in kinds]
        lats = [x for r in sel for x in r["lat_ms"]]
        return sel, lats

    print()
    for label, kinds in [("clean", {"clean"}), ("hesitation", {"hesitation"}),
                         ("multipart+pleasantry", {"multipart", "pleasantry"}),
                         ("statement", {"statement"}), ("monologue", {"monologue"}),
                         ("quiet/noisy", {"quiet", "noisy", "quietnoisy"})]:
        sel, lats = agg(kinds)
        if not sel:
            continue
        print(f"  {label:22s} n={len(lats):2d} "
              f"mean t_final={np.mean(lats):6.0f} ms  "
              f"falseE={sum(r['false_eager'] for r in sel)} "
              f"(unrecovered {sum(r['false_eager_unrecovered'] for r in sel)})  "
              f"retr={sum(r['n_retract'] for r in sel)}  "
              f"split={sum(r['split'] for r in sel):+d}  "
              f"WER={np.mean([r['wer'] for r in sel]):.3f}")
    all_lats = [x for r in rows for x in r["lat_ms"]]
    tot_unrec = sum(r["false_eager_unrecovered"] for r in rows)
    print(f"\n  ALL                    n={len(all_lats):2d} "
          f"mean t_final={np.mean(all_lats):6.0f} ms  "
          f"unrecovered false eager={tot_unrec}  "
          f"completed={sum(r['completed'] for r in rows)}/"
          f"{sum(r['n_q'] for r in rows)}")


async def main() -> None:
    ap = argparse.ArgumentParser(prog="presage.stt.bench_eot")
    ap.add_argument("--dir", default=str(BENCH_DIR))
    ap.add_argument("--kind", default=None, help="only this category")
    ap.add_argument("--clip", default=None, help="only this clip id")
    ap.add_argument("--source", default="system", choices=["mic", "system"])
    ap.add_argument("--quiet-noise", action="store_true",
                    help="only the attenuated / noisy variants")
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir)
    clips = CLIPS
    if args.quiet_noise:
        clips = [c for c in clips if c.kind in ("quiet", "noisy", "quietnoisy")]
    elif not args.kind and not args.clip:
        clips = [c for c in clips if c.kind not in ("quiet", "noisy", "quietnoisy")]
    if args.kind:
        clips = [c for c in clips if c.kind == args.kind]
    if args.clip:
        clips = [c for c in clips if c.cid == args.clip]

    infos = {}
    for c in clips:
        infos[c.cid] = render(c, d)
    print(f"{len(clips)} clips in {d}", flush=True)
    if args.render_only:
        return

    rows = []
    for c in clips:
        rows.append(await run_clip(c, infos[c.cid], d, args.source))
        await asyncio.sleep(0.6)   # let the GPU and the page cache settle
        r = rows[-1]
        warn = f"  SLOW feed late by {r['feed_late_s']:.2f}s" if r["feed_late_s"] > 0.3 else ""
        print(f"  ran {r['cid']:16s} finals={r['n_final']} eager={r['n_eager']} "
              f"falseE={r['false_eager']}/{r['false_eager_recovered']}rec "
              f"retr={r['n_retract']} lat={r['lat_ms']}{warn}", flush=True)
    print()
    print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
