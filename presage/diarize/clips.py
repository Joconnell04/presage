"""Synthetic diarization clips from macOS `say`, with exact RTTM ground truth.

    uv run python -m presage.diarize.clips              # writes docs/replay/diar/
    uv run python -m presage.diarize.clips --out /tmp/x --only far2

Every sentence is rendered on its own with `say -v VOICE --data-format=LEI16@16000`,
trimmed to its speech, then placed at a known offset. That is the whole trick: the
RTTM is not an annotation of a recording, it is the recipe the recording was built
from, so the ground truth is exact to the sample and DER measures the labeller rather
than an annotator.

Nothing is ever played. `say -o file` renders to disk and never touches an output
device.

Clips (16 kHz mono int16):

    far2.wav         2 far-side speakers, clean turn taking
    far3.wav         3 far-side speakers, clean turn taking
    far3_overlap.wav 3 far-side speakers, 3 overlapped regions (0.44 s of double talk)
    far2_me.wav      2 far-side speakers + Me leaking into the system channel
    room3.wav        one microphone in a room: Me + 2 others in a single channel
    me_enroll.wav    12 s of Me alone, different words, for enrollment
    real3.wav        3 REAL speakers, assembled the same way (see build_real)
    real2.wav        2 REAL speakers, the commonest call shape

Voices are the personas: Daniel is Me, the rest are the far side. Per-speaker gain
differs so the clip is not trivially separable by level.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs" / "replay" / "diar"

# Persona -> (macOS voice, linear gain). Gains stay under 1.0 so the mix never clips.
VOICES: dict[str, tuple[str, float]] = {
    "Me": ("Daniel", 0.95),
    "A": ("Samantha", 0.80),
    "B": ("Rishi", 0.68),
    "C": ("Karen", 0.88),
    "D": ("Moira", 0.74),
    "E": ("Fred", 0.80),
}

TRIM_DB = 40.0        # trim head/tail quieter than peak - 40 dB
TRIM_PAD_MS = 25      # keep this much of the quiet either side, so words are not clipped
GAP_SENTENCE_S = 0.18  # between two sentences by the same speaker inside one turn
GAP_TURN_S = 0.42      # between turns


# --------------------------------------------------------------------------- scripts

@dataclass(frozen=True)
class Turn:
    who: str
    text: str
    overlap: float = 0.0
    """Seconds this turn starts BEFORE the previous one ends. 0 = normal gap."""


def _t(who: str, text: str, overlap: float = 0.0) -> Turn:
    return Turn(who, text, overlap)


# Consulting-call flavour, to match what presage actually hears: figures, levers,
# headcount, margin. Sentences are short so the RTTM has tight, honest boundaries.
SCRIPTS: dict[str, list[Turn]] = {
    "far2": [
        _t("A", "Thanks for walking us through the model. I want to start with the cost side."),
        _t("B", "Same here. The headcount ramp looked steeper than what we signed off in March."),
        _t("A", "Right. Can you remind me what the fully loaded cost per head is in year two?"),
        _t("B", "And does that include the contractor pool, or is that a separate line?"),
        _t("A", "Because if it is separate we are understating the run rate by quite a bit."),
        _t("B", "My worry is the margin bridge. It moves eleven points and nothing explains it."),
        _t("A", "Let us come back to that. First, what discount rate did you use on the terminal value?"),
        _t("B", "Eight percent feels low for this market. I would have expected eleven or twelve."),
        _t("A", "If you flex it to twelve, what happens to the payback period?"),
        _t("B", "That is the number my board will ask about, so I would like it on a slide."),
        _t("A", "One more. Which of these savings are contractual and which are behavioural?"),
        _t("B", "Behavioural savings never land at a hundred percent, in my experience."),
    ],
    "far3": [
        _t("A", "Good morning everyone. Let us take the operating plan first."),
        _t("C", "Before we do, can I ask about the attrition assumption on page four?"),
        _t("D", "I had the same question. Fourteen percent seems optimistic for that team."),
        _t("A", "It is the three year average, isn't it? Or is it last year only?"),
        _t("C", "If it is last year only, that was an unusual year. We lost two managers."),
        _t("D", "What does the plan look like at eighteen percent instead?"),
        _t("A", "And would the hiring cost go up proportionally, or is there a step change?"),
        _t("C", "There is a step change once you go past forty open roles. You need another recruiter."),
        _t("D", "Which is not in this model as far as I can see."),
        _t("A", "Let us park that and look at the revenue side for a moment."),
        _t("C", "The upsell rate is the one I would push on. Where does nine percent come from?"),
        _t("D", "Is that a blended figure across both segments, or enterprise only?"),
        _t("A", "If it is blended, the enterprise number underneath it must be much higher."),
        _t("C", "Twelve or thirteen, probably. Which we have never achieved."),
    ],
    "far3_overlap": [
        _t("A", "So the second scenario is the one we would recommend to the committee."),
        _t("B", "Sorry, can I stop you there for a second?"),
        _t("C", "I had a question on that too.", overlap=0.55),
        _t("B", "Go ahead, you first."),
        _t("C", "Thanks. What changes between scenario one and scenario two, exactly?"),
        _t("A", "The capital spend and the timing of the second site."),
        _t("B", "And the working capital, surely. That was the big difference last time."),
        _t("C", "Yes, the working capital swing was nearly four million.", overlap=0.70),
        _t("A", "Let me pull up the bridge so we are all looking at the same thing."),
        _t("B", "While you do, what is the probability weighting on each scenario?"),
        _t("C", "That matters a lot for the expected value we take to the board."),
        _t("A", "Fifty five, thirty, fifteen. But those are judgement, not modelled."),
        _t("B", "Understood. I would still show them.", overlap=0.45),
        _t("C", "Agreed, show the weights and say they are judgement."),
        _t("A", "Fine. Last thing before we move on, the sensitivity grid."),
    ],
    "far2_me": [
        _t("A", "Can you take us through the working capital assumption again?"),
        _t("Me", "Of course. It is the same day sales outstanding we used in the last review."),
        _t("B", "Which was what, forty eight days?"),
        _t("Me", "Forty six, and it improves to forty one by the end of year two."),
        _t("A", "That improvement is doing a lot of work in the cash flow."),
        _t("B", "What drives it? Is it the collections team or the contract terms?"),
        _t("Me", "Mostly contract terms. Two of the three large accounts renew on better terms."),
        _t("A", "And if they do not renew on better terms?"),
        _t("B", "Then the whole cash profile shifts right by a couple of quarters."),
        _t("Me", "It does. I can show you that case, it is scenario three in the workbook."),
        _t("A", "Please do. And put the downside next to the base case if you can."),
        _t("B", "That is the version I would circulate to the committee."),
    ],
    "room3": [
        _t("Me", "Right, let us start. I will take you through the cost model first."),
        _t("A", "Before you do, is this the version you sent on Friday?"),
        _t("Me", "It is, with one change. The overhead allocation moved to the new basis."),
        _t("D", "Does that change the unit economics, or just where the cost sits?"),
        _t("Me", "Just where it sits. The total is the same, the split by product changed."),
        _t("A", "Show us the split, then. That is where the argument will be."),
        _t("D", "Particularly for the small format stores. They always look worse on allocation."),
        _t("Me", "They do, and on this basis they look better than the old one."),
        _t("A", "By how much? A point, or something that actually changes the decision?"),
        _t("Me", "About three points of contribution margin. It changes the decision."),
        _t("D", "Then I want to understand the basis properly before we go further."),
        _t("A", "Agreed. Walk us through it line by line."),
    ],
    "me_enroll": [
        _t("Me", "This is a short enrollment recording made before the call begins."),
        _t("Me", "It gives the system ten seconds of my voice with no one else speaking."),
        _t("Me", "The words do not matter, only the voice does."),
    ],
}

# Which clips are far-side only (no Me). Used by bench.py to pick the Me tests.
HAS_ME = {"far2_me", "room3", "me_enroll"}


# --------------------------------------------------------------------------- rendering

def _render(voice: str, text: str, tmp: Path) -> np.ndarray:
    """One sentence -> 16 kHz mono int16, trimmed to its speech. Never played."""
    out = tmp / "say.wav"
    subprocess.run(
        ["say", "-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000",
         "-o", str(out), text],
        check=True, capture_output=True,
    )
    with wave.open(str(out), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2), \
            f"say gave {w.getframerate()} Hz {w.getnchannels()}ch {w.getsampwidth()*8}-bit"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()
    out.unlink(missing_ok=True)
    return _trim(pcm)


def _trim(pcm: np.ndarray) -> np.ndarray:
    """Cut the silence `say` puts either side, so the RTTM onset is the real onset."""
    if pcm.size == 0:
        return pcm
    env = np.abs(pcm.astype(np.float32))
    # 10 ms moving max: robust to the zero crossings inside voiced speech.
    win = SAMPLE_RATE // 100
    if env.size > win:
        pad = np.pad(env, (0, (-env.size) % win))
        env = np.repeat(pad.reshape(-1, win).max(axis=1), win)[: pcm.size]
    peak = float(env.max())
    if peak <= 0:
        return pcm
    thresh = peak * (10.0 ** (-TRIM_DB / 20.0))
    loud = np.flatnonzero(env > thresh)
    if loud.size == 0:
        return pcm
    pad = TRIM_PAD_MS * SAMPLE_RATE // 1000
    a = max(0, int(loud[0]) - pad)
    b = min(pcm.size, int(loud[-1]) + pad)
    return pcm[a:b]


def build(name: str, turns: list[Turn], out_dir: Path) -> dict:
    """Render one clip + its RTTM. Returns a summary dict."""
    tmp = Path(tempfile.mkdtemp(prefix="presage-diar-"))
    try:
        pieces: list[tuple[str, int, np.ndarray]] = []   # (who, start_sample, pcm)
        cursor = int(0.30 * SAMPLE_RATE)                 # 300 ms of lead-in silence
        prev_who: str | None = None
        for turn in turns:
            voice, gain = VOICES[turn.who]
            # One sentence per RTTM segment: internal pauses never get called speech.
            sentences = [s.strip() for s in _split_sentences(turn.text) if s.strip()]
            for i, sentence in enumerate(sentences):
                pcm = _render(voice, sentence, tmp)
                pcm = (pcm.astype(np.float32) * gain).astype(np.int16)
                if i == 0:
                    gap = GAP_TURN_S if prev_who is not None else 0.0
                    start = cursor + int(gap * SAMPLE_RATE) - int(turn.overlap * SAMPLE_RATE)
                    start = max(start, int(0.30 * SAMPLE_RATE))
                else:
                    start = cursor + int(GAP_SENTENCE_S * SAMPLE_RATE)
                pieces.append((turn.who, start, pcm))
                cursor = max(cursor, start + pcm.size)
            prev_who = turn.who

        total = cursor + int(0.60 * SAMPLE_RATE)         # 600 ms of tail silence
        mix = np.zeros(total, dtype=np.int32)
        rttm: list[tuple[str, float, float]] = []
        for who, start, pcm in pieces:
            mix[start:start + pcm.size] += pcm.astype(np.int32)
            rttm.append((who, start / SAMPLE_RATE, pcm.size / SAMPLE_RATE))

        peak = int(np.abs(mix).max()) or 1
        if peak > 32000:                                  # only the overlap clip can get here
            mix = (mix.astype(np.float64) * (32000.0 / peak)).astype(np.int32)
        audio = np.clip(mix, -32768, 32767).astype(np.int16)

        wav_path = out_dir / f"{name}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.tobytes())

        rttm_path = out_dir / f"{name}.rttm"
        with rttm_path.open("w") as f:
            for who, t0, dur in sorted(rttm, key=lambda r: r[1]):
                f.write(f"SPEAKER {name} 1 {t0:.3f} {dur:.3f} <NA> <NA> {who} <NA> <NA>\n")

        speech = sum(d for _, _, d in rttm)
        overlap_s = _overlap_seconds(rttm)
        return {
            "clip": name,
            "seconds": round(total / SAMPLE_RATE, 2),
            "speech_seconds": round(speech, 2),
            "overlap_seconds": round(overlap_s, 2),
            "segments": len(rttm),
            "speakers": sorted({w for w, _, _ in rttm}),
            "voices": {w: VOICES[w][0] for w in sorted({w for w, _, _ in rttm})},
            "wav_kb": round(wav_path.stat().st_size / 1024),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- real voices

# Synthetic voices are a fair test of the machinery and a poor test of the models:
# `say` produces a far more consistent speaker than a person on a call does. This
# builds the same kind of clip out of real recordings of three people, taken from the
# sherpa-onnx speaker-recognition release, so the ranking can be checked against
# voices nobody synthesised. They speak Mandarin and the embedders are trained on
# English VoxCeleb, which makes it a harder test, not an easier one.

REAL_BASE = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models")
REAL_CACHE = Path.home() / ".cache" / "presage" / "diar" / "voices"
REAL_SCRIPTS: dict[str, list[tuple[str, str]]] = {
    "real3": [
        ("R1", "fangjun-sr-1.wav"), ("R2", "leijun-sr-1.wav"),
        ("R3", "liudehua-sr-1.wav"), ("R1", "fangjun-sr-2.wav"),
        ("R2", "leijun-sr-2.wav"), ("R3", "liudehua-sr-2.wav"),
        ("R1", "fangjun-sr-3.wav"), ("R2", "leijun-test-sr-1.wav"),
        ("R3", "liudehua-test-sr-1.wav"), ("R1", "fangjun-test-sr-1.wav"),
    ],
    # Two real speakers, the commonest call shape, in the other order.
    "real2": [
        ("R2", "leijun-sr-1.wav"), ("R3", "liudehua-sr-1.wav"),
        ("R2", "leijun-sr-2.wav"), ("R3", "liudehua-sr-2.wav"),
        ("R2", "leijun-test-sr-1.wav"), ("R3", "liudehua-test-sr-1.wav"),
        ("R2", "leijun-test-sr-2.wav"), ("R3", "liudehua-test-sr-2.wav"),
        ("R2", "leijun-test-sr-3.wav"),
    ],
}


def build_real(name: str, out_dir: Path) -> dict:
    """Same construction, real speakers: download, trim, place, write the RTTM."""
    import urllib.request

    REAL_CACHE.mkdir(parents=True, exist_ok=True)
    pieces: list[tuple[str, int, np.ndarray]] = []
    cursor = int(0.30 * SAMPLE_RATE)
    for who, fname in REAL_SCRIPTS[name]:
        path = REAL_CACHE / fname
        if not path.exists():
            urllib.request.urlretrieve(f"{REAL_BASE}/{fname}", path)  # noqa: S310
        with wave.open(str(path), "rb") as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2)
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()
        pcm = _trim(pcm)
        start = cursor + int(GAP_TURN_S * SAMPLE_RATE) if pieces else cursor
        pieces.append((who, start, pcm))
        cursor = start + pcm.size

    total = cursor + int(0.60 * SAMPLE_RATE)
    mix = np.zeros(total, dtype=np.int32)
    rttm = []
    for who, start, pcm in pieces:
        mix[start:start + pcm.size] += pcm.astype(np.int32)
        rttm.append((who, start / SAMPLE_RATE, pcm.size / SAMPLE_RATE))
    audio = np.clip(mix, -32768, 32767).astype(np.int16)

    wav_path = out_dir / f"{name}.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    with (out_dir / f"{name}.rttm").open("w") as f:
        for who, t0, dur in rttm:
            f.write(f"SPEAKER {name} 1 {t0:.3f} {dur:.3f} <NA> <NA> {who} <NA> <NA>\n")

    return {"clip": name, "seconds": round(total / SAMPLE_RATE, 2),
            "speech_seconds": round(sum(d for _, _, d in rttm), 2),
            "overlap_seconds": 0.0, "segments": len(rttm),
            "speakers": sorted({w for w, _, _ in rttm}),
            "voices": {w: f"{w} (real recording)"
                       for w in sorted({x for x, _, _ in rttm})},
            "wav_kb": round(wav_path.stat().st_size / 1024)}


def _split_sentences(text: str) -> list[str]:
    out, cur = [], []
    for ch in text:
        cur.append(ch)
        if ch in ".?!":
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _overlap_seconds(rttm: list[tuple[str, float, float]]) -> float:
    """Seconds where two or more speakers are active at once."""
    edges = sorted({t for _, t0, d in rttm for t in (t0, t0 + d)})
    total = 0.0
    for a, b in zip(edges, edges[1:]):
        mid = (a + b) / 2
        n = sum(1 for _, t0, d in rttm if t0 <= mid < t0 + d)
        if n > 1:
            total += b - a
    return total


def main() -> None:
    ap = argparse.ArgumentParser(prog="presage.diarize.clips")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", action="append", help="clip name; repeatable")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or [*SCRIPTS, *REAL_SCRIPTS]

    summary = []
    for name in names:
        if name in REAL_SCRIPTS:
            info = build_real(name, out_dir)
        elif name in SCRIPTS:
            info = build(name, SCRIPTS[name], out_dir)
        else:
            raise SystemExit(f"unknown clip {name!r}; have "
                             f"{', '.join([*SCRIPTS, *REAL_SCRIPTS])}")
        summary.append(info)
        print(f"{info['clip']:15s} {info['seconds']:6.1f}s  "
              f"speech {info['speech_seconds']:6.1f}s  "
              f"overlap {info['overlap_seconds']:5.2f}s  "
              f"{info['segments']:3d} seg  {','.join(info['speakers'])}  "
              f"{info['wav_kb']} kB")

    (out_dir / "clips.json").write_text(json.dumps(summary, indent=2) + "\n")
    total_kb = sum(i["wav_kb"] for i in summary)
    print(f"\n{len(summary)} clips, {total_kb/1024:.1f} MB of wav in {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
