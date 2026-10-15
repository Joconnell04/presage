"""Turn a call script into two voices on one timeline.

    uv run python -m presage.sim.render sample-01-quarterly

macOS `say` speaks every turn (Them in one voice, Me in another) and `afconvert` brings it
to the only format the replay path accepts: 16 kHz mono int16 (`presage/audio_session.py`,
`_replay_pump`). Nothing is ever played out loud; both voices go to files.

Three files land in `docs/replay/sim/clips/`:

    <id>.them.wav      the client's lines, silence where the consultant speaks
    <id>.me.wav        the consultant's lines, silence where the client speaks
    <id>.timeline.json start and end second of every turn, which is what the scorer
                       measures a card against

Both wavs share one timeline and the run feeds both: `PRESAGE_REPLAY=` is the Them wav on the
system source and `PRESAGE_REPLAY_MIC=` is the Me wav on the mic. Each is silence where the
other speaks, which keeps every question at the second the timeline says and makes the traps
that depend on the consultant's own voice real ones. The Them wav is committed because it is
the one a plain `PRESAGE_REPLAY=` smoke test needs; the Me wav is rebuilt by this module.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from presage.sim import scripts as S

CLIP_DIR = S.ROOT / "docs" / "replay" / "sim" / "clips"
RATE = 16000
THEM_VOICE = "Samantha"     # en_US
ME_VOICE = "Daniel"         # en_GB, clearly a different speaker
WORDS_PER_MIN = 185
# Silence before the first word. Long enough for the server's model prewarm to finish, so the
# first question is not the only one measured against a cold provider (the README: the first
# call to a model after idle takes several seconds).
LEAD_IN_S = 6.0
TAIL_S = 2.0

_DASH = re.compile(r"\s*[‐-―]+\s*$")      # a half sentence ends on a dash; say chokes
_MID_DASH = re.compile(r"\s*[‐-―]+\s*")


def speakable(text: str) -> str:
    t = _DASH.sub("", str(text or "").strip())
    t = _MID_DASH.sub(", ", t)
    t = t.replace("%", " percent").replace("&", " and ").replace("$", "")
    return re.sub(r"\s+", " ", t).strip()


def voices() -> set[str]:
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=20).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def say_wav(text: str, voice: str, rate: int, tmp: Path) -> np.ndarray:
    """One spoken line as 16 kHz mono int16, through `say` then `afconvert`."""
    aiff, wav = tmp / "line.aiff", tmp / "line.wav"
    for p in (aiff, wav):
        p.unlink(missing_ok=True)
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
                   check=True, capture_output=True, timeout=120)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1",
                    str(aiff), str(wav)], check=True, capture_output=True, timeout=120)
    with wave.open(str(wav), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2), \
            f"afconvert gave {w.getframerate()} Hz, {w.getnchannels()} ch, {w.getsampwidth()}B"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.astype(np.int16).tobytes())


def render(script: dict, them_voice: str = THEM_VOICE, me_voice: str = ME_VOICE,
           rate: int = WORDS_PER_MIN, out_dir: Path = CLIP_DIR) -> dict:
    have = voices()
    for v in (them_voice, me_voice):
        if v not in have:
            raise SystemExit(f"voice {v!r} is not installed (say -v '?')")

    for v in {str(t.get("voice")) for t in script["turns"] if t.get("voice")}:
        if v not in have:                    # E65: a second client, on the same Them channel
            raise SystemExit(f"voice {v!r} is not installed (say -v '?')")

    them: list[np.ndarray] = [np.zeros(int(LEAD_IN_S * RATE), dtype=np.int16)]
    me: list[np.ndarray] = [np.zeros(int(LEAD_IN_S * RATE), dtype=np.int16)]
    timeline: list[dict] = []
    cursor = LEAD_IN_S
    tmp = Path(tempfile.mkdtemp(prefix="presage-sim-"))
    try:
        for t in script["turns"]:
            voice = str(t.get("voice") or "") or (them_voice if t["speaker"] == "them"
                                                  else me_voice)
            pcm = say_wav(speakable(t["text"]), voice, rate, tmp)
            gap = np.zeros(int(round(t["pause_after_s"] * RATE)), dtype=np.int16)
            silence = np.zeros(len(pcm), dtype=np.int16)
            if t["speaker"] == "them":
                them += [pcm, gap]
                me += [silence, gap]
            else:
                them += [silence, gap]
                me += [pcm, gap]
            dur = len(pcm) / RATE
            timeline.append({"i": t["i"], "speaker": t["speaker"],
                             "start_s": round(cursor, 3), "end_s": round(cursor + dur, 3),
                             "pause_after_s": t["pause_after_s"], "text": t["text"],
                             "voice": voice})
            cursor += dur + t["pause_after_s"]
            print(f"  {t['i']:2} {t['speaker']:5} {voice:9} {dur:5.2f}s  {t['text'][:52]}",
                  file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    tail = np.zeros(int(TAIL_S * RATE), dtype=np.int16)
    them_pcm, me_pcm = np.concatenate(them + [tail]), np.concatenate(me + [tail])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / f"{script['id']}.them.wav", them_pcm)
    write_wav(out_dir / f"{script['id']}.me.wav", me_pcm)

    meta = {"id": script["id"], "context": script["context"], "title": script.get("title", ""),
            "sample_rate": RATE, "lead_in_s": LEAD_IN_S, "tail_s": TAIL_S,
            "duration_s": round(len(them_pcm) / RATE, 3),
            "them_voice": them_voice, "me_voice": me_voice, "words_per_min": rate,
            "voices": sorted({t["voice"] for t in timeline}),
            "them_wav": f"{script['id']}.them.wav", "me_wav": f"{script['id']}.me.wav",
            "turns": timeline}
    (out_dir / f"{script['id']}.timeline.json").write_text(json.dumps(meta, indent=1),
                                                           encoding="utf-8")
    return meta


def load_timeline(script_id: str, out_dir: Path = CLIP_DIR) -> dict:
    return json.loads((out_dir / f"{script_id}.timeline.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", nargs="+", help="script id(s) under docs/replay/sim/scripts")
    ap.add_argument("--them-voice", default=THEM_VOICE)
    ap.add_argument("--me-voice", default=ME_VOICE)
    ap.add_argument("--rate", type=int, default=WORDS_PER_MIN)
    a = ap.parse_args()
    total = 0.0
    for sid in a.script:
        meta = render(S.load(sid), a.them_voice, a.me_voice, a.rate)
        total += meta["duration_s"]
        mb = (CLIP_DIR / meta["them_wav"]).stat().st_size / 1e6
        print(f"{sid}: {meta['duration_s']:.1f}s, them wav {mb:.1f} MB")
    print(f"total {total:.0f}s")


if __name__ == "__main__":
    main()
