"""Smoke test: feed a wav through a provider and print what comes out, when.

    python -m presage.stt.smoke file.wav
    python -m presage.stt.smoke file.wav --fast          # no real-time pacing
    python -m presage.stt.smoke file.wav --source system
    python -m presage.stt.smoke file.wav --provider deepgram
    python -m presage.stt.smoke file.wav --json          # machine-readable summary

The number that matters is `final +NNNms`: wall clock from the last audio sample being
fed to the final transcript landing in Python.

Output legend:
    part  mic#3 [Q?]       interim for utterance 3; Q? = a question is forming,
                           p=NNN is the turn detector's P(speaker stopped here)
      text ... tail        committed text, then the unstable tail the UI should dim
    FINAL mic#3 [EAGER]    finalised early because the turn detector said so;
                           it can still be retracted
    part  mic#3 [RETRACT]  that eager final was wrong, same utterance continues

Turn the three behaviours off to compare:
    PRESAGE_STT_STABLE_PREFIX=0 PRESAGE_STT_SEMANTIC_EOT=0 PRESAGE_STT_RETRACT=0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import wave

import numpy as np

from . import make_provider

CHUNK_MS = 20
SAMPLE_RATE = 16000
CHUNK = SAMPLE_RATE * CHUNK_MS // 1000  # 320 samples


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(
                f"{path}: need 16 kHz mono int16, got {w.getframerate()} Hz "
                f"{w.getnchannels()}ch {w.getsampwidth()*8}-bit.\n"
                f"Convert with: afconvert -f WAVE -d LEI16@16000 -c 1 in.wav {path}"
            )
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


async def main() -> None:
    ap = argparse.ArgumentParser(prog="presage.stt.smoke")
    ap.add_argument("wav")
    ap.add_argument("--provider", default=None, help="local (default) | deepgram")
    ap.add_argument("--source", default="mic", choices=["mic", "system"])
    ap.add_argument("--fast", action="store_true",
                    help="skip real-time pacing; finals only, since partial cadence "
                         "is wall-clock based and no wall clock passes")
    ap.add_argument("--tail", type=float, default=2.0,
                    help="seconds of silence to append so VAD closes the utterance")
    ap.add_argument("--json", action="store_true",
                    help="also print one JSON line, for the A/B harness")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pcm = read_wav(args.wav)
    tail = np.zeros(int(args.tail * SAMPLE_RATE), dtype=np.int16)
    audio = np.concatenate([pcm, tail])
    print(f"{args.wav}: {len(pcm)/SAMPLE_RATE:.2f}s audio "
          f"+ {args.tail:.1f}s silence, source={args.source}")

    t_load = time.perf_counter()
    provider = make_provider(args.provider)
    await provider.start()
    print(f"provider={provider.name} ready in {time.perf_counter()-t_load:.2f}s\n")

    state = {"audio_end": None, "first_partial": None, "n_partial": 0,
             "finals": [], "eager": 0, "retracts": 0, "qforming": 0}

    async def drain():
        async for tr in provider.transcripts():
            now = time.perf_counter()
            rel = (now - t0) * 1000
            utt = getattr(tr, "utt", 0)
            unstable = getattr(tr, "unstable", "")
            eager = getattr(tr, "eager", False)
            retracts = getattr(tr, "retracts", False)
            qform = getattr(tr, "question_forming", False)
            cp = getattr(tr, "complete_p", 0.0)
            tags = " ".join(c for c, on in (("EAGER", eager), ("RETRACT", retracts),
                                            ("Q?", qform)) if on)
            tags = f" [{tags}]" if tags else ""
            if tr.is_final:
                ae = state["audio_end"]
                lat = (now - ae) * 1000 if ae else None
                state["finals"].append({"utt": utt, "text": tr.text, "eager": eager,
                                        "at_ms": rel, "lat_ms": lat})
                state["eager"] += bool(eager)
                print(f"[{rel:7.0f} ms] FINAL {tr.source}#{utt}{tags} "
                      f"({tr.t0:.2f}-{tr.t1:.2f}s p={cp:.3f})"
                      + (f" +{lat:.0f}ms after last sample" if lat else "")
                      + f"\n           {tr.text}")
            else:
                state["n_partial"] += 1
                state["retracts"] += bool(retracts)
                state["qforming"] += bool(qform)
                if state["first_partial"] is None:
                    state["first_partial"] = now
                dim = f"  ··· {unstable}" if unstable else ""
                print(f"[{rel:7.0f} ms] part  {tr.source}#{utt}{tags} "
                      f"({tr.t0:.2f}-{tr.t1:.2f}s p={cp:.3f})"
                      f"\n           {tr.text}{dim}")

    t0 = time.perf_counter()
    reader = asyncio.create_task(drain())

    n_real = len(pcm)
    for i in range(0, len(audio), CHUNK):
        chunk = audio[i: i + CHUNK]
        if chunk.shape[0] < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - chunk.shape[0]))
        await provider.feed(args.source, chunk)
        if i < n_real <= i + CHUNK:
            state["audio_end"] = time.perf_counter()
        if not args.fast:
            target = t0 + (i + CHUNK) / SAMPLE_RATE
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)  # let partial tasks run

    # Long enough for the LONG-silence endpoint plus the decode. The mic source uses
    # a longer threshold (1400 ms) than the system source, so size for the mic.
    await asyncio.sleep(2.2)
    await provider.stop()
    await reader

    fp, ae = state["first_partial"], state["audio_end"]
    print(f"\n{state['n_partial']} partials"
          + (f", first at +{(fp-t0)*1000:.0f} ms" if fp else "")
          + (f", audio ended at +{(ae-t0)*1000:.0f} ms" if ae else ""))
    fin = state["finals"]
    print(f"{len(fin)} finals, {state['eager']} eager, {state['retracts']} retracted, "
          f"{state['qforming']}/{state['n_partial']} partials question-forming")
    lats = [f["lat_ms"] for f in fin if f["lat_ms"] is not None]
    if lats:
        print(f"last-sample -> final: {min(lats):.0f}..{max(lats):.0f} ms")
    stats = getattr(provider, "stats", None)
    if stats and stats.get("eot_calls"):
        print(f"turn detector: {stats['eot_calls']} calls, "
              f"{stats['eot_ms']/stats['eot_calls']:.1f} ms mean")
    if args.json:
        import json
        print("JSON " + json.dumps({"wav": args.wav, "finals": fin,
                                    "partials": state["n_partial"],
                                    "eager": state["eager"],
                                    "retracts": state["retracts"],
                                    "stats": stats}))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
