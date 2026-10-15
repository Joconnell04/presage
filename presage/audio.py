"""Two labeled audio streams from macOS into Python.

Spawns the Swift helper at presage/bin/audiotap, which captures the default
microphone and all system audio (via a Core Audio process tap, no virtual audio
driver) and writes both as 16 kHz mono int16 PCM, framed, on stdout:

    [1 byte stream id][4 bytes little-endian payload length][int16 LE samples]

stream id 0 = "mic", 1 = "system". One frame is 20 ms (320 samples, 640 bytes).
The helper's status and errors arrive as JSON lines on stderr.

    async for source, samples in frames():
        ...  # source is "mic" or "system", samples is an int16 numpy array

list_devices() returns every Core Audio device (uid, name, in, out, rate, transport,
default_input, default_output), and frames(mic_uid=...) captures a specific input
device instead of the default input. A Bluetooth mic can go digitally silent with no
device or rate change (AirPods do it whenever the buds are out of your ears); the
helper notices after 3 s and falls back to the built-in mic, reporting each step as a
"mic_state" event with state ok / silent / recovering / fallback.

Smoke tests:

    python -m presage.audio --seconds 5
    python -m presage.audio --list-devices
    python -m presage.audio --seconds 16 --mic-uid BuiltInMicrophoneDevice
    python -m presage.audio --seconds 12 --debug-mute-mic-after 3   # fake a dead mic
"""

import argparse
import asyncio
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
BIN = REPO / "bin" / "audiotap"
BUILD = REPO / "native" / "audiotap" / "build.sh"

RATE = 16000
FRAME_SAMPLES = 320          # 20 ms
SOURCES = {0: "mic", 1: "system"}


def ensure_built():
    """Build the helper if it is missing or older than its source."""
    src = REPO / "native" / "audiotap" / "main.swift"
    if BIN.exists() and BIN.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.run([str(BUILD)], check=True)


def list_devices() -> list[dict]:
    """Every Core Audio device, as dicts with uid, name, in, out, rate, transport,
    default_input, default_output. transport is builtin / bluetooth / usb / virtual /
    aggregate / other; a "bluetooth" input is the one worth warning about, because it
    runs at 24 kHz and can go digitally silent (see docs/research-audio.md)."""
    ensure_built()
    done = subprocess.run([str(BIN), "--list-devices"], capture_output=True, text=True, timeout=15)
    devices = []
    for line in done.stderr.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.pop("event", None) == "device":
            devices.append(event)
    return devices


async def frames(mic=True, system=True, seconds=0, on_event=None, mic_uid=None,
                 debug_mute_mic_after=0):
    """Yield (source, samples) forever. source is "mic" or "system".

    mic_uid picks the input device by UID (from list_devices()) instead of the default
    input. debug_mute_mic_after is a test hook: it zeroes the mic samples after N
    seconds so the silent -> recovering -> fallback -> ok path can be exercised."""
    ensure_built()
    argv = [str(BIN)]
    if not system:
        argv.append("--mic-only")
    if not mic:
        argv.append("--system-only")
    if seconds:
        argv += ["--seconds", str(seconds)]
    if mic_uid:
        argv += ["--input-uid", mic_uid]
    if debug_mute_mic_after:
        argv += ["--debug-mute-mic-after", str(debug_mute_mic_after)]

    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    async def drain_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            try:
                event = json.loads(line)
            except ValueError:
                event = {"event": "raw", "line": line.decode(errors="replace").rstrip()}
            if on_event:
                on_event(event)

    stderr_task = asyncio.ensure_future(drain_stderr())
    try:
        while True:
            try:
                header = await proc.stdout.readexactly(5)
            except asyncio.IncompleteReadError:
                return
            source = SOURCES.get(header[0], "unknown")
            length = int.from_bytes(header[1:5], "little")
            try:
                payload = await proc.stdout.readexactly(length)
            except asyncio.IncompleteReadError:
                return
            yield source, np.frombuffer(payload, dtype="<i2")
    finally:
        stderr_task.cancel()
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.kill()


def _print_event(event, t):
    """mic_state is the one worth reading at a glance: it says which device the mic is
    on and whether presage had to move it."""
    if event.get("event") == "mic_state":
        extra = f"  from={event['from']}" if "from" in event else ""
        print(f"t={t:4.1f}s  MIC {event['state'].upper():10} {event.get('device', '')}"
              f"  [{event.get('uid', '')}]{extra}", flush=True)
    print("event:", json.dumps(event), flush=True)


async def _smoke(seconds, mic_uid=None, debug_mute_mic_after=0):
    events = []
    totals = {"mic": [0, 0.0], "system": [0, 0.0]}   # frames, sum of squares
    window = {"mic": [0, 0.0], "system": [0, 0.0]}
    loop = asyncio.get_event_loop()
    start = loop.time()
    next_tick = start + 1.0

    async for source, samples in frames(seconds=seconds, on_event=events.append,
                                        mic_uid=mic_uid,
                                        debug_mute_mic_after=debug_mute_mic_after):
        while events:
            _print_event(events.pop(0), loop.time() - start)
        f = samples.astype(np.float64)
        for bucket in (totals, window):
            bucket[source][0] += len(samples)
            bucket[source][1] += float(np.sum(f * f))
        now = loop.time()
        if now >= next_tick:
            parts = []
            for name in ("mic", "system"):
                n, ss = window[name]
                rms = math.sqrt(ss / n) if n else 0.0
                dbfs = 20 * math.log10(rms / 32768) if rms > 0 else float("-inf")
                parts.append(f"{name}: {n:6d} samples  rms={rms:8.1f}  {dbfs:6.1f} dBFS")
                window[name] = [0, 0.0]
            print(f"t={now - start:4.1f}s  " + " | ".join(parts), flush=True)
            next_tick += 1.0

    while events:
        _print_event(events.pop(0), loop.time() - start)
    print("---")
    for name in ("mic", "system"):
        n, ss = totals[name]
        rms = math.sqrt(ss / n) if n else 0.0
        print(f"{name}: {n} samples ({n / RATE:.2f} s of audio), overall rms={rms:.1f}")
    print("(the helper's own 'silent' events above say why a stream is quiet:")
    print(' "no_callbacks" means the device is idle, "zero_samples" means a missing permission')
    print(' or a Bluetooth mic that has stopped sending; mic_state says what was done about it)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--list-devices", action="store_true",
                        help="print the Core Audio devices and exit")
    parser.add_argument("--mic-uid", default=None,
                        help="capture this input device instead of the default input")
    parser.add_argument("--debug-mute-mic-after", type=float, default=0,
                        help="test hook: zero the mic samples after N seconds")
    args = parser.parse_args()
    if args.list_devices:
        for d in list_devices():
            flags = "".join([" default_input" if d["default_input"] else "",
                             " default_output" if d["default_output"] else ""])
            print(f"{d['name'][:32]:34} {d['transport']:10} in={d['in']} out={d['out']} "
                  f"{int(d['rate']):6} Hz  {d['uid']}{flags}")
        return
    try:
        asyncio.run(_smoke(args.seconds, mic_uid=args.mic_uid,
                           debug_mute_mic_after=args.debug_mute_mic_after))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
