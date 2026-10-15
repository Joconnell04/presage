"""E21a: the lever pick and the warm-card check, on the local multi checkpoint.

`presage/llm/intent.py` asks two typed questions that until now only hosted Jev could answer:

    pick_lever(decider, line, levers, prior=None)   which input of the model the line means
    card_answers(decider, card_q, question)         does the warm card answer what just landed

E21 trained one 421M checkpoint that answers both plus the gate's three, and E21a measured it
through these two functions on the lever table production builds today: **93.7% of E7's 300 spoken
lines to the right cell in 34 ms** (45 on torch MPS), where Jev gets 91.3% in 226, and
**100.0% precision / 95.6%
recall** on E7's 200 warm-card pairs in 28 ms, where Jev gets 93.0 / 88.9 in 168 and the shipped
0.45 cosine gets 56.3 / 84.4. This module is what puts those two answers on this machine: no round
trip, no key, no transcript leaving the Mac.

    dec = intent_local.make(choice="auto")         # None when the weights are not here
    pick = await intent.pick_lever(dec, line, index.levers())      # k=20, drawn by intent.py

Three things it is careful about.

**One load of the weights, never two.** The gate already holds a checkpoint, and a second 421M
torch load at startup once starved the transcriber for 40 s (jev.py's `shared_laya`). Everything
here goes through that same per-(directory, runtime) cache, so a process that gates on
`presage-gate-multi` and picks levers on `presage-gate-multi` holds one model, and the Core ML
package shares the torch entry it falls back to rather than building a second 421M copy.

**A shortlist, always.** At 251 options `laya.build_sequence` cuts each option to four tokens and
leaves nine for the line, and the checkpoint answers NONE to everything (14.9% over E21's 47
held-out lines, the same constant v2 gives). `intent.k_for` is what enforces that: a Laya-backed
decider is never handed more than `intent.LOCAL_SHORTLIST_K` levers, whatever the caller asked
for. The shortlist is not an optimisation here, it is the only shape the question fits in.

**A checkpoint directory you can point somewhere else.** `PRESAGE_INTENT_DIR` takes a path, which
is how a re-cut and retrained checkpoint (E21b, whose rows are cut from the table production
builds today rather than `table.json`) gets measured without a code change.

Environment:

    PRESAGE_INTENT=auto|local|jev|off    auto: local when the weights are here, else Jev with a
                                         key, else off. `local` and `jev` force it.
    PRESAGE_INTENT_DIR=<path>            the checkpoint, overriding the default search order
    PRESAGE_INTENT_COREML=0              never use the Core ML lever package, even when exported
    PRESAGE_INTENT_UNITS=<units>         ALL | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY for this
                                         package alone (E21c). The default is CPU_ONLY: the
                                         transcriber and the gate have the GPU, and an intent
                                         package beside them on it stalls the transcript (E41).
    PRESAGE_INTENT_FAST_DIR=<path>       the Core ML package directory, when it is not the one
                                         this checkpoint's name implies
    PRESAGE_INTENT_TORCH_DEVICE=cpu      the torch device, when the package is off
    PRESAGE_INTENT_PRELOAD=0             do not load at startup; the first lever pick pays for it

    uv run python -m presage.llm.intent_local                 # what is here and what it would use
    uv run python -m presage.llm.intent_local --selftest      # one lever pick, one card check
    uv run python -m presage.llm.intent_local --export        # the Core ML package (842 MB, 33 s)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("presage.llm.intent_local")

CACHE = Path.home() / ".cache" / "presage" / "laya"

# The checkpoints E21 trained, best first. The 421M is 94.1% on the lever and 98.8 / 97.6 on the
# card; the 322M student is 93.1% and 94.9 / 90.4 at half the latency, and is what to point
# `PRESAGE_INTENT_DIR` at on a machine that cannot hold the gate and the big one together.
# The 322M student first: resident beside the gate's package it keeps client transcript finals at
# 0.4 s where the 421M costs 3.2 s (docs/experiments/E21c-runtime.md); PRESAGE_INTENT_DIR overrides.
CANDIDATES = ("presage-gate-multi-small", "presage-gate-multi")

# The Core ML export of the intent shapes: batch 1, 21 option markers, one 448-token bucket. A
# separate package from the gate's, which is traced at batch 3 and 5 markers and whose shapes are
# fixed, so wiring both costs two packages on disk.
#
# 448 rather than E21's 384 because that is what covers every shape the shipped local path sends,
# measured with this checkpoint's own tokenizer over E7's sets: a lever question with no prior is
# 296 to 324 tokens, one carrying a prior is 347 to 392, and the card noul is 158. At 384 nine of
# the 100 follow-ups would fall out of the bucket and onto torch, which is a second copy of the
# weights for 9% of the traffic. Batch 1 is enough because the follow-up question does not go to
# a local checkpoint (`intent.asks_follow`), so every local request carries exactly one question.
FAST_DIR = CACHE / "fast-intent"
FAST_BUCKET = 448
FAST_BATCH = 1
FAST_K = 21                       # LOCAL_SHORTLIST_K levers + the NONE option

# The Core ML compute units for this package alone, separate from the gate's. E21a exported and
# benched it on CPU_AND_GPU, which is `FastLayaDecider`'s default and right for the gate; E21c is
# measuring what it costs the transcriber to have a second package on that GPU.
FAST_UNITS = "CPU_AND_GPU"
UNITS = ("ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY")

MODES = ("auto", "local", "jev", "off")


# ----------------------------------------------------------------------------- what is here
def weights_dir() -> Path | None:
    """The intent checkpoint to use, or None when none of them is on this machine.

    `PRESAGE_INTENT_DIR` wins and is returned even when it is incomplete, so a typo shows up as a
    "no weights in ..." message rather than as a silent fall back to a different checkpoint.
    """
    env = os.environ.get("PRESAGE_INTENT_DIR")
    if env:
        return Path(env).expanduser()
    from presage.llm.laya import weights_present
    for name in CANDIDATES:
        if weights_present(CACHE / name):
            return CACHE / name
    return None


def fast_dir(model_dir: str | Path | None = None) -> Path:
    """Where this checkpoint's batch-1 package lives.

    One directory per checkpoint, because a package carries the encoder weights it was traced
    from and `coreml_available` refuses one traced elsewhere: the 322M student under the 421M's
    package name would silently fall back to torch. The 421M keeps E21a's `fast-intent` so an
    already-exported package is still found.
    """
    env = os.environ.get("PRESAGE_INTENT_FAST_DIR")
    if env:
        return Path(env).expanduser()
    d = Path(model_dir).expanduser() if model_dir else weights_dir()
    if d is None or d.name == CANDIDATES[0]:
        return FAST_DIR
    return CACHE / f"fast-intent-{d.name}"


def units() -> str:
    """The Core ML compute units for the intent package: `PRESAGE_INTENT_UNITS`, else FAST_UNITS."""
    want = (os.environ.get("PRESAGE_INTENT_UNITS") or FAST_UNITS).strip().upper()
    if want not in UNITS:
        log.info("unknown PRESAGE_INTENT_UNITS %r, using %s", want, FAST_UNITS)
        return FAST_UNITS
    return want


def torch_device() -> str | None:
    """The torch device for the intent checkpoint when the package is off, None for laya's own
    order (MPS, then CUDA, then CPU). `PRESAGE_INTENT_TORCH_DEVICE=cpu` keeps it off the GPU."""
    return (os.environ.get("PRESAGE_INTENT_TORCH_DEVICE") or "").strip() or None


def preloads() -> bool:
    """False with `PRESAGE_INTENT_PRELOAD=0`: nothing is loaded at startup and the first lever
    pick of the call pays the load, which is the opposite trade from E17's bucket prewarm."""
    return os.environ.get("PRESAGE_INTENT_PRELOAD", "1").lower() not in ("0", "false", "off")


def why_not() -> str | None:
    """The one-line reason the local intent decider cannot run. None when it can."""
    from presage.llm.laya import FILES, missing_packages, weights_present
    gone = missing_packages()
    if gone:
        return f"{', '.join(gone)} not installed: uv sync --extra laya"
    d = weights_dir()
    if d is None:
        return (f"no intent checkpoint in {CACHE}: want one of {', '.join(CANDIDATES)} "
                f"(files: {', '.join(FILES)})")
    if not weights_present(d):
        return f"no weights in {d}"
    return None


def available() -> bool:
    """True when the lever pick and the warm-card check can run locally right now."""
    return why_not() is None


def coreml_available(packages: Path | None = None, model_dir: str | Path | None = None) -> bool:
    """True when the batch-1 package is exported, matches the shape, and was traced from this
    checkpoint. The last clause matters: a package carries the encoder weights and only the
    tokenizer comes from the directory, so the 421M's package under the 322M's name would answer
    with the 421M (`jev.shared_laya` refuses it there too, and falls back to torch)."""
    if os.environ.get("PRESAGE_INTENT_COREML", "1").lower() in ("0", "false", "off"):
        return False
    try:
        from presage.llm.laya_fast.coreml import CoreMLGate
        d = Path(model_dir).expanduser() if model_dir else weights_dir()
        g = CoreMLGate(packages or fast_dir(d))
        traced = str(g.meta.get("model_dir") or "")
        return bool(g.available() and g.meta.get("batch") == FAST_BATCH
                    and int(g.meta.get("k") or 0) >= FAST_K
                    and (not traced or not d or Path(traced).expanduser() == d))
    except Exception as e:  # noqa: BLE001  an optional runtime that will not import is not here
        log.debug("intent Core ML package unavailable: %s", e)
        return False


def mode(choice: str | None = None) -> str:
    """Which decider the intent questions should go to: `local`, `jev` or `off`.

    `PRESAGE_INTENT` wins over the argument, which is the settings value, so a run started with
    the environment variable behaves the same whatever is saved in settings.json.
    """
    want = (os.environ.get("PRESAGE_INTENT") or choice or "auto").strip().lower()
    if want not in MODES:
        log.info("unknown PRESAGE_INTENT/intent %r, using auto", want)
        want = "auto"
    if want != "auto":
        return want
    # E41 (docs/experiments/E41-stack.md): with the 421M intent package resident beside the
    # gate's on this 16 GB Mac, client transcripts settled 6 to 7 s after the audio ended (0.6 to
    # 0.8 s in E17), tier-2 verdicts took 4 to 7 s and the lever pick 2 to 4 s in a live call.
    #
    # E21c (docs/experiments/E21c-runtime.md) found the runtime that does not: the 322M student
    # on its own package, which measured 407 ms client finals, 602 ms tier 2 and 86 / 16 / 115 ms
    # picks where the 421M measured 3,201 ms and stalled eight turns. It is still not the default,
    # because the flip wanted two runs and the student's second landed on a Mac at 11.4 GB of swap
    # and a load of 12.9, with no control beside it. Auto does not pick local until that pair is
    # re-run; `intent: local` or PRESAGE_INTENT=local asks for it, and with
    # PRESAGE_INTENT_DIR=~/.cache/presage/laya/presage-gate-multi-small it asks for the one that
    # measured well.
    if available() and os.environ.get("PRESAGE_INTENT_LOCAL_AUTO", "").lower() in ("1", "true", "on"):
        return "local"
    from presage.llm.jev import jev_key
    return "jev" if jev_key() else "off"


# ----------------------------------------------------------------------------- the decider
def local_decider(model_dir: str | Path | None = None, packages: Path | None = None,
                  preload: bool = True) -> Any:
    """The shared local decider for the intent checkpoint. Raises `DecisionError` when it cannot run.

    Core ML when a package for *this* checkpoint is exported, torch MPS otherwise. Both come out
    of `jev.shared_laya`'s per-(directory, runtime) cache, so nothing here is ever a second copy
    of weights the process already holds; the Core ML decider's fallback for a shape the fixed
    package cannot take is that cache's torch entry rather than a fresh 421M load.
    """
    from presage.llm.jev import DecisionError, shared_laya
    d = Path(model_dir).expanduser() if model_dir else weights_dir()
    if d is None:
        raise DecisionError(why_not() or "no intent checkpoint")
    dev = torch_device()
    if coreml_available(packages, d):
        fast = shared_laya(d, packages=packages or fast_dir(d), units=units(), preload=preload)
        if getattr(fast, "_slow", "missing") is None:
            fast._slow = shared_laya(d, runtime="torch", device=dev, preload=False)
        return fast
    return shared_laya(d, runtime="torch", device=dev, preload=preload)


def make(choice: str | None = None, jev: Any = None, preload: bool = True) -> Any:
    """The decider `intent.pick_lever` and `intent.card_answers` should use, or None for neither.

    `choice` is the `intent` setting (`auto`, `local`, `jev`, `off`); `PRESAGE_INTENT` overrides
    it. `jev` is a hosted client the caller already has, reused rather than rebuilt so one warm
    HTTP/2 connection serves the gate, the slots and this.
    """
    want = mode(choice)
    if want == "off":
        return None
    if want == "local":
        try:
            return local_decider(preload=preload and preloads())
        except Exception as e:  # noqa: BLE001  a missing local checkpoint falls back, never raises
            log.info("local intent unavailable (%s); trying Jev", e)
    if jev is not None:
        return jev
    from presage.llm.jev import JevDecider, jev_key
    return JevDecider() if jev_key() else None


def describe(dec: Any) -> str:
    """One word for the log and the status block: `local`, `jev` or `off`."""
    if dec is None:
        return "off"
    return "local" if str(getattr(dec, "kind", "")) == "laya" else str(getattr(dec, "provider", "") or "jev")


def status(choice: str | None = None) -> dict:
    """What a Settings page would print: the mode, the checkpoint and why not, if not."""
    d = weights_dir()
    fast = coreml_available(None, d)
    return {"mode": mode(choice), "choice": (choice or os.environ.get("PRESAGE_INTENT") or "auto"),
            "dir": str(d) if d else None, "available": available(),
            "coreml": fast, "packages": str(fast_dir(d)),
            "runtime": (f"coreml/{units()}" if fast else f"torch/{torch_device() or 'auto'}"),
            "preload": preloads(), "why_not": why_not()}


# ----------------------------------------------------------------------------- the Core ML export
def export(model_dir: str | Path | None = None, out: Path | None = None) -> list[Path]:
    """Trace the intent shapes into a Core ML package: batch 1, 21 markers, one 448-token bucket.

    E21 measured 28.4 ms p50 against torch MPS's 42.4 for this question, at the cost of a second
    842 MB package (the gate's is traced at batch 3 and 5 markers and the shapes are fixed). 33 s
    to build. The ANE refuses this graph as it refuses v2's, so it runs on CPU_AND_GPU.
    """
    from presage.llm.laya_fast.coreml import export as _export
    d = Path(model_dir).expanduser() if model_dir else weights_dir()
    if d is None:
        raise SystemExit(why_not() or "no intent checkpoint")
    return _export(d, [FAST_BUCKET], Path(out or fast_dir(d)), batch=FAST_BATCH, k=FAST_K)


# ----------------------------------------------------------------------------- CLI
async def _selftest(model_dir: str | None = None) -> None:  # pragma: no cover
    import json
    import time

    from presage.llm import intent as I
    levers = json.loads((Path(__file__).resolve().parents[2]
                         / "data/gate/whatif/table.e20.json").read_text())
    dec = local_decider(model_dir)
    t0 = time.perf_counter()
    pick = await I.pick_lever(dec, "what if the platform fee were one percent", levers)
    print(f"lever   {pick.cell}  p={pick.p:.3f}  {len(pick.sent or [])} options  "
          f"{(time.perf_counter() - t0) * 1e3:.0f} ms")
    t0 = time.perf_counter()
    v = await I.card_answers(dec, "What is the platform fee?", "what's the platform fee again?")
    print(f"card    {v.answers}  p={v.p:.3f}  {(time.perf_counter() - t0) * 1e3:.0f} ms")
    v = await I.card_answers(dec, "What is the platform fee?", "how many subscribers do we have in FY27?")
    print(f"card    {v.answers}  p={v.p:.3f}   (expected False)")


def main() -> None:  # pragma: no cover
    import argparse
    import asyncio
    import json

    ap = argparse.ArgumentParser(prog="python -m presage.llm.intent_local",
                                 description="the local typed-intent decider (E21a)")
    ap.add_argument("--selftest", action="store_true", help="one lever pick and two card checks")
    ap.add_argument("--export", action="store_true", help="the Core ML lever package (842 MB, 33 s)")
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.export:
        for pkg in export(a.model_dir):
            size = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1e6
            print(f"{pkg.name:26s} {size:7.0f} MB")
        return
    if a.selftest:
        asyncio.run(_selftest(a.model_dir))
        return
    print(json.dumps(status(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
