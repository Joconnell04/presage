"""E9c smoke test: the five wiring paths `data/spoken/bench_brain.py` cannot reach.

    uv run python -m presage.spoken_smoke

The bench measures the *rates* -- which figures the guard catches over 263 lines. This checks
the plumbing around it, which a rate cannot see:

1. a cold index is silent and starts exactly one background build, off the transcript thread;
2. the same cell twice inside the window raises one card, not two;
3. and after `SPOKEN_COOLDOWN_S` the second one lands;
4. `_spoken_slow` (the time-budget fallback) still cards, through the worker thread, and a
   check that reports over budget flips it;
5. an index the guard cannot be built for costs one `guard_for` call over four lines, not four.

No model call and no network: the Brain is built with an LLM stub that raises on anything a
provider would need. The the big model folder is copied to a temp directory and opened from the copy.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIG = ROOT / "docs" / "replay" / "sim" / "big"
LINE = "So the startup cost is about nine million all in."     # contradicts $14.85M in B5

os.environ.setdefault("PRESAGE_DECIDER", "off")     # no gate weights, no Jev, no key


class _NoLLM:
    provider = "none"
    answer_model = hedge_model = classifier_model = ""

    def state(self) -> dict:
        return {"provider": "none"}

    def __getattr__(self, name: str):
        raise AssertionError(f"the spoken smoke test reached the model: {name}")


def _brain(index, cards: list):
    from presage.brain import Brain

    async def emit(msg: dict) -> None:
        if msg.get("type") == "answer":
            cards.append(msg)

    b = Brain(index, emit, llm=_NoLLM())
    b._maybe_prefetch = lambda *a, **k: None        # the speculative path is not on the Me line
    return b


async def _say(brain, text: str = LINE) -> None:
    from presage.brain import Utterance, new_id
    await brain.on_transcript(Utterance(id=new_id("t"), source="mic", text=text,
                                        final=True, ts=time.time()))


def _check(ok: bool, label: str, got: object, want: object, fails: list) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}: {got} (expected {want})")
    if not ok:
        fails.append(label)


async def main_async() -> int:
    if not BIG.exists():
        print(f"skipped: {BIG} is not here (a large workbook folder with an ANSWER_KEY.json)")
        return 0
    from presage import spoken
    from presage.ingest import Index

    fails: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="presage-e9c-smoke-"))
    try:
        shutil.copytree(BIG, tmp / "big")
        cards: list = []
        b = _brain(Index(tmp / "big"), cards)

        print("1. a cold index")
        await _say(b)
        _check(not cards and b._spoken_busy, "silent, one build started",
               f"cards={len(cards)} building={b._spoken_busy}", "cards=0 building=True", fails)
        for _ in range(60):
            await asyncio.sleep(0.25)
            if not b._spoken_busy:
                break
        _check(getattr(b.index, "_spoken_guard", None) is not None, "the guard is up",
               getattr(b.index, "_spoken_guard", None) is not None, True, fails)

        print("2. the same wrong figure three times in a row")
        await _say(b)
        await _say(b)
        await _say(b)
        _check(len(cards) == 1 and b.stats["corrections"] == 1, "one card",
               f"cards={len(cards)} corrections={b.stats['corrections']}",
               "cards=1 corrections=1", fails)

        print("3. and again after the cooldown")
        b._spoken_at = {k: t - spoken_cooldown() - 1 for k, t in b._spoken_at.items()}
        await _say(b)
        _check(len(cards) == 2, "a second card", len(cards), 2, fails)

        print("4. the time budget")
        b._spoken_slow, b._spoken_at = True, {}
        t0 = time.perf_counter()
        await _say(b)
        _check(len(cards) == 3, "off the transcript thread, still cards",
               f"cards={len(cards)} in {(time.perf_counter() - t0) * 1000:.1f} ms", "cards=3", fails)
        b2 = _brain(Index(tmp / "big"), [])
        await b2._refresh_vocab()
        g = b2.index._spoken_guard
        real = g.check

        def over_budget(line, **kw):
            sp = real(line, **kw)
            sp.ms = 99.0
            return sp

        g.check = over_budget
        await _say(b2)
        _check(b2._spoken_slow, "one slow check flips it", b2._spoken_slow, True, fails)

        print("5. an index with nothing to argue with")
        builds: list[int] = []
        real_guard_for = spoken.guard_for

        def counted(index, vocab=None):
            builds.append(1)
            return real_guard_for(index, vocab)

        spoken.guard_for = counted
        try:
            class Bare:
                version = 0

            bare_cards: list = []
            b3 = _brain(Bare(), bare_cards)
            b3._topic_hits = lambda t: []
            for _ in range(4):
                await _say(b3)
                await asyncio.sleep(0.4)
        finally:
            spoken.guard_for = real_guard_for
        _check(len(builds) == 1 and not bare_cards, "asked once over four lines",
               f"builds={len(builds)} cards={len(bare_cards)}", "builds=1 cards=0", fails)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nDONE" if not fails else "\nFAILED: " + ", ".join(fails))
    return 1 if fails else 0


def spoken_cooldown() -> float:
    from presage.brain import SPOKEN_COOLDOWN_S
    return SPOKEN_COOLDOWN_S


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
