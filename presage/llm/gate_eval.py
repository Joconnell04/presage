"""Score a question gate against the labelled set in `data/gate/`.

    uv run python -m presage.llm.gate_eval --decider jev  --split dev
    uv run python -m presage.llm.gate_eval --decider haiku --split dev --limit 100 --jsonl out.jsonl
    uv run python -m presage.llm.gate_eval --decider laya --split test

Any `Decider` can be scored: Jev over the wire, the Haiku classifier, or the local Laya gate once
it exists. Every one of them is asked the question the brain asks, in the words the brain uses,
against the state string the brain builds, so a number here is a number about the live gate and
not about a paraphrase of it.

The state and the three questions below are a **verbatim copy** of `Brain._gate_decider`
(presage/brain.py). They are a copy rather than an import because nothing in presage/ changes for
this eval; `--parity` (on by default) keeps the copy honest by building a Brain, handing it a
decider that captures whatever it is asked, and comparing byte for byte. If brain.py ever moves,
the run says so on the first line instead of quietly scoring the wrong prompt.

Reported per decider:

  kind            accuracy over question / what-if / request / objection / none, macro F1, and
                  the confusion matrix. Also the 4-way accuracy the brain acts on, where what-if
                  collapses into question the way `_gate_decider` collapses it.
  needs_lookup    accuracy, precision, recall. Recall is the one that costs a missed answer.
  complete        accuracy, precision, recall, where the positive class is a finished thought.
  calibration     expected calibration error over 10 equal-width bins on the confidence the
                  decider reports for the class it picked, per question.
  latency         mean / p50 / p95 of the round trip, sequential by default because that is how
                  the brain calls it.
  gate verdict    would the brain have answered this line, and should it have. The end-to-end
                  number: mode kinds, the what-if rule, needs_files, and question_threshold.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from presage.llm.jev import Q as DQ, Decision

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "gate"
KINDS = ["question", "what-if", "request", "objection", "none"]
BINS = 10
QUESTION_THRESHOLD = 0.5        # brain DEFAULT_SETTINGS["question_threshold"]


# ----------------------------------------------------------------------------- the brain's ask
# Verbatim from Brain._gate_decider. Do not reword: --parity compares these against the live
# strings and the whole point of the eval is that the gate is asked what it is asked on a call.

def gate_state(lines: str, mode_label: str, who: str = "Them") -> str:
    return ("A strategy consultant (Me) is on a call with a client (Them). Recent transcript, oldest first; "
            f"judge ONLY the last line. Mode: {mode_label}. The last line is spoken by {who}: a line from Me "
            "(the consultant) is never a question for the copilot, so answer kind=none and needs_lookup=false "
            f"for it unless the mode is Interview.\n\n{lines}")


def speaker_of(text: str) -> str:
    return "Me" if str(text or "").lstrip().startswith("Me:") else "Them"


def gate_questions() -> list[DQ]:
    return [
        DQ("kind", "choice", "What is the last line? question = asks for information; what-if = asks what a "
           "number would become under a different assumption (\"what if we charged 1%\", \"at 400 subscribers "
           "instead\"); request = asks Me to show, walk through or send something; objection = pushes back on "
           "a number, price, claim or plan, raises a risk or competitor, or hesitates; none = small talk, "
           "statements, agreement, filler.",
           options=["question", "what-if", "request", "objection", "none"]),
        DQ("needs_lookup", "noul", "Answering the last line well needs a looked-up number, date, name or fact "
           "(from the consultant's decks, models, documents or public sources), or, in Sales/Pitch/Interview "
           "mode, a suggested response. False for small talk, pleasantries and things answered from the head."),
        DQ("complete", "noul", "The last line is a complete thought (not cut off mid-sentence)."),
    ]


def state_for(ex: dict, mode_label: str) -> str:
    """The transcript the brain would hold: its last 4 lines, oldest first, the judged line last."""
    lines = "\n".join([*ex.get("context", []), ex["text"]][-4:])
    return gate_state(lines, mode_label, speaker_of(ex["text"]))


class _Captured(Exception):
    def __init__(self, state: str, questions: list[DQ]) -> None:
        super().__init__("captured")
        self.state, self.questions = state, questions


async def parity() -> str | None:
    """Ask a real Brain to gate one line and compare what it asks with the copy above.

    Returns None when they match, or the first difference. Never touches the network: the decider
    it is handed raises before any request is made.
    """
    from presage import brain as B

    class _Capture:
        async def decide(self, state: str, questions: list) -> dict:
            raise _Captured(state, questions)

        async def validate_key(self) -> tuple[bool, str]:
            return True, "ok"

    async def noop(_msg: dict) -> None:
        return None

    brain = B.Brain(object(), noop, settings={"mode": "sales"})
    brain.decider = _Capture()
    lines = ["Me: that is the fee model", "Them: and what would 1% do to year five"]
    brain.transcript = [
        B.Utterance(id=f"t{i}", source=("mic" if ln.startswith("Me:") else "system"),
                    text=ln.split(":", 1)[1].strip(), final=True, ts=float(i))
        for i, ln in enumerate(lines)
    ]
    u = brain.transcript[-1]
    try:
        await brain._gate_decider(u)
    except _Captured as c:
        mine_state = gate_state("\n".join(lines), B.MODES["sales"].label)
        if c.state != mine_state:
            return f"state differs:\n  brain: {c.state!r}\n  copy : {mine_state!r}"
        mine_q = gate_questions()
        if len(c.questions) != len(mine_q):
            return f"{len(c.questions)} questions in brain, {len(mine_q)} in the copy"
        for a, b in zip(c.questions, mine_q):
            if (a.name, a.kind, a.instructions, a.options) != (b.name, b.kind, b.instructions, b.options):
                return f"question {a.name!r} differs:\n  brain: {a}\n  copy : {b}"
        return None
    except Exception as e:  # noqa: BLE001
        return f"could not build a Brain to compare with ({type(e).__name__}: {str(e)[:120]})"
    return "the brain never called its decider"


# ----------------------------------------------------------------------------- deciders

def make(name: str, timeout: float | None = None) -> Any:
    name = name.strip().lower()
    if name == "jev":
        from presage.llm.jev import JevDecider
        return JevDecider(timeout=timeout) if timeout else JevDecider()
    if name in ("haiku", "llm"):
        from presage.llm import make_llm
        from presage.llm.jev import HaikuDecider
        # PRESAGE_GATE_PROVIDER=claude_code asks the same model over the Claude subscription
        # instead of the API. Same prompt, same schema, same decider; the latency it reports is a
        # CLI round trip and is not comparable with the API's.
        provider = os.environ.get("PRESAGE_GATE_PROVIDER", "anthropic")
        return HaikuDecider(make_llm(provider), os.environ.get("PRESAGE_GATE_MODEL") or None)
    if name == "laya":
        try:
            from presage.llm.laya import LayaDecider     # not written yet; lazy on purpose
        except Exception as e:  # noqa: BLE001
            print(f"laya is not available here: {type(e).__name__}: {e}")
            print("presage/llm/laya.py has not landed yet. Run --decider jev or --decider haiku.")
            raise SystemExit(2)
        return LayaDecider()
    raise SystemExit(f"unknown decider {name!r}; expected jev, haiku or laya")


# ----------------------------------------------------------------------------- metrics

def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def macro_f1(pairs: list[tuple[str, str]], labels: list[str]) -> tuple[float, dict]:
    per = {}
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        per[lab] = {"p": prf(tp, fp, fn)[0], "r": prf(tp, fp, fn)[1], "f1": prf(tp, fp, fn)[2],
                    "n": sum(1 for g, _ in pairs if g == lab)}
    present = [lab for lab in labels if per[lab]["n"]]
    return (sum(per[lab]["f1"] for lab in present) / len(present) if present else 0.0), per


def ece(rows: list[tuple[bool, float]], bins: int = BINS) -> float:
    """Expected calibration error: |accuracy - confidence| per bin, weighted by bin size."""
    if not rows:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        chunk = [r for r in rows if (lo <= r[1] < hi) or (b == bins - 1 and r[1] == 1.0)]
        if not chunk:
            continue
        acc = sum(1 for ok, _ in chunk if ok) / len(chunk)
        conf = sum(c for _, c in chunk) / len(chunk)
        total += len(chunk) / len(rows) * abs(acc - conf)
    return total


def binary(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    tn = sum(1 for g, p in pairs if not g and not p)
    p, r, f = prf(tp, fp, fn)
    n = len(pairs) or 1
    return {"acc": (tp + tn) / n, "p": p, "r": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ----------------------------------------------------------------------------- run

def would_answer(kind: str, needs: bool, conf: float, mode_id: str) -> bool:
    """The brain's own verdict, from `_gate_decider` and `_on_final`."""
    from presage.brain import MODES
    mode = MODES.get(mode_id, MODES["call"])
    if kind == "what-if":
        kind, needs = "question", True
    wanted = needs or (mode.model_knowledge and kind in ("objection", "request"))
    return bool(kind in mode.kinds and wanted and conf >= QUESTION_THRESHOLD)


async def score_one(dec: Any, ex: dict) -> dict:
    from presage.brain import MODES
    mode = MODES.get(ex["mode"], MODES["call"])
    state = state_for(ex, mode.label)
    t0 = time.time()
    out: dict[str, Decision] = await dec.decide(state, gate_questions())
    ms = (time.time() - t0) * 1000
    kind = str(out["kind"].value)
    needs_p = float(out["needs_lookup"].p)
    comp_p = float(out["complete"].p)
    needs = bool(out["needs_lookup"].value)
    comp = bool(out["complete"].value)
    conf = min(float(out["kind"].p), needs_p)          # brain's confidence, verbatim
    return {
        "id": ex.get("id"), "mode": ex["mode"], "text": ex["text"], "category": ex.get("category"),
        "source": ex.get("source"),
        "gold": {"kind": ex["kind"], "needs_lookup": ex["needs_lookup"], "complete": ex["complete"]},
        "pred": {"kind": kind, "needs_lookup": needs, "complete": comp},
        "conf": {"kind": float(out["kind"].p), "needs_lookup": max(needs_p, 1 - needs_p),
                 "complete": max(comp_p, 1 - comp_p), "gate": conf},
        "answer": would_answer(kind, needs, conf, ex["mode"]),
        "gold_answer": would_answer(ex["kind"], ex["needs_lookup"], 1.0, ex["mode"]) and ex["complete"],
        "latency_ms": ms if not out["kind"].latency_ms else float(out["kind"].latency_ms),
    }


async def run(dec: Any, rows: list[dict], concurrency: int) -> tuple[list[dict], int]:
    got: list[dict] = []
    errors = 0
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(ex: dict) -> None:
        nonlocal errors, done
        async with sem:
            try:
                got.append(await score_one(dec, ex))
            except Exception as e:  # noqa: BLE001  a gate failure is a result, not a crash
                errors += 1
                if errors <= 3:
                    print(f"  error on {ex.get('id')}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(rows)}", flush=True)

    await asyncio.gather(*(one(ex) for ex in rows))
    return got, errors


# ----------------------------------------------------------------------------- report

def report(name: str, split: str, res: list[dict], errors: int, secs: float) -> dict:
    kind_pairs = [(r["gold"]["kind"], r["pred"]["kind"]) for r in res]
    kind_acc = sum(1 for g, p in kind_pairs if g == p) / (len(res) or 1)
    f1, per = macro_f1(kind_pairs, KINDS)
    coll = [("question" if g == "what-if" else g, "question" if p == "what-if" else p)
            for g, p in kind_pairs]
    coll_acc = sum(1 for g, p in coll if g == p) / (len(res) or 1)
    needs = binary([(r["gold"]["needs_lookup"], r["pred"]["needs_lookup"]) for r in res])
    comp = binary([(r["gold"]["complete"], r["pred"]["complete"]) for r in res])
    gate = binary([(r["gold_answer"], r["answer"]) for r in res])
    e_kind = ece([(r["gold"]["kind"] == r["pred"]["kind"], r["conf"]["kind"]) for r in res])
    e_needs = ece([(r["gold"]["needs_lookup"] == r["pred"]["needs_lookup"], r["conf"]["needs_lookup"]) for r in res])
    e_comp = ece([(r["gold"]["complete"] == r["pred"]["complete"], r["conf"]["complete"]) for r in res])
    lat = sorted(r["latency_ms"] for r in res) or [0.0]

    def pct(x: float) -> str:
        return f"{100 * x:5.1f}%"

    print(f"\n{name} on {split}: {len(res)} scored, {errors} errors, {secs:.0f}s wall")
    print(f"  kind          acc {pct(kind_acc)}   macro F1 {f1:.3f}   "
          f"(brain's 4-way, what-if folded into question: {pct(coll_acc)})")
    for lab in KINDS:
        d = per[lab]
        print(f"     {lab:<10} n={d['n']:<5} P {pct(d['p'])}  R {pct(d['r'])}  F1 {d['f1']:.3f}")
    print(f"  needs_lookup  acc {pct(needs['acc'])}   P {pct(needs['p'])}  R {pct(needs['r'])}  "
          f"(fn {needs['fn']} missed, fp {needs['fp']} spurious)")
    print(f"  complete      acc {pct(comp['acc'])}   P {pct(comp['p'])}  R {pct(comp['r'])}  "
          f"(fn {comp['fn']}, fp {comp['fp']})")
    print(f"  ECE (10 bins) kind {e_kind:.3f}   needs_lookup {e_needs:.3f}   complete {e_comp:.3f}")
    print(f"  latency       mean {statistics.mean(lat):6.0f} ms   p50 {lat[len(lat) // 2]:6.0f} ms   "
          f"p95 {lat[min(len(lat) - 1, int(0.95 * len(lat)))]:6.0f} ms")
    print(f"  gate verdict  acc {pct(gate['acc'])}   P {pct(gate['p'])}  R {pct(gate['r'])}  "
          f"({gate['fn']} answers missed, {gate['fp']} spurious)")

    print("\n  confusion (gold down, predicted across)")
    print("            " + "".join(f"{k[:9]:>10}" for k in KINDS))
    for g in KINDS:
        row = [sum(1 for a, b in kind_pairs if a == g and b == p) for p in KINDS]
        print(f"  {g:<10}" + "".join(f"{v:>10}" for v in row))

    worst: dict[str, list] = {}
    for r in res:
        if r["gold"]["kind"] != r["pred"]["kind"]:
            worst.setdefault(f"{r['gold']['kind']} -> {r['pred']['kind']}", []).append(r["text"])
    if worst:
        print("\n  most common kind confusions")
        for k, v in sorted(worst.items(), key=lambda kv: -len(kv[1]))[:5]:
            print(f"    {len(v):3d}  {k:<26} e.g. {v[0][:70]}")

    return {"decider": name, "split": split, "n": len(res), "errors": errors, "wall_s": round(secs, 1),
            "kind_acc": kind_acc, "kind_macro_f1": f1, "kind_acc_collapsed": coll_acc,
            "kind_per_label": per, "needs_lookup": needs, "complete": comp, "gate_verdict": gate,
            "ece": {"kind": e_kind, "needs_lookup": e_needs, "complete": e_comp},
            "latency_ms": {"mean": statistics.mean(lat), "p50": lat[len(lat) // 2],
                           "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))]},
            "confusion": {g: {p: sum(1 for a, b in kind_pairs if a == g and b == p) for p in KINDS}
                          for g in KINDS}}


# ----------------------------------------------------------------------------- main

async def main() -> int:
    ap = argparse.ArgumentParser(description="score a question gate against data/gate")
    ap.add_argument("--decider", default="jev", help="jev | haiku | laya")
    ap.add_argument("--split", default="dev", help="train | dev | test, or a path to a jsonl file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="1 keeps latency honest; raise it when only accuracy matters")
    ap.add_argument("--timeout", type=float, default=0.0, help="per-request timeout for jev")
    ap.add_argument("--jsonl", help="write per-example predictions here")
    ap.add_argument("--summary", help="write the summary numbers here as json")
    ap.add_argument("--no-parity", action="store_true", help="skip the check against brain.py")
    a = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    if not a.no_parity:
        why = await parity()
        print("parity with brain._gate_decider: " + ("ok" if why is None else f"MISMATCH\n  {why}"))
        if why is not None:
            print("  the copy in this file no longer matches the live gate; fix it before trusting numbers")
            return 2

    path = Path(a.split) if a.split.endswith(".jsonl") else DATA / f"{a.split}.jsonl"
    if not path.exists():
        print(f"no split at {path}; build it with data/gate/assemble.py")
        return 2
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if a.limit:
        rows = rows[:a.limit]
    dec = make(a.decider, a.timeout or None)
    print(f"{type(dec).__name__} over {len(rows)} examples from {path.name}, "
          f"concurrency {a.concurrency}", flush=True)

    t0 = time.time()
    res, errors = await run(dec, rows, a.concurrency)
    secs = time.time() - t0
    if not res:
        print("nothing scored")
        return 1
    res.sort(key=lambda r: str(r["id"]))
    summary = report(a.decider, path.stem, res, errors, secs)
    if a.jsonl:
        Path(a.jsonl).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
        print(f"\nper-example predictions -> {a.jsonl}")
    if a.summary:
        Path(a.summary).write_text(json.dumps(summary, indent=2))
    close = getattr(dec, "aclose", None)
    if close:
        await close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
