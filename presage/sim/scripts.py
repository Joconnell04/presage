"""Write a call script: a two-speaker conversation with the outcome presage should produce.

    uv run python -m presage.sim.scripts --context sample_context --id sample-01-numbers \\
        --title "Quarterly review" --turns 20 --seed 1

Sonnet writes the dialogue through the project's own provider (`presage.llm.make_llm`,
`classify(..., model="claude-sonnet-5")`), in chunks of a few turns so each response fits
the classifier's output budget and every chunk sees the ones before it.

The figures are never written by the model. The context folder's answer key is a numbered
menu (`ANSWER_KEY.json` for a folder built by `presage.sim.answer_key`, otherwise the tables
in `ANSWER_KEY.md`), the model picks the id a question is asking for, and the expected
figure and citation are copied from that entry. A question with no id is still scored: a
card is expected, its figure is not checked.

Every turn carries `expect`:

    {"card": true, "figure": "$27.84M", "cells": ["... › P&L › G13"],
     "kind": "question|what-if|request|objection|none", "source": "files|derived|web|none"}

and the traps are deliberate: statements shaped like questions, the consultant's own
question-shaped lines, half sentences, follow-ups that only resolve against the previous
answer, what-ifs with a spoken number, questions the files cannot answer, and one public
fact only the web can answer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

from presage.llm import make_llm  # noqa: E402

SCRIPT_DIR = ROOT / "docs" / "replay" / "sim" / "scripts"
MODEL = "claude-sonnet-5"
CHUNK_TURNS = 5             # turns per model call: 5 x ~70 tokens fits the classify budget
KINDS = ("question", "what-if", "request", "objection", "none")
SOURCES = ("files", "derived", "web", "none")

# pauses after a turn, in seconds: long enough for a card to land, short enough to stay a call
PAUSE = {"question": 3.4, "what-if": 3.8, "request": 3.2, "objection": 3.2, "none": 1.8}
PAUSE_ME = 1.6
PAUSE_JITTER = 0.6          # deterministic per seed, keeps 1.5 <= pause <= 4.0

TRAPS = {
    "statement": "a Them line that is shaped like a question but is not one (\"so that's the "
                 "number I had in my head, right\"): kind none, card false",
    "me-question": "a Me line that is a question to the client, not to the copilot: card false",
    "half": "a Them line cut off mid-sentence, trailing off: kind none, card false",
    "follow-up": "a Them question that only makes sense against the answer before it "
                 "(\"and the year after?\"), with its own key id",
    "whatif": "a Them what-if that says a number out loud (\"what if we only land two of "
              "those instead of three\"): kind what-if, card true, source derived, no key id",
    "not-in-files": "a Them question the files cannot answer (a figure nobody has: a "
                    "competitor's margin, last year's actuals, a price nobody set): card true, "
                    "source none, no key id",
    "web": "a Them question about a public fact that is nowhere in the files (a market size, "
           "a rate, a channel's media deal): card true, source web, no key id",
    "small-talk": "a Them line of small talk or agreement: kind none, card false",
    "objection": "a Them objection that pushes back on a number in the files: kind objection, "
                 "card true, with the key id of the number it pushes on",
    "request": "a Them request to walk through something in the files: kind request, card "
               "true, with a key id",
}

TURN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["them", "me"]},
                    "text": {"type": "string", "description": "the spoken line, 5-26 words"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "card": {"type": "boolean",
                             "description": "true if presage should put a card on screen for this line"},
                    "key_ref": {"type": "string",
                                "description": "the answer-key id this line asks for, or empty"},
                    "source": {"type": "string", "enum": list(SOURCES)},
                    "trap": {"type": "string", "description": "the trap name, or empty"},
                },
                "required": ["speaker", "text", "kind", "card", "key_ref", "source", "trap"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["turns"],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------- answer keys

_ROW = re.compile(r"^\|(.+)\|\s*$")


def load_entries(folder: Path) -> list[dict]:
    """The answer key as a menu of {id, ask, figure, citation}.

    `ANSWER_KEY.json` (written by presage.sim.answer_key) when it is there; otherwise the
    markdown tables of `ANSWER_KEY.md`, which is what the sample folder ships.
    """
    js = folder / "ANSWER_KEY.json"
    if js.exists():
        return json.loads(js.read_text(encoding="utf-8"))["entries"]

    md = folder / "ANSWER_KEY.md"
    if not md.exists():
        raise SystemExit(f"no ANSWER_KEY.md or ANSWER_KEY.json in {folder}")
    entries: list[dict] = []
    in_table = False
    for line in md.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if not m:
            in_table = False
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        head = cells[0].lower()
        if head.startswith("question"):
            in_table = True
            continue
        if not in_table or set("".join(cells)) <= set("- "):
            continue
        ask, figure = cells[0], cells[1] if len(cells) > 1 else ""
        citation = cells[2] if len(cells) > 2 else ""
        if not ask or not figure or " / " in figure or not re.search(r"\d", figure):
            continue              # multi-value rows and prose rows make no single-figure question
        entries.append({"id": f"S{len(entries) + 1:02d}", "ask": ask, "figure": figure,
                        "citation": citation,
                        "null_value": "null" in figure.lower()})
    if not entries:
        raise SystemExit(f"no usable rows in {md}")
    return entries


def menu(entries: list[dict], limit: int = 60) -> str:
    return "\n".join(f"{e['id']} | {e['ask']} | {e['figure']} | {e.get('citation', '')}"
                     for e in entries[:limit])


# ----------------------------------------------------------------------------- generation

def plan_traps(seed: int, chunks: int) -> list[list[str]]:
    """Which traps each chunk must contain. Every script gets all of them, in its own order."""
    rng = random.Random(seed)
    must = ["statement", "me-question", "half", "follow-up", "whatif", "not-in-files", "web",
            "objection", "request", "small-talk"]
    rng.shuffle(must)
    out: list[list[str]] = [[] for _ in range(chunks)]
    for i, t in enumerate(must):
        out[i % chunks].append(t)
    return out


def prompt_for(folder: Path, title: str, entries: list[dict], so_far: list[dict],
               traps: list[str], n: int, them: int, first: bool, last: bool) -> str:
    lines = "\n".join(f"{'Me' if t['speaker'] == 'me' else 'Them'}: {t['text']}"
                      for t in so_far[-10:]) or "(the call has just connected)"
    trap_text = "\n".join(f"- {t}: {TRAPS[t]}" for t in traps) or "- none this chunk"
    return (
        "You are writing the transcript of a real consulting call, to test a copilot that "
        "listens to the client and answers from the consultant's own files.\n\n"
        f"The engagement: {title}. Me is the consultant, presenting their own material. Them is "
        "the client, who has the deck open and keeps interrupting with questions.\n\n"
        "ANSWER KEY (the only facts the copilot can look up; id | ask | figure | citation):\n"
        f"{menu(entries)}\n\n"
        f"CONVERSATION SO FAR:\n{lines}\n\n"
        f"Write the next {n} turns, {them} of them from Them.\n"
        "Rules:\n"
        "- Spoken English as a person actually says it: contractions, no stage directions, no "
        "names in brackets, 5 to 26 words a turn. Never say a figure from the key out loud, and "
        "never say an id or a file name.\n"
        "- A Them line the copilot should answer gets card true and, when it asks for a keyed "
        "fact, the id of that one entry in key_ref with source files. Ask for it the way a "
        "client says it, not in the key's words.\n"
        "- A Me line never gets a card: card false, source none.\n"
        "- Set kind honestly: question, what-if, request, objection, or none for statements, "
        "agreement, small talk and half sentences.\n"
        "- Use each key id at most once in the whole call.\n"
        f"- This chunk must contain these traps, one turn each, named in `trap`:\n{trap_text}\n"
        + ("- This is the start of the call: one short greeting, then straight into the material.\n"
           if first else "")
        + ("- This is the end of the call: finish on a close, not mid-thought.\n" if last else "")
        + "\nReturn only the JSON object."
    )


async def generate(folder: Path, title: str, turns: int, seed: int,
                   provider: str = "anthropic") -> list[dict]:
    entries = load_entries(folder)
    llm = make_llm(provider)
    chunks = max(1, round(turns / CHUNK_TURNS))
    traps = plan_traps(seed, chunks)
    out: list[dict] = []
    for i in range(chunks):
        n = CHUNK_TURNS if i < chunks - 1 else max(1, turns - len(out))
        them = max(2, round(n * 0.6))
        p = prompt_for(folder, title, entries, out, traps[i], n, them,
                       first=(i == 0), last=(i == chunks - 1))
        t0 = time.time()
        data = await llm.classify(p, TURN_SCHEMA, model=MODEL)
        got = [t for t in (data or {}).get("turns", []) if t.get("text")]
        print(f"  chunk {i + 1}/{chunks}: {len(got)} turns in {time.time() - t0:.1f}s "
              f"({', '.join(traps[i]) or 'no traps'})", file=sys.stderr)
        out.extend(got)
    return finalize(out, entries, seed)


# ----------------------------------------------------------------------------- shaping

def finalize(raw: list[dict], entries: list[dict], seed: int) -> list[dict]:
    """Model output -> turns with a checked `expect` block and a pause."""
    by_id = {e["id"]: e for e in entries}
    rng = random.Random(seed + 7)
    used: set[str] = set()
    turns: list[dict] = []
    for i, t in enumerate(raw):
        speaker = "me" if str(t.get("speaker", "")).lower().startswith("m") else "them"
        text = re.sub(r"\s+", " ", str(t.get("text", ""))).strip()
        kind = t.get("kind") if t.get("kind") in KINDS else "none"
        source = t.get("source") if t.get("source") in SOURCES else "none"
        card = bool(t.get("card"))
        ref = str(t.get("key_ref") or "").strip().upper()
        entry = by_id.get(ref)
        if entry is not None and ref in used:
            entry, ref = None, ""          # the same fact twice would score the same card twice
        if entry is not None:
            used.add(ref)

        if speaker == "me":                # the consultant's own lines never get a card
            card, source, ref, entry = False, "none", "", None
        if not card:
            source = "none"
        figure = ""
        cells: list[str] = []
        if entry is not None and card:
            figure = "not in files" if entry.get("null_value") else entry["figure"]
            source = "none" if entry.get("null_value") else "files"
            cells = [entry["citation"]] if entry.get("citation") else []
            if kind == "request":
                figure = ""    # "walk me through the headcount build" has no one right figure
        elif card and source == "none":
            figure = "not in files"

        base = PAUSE_ME if speaker == "me" else PAUSE.get(kind, 2.4)
        pause = round(min(4.0, max(1.5, base + rng.uniform(-PAUSE_JITTER, PAUSE_JITTER))), 2)
        turns.append({
            "i": i,
            "speaker": speaker,
            "text": text,
            "pause_after_s": pause,
            "trap": str(t.get("trap") or "").strip(),
            "expect": {"card": card, "figure": figure, "cells": cells, "kind": kind,
                       "source": source, "key_ref": ref if entry is not None else ""},
        })
    return turns


def save(script: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(script, indent=1), encoding="utf-8")
    return path


def load(script_id: str) -> dict:
    p = script_id if str(script_id).endswith(".json") else SCRIPT_DIR / f"{script_id}.json"
    return json.loads(Path(p).read_text(encoding="utf-8"))


def summarize(script: dict) -> str:
    turns = script["turns"]
    them = [t for t in turns if t["speaker"] == "them"]
    cards = [t for t in them if t["expect"]["card"]]
    keyed = [t for t in cards if t["expect"]["key_ref"]]
    traps = [t["trap"] for t in turns if t["trap"]]
    return (f"{len(turns)} turns, {len(them)} from Them, {len(cards)} expect a card "
            f"({len(keyed)} with a keyed figure), traps: {', '.join(sorted(set(traps)))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", required=True, type=Path, help="the context folder")
    ap.add_argument("--id", required=True, help="script id, e.g. sample-01-numbers")
    ap.add_argument("--title", default="", help="one line on what the call is about")
    ap.add_argument("--turns", type=int, default=20, help="total turns (Me + Them)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--provider", default="anthropic")
    a = ap.parse_args()

    folder = a.context.expanduser().resolve()
    title = a.title or folder.name
    print(f"writing {a.id} on {folder} with {MODEL}", file=sys.stderr)
    turns = asyncio.run(generate(folder, title, a.turns, a.seed, a.provider))
    try:
        where = str(folder.relative_to(ROOT))      # keep the script portable across checkouts
    except ValueError:
        where = str(folder)
    script = {"id": a.id, "context": where, "title": title, "seed": a.seed,
              "model": MODEL, "turns": turns}
    path = save(script, SCRIPT_DIR / f"{a.id}.json")
    print(f"{path}: {summarize(script)}")


if __name__ == "__main__":
    main()
