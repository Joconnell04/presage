"""Prove a script before it costs a server: every figure is in the folder, every expectation scores.

    uv run python -m presage.sim.check --all
    uv run python -m presage.sim.check big-03-deck sample-04-quiet
    uv run python -m presage.sim.run --dry --all          # the same thing, from the runner

Nothing here starts a server, reads audio, or calls a model. Two checks, and a script has to
pass both before it is worth a replay:

1. **Every expected figure is in the context folder.** The folder is copied to a temp directory
   and indexed there -- indexing writes `<folder>/.presage`, and a bench folder is never
   written to -- and then each expectation's figure is looked up in that index: by its citation
   (`file › sheet › cell`) when it carries one, and otherwise by value across every indexed
   cell, extracted document number and table row. A figure the folder cannot produce is a
   question the platform cannot answer, and it would score as the platform's failure.

2. **Every expectation parses under scorer v2.** A synthetic `frames.jsonl` is built in which
   the platform answers every turn perfectly -- the right figure, from the right place, one
   second after the line ends -- and `presage.sim.run.score(..., version=2)` has to come back
   with every question a hit, no wrong card and no miss. A `figure` the scorer cannot read, a
   `source` it treats differently than the script meant, or a `key_ref` that points at an entry
   the folder's key does not have all show up here as a miss on a perfect transcript.

The timeline comes from the rendered clip when there is one and is estimated from the word
count when there is not, so a script can be checked before `presage.sim.render` is run.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from presage.sim import render as R
from presage.sim import run as RUN
from presage.sim import scripts as S

ROOT = S.ROOT
CACHE = Path(tempfile.gettempdir()) / "presage-sim-check"   # one indexed copy per folder, reused
ANSWER_AT_S = 1.2        # where the synthetic answer lands after the line ends: inside any window
WORDS_PER_S = R.WORDS_PER_MIN / 60.0


# ----------------------------------------------------------------------------- the folder

def indexed_copy(folder: Path, rebuild: bool = False) -> Path:
    """A copy of the context folder with its own index. The original is never written to.

    `presage.ingest.Index.build` writes `<folder>/.presage`, so pointing it at
    `sample_context` or `docs/replay/sim/big` would change the folder the bench runs on. The
    copy lands under the system temp directory and is reused until `--rebuild`.
    """
    dst = CACHE / folder.name
    if dst.exists() and not rebuild:
        if (dst / ".presage" / "index.db").exists():
            return dst
        shutil.rmtree(dst, ignore_errors=True)
    elif dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    for p in folder.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    src_ix = folder / ".presage"
    if src_ix.is_dir():
        # the folder's own index comes with it, so the build below is incremental and, more to
        # the point, the `links` table survives: E16 builds it in the deep pass and E63's drift
        # check reads it, and a fresh index of a copy would have none. The database is taken
        # through sqlite's own backup, not copied: another bench writing to the folder while
        # this runs would otherwise leave a torn file and an index that reads empty.
        shutil.copytree(src_ix, dst / ".presage", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("index.db*"))
        src_db = src_ix / "index.db"
        if src_db.exists():
            src_con = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
            dst_con = sqlite3.connect(dst / ".presage" / "index.db")
            try:
                src_con.backup(dst_con)
            finally:
                dst_con.close()
                src_con.close()
    from presage.ingest import Index           # imported late: the CLI should not pay for it
    Index().build(dst)
    return dst


def _con(folder: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{folder / '.presage' / 'index.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= RUN.FIG_TOL * max(abs(a), abs(b), 1e-9)


def _values(figure: str) -> list[float]:
    """Every number the figure string could mean, the scorer's own reading."""
    return RUN.candidates(figure)


def find_figure(con: sqlite3.Connection, figure: str, citation: str = "") -> dict:
    """Where the folder holds this figure. -> {"found": bool, "where": str, "how": str}"""
    want = _values(figure)
    if not want:
        return {"found": False, "where": "", "how": "no number in the figure"}

    parts = [p.strip() for p in str(citation or "").split("›")]
    if len(parts) == 3:
        file, sheet, ref = parts
        row = con.execute("SELECT value_num, row_label FROM cells WHERE file=? AND sheet=? "
                          "AND cell_ref=?", (file, sheet, ref)).fetchone()
        if row is not None and row["value_num"] is not None:
            got = float(row["value_num"])
            if any(_same(v, got) for v in want):
                return {"found": True, "where": citation, "how": "cell"}
            return {"found": False, "where": citation,
                    "how": f"the cell holds {got:g}, the key says {figure}"}

    for v in want:                     # no citation, or a citation that is not a cell reference
        lo, hi = v - abs(v) * RUN.FIG_TOL - 1e-9, v + abs(v) * RUN.FIG_TOL + 1e-9
        row = con.execute("SELECT file, sheet, cell_ref FROM cells WHERE value_num BETWEEN ? "
                          "AND ? LIMIT 1", (lo, hi)).fetchone()
        if row is not None:
            return {"found": True, "where": f"{row['file']} › {row['sheet']} › {row['cell_ref']}",
                    "how": "by value, in a cell"}
        row = con.execute("SELECT file, loc FROM numbers WHERE value_num BETWEEN ? AND ? LIMIT 1",
                          (lo, hi)).fetchone()
        if row is not None:
            return {"found": True, "where": f"{row['file']} › {row['loc']}",
                    "how": "by value, in a document"}
    found = _in_tables(con, want)
    if found:
        return found
    return {"found": False, "where": "", "how": "no cell, number or table row holds it"}


def _in_tables(con: sqlite3.Connection, want: list[float]) -> dict | None:
    """The csv and table stores: a total that lives in rows rather than in one cell."""
    names = [r["name"] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\' "
        "AND name NOT IN ('cells','numbers','links','definitions')")]
    for name in names:
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
        nums = [c for c in cols if not c.startswith("_")]
        for col in nums:
            for v in want:
                lo, hi = v - abs(v) * RUN.FIG_TOL - 1e-9, v + abs(v) * RUN.FIG_TOL + 1e-9
                try:
                    row = con.execute(f'SELECT 1 FROM "{name}" WHERE "{col}" BETWEEN ? AND ? '
                                      f'LIMIT 1', (lo, hi)).fetchone()
                except sqlite3.Error:
                    continue
                if row is not None:
                    return {"found": True, "where": f"{name}.{col}", "how": "by value, in a table"}
                try:                      # a column of dollars against a key in thousands
                    row = con.execute(f'SELECT 1 FROM "{name}" WHERE "{col}" BETWEEN ? AND ? '
                                      f'LIMIT 1', (lo * 1000, hi * 1000)).fetchone()
                except sqlite3.Error:
                    continue
                if row is not None:
                    return {"found": True, "where": f"{name}.{col}",
                            "how": "by value x1,000, in a table"}
    return None


def _sum_in_tables(con: sqlite3.Connection, want: list[float]) -> dict | None:
    """A figure the folder holds as a total rather than as a value: the sum of a column, or the
    sum of one group of it ("churned ARR" is `sum(arr_usd) where churned = 1`)."""
    names = [r["name"] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\' "
        "AND name NOT IN ('cells','numbers','links','definitions')")]
    for name in names:
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")') if not r[1].startswith("_")]
        for col in cols:
            try:
                got = con.execute(f'SELECT SUM("{col}") FROM "{name}"').fetchone()[0]
            except sqlite3.Error:
                continue
            if got is None:
                continue
            for v in want:
                if _same(v, float(got)):
                    return {"found": True, "where": f"sum({name}.{col})", "how": "a column total"}
            for by in cols:                     # one group of it: by segment, by region, churned
                if by == col:
                    continue
                try:
                    groups = con.execute(f'SELECT "{by}" AS g, SUM("{col}") AS s, COUNT(*) AS n '
                                         f'FROM "{name}" GROUP BY "{by}"').fetchall()
                except sqlite3.Error:
                    continue
                if not groups or len(groups) > 12:
                    continue
                for g in groups:
                    if g["s"] is None:
                        continue
                    for v in want:
                        if _same(v, float(g["s"])):
                            return {"found": True, "where": f"sum({name}.{col}) where {by}={g['g']}",
                                    "how": "a group total"}
                        if _same(v, float(g["n"])):
                            return {"found": True, "where": f"count({name}) where {by}={g['g']}",
                                    "how": "a group count"}
    return None


# ----------------------------------------------------------------------------- the checks

def check_figures(script: dict, rebuild: bool = False) -> list[dict]:
    """One row per expectation that names a figure, saying where the folder holds it."""
    folder = (ROOT / script["context"]).resolve()
    copy = indexed_copy(folder, rebuild)
    keys = {e["id"]: e for e in S.load_entries(folder)}
    out: list[dict] = []
    con = _con(copy)
    try:
        for t in script["turns"]:
            e = t["expect"]
            want = RUN.wanted_figure(e, keys)
            if not e.get("card") or not want:
                continue
            entry = keys.get(e.get("key_ref") or "") or {}
            got = find_figure(con, want, entry.get("citation", "") or
                              (e.get("cells") or [""])[0])
            if not got["found"]:
                more = _sum_in_tables(con, _values(want))
                if more:
                    got = more
            out.append({"i": t["i"], "text": t["text"][:60], "key_ref": e.get("key_ref", ""),
                        "figure": want, "kind": e.get("kind"), "source": e.get("source"), **got})
    finally:
        con.close()
    return out


def open_index(folder: Path, rebuild: bool = False):
    """The folder's index, on a copy, ready for the checks that need more than SQL."""
    from presage.ingest import Index
    ix = Index()
    ix.build(indexed_copy(folder, rebuild))
    return ix


def check_unprompted(script: dict, rebuild: bool = False) -> list[dict]:
    """The cards nobody asks for: does the platform's own check actually fire on this line?

    A script can expect a correction (E9c) or a drift card (E63) all it likes; whether one is
    raised depends on a guard neither the script nor the scorer can see. Both are pure and
    local, so they are run here, on a copy of the folder, before the script costs a server:

      `source: "correction"` -> `presage.spoken.check(line, index)`, the call
        `Brain._check_spoken` makes, has to come back `contradicts` naming the cell the
        expectation cites;
      `source: "drift"` -> `presage.drift.check(index, screen)`, the call
        `Brain.on_screen_change` makes, has to report a disagreement on that cell for the page
        that turn puts on screen.
    """
    turns = [t for t in script["turns"]
             if t["expect"].get("card") and t["expect"].get("source") in ("correction", "drift")]
    me = [t for t in script["turns"]
          if t["speaker"] == "me" and t["expect"].get("source") != "correction"]
    if not turns and not me:
        return []
    folder = (ROOT / script["context"]).resolve()
    ix = open_index(folder, rebuild)
    out: list[dict] = []
    from presage import spoken
    for t in me:                    # a consultant's line that contradicts the files by accident
        v = spoken.check(t["text"], ix)
        if v.verdict == "contradicts":
            out.append({"i": t["i"], "source": "correction", "text": t["text"][:60],
                        "want_cell": "", "figure": "", "ok": False, "unwanted": True,
                        "got": f"{getattr(v, 'cell', '')} = {getattr(v, 'actual', '')}",
                        "how": f"a line with no correction expectation contradicts the files: "
                               f"{getattr(v, 'reason', '')[:60]}"})
    for t in turns:
        e, src = t["expect"], t["expect"]["source"]
        want_cell = (e.get("cells") or [""])[0]
        row = {"i": t["i"], "source": src, "text": t["text"][:60], "want_cell": want_cell,
               "figure": e.get("figure", ""), "ok": False, "how": ""}
        if src == "correction":
            from presage import spoken
            v = spoken.check(t["text"], ix)
            got = getattr(v, "cell", "") or ""
            row["how"] = f"{v.verdict}: {getattr(v, 'reason', '')[:70]}"
            row["got"] = f"{got} = {getattr(v, 'actual', '')}"
            row["ok"] = (v.verdict == "contradicts"
                         and _same_cell(got, want_cell)
                         and RUN.figure_match(e.get("figure", ""), str(getattr(v, "actual", ""))))
        else:
            from presage import drift as driftmod
            state = t.get("screen") or {}
            rep = driftmod.check(ix, {"file": state.get("file"), "loc": state.get("loc")})
            got = rep.as_dict()
            first = (got.get("drifts") or [{}])[0]
            row["how"] = (f"{got.get('links')} links, {len(got.get('drifts') or [])} drift: "
                          f"{first.get('line', '')[:70]}")
            row["got"] = f"{first.get('cell', '')} = {first.get('now', '')}"
            row["ok"] = bool(first) and _same_cell(str(first.get("cell", "")), want_cell) \
                and RUN.figure_match(e.get("figure", ""), str(first.get("now", "")))
        out.append(row)
    return out


def _same_cell(got: str, want: str) -> bool:
    """`Sheet!B45` (what the guards report) against `File › Sheet › B45` (what a script cites)."""
    a = re.split(r"\s*[›!]\s*", str(got or "").strip())
    b = re.split(r"\s*[›!]\s*", str(want or "").strip())
    return bool(a) and bool(b) and a[-2:] == b[-2:]


def check_key_refs(script: dict) -> list[str]:
    """A `key_ref` the folder's key does not have, or an `expect.figure` that disagrees with it."""
    keys = {e["id"]: e for e in S.load_entries((ROOT / script["context"]).resolve())}
    bad: list[str] = []
    for t in script["turns"]:
        e = t["expect"]
        ref = e.get("key_ref") or ""
        if not ref:
            continue
        entry = keys.get(ref)
        if entry is None:
            bad.append(f"t{t['i']}: key_ref {ref} is not in the folder's key")
            continue
        fig = str(e.get("figure") or "")
        if fig and fig != "not in files" and not RUN.figure_match(entry["figure"], fig):
            bad.append(f"t{t['i']}: expect.figure {fig} != key {ref} {entry['figure']}")
    return bad


def fake_timeline(script: dict) -> dict:
    """The clip's timeline, or an estimate of it: `say` at 185 wpm plus the scripted pauses."""
    try:
        return R.load_timeline(script["id"])
    except (OSError, ValueError):
        pass
    turns, cursor = [], R.LEAD_IN_S
    for t in script["turns"]:
        dur = max(0.8, len(R.speakable(t["text"]).split()) / WORDS_PER_S)
        turns.append({"i": t["i"], "speaker": t["speaker"], "start_s": round(cursor, 3),
                      "end_s": round(cursor + dur, 3), "pause_after_s": t["pause_after_s"],
                      "text": t["text"]})
        cursor += dur + t["pause_after_s"]
    return {"id": script["id"], "context": script["context"], "estimated": True,
            "duration_s": round(cursor + R.TAIL_S, 3), "them_wav": f"{script['id']}.them.wav",
            "me_wav": f"{script['id']}.me.wav", "turns": turns}


def perfect_frames(script: dict, timeline: dict, anchor: float) -> list[dict]:
    """The call the platform got completely right, as WebSocket frames.

    One final transcript per turn on the channel that turn was spoken on, and, for every turn
    that expects a card, one answer frame carrying the figure the script asked for, from the
    place the script said it should come from, `ANSWER_AT_S` after the line ended.
    """
    keys = RUN.key_entries(script)
    by_i = {t["i"]: t for t in script["turns"]}
    frames: list[dict] = []
    utt = {"them": 0, "me": 0}
    for t in timeline["turns"]:
        e = by_i[t["i"]]["expect"]
        spk = t["speaker"]
        u = utt[spk]
        utt[spk] += 1
        frames.append({"type": "transcript", "source": "system" if spk == "them" else "mic",
                       "utt": u, "text": t["text"], "final": True, "t0": t["start_s"],
                       "t1": t["end_s"], "_ts": anchor + t["end_s"] + 0.4})
        if not e.get("card"):
            continue
        want = RUN.wanted_figure(e, keys)
        src = e.get("source")
        unprompted = src in RUN.UNPROMPTED      # a correction or a drift card carries no `utt`
        frames.append({
            "type": "answer", "id": f"c{t['i']}", "utt": None if unprompted else u,
            "question": t["text"],
            "figure": want, "answer": (f"{want}." if want else "Not in the files."),
            "say": (f"{want}." if want else "Not in the files."),
            "source": "live", "answered_from": {"files": "files", "derived": "files",
                                                "web": "web", "none": "none",
                                                "correction": "correction",
                                                "drift": "drift"}.get(src, "files"),
            "partial": False, "cached": False, "in_files": src == "files",
            "citations": [{"file": c.split("›")[0].strip(), "loc": "›".join(c.split("›")[1:]).strip()}
                          for c in (e.get("cells") or [])],
            "_ts": anchor + t["end_s"] + ANSWER_AT_S,
        })
    return frames


def check_scores(script: dict, timeline: dict) -> dict:
    """Score the perfect call. Everything asked for has to come back a hit."""
    anchor = 1_000_000.0
    frames = perfect_frames(script, timeline, anchor)
    sc = RUN.score(script, timeline, frames, "", anchor,
                   datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), version=2)
    t = sc["totals"]
    bad = [r for r in sc["rows"]
           if (r["expect"]["card"] and r["verdict"] != "hit") or r["wrong_cards"]
           or r["noise_cards"]]
    return {"questions": sc["questions"], "hits": t["hits"], "wrong": t["wrong_cards"],
            "missed": t["missed"], "bad": [{"i": r["i"], "verdict": r["verdict"],
                                            "want": r["want_figure"], "got": r["figure"],
                                            "cards": r["card_labels"], "text": r["text"][:60]}
                                           for r in bad], "score": sc}


def check(script_id: str, rebuild: bool = False, quiet: bool = False) -> dict:
    script = S.load(script_id)
    timeline = fake_timeline(script)
    figs = check_figures(script, rebuild)
    refs = check_key_refs(script)
    sc = check_scores(script, timeline)
    guards = check_unprompted(script, rebuild)
    missing = [f for f in figs if not f["found"]]
    silent = [g for g in guards if not g["ok"]]
    ok = (not missing and not refs and not sc["bad"] and not silent
          and sc["hits"] == sc["questions"])
    if not quiet:
        cards = [t for t in script["turns"] if t["expect"]["card"]]
        them = [t for t in script["turns"] if t["speaker"] == "them"]
        print(f"{script_id}: {len(script['turns'])} turns, {len(them)} from Them, "
              f"{len(cards)} expect a card, {len(figs)} figures checked"
              + (" (timeline estimated)" if timeline.get("estimated") else ""))
        for f in missing:
            print(f"  MISSING t{f['i']} {f['key_ref'] or '-'} {f['figure']}: {f['how']}")
        for r in refs:
            print(f"  KEY {r}")
        for b in sc["bad"]:
            print(f"  SCORE t{b['i']} {b['verdict']}: wanted {b['want']!r} got {b['got']!r} "
                  f"{b['cards']} | {b['text']}")
        for g in guards:
            print(f"  {'guard' if g['ok'] else 'GUARD'} t{g['i']} {g['source']}: {g['how']}"
                  + ("" if g["ok"] else f" (wanted {g['want_cell']} = {g['figure']})"))
        print(f"  {'ok' if ok else 'FAILED'}: {sc['hits']}/{sc['questions']} hits on a perfect "
              f"call, {sc['wrong']} wrong, {sc['missed']} missed; "
              f"{len(figs) - len(missing)}/{len(figs)} figures found in the folder"
              + (f"; {len(guards) - len(silent)}/{len(guards)} unprompted cards fire"
                 if guards else ""))
    return {"script": script_id, "ok": ok, "figures": figs, "missing": len(missing),
            "key_problems": refs, "scoring": {k: v for k, v in sc.items() if k != "score"},
            "unprompted": guards, "estimated_timeline": bool(timeline.get("estimated"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", nargs="*", help="script id(s) under docs/replay/sim/scripts")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="re-index the folder copies first")
    ap.add_argument("--out", type=Path, default=None, help="write the full result as JSON")
    a = ap.parse_args()
    ids = a.script or ([p.stem for p in sorted(S.SCRIPT_DIR.glob("*.json"))] if a.all else [])
    if not ids:
        raise SystemExit("name a script, or --all")
    out = [check(sid, a.rebuild) for sid in ids]
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"-> {a.out}")
    bad = [r for r in out if not r["ok"]]
    print(f"\n{len(out) - len(bad)}/{len(out)} scripts pass")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
