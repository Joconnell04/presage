"""E23 bench and smoke test for `presage.verify`. No model calls, no network.

    uv run python -m presage.verify_smoke [--frames DIR] [--json OUT.json] [--show N]

Four corpora, all of them cards presage really produced or really would:

* **answer key** - `sample_context/ANSWER_KEY.md` and `docs/replay/sim/big/ANSWER_KEY.json`,
  the hand-verified ground truth for the two sample folders. Every one must verify.
* **Q&A cache** - the deep pass's anticipated questions, which is what the app serves
  instantly on a call. Thousands of real model-written cards carrying real citations.
* **simulator** - the answer frames from `docs/experiments/sim/*/frames.jsonl` (E5).
* **corrupted** - 40 cards built from the good ones by the four mistakes that matter:
  the figure changed, the wrong sheet, the wrong page, a number invented outright.

Prints the true-positive and false-positive rates, the per-card cost, and the level mix.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

from presage.ingest import Index
from presage.verify import ALL_FIELDS, SPOKEN, numbers_in, verify

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = [f for f in (ROOT / "sample_context", ROOT / "docs/replay/sim/big") if f.exists()]
SEED = 23
N_CORRUPT = 40
QA_SAMPLE = 400          # per folder, so one big workbook cannot swamp the rates


# --------------------------------------------------------------------------- corpora

def key_cards_md(path: Path, folder: str) -> list[dict]:
    """Cards from the markdown answer key: one per `| question | value | citation |` row."""
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[1].startswith("---") or cells[0].lower() == "question":
            continue
        q, val, cite = cells[0], cells[1], cells[2]
        if "›" not in cite or "NULL" in val or not numbers_in(val):
            continue
        f, loc = cite.split("›", 1)
        out.append({"id": f"key:{q[:40]}", "folder": folder, "question": q,
                    "answer": f"{q}: {val}.", "say": f"{q} is {val}", "figure": val,
                    "citations": [{"file": f.strip(), "loc": loc.strip()}],
                    "answered_from": "files", "working": None, "_src": "answer key"})
    return out


def key_cards_json(path: Path, folder: str) -> list[dict]:
    data = json.loads(path.read_text())
    out = []
    for e in data.get("entries", []):
        if not e.get("figure"):
            continue
        ask = e.get("ask") or e.get("id")
        out.append({"id": f"key:{e.get('id')}", "folder": folder, "question": ask,
                    "answer": f"{ask} is {e['figure']}.", "say": f"{ask} is {e['figure']}",
                    "figure": e["figure"],
                    "citations": [{"file": e["file"], "loc": e["loc"]}],
                    "answered_from": "files", "working": None, "_src": "answer key"})
    return out


def qa_cards(index: Index, folder: str, rng: random.Random) -> list[dict]:
    """The deep pass's anticipated-question cache: real cards with real citations."""
    qa = list(getattr(index, "_qa", []) or [])
    rng.shuffle(qa)
    out = []
    for a in qa:
        if not (a.figure or "").strip() or not a.citations:
            continue
        out.append({"id": f"qa:{a.question[:40]}", "folder": folder, "question": a.question,
                    "answer": a.answer, "say": a.say, "figure": a.figure,
                    "citations": [dict(c) for c in a.citations],
                    "answered_from": "derived" if a.derived else "files",
                    "working": a.working, "_src": "Q&A cache"})
        if len(out) >= QA_SAMPLE:
            break
    return out


def sim_cards(frames_dir: Path, folder_of: dict) -> list[dict]:
    """Final (non-partial) answer frames from the E5 simulator runs."""
    out = []
    for path in sorted(frames_dir.glob("*.jsonl")):
        script = path.stem
        folder = folder_of.get(script)
        if folder is None:
            continue
        latest: dict = {}
        for line in path.open():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") == "answer" and not o.get("partial"):
                latest[o.get("id")] = o
        for a in latest.values():
            a = dict(a)
            a["folder"] = folder
            a["_src"] = "simulator"
            out.append(a)
    return out


def replay_cards(scratch: Path) -> list[dict]:
    """The replay bench logs. They record `say` and `figure` and no citations at all, which
    is itself the finding: a card with no citation can never be more than unverified."""
    out = []
    for path in sorted(scratch.glob("e2e_bench*.log")):
        for m in re.finditer(r"say=(['\"])(.*?)\1\s+figure=(?:(['\"])(.*?)\3|None)",
                             path.read_text()):
            out.append({"id": f"replay:{path.stem}:{len(out)}", "folder": None,
                        "question": "", "answer": m.group(2), "say": m.group(2),
                        "figure": m.group(4), "citations": [], "answered_from": "files",
                        "working": None, "_src": "replay bench"})
    return out


# --------------------------------------------------------------------------- corruption

def _sheets(index: Index, file_: str) -> list[str]:
    return [r[0] for r in index.con.execute(
        "SELECT DISTINCT sheet FROM cells WHERE file=? AND sheet<>''", (file_,))]


def _bump(text: str, factor: float) -> str:
    """Rewrite the first number in a line, keeping its currency and its suffix."""
    def sub(m: re.Match) -> str:
        raw = m.group("num").replace(",", "")
        try:
            val = float(raw) * factor
        except ValueError:
            return m.group(0)
        shown = f"{val:,.2f}".rstrip("0").rstrip(".") if "." in raw else f"{val:,.0f}"
        return f"{m.group('cur') or ''}{shown}{m.group('unit') or ''}"
    return re.sub(r"(?<![A-Za-z0-9.])(?P<cur>[$£€])?(?P<num>\d[\d,]*(?:\.\d+)?)"
                  r"(?P<unit>%|k|M|B|bn)?", sub, text or "", count=1)


def corrupt(cards: list[dict], indexes: dict[str, Index], rng: random.Random) -> list[dict]:
    """40 cards that are wrong on purpose, ten of each mistake that actually happens."""
    out: list[dict] = []
    per = N_CORRUPT // 4
    pool = [c for c in cards if c.get("figure") and c.get("citations")]
    rng.shuffle(pool)

    def take(pred, n):
        got = [c for c in pool if pred(c)][:n * 3]
        rng.shuffle(got)
        return got[:n]

    # 1. the figure moved: the commonest real failure, a card quoting last quarter's number
    for c in take(lambda c: bool(numbers_in(c["figure"])), per):
        d = dict(c)
        f = rng.choice([1.37, 0.72, 1.15, 2.4])
        d["figure"], d["say"] = _bump(c["figure"], f), _bump(c.get("say") or "", f)
        d["answer"] = _bump(c.get("answer") or "", f)
        d["_bad"], d["_why"] = True, "figure changed"
        out.append(d)

    # 2. the right cell on the wrong sheet
    def wrong_sheet(c):
        cit = c["citations"][0]
        ix = indexes[c["folder"]]
        others = [s for s in _sheets(ix, cit.get("file", ""))
                  if s.lower() not in (cit.get("loc") or "").lower()]
        return others
    for c in take(lambda c: bool(c["citations"]) and bool(wrong_sheet(c)), per):
        d = dict(c)
        cit = dict(c["citations"][0])
        ref = re.split(r"!|›", cit.get("loc", ""))[-1].strip()
        cit["loc"] = f"{rng.choice(wrong_sheet(c))} › {ref}"
        d["citations"] = [cit] + [dict(x) for x in c["citations"][1:]]
        d["_bad"], d["_why"] = True, "wrong sheet"
        out.append(d)

    # 3. the right fact on the wrong page
    def paged(c):
        return any(re.search(r"\b(page|slide)\s+\d+", x.get("loc") or "") for x in c["citations"])
    for c in take(paged, per):
        d = dict(c)
        cits = []
        for x in c["citations"]:
            y = dict(x)
            y["loc"] = re.sub(r"\b(page|slide)\s+(\d+)",
                              lambda m: f"{m.group(1)} {int(m.group(2)) % 3 + 1}", y.get("loc") or "")
            cits.append(y)
        d["citations"] = cits
        d["_bad"], d["_why"] = True, "wrong page"
        out.append(d)

    # 4. a number from nowhere, of the shape a model invents when it pattern-matches a table
    for c in take(lambda c: True, per):
        d = dict(c)
        made = f"${rng.randint(1000, 9999) / 10:.1f}M"
        d["figure"] = made
        d["say"] = f"{(c.get('say') or 'It is').split(' ')[0]} about {made}"
        d["answer"] = f"{(c.get('answer') or '').rstrip('.')}, or {made} on the latest cut."
        d["_bad"], d["_why"] = True, "invented number"
        out.append(d)
    return out


# --------------------------------------------------------------------------- the bench

def run(cards: list[dict], indexes: dict[str, Index], fields) -> list[dict]:
    rows = []
    for c in cards:
        ix = indexes.get(c.get("folder"))
        if ix is None:
            ix = next(iter(indexes.values()))
        t0 = time.perf_counter()
        v = verify(c, ix, fields=fields)
        rows.append({"id": c["id"], "src": c.get("_src"), "bad": bool(c.get("_bad")),
                     "why": c.get("_why"), "figure": c.get("figure"),
                     "ms": (time.perf_counter() - t0) * 1000.0, **v.as_dict()})
    return rows


def rates(rows: list[dict], src: str | None = None, bad: bool | None = None) -> dict:
    sel = [r for r in rows if (src is None or r["src"] == src) and (bad is None or r["bad"] == bad)]
    if not sel:
        return {"n": 0}
    flagged = [r for r in sel if not r["ok"]]
    unver = [r for r in sel if r["ok"] and r["level"] == "unverified"]
    badge = [r for r in sel if r["ok"] and (not r["strict"] or r["level"] == "unverified")]
    noev = [r for r in sel if r["ok"] and r["level"] == "unverified"]
    mis = [r for r in sel if r["ok"] and r["level"] != "unverified" and not r["strict"]]
    ms = sorted(r["ms"] for r in sel)
    return {"n": len(sel), "flagged": len(flagged), "flag_pct": 100.0 * len(flagged) / len(sel),
            "unverified_pct": 100.0 * len(unver) / len(sel),
            "badge_pct": 100.0 * len(badge) / len(sel),
            "noev_pct": 100.0 * len(noev) / len(sel),
            "mis_pct": 100.0 * len(mis) / len(sel),
            "clean_pct": 100.0 * (len(sel) - len(flagged) - len(badge)) / len(sel),
            "verified_pct": 100.0 * (len(sel) - len(flagged) - len(unver)) / len(sel),
            "p50_ms": ms[len(ms) // 2], "p95_ms": ms[min(len(ms) - 1, int(len(ms) * 0.95))],
            "max_ms": ms[-1]}


def main(argv: list[str]) -> int:
    frames = Path(next((a.split("=", 1)[1] for a in argv if a.startswith("--frames=")),
                       ROOT / "docs/experiments/sim"))
    out_json = next((a.split("=", 1)[1] for a in argv if a.startswith("--json=")), "")
    show = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--show=")), "12"))
    scratch = Path(next((a.split("=", 1)[1] for a in argv if a.startswith("--scratch=")),
                        "/tmp"))
    rng = random.Random(SEED)

    indexes: dict[str, Index] = {}
    for f in FOLDERS:
        t0 = time.perf_counter()
        indexes[str(f)] = Index(f)
        print(f"index {f.name}: open in {1000 * (time.perf_counter() - t0):.0f} ms, "
              f"{indexes[str(f)].stats().cells} cells")

    good: list[dict] = []
    for f in FOLDERS:
        md, js = f / "ANSWER_KEY.md", f / "ANSWER_KEY.json"
        if js.exists():
            good += key_cards_json(js, str(f))
        elif md.exists():
            good += key_cards_md(md, str(f))
        good += qa_cards(indexes[str(f)], str(f), rng)

    folder_of = {"sample-01-quarterly": str(FOLDERS[0]), "sample-02-churn": str(FOLDERS[0]),
                 "sample-03-regions": str(FOLDERS[0]), "big-01-capital": str(FOLDERS[-1]),
                 "big-02-model": str(FOLDERS[-1])}
    sims = sim_cards(frames, folder_of) if frames.exists() else []
    good += sims
    replays = replay_cards(scratch)
    bad = corrupt([c for c in good if c["_src"] != "replay bench"], indexes, rng)

    cards = good + replays + bad
    all_rows = {}
    for label, fields in (("all fields", ALL_FIELDS), ("spoken only", SPOKEN)):
        rows = run(cards, indexes, fields)
        all_rows[label] = rows
        print(f"\n=== {label} ({'/'.join(fields)}) " + "=" * 40)
        print(f"{'corpus':<14}{'n':>5}{'reject':>8}{'reject %':>10}{'no evid %':>11}"
              f"{'miscited %':>12}{'clean %':>9}{'p50 ms':>9}{'p95 ms':>9}")
        for src in ("answer key", "Q&A cache", "simulator", "replay bench"):
            r = rates(rows, src=src, bad=False)
            if r["n"]:
                print(f"{src:<14}{r['n']:>5}{r['flagged']:>8}{r['flag_pct']:>10.1f}"
                      f"{r['noev_pct']:>11.1f}{r['mis_pct']:>12.1f}{r['clean_pct']:>9.1f}"
                      f"{r['p50_ms']:>9.2f}{r['p95_ms']:>9.2f}")
        r = rates(rows, bad=False)
        print(f"{'GOOD (all)':<14}{r['n']:>5}{r['flagged']:>8}{r['flag_pct']:>10.1f}"
              f"{r['noev_pct']:>11.1f}{r['mis_pct']:>12.1f}{r['clean_pct']:>9.1f}"
              f"{r['p50_ms']:>9.2f}{r['p95_ms']:>9.2f}")
        print()
        for why in ("figure changed", "wrong sheet", "wrong page", "invented number"):
            sel = [x for x in rows if x["why"] == why]
            if sel:
                rej = sum(1 for x in sel if not x["ok"])
                bdg = sum(1 for x in sel
                          if x["ok"] and (not x["strict"] or x["level"] == "unverified"))
                print(f"  corrupted · {why:<16} rejected {rej:>2}/{len(sel)}, "
                      f"badged {bdg:>2}, caught either way {rej + bdg:>2}/{len(sel)} "
                      f"({100.0 * (rej + bdg) / len(sel):.0f}%), slipped through "
                      f"{len(sel) - rej - bdg}")
        r = rates(rows, bad=True)
        print(f"  corrupted · ALL              rejected {r['flagged']}/{r['n']} "
              f"({r['flag_pct']:.1f}%), caught either way "
              f"{r['n'] - round(r['clean_pct'] * r['n'] / 100):.0f}/{r['n']} "
              f"({100 - r['clean_pct']:.1f}%), p50 {r['p50_ms']:.2f} ms")

    rows = all_rows["all fields"]
    ms = sorted(r["ms"] for r in rows)
    print(f"\ncost: p50 {statistics.median(ms):.3f} ms, p95 "
          f"{ms[int(len(ms) * 0.95)]:.3f} ms, max {ms[-1]:.3f} ms over {len(ms)} cards "
          f"(the max is the first card on an index, which builds the lookup cache)")
    levels: dict[str, int] = {}
    for r in rows:
        if not r["bad"]:
            levels[r["level"]] = levels.get(r["level"], 0) + 1
    print("levels on good cards: " + ", ".join(f"{k} {v}" for k, v in sorted(levels.items())))
    why: dict[str, int] = {}
    for r in rows:
        if not r["bad"] and not r["ok"]:
            why[re.sub(r"[-\d,.$%›]+", "N", r["reason"])[:58]] = 1 + why.get(
                re.sub(r"[-\d,.$%›]+", "N", r["reason"])[:58], 0)
    print("why good cards were rejected:")
    for k, v in sorted(why.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {v:>4}  {k}")
    noev: dict[str, int] = {}
    for r in rows:
        if not r["bad"] and r["ok"] and r["level"] == "unverified":
            k = ("no citations" if not r["reason"] else
                 "derived, no what-if result" if "no figure" not in r["reason"] and
                 "resolve" in r["reason"] else r["reason"][:44])
            noev[k] = 1 + noev.get(k, 0)
    print("why good cards had no evidence:")
    for k, v in sorted(noev.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {v:>4}  {k}")

    if show:
        print(f"\nfirst {show} flags on good cards (hand-check these):")
        for r in [x for x in rows if not x["ok"] and not x["bad"]][:show]:
            print(f"  [{r['src']}] {str(r['figure'])[:18]:<18} {r['reason'][:100]}")
        print(f"\nfirst {show} corrupted cards that slipped through entirely:")
        for r in [x for x in rows if x["ok"] and x["bad"] and x["strict"]
                  and x["level"] != "unverified"][:show]:
            print(f"  [{r['why']}] {str(r['figure'])[:18]:<18} {r['level']:<11} "
                  f"{r['reason'][:80]}")

    if out_json:
        Path(out_json).write_text(json.dumps(
            {k: [{kk: vv for kk, vv in r.items() if kk != "resolved"} for r in v]
             for k, v in all_rows.items()}, indent=1))
        print(f"\nwrote {out_json}")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
