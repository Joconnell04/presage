"""presage/feedback_report.py: nightly analysis of the intervention log (experiment E12).

Reads the append-only JSONL `presage.feedback` writes and answers the questions E12 exists to
answer: how often presage interrupts, how often it is wrong, and where the wrong cards sit, so
the "should I show this" threshold can be set from this machine's own evidence rather than from
a paper's.

    uv run python -m presage.feedback_report                  # the newest day in the log
    uv run python -m presage.feedback_report --date 2026-09-22
    uv run python -m presage.feedback_report --all            # every day pooled
    uv run python -m presage.feedback_report --no-write       # print only

It prints a table and writes `docs/experiments/feedback/<date>.md`.

Every number is MEASURED from the log. Two of them are weaker than the rest and are labelled
wherever they appear:

- **implied ignored**: a card that was pushed off the top by a later card with no verdict on it.
  It is the best evidence there is when nobody pressed a key, and it is not the same thing as a
  user saying "ignored".
- **gate association**: a card carries the gate verdict that fired around the moment its question
  landed, matched by time rather than by identity. Cards with no gate at all (prefetch, a manual
  ask, the heuristic fast path) show as `no gate`, which is a real category, not missing data.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from presage import feedback

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "experiments" / "feedback"

VERDICTS = list(feedback.VERDICTS)                       # used ignored wrong good
BANDS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
CARD_FIELDS = ("question", "utt", "utt_text", "source", "answered_from", "derived", "cached",
               "figure", "citations", "web_citations", "latency_ms", "mode", "model", "topic",
               "say", "answer", "working", "gate", "session")


# ----------------------------------------------------------------------------- folding

def fold(rows: list[dict]) -> list[dict]:
    """Append-only rows -> one record per card, newest state wins. Oldest first."""
    cards: dict[str, dict] = {}
    for r in rows:
        cid = str(r.get("id") or "")
        if not cid:
            continue
        ev = r.get("event")
        c = cards.get(cid)
        if ev == "card":
            if c is None:
                c = cards[cid] = {"id": cid, "ts": r.get("ts"), "first_ts": r.get("ts"), "updates": 0,
                                  "verdict": None, "verdict_after_s": None, "replaced": False,
                                  "up_s": None, "orphan": False}
            else:
                c["updates"] += 1
            for k in CARD_FIELDS:
                c[k] = r.get(k)
            c["ts"] = r.get("ts")
        elif ev == "replaced":
            if c is not None:
                c["replaced"] = True
                c["up_s"] = r.get("after_s")
        elif ev == "feedback":
            if c is None:
                c = cards[cid] = {"id": cid, "ts": r.get("ts"), "first_ts": r.get("ts"), "updates": 0,
                                  "verdict": None, "verdict_after_s": None, "replaced": False,
                                  "up_s": None, "orphan": True, "question": r.get("question", "")}
            c["verdict"] = r.get("verdict")
            c["verdict_after_s"] = r.get("after_s")
    return sorted(cards.values(), key=lambda c: c.get("first_ts") or 0)


def tier_of(card: dict) -> str:
    g = card.get("gate") or {}
    return str(g.get("tier") or "no gate")


def band_of(card: dict) -> str:
    g = card.get("gate") or {}
    p = g.get("confidence")
    if p is None:
        return "no gate"
    for lo, hi in BANDS:
        if lo <= float(p) < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return "no gate"


# ----------------------------------------------------------------------------- tallies

class Tally:
    """Counts for one group of cards: how many, how they were marked, how fast they were."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.cards = 0
        self.verdicts: Counter = Counter()
        self.implied = 0            # replaced with no verdict: the weak form of ignored
        self.latency: list[int] = []

    def add(self, c: dict) -> None:
        self.cards += 1
        v = c.get("verdict")
        if v:
            self.verdicts[v] += 1
        elif c.get("replaced"):
            self.implied += 1
        ms = c.get("latency_ms")
        if isinstance(ms, (int, float)):
            self.latency.append(int(ms))

    @property
    def marked(self) -> int:
        return sum(self.verdicts.values())

    @property
    def wrong_rate(self) -> float | None:
        return self.verdicts["wrong"] / self.marked if self.marked else None

    @property
    def ignored_rate(self) -> float | None:
        return self.verdicts["ignored"] / self.marked if self.marked else None

    @property
    def p50(self) -> int | None:
        return int(statistics.median(self.latency)) if self.latency else None

    def row(self) -> list[str]:
        return [self.name, str(self.cards), str(self.marked),
                *(str(self.verdicts[v]) for v in VERDICTS),
                pct(self.wrong_rate), str(self.implied),
                f"{self.p50}" if self.p50 is not None else "-"]


HEAD = ["", "cards", "marked", *VERDICTS, "wrong%", "impl.ign", "p50 ms"]


def group(cards: list[dict], key) -> list[Tally]:
    out: dict[str, Tally] = {}
    for c in cards:
        k = str(key(c) or "-")
        out.setdefault(k, Tally(k)).add(c)
    return sorted(out.values(), key=lambda t: (-t.cards, t.name))


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def short(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------- rendering

def table(head: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "  (none)"
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(head)]
    def line(cells):
        return "  " + "  ".join(str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i])
                                for i, c in enumerate(cells))
    return "\n".join([line(head), "  " + "  ".join("-" * w for w in widths), *(line(r) for r in rows)])


def md_table(head: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(none)_"
    h = [c if c else " " for c in head]
    out = ["| " + " | ".join(h) + " |",
           "|" + "|".join(["---"] + ["--:"] * (len(h) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def per_hour(cards: list[dict]) -> tuple[list[list[str]], float, float]:
    """Cards by clock hour, and the rate over the span they actually cover."""
    by_hour: Counter = Counter()
    for c in cards:
        ts = c.get("first_ts")
        if ts:
            by_hour[datetime.fromtimestamp(float(ts)).strftime("%H:00")] += 1
    ts = [float(c["first_ts"]) for c in cards if c.get("first_ts")]
    span_h = (max(ts) - min(ts)) / 3600 if len(ts) > 1 else 0.0
    rate = len(cards) / span_h if span_h > 0.02 else float(len(cards))
    rows = [[h, str(n)] for h, n in sorted(by_hour.items())]
    return rows, span_h, rate


def top_wrong(cards: list[dict], n: int = 10) -> list[list[str]]:
    wrong = [c for c in cards if c.get("verdict") == "wrong"]
    seen: dict[str, dict] = {}
    for c in wrong:
        q = (c.get("question") or c.get("utt_text") or "(no question)").strip()
        e = seen.setdefault(q.lower(), {"q": q, "n": 0, "tiers": Counter(), "from": Counter()})
        e["n"] += 1
        e["tiers"][tier_of(c)] += 1
        e["from"][str(c.get("answered_from") or "-")] += 1
    ranked = sorted(seen.values(), key=lambda e: -e["n"])[:n]
    return [[e["q"][:70], str(e["n"]),
             ", ".join(f"{k}" for k, _ in e["tiers"].most_common(2)),
             ", ".join(f"{k}" for k, _ in e["from"].most_common(2))] for e in ranked]


# ----------------------------------------------------------------------------- the report

def build(cards: list[dict], label: str, files: list[Path]) -> tuple[str, str]:
    """Returns (text for the terminal, markdown for the doc)."""
    total = Tally("all")
    for c in cards:
        total.add(c)
    hours, span_h, rate = per_hour(cards)
    updates = sum(int(c.get("updates") or 0) for c in cards)
    orphans = sum(1 for c in cards if c.get("orphan"))
    sessions = len({c.get("session") for c in cards if c.get("session")})

    headline = (f"{total.cards} cards over {span_h:.1f} h ({rate:.1f}/h) in "
                f"{plural(sessions, 'session')} · "
                f"{total.marked} marked · wrong {total.verdicts['wrong']} "
                f"({pct(total.wrong_rate)}% of marked, {pct(total.verdicts['wrong'] / total.cards if total.cards else None)}% of cards) · "
                f"ignored {total.verdicts['ignored']} ({pct(total.ignored_rate)}% of marked) · "
                f"implied ignored {total.implied} "
                f"({pct(total.implied / total.cards if total.cards else None)}% of cards)")

    groups = [("by source", lambda c: c.get("source")),
              ("by gate tier", tier_of),
              ("by mode", lambda c: c.get("mode")),
              ("by answered_from", lambda c: c.get("answered_from")),
              ("by gate confidence", band_of)]

    t = [f"presage intervention log · {label}", f"  {headline}", ""]
    m = [f"# Intervention log · {label}", "",
         f"Written by `presage/feedback_report.py` on {datetime.now().astimezone().isoformat(timespec='seconds')}. ",
         "Every number is MEASURED from the log named below; nothing here is assumed.", "",
         "Source: " + ", ".join(f"`{short(p)}`" for p in files), "", "## Bottom line", "",
         headline.replace(" · ", "  \n"), ""]
    if updates or orphans:
        note = (f"{plural(updates, 're-sent card frame')} (the same id updated in place: a streaming "
                f"answer does this on every chunk, and so does the web escalation) and "
                f"{plural(orphans, 'verdict')} on cards not in this file.")
        t += [f"  {note}", ""]
        m += [note, ""]

    t += ["  cards per clock hour", table(["hour", "cards"], hours), ""]
    m += ["## Cards per clock hour", "", md_table(["hour", "cards"], hours), ""]

    for name, key in groups:
        rows = [g.row() for g in group(cards, key)]
        t += [f"  {name}", table([name, *HEAD[1:]], rows), ""]
        m += [f"## {name.capitalize()}", "", md_table([name, *HEAD[1:]], rows), ""]

    wrong_rows = top_wrong(cards)
    t += ["  top wrong questions", table(["question", "n", "tier", "answered_from"], wrong_rows), ""]
    m += ["## Top wrong questions", "",
          md_table(["question", "n", "tier", "answered from"], wrong_rows), ""]

    hot = [g for g in group(cards, band_of) if g.verdicts["wrong"]]
    if hot:
        hot.sort(key=lambda g: -(g.wrong_rate or 0))
        worst = hot[0]
        line = (f"Wrong cards concentrate in confidence band {worst.name}: "
                f"{worst.verdicts['wrong']} of {worst.marked} marked ({pct(worst.wrong_rate)}%).")
    else:
        line = ("No card has been marked wrong yet, so there is no band to raise the threshold to. "
                "The gate columns above are the shape of the traffic, not a verdict on it.")
    t += [f"  {line}", ""]
    m += ["## What the bands say", "", line, ""]
    return "\n".join(t), "\n".join(m)


def load(log_dir: Path | None, date: str | None, every: bool) -> tuple[list[dict], str, list[Path]]:
    files = feedback.days(log_dir)
    if not files:
        return [], "no log", []
    if every:
        chosen, label = files, f"{files[0].stem} → {files[-1].stem}"
    elif date:
        chosen = [p for p in files if p.stem == date]
        label = date
        if not chosen:
            raise SystemExit(f"no log for {date}; have {', '.join(p.stem for p in files)}")
    else:
        chosen, label = [files[-1]], files[-1].stem
    rows: list[dict] = []
    for p in chosen:
        rows += feedback.read(p)
    rows.sort(key=lambda r: r.get("ts") or 0)
    return fold(rows), label, chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="nightly analysis of presage's intervention log")
    ap.add_argument("--date", help="YYYY-MM-DD; default is the newest day in the log")
    ap.add_argument("--all", action="store_true", help="pool every day")
    ap.add_argument("--log-dir", type=Path, default=None, help=f"default {feedback.LOG_DIR}")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-write", action="store_true", help="print only")
    a = ap.parse_args(argv)

    t0 = time.time()
    cards, label, files = load(a.log_dir, a.date, a.all)
    if not cards:
        where = a.log_dir or feedback.LOG_DIR
        print(f"no cards in the intervention log ({where})", file=sys.stderr)
        return 1
    text, md = build(cards, label, files)
    print(text)
    if not a.no_write:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        out = a.out_dir / f"{'all' if a.all else label}.md"
        out.write_text(md.rstrip() + "\n", encoding="utf-8")
        print(f"  wrote {out}  ({len(cards)} cards, {time.time() - t0:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
