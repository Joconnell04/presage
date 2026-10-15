"""The deep index: a second, slower pass that runs after the fast one.

The fast `build()` is mechanical and takes a few hundred milliseconds. This pass costs
model calls and a few seconds, so it runs in the background and everything it produces is
cached on disk under `<folder>/.presage/deep/` keyed by file **content hash**. An unchanged
file is never reprocessed; a changed file invalidates only its own units.

Six products, each independently cached and resumable:

1. summaries      one sentence per file, one line per sheet / slide / section   (model_fast)
2. contexts       a 50-100 token blurb per document chunk, prepended to what BM25
                  and the dense model index, never to what is displayed            (model_fast)
3. numbers        every number in the documents with its sentence, in SQLite       (no model)
4. definitions    what a formula-driven metric actually divides by, from openpyxl
                  formula text; an LLM only names it                               (model_fast)
5. qa             3-6 anticipated questions per slide / section / table with a
                  <=25 word answer, a <=8 word say-line, a figure and citations    (model_good)
6. scratch        the lever table and the sensitivity grid for each workbook: what a
                  what-if can move, what it moves, and the answer precomputed at a few
                  round values. CPU work (pycel) plus one call for spoken aliases  (model_fast)

Numbers in the Q&A are checked digit by digit against the source text before the entry is
kept, so the cache cannot carry a figure the model made up.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from . import scratch as _scratch

LLMJson = Callable[[str, dict, "str | None"], Awaitable[dict]]
# Pass real model names: presage's own classify() caps output at 300 tokens when `model` is
# None, which truncates every unit here. A truncated response is never cached, so the unit
# retries on the next run rather than caching an empty result.

DEEP_DIR = "deep"
CONCURRENCY = 6
CTX_BATCH = 5               # chunks per contextualisation call
SUM_BATCH = 12              # sections per summary call
# Both are sized so one call fits inside classify()'s 1000-token output cap.
QA_MAX_UNITS = 60           # per file, so a 200-slide deck does not run away
MAX_ATTEMPTS = 2            # after this a failing unit caches empty, so the cached
                            # path stays fast instead of retrying it every run
SUMMARY_TOKEN_BUDGET = 800
CACHE_VERSION = 3           # bump when a cache key scheme changes; old entries are dropped (3: ingest v2 headers)
SCRATCH_EXT = {".xlsx", ".xlsm"}    # what pycel can compile into a live model


@dataclass
class DeepStats:
    state: str = "idle"                 # idle | estimating | running | done | error
    done: int = 0
    total: int = 0
    qa: int = 0
    summaries: int = 0
    chunks_contextualized: int = 0
    numbers: int = 0
    definitions: int = 0
    error: str | None = None
    started: float = 0.0
    finished: float = 0.0
    # extras
    files: int = 0
    cached_units: int = 0
    llm_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    est_in_tokens: int = 0
    est_out_tokens: int = 0
    est_by_file: dict = field(default_factory=dict)
    qa_rejected: int = 0
    failed: int = 0
    wall_s: float = 0.0
    levers: int = 0
    grid: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        """The one-line progress string: what is done and what it produced."""
        return (f"{self.state} {self.done}/{self.total} units, {self.qa} q&a, "
                f"{self.summaries} summaries, {self.numbers} numbers, "
                f"{self.definitions} definitions, {self.levers} levers, "
                f"{self.grid} grid points, {self.wall_s:.1f}s")


@dataclass
class QA:
    question: str
    answer: str
    say: str = ""
    figure: str = ""
    citations: list[dict] = field(default_factory=list)
    file: str = ""
    loc: str = ""
    score: float = 0.0
    derived: bool = False       # computed by scratch paper, not read off the page
    working: str = ""           # the lever change behind a derived answer
    lever_value: float | None = None   # the input value a derived entry was computed at


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


# chars/4 is the usual rule of thumb, but these prompts are dense with numbers and table
# cells, which tokenize far worse than prose. Calibrated against measured usage on
# sample_context: chars/3 for the body plus a flat allowance for the JSON schema on each
# structured-output call lands within about 8% of the real input count.
CHARS_PER_TOKEN = 3.0
SCHEMA_TOKENS = 250


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


_DIGITS = re.compile(r"\d[\d,\.]*")


def digit_runs(text: str) -> set[str]:
    """'$14,020k' -> {'14020'}; used to check a model figure against its source."""
    out = set()
    for m in _DIGITS.finditer(text or ""):
        d = m.group(0).replace(",", "").rstrip(".")
        if not d:
            continue
        out.add(d)
        if "." in d:
            out.add(d.split(".")[0])
        out.add(d.replace(".", ""))
    return out


def numbers_are_grounded(claim: str, source: str) -> bool:
    """Every multi-digit number in `claim` must appear in `source`. Cheap, mechanical."""
    src = digit_runs(source)
    joined = "".join(sorted(src))
    for d in digit_runs(claim):
        bare = d.replace(".", "")
        if len(bare) < 2:
            continue                       # "3 regions", "one of two" are fine
        if d in src or bare in src:
            continue
        if any(bare in s or s in bare for s in src):
            continue
        if bare in joined:
            continue
        return False
    return True


def _strip_name(fname: str, sentence: str) -> str:
    """Drop a leading echo of the filename: 'Deck.pptx - Acme review' -> 'Acme review'."""
    t = (sentence or "").strip()
    if t.lower().startswith(fname.lower()):
        t = t[len(fname):].lstrip(" -\u2013\u2014:,")
    return _clip(t, 30)


def _clip(text: str, words: int) -> str:
    w = (text or "").split()
    return " ".join(w[:words])


async def _gather_limited(jobs: list, limit: int = CONCURRENCY) -> list:
    sem = asyncio.Semaphore(limit)

    async def run(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(run(j) for j in jobs), return_exceptions=True)


# --------------------------------------------------------------------------- #
# 3. numbers (no model)
# --------------------------------------------------------------------------- #
_NUM = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([$£€]\s?)?"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"\s*"
    r"(%|percent|pct|bps|bn|billion|million|mn|thousand|k|x|ppt|pp)?"
    r"(?![A-Za-z0-9])",
    re.I,
)
_SCALE_UNIT = {"bn": "$bn", "billion": "$bn", "million": "$m", "mn": "$m",
               "thousand": "$k", "k": "$k"}


def extract_numbers(chunks) -> list[tuple]:
    """-> rows for the `numbers` table: (file, loc, chunk_id, num, text, unit, sentence)."""
    from .docs import split_sentences

    rows: list[tuple] = []
    seen: set[tuple] = set()
    for c in chunks:
        # slide bullets and table lines carry no full stop, so break on newlines first
        sents = [s2 for line in c.text.splitlines() for s2 in split_sentences(line)]
        for sent in sents:
            if len(sent.split()) < 3:
                continue
            clean = sent.strip()[:400]
            for m in _NUM.finditer(sent):
                cur, num, suf = m.group(1), m.group(2), (m.group(3) or "").lower()
                raw = m.group(0).strip()
                try:
                    val = float(num.replace(",", ""))
                except ValueError:
                    continue
                if suf in ("%", "percent", "pct"):
                    unit = "%"
                elif suf in ("bps",):
                    unit = "bps"
                elif suf in ("ppt", "pp"):
                    unit = "ppt"
                elif suf == "x":
                    unit = "x"
                elif suf in _SCALE_UNIT:
                    unit = _SCALE_UNIT[suf] if cur else _SCALE_UNIT[suf].lstrip("$")
                elif cur:
                    unit = "$"
                else:
                    unit = ""
                key = (c.file, c.loc, raw, clean[:60])
                if key in seen:
                    continue
                seen.add(key)
                rows.append((c.file, c.loc, c.id, val, raw, unit, clean))
    return rows


# --------------------------------------------------------------------------- #
# 4. definitions from formulas (no model for the parse)
# --------------------------------------------------------------------------- #
METRIC_RE = re.compile(
    r"(margin|%|percent|rate|ratio|\bper\b|growth|yield|share|index|avg|average|"
    r"change|delta|cagr|conversion|uplift|multiple|\bx\b|utilisation|utilization|"
    r"productivity|efficiency|attach|mix)", re.I)
_REF = re.compile(r"(?:'([^']+)'|([A-Za-z_][A-Za-z0-9_ ]*))?!?\$?([A-Z]{1,3})\$?(\d{1,7})")
_DIV = re.compile(r"^=?\s*(.+?)\s*/\s*(.+?)\s*$")


def _refs_in(expr: str) -> list[str]:
    return [f"{m.group(3)}{m.group(4)}" for m in _REF.finditer(expr or "")]


def parse_definition(formula: str) -> tuple[str, str] | None:
    """'=B8/B6' -> ('B8','B6'); '=(B6-B7)/B6' -> ('B6-B7' refs, 'B6'). None if not a ratio."""
    body = (formula or "").lstrip("=").strip()
    if "/" not in body:
        return None
    depth, split_at = 0, -1
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "/" and depth == 0:
            split_at = i
            break
    if split_at < 0:
        return None
    num, den = body[:split_at].strip(), body[split_at + 1:].strip()
    nrefs, drefs = _refs_in(num), _refs_in(den)
    if not nrefs or not drefs:
        return None
    return ("+".join(nrefs) if len(nrefs) > 1 else nrefs[0],
            "+".join(drefs) if len(drefs) > 1 else drefs[0])


def extract_definitions(con, file: str) -> list[dict]:
    """Every formula cell whose row label reads like a metric, resolved to its operands."""
    rows = con.execute(
        "SELECT sheet, cell_ref, row_label, header, formula FROM cells "
        "WHERE file=? AND is_formula=1 AND formula IS NOT NULL", (file,)).fetchall()
    label_at: dict[tuple[str, str], str] = {}
    for r in con.execute("SELECT sheet, cell_ref, row_label FROM cells WHERE file=?", (file,)):
        label_at[(r["sheet"], r["cell_ref"])] = r["row_label"] or ""

    out: dict[tuple, dict] = {}
    for r in rows:
        label = (r["row_label"] or "").strip()
        if not label or not METRIC_RE.search(label):
            continue
        parsed = parse_definition(r["formula"])
        if not parsed:
            continue
        nref, dref = parsed
        nlab = " + ".join(label_at.get((r["sheet"], x), x) or x for x in nref.split("+"))
        dlab = " + ".join(label_at.get((r["sheet"], x), x) or x for x in dref.split("+"))
        if not nlab or not dlab or nlab == label or dlab == label:
            continue
        unit = "%" if "%" in label or "percent" in label.lower() else "ratio"
        key = (r["sheet"], label, nlab, dlab)
        if key in out:                       # one row per metric, first cell wins as the example
            continue
        out[key] = {"name": label, "formula": r["formula"], "numerator_ref": nref,
                    "denominator_ref": dref, "numerator_label": nlab,
                    "denominator_label": dlab, "file": file, "sheet": r["sheet"],
                    "cell_ref": r["cell_ref"], "unit": unit, "meaning": ""}
    return list(out.values())


# --------------------------------------------------------------------------- #
# prompts + schemas
# --------------------------------------------------------------------------- #
# Anthropic structured outputs accept `maxLength` on a string but reject `minItems` above 1
# and `maxItems` entirely (both verified, 400 invalid_request_error). Length caps are what
# keeps a reply inside classify()'s 1000-token output ceiling; counts stay in the prompt.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "file": {"type": "string", "maxLength": 200},
        "sections": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"loc": {"type": "string", "maxLength": 90},
                                     "line": {"type": "string", "maxLength": 150}},
                      "required": ["loc", "line"], "additionalProperties": False},
        },
    },
    "required": ["file", "sections"],
    "additionalProperties": False,
}

CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "contexts": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"i": {"type": "integer"},
                                     "context": {"type": "string", "maxLength": 500}},
                      "required": ["i", "context"], "additionalProperties": False},
        }
    },
    "required": ["contexts"],
    "additionalProperties": False,
}

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "qa": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "maxLength": 90},
                    "answer": {"type": "string", "maxLength": 140},
                    "say": {"type": "string", "maxLength": 60},
                    "figure": {"type": "string", "maxLength": 40},
                    "citations": {"type": "array",
                                  "items": {"type": "string", "maxLength": 90}},
                },
                "required": ["question", "answer", "say", "figure", "citations"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa"],
    "additionalProperties": False,
}

NAMES_SCHEMA = {
    "type": "object",
    "properties": {
        "definitions": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"cell_ref": {"type": "string", "maxLength": 12},
                                     "meaning": {"type": "string", "maxLength": 120}},
                      "required": ["cell_ref", "meaning"], "additionalProperties": False},
        }
    },
    "required": ["definitions"],
    "additionalProperties": False,
}


def summary_prompt(fname: str, kind: str, body: str) -> str:
    return (
        "You are indexing one file from a consultant's context folder so a live call copilot "
        "can tell at a glance what is in it.\n\n"
        f"FILE: {fname}  ({kind})\n"
        "--- CONTENT ---\n" + body[:24000] + "\n--- END ---\n\n"
        "Return:\n"
        "  file: ONE sentence, at most 22 words, saying what this file is and what period and "
        "unit it covers. No preamble, no 'This file'.\n"
        "  sections: EXACTLY one entry per bracketed section above, in the same order, no "
        "more and no fewer. For a spreadsheet table describe the table as a whole (what one "
        "row is, what the columns measure, the period and the unit), never individual rows. "
        "using the EXACT loc string given in brackets. `line` is at most 18 words: what it "
        "contains, the key figures, the period, the unit.\n"
        "Only use numbers that appear verbatim above. Never compute anything."
    )


def context_prompt(fname: str, whole: str, batch: list[tuple[int, str]]) -> str:
    parts = "\n\n".join(f"[{i}] {t[:1800]}" for i, t in batch)
    return (
        "Here is a document, then some chunks taken from it. For each chunk write a short "
        "blurb that situates it in the document, so the chunk can be found by search on its "
        "own.\n\n"
        f"DOCUMENT: {fname}\n<document>\n{whole[:40000]}\n</document>\n\n"
        f"CHUNKS\n{parts}\n\n"
        "For each chunk return `i` and `context`: 50 to 100 tokens naming the document, the "
        "section or slide it sits in, the period and entity it is about, and the words a "
        "person would use to ask about it that are not already in the chunk. Do not summarise "
        "the chunk, do not repeat it, and introduce no number that is not in the document."
    )


def qa_prompt(fname: str, loc: str, body: str) -> str:
    return (
        "You are preparing a client-question cache for a live call copilot. The consultant is "
        "presenting the material below and a client may ask about it.\n\n"
        f"SOURCE: {fname} › {loc}\n"
        "--- CONTENT (the only facts you may use) ---\n" + body[:12000] + "\n--- END ---\n\n"
        "Write 3 to 5 questions a client would actually ask out loud about this material. "
        "Spoken phrasing, not written ('so what did Q3 look like', not 'What was the Q3 "
        "revenue figure?'). Cover the obvious figure, the awkward one, and the 'why'.\n\n"
        "For each:\n"
        "  question: the client's words, under 12 words.\n"
        "  answer:   at most 20 words, the direct answer.\n"
        "  say:      at most 8 words the consultant can speak as-is.\n"
        "  figure:   the single key number exactly as written above, with its unit, or \"\".\n"
        "  citations: ONE source string (a second only if the answer truly spans two). "
        "Use the EXACT bracketed citation shown "
        "beside the fact in the content above, e.g. \"Acme_Financials.xlsx › P&L › D6\" or "
        f"\"{fname} › {loc}\".\n\n"
        "Every digit you write must appear verbatim in the content above. Do not add, "
        "subtract, divide or convert anything. If a question cannot be answered from the "
        "content, leave it out.\n"
        "Stay inside every word cap. The whole reply must fit in 800 tokens; prefer four "
        "short entries over five long ones."
    )


def names_prompt(fname: str, defs: list[dict]) -> str:
    listing = "\n".join(
        f"  {d['cell_ref']}: {d['name']} = {d['formula']}  ({d['numerator_label']} / {d['denominator_label']})"
        for d in defs)
    return (
        f"These metrics are defined by formulas in {fname}. For each, say in at most 14 words "
        "what it measures and what it is a share of, so a copilot can explain the metric "
        "without recomputing it.\n\n" + listing +
        "\n\nReturn one entry per cell_ref given, using that exact cell_ref."
    )


# --------------------------------------------------------------------------- #
# unit assembly
# --------------------------------------------------------------------------- #
def doc_units(chunks, file: str) -> list[dict]:
    """One unit per slide / page / section of a document, with its full text."""
    by_unit: dict[int, list] = {}
    for c in chunks:
        if c.file == file:
            by_unit.setdefault(c.unit, []).append(c)
    units = []
    for u in sorted(by_unit):
        cs = sorted(by_unit[u], key=lambda c: c.id)
        loc = re.sub(r"\s+notes$", "", cs[0].loc)
        body = "\n\n".join(f"[{c.file} › {c.loc}]\n{c.text}" for c in cs)
        units.append({"kind": "doc", "file": file, "loc": loc, "body": body,
                      "chunk_ids": [c.id for c in cs]})
    return units


def table_units(con, file: str, max_rows: int = 24) -> list[dict]:
    """One unit per spreadsheet table, rendered with real cell refs so citations are exact."""
    units = []
    per_sheet: dict[str, int] = {}
    for r in con.execute("SELECT sheet FROM _tables WHERE file=?", (file,)):
        per_sheet[r["sheet"] or ""] = per_sheet.get(r["sheet"] or "", 0) + 1
    for t in con.execute("SELECT * FROM _tables WHERE file=? ORDER BY sheet, first_data_row",
                         (file,)):
        cols = list(con.execute(
            "SELECT name, label, letter, type FROM _columns WHERE table_name=? ORDER BY ord",
            (t["name"],)))
        if not cols:
            continue
        rows = list(con.execute(f'SELECT * FROM "{t["name"]}" LIMIT {max_rows}'))
        head = " | ".join(f'{c["label"]} ({c["letter"]})' for c in cols)
        lines = [f'TABLE {t["name"]} from {file} › {t["sheet"]}'
                 f'{"  -- " + t["title"] if t["title"] else ""}',
                 f'columns: {head}']
        for r in rows:
            src = r["_row"]
            cells = []
            for c in cols:
                v = r[c["name"]]
                if v is None:
                    continue
                txt = f"{v:,.4g}" if isinstance(v, float) and v != int(v) else (
                    f"{int(v):,}" if isinstance(v, float) else str(v))
                cells.append(f'{c["label"]}={txt} [{file} › {t["sheet"]} › {c["letter"]}{src}]')
            if cells:
                lines.append("  " + "; ".join(cells))
        n = con.execute(f'SELECT COUNT(*) n FROM "{t["name"]}"').fetchone()["n"]
        if n > max_rows:
            lines.append(f"  ... {n - max_rows} more rows")
        sheet = t["sheet"] or ""
        loc = t["name"] if not sheet else (
            f'{sheet} ({t["name"]})' if per_sheet.get(sheet, 1) > 1 else sheet)
        units.append({"kind": "table", "file": file, "loc": loc,
                      "body": "\n".join(lines), "table": t["name"]})
    return units


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #
class DeepCache:
    """Everything the deep pass produces, keyed by file content hash."""

    FILES = ("manifest", "summaries", "contexts", "qa", "numbers", "definitions",
             "scratch")

    def __init__(self, root: Path):
        self.root = root
        self.manifest: dict = {}
        self.summaries: dict = {}
        self.contexts: dict = {}
        self.qa: dict = {}
        self.numbers: dict = {}
        self.definitions: dict = {}
        self.scratch: dict = {}

    def load(self) -> "DeepCache":
        for name in self.FILES:
            p = self.root / f"{name}.json"
            if p.exists():
                try:
                    setattr(self, name, json.loads(p.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    setattr(self, name, {})
        if (self.manifest or {}).get("v") != CACHE_VERSION:
            for name in ("summaries", "contexts", "qa", "numbers", "definitions",
                         "scratch"):
                setattr(self, name, {})
            self.manifest = {"v": CACHE_VERSION}
        return self

    def save(self, *names: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in (names or self.FILES):
            tmp = self.root / f"{name}.json.tmp"
            tmp.write_text(json.dumps(getattr(self, name), separators=(",", ":")),
                           encoding="utf-8")
            tmp.replace(self.root / f"{name}.json")

    def prune(self, live_hashes: set[str]) -> None:
        for name in ("summaries", "contexts", "qa", "numbers", "definitions", "scratch"):
            d = getattr(self, name)
            for h in [k for k in d if k.split(":")[0] not in live_hashes]:
                d.pop(h, None)


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:60] or "unit"


def plan_units(ix, cache: DeepCache, hashes: dict[str, str], force: bool) -> list[dict]:
    """Every piece of work not already in the cache. Each entry is one model call."""
    units: list[dict] = []
    con = ix.con
    tries = (cache.manifest or {}).get("attempts") or {}

    def todo(key: str) -> bool:
        if force:
            return True
        return tries.get(key, 0) < MAX_ATTEMPTS
    for rel, h in sorted(hashes.items()):
        is_doc = rel in ix._chunks_by_file
        # 1. summary, batched so a 40-slide deck does not overflow one response
        secs = _section_locs(ix, rel)
        for b in range(0, max(1, len(secs)), SUM_BATCH):
            key = h if b == 0 else f"{h}:s{b}"
            if (force or key not in cache.summaries) and todo(f"sum:{key}"):
                units.append({"kind": "summary", "file": rel, "hash": h, "key": key,
                              "secs": secs[b:b + SUM_BATCH], "first": b == 0})
        # 2. contexts, batched
        if is_doc:
            chunks = ix._chunks_by_file.get(rel, [])
            missing = [i for i in range(len(chunks))
                       if force or f"{h}:{i}" not in cache.contexts]
            for s in range(0, len(missing), CTX_BATCH):
                units.append({"kind": "context", "file": rel, "hash": h,
                              "ords": missing[s:s + CTX_BATCH]})
        # 3+4. numbers and definitions are CPU work, one unit per file
        if force or h not in cache.numbers:
            units.append({"kind": "numbers", "file": rel, "hash": h, "key": h})
        if force or h not in cache.definitions:
            units.append({"kind": "definitions", "file": rel, "hash": h, "key": h})
        # 6. scratch paper: levers + sensitivity grid, one unit per workbook
        cache.scratch = {k: v for k, v in cache.scratch.items()
                         if str(k).endswith(f":{_scratch.SCHEMA}")}
        if not is_doc and Path(rel).suffix.lower() in SCRATCH_EXT:
            # the schema is in the key: a lever table cached by an older shape of
            # scratch.py (no text levers, no combinations) has to be built again
            skey = f"{h}:{_scratch.SCHEMA}"
            if (force or skey not in cache.scratch) and todo(f"lev:{skey}"):
                units.append({"kind": "levers", "file": rel, "hash": h, "key": skey})
        # 5. Q&A per slide / section / table
        us = doc_units(ix.chunks, rel) if is_doc else (table_units(con, rel) if con else [])
        for i, u in enumerate(us[:QA_MAX_UNITS]):
            key = f"{h}:{i}:{_slug(u['loc'])}"
            if (force or key not in cache.qa) and todo(f"qa:{key}"):
                units.append({"kind": "qa", "file": rel, "hash": h, "key": key, "unit": u})
    return units


def estimate(ix, units: list[dict]) -> tuple[dict, int, int]:
    """(per-file estimated input tokens, total in, total out) before spending anything."""
    by_file: dict[str, int] = {}
    tin = tout = 0
    for u in units:
        if u["kind"] == "summary":
            body = _summary_body(ix, u["file"], only=u.get("secs"))
            n, o = approx_tokens(body) + SCHEMA_TOKENS, 100 + 40 * max(1, len(u.get("secs") or []))
        elif u["kind"] == "context":
            whole = _whole_doc(ix, u["file"])
            n = approx_tokens(whole) + 140 * len(u["ords"]) + SCHEMA_TOKENS
            o = 145 * len(u["ords"])
        elif u["kind"] == "qa":
            n, o = approx_tokens(u["unit"]["body"]) + SCHEMA_TOKENS, 800
        elif u["kind"] == "levers":
            n, o = 1200, 700          # ASSUMED: one aliases call over <=40 levers
        else:
            n = o = 0
        by_file[u["file"]] = by_file.get(u["file"], 0) + n + o
        tin += n
        tout += o
    return by_file, tin, tout


_WHOLE: dict[tuple, str] = {}


def _whole_doc(ix, rel: str) -> str:
    key = (id(ix), ix.version, rel)
    if key not in _WHOLE:
        _WHOLE.clear() if len(_WHOLE) > 64 else None
        _WHOLE[key] = "\n\n".join(c.text for c in ix.chunks if c.file == rel)[:40000]
    return _WHOLE[key]


def _summary_body(ix, rel: str, only: list[str] | None = None) -> str:
    """What the summariser sees: sheets with their columns, or sections with their text."""
    us = doc_units(ix.chunks, rel) if rel in ix._chunks_by_file else (
        table_units(ix.con, rel, max_rows=8) if ix.con is not None else [])
    if only is not None:
        keep = set(only)
        us = [u for u in us if u["loc"] in keep]
    return "\n\n".join(f"[{u['loc']}]\n" + _clip(u["body"], 220) for u in us)


def _file_kind(ix, rel: str) -> str:
    return "document" if rel in ix._chunks_by_file else "spreadsheet"


def _section_locs(ix, rel: str) -> list[str]:
    us = doc_units(ix.chunks, rel) if rel in ix._chunks_by_file else (
        table_units(ix.con, rel, max_rows=1) if ix.con is not None else [])
    return [u["loc"] for u in us]


async def _do_summary(ix, llm_json, u, model) -> dict:
    body = _summary_body(ix, u["file"], only=u.get("secs"))
    if not body.strip():
        return {"file": "", "sections": []}
    return await llm_json(summary_prompt(u["file"], _file_kind(ix, u["file"]), body),
                          SUMMARY_SCHEMA, model)


async def _do_context(ix, llm_json, u, model) -> dict:
    rel = u["file"]
    chunks = ix._chunks_by_file.get(rel, [])
    batch = [(i, chunks[i]["text"]) for i in u["ords"] if i < len(chunks)]
    if not batch:
        return {"contexts": []}
    return await llm_json(context_prompt(rel, _whole_doc(ix, rel), batch),
                          CONTEXT_SCHEMA, model)


def _do_levers(ix, u: dict) -> dict:
    """Compile the workbook, find its levers, run the grid. Pure CPU: runs in a thread.

    It opens its own read-only connection rather than borrowing `ix.con`, which the event
    loop is using at the same time, and hands the compiled graph back to the Index so the
    first live what-if does not pay for the compile again.
    """
    from . import store as _store

    con = _store.connect_ro(ix.folder)
    try:
        graph = ix._graph_for(u["file"], u["hash"])
        rec, graph = _scratch.build(con, ix.folder / u["file"], u["file"], u["hash"], graph)
    finally:
        con.close()
    ix._graphs[u["file"]] = graph
    ix._scratch[u["file"]] = rec
    return rec


async def _do_qa(llm_json, u, model) -> dict:
    return await llm_json(qa_prompt(u["file"], u["unit"]["loc"], u["unit"]["body"]),
                          QA_SCHEMA, model)


def _qa_rows(raw: dict, unit: dict) -> tuple[list[dict], int]:
    """Validate the model's Q&A against the source text. Returns (kept, rejected)."""
    kept, rejected = [], 0
    body = unit["body"]
    for e in (raw or {}).get("qa", []) or []:
        q = (e.get("question") or "").strip()
        a = (e.get("answer") or "").strip()
        if not q or not a:
            continue
        fig = (e.get("figure") or "").strip()
        say = _clip((e.get("say") or "").strip(), 8)
        if not numbers_are_grounded(a + " " + fig + " " + say, body):
            rejected += 1
            continue
        cites = []
        for c in (e.get("citations") or [])[:4]:
            c = str(c).strip()
            if not c:
                continue
            if "›" in c:
                f, _, loc = c.partition("›")
                cites.append({"file": f.strip(), "loc": loc.strip()})
            else:
                cites.append({"file": unit["file"], "loc": c})
        if not cites:
            cites = [{"file": unit["file"], "loc": unit["loc"]}]
        kept.append({"question": q, "answer": _clip(a, 25), "say": say, "figure": fig,
                     "citations": cites, "file": unit["file"], "loc": unit["loc"]})
    return kept, rejected


async def run_deep(ix, llm_json: LLMJson, on_progress=None, model_fast=None,
                   model_good=None, force: bool = False) -> DeepStats:
    st = DeepStats(state="estimating", started=time.time())
    t0 = time.perf_counter()

    def emit() -> None:
        if on_progress:
            try:
                on_progress(st.as_dict())
            except Exception:                      # a bad callback must not kill the pass
                pass

    if ix.folder is None or ix.con is None:
        st.state, st.error = "error", "index is not built"
        emit()
        return st

    root = ix._dir / DEEP_DIR
    cache = DeepCache(root).load()
    hashes = hashes_for(ix, cache)
    cache.prune(set(hashes.values()))
    st.files = len(hashes)

    units = plan_units(ix, cache, hashes, force)
    st.est_by_file, st.est_in_tokens, st.est_out_tokens = estimate(ix, units)
    st.total = len(units)
    st.cached_units = len(cache.qa) + len(cache.summaries) + len(cache.contexts)
    emit()

    st.state = "running"
    emit()
    if not units:
        try:
            _apply(ix, cache, st)
        except Exception as e:
            st.error = f"apply: {type(e).__name__}: {e}"
        st.state, st.finished = ("error" if st.error else "done"), time.time()
        st.wall_s = time.perf_counter() - t0
        emit()
        return st

    last_save = [0.0]
    lock = asyncio.Lock()

    def touch(name: str) -> None:
        now = time.perf_counter()
        if now - last_save[0] > 1.0:
            last_save[0] = now
            cache.save(name, "manifest")

    async def one(u: dict) -> None:
        kind = u["kind"]
        try:
            if kind == "summary":
                raw = await _do_summary(ix, llm_json, u, model_fast)
                st.in_tokens += (raw.get("_usage") or {}).get("in", 0)
                st.out_tokens += (raw.get("_usage") or {}).get("out", 0)
                st.llm_calls += 1
                if not (raw.get("file") or raw.get("sections")):
                    raise ValueError("empty summary response (truncated? pass model_fast)")
                sent = u.get("secs") or []
                got = (raw.get("sections") or [])
                got = got[:len(sent)] if sent else got[:SUM_BATCH]
                secs = []
                for j, sec in enumerate(got):
                    loc = sent[j] if j < len(sent) else str(sec.get("loc", ""))[:80]
                    secs.append({"loc": loc, "line": _clip(str(sec.get("line", "")), 24)})
                cache.summaries[u["key"]] = {
                    "name": u["file"],
                    "file": _strip_name(u["file"], str(raw.get("file", ""))) if u.get("first", True) else "",
                    "sections": secs,
                }
                async with lock:
                    touch("summaries")
            elif kind == "context":
                raw = await _do_context(ix, llm_json, u, model_fast)
                st.in_tokens += (raw.get("_usage") or {}).get("in", 0)
                st.out_tokens += (raw.get("_usage") or {}).get("out", 0)
                st.llm_calls += 1
                whole = _whole_doc(ix, u["file"])
                for e in (raw.get("contexts") or []):
                    i = e.get("i")
                    ctx = _clip(str(e.get("context") or ""), 120)
                    if not isinstance(i, int) or i not in u["ords"] or not ctx:
                        continue
                    if not numbers_are_grounded(ctx, whole):
                        ctx = re.sub(r"\d[\d,\.]*", "", ctx)   # keep the words, drop invented digits
                    cache.contexts[f'{u["hash"]}:{i}'] = ctx
                async with lock:
                    touch("contexts")
            elif kind == "numbers":
                rows = extract_numbers([c for c in ix.chunks if c.file == u["file"]])
                cache.numbers[u["key"]] = {"file": u["file"], "rows": rows}
                async with lock:
                    touch("numbers")
            elif kind == "definitions":
                defs = extract_definitions(ix.con, u["file"])
                if defs:
                    try:
                        raw = await llm_json(names_prompt(u["file"], defs), NAMES_SCHEMA, model_fast)
                        st.llm_calls += 1
                        st.in_tokens += (raw.get("_usage") or {}).get("in", 0)
                        st.out_tokens += (raw.get("_usage") or {}).get("out", 0)
                        meaning = {str(d.get("cell_ref")): _clip(str(d.get("meaning", "")), 16)
                                   for d in (raw.get("definitions") or [])}
                        for d in defs:
                            d["meaning"] = meaning.get(d["cell_ref"], "")
                    except Exception:                 # the parse is the product; the name is a bonus
                        pass
                cache.definitions[u["key"]] = {"file": u["file"], "rows": defs}
                async with lock:
                    touch("definitions")
            elif kind == "levers":
                rec = await asyncio.to_thread(_do_levers, ix, u)
                if rec.get("levers"):
                    try:                          # the table has to work without aliases
                        raw = await llm_json(
                            _scratch.aliases_prompt(u["file"], rec["levers"]),
                            _scratch.ALIASES_SCHEMA, model_fast)
                        st.llm_calls += 1
                        st.in_tokens += (raw.get("_usage") or {}).get("in", 0)
                        st.out_tokens += (raw.get("_usage") or {}).get("out", 0)
                        _scratch.apply_aliases(rec["levers"], raw)
                    except Exception as e:
                        rec["aliases_error"] = f"{type(e).__name__}: {e}"
                cache.scratch[u["key"]] = rec
                async with lock:
                    touch("scratch")
            elif kind == "qa":
                raw = await _do_qa(llm_json, u, model_good)
                st.in_tokens += (raw.get("_usage") or {}).get("in", 0)
                st.out_tokens += (raw.get("_usage") or {}).get("out", 0)
                st.llm_calls += 1
                if "qa" not in (raw or {}):
                    raise ValueError("empty qa response (truncated? pass model_good)")
                kept, rejected = _qa_rows(raw, u["unit"])
                st.qa_rejected += rejected
                cache.qa[u["key"]] = kept
                async with lock:
                    touch("qa")
        except Exception as e:                        # one bad unit must not kill the pass
            st.error = f"{u['kind']} {u['file']}: {type(e).__name__}: {e}"
            st.failed += 1
            if kind in ("summary", "qa"):
                att = cache.manifest.setdefault("attempts", {})
                k = f"{kind if kind == 'qa' else 'sum'}:{u['key']}"
                att[k] = att.get(k, 0) + 1
                if att[k] >= MAX_ATTEMPTS:            # give up, cache empty, stay fast
                    (cache.qa if kind == "qa" else cache.summaries)[u["key"]] = (
                        [] if kind == "qa" else {"name": u["file"], "file": "", "sections": []})
        else:
            if kind in ("summary", "qa"):
                (cache.manifest.get("attempts") or {}).pop(
                    f"{kind if kind == 'qa' else 'sum'}:{u['key']}", None)
        finally:
            st.done += 1
            emit()

    await _gather_limited([one(u) for u in units])
    cache.save()
    applied = True
    try:
        _apply(ix, cache, st)
    except Exception as e:
        applied = False
        st.error = f"apply: {type(e).__name__}: {e}"
    st.state = "error" if (not applied or st.failed >= st.total) else "done"
    st.finished = time.time()
    st.wall_s = time.perf_counter() - t0
    emit()
    return st


# --------------------------------------------------------------------------- #
# applying the cache to a live Index (no model calls)
# --------------------------------------------------------------------------- #
def hashes_for(ix, cache: DeepCache, con=None) -> dict[str, str]:
    """Content hash per indexable file, reusing the cached one when mtime+size match.

    `con` must be passed from inside build(), where ix.con is None for the duration: without
    it every spreadsheet drops out of the manifest and its cached tables and Q&A are treated
    as dead on the next pass.
    """
    sig = (cache.manifest or {}).get("sig") or {}
    old = (cache.manifest or {}).get("files") or {}
    out, newsig = {}, {}
    rels = set(ix._chunks_by_file)
    con = con if con is not None else ix.con
    if con is not None:
        rels |= {r["file"] for r in con.execute("SELECT DISTINCT file FROM _tables")}
    for rel in sorted(rels):
        p = ix.folder / rel
        if not p.exists():
            continue
        s = p.stat()
        want = [s.st_mtime, s.st_size]
        out[rel] = old[rel] if (sig.get(rel) == want and rel in old) else file_hash(p)
        newsig[rel] = want
    cache.manifest = {"v": CACHE_VERSION, "files": out, "sig": newsig, "at": time.time(),
                      "attempts": (cache.manifest or {}).get("attempts") or {}}
    return out


def attach_contexts(ix, cache: DeepCache, hashes: dict[str, str]) -> int:
    """Put the cached blurb back on every chunk. Cheap; no model call.

    Index._renumber assigns global ids by walking sorted(_chunks_by_file), so a chunk's
    global id is its file's offset plus its ordinal. No searching.
    """
    n, offset = 0, 0
    for rel in sorted(ix._chunks_by_file):
        recs = ix._chunks_by_file[rel]
        h = hashes.get(rel)
        for i, rec in enumerate(recs):
            ctx = cache.contexts.get(f"{h}:{i}") if h else None
            if ctx:
                rec["context"] = ctx
                n += 1
            else:
                ctx = rec.get("context", "")      # what chunks.json already carries
            gid = offset + i
            if 0 <= gid < len(ix.chunks):
                ix.chunks[gid].context = ctx or ""
        offset += len(recs)
    return n


def apply_sql(con_rw, cache: DeepCache, hashes: dict[str, str]) -> tuple[int, int]:
    """Replace the `numbers` and `definitions` tables from the cache. -> (numbers, defs)."""
    live = set(hashes.values())
    con_rw.execute("DELETE FROM numbers")
    con_rw.execute("DELETE FROM definitions")
    nn = nd = 0
    for h, rec in cache.numbers.items():
        if h not in live:
            continue
        rows = [tuple(r) for r in rec.get("rows", [])]
        if rows:
            con_rw.executemany("INSERT INTO numbers VALUES (?,?,?,?,?,?,?)", rows)
            nn += len(rows)
    for h, rec in cache.definitions.items():
        if h not in live:
            continue
        for d in rec.get("rows", []):
            con_rw.execute(
                "INSERT INTO definitions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (d["name"], d["formula"], d["numerator_ref"], d["denominator_ref"],
                 d["numerator_label"], d["denominator_label"], d["file"], d["sheet"],
                 d["cell_ref"], d["unit"], d.get("meaning", "")))
            nd += 1
    return nn, nd


def load_scratch(cache: DeepCache, hashes: dict[str, str]) -> dict:
    """The live scratch records, keyed by file name rather than by hash:schema."""
    live = {h: rel for rel, h in hashes.items()}
    out: dict[str, dict] = {}
    for key, rec in cache.scratch.items():
        rel = live.get(str(key).split(":")[0])
        if rel and rec.get("schema") == _scratch.SCHEMA:
            out[rel] = dict(rec, file=rel) if rec.get("file") != rel else rec   # the live name, not the one at build time
    return out


def load_qa(ix, cache: DeepCache, hashes: dict[str, str]) -> list[QA]:
    """Flatten the per-unit Q&A into one list and embed the questions for qa_matches().

    The model-written entries are kept exactly as they were cached; the grid points from
    `scratch.json` are turned into derived entries and appended, so the same instant path
    answers "what if we charged 1%" without a recalculation.
    """
    live = set(hashes.values())
    rel_of = {h: rel for rel, h in hashes.items()}     # the cache is keyed by content hash: a file
    out: list[QA] = []                                   # renamed since keeps its entries under the new name
    for key, rows in sorted(cache.qa.items()):
        h = key.split(":")[0]
        if h not in live:
            continue
        rel = rel_of.get(h)
        for r in rows:
            if rel and r.get("file") and r.get("file") != rel:
                r = dict(r, file=rel, citations=[dict(c, file=rel) if c.get("file") == r.get("file") else c
                                                 for c in r.get("citations", [])])
            out.append(QA(question=r["question"], answer=r["answer"], say=r.get("say", ""),
                          figure=r.get("figure", ""), citations=r.get("citations", []),
                          file=r.get("file", ""), loc=r.get("loc", "")))
    for rel, rec in sorted(load_scratch(cache, hashes).items()):
        try:
            rows = _scratch.derived_qa(rec)
        except Exception:                      # a broken scratch record must not lose the Q&A
            continue
        for r in rows:
            out.append(QA(question=r["question"], answer=r["answer"], say=r.get("say", ""),
                          figure=r.get("figure", ""), citations=r.get("citations", []),
                          file=r.get("file", ""), loc=r.get("loc", ""),
                          derived=True, working=r.get("working", ""),
                          lever_value=r.get("lever_value")))
    return out


def embed_questions(questions: list[str]):
    """model2vec vectors for qa_matches. Loaded even when PRESAGE_DENSE is off."""
    if not questions:
        return None
    from . import search as _search
    if not _search.Dense.available(allow_download=False):
        return None
    import numpy as np
    v = _search.Dense().model().encode(questions).astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def summaries_text(cache: DeepCache, hashes: dict[str, str],
                   budget_tokens: int = SUMMARY_TOKEN_BUDGET) -> str:
    """One line per file plus per-section lines, trimmed to fit the prompt budget.

    File lines always survive; section lines are dropped from the end until it fits, with a
    count of what was left out so the model knows to use search_text instead of guessing.
    """
    live = {h: rel for rel, h in hashes.items()}
    merged: dict[str, dict] = {}
    for key in sorted(cache.summaries):
        h = key.split(":")[0]
        rel = live.get(h)
        if rel is None:
            continue
        rec = cache.summaries[key]
        m = merged.setdefault(rel, {"file": "", "sections": []})
        if rec.get("file") and not m["file"]:
            m["file"] = rec["file"]
        m["sections"].extend(rec.get("sections", []))
    if not merged:
        return ""

    head = "== WHAT IS IN EACH FILE =="
    budget = budget_tokens * 4 - 60     # room for the trailing "(+N more)" line
    used = len(head) + 1
    file_line = {}
    for rel, m in sorted(merged.items()):
        file_line[rel] = f'{rel} — {m["file"]}' if m["file"] else rel
        used += len(file_line[rel]) + 1

    kept: dict[str, list[str]] = {}
    dropped = 0
    for rel, m in sorted(merged.items()):
        for sec in m["sections"]:
            loc, line = str(sec.get("loc", "")).strip(), str(sec.get("line", "")).strip()
            if not loc or not line:
                continue
            text = f"    {loc} — {line}"
            if used + len(text) + 1 > budget:
                dropped += 1
                continue
            kept.setdefault(rel, []).append(text)
            used += len(text) + 1

    out = [head]
    for rel in sorted(merged):
        out.append(file_line[rel])
        out.extend(kept.get(rel, []))
    if dropped:
        out.append(f"    (+{dropped} more sections; use search_text or read)")
    return "\n".join(out)


def definitions_block(con, limit: int = 20) -> str:
    """The short Definitions block that goes into schema_summary()."""
    try:
        rows = list(con.execute(
            "SELECT name, formula, numerator_label, denominator_label, sheet, cell_ref, "
            "unit, meaning FROM definitions ORDER BY file, sheet, cell_ref LIMIT ?", (limit,)))
    except Exception:
        return ""
    if not rows:
        return ""
    L = ["", "== DEFINITIONS (what a formula-driven metric divides by) =="]
    for r in rows:
        line = (f'  {r["name"]} ({r["sheet"]} {r["cell_ref"]}) = {r["formula"]}  '
                f'-> {r["numerator_label"]} / {r["denominator_label"]}')
        if r["meaning"]:
            line += f'  ({r["meaning"]})'
        L.append(line)
    L.append("  Read both operands from the cells and report them; never compute the ratio "
             "yourself if the metric cell is NULL.")
    return "\n".join(L)


def apply_cached(ix, cache: DeepCache | None = None, rebuild_search: bool = True) -> dict:
    """Push everything the cache holds into a live Index. No model calls.

    This closes and reopens the read-only connection. It runs on the event loop with no
    await between the two, so a concurrent `ix.sql()` on the same loop cannot land in the
    gap. Do not call it from a worker thread while the brain is querying.
    """
    from . import search as _search
    from . import store as _store

    root = ix._dir / DEEP_DIR
    cache = cache or DeepCache(root).load()
    hashes = hashes_for(ix, cache)
    n_ctx = attach_contexts(ix, cache, hashes)

    con = _store.connect_rw(ix.folder)
    try:
        nn, nd = apply_sql(con, cache, hashes)
        con.commit()
    finally:
        con.close()
    if ix.con is not None:
        ix.con.close()
    ix.con = _store.connect_ro(ix.folder)

    ix._scratch = load_scratch(cache, hashes)
    ix._qa = load_qa(ix, cache, hashes)
    ix._qa_vecs = embed_questions([q.question for q in ix._qa])
    ix._summaries = summaries_text(cache, hashes)

    if rebuild_search and n_ctx:
        ix.bm25 = _search.Bm25.build([c.indexed() for c in ix.chunks])
        _search.save_json(ix._dir / "bm25.json", ix.bm25.to_json())
        if ix.dense is not None:
            try:
                ix.dense = _search.Dense().build([c.indexed() for c in ix.chunks])
                ix.dense.save(ix._dir / "dense.npy")
            except Exception:
                pass
        _search.save_json(ix._dir / "chunks.json", ix._chunks_by_file)
    cache.save("manifest")
    if ix._qa_vecs is not None:
        import numpy as np
        np.save(root / "qa_vecs.npy", ix._qa_vecs)
    return {"contexts": n_ctx, "numbers": nn, "definitions": nd, "qa": len(ix._qa),
            "summaries": len([1 for h in cache.summaries if h in set(hashes.values())]),
            "levers": sum(len(r.get("levers", [])) for r in ix._scratch.values()),
            "grid": sum(len(r.get("grid", [])) for r in ix._scratch.values())}


def _apply(ix, cache: DeepCache, st: DeepStats) -> None:
    got = apply_cached(ix, cache)
    st.chunks_contextualized = got["contexts"]
    st.numbers = got["numbers"]
    st.definitions = got["definitions"]
    st.qa = got["qa"]
    st.summaries = got["summaries"]
    st.levers = got.get("levers", 0)
    st.grid = got.get("grid", 0)


def apply_on_build(ix, con_rw) -> dict:
    """Called from the middle of the fast build(), with its rw connection already open.

    Puts back everything the deep pass produced for these exact file bytes. No model call,
    no network: JSON reads plus a few thousand inserts. Files whose bytes changed simply
    have no cache entry and come back empty until the next deep_build.
    """
    cache = DeepCache(ix._dir / DEEP_DIR).load()
    hashes = hashes_for(ix, cache, con_rw)      # ix.con is None inside build()
    n_ctx = attach_contexts(ix, cache, hashes)
    nn, nd = apply_sql(con_rw, cache, hashes)
    ix._scratch = load_scratch(cache, hashes)
    ix._qa = load_qa(ix, cache, hashes)
    ix._qa_vecs = _load_or_embed(ix, cache)
    ix._summaries = summaries_text(cache, hashes)
    if cache.summaries or cache.contexts or cache.qa:
        cache.save("manifest")
    return {"contexts": n_ctx, "numbers": nn, "definitions": nd, "qa": len(ix._qa)}


def load_light(ix) -> dict:
    """Called from Index.open(). Chunk contexts already ride in chunks.json and the numbers
    and definitions tables are already in index.db, so this only rebuilds the in-memory
    Q&A list, its vectors and the summaries block."""
    root = ix._dir / DEEP_DIR
    if not root.exists():
        return {}
    cache = DeepCache(root).load()
    hashes = (cache.manifest or {}).get("files") or {}
    ix._scratch = load_scratch(cache, hashes)
    ix._qa = load_qa(ix, cache, hashes)
    ix._qa_vecs = _load_or_embed(ix, cache)
    ix._summaries = summaries_text(cache, hashes)
    return {"qa": len(ix._qa)}


def _load_or_embed(ix, cache: DeepCache):
    """Reuse qa_vecs.npy when it matches the question list; otherwise embed (0.3 ms each)."""
    if not ix._qa:
        return None
    p = ix._dir / DEEP_DIR / "qa_vecs.npy"
    if p.exists():
        try:
            import numpy as np
            v = np.load(p)
            if len(v) == len(ix._qa):
                return v
        except Exception:
            pass
    v = embed_questions([q.question for q in ix._qa])
    if v is not None:
        try:
            import numpy as np
            (ix._dir / DEEP_DIR).mkdir(parents=True, exist_ok=True)
            np.save(p, v)
        except Exception:
            pass
    return v
