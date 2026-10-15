"""presage.ingest - one folder of context files -> SQL tables, chunks, topics.

    from presage.ingest import Index
    ix = Index()
    ix.build(Path("sample_context"))
    ix.sql("SELECT fy25_actual_q3 FROM p_and_l WHERE line_item='Revenue'")
    ix.search_text("enterprise churn")
    ix.match_topics("so what did revenue look like in q3")

Spreadsheets go to SQLite and are only ever read with SQL: no vector search over
rows, so a number in an answer is a number that was in a cell. Documents go to
plain chunks with a printable citation.

Everything persists under <folder>/.presage/ and rebuilds incrementally by mtime.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import replace, asdict, dataclass, field
from pathlib import Path

from . import deep as _deep
from . import docs as _docs
from . import scratch as _scratch
from . import search as _search
from . import store as _store
from . import tables as _tables
from .deep import QA, DeepStats
from .docs import Chunk
from .topics import TopicHit, TopicIndex, candidate_terms

__all__ = ["Index", "Stats", "Chunk", "TopicHit", "QA", "DeepStats"]

log = logging.getLogger(__name__)

SKIP_PREFIXES = ("ANSWER_KEY", ".", "~$")
MAX_ROW_LABEL_ROWS = 100      # do not turn a 200-row export's id column into topics
MAX_DISTINCT_VALUES = 20      # segment / region columns become topics, names do not
SUMMARY_TOKEN_BUDGET = 6000


@dataclass
class Stats:
    folder: str = ""
    files: int = 0
    spreadsheets: int = 0
    documents: int = 0
    tables: int = 0
    table_rows: int = 0
    cells: int = 0
    null_formula_cells: int = 0
    chunks: int = 0
    topics: int = 0
    dense: bool = False
    rebuilt: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    build_ms: float = 0.0

    def __str__(self) -> str:
        return (f"{self.files} files ({self.spreadsheets} spreadsheets, {self.documents} documents) -> "
                f"{self.tables} tables / {self.table_rows} rows, {self.cells} cells "
                f"({self.null_formula_cells} uncached formulas), {self.chunks} chunks, "
                f"{self.topics} topics, dense={'on' if self.dense else 'off'}, "
                f"built in {self.build_ms:.0f} ms")


def _is_indexable(p: Path) -> bool:
    if not p.is_file() or p.name.startswith(SKIP_PREFIXES):
        return False
    ext = p.suffix.lower()
    return ext in _tables.SHEET_EXT or ext in _tables.CSV_EXT or ext in _docs.DOC_EXT


class Index:
    def __init__(self, folder: Path | str | None = None):
        self.folder: Path | None = None
        self.con = None
        self.version = 0          # bumped on every build so callers can cache derived prompts
        self._chunks_by_file: dict[str, list[dict]] = {}
        self.chunks: list[Chunk] = []
        self.bm25 = _search.Bm25()
        self.dense: _search.Dense | None = None
        self.topics = TopicIndex()
        self._stats = Stats()
        # deep pass (presage/ingest/deep.py); all optional, all cached on disk
        self._qa: list[QA] = []
        self._qa_vecs = None
        self._summaries: str = ""
        self._deep = DeepStats()
        # scratch paper (presage/ingest/scratch.py): one compiled model per workbook, kept
        # in memory, plus the cached lever table and sensitivity grid per file name.
        self._graphs: dict[str, _scratch.ModelGraph] = {}
        self._scratch: dict[str, dict] = {}
        self._hashes: dict[str, tuple] = {}
        if folder is not None:
            self.open(Path(folder))

    # ------------------------------------------------------------------ io
    @property
    def _dir(self) -> Path:
        return self.folder / _store.INDEX_DIR

    def open(self, folder: Path) -> "Index":
        """Attach to an already-built index without re-reading source files."""
        self.folder = Path(folder).resolve()
        if not _store.db_path(self.folder).exists():
            return self
        self.con = _store.connect_ro(self.folder)
        self._chunks_by_file = _search.load_json(self._dir / "chunks.json", {}) or {}
        self._renumber()
        self.bm25 = _search.Bm25(_search.load_json(self._dir / "bm25.json", {}))
        self.topics = TopicIndex(_search.load_json(self._dir / "topics.json", {}))
        meta = _search.load_json(self._dir / "meta.json", {}) or {}
        if meta.get("dense"):
            self.dense = _search.Dense().load(self._dir / "dense.npy")
            self.dense.model()          # warm now, not on the first question of a call
        if meta.get("stats"):
            self._stats = Stats(**meta["stats"])
        try:
            _deep.load_light(self)        # qa + summaries; contexts already ride in chunks.json
        except Exception:                 # a broken deep cache must never stop the fast path
            pass
        return self

    def _renumber(self) -> None:
        self.chunks = []
        for rel in sorted(self._chunks_by_file):
            for d in self._chunks_by_file[rel]:
                c = Chunk(**d)
                c.id = len(self.chunks)
                self.chunks.append(c)

    # --------------------------------------------------------------- build
    def build(self, folder: Path | str | None = None, force: bool = False) -> Stats:
        t0 = time.perf_counter()
        if folder is not None:
            self.folder = Path(folder).resolve()
        if self.folder is None:
            raise ValueError("no folder given")
        if self.con is not None:
            self.con.close()
            self.con = None
        self._dir.mkdir(parents=True, exist_ok=True)
        con = _store.connect_rw(self.folder)

        seen = {p.name: p for p in sorted(self.folder.iterdir()) if _is_indexable(p)}
        known = {r["path"]: (r["mtime"], r["size"])
                 for r in con.execute("SELECT path, mtime, size FROM _files").fetchall()}
        self._chunks_by_file = _search.load_json(self._dir / "chunks.json", {}) or {}

        for gone in set(known) - set(seen):
            _tables.drop_file_tables(con, gone)
            con.execute("DELETE FROM _documents WHERE file=?", (gone,))
            con.execute("DELETE FROM _files WHERE path=?", (gone,))
            self._chunks_by_file.pop(gone, None)

        stats = Stats(folder=str(self.folder))
        errors: dict[str, str] = {}
        taken = {r[0] for r in con.execute("SELECT name FROM _tables").fetchall()}
        meta_now = _search.load_json(self._dir / "meta.json", {}) or {}
        same_ingest = meta_now.get("ingest") == _tables.INGEST_VERSION
        for name, path in seen.items():
            st = path.stat()
            fresh = known.get(name) == (st.st_mtime, st.st_size) and same_ingest
            if fresh and not force:
                continue
            stats.rebuilt.append(name)
            _tables.drop_file_tables(con, name)
            con.execute("DELETE FROM _documents WHERE file=?", (name,))
            self._chunks_by_file.pop(name, None)
            taken = {r[0] for r in con.execute("SELECT name FROM _tables").fetchall()}
            try:
                self._ingest_one(con, path, name, taken)
            except Exception as e:                       # one bad file must not kill a build
                errors[name] = f"{type(e).__name__}: {e}"
                con.execute("INSERT OR REPLACE INTO _files VALUES (?,?,?,?,?,?,?,?,?)",
                            (name, st.st_mtime, st.st_size, path.suffix.lstrip("."),
                             0, 0, 0, 0, errors[name]))
        con.commit()

        self._renumber()
        # Whatever the deep pass already produced for these exact file bytes comes back for
        # free: contextual blurbs, the numbers and definitions tables, the Q&A cache. No
        # model call, no network, a few milliseconds. Nothing here waits on deep work.
        try:
            _deep.apply_on_build(self, con)
        except Exception as e:
            errors["_deep"] = f"{type(e).__name__}: {e}"
        self.bm25 = _search.Bm25.build([c.indexed() for c in self.chunks])
        self.topics = TopicIndex.build(self._topic_sources(con))
        self.dense = None
        if self.chunks and _search.Dense.available():
            try:
                self.dense = _search.Dense().build([c.indexed() for c in self.chunks])
            except Exception as e:
                errors["_dense"] = f"{type(e).__name__}: {e}"
                self.dense = None

        _search.save_json(self._dir / "chunks.json", self._chunks_by_file)
        _search.save_json(self._dir / "bm25.json", self.bm25.to_json())
        _search.save_json(self._dir / "topics.json", self.topics.to_json())
        if self.dense is not None:
            self.dense.save(self._dir / "dense.npy")

        row = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(n_rows),0) FROM _tables").fetchone()
        stats.tables, stats.table_rows = row[0], int(row[1])
        stats.cells = con.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
        stats.null_formula_cells = con.execute(
            "SELECT COUNT(*) FROM cells WHERE is_formula=1 AND value_num IS NULL "
            "AND value_text IS NULL").fetchone()[0]
        stats.files = len(seen)
        stats.spreadsheets = con.execute(
            "SELECT COUNT(DISTINCT file) FROM _tables").fetchone()[0]
        stats.documents = con.execute("SELECT COUNT(*) FROM _documents").fetchone()[0]
        stats.chunks = len(self.chunks)
        stats.topics = len(self.topics.topics)
        stats.dense = self.dense is not None and self.dense.vecs is not None
        stats.skipped = sorted(p.name for p in self.folder.iterdir()
                               if p.is_file() and not _is_indexable(p))
        stats.errors = errors
        con.close()

        self.con = _store.connect_ro(self.folder)
        stats.build_ms = (time.perf_counter() - t0) * 1000
        self._stats = stats
        _search.save_json(self._dir / "meta.json",
                          {"dense": stats.dense, "stats": asdict(stats), "ingest": _tables.INGEST_VERSION})
        self.version += 1
        return stats

    def _ingest_one(self, con, path: Path, rel: str, taken: set[str]) -> None:
        st = path.stat()
        ext = path.suffix.lower()
        n_tables = n_chunks = n_cells = n_null = 0
        if ext in _tables.SHEET_EXT or ext in _tables.CSV_EXT:
            res = _tables.ingest_spreadsheet(con, path, rel, taken)
            n_tables, n_cells, n_null = len(res["tables"]), res["n_cells"], res["n_null_formulas"]
        else:
            res = _docs.read_document(path, rel, 0)
            self._chunks_by_file[rel] = [
                {k: v for k, v in c.__dict__.items() if k != "score"} for c in res.chunks
            ]
            n_chunks = len(res.chunks)
            con.execute("INSERT OR REPLACE INTO _documents VALUES (?,?,?,?,?)",
                        (rel, res.kind, res.unit, res.n_units, n_chunks))
        con.execute("INSERT OR REPLACE INTO _files VALUES (?,?,?,?,?,?,?,?,?)",
                    (rel, st.st_mtime, st.st_size, ext.lstrip("."),
                     n_tables, n_chunks, n_cells, n_null, None))

    # -------------------------------------------------------------- topics
    def _topic_sources(self, con) -> list[dict]:
        src: list[dict] = []
        tmeta = {r["name"]: dict(r) for r in con.execute("SELECT * FROM _tables")}
        for name, t in tmeta.items():
            tgt = {"kind": "table", "table": name, "file": t["file"], "sheet": t["sheet"]}
            src.append({"topic": name.replace("_", " "), "kind": "table", "target": tgt})
            if t["sheet"]:
                src.append({"topic": t["sheet"], "kind": "sheet", "target": tgt})
            if t["title"]:
                for piece in t["title"].split(" / "):
                    src.append({"topic": piece, "kind": "table", "target": tgt})

            cols = list(con.execute(
                "SELECT name, label, letter, type FROM _columns WHERE table_name=? ORDER BY ord",
                (name,)))
            for c in cols:
                src.append({"topic": c["label"], "kind": "column",
                            "target": {"kind": "column", "table": name, "column": c["name"],
                                       "label": c["label"], "letter": c["letter"],
                                       "file": t["file"], "sheet": t["sheet"]}})
            if not cols:
                continue

            first = cols[0]
            if first["type"] == "TEXT" and t["n_rows"] <= MAX_ROW_LABEL_ROWS:
                q = f'SELECT "{first["name"]}" AS v, _row FROM "{name}" WHERE v IS NOT NULL'
                for r in con.execute(q):
                    v = str(r["v"]).strip()
                    if len(v) < 3 or v.replace(" ", "").isdigit():
                        continue
                    src.append({"topic": v, "kind": "row_label",
                                "target": {"kind": "row", "table": name, "label": v,
                                           "row": r["_row"], "file": t["file"],
                                           "sheet": t["sheet"],
                                           "cell_ref": f'{first["letter"]}{r["_row"]}'}})
            for c in cols:
                if c["type"] != "TEXT" or c is first and t["n_rows"] <= MAX_ROW_LABEL_ROWS:
                    continue
                vals = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{c["name"]}" FROM "{name}" '
                    f'WHERE "{c["name"]}" IS NOT NULL LIMIT {MAX_DISTINCT_VALUES + 1}')]
                if not vals or len(vals) > MAX_DISTINCT_VALUES:
                    continue
                for v in vals:
                    v = str(v).strip()
                    if len(v) < 3:
                        continue
                    src.append({"topic": v, "kind": "value",
                                "target": {"kind": "value", "table": name,
                                           "column": c["name"], "value": v,
                                           "file": t["file"], "sheet": t["sheet"]}})

        by_file_unit: dict[tuple[str, int], list[int]] = {}
        for c in self.chunks:
            by_file_unit.setdefault((c.file, c.unit), []).append(c.id)
        for c in self.chunks:
            if not c.heading:
                continue
            kind = "slide" if c.kind.startswith("pptx") else "heading"
            src.append({"topic": c.heading, "kind": kind,
                        "target": {"kind": "chunk", "file": c.file, "loc": c.loc,
                                   "chunk_ids": by_file_unit[(c.file, c.unit)]}})

        per_file: dict[str, str] = {}
        for c in self.chunks:
            per_file[c.file] = per_file.get(c.file, "") + " " + c.text
        for f, text in per_file.items():
            for term in candidate_terms(text):
                top = sorted(self.bm25.score(term).items(), key=lambda kv: -kv[1])[:3]
                ids = [i for i, _ in top if self.chunks[i].file == f] or [i for i, _ in top]
                if ids:
                    src.append({"topic": term, "kind": "term",
                                "target": {"kind": "chunk", "file": f, "chunk_ids": ids}})
        return src

    # ------------------------------------------------------------ querying
    def sql(self, query: str, limit: int = 50) -> list[dict]:
        if self.con is None:
            return [{"error": "index is not open; call build(folder) first"}]
        return _store.run_select(self.con, query, limit)

    def search_text(self, query: str, k: int = 6) -> list[Chunk]:
        out: list[Chunk] = []
        for i, sc in _search.hybrid(self.bm25, self.dense, query, k):
            c = self.chunks[i]
            out.append(Chunk(c.id, c.file, c.loc, c.text, round(sc, 4), c.kind, c.unit,
                             c.heading, c.context))
        return out

    def match_topics(self, text: str, limit: int = 12) -> list[TopicHit]:
        return self.topics.match(text, limit)

    def stats(self) -> Stats:
        return self._stats

    # ----------------------------------------------------------- deep index
    async def deep_build(self, llm_json, on_progress=None, model_fast: str | None = None,
                         model_good: str | None = None, force: bool = False) -> DeepStats:
        """Second pass: summaries, contextual chunks, numbers, definitions, a Q&A cache.

        `llm_json(prompt, json_schema, model) -> dict` is the server's `brain.llm.classify`.
        Everything is cached under <folder>/.presage/deep/ by file content hash, so an
        unchanged folder finishes from cache without a single model call. Safe to run in the
        background; the fast build() never waits on it.
        """
        # the page reads `deep_stats()` while the pass runs: show the live pass, not the last
        # result (a folder change used to leave an old "index is not built" error on screen for
        # the whole of the next run, 2026-09-30)
        self._deep_live = {"state": "running", "done": 0, "total": 0, "error": None}

        def progress(d: dict) -> None:
            self._deep_live = dict(d)
            if on_progress:
                on_progress(d)

        try:
            self._deep = await _deep.run_deep(self, llm_json, on_progress=progress,
                                              model_fast=model_fast, model_good=model_good,
                                              force=force)
        finally:
            self._deep_live = None
        self.warm_graphs()        # the first what-if of the call must not pay for a compile
        try:                      # E16's slide-to-cell links; nothing else built them (E43a found
            import asyncio        # the table empty on every fresh folder)
            stats = await asyncio.to_thread(self.link_build)
            log.info("links built: %s", stats)
        except Exception as e:  # noqa: BLE001  the links are an extra, never the deep pass itself
            log.warning("link build failed: %s", e)
        self.version += 1
        return self._deep

    def deep_stats(self) -> dict:
        live = getattr(self, "_deep_live", None)
        return dict(live) if live else self._deep.as_dict()

    def qa_matches(self, text: str, k: int = 3) -> list[QA]:
        """Anticipated questions closest to what was just asked. model2vec, well under 1 ms."""
        if not self._qa or self._qa_vecs is None or not text.strip():
            return []
        import numpy as np
        try:
            q = _search.Dense().model().encode([text]).astype("float32")[0]
        except Exception:
            return []
        n = float(np.linalg.norm(q)) or 1.0
        sims = self._qa_vecs @ (q / n)
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k] if len(sims) > k else range(len(sims))
        hits = [(int(i), float(sims[i])) for i in top]
        hits.sort(key=lambda kv: -kv[1])
        out = []
        for i, s in hits[:k]:
            a = self._qa[i]
            out.append(replace(a, score=round(s, 4)))
        return out

    def summaries(self) -> str:
        """One line per file and per section, for the system prompt. Empty until deep_build."""
        return self._summaries

    # --------------------------------------------------------- scratch paper
    # A lever is an input cell a what-if can move; an output is what it moves. Everything
    # here is sync and CPU-bound, and the first call on a cold workbook pays for the pycel
    # compile (about 2 s on a 3,500-formula model), so the server calls it through
    # asyncio.to_thread. `levers_block` is the exception: it never compiles on the
    # caller's thread, because it runs inside the system-prompt build.
    def _read(self, query: str, params: tuple = ()) -> list[dict]:
        """One read-only query on its own connection.

        `self.con` is a single handle the event loop is already using; a what-if runs in a
        worker thread at the same time and two threads sharing one sqlite connection
        silently hand each other empty result sets. These reads are rare and tiny, so they
        pay for their own connection instead.
        """
        if self.folder is None or not _store.db_path(self.folder).exists():
            return []
        con = _store.connect_ro(self.folder)
        try:
            return [dict(r) for r in con.execute(query, params)]
        except Exception as e:  # noqa: BLE001
            log.debug("scratch: query failed: %s", e)
            return []
        finally:
            con.close()

    def _workbooks(self) -> list[str]:
        return [r["path"] for r in self._read(
            "SELECT path FROM _files WHERE kind IN ('xlsx','xlsm') ORDER BY path")]

    def _one_workbook(self, file: str | None) -> str:
        books = self._workbooks()
        if file:
            if file in books:
                return file
            match = [b for b in books if b.lower() == file.lower()]
            if match:
                return match[0]
            raise ValueError(f"{file!r} is not a workbook in this folder "
                             f"({', '.join(books) or 'none'})")
        if len(books) == 1:
            return books[0]
        if not books:
            raise ValueError("no workbook is loaded")
        raise ValueError(f"say which workbook: {', '.join(books)}")

    def _file_key(self, rel: str) -> str:
        """Content hash, recomputed only when mtime or size moved."""
        path = self.folder / rel
        st = path.stat()
        sig = (st.st_mtime, st.st_size)
        known = self._hashes.get(rel)
        if known and known[0] == sig:
            return known[1]
        key = _deep.file_hash(path)
        self._hashes[rel] = (sig, key)
        return key

    def _graph_for(self, rel: str, key: str | None = None) -> _scratch.ModelGraph:
        """The compiled model for one workbook, rebuilt when the file bytes change."""
        key = key or self._file_key(rel)
        graph = self._graphs.get(rel)
        if graph is not None and graph.key == key:
            return graph
        graph = _scratch.ModelGraph(self.folder / rel, rel, key)
        self._graphs[rel] = graph
        return graph

    def _scratch_for(self, rel: str, build: bool = True) -> dict | None:
        """The cached lever table + grid for one workbook, built on demand."""
        key = self._file_key(rel)
        rec = self._scratch.get(rel)
        if rec is not None and rec.get("hash", key) == key:
            return rec
        if not build:
            return None
        con = _store.connect_ro(self.folder)
        try:
            rec, graph = _scratch.build(con, self.folder / rel, rel, key,
                                        self._graph_for(rel, key))
        finally:
            con.close()
        self._graphs[rel] = graph
        self._scratch[rel] = rec
        self.version += 1                  # so a cached system prompt picks the table up
        return rec

    def warm_graphs(self, file: str | None = None) -> None:
        """Compile in the background so the first what-if of a call is not the slow one."""
        for rel in ([file] if file else self._workbooks()):
            try:
                self._graph_for(rel).start()
            except Exception as e:  # noqa: BLE001
                log.debug("scratch: could not warm %s: %s", rel, e)

    def levers(self, file: str | None = None) -> list[dict]:
        """The inputs a what-if can move. Compiles the workbook if the cache is cold."""
        out: list[dict] = []
        for rel in ([self._one_workbook(file)] if file else self._workbooks()):
            try:
                rec = self._scratch_for(rel)
            except Exception as e:  # noqa: BLE001  a bad workbook leaves the rest usable
                log.warning("scratch: no levers for %s: %s", rel, e)
                continue
            out.extend((rec or {}).get("levers", []))
        return out

    def levers_block(self, limit: int = 40) -> str:
        """The LEVERS block for the system prompt. Never compiles on this thread: a missing
        record is built in the background (the deep pass normally did it already) and bumps
        `version` when it lands, so the next prompt build picks it up."""
        levers: list[dict] = []
        for rel in self._workbooks():
            rec = self._scratch_for(rel, build=False)
            if rec is None:
                self._build_scratch_later(rel)
                continue
            levers.extend(rec.get("levers", []))
        return _scratch.levers_block(levers, limit)

    def _build_scratch_later(self, rel: str) -> None:
        building = self.__dict__.setdefault("_scratch_building", set())
        if rel in building:
            return
        building.add(rel)

        def run() -> None:
            try:
                self._scratch_for(rel)
            except Exception as e:  # noqa: BLE001
                log.warning("scratch: background build failed for %s: %s", rel, e)
            finally:
                building.discard(rel)
        threading.Thread(target=run, name=f"scratch:{rel}", daemon=True).start()

    def whatif(self, file: str | None, sets: dict, watch: list[str] | None = None) -> dict:
        """Set input cells, re-evaluate, and report what moved. Restores the model after.

        `sets` maps `Sheet!A1` to a number or a spoken value ("1%", "$4.2M"). `watch`
        defaults to the outputs the changed levers are known to move. Raises ValueError
        for an unknown cell, a formula cell, or a workbook that will not compile.
        """
        t0 = time.perf_counter()
        rel = self._one_workbook(file)
        graph = self._graph_for(rel)
        try:
            graph.compile()
        except Exception as e:
            raise ValueError(f"{rel} did not compile: {e}") from None
        if not sets:
            raise ValueError("whatif needs at least one cell to change")

        table = {_scratch.canon(lv["sheet"], lv["cell"]): lv
                 for lv in (self._scratch_for(rel, build=False) or {}).get("levers", [])}
        changes, values = [], {}
        for ref, raw in sets.items():
            sheet, cell = _scratch.split_ref(ref)
            key = _scratch.canon(sheet, cell)
            if not graph.known(sheet, cell):
                raise ValueError(f"{ref} is not a cell that {rel} uses")
            if graph.is_formula(sheet, cell):
                raise ValueError(f"{ref} is a formula, not an input; pick a cell from the "
                                 "LEVERS table")
            was = graph.value(sheet, cell)
            lever = table.get(key) or {}
            if isinstance(was, str):
                now, kind = self._text_value(ref, raw, was, lever), "text"
            elif isinstance(was, (int, float)) and not isinstance(was, bool):
                now = _scratch.parse_value(raw)
                kind = lever.get("fmt") or _scratch.fmt_kind("", was)
            else:
                raise ValueError(f"{ref} holds {was!r}, which a what-if cannot move")
            changes.append({"cell": key, "label": lever.get("label") or self._label_of(rel, sheet, cell),
                            "fmt": kind, "was": was, "now": now,
                            "was_s": _scratch.fmt_value(was, kind),
                            "now_s": _scratch.fmt_value(now, kind)})
            values[key] = now

        outs: dict[str, dict] = {}
        for key in values:
            lever = table.get(key)
            found = lever["outputs"] if lever else self._outputs_of(rel, graph, key)
            for o in found:
                outs.setdefault(o["cell"], o)
        for ref in (watch or []):
            sheet, cell = _scratch.split_ref(ref)
            key = _scratch.canon(sheet, cell)
            if not graph.known(sheet, cell):
                raise ValueError(f"{ref} is not a cell that {rel} uses")
            outs.setdefault(key, {"cell": key, "label": self._label_of(rel, sheet, cell),
                                  "fmt": _scratch.fmt_kind("", graph.value(sheet, cell)),
                                  "value": graph.value(sheet, cell)})

        got = graph.whatif(values, list(outs))
        rows = []
        for key, meta in outs.items():
            was, now = meta.get("value"), got["values"].get(key)
            kind = meta.get("fmt") or "number"
            row = {"cell": key, "label": meta.get("label", ""), "fmt": kind,
                   "was": was, "now": now, "delta": None, "pct": None,
                   "was_s": _scratch.fmt_value(was, kind),
                   "now_s": _scratch.fmt_value(now, kind)}
            if isinstance(was, (int, float)) and isinstance(now, (int, float)):
                row["delta"] = now - was
                row["pct"] = (now - was) / abs(was) if was else None
            rows.append(row)
        return {"file": rel, "changes": changes, "outputs": rows,
                "partial": got["partial"], "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def _text_value(self, ref: str, raw, was: str, lever: dict) -> str:
        """The workbook's own spelling of a scenario the asker named.

        A selector holds one of a fixed set of strings and a client says a shorter version
        of one of them ("downside", "flat dues"), so the value is matched against the
        choices rather than taken as typed; a value that is not one of them is refused with
        the list, which is what lets the model pick again without a second round trip.
        """
        choices = list(lever.get("choices") or [])
        want = str(raw).strip()
        if not choices:
            if not want:
                raise ValueError(f"{ref} holds text; say which value to put in it")
            return want
        match = _scratch.match_choice(want, choices)
        if match is None:
            raise ValueError(f"{want!r} is not one of the values {ref} takes "
                             f"({', '.join(choices)})")
        return match

    def _outputs_of(self, rel: str, graph, key: str) -> list[dict]:
        """Outputs for a cell that is not in the lever table (a hand-picked what-if)."""
        con = _store.connect_ro(self.folder)
        try:
            sheet, cell = _scratch.split_ref(key)
            return _scratch.outputs_for(con, self.folder / rel, rel, graph, sheet, cell)
        except Exception as e:  # noqa: BLE001
            log.debug("scratch: no outputs for %s: %s", key, e)
            return []
        finally:
            con.close()

    def _label_of(self, rel: str, sheet: str, cell: str) -> str:
        rows = self._read("SELECT row_label FROM cells WHERE file=? AND sheet=? AND "
                          "cell_ref=? LIMIT 1", (rel, sheet, cell))
        return (rows[0].get("row_label") or "") if rows else ""

    def _ready_graph(self, file: str | None):
        """(workbook name, compiled graph). Raises the same ValueError `whatif` does."""
        rel = self._one_workbook(file)
        graph = self._graph_for(rel)
        try:
            graph.compile()
        except Exception as e:
            raise ValueError(f"{rel} did not compile: {e}") from None
        return rel, graph

    def _book_for(self, rel: str, graph):
        """The `cells` rows for one workbook, so a chain can be printed with row labels."""
        con = _store.connect_ro(self.folder)
        try:
            return graph.context(con, self.folder / rel, rel)[0]
        except Exception as e:  # noqa: BLE001
            log.debug("scratch: no cell rows for %s: %s", rel, e)
            return None
        finally:
            con.close()

    def _lever_at(self, rel: str, key: str) -> dict:
        for lv in (self._scratch_for(rel, build=False) or {}).get("levers", []):
            if _scratch.canon(lv["sheet"], lv["cell"]) == key:
                return lv
        return {}

    def goalseek(self, file: str | None, target: str, value, lever: str,
                 lo=None, hi=None) -> dict:
        """The lever value that puts `target` at `value`: "what fee gets us to $10M".

        Bisection on the compiled model, so the answer is the model's own, not a linear
        guess off one recalculation. Returns `error` (with the nearest value it reached)
        when the target is out of the lever's range, which is itself the answer to
        "can we get there at all".
        """
        t0 = time.perf_counter()
        rel, graph = self._ready_graph(file)
        lsheet, lcell = _scratch.split_ref(lever)
        lkey = _scratch.canon(lsheet, lcell)
        if not graph.known(lsheet, lcell):
            raise ValueError(f"{lever} is not a cell that {rel} uses")
        if graph.is_formula(lsheet, lcell):
            raise ValueError(f"{lever} is a formula, not an input; pick a cell from the "
                             "LEVERS table")
        was = graph.value(lsheet, lcell)
        if not isinstance(was, (int, float)) or isinstance(was, bool):
            raise ValueError(f"{lever} holds text, so there is nothing to solve for; use "
                             "whatif with one of its values")
        tsheet, tcell = _scratch.split_ref(target)
        if not graph.known(tsheet, tcell):
            raise ValueError(f"{target} is not a cell that {rel} uses")
        want = _scratch.parse_value(value)
        got = _scratch.solve(graph, lkey, _scratch.canon(tsheet, tcell), want,
                             None if lo is None else _scratch.parse_value(lo),
                             None if hi is None else _scratch.parse_value(hi))
        lever_row = self._lever_at(rel, lkey)
        kind = lever_row.get("fmt") or _scratch.fmt_kind("", was)
        out = {"file": rel, "lever": lkey,
               "label": lever_row.get("label") or self._label_of(rel, lsheet, lcell),
               "fmt": kind, "was": was, "was_s": _scratch.fmt_value(was, kind),
               "target": _scratch.canon(tsheet, tcell),
               "target_label": self._label_of(rel, tsheet, tcell),
               "want": want, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if got.get("error"):
            out.update({k: v for k, v in got.items() if k != "ms"})
            return out
        tkind = _scratch.fmt_kind("", got.get("achieved"))
        out.update({"value": got["value"], "value_s": _scratch.fmt_value(got["value"], kind),
                    "achieved": got["achieved"],
                    "achieved_s": _scratch.fmt_value(got["achieved"], tkind),
                    "want_s": _scratch.fmt_value(want, tkind),
                    "monotone": got["monotone"], "tries": got["tries"]})
        return out

    def explain(self, file: str | None, cell: str, sets: dict | None = None,
                hops: int = _scratch.EXPLAIN_HOPS) -> dict:
        """Why a number is what it is, or why it moved: the formula chain, hop by hop.

        With `sets` it is the route the change took from the input to this cell, each hop
        before and after; without, it is what the cell is built out of. Either way every
        hop is a real cell with a citation, so the card can say the mechanism instead of
        asserting one.
        """
        t0 = time.perf_counter()
        rel, graph = self._ready_graph(file)
        sheet, ref = _scratch.split_ref(cell)
        if not graph.known(sheet, ref):
            raise ValueError(f"{cell} is not a cell that {rel} uses")
        resolved = {}
        for key, raw in (sets or {}).items():
            ssheet, scell = _scratch.split_ref(key)
            if not graph.known(ssheet, scell):
                raise ValueError(f"{key} is not a cell that {rel} uses")
            was = graph.value(ssheet, scell)
            lever = self._lever_at(rel, _scratch.canon(ssheet, scell))
            resolved[_scratch.canon(ssheet, scell)] = (
                self._text_value(key, raw, was, lever) if isinstance(was, str)
                else _scratch.parse_value(raw))
        book = self._book_for(rel, graph)
        chain = _scratch.trace(graph, book, _scratch.canon(sheet, ref), resolved or None, hops)
        for hop in chain:
            hsheet, hcell = _scratch.split_ref(hop["cell"])
            kind = _scratch.fmt_kind("", hop["was"])
            hop["was_s"] = _scratch.fmt_value(hop["was"], kind)
            hop["now_s"] = _scratch.fmt_value(hop["now"], kind)
            hop["loc"] = f"{rel} › {_scratch.pretty(hsheet, hcell)}"
        last = chain[-1] if chain else {}
        out = {"file": rel, "cell": _scratch.canon(sheet, ref),
               "label": self._label_of(rel, sheet, ref),
               "was": last.get("was"), "now": last.get("now"),
               "moved": bool(last.get("delta")), "hops": len(chain), "chain": chain,
               "lever": list(resolved) or None, "reaches": True,
               "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if resolved and (not chain or chain[0]["cell"] not in resolved):
            # No route from the input to this cell. Saying so is the answer ("churn does not
            # touch the cost to serve"), and it is only half of it: what the input does move
            # is the other half, and a chain that starts somewhere else would imply a
            # mechanism that is not there.
            out["reaches"] = False
            try:
                got = self.whatif(rel, dict(resolved), None)
                moved = sorted((o for o in got["outputs"] if o.get("delta")),
                               key=lambda o: -abs(o.get("pct") or 0))[:3]
            except Exception:  # noqa: BLE001
                moved = []
            out["moves_instead"] = [{"cell": o["cell"], "label": o["label"],
                                     "was_s": o["was_s"], "now_s": o["now_s"]}
                                    for o in moved]
            names = ", ".join(resolved)
            out["note"] = (
                f"{names} does not feed {out['cell']} ({out['label'] or 'that cell'}); "
                + ("it moves " + ", ".join(f'{o["label"]} ({o["now_s"]})' for o in moved)
                   if moved else "nothing on the summary moves"))
        return out

    def compare(self, file: str | None, cases: list, watch: list | None = None,
                labels: list | None = None) -> dict:
        """Two or more whole cases side by side: "premium vs market at 1%".

        One call, one set of output rows, so the answer compares like with like instead of
        stitching two recalculations together. Case 0 is the reference every delta is
        measured from; pass `{}` for "as the model stands".
        """
        t0 = time.perf_counter()
        rel, _ = self._ready_graph(file)
        if not cases or len(cases) < 2:
            raise ValueError("compare needs at least two cases, e.g. "
                             "[{\"Summary Scenarios!B5\": \"Premium\"}, "
                             "{\"Summary Scenarios!B5\": \"Market\"}]")
        watch = [w for w in (watch or []) if str(w).strip()]
        _, graph = self._ready_graph(rel)
        moved = list(dict.fromkeys(k for case in cases for k in (case or {})))
        runs, cells = [], []
        for case in cases:
            # "as it stands" is run as a what-if that puts every cell back where it is, so
            # the case comes back through the same path with the same labels and formats
            sets = dict(case or {}) or {
                k: graph.value(*_scratch.split_ref(k)) for k in moved}
            if not sets:
                raise ValueError("compare needs at least one case that changes something")
            got = self.whatif(rel, sets, watch or None)
            runs.append(got)
            cells += [o["cell"] for o in got["outputs"]]
        order = list(dict.fromkeys(cells))
        names = [str(n) for n in (labels or [])] + [
            ", ".join(c.get("now_s", str(c.get("now"))) for c in (run["changes"] or []))
            or "as it stands" for run in runs[len(labels or []):]]
        rows = []
        for ref in order:
            by_case = [next((o for o in run["outputs"] if o["cell"] == ref), None)
                       for run in runs]
            values = [None if o is None else o["now"] for o in by_case]
            meta = next(o for o in by_case if o is not None)
            nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            rows.append({"cell": ref, "label": meta["label"], "fmt": meta["fmt"],
                         "values": values,
                         "values_s": [_scratch.fmt_value(v, meta["fmt"]) for v in values],
                         "best": max(nums) if nums else None,
                         "worst": min(nums) if nums else None,
                         "spread": (max(nums) - min(nums)) if len(nums) > 1 else None})
        return {"file": rel,
                "cases": [{"label": names[i], "sets": dict(cases[i] or {}),
                           "changes": run["changes"],
                           "outputs": {o["cell"]: o["now"] for o in run["outputs"]},
                           "partial": run["partial"]}
                          for i, run in enumerate(runs)],
                "rows": rows, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def sweep(self, file: str | None, cell: str, values: list,
              watch: list | None = None) -> dict:
        """One lever at several values in one call: "1, 1.25 and 1.5 percent".

        A range asked in one breath is one card, not three, and the model should not have
        to make three tool calls (and three chances to stop after the first).
        """
        t0 = time.perf_counter()
        rel, _ = self._ready_graph(file)
        if not values:
            raise ValueError("sweep needs at least one value")
        points, order, meta = [], [], {}
        for raw in list(values)[:12]:
            got = self.whatif(rel, {cell: raw}, watch or None)
            change = (got["changes"] or [{}])[0]
            points.append({"value": change.get("now"), "value_s": change.get("now_s"),
                           "outputs": {o["cell"]: o["now"] for o in got["outputs"]},
                           "outputs_s": {o["cell"]: o["now_s"] for o in got["outputs"]},
                           "partial": got["partial"]})
            for o in got["outputs"]:
                if o["cell"] not in meta:
                    order.append(o["cell"])
                    meta[o["cell"]] = o
        sheet, ref = _scratch.split_ref(cell)
        lever = self._lever_at(rel, _scratch.canon(sheet, ref))
        return {"file": rel, "cell": _scratch.canon(sheet, ref),
                "label": lever.get("label") or self._label_of(rel, sheet, ref),
                "fmt": lever.get("fmt") or "number",
                "was": meta and lever.get("value"), "points": points,
                "rows": [{"cell": c, "label": meta[c]["label"], "fmt": meta[c]["fmt"],
                          "was": meta[c]["was"], "was_s": meta[c]["was_s"],
                          "values": [p["outputs"].get(c) for p in points],
                          "values_s": [p["outputs_s"].get(c) for p in points]}
                         for c in order],
                "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def calc(self, expr: str, inputs: dict) -> dict:
        """Arithmetic over named inputs bound to cited cells, document numbers or literals.

        Every input is resolved and returned beside the answer, so the working can be
        shown and checked. The grammar is in `scratch.eval_expr`; `eval` is never used.
        """
        values, rows = {}, []
        for name, ref in (inputs or {}).items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"calc: {name!r} is not a usable input name")
            value, shown, label = self._resolve_input(ref)
            values[name] = value
            rows.append({"name": name, "ref": shown, "value": value, "label": label})
        return {"value": _scratch.eval_expr(expr, values), "expr": expr, "inputs": rows}

    def _resolve_input(self, ref) -> tuple[float, str, str]:
        """-> (value, how to cite it, what it is). Cell ref, document number, or literal."""
        if isinstance(ref, (int, float)) and not isinstance(ref, bool):
            return float(ref), "given", "from the asker"
        text = str(ref or "").strip()
        if not text:
            raise ValueError("calc: an input is empty")

        if "#" in text:                       # file#loc#value_text, as find_numbers returns
            parts = text.split("#")
            file, loc, value_text = parts[0], parts[1] if len(parts) > 1 else "", \
                parts[2] if len(parts) > 2 else ""
            rows = self._read(
                "SELECT value_num, value_text, sentence FROM numbers WHERE file=? AND "
                "loc=?" + (" AND value_text=?" if value_text else "") + " LIMIT 1",
                (file, loc, value_text) if value_text else (file, loc))
            if not rows or rows[0].get("value_num") is None:
                raise ValueError(f"calc: no number found at {text!r}")
            return (float(rows[0]["value_num"]), f"{file} › {loc}",
                    (rows[0].get("sentence") or "")[:120])

        parts = text.split("!")
        if len(parts) >= 2:
            file = None
            if len(parts) >= 3:
                file, parts = parts[0], parts[1:]
            sheet, cell = _scratch.split_ref("!".join(parts))
            rel = self._one_workbook(file)
            rows = self._read(
                "SELECT value_num, value_text, row_label, header FROM cells WHERE file=? "
                "AND sheet=? AND cell_ref=? LIMIT 1", (rel, sheet, cell))
            row = rows[0] if rows else {}
            value = row.get("value_num")
            if value is None:                 # an uncached formula: ask the compiled model
                graph = self._graph_for(rel)
                try:
                    graph.compile()
                    value = graph.value(sheet, cell)
                except Exception:
                    value = None
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"calc: {text!r} holds no number")
            label = (row.get("row_label") or "").strip() or (row.get("header") or "").strip()
            return float(value), f"{rel} › {_scratch.pretty(sheet, cell)}", label

        try:                                   # a literal the asker supplied: "1%", "4.2M"
            return _scratch.parse_value(text), "given", "from the asker"
        except ValueError:
            raise ValueError(f"calc: {text!r} is neither a cell reference nor a number") \
                from None

    def find_numbers(self, pattern: str, limit: int = 40) -> list[dict]:
        """Every number in the documents whose sentence or value matches. find_cells for prose."""
        pat = (pattern or "").strip().lower().replace("'", "''").replace("%", "")
        if len(pat) < 2:
            return [{"error": "pattern too short"}]
        like = f"%{pat}%"
        return self.sql(
            "SELECT file, loc, value_text, value_num, unit, sentence FROM numbers "
            f"WHERE lower(sentence) LIKE '{like}' OR lower(value_text) LIKE '{like}' "
            "ORDER BY file, loc", limit=limit)

    # ------------------------------------------------------- schema summary
    def schema_summary(self, budget_tokens: int = SUMMARY_TOKEN_BUDGET) -> str:
        if self.con is None:
            return "(no index)"
        defs = _deep.definitions_block(self.con) if self.con is not None else ""
        room = budget_tokens - len(defs) // 4
        for detail in (3, 2, 1, 0, -1):
            text = self._summary(detail)
            if len(text) // 4 <= room:
                break
        return text + defs

    def _summary(self, n_samples: int) -> str:
        con = self.con
        L: list[str] = []
        folder = Path(self._stats.folder or (self.folder or "."))
        L.append(f"CONTEXT FOLDER: {folder}")
        L.append("Numbers come from SQL over the tables below, or from the `cells` table. "
                 "Never state a number that is not in a query result. Cite as "
                 "file › sheet › cell, e.g. Acme_Financials.xlsx › P&L › D6.")
        L.append("")
        L.append("== SPREADSHEET TABLES (read-only sqlite; use the `sql` tool) ==")
        if n_samples < 0:
            L.append("(compact listing: column name(letter,type); every table also has _row and _is_total; "
                     "column labels and sample values are in _columns(table_name, name, label, letter, samples))")
        for t in con.execute("SELECT * FROM _tables ORDER BY file, sheet, first_data_row"):
            where = f'{t["file"]}' + (f' › {t["sheet"]}' if t["sheet"] else "")
            L.append("")
            L.append(f'{t["name"]}  <-  {where}   ({t["n_rows"]} rows x {t["n_cols"]} cols)')
            if t["title"] and n_samples >= 2:
                L.append(f'  about: {t["title"][:160]}')
            loc = (f'  header row {t["header_rows"]}, data rows {t["first_data_row"]}'
                   f'-{t["last_data_row"]}, first column {t["first_col_letter"]}')
            if t["total_rows"]:
                loc += f', totals row(s) {t["total_rows"]} (do not double count)'
            L.append(loc)
            cols = list(con.execute("SELECT * FROM _columns WHERE table_name=? ORDER BY ord", (t["name"],)))
            if n_samples < 0:   # compact form for big workbooks: one line of columns per table
                L.append("  cols: " + ", ".join(f'{c["name"]}({c["letter"]},{c["type"][0]})' for c in cols))
                continue
            for c in cols:
                line = f'  {c["name"]:<26} {c["type"]:<5} col {c["letter"]:<3} "{c["label"]}"'
                if n_samples and c["samples"]:
                    samp = " | ".join(c["samples"].split(" | ")[:n_samples])
                    line += f"  e.g. {samp}"
                if c["n_null"]:
                    line += f'  [{c["n_null"]} null]'
                L.append(line)
            L.append('  every row also carries _row (source spreadsheet row) and _is_total (0/1).')

        nulls = self._stats.null_formula_cells
        L.append("")
        L.append("== cells (universal fallback) ==")
        L.append(
            "cells(file, sheet, row, col, cell_ref, header, row_label, value_num, value_text, "
            "is_formula, formula) holds every non-empty cell of every sheet, with the header label "
            "above it and the row label to its left. Use it when a question does not fit the shaped "
            "tables above, when a sheet was split wrongly, or to confirm a citation: "
            "SELECT cell_ref, value_num FROM cells WHERE row_label LIKE '%revenue%' AND header LIKE "
            "'%Q3%'. value_num is the number, value_text the string; exactly one is set. "
            f"{nulls} cells are formulas whose workbook was never recalculated, so value_num is NULL "
            "and `formula` holds the expression: report those as not available, never guess them.")

        L.append("")
        L.append("== DOCUMENTS (use the `search_text` tool; citations are already formatted) ==")
        for d in con.execute("SELECT * FROM _documents ORDER BY file"):
            heads = [c.heading for c in self.chunks if c.file == d["file"] and c.heading]
            seen, uniq = set(), []
            for h in heads:
                if h not in seen:
                    seen.add(h)
                    uniq.append(h)
            line = (f'  {d["file"]:<22} {d["kind"]:<5} {d["n_units"]} {d["unit"]}s, '
                    f'{d["n_chunks"]} chunks')
            L.append(line)
            if uniq and n_samples >= 1:
                L.append("      " + "; ".join(uniq[:8])[:300])
        if self._stats.errors:
            L.append("")
            L.append("== FILES THAT FAILED TO INDEX ==")
            for f, e in self._stats.errors.items():
                L.append(f"  {f}: {e}")
        return "\n".join(L)

    # --------------------------------------------------------------- watch
    def watch(self, folder: Path | str | None = None, on_change=None, debounce: float = 0.75):
        """Re-ingest changed files in the background. Returns the watchdog observer."""
        import threading

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        target = Path(folder or self.folder).resolve()
        pending = threading.Event()
        stop = threading.Event()

        class H(FileSystemEventHandler):
            def on_any_event(self, ev):
                if ev.is_directory or _store.INDEX_DIR in str(ev.src_path):
                    return
                if Path(str(ev.src_path)).name.startswith(SKIP_PREFIXES):
                    return
                pending.set()

        def loop():
            while not stop.wait(debounce):
                if pending.is_set():
                    pending.clear()
                    time.sleep(debounce)          # let a burst of writes settle
                    pending.clear()
                    try:
                        s = self.build(target)
                    except Exception as e:
                        s = Stats(folder=str(target), errors={"_build": repr(e)})
                    if on_change:
                        on_change(s)

        obs = Observer()
        obs.schedule(H(), str(target), recursive=False)
        obs.start()
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        obs._presage_stop = stop                   # caller: obs._presage_stop.set(); obs.stop()
        return obs

    # ------------------------------------------------- deck -> model cell (E16)
    # A figure on a slide is the model's number in the presenter's units. These
    # three keep the map: build it once, read it forwards from a slide, read it
    # backwards from a cell. CPU only, no model call. See presage/ingest/link.py.
    def link_build(self, accept: float | None = None) -> dict:
        """Resolve every document figure to the cell it came from. -> stats dict."""
        from . import link as _link

        if self.folder is None:
            raise ValueError("no folder is open")
        kw = {} if accept is None else {"accept": accept}
        stats = _link.build_links(self.folder, **kw)
        if self.con is not None:                  # the ro handle predates the new table
            self.con.close()
            self.con = _store.connect_ro(self.folder)
        return stats

    def links_for(self, file: str, loc: str | None = None, limit: int = 60) -> list[dict]:
        """Every figure on one slide or page with the cell behind it, best first.

        `loc` takes the chunk's own citation ("slide 4 'FY26 plan by region'") or
        just its head ("slide 4"), because that is what a card has in hand.
        """
        if loc is None:
            return self._read(
                "SELECT * FROM links WHERE file=? ORDER BY loc, score DESC LIMIT ?",
                (file, limit))
        return self._read(
            "SELECT * FROM links WHERE file=? AND (loc=? OR loc LIKE ?) "
            "ORDER BY score DESC LIMIT ?",
            (file, loc, loc.rstrip() + "%", limit))

    def cell_sources(self, sheet: str, cell: str | None = None, file: str | None = None,
                     limit: int = 20) -> list[dict]:
        """Backwards: every slide or page that quotes this cell. 'Sheet!G57' also works."""
        if cell is None and "!" in (sheet or ""):
            sheet, cell = sheet.split("!", 1)
        if not cell:
            return [{"error": "cell_sources needs a cell reference"}]
        q = ("SELECT * FROM links WHERE lower(cell_sheet)=lower(?) "
             "AND upper(cell_ref)=upper(?)")
        params: tuple = (sheet or "", cell)
        if file:
            q += " AND cell_file=?"
            params += (file,)
        return self._read(q + " ORDER BY score DESC LIMIT ?", params + (limit,))


def _cli() -> int:
    import sys
    ix = Index()
    print(ix.build(Path(sys.argv[1] if len(sys.argv) > 1 else "sample_context")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
