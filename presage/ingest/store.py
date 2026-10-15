"""SQLite store: schema, the read-only query surface, and the file manifest.

One db per context folder at <folder>/.presage/index.db. Spreadsheets become real
tables in here; the `sql` tool the model gets is a read-only connection to this file.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

INDEX_DIR = ".presage"
DB_NAME = "index.db"

# Bookkeeping tables. Leading underscore keeps them visually apart from data tables.
SCHEMA = """
CREATE TABLE IF NOT EXISTS _files (
    path TEXT PRIMARY KEY, mtime REAL, size INTEGER, kind TEXT,
    n_tables INTEGER DEFAULT 0, n_chunks INTEGER DEFAULT 0,
    n_cells INTEGER DEFAULT 0, n_null_formulas INTEGER DEFAULT 0, error TEXT
);
CREATE TABLE IF NOT EXISTS _tables (
    name TEXT PRIMARY KEY, file TEXT, sheet TEXT, title TEXT,
    header_rows TEXT, first_data_row INTEGER, last_data_row INTEGER,
    first_col_letter TEXT, n_rows INTEGER, n_cols INTEGER, total_rows TEXT
);
CREATE TABLE IF NOT EXISTS _columns (
    table_name TEXT, ord INTEGER, name TEXT, label TEXT, letter TEXT,
    type TEXT, n_null INTEGER, samples TEXT,
    PRIMARY KEY (table_name, name)
);
CREATE TABLE IF NOT EXISTS _documents (
    file TEXT PRIMARY KEY, kind TEXT, unit TEXT, n_units INTEGER, n_chunks INTEGER
);
CREATE TABLE IF NOT EXISTS cells (
    file TEXT, sheet TEXT, row INTEGER, col INTEGER, cell_ref TEXT,
    header TEXT, row_label TEXT, value_num REAL, value_text TEXT,
    is_formula INTEGER DEFAULT 0, formula TEXT
);
CREATE INDEX IF NOT EXISTS ix_cells_file ON cells(file, sheet);
CREATE INDEX IF NOT EXISTS ix_cells_rowlabel ON cells(row_label);
CREATE INDEX IF NOT EXISTS ix_cells_header ON cells(header);
CREATE INDEX IF NOT EXISTS ix_cells_num ON cells(value_num);

-- filled by the deep pass (presage/ingest/deep.py), re-applied from its cache on
-- every fast build so the model's `sql` tool always sees them.
CREATE TABLE IF NOT EXISTS numbers (
    file TEXT, loc TEXT, chunk_id INTEGER, value_num REAL, value_text TEXT,
    unit TEXT, sentence TEXT
);
CREATE INDEX IF NOT EXISTS ix_numbers_file ON numbers(file);
CREATE INDEX IF NOT EXISTS ix_numbers_num ON numbers(value_num);
CREATE TABLE IF NOT EXISTS definitions (
    name TEXT, formula TEXT, numerator_ref TEXT, denominator_ref TEXT,
    numerator_label TEXT, denominator_label TEXT, file TEXT, sheet TEXT,
    cell_ref TEXT, unit TEXT, meaning TEXT
);
CREATE INDEX IF NOT EXISTS ix_definitions_file ON definitions(file, sheet);
"""

# filled by presage/ingest/link.py: one row per figure in a deck or memo that was
# resolved to the model cell it came from. `loc` is the chunk's own citation, so a
# topic hit on slide 12 joins straight to the cells slide 12 quotes.
LINKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS links (
    file TEXT, loc TEXT, kind TEXT, value_text TEXT, value_num REAL, unit TEXT,
    cell_file TEXT, cell_sheet TEXT, cell_ref TEXT, cell_num REAL,
    label TEXT, score REAL, method TEXT, n_alt INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_links_loc ON links(file, loc);
CREATE INDEX IF NOT EXISTS ix_links_cell ON links(cell_sheet, cell_ref);
"""

SCHEMA += LINKS_SCHEMA

# --------------------------------------------------------------------------- #
# Read-only SQL guard
# --------------------------------------------------------------------------- #
_FORBIDDEN = re.compile(
    r"\b(attach|detach|pragma|insert|update|delete|drop|create|alter|replace|vacuum|"
    r"reindex|analyze|begin|commit|rollback|savepoint|trigger|load_extension)\b",
    re.I,
)


def _strip_literals(sql: str) -> str:
    """Blank out string/identifier literals and comments so the guard can't be fooled."""
    out, i, n = [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in "'\"`":
            j = i + 1
            while j < n and sql[j] != ch:
                j += 1
            out.append(" ")
            i = j + 1
        elif ch == "[":
            j = sql.find("]", i)
            j = n if j < 0 else j
            out.append(" ")
            i = j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i)
            i = n if j < 0 else j + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def check_select(query: str) -> str | None:
    """Return an error message, or None if the query is a single safe SELECT/WITH."""
    if not query or not query.strip():
        return "empty query"
    bare = _strip_literals(query).strip()
    if bare.endswith(";"):
        bare = bare[:-1]
    if ";" in bare:
        return "only one statement is allowed (found ';')"
    head = bare.lstrip("( \t\r\n")[:6].lower()
    if not (head.startswith("select") or head.startswith("with")):
        return "only SELECT / WITH queries are allowed"
    bad = _FORBIDDEN.search(bare)
    if bad:
        return f"keyword '{bad.group(0)}' is not allowed in a read-only query"
    return None


# --------------------------------------------------------------------------- #
def db_path(folder: Path) -> Path:
    return Path(folder) / INDEX_DIR / DB_NAME


def connect_rw(folder: Path) -> sqlite3.Connection:
    p = db_path(folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def connect_ro(folder: Path) -> sqlite3.Connection:
    p = db_path(folder)
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def run_select(con: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """Single read-only SELECT. Errors come back as [{'error': ...}], never raised."""
    err = check_select(query)
    if err:
        return [{"error": err}]
    try:
        cur = con.execute(query)
        rows = cur.fetchmany(max(1, limit))
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        return [{"error": f"{type(e).__name__}: {e}"}]


# --------------------------------------------------------------------------- #
# Identifier sanitising
# --------------------------------------------------------------------------- #
_SQL_KEYWORDS = {
    "select", "from", "where", "group", "order", "table", "index", "by", "as", "on",
    "and", "or", "not", "null", "in", "is", "like", "join", "left", "right", "inner",
    "union", "all", "limit", "offset", "case", "when", "then", "else", "end", "values",
    "default", "primary", "key", "references", "check", "unique", "cast", "row",
}


def ident(text: str, fallback: str = "col") -> str:
    s = (text or "").strip().lower()
    s = s.replace("&", " and ").replace("%", " pct ").replace("#", " num ")
    s = s.replace("$", " usd ").replace("/", " per ").replace("+", " plus ")
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = fallback
    if s[0].isdigit():
        s = "c_" + s
    if s in _SQL_KEYWORDS:
        s = s + "_"
    return s[:58]


def unique_ident(base: str, taken: set[str]) -> str:
    name = base
    i = 2
    while name in taken:
        name = f"{base}_{i}"
        i += 1
    taken.add(name)
    return name


def col_letter(n: int) -> str:
    """1 -> A, 27 -> AA."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s
