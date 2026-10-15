"""A shortlist of levers for the typed choice, for a workbook past the 255-option ceiling.

One Jev `choice` takes 254 options plus a NONE, so `intent.pick_lever` can put a whole lever
table in front of the model only while the workbook has 254 levers or fewer. the big model has 250 and
fits; a bigger model does not, and then the choice has to be asked over a shortlist. E7 measured
what happens when the word matcher draws that shortlist: its top 20 holds the gold lever on only
81.3% of 300 spoken lines, and the two-stage path scores 78.3% against 95.7% for the whole table.
The shortlist was the ceiling, not the model.

This module draws the shortlist with the same static embeddings the warm-card match already uses
(model2vec potion-base-8M, ~0.03 ms a text, no torch), fused with the rarity-weighted word score
`prerun.match_levers` returns. Measured on the same 300 lines over the same 250 levers
(`docs/experiments/E20-shortlist.md`; 9 of the 300 gold cells are no longer in the table, so 97.0%
is the ceiling every row below is read against):

    recall@k on 300 lines          @5     @10    @20    @50
    Brain._lever_matches          80.0   82.0   82.0   82.0     (E7's shortlist)
    prerun.match_levers           85.7   89.3   92.0   92.0     (the shipped matcher)
    model2vec cosine              83.0   86.3   92.3   95.7
    hybrid (this module)          89.7   92.7   95.7   97.0

End to end on Jev, same lines and same session: the choice over the hybrid top 50 scores 93.0%
against 92.3% over the whole table, in 233 ms against 387; the top 20 scores 91.0%. Given the gold
is in the list the model picks it at the same rate whatever k is (95.9% / 95.2% / 95.1%), so k is
a recall decision and nothing else. On a simulated 600-lever workbook, where the whole table
cannot be asked at all, the top 50 holds the gold on 95.0% and the end-to-end pick is 86.0%.

So: while the table fits in one question, ask it all; the shortlist is not free accuracy, only
free latency. Past 254 levers, shortlist at k=50, which is `DEFAULT_K`.

    vecs = shortlist.vecs_for(index)                       # once per index version, 4 ms / 250
    pick = await shortlist.pick(decider, line, levers, vecs=vecs, k=50, prior=last_ask)

Nothing here is wired into `brain.py`. `pick` is the wiring, written where it can be measured.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from presage.llm.intent import Pick, cell_of, option_text, pick_lever

DEFAULT_K = 50             # what to ask for past 254 levers (E20)
RRF_K = 20                 # reciprocal-rank fusion constant; 10 to 30 score the same on E20's set
MAX_MATCH = 60             # word-matcher hits fused, best first; its own list is shorter than this
VECS_FILE = "lever_vecs.npy"


def embed_text(lever: dict) -> str:
    """What gets embedded for one lever: the option string the choice question itself reads.

    `label · header (sheet › section) = value [spoken names]`. Four shorter variants were
    measured (label alone, label plus names, the fields without the sheet, one vector per field
    max-pooled) and the option text won at the k that matters: 92.3% recall@20 and 95.7%@50
    against 91.7% / 94.0% for label-plus-names, the best of the others at 20. Max-pooling one
    vector per name wins at k=1 (70.0% against 62.3%) and loses from k=10 on, which is the wrong
    end of the list for a shortlist.
    """
    return option_text(lever)


# ----------------------------------------------------------------------------- the vectors
@dataclass
class LeverVecs:
    """Static embeddings of one lever table's option text, built once per index version.

    `vecs` is None when model2vec is not installed or its model is not in the cache, and every
    caller here falls back to the word matcher rather than failing: the dense half is optional
    exactly as it is in `ingest/search.py`.
    """
    cells: list[str] = field(default_factory=list)
    vecs: Any = None                       # (n, 256) float32, L2-normalised
    version: int = 0
    build_ms: float = 0.0

    def ok(self) -> bool:
        return self.vecs is not None and len(self.vecs) == len(self.cells) and bool(self.cells)

    @staticmethod
    def build(levers: list[dict], version: int = 0) -> "LeverVecs":
        cells = [cell_of(lv) for lv in levers]
        t0 = time.perf_counter()
        vecs = _embed([embed_text(lv) for lv in levers])
        return LeverVecs(cells, vecs, version, (time.perf_counter() - t0) * 1000)

    def matches(self, levers: list[dict]) -> bool:
        """True when this was built for exactly this table, in this order."""
        return self.ok() and self.cells == [cell_of(lv) for lv in levers]

    def cosine(self, line: str):
        """Cosine of the line against every lever, as a numpy array. None when the model is not
        there, the line is empty, or the encode fails."""
        if not self.ok() or not (line or "").strip():
            return None
        q = _embed([line])
        if q is None:
            return None
        return self.vecs @ q[0]

    # ------------------------------------------------------------------ disk
    def save(self, path: Path) -> None:
        if not self.ok():
            return
        try:
            import numpy as np
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.vecs)
        except Exception:  # noqa: BLE001  a cache that will not write is not an error
            pass

    @staticmethod
    def load(path: Path, levers: list[dict], version: int = 0) -> "LeverVecs | None":
        """The saved vectors, when there are as many of them as there are levers."""
        try:
            import numpy as np
            if not Path(path).exists():
                return None
            v = np.load(path)
        except Exception:  # noqa: BLE001
            return None
        if len(v) != len(levers):
            return None
        return LeverVecs([cell_of(lv) for lv in levers], v, version)


def _embed(texts: list[str]):
    """model2vec vectors, L2-normalised. None when the model is not on this machine."""
    if not texts:
        return None
    try:
        from presage.ingest import search as _search
        if not _search.Dense.available(allow_download=False):
            return None
        import numpy as np
        v = _search.Dense().model().encode(texts).astype("float32")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1, n)
    except Exception:  # noqa: BLE001  the shortlist must never take a decision down
        return None


def vecs_for(index: Any, levers: list[dict] | None = None, file: str | None = None) -> LeverVecs:
    """The vectors for an Index's lever table, built once per `index.version`.

    Kept on the Index object and on disk beside the deep cache, so a re-opened index pays the
    ~10 ms per 250 levers only when the table itself changed.
    """
    levers = levers if levers is not None else _levers_of(index, file)
    version = int(getattr(index, "version", 0) or 0)
    held = getattr(index, "_lever_vecs", None)
    if isinstance(held, LeverVecs) and held.version == version and held.matches(levers):
        return held
    path = _vecs_path(index)
    got = LeverVecs.load(path, levers, version) if path else None
    if got is None:
        got = LeverVecs.build(levers, version)
        if path:
            got.save(path)
    try:
        index._lever_vecs = got
    except Exception:  # noqa: BLE001  a stub index without __dict__ is still usable
        pass
    return got


# A caller that has a lever table but no Index -- which is every caller of `intent.pick_lever` --
# still must not re-embed 250 levers on every spoken line. `Index.levers()` builds a fresh list
# each call but fills it from the cached scratch record, so the *dicts* are the same objects until
# the record is rebuilt: the first and last of them identify the table, and holding a reference to
# both stops their ids being handed to anything else.
_TABLE_CACHE: list[tuple[tuple, dict, dict, "LeverVecs"]] = []
_TABLE_CACHE_MAX = 4


def vecs_for_table(levers: list[dict], version: Any = None) -> "LeverVecs":
    """Vectors for a lever table, memoised on the identity of the table's own records.

    `version` is an extra key for a caller that has one (`index.version`); it is not needed for
    correctness, because a rebuilt table brings new dicts with it.
    """
    if not levers:
        return LeverVecs()
    key = (len(levers), id(levers[0]), id(levers[-1]), version)
    for i, (k, _first, _last, vecs) in enumerate(_TABLE_CACHE):
        if k == key:
            if i:
                _TABLE_CACHE.insert(0, _TABLE_CACHE.pop(i))
            return vecs
    got = LeverVecs.build(levers, int(version or 0))
    _TABLE_CACHE.insert(0, (key, levers[0], levers[-1], got))
    del _TABLE_CACHE[_TABLE_CACHE_MAX:]
    return got


def _vecs_path(index: Any) -> Path | None:
    try:
        from presage.ingest.deep import DEEP_DIR
        return Path(index._dir) / DEEP_DIR / VECS_FILE
    except Exception:  # noqa: BLE001
        return None


def _levers_of(index: Any, file: str | None = None) -> list[dict]:
    try:
        return list(index.levers(file) if file else index.levers() or [])
    except Exception:  # noqa: BLE001
        return []


# ----------------------------------------------------------------------------- the word half
def word_scores(line: str, levers: list[dict]) -> dict[int, float]:
    """`prerun.match_levers` by position in the table: the rarity-weighted share of a label's
    weight the line covers, plus the block title, the header and a text lever's own values.

    This is the matcher the pre-run ships (E32), not `Brain._lever_matches`: on E20's set it
    holds the gold in its hits on 92.0% of lines against 82.0%, because the older one keeps
    only what clears a hard threshold and returns 3.4 levers a line.
    """
    try:
        from presage import prerun
        hits = prerun.match_levers(line, levers)
    except Exception:  # noqa: BLE001
        return {}
    at = {cell_of(lv): i for i, lv in enumerate(levers)}
    out: dict[int, float] = {}
    for h in hits[:MAX_MATCH]:
        i = at.get(h.cell)
        if i is not None and i not in out:
            out[i] = float(h.score)
    return out


# ----------------------------------------------------------------------------- the ranking
def rank(line: str, levers: list[dict], vecs: LeverVecs | None = None,
         method: str = "hybrid") -> list[tuple[float, int]]:
    """Every lever scored against the line, best first, as (score, position in the table).

    `method` is "hybrid" (reciprocal-rank fusion of the two lists), "dense" (cosine alone) or
    "words" (the matcher alone). The hybrid is the default because neither half is enough on its
    own: the cosine finds the lever whose words are nowhere in the line ("the bottom tier price"
    for `Bracket 1`) and the matcher finds the one whose rare word is the whole of the evidence
    ("MBNA"), and fusing them beats both at every k measured (E20).
    """
    n = len(levers)
    if not n:
        return []
    cos = vecs.cosine(line) if (vecs is not None and method != "words") else None
    words = word_scores(line, levers) if method != "dense" else {}
    if cos is None and not words:
        return [(0.0, i) for i in range(n)]
    if cos is None:                       # no model on this machine: the matcher, then the table
        order = sorted(words, key=lambda i: -words[i])
        return [(words[i], i) for i in order] + [(0.0, i) for i in range(n) if i not in words]
    if method == "dense" or not words:
        return sorted(((float(cos[i]), i) for i in range(n)), key=lambda si: -si[0])

    # reciprocal-rank fusion: rank-based, so the cosine's narrow spread (0.3 to 0.8 over a whole
    # table) and the matcher's absolute score never have to be put on one scale. A blend of
    # min-max normalised scores was measured too and is a point worse at k=20 (E20).
    import numpy as np
    dense_order = list(np.argsort(-cos))
    dense_rank = {int(j): p for p, j in enumerate(dense_order)}
    word_rank = {j: p for p, j in enumerate(sorted(words, key=lambda i: -words[i]))}
    fused = []
    for i in range(n):
        s = 1.0 / (RRF_K + dense_rank.get(i, n))
        if i in word_rank:
            s += 1.0 / (RRF_K + word_rank[i])
        fused.append((s, i))
    fused.sort(key=lambda si: (-si[0], dense_rank.get(si[1], n)))
    return fused


def shortlist(line: str, levers: list[dict], k: int = DEFAULT_K, vecs: LeverVecs | None = None,
              method: str = "hybrid", keep: list[str] | None = None) -> list[dict]:
    """The k levers most likely to be the one the line means, best first.

    `keep` is cells that must be in the list whatever it scores -- the previous ask's lever, so a
    follow-up can resolve to it. They go in front, and the shortlist stays k long.
    """
    if k <= 0 or not levers:
        return []
    if len(levers) <= k and not keep:
        return list(levers)
    scored = rank(line, levers, vecs, method)
    chosen = [i for _, i in scored[:k]]
    if keep:
        at = {cell_of(lv): i for i, lv in enumerate(levers)}
        front = [at[c] for c in keep if c in at and at[c] not in chosen[:k]]
        if front:
            chosen = front + [i for i in chosen if i not in set(front)]
    return [levers[i] for i in chosen[:k]]


# ----------------------------------------------------------------------------- the wiring
async def pick(decider, line: str, levers: list[dict], k: int = DEFAULT_K,
               vecs: LeverVecs | None = None, prior: dict | None = None,
               method: str = "hybrid") -> Pick:
    """`intent.pick_lever` over the shortlist: the whole path for a workbook past the ceiling.

    Since E20 was wired, `intent.pick_lever` does this itself at `intent.SHORTLIST_K`, so the only
    reason to call this is to ask for a different k or a single-half `method`. It cuts the list
    and then tells `pick_lever` not to cut it again (`k=None`).
    """
    if not levers:
        return await pick_lever(decider, line, levers, prior=prior)
    keep = []
    if prior and (prior.get("lever") or None):
        keep = [cell_of(prior["lever"])]
    opts = shortlist(line, levers, k, vecs, method, keep=keep)
    return await pick_lever(decider, line, opts, prior=prior, k=None)


__all__ = ["LeverVecs", "DEFAULT_K", "RRF_K", "embed_text", "vecs_for", "vecs_for_table",
           "word_scores", "rank", "shortlist", "pick"]
