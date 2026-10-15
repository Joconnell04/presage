"""Chunk retrieval: BM25 over a plain inverted index, plus an optional dense rescore.

BM25 here is ~60 lines of stdlib rather than bm25s or rank_bm25 because at this
scale (a few thousand chunks) a dict of posting lists only touches the query's own
terms, which is faster than either library's array scan, serialises to JSON with no
extra format, and removes numpy/scipy from the hot path. See docs/research-ingest.md
for the measured comparison.

The dense half is model2vec static embeddings (potion-base-8M): numpy only, no
torch, ~0.03 ms to embed a query. It is optional and off unless the model is
already on disk, so an index build never blocks on a network fetch mid-call.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .topics import normalize

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

K1 = 1.2
B = 0.75
DENSE_MODEL = os.environ.get("PRESAGE_DENSE_MODEL", "minishlab/potion-base-8M")
DENSE_WEIGHT = 0.35


class Bm25:
    def __init__(self, data: dict | None = None):
        d = data or {}
        self.postings: dict[str, list[list[int]]] = d.get("postings", {})
        self.doc_len: list[int] = d.get("doc_len", [])
        self.n: int = len(self.doc_len)
        self.avgdl: float = d.get("avgdl", 1.0) or 1.0

    @staticmethod
    def build(texts: list[str]) -> "Bm25":
        postings: dict[str, list[list[int]]] = {}
        doc_len: list[int] = []
        for i, t in enumerate(texts):
            toks = normalize(t)
            doc_len.append(len(toks))
            tf: dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            for w, f in tf.items():
                postings.setdefault(w, []).append([i, f])
        avg = (sum(doc_len) / len(doc_len)) if doc_len else 1.0
        return Bm25({"postings": postings, "doc_len": doc_len, "avgdl": avg})

    def to_json(self) -> dict:
        return {"postings": self.postings, "doc_len": self.doc_len, "avgdl": self.avgdl}

    def score(self, query: str) -> dict[int, float]:
        toks = normalize(query)
        if not toks or not self.n:
            return {}
        scores: dict[int, float] = {}
        seen: set[str] = set()
        for w in toks:
            if w in seen:
                continue
            seen.add(w)
            plist = self.postings.get(w)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, f in plist:
                denom = f + K1 * (1 - B + B * self.doc_len[i] / self.avgdl)
                scores[i] = scores.get(i, 0.0) + idf * f * (K1 + 1) / denom
        return scores


# --------------------------------------------------------------------------- #
_MODEL = None       # one StaticModel per process; loading it costs ~0.4 s


class Dense:
    """model2vec static embeddings. Silently absent if the package or model is not.

    Off unless the model is already in the HuggingFace cache, so an index build
    never blocks on a download during a call. Set PRESAGE_DENSE=1 to allow the
    one-time ~30 MB fetch.
    """

    def __init__(self, vecs=None):
        self.vecs = vecs

    @staticmethod
    def available(allow_download: bool | None = None) -> bool:
        try:
            import model2vec  # noqa: F401
        except ImportError:
            return False
        if allow_download is None:
            allow_download = os.environ.get("PRESAGE_DENSE", "") == "1"
        if allow_download:
            return True
        from huggingface_hub import constants
        cache = Path(constants.HF_HUB_CACHE)
        want = "models--" + DENSE_MODEL.replace("/", "--")
        return (cache / want).exists() or Path(DENSE_MODEL).exists()

    def model(self):
        global _MODEL
        if _MODEL is None:
            from model2vec import StaticModel
            _MODEL = StaticModel.from_pretrained(DENSE_MODEL)
        return _MODEL

    def build(self, texts: list[str]):
        import numpy as np

        v = self.model().encode(texts).astype("float32")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        self.vecs = v / np.where(n == 0, 1, n)
        return self

    def save(self, path: Path):
        import numpy as np

        if self.vecs is not None:
            np.save(path, self.vecs)

    def load(self, path: Path):
        import numpy as np

        if path.exists():
            self.vecs = np.load(path)
        return self

    def score(self, query: str) -> dict[int, float]:
        if self.vecs is None or not len(self.vecs):
            return {}
        import numpy as np

        q = self.model().encode([query]).astype("float32")[0]
        nq = np.linalg.norm(q) or 1.0
        sims = self.vecs @ (q / nq)
        return {int(i): float(s) for i, s in enumerate(sims)}


# --------------------------------------------------------------------------- #
def _minmax(d: dict[int, float]) -> dict[int, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def hybrid(bm: Bm25, dense: Dense | None, query: str, k: int) -> list[tuple[int, float]]:
    sparse = bm.score(query)
    if dense is None or dense.vecs is None:
        return sorted(sparse.items(), key=lambda kv: -kv[1])[:k]
    # only rescore what BM25 already likes, plus the dense top slice
    dn = dense.score(query)
    pool = set(sorted(sparse, key=sparse.get, reverse=True)[:50])
    pool |= set(sorted(dn, key=dn.get, reverse=True)[:20])
    s, d = _minmax({i: sparse.get(i, 0.0) for i in pool}), _minmax({i: dn.get(i, 0.0) for i in pool})
    blended = {i: (1 - DENSE_WEIGHT) * s.get(i, 0.0) + DENSE_WEIGHT * d.get(i, 0.0) for i in pool}
    return sorted(blended.items(), key=lambda kv: -kv[1])[:k]


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
