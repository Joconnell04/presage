"""Topic index for speculative prefetch.

At ingest we collect every phrase a question is likely to name: table names and
titles, column labels, row labels, sheet names, document headings, slide titles,
and distinctive repeated multiword terms from the document text.

At call time `match_topics(utterance)` looks up n-grams of the normalised
utterance in a plain dict. No model, no embeddings, target well under 5 ms so it
can run on every partial transcript line.

The trick that makes finance speech match finance spreadsheets is canonicalising
both sides the same way: "third quarter" and "Q3" both normalise to `q3`,
"cost of goods sold" and "COGS" both normalise to `cogs`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_NGRAM = 4

# Abbreviation <-> expansion. Everything on the right collapses to the key.
ALIASES: dict[str, list[str]] = {
    "q1": ["first quarter", "quarter one", "quarter 1"],
    "q2": ["second quarter", "quarter two", "quarter 2"],
    "q3": ["third quarter", "quarter three", "quarter 3"],
    "q4": ["fourth quarter", "quarter four", "quarter 4"],
    "h1": ["first half", "half one"],
    "h2": ["second half", "half two"],
    "fy": ["fiscal year", "financial year"],
    "ytd": ["year to date"],
    "yoy": ["year over year", "year on year"],
    "qoq": ["quarter over quarter", "quarter on quarter"],
    "mom": ["month over month", "month on month"],
    "ebitda": ["earnings before interest tax depreciation and amortisation",
               "earnings before interest tax depreciation and amortization",
               "earnings before interest taxes depreciation and amortization"],
    "ebit": ["earnings before interest and tax"],
    "arr": ["annual recurring revenue", "annualised recurring revenue",
            "annualized recurring revenue"],
    "mrr": ["monthly recurring revenue"],
    "acv": ["annual contract value"],
    "tcv": ["total contract value"],
    "cogs": ["cost of goods sold", "cost of sales", "cost of revenue"],
    "opex": ["operating expenses", "operating expense", "operating cost",
             "operating costs", "opex spend"],
    "capex": ["capital expenditure", "capital expenditures", "capital spend"],
    "gm": ["gross margin"],
    "nm": ["net margin"],
    "hc": ["headcount", "head count", "fte", "full time equivalent",
           "full time equivalents"],
    "sm": ["sales and marketing", "s and m", "sales marketing"],
    "rd": ["research and development", "r and d"],
    "ga": ["general and administrative", "g and a", "general admin"],
    "cac": ["customer acquisition cost"],
    "ltv": ["lifetime value", "life time value"],
    "nrr": ["net revenue retention", "net dollar retention", "ndr"],
    "grr": ["gross revenue retention", "gross retention"],
    "pnl": ["p and l", "profit and loss", "income statement"],
    "kpi": ["key performance indicator", "key performance indicators"],
    "tam": ["total addressable market"],
    "sam": ["serviceable addressable market"],
    "som": ["serviceable obtainable market"],
    "cagr": ["compound annual growth rate"],
    "ent": ["enterprise"],
    "mm": ["mid market", "midmarket", "middle market"],
    "smb": ["small and medium business", "small business", "small medium business"],
    "na": ["north america", "north american", "nam", "amer", "americas"],
    "emea": ["europe middle east and africa", "europe"],
    "apac": ["asia pacific", "asia pac"],
    "latam": ["latin america"],
    "churn": ["churned", "attrition", "logo loss", "logo losses"],
    "revenue": ["topline", "top line", "sales revenue", "turnover"],
    "plan": ["budget", "forecast"],
}

# Words that carry no retrieval signal in a spoken question.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "am", "do", "does", "did", "we", "i", "you", "he", "she", "it", "they", "them",
    "our", "your", "their", "this", "that", "these", "those", "there", "here",
    "what", "which", "who", "how", "when", "where", "why", "can", "could", "would",
    "should", "will", "shall", "may", "might", "just", "like", "kind", "sort",
    "okay", "ok", "yeah", "um", "uh", "right", "well", "really", "actually",
    "about", "into", "over", "than", "then", "some", "any", "all", "got", "get",
    "me", "my", "us", "look", "looks", "looked", "say", "said", "talk", "tell",
    "know", "think", "going", "gonna", "walk", "through", "give", "take", "let",
    "lets", "one", "two", "up", "out", "down", "back", "again", "also", "very",
    "much", "many", "more", "most", "now", "sorry", "please", "thanks", "hey",
}

_WORD = re.compile(r"[a-z0-9&%$][a-z0-9&%$'\.]*")
_EXPANSION: dict[tuple[str, ...], str] = {}
_SINGLE: dict[str, str] = {}
for _k, _vs in ALIASES.items():
    for _v in _vs:
        _toks = tuple(_v.split())
        if len(_toks) == 1:
            _SINGLE[_toks[0]] = _k
        else:
            _EXPANSION[_toks] = _k
_MAX_ALIAS = max(len(t) for t in _EXPANSION) if _EXPANSION else 1


def _stem(w: str) -> str:
    if len(w) > 4:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("sses") or w.endswith("shes") or w.endswith("ches"):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
            return w[:-1]
    return w


def normalize(text: str) -> list[str]:
    """Lowercase, expand abbreviations both ways, drop stopwords, light stem."""
    if not text:
        return []
    low = text.lower().replace("&", " and ").replace("/", " ").replace("-", " ")
    raw = [m.group(0).strip(".'") for m in _WORD.finditer(low)]
    raw = [w for w in raw if w]
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        hit = None
        for span in range(min(_MAX_ALIAS, n - i), 1, -1):
            key = tuple(raw[i:i + span])
            if key in _EXPANSION:
                hit = (_EXPANSION[key], span)
                break
        if hit:
            out.append(hit[0])
            i += hit[1]
            continue
        w = raw[i]
        w = _SINGLE.get(w, w)
        # "q3" / "fy25" / "fy2025" split into their parts so both halves match
        m = re.fullmatch(r"(fy|cy)(\d{2,4})", w)
        if m:
            out.append(m.group(1) + m.group(2)[-2:])
        else:
            out.append(_stem(w))
        i += 1
    return [w for w in out if w not in STOPWORDS and len(w) > 1]


def norm_key(text: str) -> str:
    return " ".join(normalize(text))


# --------------------------------------------------------------------------- #
@dataclass
class TopicHit:
    topic: str
    score: float
    kind: str
    targets: list[dict] = field(default_factory=list)


KIND_WEIGHT = {
    "row_label": 1.6, "table": 1.5, "sheet": 1.4, "column": 1.4, "value": 1.3,
    "heading": 1.2, "slide": 1.2, "term": 0.8,
}


class TopicIndex:
    """phrase -> topics. Built at ingest, queried on every transcript line.

    Every contiguous sub-phrase of a topic is indexed too, so a bare "q3" finds the
    column labelled "FY25 actual Q3". Sub-phrase hits score lower (n-gram length
    squared) and a phrase that points at many topics is discounted, so the exact
    long match always wins.
    """

    def __init__(self, data: dict | None = None):
        d = data or {}
        self.topics: list[dict] = d.get("topics", [])
        self.phrases: dict[str, list[list[int]]] = d.get("phrases", {})

    # ---------------------------------------------------------------- build
    @staticmethod
    def build(sources: list[dict]) -> "TopicIndex":
        """sources: [{'topic','kind','target': {...}}]. Same phrase merges targets."""
        by_key: dict[str, int] = {}
        topics: list[dict] = []
        phrases: dict[str, dict[int, int]] = {}
        for s in sources:
            key = norm_key(s["topic"])
            if not key:
                continue
            if key not in by_key:
                by_key[key] = len(topics)
                topics.append({"topic": s["topic"], "key": key, "kind": s["kind"], "targets": []})
            tid = by_key[key]
            t = topics[tid]
            if KIND_WEIGHT.get(s["kind"], 0) > KIND_WEIGHT.get(t["kind"], 0):
                t["kind"] = s["kind"]
                t["topic"] = s["topic"]
            if s["target"] not in t["targets"]:
                t["targets"].append(s["target"])
            toks = key.split()
            for size in range(1, min(MAX_NGRAM, len(toks)) + 1):
                for i in range(len(toks) - size + 1):
                    sub = " ".join(toks[i:i + size])
                    exact = 1 if sub == key else 0
                    slot = phrases.setdefault(sub, {})
                    slot[tid] = max(slot.get(tid, 0), exact)
        return TopicIndex({
            "topics": topics,
            "phrases": {k: [[tid, ex] for tid, ex in v.items()] for k, v in phrases.items()},
        })

    def to_json(self) -> dict:
        return {"topics": self.topics, "phrases": self.phrases}

    # ---------------------------------------------------------------- query
    def match(self, text: str, limit: int = 12) -> list[TopicHit]:
        toks = normalize(text)
        if not toks:
            return []
        best: dict[int, float] = {}
        n = len(toks)
        for size in range(min(MAX_NGRAM, n), 0, -1):
            for i in range(n - size + 1):
                entries = self.phrases.get(" ".join(toks[i:i + size]))
                if not entries:
                    continue
                # long phrases are specific; a phrase pointing at many topics is not
                # a phrase pointing at many topics is less specific, but a word like "revenue"
                # that names 30 rows must still fire, so the discount is capped at 2x
                base = size * size / min(1 + 0.5 * (len(entries) - 1), 2.0)
                for tid, exact in entries:
                    t = self.topics[tid]
                    sc = base * KIND_WEIGHT.get(t["kind"], 1.0) * (1.25 if exact else 1.0)
                    if len(t["targets"]) > 3:
                        sc *= 0.8
                    if sc > best.get(tid, 0.0):
                        best[tid] = sc
        hits = [TopicHit(self.topics[i]["topic"], round(s, 3), self.topics[i]["kind"],
                         self.topics[i]["targets"])
                for i, s in best.items()]
        hits.sort(key=lambda h: (-h.score, h.topic))
        return hits[:limit]


# --------------------------------------------------------------------------- #
_TERM_STOP = STOPWORDS | {"table", "column", "row", "page", "slide", "section",
                          "percent", "pct", "thousand", "million", "billion"}


def candidate_terms(text: str, min_count: int = 2, max_terms: int = 40) -> list[str]:
    """Repeated 2-3 word phrases that are not all stopwords: cheap topic candidates."""
    words = [m.group(0) for m in _WORD.finditer(text.lower())]
    counts: dict[str, int] = {}
    surface: dict[str, str] = {}
    for size in (2, 3):
        for i in range(len(words) - size + 1):
            gram = [w.strip(".,;:'$%") for w in words[i:i + size]]
            if gram[0] in _TERM_STOP or gram[-1] in _TERM_STOP:
                continue
            # words only: "$34,500 thousand" and "50% of revenue" are not topics
            if not all(w.isalpha() and len(w) > 2 for w in gram):
                continue
            key = norm_key(" ".join(gram))
            if not key or len(key.split()) < 2:
                continue
            counts[key] = counts.get(key, 0) + 1
            surface.setdefault(key, " ".join(gram))
    ranked = sorted(((c, k) for k, c in counts.items() if c >= min_count),
                    key=lambda x: (-x[0], -len(x[1])))
    return [surface[k] for _, k in ranked[:max_terms]]
