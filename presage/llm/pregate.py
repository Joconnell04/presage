"""PreGate: the question gate's first stage, on the embedding presage already computes.

`Brain._gate_decider` asks three questions of every finished line -- `kind`, `needs_lookup`,
`complete` -- and pays 150-250 ms to hosted Jev for each one. Most lines do not need a 300M-parameter
model to answer: "yeah exactly", "Me: so on the revenue side", "what if the fee were 40 bps" are not
close calls. PreGate answers those locally in about 0.2 ms from three linear heads over the
potion-base-8M static embedding `presage/ingest/search.py` already loads, plus ~70 cheap features
(speaker, mode, final punctuation mark, trailing filler, question and what-if cues, length). When it
is not confident it says so and returns nothing, and the hosted gate runs as before.

  weights      ~/.cache/presage/pregate/  (heads.npz + meta.json, 311 KB; rebuilt by
               data/gate/pregate/train.py, which imports the feature builder below so the
               features a head was trained on and the features it is served cannot drift apart)
  cost         one model2vec encode of two short strings (~0.09 ms) plus three matmuls
  calibration  a temperature on each head, fit on the dev splits (isotonic loses a 5-fold CV
               inside dev); test ECE 0.006 to 0.033. `p` is the minimum over the three heads and
               `p_gate` the minimum over the two the brain acts on -- which is the one that
               decides whether to defer, since nothing reads `complete` today.
               `PRESAGE_CALIB=real` serves `meta.real.json` instead: the same heads with the
               temperatures E61 refit on real dev rows and the simulator transcripts.
  threshold    0.75, chosen on dev: the lowest threshold whose locally-decided dev rows clear
               97.5% on `kind` and 98.5% on `needs_lookup` on the synthetic AND the real dev
               split. Test then measured at it.

Two ways to use it.

  PreGate.load().decide(lines, mode, speaker) -> dict | None
      The direct call. `lines` is the last few transcript lines, newest last, each prefixed
      "Me: " or "Them: " (a list, or one newline-joined string); `mode` is a presage mode id or
      label; `speaker` is "Me" or "Them" and defaults to the last line's own prefix. Returns
      `{"kind", "needs_lookup", "complete", "p", "p_gate", "p_defer", "p_kind",
      "p_needs_lookup", "p_complete", "probs", "ms", "decided"}`, or None when `p_defer` is under
      the threshold. `decide(..., force=True)` always answers, which is what an eval wants.

  PreGateDecider  -- the `presage.llm.jev.Decider` contract, with one documented extension:
      `decide(state, questions)` returns None instead of `dict[str, Decision]` when the line is
      not one it can call, or when the questions are not the gate's three, or when the state is
      not the string `_gate_decider` builds. A caller that cannot handle None must wrap it:

          decider = cascade(PreGateDecider(), JevDecider())

      `cascade` is a plain `Decider` -- it never returns None -- so it drops into `Brain.decider`
      with no change to brain.py, and `cascade(...).stats` counts how many lines each stage took.

  uv run python -m presage.llm.pregate --selftest
  uv run python -m presage.llm.pregate --bench
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("presage.llm.pregate")

WEIGHTS_DIR = Path(os.environ.get("PRESAGE_PREGATE_DIR") or (Path.home() / ".cache" / "presage" / "pregate"))
DENSE_MODEL = os.environ.get("PRESAGE_DENSE_MODEL", "minishlab/potion-base-8M")
META_NAME = "meta.json"
KINDS = ["question", "what-if", "request", "objection", "none"]
MODES = ["call", "sales", "interview", "pitch"]
MODE_LABELS = {"call": "call", "client call": "call", "sales": "sales", "sales call": "sales",
               "interview": "interview", "pitch": "pitch"}
DEFAULT_THRESHOLD = 0.75        # chosen on the dev splits, not on test; see docs/experiments/E22-pregate.md
# Which confidence has to clear the threshold.
#   "gate"  min(p_kind, p_needs_lookup) -- the brain's own `conf`, and all the live path reads:
#           `Brain._on_final` acts on kind, needs_files and confidence and never looks at
#           `complete`, so letting `complete` veto a line costs coverage and buys nothing.
#   "all3"  min over all three heads. Switch to this the day anything acts on `complete`.
DEFAULT_DEFER_ON = "gate"

# --------------------------------------------------------------------------- the lexicon
# A lexicon, not a tagger, and deliberately the same one `data/gate/truncate2.py` scores cut-off
# lines with: the point of the last-word-class features is that "where the line stops" means the
# same thing here as it did when the training rows were cut.

FILLER = {"um", "uh", "uhm", "erm", "er", "ah", "hmm", "mm", "mhm", "hm", "huh"}
DISCOURSE = {"like", "right", "okay", "ok", "well", "actually", "basically", "literally",
             "obviously", "honestly", "anyway", "anyways", "yeah", "yep", "yes", "no", "mean",
             "know", "sorry"}
DET = {"the", "a", "an", "this", "that", "these", "those", "my", "our", "your", "their", "his",
       "her", "its", "some", "any", "no", "every", "each", "another", "both", "either", "neither",
       "which", "what"}
PREP = {"of", "to", "in", "on", "at", "by", "for", "from", "with", "into", "about", "over",
        "under", "between", "through", "during", "against", "across", "around", "after", "before",
        "than", "per", "via", "within", "without", "upon", "toward", "towards", "onto", "off",
        "out", "up", "down", "along", "among", "versus", "vs", "like", "as", "near", "behind",
        "beyond", "past", "since", "until", "till"}
CONJ = {"and", "or", "but", "so", "because", "if", "when", "while", "as", "that", "whether",
        "since", "unless", "although", "though", "plus", "nor", "yet", "once", "whereas",
        "cause", "cuz"}
PRON = {"i", "we", "you", "they", "he", "she", "it", "me", "us", "them", "him", "her", "who",
        "whom", "whose", "something", "anything", "everything", "nothing", "someone", "anyone",
        "everyone", "one", "ours", "yours", "theirs", "mine", "himself", "herself", "themselves",
        "itself", "myself", "ourselves"}
AUX = {"is", "are", "was", "were", "am", "be", "been", "being", "do", "does", "did", "done",
       "have", "has", "had", "will", "would", "can", "could", "should", "shall", "may", "might",
       "must", "gonna", "wanna", "going", "let", "lets", "ain't", "isn't", "aren't", "wasn't",
       "weren't", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "shouldn't",
       "wouldn't", "haven't", "hasn't"}
WH = {"what", "when", "where", "why", "how", "which", "who"}
DEG = {"just", "really", "very", "quite", "pretty", "kind", "sort", "much", "more", "less",
       "most", "all", "even", "still", "also", "too", "almost", "maybe", "probably",
       "definitely", "totally"}
NEG = {"not", "n't", "never"}
NUMWORD = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
           "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
           "ninety", "hundred", "thousand", "million", "billion", "percent", "half", "first",
           "second", "third", "k", "x"}
BACKCHANNEL = {"yeah", "yep", "yes", "no", "ok", "okay", "right", "sure", "exactly", "totally",
               "cool", "nice", "great", "perfect", "gotcha", "mhm", "mm", "uh-huh", "absolutely",
               "definitely", "agreed", "true", "fair", "correct", "understood", "thanks", "thank"}
MODAL_START = {"do", "does", "did", "can", "could", "would", "will", "is", "are", "was", "were",
               "have", "has", "had", "should", "shall", "may", "might", "am"}

ELL = "…"
WORD_RE = re.compile(r"[A-Za-z0-9$%'’°.-]*[A-Za-z0-9$%]+[A-Za-z0-9$%'’.-]*")
TRAIL_RE = re.compile(r"[\s\"')\]]+$")
SPEAKER_RE = re.compile(r"^\s*(Me|Them)\s*:\s*", re.I)
CLASSES = ["content", "number", "filler", "discourse", "aux", "pron", "det", "wh", "conj",
           "prep", "neg", "deg"]


def calib_tag(calib: str | None = None) -> str:
    """The calibration set to serve. `PRESAGE_CALIB=real` picks the E61 refit; unset is shipped."""
    tag = (calib if calib is not None else os.environ.get("PRESAGE_CALIB", "")).strip().lower()
    return "" if tag in ("", "default", "ship", "shipped") else tag


def meta_path(path: str | Path | None = None, calib: str | None = None) -> Path:
    """Which `meta.json` the heads are served with.

    The three heads are the same arrays either way -- a calibration is only the per-head
    temperature and the defer threshold -- so a refit is a sibling meta beside `heads.npz`:

        meta.json        shipped, temperatures fitted on the pooled dev splits (mostly synthetic)
        meta.real.json   E61, refit on real dev rows + the simulator transcripts

    `PRESAGE_CALIB=real` selects the second. A tag whose file is not there warns and falls back.
    """
    d = Path(path).expanduser() if path else WEIGHTS_DIR
    tag = calib_tag(calib)
    if tag:
        alt = d / f"meta.{tag}.json"
        if alt.is_file():
            return alt
        log.warning("PRESAGE_CALIB=%s but no %s; using the shipped calibration", tag, alt.name)
    return d / META_NAME


def tclass(tok: str) -> str:
    """The word class of one token, by the same rules data/gate/truncate2.py cuts by."""
    t = tok.lower().strip(".,;:!?…’'\"()")
    if not t:
        return "other"
    if re.fullmatch(r"[\d$%.,'-]*\d[\d$%.,'-]*", t) or t in NUMWORD:
        return "number"
    for s, name in ((FILLER, "filler"), (DISCOURSE, "discourse"), (AUX, "aux"), (PRON, "pron"),
                    (DET, "det"), (WH, "wh"), (CONJ, "conj"), (PREP, "prep"), (NEG, "neg"),
                    (DEG, "deg")):
        if t in s:
            return name
    return "content"


def strip_speaker(line: str) -> tuple[str, str | None]:
    """("what if we charged 1%", "Them") from "Them: what if we charged 1%"."""
    s = str(line or "")
    m = SPEAKER_RE.match(s)
    if not m:
        return s.strip(), None
    return s[m.end():].strip(), m.group(1).capitalize()


# --------------------------------------------------------------------------- cheap features
# Every entry is (name, fn) so the names travel with the weights and a head trained on one layout
# refuses to load against another. Order is the vector's order; append, never insert.

def _mark(text: str) -> str:
    t = TRAIL_RE.sub("", text)
    if t.endswith(ELL) or t.endswith("..") or t.endswith(". . ."):
        return "ellipsis"
    if not t:
        return "none"
    c = t[-1]
    return {"?": "question", ".": "period", "!": "bang", ",": "comma", ";": "comma", ":": "comma",
            "-": "dash", "–": "dash", "—": "dash"}.get(c, "none" if c.isalnum() else "other")


def _feat_names() -> list[str]:
    names = ["is_me", *[f"mode_{m}" for m in MODES],
             *[f"mark_{m}" for m in ("question", "period", "bang", "comma", "ellipsis", "dash",
                                     "other", "none")],
             "inner_ellipsis", "qmark_anywhere",
             "n_words", "log_words", "n_chars", "words_le3", "words_ge15",
             *[f"last_{c}" for c in CLASSES],
             "last_filler", "trailing_filler_phrase", "last_is_function",
             "starts_wh", "starts_modal", "has_wh", "how_many_much", "indirect_ask",
             "starts_upper",
             "cue_what_if", "cue_if_clause", "cue_instead", "cue_assume", "cue_scenario",
             "cue_can_you", "cue_show_me", "cue_send", "cue_lets_look",
             "cue_but", "cue_worry", "cue_too", "cue_unsure", "cue_however",
             "backchannel_only", "starts_backchannel", "short_no_verb",
             "has_digit", "has_pct", "has_numword", "has_currency",
             "prev_is_me", "prev_is_question", "prev_exists", "bias"]
    return names


FEAT_NAMES = _feat_names()
N_CHEAP = len(FEAT_NAMES)
FEAT_IX = {n: i for i, n in enumerate(FEAT_NAMES)}
# The two features the transcriber's trailing "..." lights up. E18: the marker alone is nearly
# decisive for `complete`, so the training script also fits a head with these zeroed and both
# numbers are reported.
ELLIPSIS_FEATURES = (FEAT_IX["mark_ellipsis"], FEAT_IX["inner_ellipsis"])

_HAS = {
    "cue_what_if": ("what if", "what would happen if", "what happens if"),
    "cue_if_clause": ("if we ", "if you ", "if the ", "if it ", "if that ", "if they "),
    "cue_instead": ("instead", "rather than", "in place of"),
    "cue_assume": ("assuming", "suppose", "say we", "let's say", "lets say", "imagine"),
    "cue_scenario": ("scenario", "downside case", "upside case", "sensitivit", "best case",
                     "worst case"),
    "cue_can_you": ("can you", "could you", "would you", "can we", "could we", "mind if"),
    "cue_show_me": ("show me", "show us", "walk me", "walk us", "walk through", "pull up",
                    "bring up", "go back to", "open the", "take us through"),
    "cue_send": ("send me", "send us", "send over", "share the", "share that", "email me",
                 "follow up with"),
    "cue_lets_look": ("let's look", "lets look", "let's see", "lets see", "let's go", "can i see"),
    "cue_worry": ("concern", "worried", "worry", "nervous", "risk", "problem", "issue with",
                  "pushback", "skeptic", "sceptic"),
    "cue_too": ("too high", "too low", "too expensive", "too much", "too aggressive",
                "too optimistic", "too conservative", "seems high", "seems low", "feels high"),
    "cue_unsure": ("not sure", "i don't know", "i dont know", "not convinced", "hard to see",
                   "struggling to", "i doubt", "hesitant"),
    "cue_however": ("however", "that said", "although", "even so", "on the other hand",
                    "the thing is"),
    "indirect_ask": ("i'm curious", "im curious", "curious", "wondering", "any idea", "do you know",
                     "tell me", "remind me", "i'd love to know", "talk me through", "help me"),
    "has_pct": ("%", "percent", "basis point", " bps", "bps ", "per cent"),
    "has_currency": ("$", "£", "€", "dollar", "million", "billion", " k ", "usd"),
}
_PHRASE_FILLER = ("you know", "i mean", "sort of", "kind of", "like", "um", "uh", "so", "and",
                  "but", "or", "right")


def cheap_features(text: str, prev: str = "", mode: str = "call", speaker: str | None = None) -> list[float]:
    """The ~70 hand-written features for one line. Pure Python; no numpy, no model."""
    body, pfx = strip_speaker(text)
    who = (speaker or pfx or "Them").capitalize()
    prev_body, prev_pfx = strip_speaker(prev or "")
    mode_id = MODE_LABELS.get(str(mode or "call").strip().lower(), str(mode or "call").strip().lower())
    low = body.lower()
    toks = WORD_RE.findall(body)
    nw = len(toks)
    last = tclass(toks[-1]) if toks else "other"
    mark = _mark(body)
    core = TRAIL_RE.sub("", body)
    v: list[float] = [1.0 if who == "Me" else 0.0]
    v += [1.0 if mode_id == m else 0.0 for m in MODES]
    v += [1.0 if mark == m else 0.0 for m in ("question", "period", "bang", "comma", "ellipsis",
                                              "dash", "other", "none")]
    v.append(1.0 if (ELL in core[:-1] or "..." in core[:-3]) else 0.0)
    v.append(1.0 if "?" in body else 0.0)
    v += [min(nw, 40) / 20.0, math.log1p(nw) / 4.0, min(len(body), 240) / 120.0,
          1.0 if nw <= 3 else 0.0, 1.0 if nw >= 15 else 0.0]
    v += [1.0 if last == c else 0.0 for c in CLASSES]
    v.append(1.0 if last == "filler" else 0.0)
    v.append(1.0 if any(low.rstrip(" .,?!" + ELL).endswith(p) for p in _PHRASE_FILLER) else 0.0)
    v.append(1.0 if last in ("det", "prep", "conj", "pron", "aux", "wh", "deg", "neg") else 0.0)
    first = toks[0].lower() if toks else ""
    v.append(1.0 if first in WH else 0.0)
    v.append(1.0 if first in MODAL_START else 0.0)
    v.append(1.0 if any(w.lower() in WH for w in toks) else 0.0)
    v.append(1.0 if ("how many" in low or "how much" in low) else 0.0)
    v.append(1.0 if any(s in low for s in _HAS["indirect_ask"]) else 0.0)
    v.append(1.0 if body[:1].isupper() else 0.0)
    for key in ("cue_what_if", "cue_if_clause", "cue_instead", "cue_assume", "cue_scenario",
                "cue_can_you", "cue_show_me", "cue_send", "cue_lets_look"):
        v.append(1.0 if any(s in low for s in _HAS[key]) else 0.0)
    v.append(1.0 if (low.startswith("but ") or " but " in low) else 0.0)
    for key in ("cue_worry", "cue_too", "cue_unsure", "cue_however"):
        v.append(1.0 if any(s in low for s in _HAS[key]) else 0.0)
    words_l = [w.lower() for w in toks]
    v.append(1.0 if words_l and all(w in BACKCHANNEL or w in FILLER for w in words_l) else 0.0)
    v.append(1.0 if first in BACKCHANNEL else 0.0)
    v.append(1.0 if (nw <= 4 and not any(tclass(w) == "aux" for w in toks)) else 0.0)
    v.append(1.0 if any(ch.isdigit() for ch in body) else 0.0)
    v.append(1.0 if any(s in low for s in _HAS["has_pct"]) else 0.0)
    v.append(1.0 if any(w in NUMWORD for w in words_l) else 0.0)
    v.append(1.0 if any(s in low for s in _HAS["has_currency"]) else 0.0)
    v.append(1.0 if (prev_pfx or "").capitalize() == "Me" else 0.0)
    v.append(1.0 if prev_body.rstrip().endswith("?") else 0.0)
    v.append(1.0 if prev_body else 0.0)
    v.append(1.0)                                   # bias, so a head needs no separate intercept
    assert len(v) == N_CHEAP, f"{len(v)} features, layout says {N_CHEAP}"
    return v


def split_lines(lines: Sequence[str] | str) -> list[str]:
    if isinstance(lines, str):
        return [ln for ln in lines.splitlines() if ln.strip()]
    return [str(ln) for ln in lines if str(ln).strip()]


def build_matrix(rows: list[dict], embed) -> Any:
    """(n, 2*dim + N_CHEAP) for a list of `{text, context, mode}` rows. `embed` takes a list of
    strings and returns L2-normalised vectors; the training script hands in a batched encoder and
    `PreGate` hands in its own one-shot one, so the layout cannot differ between the two."""
    import numpy as np

    lasts, prevs, cheap = [], [], []
    for r in rows:
        ctx = list(r.get("context") or [])
        prev = ctx[-1] if ctx else ""
        body, pfx = strip_speaker(r["text"])
        lasts.append(body)
        prevs.append(strip_speaker(prev)[0])
        cheap.append(cheap_features(r["text"], prev, r.get("mode", "call"), r.get("speaker") or pfx))
    a = np.asarray(embed(lasts), dtype="float32")
    b = np.asarray(embed(prevs), dtype="float32")
    b[[i for i, p in enumerate(prevs) if not p]] = 0.0
    return np.hstack([a, b, np.asarray(cheap, dtype="float32")])


# --------------------------------------------------------------------------- heads
def _softmax(z):
    import numpy as np

    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(z):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _forward(head: dict, x):
    """Logits from a head's arrays. `W` alone is linear; `W1/b1/W2/b2` is one relu layer."""
    import numpy as np

    if "W1" in head:
        h = np.maximum(0.0, x @ head["W1"] + head["b1"])
        return h @ head["W2"] + head["b2"]
    return x @ head["W"] + head.get("b", 0.0)


def _apply_iso(p, x_knots, y_knots):
    import numpy as np

    return np.interp(p, x_knots, y_knots)


class PreGate:
    """Three calibrated heads over the potion embedding. Load once, call per line."""

    def __init__(self, weights: dict, meta: dict, threshold: float | None = None) -> None:
        self.w, self.meta = weights, meta
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", DEFAULT_THRESHOLD))
        self.defer_on = str(meta.get("defer_on", DEFAULT_DEFER_ON))
        self.dim = int(meta["dim"])
        self.meta_file = META_NAME
        self._model = None
        if list(meta.get("features", FEAT_NAMES)) != FEAT_NAMES:
            raise RuntimeError("pregate weights were trained on a different feature layout; "
                               "rebuild them with data/gate/pregate/train.py")

    # ---- load ----
    @classmethod
    def load(cls, path: str | Path | None = None, threshold: float | None = None,
             calib: str | None = None) -> "PreGate":
        import numpy as np

        d = Path(path).expanduser() if path else WEIGHTS_DIR
        npz, meta = d / "heads.npz", meta_path(d, calib)
        if not npz.exists():
            raise FileNotFoundError(f"no pregate weights at {npz}; build them with "
                                    f"`uv run python data/gate/pregate/train.py`")
        z = np.load(npz)
        w: dict[str, dict] = {}
        for key in z.files:
            head, _, name = key.partition(".")
            w.setdefault(head, {})[name] = z[key]
        g = cls(w, json.loads(meta.read_text()), threshold)
        g.meta_file = meta.name
        g.model()                                   # pay the 0.8 s model2vec load now, not mid-call
        return g

    @classmethod
    def available(cls, path: str | Path | None = None) -> bool:
        d = Path(path).expanduser() if path else WEIGHTS_DIR
        return (d / "heads.npz").exists() and (d / "meta.json").exists()

    def model(self):
        if self._model is None:
            from presage.ingest.search import Dense
            self._model = Dense().model()           # the same StaticModel the index uses
        return self._model

    def _embed(self, texts: list[str]):
        import numpy as np

        v = np.asarray(self.model().encode(texts), dtype="float32")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1, n)

    # ---- decide ----
    def vector(self, lines: Sequence[str] | str, mode: str = "call", speaker: str | None = None):
        import numpy as np

        got = split_lines(lines)
        if not got:
            raise ValueError("pregate needs at least one line")
        last, prev = got[-1], (got[-2] if len(got) > 1 else "")
        lb, pfx = strip_speaker(last)
        pb = strip_speaker(prev)[0]
        e = self._embed([lb, pb]) if pb else None
        if e is None:
            e = self._embed([lb])
            emb = np.hstack([e[0], np.zeros(self.dim, dtype="float32")])
        else:
            emb = np.hstack([e[0], e[1]])
        cheap = np.asarray(cheap_features(last, prev, mode, speaker or pfx), dtype="float32")
        return np.hstack([emb, cheap])[None, :]

    def decide(self, lines: Sequence[str] | str, mode: str = "call", speaker: str | None = None,
               force: bool = False) -> dict | None:
        """The three answers and the confidence, or None when the line should go to the slow gate."""
        t0 = time.perf_counter()
        x = self.vector(lines, mode, speaker)
        out = self.scores(x)
        out["ms"] = (time.perf_counter() - t0) * 1000.0
        out["p_defer"] = out["p_gate"] if self.defer_on == "gate" else out["p"]
        out["decided"] = bool(out["p_defer"] >= self.threshold)
        return out if (force or out["decided"]) else None

    def scores(self, x) -> dict:
        """Calibrated answers for a (1, D) or (n, D) feature matrix. No threshold applied."""
        import numpy as np

        single = x.ndim == 1
        if single:
            x = x[None, :]
        pk = _softmax(_forward(self.w["kind"], x) / float(self.meta["calib"]["kind"]["T"]))
        pn = self._binary("needs_lookup", x)
        pc = self._binary("complete", x)
        ki = pk.argmax(axis=1)
        p_kind = pk[np.arange(len(ki)), ki]
        p_needs = np.maximum(pn, 1 - pn)
        p_comp = np.maximum(pc, 1 - pc)
        p_gate = np.minimum(p_kind, p_needs)            # the brain's own confidence, verbatim
        p = np.minimum(p_gate, p_comp)
        rows = {
            "kind": [KINDS[i] for i in ki],
            "needs_lookup": (pn >= 0.5),
            "complete": (pc >= 0.5),
            "p": p, "p_gate": p_gate, "p_kind": p_kind, "p_needs_lookup": p_needs,
            "p_complete": p_comp, "probs": pk,
        }
        if not single and len(x) > 1:
            return rows
        return {"kind": rows["kind"][0], "needs_lookup": bool(rows["needs_lookup"][0]),
                "complete": bool(rows["complete"][0]), "p": float(p[0]),
                "p_gate": float(p_gate[0]),
                "p_kind": float(p_kind[0]), "p_needs_lookup": float(p_needs[0]),
                "p_complete": float(p_comp[0]),
                "probs": {k: float(v) for k, v in zip(KINDS, pk[0])}}

    def _binary(self, name: str, x):
        c = self.meta["calib"][name]
        z = _forward(self.w[name], x).reshape(-1)
        if c["how"] == "isotonic":
            return _apply_iso(_sigmoid(z / float(c.get("T", 1.0))),
                              self.w[name]["iso_x"], self.w[name]["iso_y"])
        return _sigmoid(z / float(c["T"]))


# --------------------------------------------------------------------------- the typed ask's closed slots
ASK_NPZ, ASK_META = "ask_heads.npz", "ask_meta.json"


class AskHeads:
    """Two linear heads for the typed ask's closed slots, beside the gate's own three.

    `presage.llm.ask` asks Jev five questions about a spoken line. Two of them -- `basis` and
    `compare_to` -- have no workbook vocabulary behind them: they are closed sets four and five
    options wide, written once. A linear head on the same potion embedding beats the 300M model on
    both, so they come off the request and ride here instead: E38 measured 92.5% and 95.0% against
    Jev's 75.0% and 90.4%, and the shipped heads measure 91.7% and 95.8% on the same held-out half
    (E38a), for 0.04 ms including the encode and no tokens.

    The vector is the line's own normalised embedding, which `Brain._embed` already computes; no
    cheap features, no previous line. Built by `data/gate/ask/train_heads.py`.
    """

    def __init__(self, weights: dict, meta: dict, threshold: float | None = None) -> None:
        self.w, self.meta = weights, meta
        self.dim = int(meta.get("dim", 0))
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.classes: dict[str, list] = {name: [None if c == "None" else c for c in cfg["classes"]]
                                         for name, cfg in (meta.get("slots") or {}).items()}
        self.temp: dict[str, float] = {name: float(cfg.get("T", 1.0))
                                       for name, cfg in (meta.get("slots") or {}).items()}
        self.floors: dict[str, float] = {name: float(cfg.get("threshold", self.threshold))
                                         for name, cfg in (meta.get("slots") or {}).items()}
        self._model = None

    @classmethod
    def available(cls, path: str | Path | None = None) -> bool:
        d = Path(path).expanduser() if path else WEIGHTS_DIR
        return (d / ASK_NPZ).exists() and (d / ASK_META).exists()

    @classmethod
    def load(cls, path: str | Path | None = None, threshold: float | None = None) -> "AskHeads":
        import numpy as np

        d = Path(path).expanduser() if path else WEIGHTS_DIR
        npz = d / ASK_NPZ
        if not npz.exists():
            raise FileNotFoundError(f"no ask heads at {npz}; build them with "
                                    f"`uv run python data/gate/ask/train_heads.py`")
        z = np.load(npz)
        w: dict[str, dict] = {}
        for key in z.files:
            head, _, name = key.partition(".")
            w.setdefault(head, {})[name] = z[key]
        return cls(w, json.loads((d / ASK_META).read_text()), threshold)

    def model(self):
        if self._model is None:
            from presage.ingest.search import Dense
            self._model = Dense().model()
        return self._model

    def embed(self, texts: list[str]):
        """`Brain._embed`'s contract, for a caller that has no embedder of its own."""
        import numpy as np

        v = np.asarray(self.model().encode(texts), dtype="float32")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1, n)

    def slots(self, vec, threshold: float | None = None) -> dict[str, tuple[Any, float]]:
        """`{slot: (value, p)}` for every head that clears the threshold; the others are absent
        and stay on the request. `vec` is one line's normalised embedding, or the line itself."""
        import numpy as np

        if isinstance(vec, str):
            vec = self.embed([vec])
        x = np.asarray(vec, dtype="float32").reshape(1, -1)
        if self.dim and x.shape[1] != self.dim:
            return {}
        out: dict[str, tuple[Any, float]] = {}
        for name, cls in self.classes.items():
            head = self.w.get(name)
            if not head:
                continue
            floor = float(self.floors.get(name, self.threshold) if threshold is None else threshold)
            p = _softmax(_forward(head, x) / max(self.temp.get(name, 1.0), 1e-6))[0]
            i = int(p.argmax())
            if float(p[i]) >= floor:
                out[name] = (cls[i], round(float(p[i]), 3))
        return out


# --------------------------------------------------------------------------- the Decider contract
try:                                                # jev.py is pure httpx; importing it is cheap
    from presage.llm.jev import Decision, Q
except Exception:                                   # pragma: no cover - keeps this module standalone
    Decision = Q = None                             # type: ignore[assignment]

GATE_QUESTIONS = ("kind", "needs_lookup", "complete")
_STATE_MODE = re.compile(r"Mode:\s*([A-Za-z ]+?)\s*\.")
_STATE_WHO = re.compile(r"last line is spoken by\s+(Me|Them)\s*:")


def parse_gate_state(state: str) -> tuple[list[str], str, str] | None:
    """(lines, mode, speaker) out of the string `Brain._gate_decider` builds, or None.

    The brain's state is a fixed header, a blank line, then the last four transcript lines. This
    reads it back rather than asking the brain to pass structure, which is what lets PreGate sit
    behind the existing `Decider` contract with brain.py untouched.
    """
    s = str(state or "")
    head, sep, body = s.partition("\n\n")
    if not sep or "judge ONLY the last line" not in head:
        return None
    m, w = _STATE_MODE.search(head), _STATE_WHO.search(head)
    lines = split_lines(body)
    if not m or not w or not lines:
        return None
    mode = MODE_LABELS.get(m.group(1).strip().lower())
    return (lines, mode, w.group(1).capitalize()) if mode else None


class PreGateDecider:
    """`Decider`, with one documented extension: `decide()` may return None.

    None means "not mine": the weights are missing, the questions are not the gate's three, the
    state is not the brain's gate state, or the heads are under the threshold. The caller then
    runs the real decider. `cascade()` below is the wrapper that does that and is itself an
    ordinary `Decider`, so nothing in brain.py has to learn about the extension.
    """

    provider = "pregate"
    kind = "pregate"

    def __init__(self, gate: PreGate | None = None, threshold: float | None = None,
                 weights: str | Path | None = None, calib: str | None = None) -> None:
        self.threshold = threshold
        self._weights = weights
        self._calib = calib                     # None reads PRESAGE_CALIB; "" forces the shipped one
        self.gate = gate
        self.stats = {"asked": 0, "answered": 0, "deferred": 0, "ms": 0.0}

    def ready(self) -> bool:
        if self.gate is None:
            if not PreGate.available(self._weights):
                return False
            try:
                self.gate = PreGate.load(self._weights, self.threshold, self._calib)
            except Exception as e:                  # noqa: BLE001  a missing head is a defer, not a crash
                log.warning("pregate not loaded: %s", e)
                return False
        return True

    async def decide(self, state: str, questions: list) -> dict | None:
        self.stats["asked"] += 1
        if not self.ready() or [q.name for q in questions] != list(GATE_QUESTIONS):
            self.stats["deferred"] += 1
            return None
        parsed = parse_gate_state(state)
        if parsed is None:
            self.stats["deferred"] += 1
            return None
        lines, mode, who = parsed
        got = self.gate.decide(lines, mode, who)
        if got is None:
            self.stats["deferred"] += 1
            return None
        self.stats["answered"] += 1
        self.stats["ms"] += got["ms"]
        ms = int(round(got["ms"]))
        return {
            "kind": Decision(got["kind"], got["p_kind"], got["p_kind"], probs=got["probs"], latency_ms=ms),
            "needs_lookup": Decision(got["needs_lookup"], got["p_needs_lookup"] if got["needs_lookup"]
                                     else 1 - got["p_needs_lookup"], got["p_needs_lookup"], latency_ms=ms),
            "complete": Decision(got["complete"], got["p_complete"] if got["complete"]
                                 else 1 - got["p_complete"], got["p_complete"], latency_ms=ms),
        }

    async def validate_key(self) -> tuple[bool, str]:
        return (True, "pregate ready") if self.ready() else (False, "no pregate weights")


class CascadeDecider:
    """PreGate first, the real decider when PreGate declines. An ordinary `Decider`."""

    def __init__(self, fast: PreGateDecider, slow: Any) -> None:
        self.fast, self.slow = fast, slow
        self.provider = getattr(slow, "provider", "cascade")
        self.kind = "pregate+" + str(getattr(slow, "kind", None) or type(slow).__name__.replace("Decider", "")).lower()
        self.stats = {"fast": 0, "slow": 0}

    def __repr__(self) -> str:
        return f"CascadeDecider(pregate -> {type(self.slow).__name__})"

    async def decide(self, state: str, questions: list) -> dict:
        got = await self.fast.decide(state, questions)
        if got is not None:
            self.stats["fast"] += 1
            return got
        self.stats["slow"] += 1
        return await self.slow.decide(state, questions)

    async def validate_key(self) -> tuple[bool, str]:
        return await self.slow.validate_key()

    async def aclose(self) -> None:
        close = getattr(self.slow, "aclose", None)
        if close:
            await close()


def cascade(fast: PreGateDecider | None, slow: Any) -> Any:
    """`cascade(PreGateDecider(), JevDecider())`. Returns `slow` untouched when PreGate is absent."""
    fast = fast or PreGateDecider()
    return CascadeDecider(fast, slow) if fast.ready() else slow


# --------------------------------------------------------------------------- cli
def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import statistics

    ap = argparse.ArgumentParser(description="the local pre-gate")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--calib", help="calibration tag, or the PRESAGE_CALIB env (e.g. real)")
    a = ap.parse_args(argv)
    g = PreGate.load(a.weights, a.threshold, a.calib)
    print(f"pregate {WEIGHTS_DIR if not a.weights else a.weights}  dim {g.dim}  "
          f"defer on {g.defer_on} below {g.threshold:.2f}  trained {g.meta.get('trained_utc')}  "
          f"calib {g.meta_file}")
    script = [
        ["Me: so this page is the fee waterfall", "Them: what if the fee moved to 40 bps instead of 25?"],
        ["Them: right", "Me: and that flows into the year five number"],
        ["Them: how many subscribers are in the platinum tier?"],
        ["Them: yeah, exactly"],
        ["Me: here is the churn view", "Them: could you pull up the cohort tab"],
        ["Them: I think that retention number feels high to me"],
        ["Them: so on the revenue side…"],
    ]
    if a.selftest or not a.bench:
        for lines in script:
            d = g.decide(lines, "call", force=True)
            flag = " " if d["decided"] else "*"
            print(f"  {flag} {d['kind']:<9} needs={str(d['needs_lookup']):<5} "
                  f"complete={str(d['complete']):<5} p={d['p_defer']:.3f}  {d['ms']:.2f} ms  "
                  f"| {lines[-1][:58]}")
        print("  * = under the threshold, so the hosted gate would run")
    if a.bench:
        for lines in script[:1]:
            for _ in range(20):
                g.decide(lines, "call", force=True)
        ms = []
        for _ in range(400):
            lines = script[len(ms) % len(script)]
            t = time.perf_counter()
            g.decide(lines, "call", force=True)
            ms.append((time.perf_counter() - t) * 1000)
        ms.sort()
        print(f"  decide()  mean {statistics.mean(ms):.3f} ms  p50 {ms[len(ms)//2]:.3f}  "
              f"p95 {ms[int(0.95*len(ms))]:.3f}  p99 {ms[int(0.99*len(ms))]:.3f}  (n={len(ms)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
