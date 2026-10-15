"""Everything either side of the forward pass: tokens in, Jev's wire body out.

The backends (`coreml`, `onnx_rt`, `mlxrt`, and torch) differ only in how they turn four arrays
into one `(B, K)` logits array. This module owns the rest, and it owns it once, so a backend
cannot quietly answer a different question from the one `presage/llm/laya.py` answers:

  sequences   `laya.build_sequence`, imported, not copied
  answers     `laya.temp_bucket` and `laya.confidence_from_probs`, imported, then `jev.parse_answers`
  prefixes    the instruction head and the option texts are tokenized once per question object and
              cached, so a gate call only tokenizes the four transcript lines

The tokenizer is `tokenizers.Tokenizer`, which presage already depends on for the STT path, rather
than `transformers.AutoTokenizer`. Same vocabulary, same ids (checked in `bench.py --check`), and
it means the exported backends need neither transformers nor torch at run time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from presage.llm.jev import Decision, DecisionError, Q, parse_answers
from presage.llm.laya import (
    QTYPES, LayaDecider, build_sequence, confidence_from_probs, render_options, serialize_state,
    temp_bucket,
)

NEG = -1e4


# ----------------------------------------------------------------------------- encoder shape tables
# The band mask and the rope tables depend only on the sequence length, so the host builds them
# once per bucket and the exported graph takes them as inputs. numpy only: the whole point of the
# Core ML path is that running the gate needs no torch.

def layer_kinds(enc: dict) -> list[str]:
    if enc.get("layer_types"):
        return list(enc["layer_types"])
    every = int(enc.get("global_attn_every_n_layers", 3))
    return ["full_attention" if i % every == 0 else "sliding_attention"
            for i in range(int(enc["num_hidden_layers"]))]


def rope_thetas(enc: dict) -> dict[str, float]:
    rp = enc.get("rope_parameters") or {}
    return {"full_attention": float((rp.get("full_attention") or {}).get(
                "rope_theta", enc.get("global_rope_theta", 160000.0))),
            "sliding_attention": float((rp.get("sliding_attention") or {}).get(
                "rope_theta", enc.get("local_rope_theta", 10000.0)))}


def band_np(L: int, window: int, dtype=np.float16) -> np.ndarray:
    i, j = np.arange(L)[:, None], np.arange(L)[None, :]
    return np.where(np.abs(i - j) <= window // 2, 0.0, NEG).astype(dtype)[None, None]


def rope_np(L: int, head_dim: int, theta: float, dtype=np.float16):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(L, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([t, t], -1)
    return np.cos(emb).astype(dtype)[None, None], np.sin(emb).astype(dtype)[None, None]


def table_names(enc: dict) -> list[str]:
    kinds = sorted(set(layer_kinds(enc)))
    return ["band", *(f"{p}_{k}" for k in kinds for p in ("cos", "sin"))]


def tables_np(enc: dict, L: int, dtype=np.float16) -> dict[str, np.ndarray]:
    head_dim = int(enc["hidden_size"]) // int(enc["num_attention_heads"])
    th = rope_thetas(enc)
    out = {"band": band_np(L, int(enc.get("local_attention", 128)), dtype)}
    for k in sorted(set(layer_kinds(enc))):
        out[f"cos_{k}"], out[f"sin_{k}"] = rope_np(L, head_dim, th[k], dtype)
    return out


# ----------------------------------------------------------------------------- tokenizer
class FastTok:
    """The five attributes `build_sequence` asks of a tokenizer, over `tokenizers.Tokenizer`."""

    def __init__(self, d: str | Path) -> None:
        from tokenizers import Tokenizer

        d = Path(d).expanduser()
        self.t = Tokenizer.from_file(str(d / "tokenizer.json"))
        cfg = json.loads((d / "tokenizer_config.json").read_text())

        def tok_id(key: str, default: str) -> tuple[str, int]:
            v = cfg.get(key) or default
            name = v["content"] if isinstance(v, dict) else v
            got = self.t.token_to_id(name)
            if got is None:
                raise DecisionError(f"tokenizer has no {key} {name!r}")
            return name, got

        self.mask_token, self.mask_token_id = tok_id("mask_token", "[MASK]")
        _, self.cls_token_id = tok_id("cls_token", "[CLS]")
        _, self.sep_token_id = tok_id("sep_token", "[SEP]")
        _, self.pad_token_id = tok_id("pad_token", "[PAD]")

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": self.t.encode(text, add_special_tokens=add_special_tokens).ids}


def load_tokenizer(model_dir: str | Path):
    d = Path(model_dir).expanduser() / "tokenizer"
    try:
        return FastTok(d)
    except Exception:  # noqa: BLE001  any tokenizers problem falls back to the reference loader
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(d))


# ----------------------------------------------------------------------------- one question
def question_shape(q: Q) -> dict:
    """`LayaDecider._internal` without an instance. Called through the class on purpose: the
    option rendering (`{label: None}` rather than `{label: label}`) is the difference between the
    numbers in data/gate/RESULTS.md and a different model, and it must not be re-typed here."""
    return LayaDecider._internal(None, q)  # type: ignore[arg-type]


class Prefixes:
    """Per-question token prefix, cached. The state is the only thing that changes call to call."""

    def __init__(self, tok, cfg: dict) -> None:
        self.tok, self.cfg = tok, cfg
        self._cache: dict[tuple, tuple] = {}

    def key(self, q: Q) -> tuple:
        return (q.name, q.kind, q.instructions, tuple(q.options or ()), q.lo, q.hi)

    def head_for(self, q: Q) -> tuple[dict, list[int], list[int], list[int], int]:
        k = self.key(q)
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        qi = question_shape(q)
        mask = self.tok.mask_token
        opts = render_options(qi)
        ins = str(qi["ins"]).replace(mask, " ")
        head = self.tok("%s question: %s" % (qi["t"], ins), add_special_tokens=False)["input_ids"]
        opt_ids = [[self.tok.mask_token_id]
                   + self.tok(" " + o.replace(mask, " "), add_special_tokens=False)["input_ids"][:48]
                   for o in opts]
        hit = (qi, head, [i for o in opt_ids for i in o], [len(o) for o in opt_ids], len(opts))
        self._cache[k] = hit
        return hit


def build_ids(tok, prefixes: Prefixes, state: str, q: Q, cfg: dict) -> tuple[list[int], list[int], dict]:
    """`laya.build_sequence` with the head half taken from the cache."""
    qi, head, flat, lens, n = prefixes.head_for(q)
    budget = cfg["head_max_len"] - sum(lens)
    if budget < 16:                         # the slow path handles its own re-truncation
        ids, markers = build_sequence(tok, state, qi, cfg["max_len"], cfg["head_max_len"])
        return ids, markers, qi
    head = head[:max(8, budget)]
    ids = [tok.cls_token_id, *head, tok.sep_token_id]
    markers, at = [], len(ids)
    for ln in lens:
        markers.append(at)
        at += ln
    ids.extend(flat)
    ids.append(tok.sep_token_id)
    room = max(0, cfg["max_len"] - len(ids) - 1)
    st = tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    ids = ids + st[:room] + [tok.sep_token_id]
    return ids[:cfg["max_len"]], [m for m in markers if m < cfg["max_len"]], qi


# ----------------------------------------------------------------------------- arrays
def pack(rows: list[list[int]], markers: list[list[int]], qtypes: list[int], type_table: np.ndarray,
         pad_id: int, L: int, K: int) -> dict[str, np.ndarray]:
    n = len(rows)
    ids = np.full((n, L), pad_id, dtype=np.int32)
    bias = np.full((n, 1, 1, L), NEG, dtype=np.float32)
    mm = np.zeros((n, K, L), dtype=np.float32)
    for r, row in enumerate(rows):
        ids[r, :len(row)] = row
        bias[r, 0, 0, :len(row)] = 0.0
        for i, p in enumerate(markers[r][:K]):
            mm[r, i, p] = 1.0
    return {"input_ids": ids, "pad_bias": bias, "type_vec": type_table[qtypes].astype(np.float32),
            "marker_map": mm}


# ----------------------------------------------------------------------------- answers
def wire_body(questions: list[Q], shapes: list[dict], ks: list[int], logits: np.ndarray,
              cfg: dict, tokens: int, model_name: str = "laya-fast") -> dict:
    """Jev's response body from raw marker logits. Same temperatures, same rounding as `laya.py`."""
    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    by_opts = cfg.get("temperature_by_options", {})
    answers: dict[str, dict] = {}
    for r, q in enumerate(questions):
        qi, k = shapes[r], ks[r]
        qt = QTYPES[qi["t"]]
        temp = by_opts.get(temp_bucket(qt, k), temps[qt])
        z = np.asarray(logits[r, :k], dtype=np.float64) / temp
        p = np.exp(z - z.max())
        p = p / p.sum()
        ext = {"act_probability": 0.0, "temperature": float(temp), "bucket": temp_bucket(qt, k)}
        if qi["t"] == "choice":
            keys = list(qi["crit"].keys())
            answers[q.name] = {"type": "choice", "choice": keys[int(p.argmax())],
                               "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                               "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
        elif qi["t"] == "score":
            answers[q.name] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                               "legend": {str(i): c for i, c in enumerate(qi["crit"])},
                               "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                               "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
        else:
            answers[q.name] = {"type": "noul", "noul": round(float(p[1]), 4), "rl_agent": ext}
    return {"model": model_name, "answers": answers,
            "usage": {"input_tokens": int(tokens), "output_tokens": 0}}


# ----------------------------------------------------------------------------- backends
class Backend(Protocol):
    name: str

    def buckets(self) -> tuple[int, ...]: ...
    def run(self, ids: np.ndarray, pad_bias: np.ndarray, type_vec: np.ndarray,
            marker_map: np.ndarray) -> np.ndarray: ...


# ----------------------------------------------------------------------------- bench helper
def gate_inputs(dec: Any, state: str, L: int, questions: list[Q] | None = None):
    """The four arrays for one gate call, built off a loaded `LayaDecider` (bench and check only)."""
    from presage.llm.laya import GATE_QUESTIONS
    questions = questions or GATE_QUESTIONS
    pre = Prefixes(dec.tok, dec.cfg)
    rows, markers, qtypes, shapes = [], [], [], []
    for q in questions:
        ids, mk, qi = build_ids(dec.tok, pre, state, q, dec.cfg)
        rows.append(ids)
        markers.append(mk)
        qtypes.append(QTYPES[qi["t"]])
        shapes.append(qi)
    K = max(len(m) for m in markers)
    table = np.asarray(dec.model.type_emb.weight.detach().float().cpu().numpy())
    b = pack(rows, markers, qtypes, table, dec.tok.pad_token_id, L, K)
    return (b["input_ids"], b["pad_bias"], b["type_vec"], b["marker_map"],
            [len(m) for m in markers])


def percentiles(xs: list[float]) -> tuple[float, float]:
    s = sorted(xs)
    return (s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2]),
            s[min(len(s) - 1, int(0.95 * len(s)))])


def rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
