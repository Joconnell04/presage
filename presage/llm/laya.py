"""Laya: presage's question gate, local, on open weights, with `jev.py`'s exact contract.

Laya (convaiinnovations/laya, Apache 2.0) answers the same three primitives as TypeSafe's hosted
Jev -- `noul` (yes/no), `choice` (pick one), `score` (rate it) -- in one forward pass, with no text
generation, no network and no key. This module is the inference half only: it takes the `Q` list
`presage/llm/jev.py` takes and returns the same `dict[str, Decision]`, so `Brain._gate_decider`
does not change. `docs/research-laya.md` is the write-up; this file is that document's probe,
cleaned up, with the answer mapping imported from `jev.py` instead of copied.

How it works, read out of the model repo's own code (`rl_agent_api.py`, `rl_common.py`):

  sequence    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
  scoring     one [MASK] marker per option, a 2-layer transformer head over the encoder states,
              a linear scorer at each marker, softmax across that question's markers only
  batching    every question is one row of one batch, so N questions cost one forward pass
  answers     noul -> p[1]; choice -> argmax label; score -> sum(i * p_i); confidence -> 1 - H/log k
  calibration per (question type, option count) temperature from `rl_agent_config.json`, or from
              `rl_agent_config.<tag>.json` when `PRESAGE_CALIB=<tag>` is set (E61 writes `real`)

Measured on an M5 (docs/research-laya.md section 4): 81 ms p50 for the three-question gate on MPS
fp16, against 205 ms for hosted Jev; 20 s to load, which `preload=True` pays in the background.
fp16 is 2.1x faster than fp32 on MPS and the answers are identical to three decimals, so it is the
default there; torch CPU fp16 is emulated and 3.5 s, so CPU defaults to fp32.

The stock weights answer presage's own gate badly (3/10 on `kind`, 1/10 on `needs_lookup`), which is
why `make_decider` only reaches for this when asked or when Jev has no key. The fix is a fine-tune on
presage's own gate traffic; the training set and recipe live under `data/gate`.

  uv sync --extra laya
  uv run python -m presage.llm.laya --download      # 846 MB into ~/.cache/presage/laya
  uv run python -m presage.llm.laya --selftest      # the real gate, two states, printed
  uv run python -m presage.llm.laya --bench         # p50/p95 over 30 runs
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from presage.llm.jev import Decision, DecisionError, Q, parse_answers

log = logging.getLogger("presage.llm.laya")

REPO_ID = "convaiinnovations/laya"
CHECKPOINT = "typed-decisions"          # the fine-tuned checkpoint; `root` is near chance on these
DEFAULT_DIR = Path.home() / ".cache" / "presage" / "laya" / CHECKPOINT
FILES = ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
         "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")
PACKAGES = ("torch", "transformers", "safetensors")
DTYPES = ("fp32", "fp16", "bf16")


# ----------------------------------------------------------------------------- is it here at all
TUNED_DIR = DEFAULT_DIR.parent / "presage-gate"   # data/gate/train_laya.py writes the fine-tuned gate here


def weights_dir() -> Path:
    """Where the checkpoint lives. `PRESAGE_LAYA_DIR` overrides it; otherwise the gate fine-tuned on
    presage's own data (`presage-gate`) wins over the stock typed-decisions checkpoint, which
    measured 42% on the gate set against 98% after fine-tuning (data/gate/RESULTS.md)."""
    env = os.environ.get("PRESAGE_LAYA_DIR")
    if env:
        return Path(env).expanduser()
    if all((TUNED_DIR / f).is_file() for f in FILES):
        return TUNED_DIR
    return DEFAULT_DIR


def tuned_present() -> bool:
    """The fine-tuned gate is on disk (the only Laya worth using without a flag)."""
    return all((TUNED_DIR / f).is_file() for f in FILES)


# ----------------------------------------------------------------------------- which calibration
CONFIG_NAME = "rl_agent_config.json"


def calib_tag(calib: str | None = None) -> str:
    """The calibration set to serve. `PRESAGE_CALIB=real` picks the E61 refit; unset is shipped."""
    tag = (calib if calib is not None else os.environ.get("PRESAGE_CALIB", "")).strip().lower()
    return "" if tag in ("", "default", "ship", "shipped") else tag


def config_path(model_dir: str | Path | None = None, calib: str | None = None) -> Path:
    """Which `rl_agent_config` the gate reads.

    Everything about the checkpoint except the temperatures is the same file either way, so a
    calibration is a sibling config rather than a second copy of 842 MB of weights:

        rl_agent_config.json        shipped, fitted on the dev splits (mostly synthetic)
        rl_agent_config.real.json   E61, refit on real dev rows + the simulator transcripts

    `PRESAGE_CALIB=real` selects the second. A tag whose file is not there logs once and falls
    back to the shipped config rather than failing the gate, because a missing calibration is a
    worse answer, not no answer.
    """
    d = Path(model_dir).expanduser() if model_dir else weights_dir()
    tag = calib_tag(calib)
    if tag:
        alt = d / f"rl_agent_config.{tag}.json"
        if alt.is_file():
            return alt
        log.warning("PRESAGE_CALIB=%s but no %s; using the shipped calibration", tag, alt.name)
    return d / CONFIG_NAME


def load_config(model_dir: str | Path | None = None, calib: str | None = None) -> dict:
    """The gate's config, with `calib_file` recording which one was served.

    `presage/llm/laya_fast/decider.py` reads the same function, so the Core ML path and the torch
    path cannot be running different temperatures against the same weights.
    """
    p = config_path(model_dir, calib)
    cfg = json.loads(p.read_text())
    cfg["calib_file"] = p.name
    return cfg


def weights_present(model_dir: str | Path | None = None) -> bool:
    d = Path(model_dir).expanduser() if model_dir else weights_dir()
    return all((d / f).is_file() for f in FILES)


def missing_packages() -> list[str]:
    """The `laya` extra's imports that are not installed. Never imports them."""
    out = []
    for name in PACKAGES:
        try:
            if find_spec(name) is None:
                out.append(name)
        except Exception:  # noqa: BLE001  a broken install is a missing one
            out.append(name)
    return out


def available() -> bool:
    """True when this machine can run the local gate right now: weights on disk, torch installed."""
    return weights_present() and not missing_packages()


def _short(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def _why_not(model_dir: Path) -> str | None:
    """The one-line reason the gate cannot run, for the Settings page. None when it can."""
    gone = missing_packages()
    if gone:
        return f"{', '.join(gone)} not installed: uv sync --extra laya"
    if not weights_present(model_dir):
        return f"no weights in {_short(model_dir)}: uv run python -m presage.llm.laya --download"
    return None


# ----------------------------------------------------------------------------- Laya's own encoding
# Verbatim from the model repo's `rl_common.py`. Every byte of this matters: the checkpoint was
# trained against exactly this layout, and "level 0: " or a missing marker changes the answer.

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_options(q: dict) -> list[str]:
    """Option texts in label-index order. Noul is always [false, true], so p[1] is the noul."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(tok, state, q: dict, max_len: int, head_max_len: int,
                   option_order: list[int] | None = None, truncate_left: bool = False):
    """[CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]."""
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append([tok.mask_token_id]
                       + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def confidence_from_probs(p, k: int) -> float:
    """Jev-style confidence: 1 - normalised entropy of the answer distribution."""
    import numpy as np
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


def collate(items: list[dict], pad_id: int) -> dict:
    import torch
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "qtype": torch.tensor([it["qtype"] for it in items])}


# ----------------------------------------------------------------------------- where it runs
def pick_device(name: str | None = None):
    """The `torch.device` to run on. `name` wins; otherwise MPS, then CUDA, then CPU.

    Both the gate and `data/gate/train_laya.py` come through here, which is why it lives in the
    module the gate ships rather than in the trainer: inference on this Mac lands on MPS, and
    training on a rented CUDA box gets there with `--device cuda`.

    A device asked for by name and not actually present is an error, not a quiet fall back to CPU.
    Four epochs on CPU is days rather than minutes, and a run that silently starts is worse than a
    run that refuses to.
    """
    import torch

    def has(kind: str) -> bool:
        if kind == "mps":
            b = getattr(torch.backends, "mps", None)
            return bool(b is not None and b.is_available())
        if kind == "cuda":
            return bool(torch.cuda.is_available())
        return kind == "cpu"

    def fix(dev):
        # `torch.device("cuda")` carries no index, and `torch.cuda.set_device` refuses that. Pin it
        # to the current device so every caller gets a device it can name.
        if dev.type == "cuda" and dev.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return dev

    if name and name != "auto":
        dev = torch.device(name)
        if not has(dev.type):
            raise DecisionError(f"device {name!r} asked for, but torch reports no {dev.type} on this machine")
        return fix(dev)
    for kind in ("mps", "cuda"):
        if has(kind):
            return fix(torch.device(kind))
    return torch.device("cpu")


def sync(device) -> None:
    """Wait for the queued kernels. A no-op on CPU; both accelerators need it before a timer."""
    import torch
    if device is None:
        return
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache(device) -> None:
    """Hand the allocator's free blocks back. Called on a schedule during training, never per step."""
    import torch
    try:
        if device is not None and device.type == "mps":
            torch.mps.empty_cache()
        elif device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001  freeing cache must never take a run down
        pass


# ----------------------------------------------------------------------------- the model
_MODEL_CLS = None


def _decision_model_cls():
    """The published architecture, defined on first use so importing this module never needs torch.

    Inference only: no checkpointing, no dropout paths, no reward machinery. Parameter names match
    the published state dict exactly, which is what lets `load_state_dict` run strict.
    """
    global _MODEL_CLS
    if _MODEL_CLS is not None:
        return _MODEL_CLS
    import torch
    import torch.nn as nn

    class DecisionModel(nn.Module):
        def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2) -> None:
            super().__init__()
            self.encoder = encoder
            d = encoder.config.hidden_size
            layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.1, batch_first=True, norm_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            h = h + self.type_emb(qtype)[:, None, :]
            if self.head is not None:
                pad = ~attention_mask.bool()
                for layer in self.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx)
            logits = self.scorer(m).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)
            p = torch.softmax(logits.detach(), -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            top2 = p.topk(2, -1).values
            feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
            # The repo's own path only ever runs this under CUDA autocast, which casts for it; cast
            # explicitly so fp16 and bf16 work on CPU and MPS too.
            act_in = torch.cat([h[:, 0].float(), feats], -1).to(self.act_head[0].weight.dtype)
            return logits, self.act_head(act_in)

    _MODEL_CLS = DecisionModel
    return _MODEL_CLS


def __getattr__(name: str):
    """`laya.DecisionModel` builds the class on first touch, so the import still costs no torch.

    `data/gate/train_laya.py` asks this module for `DecisionModel`, `collate`, `QTYPES`,
    `render_options` and `pick_device` by name. Handing it the class defined here means the
    trainer's graph and the gate's graph are the same object rather than two copies that drift.
    Without torch installed this raises, and the trainer falls back to the probe.
    """
    if name == "DecisionModel":
        return _decision_model_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ----------------------------------------------------------------------------- heavy imports
def _heavy_imports(tries: int = 5, wait: float = 0.25):
    """`load_file` and the three `Auto*` classes, retried past a concurrent import of the same
    packages.

    `preload()` runs on a background thread on purpose, and the rest of presage is importing on the
    main one at that moment. Two threads reaching `from transformers import AutoConfig` at once is
    a race Python does not arbitrate: the second one is handed the module object while the first is
    still filling it in, finds no `AutoConfig` on it yet, and raises ImportError. `gate_eval
    --decider laya` hit exactly that on its first row, and in the app the same race would show up
    as `laya preload: cannot import name 'AutoConfig'` and a gate that quietly fell back.

    Waiting and asking again is the whole fix: the other thread finishes in well under a second and
    the names are there. Anything genuinely missing still raises, from the last attempt.
    """
    for attempt in range(tries):
        try:
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            return load_file, AutoConfig, AutoModel, AutoTokenizer
        except ImportError:
            if attempt == tries - 1:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


# ----------------------------------------------------------------------------- the Decider
class LayaDecider:
    """Local typed decisions. Same `decide()` contract as `JevDecider`: no network, no key.

    The forward pass is 80 to 430 ms of solid compute, so `decide()` hands it to a thread and holds
    a lock while it runs: one model, one GPU queue, serial calls.
    """

    provider = "laya"
    kind = "laya"

    def __init__(self, model_dir: str | Path | None = None, device: str | None = None,
                 dtype: str | None = None, preload: bool = False, calib: str | None = None) -> None:
        self.model_dir = Path(model_dir).expanduser() if model_dir else weights_dir()
        self.want_device = device
        self.want_dtype = dtype
        self.want_calib = calib                 # None reads PRESAGE_CALIB; "" forces the shipped one
        self.device = None                      # a torch.device, once loaded
        self.dtype_name: str | None = None      # fp16 on mps, fp32 on cpu, unless asked otherwise
        self.cfg: dict = {}
        self.tok = None
        self.model = None
        self.load_ms: dict[str, float] = {}
        self._load_lock = threading.Lock()      # one loader, whichever thread gets there first
        self._alock: asyncio.Lock | None = None
        self._loader: threading.Thread | None = None
        self.warm_ms = 0
        if preload:
            self.preload()

    def __repr__(self) -> str:
        where = f"{self.device.type}/{self.dtype_name}" if self.model is not None else "not loaded"
        return f"LayaDecider({_short(self.model_dir)}, {where})"

    # ---- load ----
    def preload(self) -> None:
        """Start the 20 s load in the background so the first gate does not pay for it."""
        if self.model is not None or (self._loader is not None and self._loader.is_alive()):
            return
        self._loader = threading.Thread(target=self._load_quiet, name="laya-load", daemon=True)
        self._loader.start()

    def _load_quiet(self) -> None:
        try:
            self.load()
        except Exception as e:  # noqa: BLE001  preloading must never take the process down
            log.warning("laya preload: %s", e)

    def load(self) -> "LayaDecider":
        with self._load_lock:
            if self.model is None:
                self._load()
        return self

    def _resolve(self):
        import torch
        dev = pick_device(self.want_device)
        # fp16 is 2.1x faster than fp32 on MPS with identical answers (the weights are stored fp16
        # anyway); torch CPU fp16 is emulated and 8x slower than CPU fp32, so CPU stays fp32.
        name = self.want_dtype or ("fp32" if dev.type == "cpu" else "fp16")
        if name not in DTYPES:
            raise DecisionError(f"unknown dtype {name!r}, expected one of {', '.join(DTYPES)}")
        return dev, name, {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]

    def _load(self) -> None:
        why = _why_not(self.model_dir)
        if why:
            raise DecisionError(why)
        os.environ.setdefault("USE_TF", "0")          # model card: a TF probe can deadlock construction
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        load_file, AutoConfig, AutoModel, AutoTokenizer = _heavy_imports()

        d, ms = self.model_dir, self.load_ms
        t = time.perf_counter()
        self.cfg = load_config(d, self.want_calib)
        self.tok = AutoTokenizer.from_pretrained(str(d / "tokenizer"))
        ms["tokenizer"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        enc = AutoModel.from_config(AutoConfig.from_pretrained(str(d / "encoder")), attn_implementation="sdpa")
        model = _decision_model_cls()(enc, self.cfg["head_layers"],
                                      len(self.cfg.get("act_costs", {"escalate": 0.5})) + 1)
        ms["build"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        missing, unexpected = model.load_state_dict(load_file(str(d / "model.safetensors")), strict=False)
        if missing or unexpected:
            raise DecisionError(f"state dict mismatch: missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
        ms["weights"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        dev, dtype_name, dtype = self._resolve()
        model.encoder.config.reference_compile = False     # repo note: torch.compile loses on small batches
        model.eval().to(dev, dtype=dtype)
        sync(dev)
        ms["to_device"] = (time.perf_counter() - t) * 1e3
        ms["total"] = sum(v for k, v in ms.items() if k != "total")

        self.device, self.dtype_name, self.model = dev, dtype_name, model
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        log.info("laya: %s loaded in %.1f s on %s/%s (max_len=%d, calib=%s)",
                 self.cfg.get("model_name", "laya"), ms["total"] / 1e3, dev.type, dtype_name,
                 self.cfg["max_len"], self.cfg.get("calib_file", CONFIG_NAME))

    # ---- request shaping ----
    def _internal(self, q: Q) -> dict:
        """One `Q` in Laya's internal question shape.

        `jev.py` sends a choice question as `criteria={label: label}`, which Jev is happy with but
        which Laya renders as "question: question", because its criteria map label -> description.
        Sending `{label: None}` makes the option text the bare label, which is what was measured.
        """
        if q.kind == "choice":
            opts = list(q.options or [])
            if not opts:
                raise DecisionError(f"choice question {q.name!r} has no options")
            crit: Any = {o: None for o in opts}
        elif q.kind == "score":
            crit = q.levels()
            if len(crit) < 2:
                raise DecisionError(f"score question {q.name!r} needs at least 2 levels")
        elif q.kind == "noul":
            crit = None
        else:
            raise DecisionError(f"unknown question kind {q.kind!r}")
        return {"t": q.kind, "ins": q.instructions, "crit": crit}

    # ---- the forward pass ----
    def system_one(self, state: Any, questions: list[Q]) -> dict:
        """Jev's wire response, computed locally. One forward pass for every question."""
        import numpy as np
        import torch

        if self.model is None:
            self.load()
        if not questions:
            raise DecisionError("no questions")
        names = [q.name for q in questions]
        if len(set(names)) != len(names):
            raise DecisionError(f"duplicate question names: {names}")

        internals, items = [], []
        for q in questions:
            qi = self._internal(q)
            seq, markers = build_sequence(self.tok, state, qi, self.cfg["max_len"], self.cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                raise DecisionError(f"question {q.name!r}: options do not fit in "
                                    f"head_max_len={self.cfg['head_max_len']} tokens")
            internals.append(qi)
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[qi["t"]]})

        b = collate(items, self.tok.pad_token_id)
        dev = self.device
        with torch.no_grad():
            logits, act = self.model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                     b["marker_pos"].to(dev), b["marker_mask"].to(dev), b["qtype"].to(dev))
        sync(dev)
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()

        answers: dict[str, dict] = {}
        for r, q in enumerate(questions):
            qi, k = internals[r], len(items[r]["markers"])
            qt = QTYPES[qi["t"]]
            temp = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[r, :k] / temp
            p = np.exp(z - z.max())
            p = p / p.sum()
            ext = {"act_probability": float(act[r, 0]), "temperature": float(temp),
                   "bucket": temp_bucket(qt, k)}
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
        return {"model": self.cfg.get("model_name", "laya"), "answers": answers,
                "usage": {"input_tokens": int(b["attention_mask"].sum()), "output_tokens": 0}}

    # ---- the Decider contract ----
    def decide_sync(self, state: str, questions: list[Q]) -> dict[str, Decision]:
        """Blocking. Loads the model on the first call, which costs about 20 s."""
        t0 = time.perf_counter()
        body = self.system_one(state, questions)
        ms = int((time.perf_counter() - t0) * 1e3)
        return parse_answers(questions, body, ms)

    def _lock(self) -> asyncio.Lock:
        if self._alock is None:
            self._alock = asyncio.Lock()
        return self._alock

    async def decide(self, state: str, questions: list[Q]) -> dict[str, Decision]:
        """Off the event loop: the forward pass is 80 to 430 ms of compute and would otherwise
        stall the audio and websocket tasks for its whole duration."""
        async with self._lock():
            return await asyncio.to_thread(self.decide_sync, state, questions)

    async def validate_key(self) -> tuple[bool, str]:
        """No key to check, so this checks what can actually be wrong: that the weights are here,
        that they load, and how long a warm gate takes on this machine."""
        why = _why_not(self.model_dir)
        if why:
            return False, why[:140]
        async with self._lock():
            try:
                self.warm_ms = await asyncio.to_thread(self._warm_gate)
            except DecisionError as e:
                return False, str(e)[:140]
            except Exception as e:  # noqa: BLE001
                return False, f"{type(e).__name__}: {e}"[:140]
        return True, f"ok ({self.warm_ms} ms)"

    def _warm_gate(self) -> int:
        """One gate call to warm the graph, a second one for the number that gets reported."""
        state = SAMPLES[0][1]
        self.decide_sync(state, GATE_QUESTIONS)
        return self.decide_sync(state, GATE_QUESTIONS)["kind"].latency_ms

    async def warm(self) -> float | None:
        """Load and warm up now. Returns the warm gate in ms, or None if it cannot run."""
        ok, msg = await self.validate_key()
        if not ok:
            log.info("laya warm failed: %s", msg)
            return None
        return float(self.warm_ms)

    async def aclose(self) -> None:
        dev = self.device
        self.model, self.tok = None, None
        if sys.modules.get("torch") is not None and dev is not None:
            empty_cache(dev)


# ----------------------------------------------------------------------------- presage's real gate
# Copied from `Brain._gate_decider` so `--selftest` and `--bench` measure the real thing.

GATE_QUESTIONS: list[Q] = [
    Q("kind", "choice", "What is the last line? question = asks for information; what-if = asks what a "
      "number would become under a different assumption (\"what if we charged 1%\", \"at 400 subscribers "
      "instead\"); request = asks Me to show, walk through or send something; objection = pushes back on "
      "a number, price, claim or plan, raises a risk or competitor, or hesitates; none = small talk, "
      "statements, agreement, filler.",
      options=["question", "what-if", "request", "objection", "none"]),
    Q("needs_lookup", "noul", "Answering the last line well needs a looked-up number, date, name or fact "
      "(from the consultant's decks, models, documents or public sources), or, in Sales/Pitch/Interview "
      "mode, a suggested response. False for small talk, pleasantries and things answered from the head."),
    Q("complete", "noul", "The last line is a complete thought (not cut off mid-sentence)."),
]


def gate_state(lines: list[tuple[str, str]], mode: str = "Consulting") -> str:
    body = "\n".join(f"{'Me' if src == 'mic' else 'Them'}: {text}" for src, text in lines)
    return ("A strategy consultant (Me) is on a call with a client (Them). Recent transcript, oldest first; "
            f"judge ONLY the last line. Mode: {mode}.\n\n{body}")


# Two states that should land on opposite sides of the gate: an information question that needs the
# files, and pure filler that needs nothing. Expected answers in the comments, not asserted: the
# stock weights get them wrong (docs/research-laya.md section 5), and that is the point of printing.
SAMPLES: list[tuple[str, str]] = [
    ("question / lookup / complete", gate_state([
        ("mic", "So that covers the cost side of the model."),
        ("them", "Mm, got it."),
        ("mic", "Anything you want to dig into before I move on?"),
        ("them", "So on the revenue side, what did Q3 actually come in at versus plan?")])),
    ("none / no lookup / complete", gate_state([
        ("mic", "Morning, thanks for making the time."),
        ("them", "Of course."),
        ("mic", "Did you have any trouble getting in?"),
        ("them", "Ha, no, the traffic on the bridge was brutal this morning though.")])),
]


# ----------------------------------------------------------------------------- getting the weights
def copy_from(src: str | Path, target: str | Path | None = None) -> Path:
    """Copy a checkpoint that is already on this machine into the weights dir."""
    src, target = Path(src).expanduser(), Path(target).expanduser() if target else weights_dir()
    if not weights_present(src):
        raise SystemExit(f"{src} is not a Laya checkpoint (want {', '.join(FILES)})")
    for name in FILES:
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name, dst)
        print(f"  {name:34s} {dst.stat().st_size / 1e6:8.1f} MB")
    print(f"copied {_short(target)}")
    return target


def download(target: str | Path | None = None, force: bool = False) -> Path:
    """Pull just the typed-decisions checkpoint (846 MB) from the Hub into the weights dir."""
    target = Path(target).expanduser() if target else weights_dir()
    if weights_present(target) and not force:
        print(f"already here: {_short(target)}")
        return target
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub is not installed: uv sync --extra laya") from None
    stage = target.parent / f".{target.name}.download"
    print(f"{REPO_ID} / {CHECKPOINT} -> {_short(target)}")
    snapshot_download(repo_id=REPO_ID, allow_patterns=[f"{CHECKPOINT}/*", f"{CHECKPOINT}/**"],
                      local_dir=str(stage))
    got = stage / CHECKPOINT
    for name in FILES:
        if not (got / name).is_file():
            raise SystemExit(f"the Hub did not give us {CHECKPOINT}/{name}")
    for name in FILES:
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got / name), str(dst))
        print(f"  {name:34s} {dst.stat().st_size / 1e6:8.1f} MB")
    shutil.rmtree(stage, ignore_errors=True)
    print(f"downloaded {_short(target)}")
    return target


# ----------------------------------------------------------------------------- CLI
def _fmt(d: dict[str, Decision]) -> str:
    bits = []
    for name, dec in d.items():
        v = f"{dec.value:.3f}" if isinstance(dec.value, float) else str(dec.value)
        bits.append(f"{name}={v} p={dec.p:.3f} conf={dec.confidence:.3f}")
    return "  ".join(bits)


def selftest(dec: LayaDecider) -> None:
    """The real three-question gate on two states, with the full distributions."""
    for label, state in SAMPLES:
        out = dec.decide_sync(state, GATE_QUESTIONS)
        body = dec.system_one(state, GATE_QUESTIONS)
        print(f"\n{state.splitlines()[-1]}")
        print(f"  expected: {label}")
        for q in GATE_QUESTIONS:
            a = body["answers"][q.name]
            probs = a.get("probabilities") or {"true": a.get("noul")}
            print(f"  {q.name:13s} {str(out[q.name].value):10s} "
                  f"p={out[q.name].p:.3f} conf={out[q.name].confidence:.3f}  "
                  f"{json.dumps({k: round(float(v), 3) for k, v in probs.items()})}")
        print(f"  {body['usage']['input_tokens']} tokens, {out['kind'].latency_ms} ms")


def bench(dec: LayaDecider, runs: int, warmup: int) -> None:
    import statistics
    state = SAMPLES[0][1]
    for _ in range(warmup):
        dec.system_one(state, GATE_QUESTIONS)
    took = []
    for _ in range(runs):
        t = time.perf_counter()
        dec.system_one(state, GATE_QUESTIONS)
        took.append((time.perf_counter() - t) * 1e3)
    p95 = statistics.quantiles(took, n=100)[94] if len(took) > 2 else max(took)
    print(f"{dec.device.type}/{dec.dtype_name}  load {dec.load_ms.get('total', 0) / 1e3:.1f} s  "
          f"{len(GATE_QUESTIONS)} questions x {runs}: p50 {statistics.median(took):.1f} ms  "
          f"p95 {p95:.1f} ms  min {min(took):.1f} ms")


def status() -> int:
    d = weights_dir()
    why = _why_not(d)
    print(f"weights   {_short(d)}{'' if weights_present(d) else '  (not downloaded)'}")
    gone = missing_packages()
    print(f"packages  {'torch, transformers, safetensors' if not gone else 'missing: ' + ', '.join(gone)}")
    cp = config_path(d)
    print(f"calib     {cp.name}{'' if calib_tag() else '  (shipped; PRESAGE_CALIB unset)'}")
    if weights_present(d):
        cfg = load_config(d)
        print(f"          {json.dumps(cfg.get('temperature_by_options', {}))}")
    print(f"gate      {'ready' if not why else why}")
    return 0 if not why else 1


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m presage.llm.laya",
                                 description="presage's local typed-decision gate")
    ap.add_argument("--download", action="store_true", help=f"fetch {CHECKPOINT} (846 MB) into the weights dir")
    ap.add_argument("--from-dir", help="copy a checkpoint already on this machine instead of downloading")
    ap.add_argument("--force", action="store_true", help="download even when the weights are already here")
    ap.add_argument("--selftest", action="store_true", help="the three-question gate on two sample states")
    ap.add_argument("--bench", action="store_true", help="p50/p95 for the gate")
    ap.add_argument("--model-dir", help=f"where the weights are (default {_short(DEFAULT_DIR)})")
    ap.add_argument("--device", help="mps | cuda | cpu (default: mps, then cuda, then cpu)")
    ap.add_argument("--dtype", choices=DTYPES, help="default: fp16 on mps, fp32 on cpu")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    if a.from_dir:
        copy_from(a.from_dir, a.model_dir)
    elif a.download:
        download(a.model_dir, force=a.force)
    elif not (a.selftest or a.bench):
        raise SystemExit(status())
    if not (a.selftest or a.bench):
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dec = LayaDecider(a.model_dir, device=a.device, dtype=a.dtype).load()
    if a.selftest:
        selftest(dec)
    if a.bench:
        bench(dec, a.runs, a.warmup)


if __name__ == "__main__":
    main()
