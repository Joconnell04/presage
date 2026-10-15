"""ModernBERT + Laya's decision head, written so it can leave PyTorch.

`presage/llm/laya.py` builds the encoder with `AutoModel.from_config`, which is the right thing for
a model you only ever run in torch: it is the reference implementation and it cannot drift. It is
the wrong thing for an export. `transformers` builds its attention masks inside the forward pass
from `torch.arange`, `new_ones` and Python `int()` on tensors, and every one of those is a node a
tracer either refuses or bakes in wrongly. Converting it to Core ML fails on the third op.

This module is the same 28-layer encoder with every data-dependent step lifted out:

  masks          an additive padding bias `(B, 1, 1, L)` and a sliding-window band `(1, 1, L, L)`
                 come in as inputs; the host builds the band once per length bucket
  rope           cos/sin for both thetas come in the same way, not computed from the sequence
  markers        a one-hot `marker_map (B, K, L)` matmul instead of `torch.gather`, which is what
                 the published Core ML exports of this model use and what the Neural Engine wants
  type embedding looked up on the host and passed in as `type_vec (B, D)`

What is left is dense linear algebra and one softmax per layer. It traces, it exports, and it
agrees with `AutoModel` to 4.8e-07 on the gate's own logits in fp32 (`bench.py --check`).

Weights load straight out of the same `model.safetensors` the torch path uses, so there is one set
of weights on disk and no conversion step to keep in sync.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from presage.llm.laya_fast import runtime as R

NEG = -1e4          # fp16-safe stand-in for -inf; ModernBERT's own sdpa path uses finfo.min in fp32


# ----------------------------------------------------------------------------- config
class EncCfg:
    def __init__(self, d: dict) -> None:
        self.hidden = int(d["hidden_size"])
        self.layers = int(d["num_hidden_layers"])
        self.heads = int(d["num_attention_heads"])
        self.inter = int(d["intermediate_size"])
        self.vocab = int(d["vocab_size"])
        self.eps = float(d.get("norm_eps", d.get("layer_norm_eps", 1e-5)))
        self.norm_bias = bool(d.get("norm_bias", False))
        self.local = int(d.get("local_attention", 128))
        self.raw = d
        self.layer_types = R.layer_kinds(d)
        self.theta = R.rope_thetas(d)
        self.head_dim = self.hidden // self.heads

    @property
    def kinds(self) -> list[str]:
        return sorted(set(self.layer_types))


def rotate_half(x: torch.Tensor, h: int) -> torch.Tensor:
    return torch.cat([-x[..., h:], x[..., :h]], -1)


# ----------------------------------------------------------------------------- encoder
class Layer(nn.Module):
    def __init__(self, c: EncCfg, idx: int) -> None:
        super().__init__()
        self.c, self.idx = c, idx
        self.attn_norm = nn.Identity() if idx == 0 else nn.LayerNorm(c.hidden, eps=c.eps, bias=c.norm_bias)
        self.Wqkv = nn.Linear(c.hidden, 3 * c.hidden, bias=False)
        self.Wo = nn.Linear(c.hidden, c.hidden, bias=False)
        self.mlp_norm = nn.LayerNorm(c.hidden, eps=c.eps, bias=c.norm_bias)
        self.Wi = nn.Linear(c.hidden, 2 * c.inter, bias=False)
        self.Wmlp = nn.Linear(c.inter, c.hidden, bias=False)

    def forward(self, h, bias, cos, sin, B: int):
        """B is a Python int from the export and the sequence dimension is `-1`. Neither is read
        off a tensor: `h.shape[1]` traces to an `aten::Int` node that Core ML has no translation
        for, and baking the length in as a constant would cost one compiled model per length."""
        c, half = self.c, self.c.head_dim // 2
        x = self.attn_norm(h)
        qkv = self.Wqkv(x).view(B, -1, 3, c.heads, c.head_dim)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        dt = q.dtype
        q = (q.float() * cos + rotate_half(q.float(), half) * sin).to(dt)
        k = (k.float() * cos + rotate_half(k.float(), half) * sin).to(dt)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        h = h + self.Wo(a.transpose(1, 2).reshape(B, -1, c.hidden))
        wi = self.Wi(self.mlp_norm(h))
        return h + self.Wmlp(F.gelu(wi[..., :c.inter]) * wi[..., c.inter:])


class Encoder(nn.Module):
    """ModernBERT with the masks and the rope tables taken as inputs rather than built inside.

    They could be constants -- they only depend on the sequence length -- and the first export did
    exactly that. But then each length is a separate compiled model, and each compiled model is
    another 843 MB copy of the weights on disk. Passing them in lets one Core ML package carry
    every length as an enumerated shape with one set of weights. The host builds them once per
    bucket and reuses the arrays (`tables`), so they cost nothing per call.
    """

    def __init__(self, c: EncCfg) -> None:
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.hidden)
        self.emb_norm = nn.LayerNorm(c.hidden, eps=c.eps, bias=c.norm_bias)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.layers))
        self.final_norm = nn.LayerNorm(c.hidden, eps=c.eps, bias=c.norm_bias)

    def forward(self, input_ids, pad_bias, band, rope: dict, B: int):
        h = self.emb_norm(self.tok(input_ids))
        sliding = pad_bias + band
        for i, layer in enumerate(self.layers):
            kind = self.c.layer_types[i]
            bias = pad_bias if kind == "full_attention" else sliding
            cos, sin = rope[kind]
            h = layer(h, bias.to(h.dtype), cos, sin, B)
        return self.final_norm(h)


def tables(c: EncCfg, L: int, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """The band mask and the rope tables for one bucket, as torch tensors. One builder, in
    `runtime.py`, in numpy, so the exported backends never reach for torch to make them."""
    np_dtype = {torch.float32: "float32", torch.float16: "float16"}.get(dtype, "float32")
    return {k: torch.from_numpy(v.copy()) for k, v in R.tables_np(c.raw, L, np_dtype).items()}


# ----------------------------------------------------------------------------- decision head
class HeadLayer(nn.Module):
    """`nn.TransformerEncoderLayer(d, d//64, 4d, norm_first=True)`, unrolled.

    Same maths, same parameter names, no `src_key_padding_mask` bool path: the padding arrives as
    the additive bias the rest of the graph already carries.
    """

    def __init__(self, d: int, heads: int, ff: int) -> None:
        super().__init__()
        self.d, self.heads, self.hd = d, heads, d // heads
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.linear1 = nn.Linear(d, ff)
        self.linear2 = nn.Linear(ff, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, pad_bias, B: int):
        y = self.norm1(x)
        qkv = F.linear(y, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        qkv = qkv.view(B, -1, 3, self.heads, self.hd)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_bias.to(x.dtype))
        x = x + self.self_attn.out_proj(a.transpose(1, 2).reshape(B, -1, self.d))
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class Gate(nn.Module):
    """Encoder, head, scorer: everything between the token ids and the per-option logits.

    `forward` takes the band mask and the rope tables positionally, in `table_names` order, so the
    exported signature is fixed and flat: tracers and Core ML both want that rather than a dict.
    """

    def __init__(self, c: EncCfg, head_layers: int = 2, batch: int = 3) -> None:
        super().__init__()
        self.c, self.batch = c, batch
        self.enc = Encoder(c)
        self.head = nn.ModuleList(HeadLayer(c.hidden, max(1, c.hidden // 64), 4 * c.hidden)
                                  for _ in range(head_layers))
        self.scorer = nn.Sequential(nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.hidden),
                                    nn.GELU(), nn.Linear(c.hidden, 1))

    @property
    def table_names(self) -> list[str]:
        return ["band", *(f"{p}_{k}" for k in self.c.kinds for p in ("cos", "sin"))]

    def forward(self, input_ids, pad_bias, type_vec, marker_map, *tabs):
        B = self.batch
        t = dict(zip(self.table_names, tabs))
        rope = {k: (t[f"cos_{k}"], t[f"sin_{k}"]) for k in self.c.kinds}
        h = self.enc(input_ids, pad_bias, t["band"], rope, B) + type_vec[:, None, :]
        for layer in self.head:
            h = layer(h, pad_bias, B)
        return self.scorer(torch.bmm(marker_map.to(h.dtype), h)).squeeze(-1).float()


# ----------------------------------------------------------------------------- weights
def _remap(sd: dict[str, torch.Tensor], c: EncCfg, head_layers: int) -> dict[str, torch.Tensor]:
    """Laya's published state dict -> this module's names. Every tensor is used or named."""
    out: dict[str, torch.Tensor] = {
        "enc.tok.weight": sd["encoder.embeddings.tok_embeddings.weight"],
        "enc.emb_norm.weight": sd["encoder.embeddings.norm.weight"],
        "enc.final_norm.weight": sd["encoder.final_norm.weight"],
    }
    for i in range(c.layers):
        p, q = f"encoder.layers.{i}.", f"enc.layers.{i}."
        if i:
            out[q + "attn_norm.weight"] = sd[p + "attn_norm.weight"]
        out[q + "Wqkv.weight"] = sd[p + "attn.Wqkv.weight"]
        out[q + "Wo.weight"] = sd[p + "attn.Wo.weight"]
        out[q + "mlp_norm.weight"] = sd[p + "mlp_norm.weight"]
        out[q + "Wi.weight"] = sd[p + "mlp.Wi.weight"]
        out[q + "Wmlp.weight"] = sd[p + "mlp.Wo.weight"]
    for i in range(head_layers):
        p, q = f"head.layers.{i}.", f"head.{i}."
        for name in ("self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight",
                     "self_attn.out_proj.bias", "linear1.weight", "linear1.bias", "linear2.weight",
                     "linear2.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            out[q + name] = sd[p + name]
    for i in (0, 1, 3):
        out[f"scorer.{i}.weight"] = sd[f"scorer.{i}.weight"]
        out[f"scorer.{i}.bias"] = sd[f"scorer.{i}.bias"]
    return out


def load_gate(model_dir: str | Path, dtype=torch.float32, batch: int = 3) -> tuple[Gate, dict[str, Any], torch.Tensor]:
    """The Gate, plus `rl_agent_config.json` and the type-embedding table. Length-agnostic."""
    from safetensors.torch import load_file

    d = Path(model_dir).expanduser()
    cfg = json.loads((d / "rl_agent_config.json").read_text())
    c = EncCfg(json.loads((d / "encoder" / "config.json").read_text()))
    sd = load_file(str(d / "model.safetensors"))
    g = Gate(c, int(cfg.get("head_layers", 2)), batch)
    missing, unexpected = g.load_state_dict(_remap(sd, c, int(cfg.get("head_layers", 2))), strict=True)
    assert not missing and not unexpected
    return g.eval().to(dtype), cfg, sd["type_emb.weight"].float()


# ----------------------------------------------------------------------------- host-side inputs
def pad_bias_of(attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, L) 1/0 -> (B, 1, 1, L) additive bias."""
    return torch.where(attention_mask.bool(), 0.0, NEG)[:, None, None, :].float()


def marker_map_of(markers: list[list[int]], K: int, L: int) -> torch.Tensor:
    """One-hot selector for the option markers. Rows past a question's option count stay zero;
    the caller slices `logits[r, :k]` exactly as `laya.py` does, so they are never read."""
    m = torch.zeros(len(markers), K, L)
    for r, pos in enumerate(markers):
        for i, p in enumerate(pos[:K]):
            if p < L:
                m[r, i, p] = 1.0
    return m


def seq_len_for(n: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} tokens is past the largest bucket {buckets[-1]}")
