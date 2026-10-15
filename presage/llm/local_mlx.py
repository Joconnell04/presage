"""The answer path with no network: a quantised model on this Mac's GPU through MLX.

    apply_chat_template(system + user, tools=...)   ->  token ids
    stream_generate(..., prompt_cache=<the system prefix, kept warm>)
    -> stream text, pull the `answer` field out of the partial JSON as it grows
    -> a <tool_call> block runs the brain's tool and the loop goes round again

Why it is shaped like this. The prefix is 15-20k tokens (`system_text()` plus the pre-fetched
evidence) and prefilling it is the whole latency budget: on an M-series GPU a 4B model reads a few
thousand tokens a second, so a cold call cannot be fast and a warm one can. So the provider keeps
ONE KV cache alive, matches each new prompt against it token by token, trims to the common prefix
and prefills only the tail. Inside a tool loop round 2's prompt is round 1's prompt plus the tool
result, so the reuse is total and only the tool result is read.

The answer JSON is not constrained-decoded. mlx-lm 0.31 ships no grammar sampler, and a logits
processor that walks a JSON schema costs more per token than it saves; instead the object is parsed
strictly, and a failure gets ONE repair pass, re-asked with the assistant turn prefilled with
`{"answer": "` so the model cannot open with prose. The repair reads almost nothing new because the
whole conversation is already in the KV cache.

One model is resident at a time. `unload()` drops it and returns the Metal buffers, because the STT
(Parakeet, MLX) and the gate (Core ML) share this GPU and 16 GB.

No key, no network once the weights are in ~/.cache/huggingface.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from typing import Any

from presage.llm.base import (LLM, PARTIAL_EVERY_S, OnPartial, RunTool, ToolSpec,
                              fallback_answer, parse_json, partial_field)

log = logging.getLogger("presage.llm.local")

# The default is E15's winner; override with PRESAGE_MLX_MODEL=<hf repo or local path>.
DEFAULT_MODEL = os.environ.get("PRESAGE_MLX_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit")

ANSWER_MAX_TOKENS = 320       # the object is ~90 tokens; the headroom is for a long `working`
TOOL_MAX_TOKENS = 384         # a whatif call with several sets
CLASSIFY_MAX_TOKENS = 192
TOOL_ROUNDS = 3               # the brain's own budget is 3 calls; a 4B model does not need more
MIN_REUSE_TOKENS = 128        # below this a prefix match is not worth a trim
MAX_CACHE_TOKENS = 32_768     # past this the KV cache is dropped rather than grown

# Generation knobs. Greedy: the answer is a lookup, not prose, and temperature only invents numbers.
TEMP = float(os.environ.get("PRESAGE_MLX_TEMP", "0.0"))
KV_BITS = int(os.environ["PRESAGE_MLX_KV_BITS"]) if os.environ.get("PRESAGE_MLX_KV_BITS") else None
KV_START = int(os.environ.get("PRESAGE_MLX_KV_START", "2048"))
PREFILL_STEP = int(os.environ.get("PRESAGE_MLX_PREFILL_STEP", "2048"))

def _tool_fallback(tools: list[ToolSpec]) -> str:
    """The tool surface for a model whose chat template ignores `tools` (Phi-4-mini drops the
    argument silently, so without this the model never learns the tools exist)."""
    if not tools:
        return ""
    lines = [f"- {t.name}({', '.join((t.input_schema.get('properties') or {}))}): "
             f"{t.description.split('.')[0]}." for t in tools]
    return ("\n\nTools you may call, one per turn:\n" + "\n".join(lines) +
            "\n\nTo call one emit exactly:\n"
            '<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>\n'
            "and nothing else in that turn. Otherwise reply with the answer JSON object only.")

_JSON_NUDGE = (
    "\n\nReply with ONE JSON object and nothing else: no prose before it, no ``` fence, no trailing "
    "text. Keys, in this order: \"answer\" (<=25 words), \"say\" (4-8 spoken words), \"figure\" "
    "(the number with its unit, or null), \"citations\" (a list of {\"file\", \"loc\"}), "
    "\"answered_from\" (\"files\", \"derived\" or \"none\"). Add \"working\" only when you computed "
    "something. Start your reply with {"
)

_ANSWER_PREFILL = '{"answer": "'


# ----------------------------------------------------------------------------- the resident model

class _Resident:
    """The one model in this process. Loading a second one unloads the first."""

    def __init__(self) -> None:
        self.repo: str | None = None
        self.model: Any = None
        self.tok: Any = None
        self.cache: _PrefixCache | None = None
        self.load_s: float = 0.0
        self.lock = threading.Lock()      # MLX generation is not re-entrant

    def get(self, repo: str) -> "_Resident":
        if self.repo == repo:
            return self
        self.unload()
        from mlx_lm import load           # imported here: `import presage.llm` must stay cheap
        t0 = time.time()
        log.info("local: loading %s", repo)
        self.model, self.tok = load(repo)
        self.repo, self.load_s = repo, time.time() - t0
        self.cache = _PrefixCache(self.model)
        log.info("local: %s loaded in %.1fs, %s", repo, self.load_s, mem_mb())
        return self

    def unload(self) -> None:
        if self.repo is None:
            return
        log.info("local: unloading %s", self.repo)
        self.repo, self.model, self.tok, self.cache = None, None, None, None
        import gc

        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
            mx.reset_peak_memory()
        except Exception as e:  # noqa: BLE001
            log.debug("clear_cache failed: %s", e)


_RESIDENT = _Resident()


def unload() -> None:
    """Drop the resident model and give the Metal buffers back. The bench calls this between
    candidates; the app calls it when the private mode is switched off."""
    _RESIDENT.unload()


def mem_mb() -> dict:
    """Active and peak GPU-visible bytes, and this process's RSS, all in MB."""
    out: dict[str, int] = {}
    try:
        import mlx.core as mx

        out["mlx_active_mb"] = int(mx.get_active_memory() / 1e6)
        out["mlx_peak_mb"] = int(mx.get_peak_memory() / 1e6)
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource

        # macOS reports ru_maxrss in bytes
        out["rss_peak_mb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    except Exception:  # noqa: BLE001
        pass
    return out


class _PrefixCache:
    """The system prefix, prefilled once and reused by every call after it.

    This is the local stand-in for a hosted prompt cache and it is the whole experiment:
    `system_text()` is ~18k of the ~20k tokens in every prompt, and re-reading it costs 15 to 20
    seconds on this GPU.

    Two ways to reuse it, because the models differ.

    - **Trim.** A model whose cache is only `KVCache` can be rewound: keep one live cache, walk the
      new prompt against what it holds, drop the tail, prefill the rest. No copying.
    - **Copy.** Qwen3.5 is a hybrid: its linear layers hold a recurrent `ArraysCache` state, which
      is a running summary and cannot be rewound, so `can_trim_prompt_cache` is False for the whole
      cache. Without the base snapshot this class silently fell back to a fresh cache and prefilled
      all 24,415 tokens on every question, at 1,200 tok/s: first text 19.5 s (measured, first
      qwen35-4b run). The fix is to hold a pristine cache of the system prefix and deep-copy it per
      call, which is what mlx-lm's own `LRUPromptCache` does.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.cache: list | None = None
        self.tokens: list[int] = []
        self.base: list | None = None       # pristine, holds exactly `base_tokens`
        self.base_tokens: list[int] = []

    def reset(self) -> None:
        self.cache, self.tokens = None, []

    def _fresh(self) -> list:
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)

    def set_base(self, tokens: list[int]) -> int:
        """Prefill `tokens` into a cache kept pristine, and return how many were read.

        Called from `prewarm()` with the prompt's constant head (system message plus the user
        turn's opening tags). Costs one full prefill, once per system prompt.
        """
        import mlx.core as mx

        if not tokens or self.base_tokens == list(tokens):
            return len(self.base_tokens)
        cache = self._fresh()
        for i in range(0, len(tokens), PREFILL_STEP):
            mx.eval(self.model(mx.array(tokens[i:i + PREFILL_STEP])[None], cache=cache))
        self.base, self.base_tokens = cache, list(tokens)
        log.info("local: system prefix cached, %d tokens, %s", len(tokens), mem_mb())
        return len(tokens)

    def prepare(self, tokens: list[int]) -> tuple[list, list[int], int]:
        """(cache to generate with, the tokens still to prefill, how many were reused)."""
        import copy

        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        # Trim first, because it copies nothing. Only a cache that cannot be rewound falls
        # through to the snapshot: a pure-KVCache model deep-copying a 21k-token cache on every
        # question doubles its resident KV for no gain, and on a 16 GB machine that is the
        # difference between answering and swapping.
        if (self.cache is not None and len(self.tokens) <= MAX_CACHE_TOKENS
                and can_trim_prompt_cache(self.cache)):
            n = 0
            limit = min(len(self.tokens), len(tokens) - 1)   # always leave one token to feed in
            while n < limit and self.tokens[n] == tokens[n]:
                n += 1
            if n >= MIN_REUSE_TOKENS:
                drop = len(self.tokens) - n
                if drop:
                    trim_prompt_cache(self.cache, drop)
                self.tokens = list(tokens[:n])
                return self.cache, list(tokens[n:]), n

        n_base = len(self.base_tokens)
        if (self.base is not None and n_base and len(tokens) > n_base
                and tokens[:n_base] == self.base_tokens):
            self.cache = copy.deepcopy(self.base)
            self.tokens = list(self.base_tokens)
            return self.cache, list(tokens[n_base:]), n_base

        self.cache, self.tokens = self._fresh(), []
        return self.cache, list(tokens), 0

    def advance(self, tokens: list[int]) -> None:
        self.tokens.extend(tokens)


# ----------------------------------------------------------------------------- tool-call parsing

_FALLBACK_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _tool_calls(text: str, tok: Any, names: set[str]) -> list[dict]:
    """Every tool call in one assistant turn, as [{"name", "arguments"}]."""
    out: list[dict] = []
    start = getattr(tok, "tool_call_start", None) or "<tool_call>"
    end = getattr(tok, "tool_call_end", None) or "</tool_call>"
    parser = getattr(tok, "tool_parser", None)
    i = 0
    while True:
        a = text.find(start, i)
        if a < 0:
            break
        b = text.find(end, a + len(start))
        body = text[a + len(start): b if b >= 0 else len(text)]
        i = (b + len(end)) if b >= 0 else len(text)
        call = None
        if parser is not None:
            try:
                call = parser(body, None)
            except Exception:  # noqa: BLE001
                call = None
        if call is None:
            try:
                call = json.loads(body.strip())
            except json.JSONDecodeError:
                call = None
        if isinstance(call, dict):
            out.append(call)
    if not out:            # a model that ignored its own grammar and wrote the block as prose
        for m in _FALLBACK_CALL.finditer(text):
            try:
                out.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass
    fixed = []
    for c in out:
        fn = c.get("function") if isinstance(c.get("function"), dict) else c
        name = str(fn.get("name") or "")
        args = fn.get("arguments") if "arguments" in fn else fn.get("parameters")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name in names:
            fixed.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return fixed


def _tool_decls(tools: list[ToolSpec]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
            for t in tools]


# ----------------------------------------------------------------------------- the provider

class LocalMLXLLM(LLM):
    """A local MLX model behind the same interface as the hosted providers.

    `answer_model`, `hedge_model` and `classifier_model` are Hugging Face repo ids (or local
    paths). Only one is resident at a time, so setting two different ones costs a reload per
    switch; the sane configuration is one repo for all three, and the hedge off.
    """

    provider = "local"

    @classmethod
    def default_models(cls) -> dict:
        return {"answer": DEFAULT_MODEL, "hedge": "", "classifier": DEFAULT_MODEL}

    def __init__(self, api_key: str | None = None, models: dict | None = None) -> None:
        super().__init__(api_key, models)
        self._sem = asyncio.Semaphore(1)       # one generation at a time on one GPU

    # ------------------------------------------------------------------ generation

    def _render(self, tok: Any, messages: list[dict], tools: list[ToolSpec] | None,
                prefill: str = "") -> list[int]:
        kw: dict[str, Any] = dict(add_generation_prompt=True, tokenize=True, enable_thinking=False)
        if tools:
            kw["tools"] = _tool_decls(tools)
        attempts = [messages]
        if messages and messages[0]["role"] == "system":
            # Gemma-shaped templates raise on a system role; fold it into the first user turn
            merged = [dict(m) for m in messages[1:]]
            head = messages[0]["content"]
            if merged and merged[0]["role"] == "user":
                merged[0]["content"] = f"{head}\n\n{merged[0]['content']}"
            else:
                merged.insert(0, {"role": "user", "content": head})
            attempts.append(merged)
        ids = None
        for msgs in attempts:
            for drop in ((), ("enable_thinking",), ("enable_thinking", "tools")):
                k = {x: v for x, v in kw.items() if x not in drop}
                if "tools" in k and "enable_thinking" in drop:
                    k["tools"] = [d["function"] for d in _tool_decls(tools or [])]
                try:
                    ids = tok.apply_chat_template(msgs, **k)
                    break
                except Exception as e:  # noqa: BLE001
                    log.debug("chat template refused %s: %s", sorted(k), e)
            if ids is not None:
                break
        if ids is None:
            raise RuntimeError("this model's chat template accepted none of the message shapes")
        ids = list(ids)
        if prefill:
            ids += list(tok.encode(prefill, add_special_tokens=False))
        return ids

    def _constant_head(self, tok: Any, system: str, tools: list[ToolSpec] | None) -> list[int]:
        """The token prefix every prompt on this system text shares.

        Found by rendering the same conversation with two different user turns and taking the
        common prefix, which is exact whatever the template puts between the roles, and needs no
        knowledge of the template itself.
        """
        a = self._render(tok, [{"role": "system", "content": system},
                               {"role": "user", "content": "alpha"}], tools)
        b = self._render(tok, [{"role": "system", "content": system},
                               {"role": "user", "content": "bravo zulu"}], tools)
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return a[:n]

    def _generate(self, res: _Resident, ids: list[int], max_tokens: int, rec: dict,
                  on_chunk: Any, prefill: str = "") -> tuple[str, str]:
        """Blocking. Runs in a worker thread; `on_chunk(text_so_far)` is called from there."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        cache, todo, reused = res.cache.prepare(ids)
        rec["cache_read"] = rec.get("cache_read", 0) + reused
        rec["prefill"] = rec.get("prefill", 0) + len(todo)
        rec["in"] = rec.get("in", 0) + len(ids)
        kw: dict[str, Any] = dict(prompt_cache=cache, sampler=make_sampler(temp=TEMP),
                                  prefill_step_size=PREFILL_STEP)
        if KV_BITS:
            kw["kv_bits"], kw["quantized_kv_start"] = KV_BITS, KV_START
        buf, out_ids, finish = prefill, [], None
        t0 = time.time()
        for r in stream_generate(res.model, res.tok, todo, max_tokens=max_tokens, **kw):
            if not rec.get("first_text_s"):
                rec["first_text_s"] = round(time.time() - t0, 3)
                rec["prompt_tps"] = round(getattr(r, "prompt_tps", 0.0) or 0.0, 1)
            buf += r.text
            out_ids.append(int(r.token))
            on_chunk(buf)
            finish = r.finish_reason
            rec["gen_tps"] = round(getattr(r, "generation_tps", 0.0) or 0.0, 1)
        rec["out"] = rec.get("out", 0) + len(out_ids)
        res.cache.advance(todo + out_ids)
        return buf, str(finish or "stop")

    async def _run(self, repo: str, messages: list[dict], tools: list[ToolSpec] | None,
                   max_tokens: int, rec: dict, on_partial: OnPartial | None,
                   prefill: str = "") -> tuple[str, str]:
        """One streamed turn, off the event loop, with partials pushed back onto it."""
        res = await asyncio.to_thread(_RESIDENT.get, repo)
        ids = self._render(res.tok, messages, tools, prefill)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        last = [0.0]

        def on_chunk(buf: str) -> None:
            if on_partial is None:
                return
            now = time.time()
            if now - last[0] < PARTIAL_EVERY_S:
                return
            last[0] = now
            loop.call_soon_threadsafe(q.put_nowait, buf)

        async def pump() -> None:
            sent = None
            while True:
                buf = await q.get()
                if buf is None:
                    return
                text = partial_field(buf)
                if text and text != sent and on_partial is not None:
                    sent = text
                    if not rec.get("first_answer_s"):
                        rec["first_answer_s"] = round(time.time() - rec["_t0"], 3)
                    try:
                        await on_partial(text, repo)
                    except Exception as e:  # noqa: BLE001
                        log.debug("on_partial failed: %s", e)

        task = asyncio.create_task(pump()) if on_partial is not None else None
        try:
            def work() -> tuple[str, str]:
                with res.lock:
                    return self._generate(res, ids, max_tokens, rec, on_chunk, prefill)

            text, finish = await asyncio.to_thread(work)
        finally:
            if task is not None:
                loop.call_soon_threadsafe(q.put_nowait, None)
                try:
                    await asyncio.wait_for(task, 2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
        return text, finish

    # ------------------------------------------------------------------ answer

    async def answer(self, model: str, system: str, user: str, tools: list[ToolSpec], schema: dict,
                     run_tool: RunTool, on_partial: OnPartial | None = None,
                     web: bool = False, lane: str = "bg") -> tuple[dict, dict]:
        """The whole tool loop for one answer.

        `web` is ignored: there is no network in this mode, which is the point of it; the brain's
        "not in files" path covers what the web would have.

        `lane` (E52) is noted and not acted on. There is no parallelism here to prioritise:
        `self._sem` is 1 because two generations on one Metal queue serialize anyway, so a live
        question already waits for at most one draft rather than for a pool. Preempting a draft
        mid-generation would need the generator to check a flag between tokens; worth doing if
        private mode ships, and cheap, but it is not measured in E15.
        """
        repo = model or self.answer_model
        t0 = time.time()
        rec = self.record(repo, self.provider)
        rec["_t0"] = t0
        if web:
            log.info("local: web search is not available offline; answering from files only")
        res = await asyncio.to_thread(_RESIDENT.get, repo)
        names = {t.name for t in tools}
        sys_text = system + ("" if getattr(res.tok, "has_tool_calling", False)
                             else _tool_fallback(tools))
        messages: list[dict] = [{"role": "system", "content": sys_text},
                                {"role": "user", "content": user + _JSON_NUDGE}]
        text, finish = "", "stop"
        async with self._sem:
            for _ in range(TOOL_ROUNDS + 1):
                text, finish = await self._run(repo, messages, tools,
                                               ANSWER_MAX_TOKENS + TOOL_MAX_TOKENS, rec, on_partial)
                calls = _tool_calls(text, res.tok, names)
                if not calls:
                    break
                rec["tool_rounds"] += 1
                messages.append({"role": "assistant", "content": text})
                for c in calls[:2]:
                    out = run_tool(c["name"], c["arguments"])
                    if asyncio.isfuture(out) or asyncio.iscoroutine(out):
                        out = await out
                    messages.append({"role": "tool", "name": c["name"], "content": str(out)[:6000]})

            data = parse_json(_strip_calls(text, res.tok))
            if data is None or not data.get("answer"):
                # one repair pass: the conversation is already cached, so this reads almost nothing
                rec["repairs"] = rec.get("repairs", 0) + 1
                messages.append({"role": "assistant", "content": text[:1500]})
                messages.append({"role": "user",
                                 "content": "That was not the JSON object. Respond with the object "
                                            "only, starting with {." + _JSON_NUDGE})
                text, finish = await self._run(repo, messages, None, ANSWER_MAX_TOKENS, rec,
                                               on_partial, prefill=_ANSWER_PREFILL)
                data = parse_json(text)
        rec["stop"] = finish
        rec["total_s"] = round(time.time() - t0, 3)
        rec.update(mem_mb())
        rec.pop("_t0", None)
        data = data or fallback_answer(text, user)
        data.setdefault("citations", [])
        return data, rec

    # ------------------------------------------------------------------ classifier

    async def classify(self, prompt: str, schema: dict, model: str | None = None) -> dict:
        repo = model or self.classifier_model
        rec = self.record(repo, self.provider)
        rec["_t0"] = time.time()
        keys = ", ".join(f'"{k}"' for k in (schema.get("properties") or {}))
        messages = [{"role": "user",
                     "content": f"{prompt}\n\nReply with one JSON object only, keys: {keys}. "
                                f"No prose, no fence."}]
        async with self._sem:
            text, _ = await self._run(repo, messages, None, CLASSIFY_MAX_TOKENS, rec, None,
                                      prefill="{")
        return parse_json(text) or {}

    # ------------------------------------------------------------------ warmup + key

    async def prewarm(self, system: str, tools: list[ToolSpec]) -> None:
        """Load the weights and leave the system prefix sitting in the KV cache.

        This is the whole ball game locally: without it the first answer prefills 15k tokens and
        takes seconds. Called on startup and again whenever `system_text()` changes.
        """
        for repo in dict.fromkeys(m for m in (self.answer_model, self.hedge_model,
                                              self.classifier_model) if m):
            t = time.time()
            rec = self.record(repo, self.provider)
            rec["_t0"] = t
            res = await asyncio.to_thread(_RESIDENT.get, repo)
            sys_text = system + ("" if getattr(res.tok, "has_tool_calling", False)
                                 else _tool_fallback(tools))
            try:
                async with self._sem:
                    head = self._constant_head(res.tok, sys_text, tools)
                    n = await asyncio.to_thread(res.cache.set_base, head)
                    await self._run(repo, [{"role": "system", "content": sys_text},
                                           {"role": "user", "content": "warmup"}],
                                    tools, 1, rec, None)
                log.info("prewarm %s: %.1fs base=%d prefill=%s reused=%s %s", repo,
                         time.time() - t, n, rec.get("prefill"), rec.get("cache_read"), mem_mb())
            except Exception as e:  # noqa: BLE001
                log.warning("prewarm %s failed: %s", repo, e)
            if len(dict.fromkeys(m for m in (self.answer_model, self.hedge_model,
                                             self.classifier_model) if m)) > 1:
                break    # a second repo would evict the first; one resident model is the design

    async def validate_key(self) -> tuple[bool, str]:
        """No key. The check is whether the weights are on disk or the Hub is reachable."""
        repo = self.answer_model
        if os.path.isdir(repo):
            return True, "ok"
        try:
            from huggingface_hub import snapshot_download

            await asyncio.to_thread(snapshot_download, repo, local_files_only=True)
            return True, "ok"
        except Exception:  # noqa: BLE001
            pass
        try:
            from huggingface_hub import model_info

            await asyncio.to_thread(model_info, repo)
            return True, f"{repo} is not downloaded yet; the first answer will fetch it"
        except Exception as e:  # noqa: BLE001
            return False, f"{repo} is neither on disk nor reachable ({type(e).__name__})"


def _strip_calls(text: str, tok: Any) -> str:
    """Drop any tool-call block so the JSON parser sees only the answer."""
    start = getattr(tok, "tool_call_start", None) or "<tool_call>"
    end = getattr(tok, "tool_call_end", None) or "</tool_call>"
    out, i = [], 0
    while True:
        a = text.find(start, i)
        if a < 0:
            out.append(text[i:])
            break
        out.append(text[i:a])
        b = text.find(end, a)
        if b < 0:
            break
        i = b + len(end)
    return "".join(out)
