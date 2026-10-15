"""Gemini provider (google-genai SDK, async client).

Written from the SDK source and the public docs; there is no Gemini key on this machine, so
every line here is verified structurally (signatures, field names, and a scripted fake client in
`presage/llm/smoke.py`) and never against the live API. `docs/research-llm.md` lists what is
VERIFIED versus ASSUMED; read it before trusting a default.

Shape of one answer, mirroring the Anthropic path:
  generate_content_stream (system_instruction + tools + response_json_schema)
    -> stream text deltas, extract the `answer` field as it grows
    -> function_call parts -> run them locally -> feed back as function_response parts
    -> up to 4 rounds, then parse the JSON object.

Automatic function calling is off: we own the loop so the answer can stream. Thinking is set
to the lowest level each model allows, because a thinking model spends its first seconds
silently and this whole program exists to put words on screen fast.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import errors, types

from presage.llm.base import (MAX_TOOL_ROUNDS, PARTIAL_EVERY_S, TIMEOUT_S, LLM, OnPartial,
                              RunTool, ToolSpec, collect_web_urls, fallback_answer, parse_json,
                              partial_field)

log = logging.getLogger("presage.llm.gemini")

MAX_OUTPUT_TOKENS = 4000
CLASSIFY_MAX_TOKENS = 512
CACHE_MIN_TOKENS = 4096          # docs: explicit + implicit cache minimum for Gemini 3 Flash
CACHE_TTL_S = 3600

# Models whose thinking can be pushed all the way down to MINIMAL (docs/thinking, 2026-09-21).
# Anything else on the 3.x line starts at LOW; 2.5-class models take thinking_budget=0 instead.
MINIMAL_THINKING = ("gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                    "gemini-flash-lite-latest")

SAFETY_STOPS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


def _env_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None


def _cache_enabled() -> bool:
    return os.environ.get("PRESAGE_GEMINI_CACHE", "0").lower() in ("1", "true", "on", "yes")


def thinking_config(model: str) -> types.ThinkingConfig | None:
    """The cheapest thinking setting this model accepts. None means "send nothing"."""
    if model.startswith("gemini-2.") or model.startswith("gemini-1."):
        return types.ThinkingConfig(thinking_budget=0, include_thoughts=False)   # 0 = DISABLED
    if model.startswith("gemini-3") or model.startswith("gemini-flash"):
        level = (types.ThinkingLevel.MINIMAL if model.startswith(MINIMAL_THINKING)
                 else types.ThinkingLevel.LOW)
        return types.ThinkingConfig(thinking_level=level, include_thoughts=False)
    return None


@dataclass
class Caps:
    """What we still believe this model accepts. Each flag is switched off for good the
    first time the API answers 400 to a request that used it."""
    thinking: bool = True
    schema_with_tools: bool = True       # response_json_schema in the same call as tools
    search_with_tools: bool = True       # google_search alongside our function declarations


def _err_text(e: Exception) -> str:
    return f"{getattr(e, 'message', '') or ''} {e}".strip().lower()


class GeminiLLM(LLM):
    provider = "gemini"

    @classmethod
    def default_models(cls) -> dict:
        # Chosen 2026-09-21 from ai.google.dev/gemini-api/docs/models + the 2026-07-21 launch post.
        # 3.6 Flash is the only Gemini 3 Flash that accepts thinking_level MINIMAL; 3.5 Flash-Lite
        # is the fastest of the line ("350 output tokens per second") and thinks minimally by default.
        return {"answer": "gemini-3.6-flash",
                "hedge": "gemini-3.5-flash-lite",
                "classifier": "gemini-3.5-flash-lite"}

    def __init__(self, api_key: str | None = None, models: dict | None = None) -> None:
        super().__init__(api_key, models)
        self._client: genai.Client | None = None
        self._caps: dict[str, Caps] = {}
        self._cache: dict[str, str] = {}      # model -> cachedContents/... name
        self._cache_key: str | None = None    # system text the caches were built from
        self._cache_off = not _cache_enabled()

    # ------------------------------------------------------------------ client

    def key(self) -> str | None:
        return self.api_key or _env_key()

    @property
    def client(self) -> genai.Client:
        """Built on first use, never in __init__: a Brain may exist long before a key does."""
        if self._client is None:
            key = self.key()
            if not key:
                raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
            self._client = genai.Client(api_key=key,
                                        http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)))
        return self._client

    def caps(self, model: str) -> Caps:
        return self._caps.setdefault(model, Caps())

    # ------------------------------------------------------------------ request shape

    @staticmethod
    def _tool_decls(tools: list[ToolSpec]) -> list[types.Tool]:
        """Anthropic-style JSON schemas go straight in as `parameters_json_schema`, which takes
        standard JSON Schema (including additionalProperties) rather than the OpenAPI subset."""
        return [types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t.name, description=t.description,
                                      parameters_json_schema=t.input_schema)
            for t in tools])]

    def _config(self, model: str, system: str | None = None, tools: list[ToolSpec] | None = None,
                schema: dict | None = None, max_tokens: int = MAX_OUTPUT_TOKENS,
                web: bool = False) -> types.GenerateContentConfig:
        caps = self.caps(model)
        kw: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        cached = self._cache.get(model) if (system and tools) else None
        if cached:
            kw["cached_content"] = cached          # system prompt + tools live in the cache
        else:
            if system:
                kw["system_instruction"] = system
            if tools:
                kw["tools"] = self._tool_decls(tools)
                kw["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.AUTO))
            if web and (not tools or caps.search_with_tools):
                kw["tools"] = (kw.get("tools") or []) + [types.Tool(google_search=types.GoogleSearch())]
        if caps.thinking:
            th = thinking_config(model)
            if th is not None:
                kw["thinking_config"] = th
        if schema is not None and (not tools or caps.schema_with_tools):
            kw["response_mime_type"] = "application/json"
            kw["response_json_schema"] = schema
        return types.GenerateContentConfig(**kw)

    def _degrade(self, model: str, e: Exception, tools: bool, schema: bool, web: bool = False) -> bool:
        """A 400 on a config we have never had accepted: drop the most likely culprit and retry."""
        if getattr(e, "code", None) != 400:
            return False
        caps, msg = self.caps(model), _err_text(e)
        if caps.thinking and ("thinking" in msg or "thought" in msg):
            caps.thinking = False
            log.warning("gemini %s rejected the thinking config, retrying without it: %s", model, msg[:160])
            return True
        if caps.search_with_tools and web and tools:
            caps.search_with_tools = False
            log.warning("gemini %s rejected google_search alongside function declarations, "
                        "falling back to a grounded final call: %s", model, msg[:160])
            return True
        if caps.schema_with_tools and tools and schema:
            caps.schema_with_tools = False
            log.warning("gemini %s rejected schema+tools in one call, falling back to a prompted "
                        "JSON loop with a formatting pass: %s", model, msg[:160])
            return True
        if caps.thinking:
            caps.thinking = False
            log.warning("gemini %s 400, retrying without the thinking config: %s", model, msg[:160])
            return True
        return False

    # ------------------------------------------------------------------ answer

    async def answer(self, model: str, system: str, user: str, tools: list[ToolSpec], schema: dict,
                     run_tool: RunTool, on_partial: OnPartial | None = None,
                     web: bool = False, lane: str = "bg") -> tuple[dict, dict]:
        t0 = time.time()
        rec = self.record(model, self.provider)
        contents: list[types.Content] = [types.Content(role="user", parts=[types.Part.from_text(text=user)])]
        text, parts, finish = "", [], None
        for _ in range(MAX_TOOL_ROUNDS):
            text, parts, finish = await self._stream(model, contents, on_partial, rec,
                                                     system=system, tools=tools, schema=schema, web=web)
            calls = [p.function_call for p in parts if p.function_call is not None]
            if not calls:
                break
            rec["tool_rounds"] += 1
            # the model turn goes back verbatim so thought signatures survive the round trip
            contents.append(types.Content(role="model", parts=parts))
            contents.append(types.Content(role="user",
                                          parts=[await self._tool_result(fc, run_tool) for fc in calls]))
        if web and not self.caps(model).search_with_tools:
            # degraded: the file loop ran ungrounded, so one final call carries google_search
            contents.append(types.Content(role="model", parts=parts or [types.Part.from_text(text=text or " ")]))
            contents.append(types.Content(role="user", parts=[types.Part.from_text(
                text="Now use Google Search if the files did not answer it, then respond with the "
                     "JSON object only.")]))
            text, parts, finish = await self._stream(model, contents, on_partial, rec,
                                                     system=system, tools=None, schema=schema, web=True)
            rec["tool_rounds"] += 1
        rec["stop"] = str(finish) if finish else None
        if finish and str(finish).split(".")[-1] in SAFETY_STOPS:
            rec["total_s"] = round(time.time() - t0, 2)
            return ({"question": user, "answer": "Model declined to answer.", "figure": None,
                     "citations": [], "in_files": False}, rec)
        data = parse_json(text)
        if data is None and text.strip() and not self.caps(model).schema_with_tools:
            # degraded loop: the tool turns carried no schema, so ask once more for the object
            data = await self._reformat(model, system, contents, parts, schema, on_partial, rec)
        rec["total_s"] = round(time.time() - t0, 2)
        data = data or fallback_answer(text, user)
        data.setdefault("citations", [])
        return data, rec

    async def _reformat(self, model: str, system: str, contents: list, parts: list, schema: dict,
                        on_partial: OnPartial | None, rec: dict) -> dict | None:
        """One extra request, schema on and tools off, when the prompted JSON came back as prose."""
        convo = list(contents)
        if parts:
            convo.append(types.Content(role="model", parts=parts))
        convo.append(types.Content(role="user", parts=[types.Part.from_text(
            text="Respond with the JSON object only, matching the required schema.")]))
        rec["tool_rounds"] += 1
        text, _, _ = await self._stream(model, convo, on_partial, rec, system=system, tools=None, schema=schema)
        return parse_json(text)

    async def _tool_result(self, fc: Any, run_tool: RunTool) -> types.Part:
        out = run_tool(fc.name or "", self.tool_args(fc.args))
        if inspect.isawaitable(out):
            out = await out            # a tool the brain runs in a thread (whatif, calc)
        try:
            payload: Any = json.loads(out)
        except (TypeError, ValueError):
            payload = out
        return types.Part(function_response=types.FunctionResponse(
            id=getattr(fc, "id", None), name=fc.name, response={"result": payload}))

    # ------------------------------------------------------------------ streaming

    async def _stream(self, model: str, contents: list, on_partial: OnPartial | None, rec: dict, *,
                      system: str | None = None, tools: list[ToolSpec] | None = None,
                      schema: dict | None = None, web: bool = False,
                      max_tokens: int = MAX_OUTPUT_TOKENS) -> tuple[str, list, Any]:
        """One streamed request, retried with fewer features if the API rejects the config.
        A 400 lands before any text does, so nothing that reached the screen is ever replayed."""
        for attempt in range(4):
            cfg = self._config(model, system=system, tools=tools, schema=schema,
                               max_tokens=max_tokens, web=web)
            try:
                return await self._stream_once(model, contents, cfg, on_partial, rec)
            except errors.ClientError as e:
                if attempt == 3 or not self._degrade(model, e, bool(tools), schema is not None, web):
                    raise
        raise RuntimeError("unreachable")

    async def _stream_once(self, model: str, contents: list, cfg: types.GenerateContentConfig,
                           on_partial: OnPartial | None, rec: dict) -> tuple[str, list, Any]:
        buf, parts, finish = "", [], None
        cur: dict[str, int] = {}
        t0 = time.time()
        last_emit, last_sent = 0.0, None
        stream = await self.client.aio.models.generate_content_stream(
            model=model, contents=contents, config=cfg)
        async for chunk in stream:
            if not rec.get("first_event_s"):
                rec["first_event_s"] = round(time.time() - t0, 2)
            usage = getattr(chunk, "usage_metadata", None)
            if usage is not None:
                cur = self._usage(usage)   # cumulative within one stream: keep the newest
            for cand in (getattr(chunk, "candidates", None) or [])[:1]:
                if cand.finish_reason:
                    finish = cand.finish_reason
                gm = getattr(cand, "grounding_metadata", None)
                if gm is not None:
                    known = {u["url"] for u in rec["web_urls"]}
                    dump = gm.model_dump() if hasattr(gm, "model_dump") else gm
                    for u in collect_web_urls(dump):
                        if u["url"] not in known:
                            known.add(u["url"])
                            rec["web_urls"].append(u)
                content = getattr(cand, "content", None)
                for p in (getattr(content, "parts", None) or []):
                    parts.append(p)
                    if not p.text or p.thought:
                        continue
                    if not rec["first_text_s"]:
                        rec["first_text_s"] = round(time.time() - t0, 2)
                    buf += p.text
                    if on_partial is None:
                        continue
                    now = time.time()
                    if now - last_emit >= PARTIAL_EVERY_S:
                        text = partial_field(buf)
                        if text and text != last_sent:
                            last_emit, last_sent = now, text
                            await on_partial(text, model)
        for k, v in cur.items():        # tool rounds add up across requests
            rec[k] = rec.get(k, 0) + v
        return buf, parts, finish

    @staticmethod
    def _usage(u: Any) -> dict[str, int]:
        thoughts = getattr(u, "thoughts_token_count", 0) or 0
        return {"in": getattr(u, "prompt_token_count", 0) or 0,
                "out": (getattr(u, "candidates_token_count", 0) or 0) + thoughts,
                "cache_read": getattr(u, "cached_content_token_count", 0) or 0,
                "thinking": thoughts}

    # ------------------------------------------------------------------ classifier

    async def classify(self, prompt: str, schema: dict, model: str | None = None) -> dict:
        max_tokens = CLASSIFY_MAX_TOKENS if model is None else MAX_OUTPUT_TOKENS
        model = model or self.classifier_model
        for attempt in range(3):
            cfg = self._config(model, schema=schema, max_tokens=max_tokens)
            try:
                resp = await self.client.aio.models.generate_content(
                    model=model, contents=prompt, config=cfg)
                return parse_json(self._text(resp)) or {}
            except errors.ClientError as e:
                if attempt == 2 or not self._degrade(model, e, tools=False, schema=True):
                    raise
        return {}

    @staticmethod
    def _text(resp: Any) -> str:
        out = []
        for cand in (getattr(resp, "candidates", None) or [])[:1]:
            content = getattr(cand, "content", None)
            for p in (getattr(content, "parts", None) or []):
                if p.text and not p.thought:
                    out.append(p.text)
        return "".join(out)

    # ------------------------------------------------------------------ warmup + key

    async def prewarm(self, system: str, tools: list[ToolSpec]) -> None:
        """Ping every model with the real prefix so routing is hot and the implicit cache has
        seen the system prompt. Explicit caches are built here too when they are switched on."""
        try:
            await self._ensure_cache(system, tools)
        except Exception as e:  # noqa: BLE001
            log.warning("gemini explicit cache unavailable, using implicit caching: %s", e)
            self._cache_off, self._cache = True, {}

        async def warm(model: str, with_prefix: bool) -> None:
            try:
                t = time.time()
                cfg = self._config(model, system=system if with_prefix else None,
                                   tools=tools if with_prefix else None, max_tokens=8)
                resp = await self.client.aio.models.generate_content(
                    model=model, contents="warmup: reply ok", config=cfg)
                u = getattr(resp, "usage_metadata", None)
                log.info("prewarm %s: %.1fs in=%s cached=%s", model, time.time() - t,
                         getattr(u, "prompt_token_count", None), getattr(u, "cached_content_token_count", None))
            except Exception as e:  # noqa: BLE001
                log.warning("prewarm %s failed: %s", model, e)

        jobs = [warm(self.answer_model, True)]
        if self.hedge_model and self.hedge_model != self.answer_model:
            jobs.append(warm(self.hedge_model, True))
        if self.classifier_model not in (self.answer_model, self.hedge_model):
            jobs.append(warm(self.classifier_model, False))
        await asyncio.gather(*jobs)

    async def _ensure_cache(self, system: str, tools: list[ToolSpec]) -> None:
        """Explicit context cache of the system prompt + tools, one per answering model.

        Off by default: the documented minimum is 4,096 tokens, which is roughly the size of
        our whole prefix, and implicit caching already covers the same prefix at the same
        threshold for free. `PRESAGE_GEMINI_CACHE=1` turns it on."""
        if self._cache_off or self._cache_key == system:
            return
        n = await self.count_tokens(self.answer_model, system)
        if n < CACHE_MIN_TOKENS:
            log.info("gemini: system prompt is %d tokens, under the %d-token explicit cache minimum; "
                     "leaving it to implicit caching", n, CACHE_MIN_TOKENS)
            self._cache_off = True
            return
        old, self._cache = dict(self._cache), {}
        for model in dict.fromkeys(m for m in (self.answer_model, self.hedge_model) if m):
            cache = await self.client.aio.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    display_name="presage-system", system_instruction=system,
                    tools=self._tool_decls(tools), ttl=f"{CACHE_TTL_S}s"))
            self._cache[model] = cache.name
            log.info("gemini cache %s for %s (%d tokens)", cache.name, model, n)
        self._cache_key = system
        for name in old.values():
            try:
                await self.client.aio.caches.delete(name=name)
            except Exception:  # noqa: BLE001
                pass

    async def count_tokens(self, model: str, text: str) -> int:
        resp = await self.client.aio.models.count_tokens(model=model, contents=text)
        return int(getattr(resp, "total_tokens", 0) or 0)

    async def validate_key(self) -> tuple[bool, str]:
        if not self.key():
            return False, "GEMINI_API_KEY (or GOOGLE_API_KEY) not set"
        try:
            client = self.client
        except Exception as e:  # noqa: BLE001
            return False, str(e)[:120]
        try:
            await client.aio.models.count_tokens(model=self.classifier_model, contents="ok")
            return True, "ok"
        except errors.ClientError as e:
            code, msg = getattr(e, "code", None), _err_text(e)
            if code in (401, 403) or "api key not valid" in msg or "api_key_invalid" in msg:
                return False, f"GEMINI_API_KEY rejected ({code or 400})"
            if code == 429:
                return False, "Gemini quota exhausted (429)"
            if code == 404:
                try:
                    await client.aio.models.list(config=types.ListModelsConfig(page_size=1))
                except Exception:  # noqa: BLE001
                    return False, f"model {self.classifier_model!r} not found"
                return False, f"key works but model {self.classifier_model!r} is not available to it"
            return False, f"Gemini rejected the request ({code}): {msg[:100]}"
        except errors.ServerError as e:
            return False, f"Gemini API unavailable ({getattr(e, 'code', '5xx')})"
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            if "Connect" in name or "Timeout" in name or isinstance(e, OSError):
                return False, "cannot reach generativelanguage.googleapis.com"
            return False, f"{name}: {str(e)[:120]}"
