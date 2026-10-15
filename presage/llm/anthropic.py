"""Anthropic provider: the call path presage was built and measured on.

Ported verbatim from the original `Brain._stream` / `_llm_answer` / `prewarm` / `_classify`:
streamed `messages.stream`, a 1-hour `cache_control` block on the system prompt, `effort=low`
where the model supports it, structured output through `output_config.format`, partial
extraction of the `answer` field, and a 4-round tool loop.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any

import anthropic

from presage.llm.base import (FILE_TOOL_ROUNDS, MAX_TOOL_ROUNDS, PARTIAL_EVERY_S, TIMEOUT_S, LLM, OnPartial,
                              RunTool, ToolSpec, collect_web_urls, fallback_answer, parse_json,
                              partial_field)

log = logging.getLogger("presage.llm.anthropic")

EFFORT = os.environ.get("PRESAGE_EFFORT", "")   # empty = no effort field: measured 644 ms faster TTFT than "low" on Sonnet 5 (docs/experiments/E2-landscape.md)
THINKING = os.environ.get("PRESAGE_THINKING", "0").lower() in ("1", "true", "on")

# Dynamic-filtering server tools; they need Opus 4.6+ / Sonnet 4.6+. Haiku 4.5 and older take the
# basic variants. A 400 naming the tool type demotes the model for the rest of the process.
WEB_LATEST = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3},
              {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 2}]
WEB_BASIC = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
             {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 2}]
_BASIC_ONLY: set[str] = set()


def supports_effort(model: str) -> bool:
    return "haiku" not in model


def web_tools(model: str) -> list[dict]:
    older = "haiku" in model or "sonnet-4-5" in model or "sonnet-3" in model or "opus-4-5" in model
    return WEB_BASIC if (older or model in _BASIC_ONLY) else WEB_LATEST


class AnthropicLLM(LLM):
    provider = "anthropic"

    @classmethod
    def default_models(cls) -> dict:
        # Measured on this account (docs/research-llm.md): Opus 5 4.3 s, Sonnet 5 2.2 s, Haiku 4.5 1.2 s.
        return {"answer": "claude-sonnet-5",
                "hedge": "claude-haiku-4-5-20251001",
                "classifier": "claude-haiku-4-5-20251001"}

    def __init__(self, api_key: str | None = None, models: dict | None = None) -> None:
        super().__init__(api_key, models)
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            kw: dict[str, Any] = {"timeout": TIMEOUT_S, "max_retries": 1}
            if self.api_key:
                kw["api_key"] = self.api_key
            self._client = anthropic.AsyncAnthropic(**kw)
        return self._client

    # ------------------------------------------------------------------ request shape

    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict]:
        return [{"name": t.name, "description": t.description, "strict": True,
                 "input_schema": t.input_schema} for t in tools]

    def _request_kwargs(self, model: str, messages: list, system: str, tools: list[ToolSpec],
                        schema: dict | None, max_tokens: int = 4000, web: bool = False,
                        tool_choice: dict | None = None) -> dict:
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            tools=self._tools(tools) + (web_tools(model) if web else []),
            messages=messages,
        )
        output_config: dict[str, Any] = {}
        if EFFORT and supports_effort(model):
            output_config["effort"] = EFFORT
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config
        if not THINKING and supports_effort(model):
            # a live answer is a lookup plus a sentence; thinking blocks measured 1-3 s on Sonnet 5
            kwargs["thinking"] = {"type": "disabled"}
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return kwargs

    # ------------------------------------------------------------------ answer

    async def answer(self, model: str, system: str, user: str, tools: list[ToolSpec], schema: dict,
                     run_tool: RunTool, on_partial: OnPartial | None = None,
                     web: bool = False, lane: str = "bg") -> tuple[dict, dict]:
        t0 = time.time()
        rec = self.record(model, self.provider)
        messages: list[dict] = [{"role": "user", "content": user}]
        resp = None
        for _ in range(MAX_TOOL_ROUNDS if web else FILE_TOOL_ROUNDS):
            resp = await self._stream(model, messages, system, tools, schema, on_partial, rec, web)
            if web:
                self._web_urls(rec, resp.content)
            if resp.stop_reason == "tool_use":
                rec["tool_rounds"] += 1
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for b in resp.content:
                    if b.type == "tool_use":
                        out = run_tool(b.name, self.tool_args(b.input))
                        if inspect.isawaitable(out):
                            out = await out        # a tool the brain runs in a thread (whatif, calc)
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
                messages.append({"role": "user", "content": results})
                continue
            if resp.stop_reason == "pause_turn":
                # server-side tool loop hit its iteration limit: resend as-is, no extra user turn
                rec["tool_rounds"] += 1
                messages.append({"role": "assistant", "content": resp.content})
                continue
            break
        if resp is not None and resp.stop_reason == "tool_use":
            # the round cap hit while the model still wanted tools: one more request, tools off,
            # so it answers from what it has instead of returning nothing
            messages[-1]["content"].append({"type": "text", "text": "Tool budget spent. Answer now from the "
                                            "results above; the JSON object only."})
            rec["tool_rounds"] += 1
            resp = await self._stream(model, messages, system, tools, schema, on_partial, rec, web,
                                      tool_choice={"type": "none"})
        rec["total_s"] = round(time.time() - t0, 2)
        if resp is None:
            raise RuntimeError("no response")
        rec["stop"] = resp.stop_reason
        if resp.stop_reason == "refusal":
            return ({"question": user, "answer": "Model declined to answer.", "figure": None,
                     "citations": [], "in_files": False}, rec)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        data = parse_json(text) or fallback_answer(text, user)
        data.setdefault("citations", [])
        return data, rec

    @staticmethod
    def _web_urls(rec: dict, content: Any) -> None:
        """Server-tool results are blocks, and an error returns an object where a success returns
        a list, so walk them blindly rather than indexing."""
        known = {u["url"] for u in rec["web_urls"]}
        for b in content:
            if getattr(b, "type", "") in ("web_search_tool_result", "web_fetch_tool_result"):
                for u in collect_web_urls(getattr(b, "content", None)):
                    if u["url"] not in known:
                        known.add(u["url"])
                        rec["web_urls"].append(u)

    async def _stream(self, model: str, messages: list, system: str, tools: list[ToolSpec],
                      schema: dict | None, on_partial: OnPartial | None, rec: dict,
                      web: bool = False, tool_choice: dict | None = None):
        """One streamed request. Text deltas feed on_partial with the `answer` field as it grows."""
        kwargs = self._request_kwargs(model, messages, system, tools, schema, web=web, tool_choice=tool_choice)
        try:
            return await self._stream_once(kwargs, model, on_partial, rec)
        except anthropic.BadRequestError as e:
            msg = str(e)
            if not web or model in _BASIC_ONLY or "web_" not in msg:
                raise
            _BASIC_ONLY.add(model)      # this model only has the basic web tools
            log.warning("%s rejected the dynamic-filtering web tools, retrying basic: %s", model, msg[:160])
            kwargs = self._request_kwargs(model, messages, system, tools, schema, web=True)
            return await self._stream_once(kwargs, model, on_partial, rec)

    async def _stream_once(self, kwargs: dict, model: str, on_partial: OnPartial | None, rec: dict):
        buf = ""
        last_emit = 0.0
        last_sent = None
        t0 = time.time()
        t_first = t_text = None
        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if t_first is None:
                    t_first = time.time()
                    if not rec.get("first_event_s"):
                        rec["first_event_s"] = round(t_first - t0, 2)
                if event.type != "content_block_delta" or getattr(event.delta, "type", "") != "text_delta":
                    continue
                if t_text is None:
                    t_text = time.time()
                    if not rec["first_text_s"]:
                        rec["first_text_s"] = round(t_text - t0, 2)
                if on_partial is None:
                    continue
                buf += event.delta.text
                now = time.time()
                if now - last_emit >= PARTIAL_EVERY_S:
                    text = partial_field(buf)
                    if text and text != last_sent:
                        last_emit, last_sent = now, text
                        await on_partial(text, model)
            resp = await stream.get_final_message()
        u = resp.usage
        rec["in"] += u.input_tokens or 0
        rec["out"] += u.output_tokens or 0
        rec["cache_read"] += u.cache_read_input_tokens or 0
        rec["cache_write"] += u.cache_creation_input_tokens or 0
        rec["thinking"] += sum(1 for b in resp.content if b.type == "thinking")
        return resp

    # ------------------------------------------------------------------ classifier

    async def classify(self, prompt: str, schema: dict, model: str | None = None) -> dict:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if EFFORT and model and supports_effort(model):
            output_config["effort"] = EFFORT
        resp = await self.client.messages.create(
            model=model or self.classifier_model,
            max_tokens=300 if model is None else 1000,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        return parse_json(text) or {}

    # ------------------------------------------------------------------ warmup + key

    async def prewarm(self, system: str, tools: list[ToolSpec]) -> None:
        """Write the system prompt + tools into the prompt cache on the answer model (Haiku's
        4096-token minimum is above our prefix) and ping every model so routing is hot.
        Measured: the first call to a model after idle takes 7-19 s, warm calls 1-3 s."""
        async def warm(model: str, max_tokens: int) -> None:
            try:
                kwargs = self._request_kwargs(model, [{"role": "user", "content": "warmup: reply ok"}],
                                              system, tools, schema=None, max_tokens=max_tokens)
                t = time.time()
                resp = await self.client.messages.create(**kwargs)
                u = resp.usage
                log.info("prewarm %s: %.1fs cache_write=%s cache_read=%s", model, time.time() - t,
                         getattr(u, "cache_creation_input_tokens", None), getattr(u, "cache_read_input_tokens", None))
            except Exception as e:  # noqa: BLE001
                log.warning("prewarm %s failed: %s", model, e)

        jobs = [warm(self.answer_model, 0)]
        if self.hedge_model and self.hedge_model != self.answer_model:
            jobs.append(warm(self.hedge_model, 1))
        if self.classifier_model not in (self.answer_model, self.hedge_model):
            jobs.append(warm(self.classifier_model, 1))
        await asyncio.gather(*jobs)

    async def validate_key(self) -> tuple[bool, str]:
        if not (self.api_key or os.environ.get("ANTHROPIC_API_KEY")):
            return False, "ANTHROPIC_API_KEY not set"
        try:
            await self.client.models.list(limit=1)
            return True, "ok"
        except anthropic.AuthenticationError:
            return False, "ANTHROPIC_API_KEY rejected (401)"
        except anthropic.PermissionDeniedError:
            return False, "key has no permission for this workspace (403)"
        except anthropic.RateLimitError:
            return True, "ok (rate limited right now)"
        except anthropic.APIConnectionError:
            return False, "cannot reach api.anthropic.com"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {str(e)[:120]}"
