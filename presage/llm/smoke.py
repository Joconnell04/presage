"""Offline checks for the LLM layer: `uv run python -m presage.llm.smoke`.

No network. The Gemini path is exercised against a fake client whose
`generate_content_stream` yields scripted `types.GenerateContentResponse` chunks (a function
call, then a streamed JSON answer), which is the only way to test it on a machine with no
Gemini key. Everything the fake returns is a real SDK type, so field names and shapes are
checked by pydantic rather than by hope.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from types import SimpleNamespace
from typing import Any

from google.genai import errors, types

from presage.llm import ToolSpec, make_llm, parse_json, partial_field
from presage.llm.base import collect_web_urls
from presage.llm.gemini import GeminiLLM

FAILURES: list[str] = []
TOOLS = [ToolSpec(name="sql", description="Run one SELECT.",
                  input_schema={"type": "object", "properties": {"query": {"type": "string"}},
                                "required": ["query"], "additionalProperties": False})]
SCHEMA = {"type": "object",
          "properties": {"answer": {"type": "string"},
                         "figure": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                         "citations": {"type": "array", "items": {
                             "type": "object",
                             "properties": {"file": {"type": "string"}, "loc": {"type": "string"}},
                             "required": ["file", "loc"], "additionalProperties": False}},
                         "in_files": {"type": "boolean"},
                         "question": {"type": "string"}},
          "required": ["answer", "figure", "citations", "in_files", "question"],
          "additionalProperties": False}

ANSWER_JSON = ('{"answer": "Q3 FY25 revenue was $14.0M, 3% ahead of plan.", "figure": "$14.0M", '
               '"citations": [{"file": "Acme.xlsx", "loc": "P&L \\u203a D6"}], "in_files": true, '
               '"question": "What was Q3 revenue?"}')


def check(name: str, cond: Any, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  <- ' + str(detail)[:300]}")
    if not cond:
        FAILURES.append(name)


# ----------------------------------------------------------------------------- fake client

def chunk(parts: list[types.Part], finish: str | None = None, usage: tuple[int, int] | None = None
          ) -> types.GenerateContentResponse:
    u = None
    if usage:
        u = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=usage[0], candidates_token_count=usage[1], thoughts_token_count=0,
            cached_content_token_count=0, total_token_count=usage[0] + usage[1])
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts),
                                    finish_reason=finish)],
        usage_metadata=u)


def text_chunks(text: str, n: int = 4, usage: tuple[int, int] = (4100, 90)) -> list:
    step = max(1, len(text) // n)
    pieces = [text[i:i + step] for i in range(0, len(text), step)]
    return [chunk([types.Part(text=p)], usage=usage) for p in pieces[:-1]] + \
           [chunk([types.Part(text=pieces[-1])], finish="STOP", usage=usage)]


class FakeModels:
    def __init__(self, script: list) -> None:
        self.script, self.calls = list(script), []

    def _next(self, kind: str, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"kind": kind, "model": model, "contents": contents, "config": config})
        if not self.script:
            raise AssertionError("fake client ran out of scripted steps")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def generate_content_stream(self, *, model, contents, config):
        chunks = self._next("stream", model, contents, config)

        async def gen():
            for c in chunks:
                yield c
        return gen()

    async def generate_content(self, *, model, contents, config=None):
        step = self._next("single", model, contents, config)
        return step if isinstance(step, types.GenerateContentResponse) else step[-1]

    async def count_tokens(self, *, model, contents, config=None):
        self.calls.append({"kind": "count", "model": model})
        return types.CountTokensResponse(total_tokens=4200)


class FakeClient:
    def __init__(self, script: list) -> None:
        self.models = FakeModels(script)
        self.aio = SimpleNamespace(models=self.models, caches=SimpleNamespace())


def gemini(script: list, **kw) -> tuple[GeminiLLM, FakeModels]:
    llm = make_llm("gemini", api_key="fake-key-for-tests", **kw)
    fake = FakeClient(script)
    llm._client = fake                      # noqa: SLF001 - the point of the test
    return llm, fake.models


def run_tool_recorder(out: str = '[{"cell": "D6", "value": 14020}]'):
    seen: list[tuple[str, dict]] = []

    def run_tool(name: str, args: dict) -> str:
        seen.append((name, args))
        return out
    return run_tool, seen


# ----------------------------------------------------------------------------- tests

async def t_request_shape() -> None:
    print("\ngemini: request construction")
    llm, fake = gemini([text_chunks(ANSWER_JSON)])
    data, rec = await llm.answer("gemini-3.6-flash", "SYSTEM PROMPT", "USER PROMPT", TOOLS, SCHEMA,
                                 run_tool_recorder()[0])
    cfg = fake.calls[0]["config"]
    check("system_instruction passed through", cfg.system_instruction == "SYSTEM PROMPT", cfg.system_instruction)
    check("user content is one user Content", fake.calls[0]["contents"][0].role == "user"
          and fake.calls[0]["contents"][0].parts[0].text == "USER PROMPT")
    decl = cfg.tools[0].function_declarations[0]
    check("tool declared with parameters_json_schema",
          decl.name == "sql" and decl.parameters_json_schema == TOOLS[0].input_schema, decl)
    check("automatic function calling disabled", cfg.automatic_function_calling.disable is True)
    check("function calling mode AUTO",
          cfg.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO)
    check("json mime + schema", cfg.response_mime_type == "application/json"
          and cfg.response_json_schema == SCHEMA)
    check("response_schema left unset (mutually exclusive)", cfg.response_schema is None)
    check("thinking MINIMAL on 3.6-flash",
          cfg.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL, cfg.thinking_config)
    check("answer parsed", data["figure"] == "$14.0M" and data["in_files"] is True, data)
    check("record shape", rec["tool_rounds"] == 0 and rec["in"] == 4100 and rec["out"] == 90
          and rec["stop"] and rec["total_s"] >= 0 and rec["first_text_s"] >= 0, rec)

    llm2, fake2 = gemini([text_chunks(ANSWER_JSON)], models={"answer": "gemini-2.5-flash"})
    await llm2.answer("gemini-2.5-flash", "S", "U", TOOLS, SCHEMA, run_tool_recorder()[0])
    check("thinking_budget=0 on 2.5-class",
          fake2.calls[0]["config"].thinking_config.thinking_budget == 0, fake2.calls[0]["config"].thinking_config)


async def t_tool_loop() -> None:
    print("\ngemini: tool loop + partial streaming")
    call = types.Part(function_call=types.FunctionCall(
        id="fc-1", name="sql", args={"query": "SELECT * FROM pl WHERE _row = 6"}))
    script = [
        [chunk([types.Part(text="")], usage=(4100, 12)), chunk([call], finish="STOP", usage=(4100, 24))],
        text_chunks(ANSWER_JSON, n=6, usage=(4400, 96)),
    ]
    llm, fake = gemini(script)
    run_tool, seen = run_tool_recorder()
    partials: list[tuple[str, str]] = []

    async def on_partial(text: str, model: str) -> None:
        partials.append((text, model))

    data, rec = await llm.answer("gemini-3.6-flash", "SYS", "USER", TOOLS, SCHEMA, run_tool, on_partial)
    check("tool ran once with parsed args", seen == [("sql", {"query": "SELECT * FROM pl WHERE _row = 6"})], seen)
    check("two requests", len(fake.calls) == 2, fake.calls)
    convo = fake.calls[1]["contents"]
    check("model turn replayed verbatim (thought signatures survive)",
          convo[1].role == "model" and convo[1].parts[-1].function_call.name == "sql", convo)
    fr = convo[2].parts[0].function_response
    check("function_response carries the call id", convo[2].role == "user" and fr.id == "fc-1"
          and fr.name == "sql" and fr.response["result"][0]["value"] == 14020, fr)
    check("partial answers streamed", partials and all(m == "gemini-3.6-flash" for _, m in partials)
          and partials[-1][0].startswith("Q3 FY25 revenue"), partials)
    check("partials never leak JSON syntax", not any('"' in p or "{" in p for p, _ in partials), partials)
    check("final answer parsed", data["answer"].startswith("Q3 FY25 revenue")
          and data["citations"][0]["loc"] == "P&L › D6", data)
    check("tokens summed across rounds", rec["tool_rounds"] == 1 and rec["in"] == 8500
          and rec["out"] == 120, rec)


async def t_degrade_schema() -> None:
    print("\ngemini: 400 on schema+tools -> prompted loop + one formatting pass")
    err = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                             "message": "Function calling is not enabled for models with a response schema"}})
    script = [err,
              text_chunks("Q3 revenue was $14.0M, 3% ahead of plan.", n=2),   # prose, no JSON
              text_chunks(ANSWER_JSON, n=3)]                                   # the formatting pass
    llm, fake = gemini(script)
    data, rec = await llm.answer("gemini-3.6-flash", "SYS", "USER", TOOLS, SCHEMA, run_tool_recorder()[0])
    check("cap remembered", llm.caps("gemini-3.6-flash").schema_with_tools is False)
    check("retried without the schema", fake.calls[1]["config"].response_json_schema is None
          and fake.calls[1]["config"].tools is not None, fake.calls[1]["config"])
    check("formatting pass has schema and no tools",
          fake.calls[2]["config"].response_json_schema == SCHEMA and not fake.calls[2]["config"].tools)
    check("formatting pass keeps the conversation", len(fake.calls[2]["contents"]) == 3
          and fake.calls[2]["contents"][-1].role == "user", fake.calls[2]["contents"])
    check("answer recovered", data["figure"] == "$14.0M", data)
    check("second answer on the same llm skips the doomed combination", True)
    llm.caps("gemini-3.6-flash").schema_with_tools is False
    fake.script.append(text_chunks(ANSWER_JSON, n=2))
    await llm.answer("gemini-3.6-flash", "SYS", "USER2", TOOLS, SCHEMA, run_tool_recorder()[0])
    check("no schema sent on the retry-free path", fake.calls[3]["config"].response_json_schema is None)


async def t_degrade_thinking() -> None:
    print("\ngemini: 400 on the thinking config")
    err = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                             "message": "thinking_level is not supported for this model"}})
    llm, fake = gemini([err, text_chunks(ANSWER_JSON, n=2)])
    data, _ = await llm.answer("gemini-3.8-flash", "SYS", "USER", TOOLS, SCHEMA, run_tool_recorder()[0])
    check("thinking LOW chosen for a non-MINIMAL model",
          fake.calls[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW)
    check("retried with no thinking config", fake.calls[1]["config"].thinking_config is None)
    check("schema kept (only thinking was dropped)", fake.calls[1]["config"].response_json_schema == SCHEMA)
    check("answer still returned", data["figure"] == "$14.0M", data)


async def t_non_400_propagates() -> None:
    print("\ngemini: 429 and safety stops")
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota"}})
    llm, fake = gemini([err])
    try:
        await llm.answer("gemini-3.6-flash", "S", "U", TOOLS, SCHEMA, run_tool_recorder()[0])
        check("429 raises", False, "no exception")
    except errors.ClientError as e:
        check("429 raises instead of degrading", e.code == 429 and len(fake.calls) == 1)
    llm2, _ = gemini([[chunk([types.Part(text="")], finish="SAFETY", usage=(10, 0))]])
    data, rec = await llm2.answer("gemini-3.6-flash", "S", "U", TOOLS, SCHEMA, run_tool_recorder()[0])
    check("safety stop renders as a refusal", data["in_files"] is False
          and data["answer"] == "Model declined to answer.", data)


async def t_classify() -> None:
    print("\ngemini: classifier")
    verdict = '{"kind": "question", "needs_files": true, "confidence": 0.9, "question": "What was Q3 revenue?"}'
    llm, fake = gemini([chunk([types.Part(text=verdict)], finish="STOP", usage=(120, 30))])
    out = await llm.classify("classify this", SCHEMA)
    cfg = fake.calls[0]["config"]
    check("classifier model used", fake.calls[0]["model"] == "gemini-3.5-flash-lite", fake.calls[0]["model"])
    check("no tools, schema on", not cfg.tools and cfg.response_json_schema == SCHEMA)
    check("verdict parsed", out["kind"] == "question" and out["confidence"] == 0.9, out)
    llm2, fake2 = gemini([chunk([types.Part(text=verdict)], finish="STOP", usage=(120, 30))])
    await llm2.classify("a bigger json job", SCHEMA, model="gemini-3.6-flash")
    check("explicit model overrides the classifier", fake2.calls[0]["model"] == "gemini-3.6-flash")
    check("and gets more output room",
          fake2.calls[0]["config"].max_output_tokens > cfg.max_output_tokens,
          (fake2.calls[0]["config"].max_output_tokens, cfg.max_output_tokens))
    a = make_llm("anthropic", api_key="fake")
    import inspect
    for name, cls in (("anthropic", type(a)), ("claude_code", type(make_llm("claude_code")))):
        params = inspect.signature(cls.classify).parameters
        check(f"{name}.classify takes model=None",
              "model" in params and params["model"].default is None, list(params))


async def t_no_key() -> None:
    print("\ngemini: no key on this machine")
    import os
    saved = {k: os.environ.pop(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") if k in os.environ}
    try:
        llm = make_llm("gemini")            # must not raise: the key is checked later
        check("constructed without a key", llm.provider == "gemini")
        ok, why = await llm.validate_key()
        check("validate_key explains itself", ok is False and "GEMINI_API_KEY" in why, why)
        try:
            llm.client
            check("client raises without a key", False, "no exception")
        except RuntimeError as e:
            check("client raises without a key", "GEMINI_API_KEY" in str(e))
        check("defaults still reportable", make_llm("gemini").state()["answer"] == "gemini-3.6-flash")
    finally:
        os.environ.update(saved)
    try:
        make_llm("llama")
        check("unknown provider raises", False, "no exception")
    except ValueError as e:
        check("unknown provider raises", "unknown LLM provider" in str(e))


async def t_helpers() -> None:
    print("\nshared helpers")
    check("partial mid-string", partial_field('{"answer": "Q3 revenue was $14.0') == "Q3 revenue was $14.0")
    check("partial with an escape", partial_field('{"answer": "P&L \\u203a D6') == "P&L › D6")
    check("partial before the field", partial_field('{"ans') is None)
    check("parse fenced json", parse_json('```json\n{"a": 1}\n```') == {"a": 1})
    check("parse prose-wrapped json", parse_json('Sure:\n{"a": 1}\nhope that helps') == {"a": 1})
    check("parse failure is None", parse_json("not json at all") is None)


async def t_swap() -> None:
    print("\nbrain: provider round trip")
    from presage.brain import Brain

    class Index:
        version = 1

        def schema_summary(self) -> str:
            return "(no files)"

    async def emit(msg: dict) -> None:
        pass

    brain = Brain(Index(), emit)
    check("starts on anthropic", brain.llm.provider == "anthropic" and brain.state()["models"]["provider"] == "anthropic")
    import os
    saved = {k: os.environ.pop(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") if k in os.environ}
    try:
        state = brain.set_provider("gemini")
        check("swapped to gemini", brain.llm.provider == "gemini" and state["answer"] == "gemini-3.6-flash", state)
        ok, why = await brain.check_key()
        check("key_ok false with a clear reason", ok is False and "GEMINI_API_KEY" in why, why)
        check("state reports the gemini models", brain.state()["models"]["hedge"] == "gemini-3.5-flash-lite")
        await brain.prewarm()      # must not raise even with no key
        check("prewarm survives a keyless provider", True)
    finally:
        os.environ.update(saved)
    brain.set_provider("anthropic")
    check("swapped back", brain.llm.provider == "anthropic"
          and brain.llm.answer_model == "claude-sonnet-5" and brain.state()["models"]["provider"] == "anthropic")
    ok, why = await brain.check_key()
    check("anthropic key still usable", ok, why)




# ----------------------------------------------------------------------------- web search

async def t_web_shapes() -> None:
    print("\nweb search: request shapes (offline)")
    a = make_llm("anthropic", api_key="fake")
    plain = a._request_kwargs("claude-sonnet-5", [], "SYS", TOOLS, SCHEMA)
    check("web=False sends only our tools", [t["name"] for t in plain["tools"]] == ["sql"], plain["tools"])
    son = a._request_kwargs("claude-sonnet-5", [], "SYS", TOOLS, SCHEMA, web=True)
    types_ = [t.get("type") for t in son["tools"] if t.get("type")]
    check("sonnet 5 gets the dynamic-filtering web tools",
          types_ == ["web_search_20260209", "web_fetch_20260209"], types_)
    hai = a._request_kwargs("claude-haiku-4-5-20251001", [], "SYS", TOOLS, SCHEMA, web=True)
    types_ = [t.get("type") for t in hai["tools"] if t.get("type")]
    check("haiku 4.5 gets the basic web tools",
          types_ == ["web_search_20250305", "web_fetch_20250910"], types_)
    check("web tools are capped", all(t.get("max_uses") for t in hai["tools"] if t.get("type")))
    check("our tools survive alongside", [t["name"] for t in son["tools"]][0] == "sql")
    check("rest of the request is unchanged",
          {k: v for k, v in son.items() if k != "tools"} == {k: v for k, v in plain.items() if k != "tools"})

    g, _ = gemini([])
    cfg = g._config("gemini-3.6-flash", system="S", tools=TOOLS, schema=SCHEMA, web=True)
    kinds = [("search" if t.google_search else "functions") for t in cfg.tools]
    check("gemini sends google_search next to the declarations", kinds == ["functions", "search"], kinds)
    g.caps("gemini-3.6-flash").search_with_tools = False
    cfg = g._config("gemini-3.6-flash", system="S", tools=TOOLS, schema=SCHEMA, web=True)
    check("degraded gemini keeps only the declarations",
          [t.google_search for t in cfg.tools] == [None], cfg.tools)
    cfg = g._config("gemini-3.6-flash", system="S", tools=None, schema=SCHEMA, web=True)
    check("grounded final call carries search alone", cfg.tools[0].google_search is not None)

    from presage.llm.claude_code import PREFIX, ClaudeCodeLLM, _Session
    cc = ClaudeCodeLLM()
    sess = _Session(key="k", model="claude-sonnet-5", client=None)
    o = cc._options(sess, "claude-sonnet-5", "SYS", TOOLS, SCHEMA, web=True)
    check("agent sdk enables the built-in web tools", o.tools == ["WebSearch", "WebFetch"], o.tools)
    check("and allows them plus ours",
          o.allowed_tools == [PREFIX + "sql", "WebSearch", "WebFetch"], o.allowed_tools)
    o = cc._options(sess, "claude-sonnet-5", "SYS", TOOLS, SCHEMA, web=False)
    check("web=False leaves every built-in tool off", o.tools == [] and o.allowed_tools == [PREFIX + "sql"])


async def t_web_urls() -> None:
    print("\nweb search: url collection")
    anthropic_shape = [{"type": "web_search_result", "title": "Fed funds rate",
                        "url": "https://www.federalreserve.gov/x.htm"}]
    gemini_shape = {"grounding_chunks": [{"web": {"uri": "https://example.gov/rate", "title": "Rate"}}]}
    urls = collect_web_urls([anthropic_shape, gemini_shape])
    check("both provider shapes yield urls", len(urls) == 2, urls)
    check("titles kept", urls[0]["title"] == "Fed funds rate" and urls[1]["url"] == "https://example.gov/rate")
    check("duplicates collapse", len(collect_web_urls([anthropic_shape, anthropic_shape])) == 1)
    check("error objects do not raise", collect_web_urls({"error_code": "max_uses_exceeded"}) == [])


# ----------------------------------------------------------------------------- claude_code

async def t_claude_code_env() -> None:
    print("\nclaude_code: the API key must never reach the child")
    import os
    import tempfile
    from claude_agent_sdk import ClaudeSDKClient

    from presage.llm.claude_code import ClaudeCodeLLM, _Session, api_key_hidden, child_env_preview
    saved = {k: os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-poison-do-not-bill"
    os.environ["ANTHROPIC_AUTH_TOKEN"] = "oauth-poison"
    try:
        env = child_env_preview()
        check("preview drops the key", "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env)
        check("os.environ restored", os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-poison-do-not-bill")
        with api_key_hidden():
            check("hidden inside the block", "ANTHROPIC_API_KEY" not in os.environ)

        # spawn a fake CLI that dumps its real environment, and check the child process itself
        d = tempfile.mkdtemp(prefix="presage-cc-")
        dump, fake = os.path.join(d, "env.json"), os.path.join(d, "claude")
        with open(fake, "w") as f:
            f.write("#!/usr/bin/env python3\nimport json, os, sys\n"
                    f"json.dump(dict(os.environ), open({dump!r}, 'w'))\nsys.exit(0)\n")
        os.chmod(fake, 0o755)
        os.environ["CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"] = "1"
        llm = ClaudeCodeLLM()
        sess = _Session(key="k", model="claude-sonnet-5", client=None)
        opts = llm._options(sess, "claude-sonnet-5", "SYS", TOOLS, SCHEMA, web=False)
        opts.cli_path = fake
        client = ClaudeSDKClient(options=opts)
        try:
            async with asyncio.timeout(30):
                with api_key_hidden():
                    await client.connect()
        except Exception:
            pass          # the fake exits immediately; we only care about the env it saw
        finally:
            with contextlib_suppress():
                await client.disconnect()
        child = json.load(open(dump)) if os.path.exists(dump) else None
        check("the child process actually spawned", child is not None, dump)
        if child:
            check("child env has no ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY" not in child,
                  sorted(k for k in child if k.startswith("ANTHROPIC")))
            check("child env has no ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN" not in child)
            check("child still gets the rest of the env", child.get("HOME") == os.environ.get("HOME"))
            check("child is marked as an SDK entrypoint", child.get("CLAUDE_CODE_ENTRYPOINT") == "sdk-py")
    finally:
        os.environ.pop("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", None)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    check("key restored after the spawn", os.environ.get("ANTHROPIC_API_KEY") in (None, saved["ANTHROPIC_API_KEY"]))


def contextlib_suppress():
    import contextlib
    return contextlib.suppress(Exception)


async def t_claude_code_login() -> None:
    print("\nclaude_code: tools and login check")
    from presage.llm.claude_code import PREFIX, ClaudeCodeLLM, _Session
    llm = ClaudeCodeLLM()
    seen: list = []
    sess = _Session(key="k", model="m", client=None, run_tool=lambda n, a: seen.append((n, a)) or "[]")
    check("mcp prefix stripped before run_tool",
          sess.call(PREFIX + "sql", {"query": "SELECT 1"}) == "[]" and seen == [("sql", {"query": "SELECT 1"})], seen)
    check("bare names work too", sess.call("search", {"query": "x"}) is not None)
    sess.run_tool = None
    check("unbound callback answers with an error", "error" in sess.call("sql", {}))
    ok, why = await llm.validate_key()
    print(f"       claude auth status -> ok={ok} why={why!r}")
    check("validate_key answers without raising", isinstance(ok, bool) and isinstance(why, str))
    if not ok:
        check("and tells the user how to fix it", "log in" in why or "not found" in why, why)


# ----------------------------------------------------------------------------- jev

def jev_server(capture: dict, answers: dict | None = None, status: int = 200, body: Any = None):
    """A fake api.typesafe.ai speaking the shapes measured on 2026-09-21."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        capture["url"] = str(request.url)
        capture["auth"] = request.headers.get("authorization")
        capture["ua"] = request.headers.get("user-agent")
        capture["body"] = json.loads(request.content)
        if status != 200:
            return httpx.Response(status, json=body, headers={"x-typesafe-request-id": "req_test"})
        return httpx.Response(200, headers={"x-typesafe-request-id": "req_test"}, json={
            "model": "jev-1.13.0",
            "answers": answers if answers is not None else {
                "is_question": {"type": "noul", "noul": 0.98},
                "kind": {"type": "choice", "choice": "question", "confidence": 0.99,
                         "probabilities": {"question": 0.99, "request": 0.01, "statement": 0.0}},
                "urgency": {"type": "score", "score": 1.31, "confidence": 0.46,
                            "legend": {"0": "0", "1": "0.25", "2": "0.5", "3": "0.75", "4": "1"},
                            "probabilities": {"0": 0.17, "1": 0.4, "2": 0.4, "3": 0.03, "4": 0.0}},
            },
            "usage": {"input_tokens": 661, "output_tokens": 96},
        })
    return httpx.MockTransport(handler)


JEV_QS = None


def _jev_qs():
    from presage.llm.jev import Q
    return [Q("is_question", "noul", "Is the last line a question aimed at the speaker?"),
            Q("kind", "choice", "What is the last line?",
              options=["question", "request", "statement"]),
            Q("urgency", "score", "How urgent does the last line sound?", lo=0.0, hi=1.0)]


async def t_jev_wire() -> None:
    print("\njev: wire format (fake server)")
    from presage.llm.jev import JevDecider, Q
    cap: dict = {}
    d = JevDecider(api_key="test-key", transport=jev_server(cap))
    out = await d.decide("Them: what did Q3 look like versus plan?", _jev_qs())

    check("posts to /v1/systemone", cap["url"] == "https://api.typesafe.ai/v1/systemone", cap["url"])
    check("bearer auth", cap["auth"] == "Bearer test-key")
    check("own user agent, not langchain", "presage" in (cap["ua"] or ""), cap["ua"])
    body = cap["body"]
    check("body is state + model + questions", set(body) == {"state", "model", "questions"}, list(body))
    check("model defaults to jev-latest", body["model"] == "jev-latest", body["model"])
    check("state is the raw string", body["state"].startswith("Them:"))
    check("all questions in ONE request", len(body["questions"]) == 3, body["questions"])
    q = body["questions"]
    check("noul carries no criteria", q["is_question"] == {"type": "noul",
          "instructions": "Is the last line a question aimed at the speaker?"}, q["is_question"])
    check("choice criteria maps label to description",
          q["kind"]["criteria"] == {"question": "question", "request": "request", "statement": "statement"},
          q["kind"])
    check("score criteria is an ordered rubric of 5 levels",
          q["urgency"]["criteria"] == ["0", "0.25", "0.5", "0.75", "1"], q["urgency"])

    n, c, sc = out["is_question"], out["kind"], out["urgency"]
    check("noul -> bool at p>=0.5", n.value is True and n.p == 0.98, n)
    check("noul confidence derived from the coin flip", abs(n.confidence - 0.96) < 1e-9, n.confidence)
    check("noul probs both ways", n.probs["true"] == 0.98 and abs(n.probs["false"] - 0.02) < 1e-9, n.probs)
    check("choice -> option + its own p", c.value == "question" and c.p == 0.99, c)
    check("choice confidence is the API's", c.confidence == 0.99 and c.probs["request"] == 0.01, c)
    check("score maps level space onto lo..hi", abs(sc.value - 0.3275) < 1e-9, sc.value)
    check("score p is the confidence", sc.p == 0.46 and sc.confidence == 0.46, sc)
    check("every decision carries latency_ms", all(x.latency_ms >= 0 for x in out.values()))
    await d.aclose()

    cap2: dict = {}
    d2 = JevDecider(api_key="k", transport=jev_server(cap2, answers={"m": {"type": "score", "score": 3.0,
                    "confidence": 0.8, "probabilities": {"3": 0.8}}}))
    out2 = await d2.decide("x", [Q("m", "score", "rate it", options=["a", "b", "c", "d"], lo=1, hi=10)])
    check("explicit rubric and range respected", abs(out2["m"].value - 10.0) < 1e-9, out2["m"].value)
    check("custom levels sent verbatim", cap2["body"]["questions"]["m"]["criteria"] == ["a", "b", "c", "d"])
    await d2.aclose()


async def t_jev_errors() -> None:
    print("\njev: failure modes")
    import httpx

    from presage.llm.jev import DecisionError, JevDecider, Q
    cases = [
        (401, {"detail": {"error_type": "authentication_error", "message": "Cannot authenticate with the server."}},
         "Cannot authenticate"),
        (400, {"detail": "Choice question must have at least one choice: a"}, "at least one choice"),
        (422, {"detail": [{"type": "too_short", "loc": ["body", "questions"],
                           "msg": "Dictionary should have at least 1 item after validation, not 0"}]},
         "body.questions"),
        (529, {"detail": {"message": "Overloaded"}}, "Overloaded"),
    ]
    for status, body, needle in cases:
        d = JevDecider(api_key="k", transport=jev_server({}, status=status, body=body))
        try:
            await d.decide("x", [Q("a", "noul", "ok?")])
            check(f"{status} raises", False, "no exception")
        except DecisionError as e:
            check(f"{status} raises with the server's message",
                  e.status == status and needle in str(e), str(e))
            check(f"{status} keeps the request id", e.request_id == "req_test", e.request_id)
        await d.aclose()

    def boom(request):
        raise httpx.ConnectTimeout("timed out")
    d = JevDecider(api_key="k", timeout=0.25, transport=httpx.MockTransport(boom))
    try:
        await d.decide("x", [Q("a", "noul", "ok?")])
        check("timeout raises", False, "no exception")
    except DecisionError as e:
        check("timeout raises a clear error", "timed out" in str(e), str(e))
    await d.aclose()

    import os
    saved = {k: os.environ.pop(k) for k in ("JEV_API_KEY", "TYPESAFE_API_KEY") if k in os.environ}
    try:
        d = JevDecider()
        ok, why = await d.validate_key()
        check("no key is reported, not raised", ok is False and "JEV_API_KEY" in why, why)
        try:
            await d.decide("x", [Q("a", "noul", "ok?")])
            check("decide without a key raises", False, "no exception")
        except DecisionError as e:
            check("decide without a key raises", "JEV_API_KEY" in str(e))
    finally:
        os.environ.update(saved)

    d = JevDecider(api_key="k", transport=jev_server({}, answers={}))
    try:
        await d.decide("x", [Q("a", "noul", "ok?")])
        check("a missing answer raises", False, "no exception")
    except DecisionError as e:
        check("a missing answer raises", "no answer" in str(e), str(e))
    await d.aclose()
    try:
        from presage.llm.jev import build_payload
        build_payload("x", [Q("a", "choice", "pick", options=[])])
        check("a choice with no options raises before sending", False, "no exception")
    except DecisionError as e:
        check("a choice with no options raises before sending", "no options" in str(e))


async def t_haiku_decider() -> None:
    print("\njev: haiku fallback and factory")
    from presage.llm.jev import Decider, HaikuDecider, JevDecider, make_decider

    class FakeLLM:
        def __init__(self) -> None:
            self.calls: list = []

        async def classify(self, prompt: str, schema: dict, model: str | None = None) -> dict:
            self.calls.append({"prompt": prompt, "schema": schema, "model": model})
            return {"is_question": {"answer": True, "confidence": 0.9},
                    "kind": {"answer": "question", "confidence": 0.8},
                    "urgency": {"answer": 0.25, "confidence": 0.6}}

        async def validate_key(self):
            return True, "ok"

    llm = FakeLLM()
    d = HaikuDecider(llm)
    out = await d.decide("Them: what did Q3 look like?", _jev_qs())
    sch = llm.calls[0]["schema"]["properties"]
    check("one call for every question", len(llm.calls) == 1 and len(sch) == 3, len(llm.calls))
    check("noul becomes a boolean field", sch["is_question"]["properties"]["answer"]["type"] == "boolean")
    check("choice becomes an enum", sch["kind"]["properties"]["answer"]["enum"] == ["question", "request", "statement"])
    check("score becomes a bounded number",
          sch["urgency"]["properties"]["answer"]["maximum"] == 1.0, sch["urgency"])
    check("noul answer mapped", out["is_question"].value is True and out["is_question"].p == 0.9, out["is_question"])
    check("choice answer mapped", out["kind"].value == "question" and out["kind"].probs["question"] == 0.8)
    check("score answer clamped into range", out["urgency"].value == 0.25)
    check("latency recorded", all(x.latency_ms >= 0 for x in out.values()))

    class Broken(FakeLLM):
        async def classify(self, prompt, schema, model=None):
            raise RuntimeError("api down")
    try:
        await HaikuDecider(Broken()).decide("x", _jev_qs())
        check("provider failure raises DecisionError", False, "no exception")
    except Exception as e:  # noqa: BLE001
        check("provider failure raises DecisionError", type(e).__name__ == "DecisionError", type(e).__name__)

    import os
    saved = {k: os.environ.pop(k) for k in ("JEV_API_KEY", "TYPESAFE_API_KEY", "PRESAGE_DECIDER")
             if k in os.environ}
    try:
        check("no key -> haiku", isinstance(make_decider(llm), HaikuDecider))
        os.environ["JEV_API_KEY"] = "x"
        check("key -> jev", isinstance(make_decider(llm), JevDecider))
        check("privacy override -> haiku", isinstance(make_decider(llm, prefer="haiku"), HaikuDecider))
        check("both satisfy the Decider protocol",
              isinstance(make_decider(llm), Decider) and isinstance(HaikuDecider(llm), Decider))
    finally:
        os.environ.pop("JEV_API_KEY", None)
        os.environ.update(saved)


def t_jev_live():
    async def run() -> None:
        print("\njev: live endpoint")
        from presage.llm.jev import JevDecider, jev_key
        if not jev_key():
            check("JEV_API_KEY present", False, "not set")
            return
        d = JevDecider()
        ok, why = await d.validate_key()
        check("validate_key", ok, why)
        t0 = time.time()
        out = await d.decide(
            "Them: Thanks for the deck. So on the revenue side, what did Q3 actually look like "
            "versus plan? We also want the enterprise churn story before we get to headcount.",
            _jev_qs())
        print(f"       {int((time.time() - t0) * 1000)} ms, http2={d.http2}")
        for k, v in out.items():
            print(f"       {k}: {v.value!r} p={v.p:.2f} conf={v.confidence:.2f}")
        check("noul answered", isinstance(out["is_question"].value, bool))
        check("choice picked one of the options", out["kind"].value in ("question", "request", "statement"))
        check("score inside lo..hi", 0.0 <= out["urgency"].value <= 1.0, out["urgency"].value)
        check("probabilities came back", out["kind"].probs and sum(out["kind"].probs.values()) > 0.9)
        await d.aclose()
    run.__name__ = "t_jev_live"
    return run


async def main(provider: str = "all", web_live: bool = False, jev_live: bool = False) -> int:
    groups = {
        "gemini": [t_request_shape, t_tool_loop, t_degrade_schema, t_degrade_thinking,
                   t_non_400_propagates, t_classify, t_no_key],
        "claude_code": [t_claude_code_env, t_claude_code_login],
        "anthropic": [t_swap],
        "all": [],
    }
    groups["jev"] = [t_jev_wire, t_jev_errors, t_haiku_decider]
    shared = [t_helpers, t_web_shapes, t_web_urls]
    chosen = (shared + groups["gemini"] + groups["claude_code"] + groups["anthropic"] + groups["jev"]
              if provider == "all" else shared + groups.get(provider, []))
    if web_live:
        chosen.append(t_web_live(provider))
    if jev_live:
        chosen.append(t_jev_live())
    for t in chosen:
        try:
            await t()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILURES.append(f"{t.__name__}: {type(e).__name__}: {e}")
    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} FAILED: {json.dumps(FAILURES)}"))
    return 1 if FAILURES else 0


def t_web_live(provider: str):
    """One real web-search question against a provider that has working credentials."""
    async def run() -> None:
        name = "anthropic" if provider in ("all", "anthropic") else provider
        print(f"\n{name}: live web search")
        llm = make_llm(name)
        ok, why = await llm.validate_key()
        if not ok:
            check(f"{name} credentials usable", False, why)
            return
        rec_out: dict = {}
        data, rec = await llm.answer(
            model=llm.answer_model, system="You answer from the web in at most 25 words.",
            user="What is the current US federal funds rate? Respond with the JSON object only.",
            tools=[], schema=SCHEMA, run_tool=lambda n, a: "{}", web=True)
        rec_out.update(rec)
        print(f"       {rec['total_s']}s first_text={rec['first_text_s']}s -> {str(data.get('answer'))[:90]}")
        print(f"       urls: {[u['url'][:60] for u in rec['web_urls']][:3]}")
        check("web answer came back", bool(data.get("answer")), data)
        check("urls were captured", bool(rec["web_urls"]), rec)
    run.__name__ = "t_web_live"
    return run


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    ap = argparse.ArgumentParser(description="offline checks for the LLM layer")
    ap.add_argument("--provider", default="all",
                    choices=["all", "anthropic", "gemini", "claude_code", "jev"])
    ap.add_argument("--web", action="store_true", help="also run one live web-search question")
    ap.add_argument("--jev", action="store_true", help="also call the live Jev endpoint")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.provider, a.web, a.jev)))
