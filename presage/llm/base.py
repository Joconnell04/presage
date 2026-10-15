"""Provider-neutral LLM interface for the brain.

One `answer()` call runs a whole tool loop and returns the parsed JSON object the brain
renders, plus a record of what it cost. Providers live in `anthropic.py` and `gemini.py`;
`make_llm()` picks one. Nothing here imports a vendor SDK, so `import presage.llm` is cheap
and a missing key surfaces in `validate_key()`, never at import time.
"""
from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

MAX_TOOL_ROUNDS = 8          # web search needs the headroom (server-side pause_turn rounds)
FILE_TOOL_ROUNDS = 3         # files-only answers: the prompt asks for one call, three at most
PARTIAL_EVERY_S = 0.12       # min gap between partial-answer emissions
TIMEOUT_S = 25.0

PROVIDERS = ("anthropic", "gemini", "claude_code", "local")

# the JSON string to hand back, or an awaitable of one when the caller runs that tool in a thread
RunTool = Callable[[str, dict], Any]
# (answer_so_far, model) and, since E52, an optional dict of the other fields that have closed in
# the streamed JSON -- today just {"figure": ...}. Providers that send only the first two still
# work: the brain's callback defaults the third.
OnPartial = Callable[..., Awaitable[None]]


@dataclass
class ToolSpec:
    """A tool the model may call. `input_schema` is an Anthropic-style JSON schema."""
    name: str
    description: str
    input_schema: dict


class DraftDropped(Exception):
    """A speculative draft gave up its place in a provider's queue so a live answer could go
    first (E52). It is a decision, not a failure: the caller expires the draft and does not fall
    back to another provider, because nobody is waiting for the card it would have drawn."""


# ----------------------------------------------------------------------------- partial JSON

_PARTIAL = {}


def partial_field(buf: str, field: str = "answer", closed: bool = False) -> str | None:
    """The value of a string field from a partially generated JSON object, or None.

    The answer schema puts `answer` first on purpose, so this yields words to the screen
    while the rest of the object is still being generated.

    `closed=True` waits for the field's closing quote, so the caller sees the whole value or
    nothing. Half a sentence on the screen reads as typing; half a number reads as a different
    number, so E52 streams `figure` closed and `answer` open.
    """
    key = (field, closed)
    rx = _PARTIAL.get(key)
    if rx is None:
        rx = _PARTIAL[key] = re.compile(
            r'"' + re.escape(field) + r'"\s*:\s*"((?:[^"\\]|\\.)*)' + ('"' if closed else ''))
    m = rx.search(buf)
    if not m:
        return None
    try:
        return json.loads('"' + m.group(1) + '"')
    except json.JSONDecodeError:
        return m.group(1)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_json(text: str) -> dict | None:
    """Parse a model's JSON answer, tolerating code fences and leading prose."""
    if not text:
        return None
    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                data = json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            return data
    return None


_TITLED_URL = re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"url"\s*:\s*"(https?://[^"]+)"')
_BARE_URL = re.compile(r'https?://[^\s"\'<>)\]]+')


def collect_web_urls(obj: Any, out: list | None = None, seen: set | None = None) -> list[dict]:
    """Every {"title", "url"} pair anywhere inside a provider's web-search result.

    The three providers shape their results differently and all three sometimes answer without
    listing a citation, so the brain keeps these as a fallback. Walks blindly rather than
    matching a schema, which is what makes it survive all three.
    """
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if isinstance(obj, dict):
        url = obj.get("url") or obj.get("uri")
        if isinstance(url, str) and url.startswith("http") and url not in seen:
            seen.add(url)
            title = obj.get("title") or obj.get("page_title") or obj.get("name") or ""
            out.append({"title": str(title)[:200], "url": url})
        for v in obj.values():
            collect_web_urls(v, out, seen)
    elif isinstance(obj, str):
        # the Agent SDK hands back web results as one text blob with a `Links: [...]` array
        for m in _TITLED_URL.finditer(obj):
            if m.group(2) not in seen:
                seen.add(m.group(2))
                out.append({"title": m.group(1).encode().decode("unicode_escape", "ignore")[:200],
                            "url": m.group(2)})
        for m in _BARE_URL.finditer(obj):
            url = m.group(0).rstrip(".,")
            if url not in seen:
                seen.add(url)
                out.append({"title": "", "url": url})
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_web_urls(v, out, seen)
    elif hasattr(obj, "model_dump"):          # pydantic blocks from either SDK
        try:
            collect_web_urls(obj.model_dump(), out, seen)
        except Exception:  # noqa: BLE001
            pass
    else:
        for attr in ("url", "uri", "title", "content", "input", "document"):
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if attr in ("url", "uri") and isinstance(value, str) and value.startswith("http"):
                    if value not in seen:
                        seen.add(value)
                        out.append({"title": str(getattr(obj, "title", "") or "")[:200], "url": value})
                elif attr in ("content", "input", "document"):
                    collect_web_urls(value, out, seen)
    return out


def fallback_answer(text: str, question: str) -> dict:
    """What to render when the model returned something that is not our JSON object."""
    return {"question": question, "answer": (text.strip() or "No answer returned")[:200],
            "figure": None, "citations": [], "in_files": False}


# ----------------------------------------------------------------------------- interface

class LLM(abc.ABC):
    """One provider, three roles (answer / hedge / classifier).

    Subclasses must not touch the network in `__init__`: a Brain is constructed before
    anyone has checked whether a key exists, and `make_llm("gemini")` on a machine with no
    Gemini key has to survive until `validate_key()` says why.
    """

    provider: str = "?"

    def __init__(self, api_key: str | None = None, models: dict | None = None) -> None:
        d = self.default_models()
        m = dict(d)
        m.update({k: v for k, v in (models or {}).items() if v is not None and k in d})
        self.api_key = api_key or None
        self.answer_model: str = m["answer"] or d["answer"]
        self.hedge_model: str | None = (m.get("hedge") or None)     # "" disables hedging
        self.classifier_model: str = m["classifier"] or d["classifier"]

    # -- models ---------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def default_models(cls) -> dict:
        """{"answer": ..., "hedge": ..., "classifier": ...} for this provider."""

    def models(self) -> dict:
        return {"answer": self.answer_model, "hedge": self.hedge_model, "classifier": self.classifier_model}

    def state(self) -> dict:
        return {"provider": self.provider, **self.models(), "defaults": self.default_models()}

    # -- calls ----------------------------------------------------------------

    @abc.abstractmethod
    async def answer(self, model: str, system: str, user: str, tools: list[ToolSpec], schema: dict,
                     run_tool: RunTool, on_partial: OnPartial | None = None,
                     web: bool = False, lane: str = "bg") -> tuple[dict, dict]:
        """Run the full tool loop for one answer.

        Returns (data, record): `data` is the parsed JSON object matching `schema`;
        `record` has model, total_s, first_text_s, in, out, cache_read, tool_rounds, stop, and
        `web_urls` when web search ran. `on_partial(text, model)` is awaited with the `answer`
        field as it streams. `web=True` adds the provider's own web search and fetch tools; with
        `web=False` the request is byte-for-byte what it was before web search existed.

        `lane` is how urgent this call is: "live" is a question somebody asked and is waiting on,
        "draft" a speculative guess, "bg" anything else. A provider that queues calls behind a
        shared resource serves "live" first (E52, `claude_code`); the rest ignore it.
        """

    @abc.abstractmethod
    async def classify(self, prompt: str, schema: dict, model: str | None = None) -> dict:
        """One cheap structured JSON call. No tools, no streaming.

        `model` defaults to the classifier model. Passing one explicitly means a different job
        (rolling notes rather than a yes/no verdict), so providers give it more output room and,
        where the model takes it, low effort; the default path is byte-identical to before.
        """

    @abc.abstractmethod
    async def prewarm(self, system: str, tools: list[ToolSpec]) -> None:
        """Ping every model so routing, connections and any prompt cache are hot."""

    @abc.abstractmethod
    async def validate_key(self) -> tuple[bool, str]:
        """One cheap request. (True, "ok") or (False, short human-readable reason)."""

    # -- helpers for subclasses ----------------------------------------------

    @staticmethod
    def record(model: str, provider: str) -> dict:
        return {"provider": provider, "model": model, "total_s": 0.0, "first_text_s": 0.0,
                "in": 0, "out": 0, "cache_read": 0, "cache_write": 0, "thinking": 0,
                "tool_rounds": 0, "stop": None, "web_urls": []}

    @staticmethod
    def tool_args(args: Any) -> dict:
        return dict(args) if isinstance(args, dict) else {}


# ----------------------------------------------------------------------------- factory

def make_llm(provider: str, api_key: str | None = None, models: dict | None = None) -> LLM:
    """Build a provider by name. Import errors and missing keys surface here as ValueError /
    later at validate_key, never as a stray exception from the answer path."""
    name = (provider or "claude_code").strip().lower()   # the authed Claude subscription by default
    if name in ("anthropic", "claude"):
        from presage.llm.anthropic import AnthropicLLM
        return AnthropicLLM(api_key=api_key, models=models)
    if name in ("gemini", "google", "google-genai"):
        from presage.llm.gemini import GeminiLLM
        return GeminiLLM(api_key=api_key, models=models)
    if name in ("claude_code", "claude-code", "subscription", "max"):
        from presage.llm.claude_code import ClaudeCodeLLM
        return ClaudeCodeLLM(api_key=api_key, models=models)
    if name in ("local", "mlx", "local_mlx", "private", "offline"):
        from presage.llm.local_mlx import LocalMLXLLM     # a quantised model on this Mac, no network
        return LocalMLXLLM(api_key=api_key, models=models)
    raise ValueError(f"unknown LLM provider {provider!r}; expected one of {', '.join(PROVIDERS)}")
