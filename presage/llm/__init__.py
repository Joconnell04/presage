"""LLM providers for the brain.

`make_llm(provider)` returns an `LLM`: one `answer()` call per answer (tool loop, streaming
partials, structured JSON), plus `classify()`, `prewarm()` and `validate_key()`. Providers are
imported on demand, so this package costs nothing to import and a provider whose SDK or key is
missing fails where it can be reported, not at import.
"""
from presage.llm.base import (MAX_TOOL_ROUNDS, PARTIAL_EVERY_S, PROVIDERS, LLM, ToolSpec,
                              fallback_answer, make_llm, parse_json, partial_field)

__all__ = ["LLM", "ToolSpec", "make_llm", "partial_field", "parse_json", "fallback_answer",
           "PROVIDERS", "MAX_TOOL_ROUNDS", "PARTIAL_EVERY_S"]
