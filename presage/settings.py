"""Everything the page can change, persisted so a restart keeps it.

`settings.json` at the repo root (gitignored) holds the choices:

    {"context_dir": null,            # null = <repo>/context, else an absolute path
     "mic_uid": null,                # null = system default input
     "llm_provider": "claude_code",   # the authed Claude subscription; "anthropic" needs an API key with credit    # "anthropic" | "gemini"
     "models": null,                 # null = the provider's defaults
     "tune": {...the five Tune keys...},
     "auto_start_audio": true}

API keys never go in there: they live in `.env` next to it, one `NAME=value` per
line, mode 600. `set_key()` rewrites only its own line and leaves the rest of the
file alone, so a hand-edited .env survives a key change from the page.

Precedence: settings.json wins over the environment; the environment is the
default when a key is null, so `PRESAGE_CONTEXT=... ./run.sh` still works.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("presage.settings")

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "settings.json"
ENV_PATH = ROOT / ".env"
DEFAULT_CONTEXT = ROOT / "context"

PROVIDERS = ("anthropic", "gemini", "claude_code")
# claude_code has no key: it bills the Claude subscription the `claude` CLI is logged into.
ENV_VAR = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
           # not an answer provider: the typed-decision gate (docs/research-next.md, bet 6)
           "jev": "JEV_API_KEY"}

# The Tune popover: five sliders and one switch. brain owns the ranges; this owns the disk.
TUNE_KEYS = ("topic_threshold", "question_threshold", "prefetch_cooldown_s", "max_warm",
             "answer_ttl_s", "web_mode", "mode")
WEB_MODES = ("auto", "off")
LIVE_CHOICES = ("auto", "same", "anthropic", "claude_code", "gemini")
DECIDER_CHOICES = ("auto", "jev", "laya", "haiku")   # the question gate: hosted Jev, local Laya, Haiku
# The two typed-intent questions (which lever a spoken what-if means, whether a warm card answers
# what landed) are a separate choice from the gate, because since E21a they have a different best
# answer: the local multi checkpoint beats Jev on both. "auto" is local when its weights are in
# ~/.cache/presage/laya, else Jev when a key is set, else off.
INTENT_CHOICES = ("auto", "local", "jev", "off")
DEFAULT_MODE = "call"
MODEL_KEYS = ("answer", "hedge", "classifier")

# Used when presage.llm is not importable or does not publish its own defaults.
FALLBACK_MODELS = {
    "anthropic": {"answer": "claude-sonnet-5",
                  "hedge": "claude-haiku-4-5-20251001",
                  "classifier": "claude-haiku-4-5-20251001"},
    "gemini": {"answer": "gemini-2.5-flash",
               "hedge": "gemini-2.5-flash-lite",
               "classifier": "gemini-2.5-flash-lite"},
    "claude_code": {"answer": "claude-sonnet-5",
                    "hedge": "claude-haiku-4-5-20251001",
                    "classifier": "claude-haiku-4-5-20251001"},
}

DEFAULTS: dict[str, Any] = {
    "context_dir": None,
    "mic_uid": None,
    "llm_provider": "claude_code",
    "models": None,
    "tune": {},
    "auto_start_audio": True,
    "deep_index": True,          # the background pass that builds the Q&A cache
    "live_choice": "same",       # which provider answers live: same (the main provider, the Claude subscription by default) | auto | a provider name
    "decider": "auto",           # question gate: auto (Jev with a key, else Laya if downloaded, else Haiku) | jev | laya | haiku
    "intent": "auto",            # the lever pick and the warm-card check: auto (local when the E21 weights are here, else Jev with a key, else off) | local | jev | off
}

_cache: dict[str, Any] | None = None


# ----------------------------------------------------------------------------- file

def load(force: bool = False) -> dict[str, Any]:
    """The settings file merged over the defaults. Cached; a bad file is ignored."""
    global _cache
    if _cache is not None and not force:
        return _cache
    data = dict(DEFAULTS)
    data["tune"] = {}
    try:
        raw = json.loads(SETTINGS_PATH.read_text())
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k == "tune" and isinstance(v, dict):
                    data["tune"] = {str(kk): vv for kk, vv in v.items()}
                else:
                    data[k] = v
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001  a corrupt file must not stop the server
        log.warning("settings.json unreadable, using defaults: %s", e)
    if data.get("llm_provider") not in PROVIDERS:
        data["llm_provider"] = os.environ.get("PRESAGE_LLM_PROVIDER", "claude_code")
        if data["llm_provider"] not in PROVIDERS:
            data["llm_provider"] = "anthropic"
    _cache = data
    return data


def save(data: dict[str, Any]) -> dict[str, Any]:
    """Write the whole blob atomically and return what was written."""
    global _cache
    out = {k: data.get(k, v) for k, v in DEFAULTS.items()}
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(SETTINGS_PATH.parent), prefix=".settings-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, SETTINGS_PATH)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    _cache = out
    return out


def update(**patch: Any) -> dict[str, Any]:
    """Patch a few keys and save. `tune=` merges, everything else replaces."""
    data = dict(load())
    for k, v in patch.items():
        if k == "tune" and isinstance(v, dict):
            merged = dict(data.get("tune") or {})
            merged.update(v)
            data["tune"] = merged
        else:
            data[k] = v
    return save(data)


# ----------------------------------------------------------------------------- context folder

def context_dir() -> Path:
    """settings.json, else PRESAGE_CONTEXT, else <repo>/context."""
    chosen = load().get("context_dir")
    if chosen:
        return Path(str(chosen)).expanduser().resolve()
    env = os.environ.get("PRESAGE_CONTEXT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CONTEXT.resolve()


def is_own_context(path: Path | str | None = None) -> bool:
    """True when the folder is the one presage ships, so it is ours to add to and remove from."""
    p = Path(path).expanduser().resolve() if path else context_dir()
    return p == DEFAULT_CONTEXT.resolve()


# ----------------------------------------------------------------------------- models

_llm_mod: Any = ...          # not looked up yet


def _llm_module():
    """presage.llm once it exists, remembered either way so this is not a hot import."""
    global _llm_mod
    if _llm_mod is ...:
        try:
            from presage import llm as mod  # noqa: PLC0415  optional, owned by another module
            _llm_mod = mod
        except Exception:  # noqa: BLE001
            _llm_mod = None
    return _llm_mod


def provider_defaults(provider: str) -> dict[str, str]:
    """The provider layer's defaults when it is importable, otherwise ours."""
    provider = provider if provider in PROVIDERS else "anthropic"
    try:                        # the provider class publishes its own, no key needed
        import importlib
        mod = importlib.import_module(f"presage.llm.{provider}")
        for obj in vars(mod).values():
            if isinstance(obj, type) and getattr(obj, "provider", None) == provider:
                fn = getattr(obj, "default_models", None)
                got = fn() if callable(fn) else None
                if isinstance(got, dict) and got:
                    return {k: str(v) for k, v in got.items() if k in MODEL_KEYS}
    except Exception:  # noqa: BLE001  presage.llm is optional and owns its own shape
        pass
    _llm = _llm_module()
    if _llm is not None:
        try:
            for attr in ("provider_defaults", "defaults", "default_models"):
                fn = getattr(_llm, attr, None)
                if callable(fn):
                    got = fn(provider)
                    if isinstance(got, dict) and got:
                        return {k: str(v) for k, v in got.items() if k in MODEL_KEYS}
            table = getattr(_llm, "DEFAULT_MODELS", None)
            if isinstance(table, dict) and isinstance(table.get(provider), dict):
                return {k: str(v) for k, v in table[provider].items() if k in MODEL_KEYS}
        except Exception:  # noqa: BLE001  presage.llm owns its own shape
            pass
    if provider in ("anthropic", "claude_code"):
        env = {"answer": os.environ.get("PRESAGE_ANSWER_MODEL"),
               "hedge": os.environ.get("PRESAGE_HEDGE_MODEL"),
               "classifier": os.environ.get("PRESAGE_CLASSIFIER_MODEL")}
        out = dict(FALLBACK_MODELS[provider])
        out.update({k: v for k, v in env.items() if v is not None})
        return out
    return dict(FALLBACK_MODELS[provider])


def models_for(provider: str) -> dict[str, str]:
    """Saved models for this provider, filled in from the defaults."""
    out = dict(provider_defaults(provider))
    saved = load().get("models")
    if isinstance(saved, dict):
        for k in MODEL_KEYS:
            v = saved.get(k)
            if v is not None:
                out[k] = str(v)
    return out


def llm_provider() -> str:
    p = str(load().get("llm_provider") or "anthropic").lower()
    return p if p in PROVIDERS else "anthropic"


# ----------------------------------------------------------------------------- api keys (.env)

def _env_lines() -> list[str]:
    try:
        return ENV_PATH.read_text().splitlines()
    except FileNotFoundError:
        return []
    except Exception as e:  # noqa: BLE001
        log.warning(".env unreadable: %s", e)
        return []


def set_key(provider: str, key: str | None) -> None:
    """Rewrite one NAME=value line in .env and update os.environ. Never logged."""
    if provider not in ENV_VAR:
        raise ValueError(f"unknown provider {provider!r}")
    var = ENV_VAR[provider]
    key = (key or "").strip()
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(var)}\s*=")
    out: list[str] = []
    written = False
    for line in _env_lines():
        if pat.match(line):
            if key and not written:
                out.append(f"{var}={key}")
                written = True
            continue          # drop the old line (and any duplicates)
        out.append(line)
    if key and not written:
        out.append(f"{var}={key}")
    text = "\n".join(out).strip("\n")
    ENV_PATH.write_text(text + "\n" if text else "")
    try:
        os.chmod(ENV_PATH, 0o600)
    except OSError as e:
        log.warning("could not chmod .env: %s", e)
    if key:
        os.environ[var] = key
    else:
        os.environ.pop(var, None)


def api_key(provider: str) -> str | None:
    """The live key: environment first (that is what the SDKs read), then .env."""
    var = ENV_VAR.get(provider)
    if not var:
        return None
    val = os.environ.get(var)
    if val:
        return val.strip()
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(var)}\s*=\s*(.*)$")
    for line in _env_lines():
        m = pat.match(line)
        if m:
            return m.group(1).strip().strip("'\"") or None
    return None


def key_status(provider: str) -> dict[str, Any]:
    """{"set": bool, "hint": "…AB12"} — enough to show, never enough to use."""
    key = api_key(provider)
    if not key:
        return {"set": False, "hint": ""}
    return {"set": True, "hint": "…" + key[-4:]}
