"""presage server: audio -> STT -> brain -> WebSocket UI, plus the control API.

Run:  uv run python -m presage.server            (serves http://0.0.0.0:8766)
Env:  PRESAGE_CONTEXT=./context   default context folder (settings.json wins)
      STT_PROVIDER=local|deepgram
      PRESAGE_REPLAY=file.wav     feed a 16 kHz mono wav as the "system" (Them) source
      PRESAGE_NO_AUDIO=1          UI + manual ask only, never opens the mic
      PRESAGE_OPEN_BROWSER=1      open the page once the server is up
      PRESAGE_PORT=8766

Everything else (context folder, mic, provider, API key, models) is set from the
page and persisted in settings.json / .env by presage.settings; see docs/ui-protocol.md.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from presage import digest  # noqa: E402
from presage import feedback  # noqa: E402
from presage import screen as screenmod  # noqa: E402
from presage import settings as prefs  # noqa: E402
from presage.audio_session import AudioSession  # noqa: E402
from presage.brain import Brain, Utterance, new_id  # noqa: E402
from presage.ingest import Index  # noqa: E402

log = logging.getLogger("presage")
logging.basicConfig(level=os.environ.get("PRESAGE_LOG", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

PORT = int(os.environ.get("PRESAGE_PORT", "8766"))
UI_FILE = ROOT / "ui" / "index.html"
BLUETOOTH_WARNING = "Bluetooth mics can drop out; prefer the built-in mic"
CLAUDE_CODE_HINT = "Run `claude` once in Terminal and log in with your Claude account"

# What each mode does, in the words the page shows as the empty state of the answers column.
# brain.modes() wins when it exists; this is the fallback so the page never hardcodes copy.
MODES = [
    {"id": "call", "label": "Call", "hint": "Answers from your files appear here."},
    {"id": "sales", "label": "Sales", "hint": "Objection handling and talk tracks appear here."},
    {"id": "interview", "label": "Interview", "hint": "Suggested answers to their questions appear here."},
    {"id": "pitch", "label": "Pitch", "hint": "Answers and tough-question responses appear here."},
    {"id": "notes", "label": "Notes", "hint": "Notes build as the call goes on."},
    {"id": "quiet", "label": "Quiet", "hint": "Nothing automatic. Ask a question below."},
]
LOGIN_TTL_S = 60.0          # the claude_code login probe can take a second
DEEP_DEBOUNCE_S = 3.0       # a burst of file changes starts one deep pass, not five
DEEP_PUSH_EVERY_S = 0.5     # at most two progress broadcasts a second

app = FastAPI()


class Hub:
    """Shared state + fan-out to browser clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.transcript: deque[dict] = deque(maxlen=80)
        self.status: dict = {"mic": False, "system": False, "stt": os.environ.get("STT_PROVIDER", "local"),
                             "index": {}, "listening": False, "levels": {"mic": 0.0, "system": 0.0}}
        self.levels: dict[str, float] = {"mic": 0.0, "system": 0.0}
        self.index: Index | None = None
        self.brain: Brain | None = None
        self.audio: AudioSession | None = None
        self.observer = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.devices: list[dict] = []
        self.files_count: int = 0
        self.key_ok: bool | None = None
        self.key_msg: str = ""
        self.provider_error: str = ""       # set when the chosen provider did not load
        self.notes: dict | None = None      # last notes frame, replayed in hello
        self.scratch: list[dict] = []       # last scratch frame's items, replayed in hello
        self.deep_task: asyncio.Task | None = None   # background deep index pass
        self.deep_push_at: float = 0.0
        self.decider: dict | None = None    # typed-decision gate state, cached
        self.decider_at: float = 0.0
        self._decider_job = None
        self.login: dict | None = None      # claude_code login state, cached
        self.login_at: float = 0.0
        self._login_job = None
        self.open_utterance: dict[str, str] = {}   # source -> transcript id of the utterance in progress
        self.recent_system: list = []      # (ts, words) of the far side's recent lines, for the echo check
        self.echo_dropped = 0
        self.echo_ids: dict = {}
        self._save_tune: asyncio.Task | None = None

    async def broadcast(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "transcript":
            self._remember_transcript(msg)
        elif kind == "notes":
            self.notes = msg                  # the page rebuilds from hello, so keep the last one
        elif kind == "scratch":
            self.scratch = list(msg.get("items") or [])
        elif kind == "cleared":
            self.notes = None
            self.scratch = []
        elif kind == "answer":
            log_card(msg)                     # E12: every card that reaches a page is logged
        digest.recorder().observe(msg)        # E44: the whole call, kept for the post-call digest
        dead = []
        data = json.dumps(msg, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def push(self, msg: dict) -> None:
        """Broadcast from a sync callback."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)

    def _remember_transcript(self, msg: dict) -> None:
        for i, old in enumerate(self.transcript):
            if old["id"] == msg["id"]:
                self.transcript[i] = msg
                return
        self.transcript.append(msg)

    def refresh_status(self) -> dict:
        audio = self.audio.state() if self.audio else {"state": "stopped", "mic_state": "unknown",
                                                       "mic_device": None, "mic_uid": None,
                                                       "output_device": None, "error": None}
        self.status["audio"] = audio
        self.status["mic"] = bool(audio.get("mic"))
        self.status["system"] = bool(audio.get("system"))
        self.status["listening"] = audio.get("state") == "running"
        self.status["me_speaking"] = bool(audio.get("me_speaking"))
        self.status["stt"] = audio.get("stt") or self.status.get("stt")
        self.status["llm"] = llm_block()
        self.status["context"] = context_block(full=False)
        return self.status

    def hello(self) -> dict:
        state = self.brain.state() if self.brain else {"settings": {}, "answers": [], "candidates": []}
        self.refresh_status()
        return {"type": "hello", "settings": settings_view(), "status": self.status,
                "transcript": list(self.transcript), "answers": state["answers"],
                "candidates": state["candidates"], "models": state.get("models", {}),
                "audio": self.status["audio"], "llm": self.status["llm"],
                "context": context_block(full=True), "devices": self.devices,
                "modes": modes_list(), "notes": state.get("notes") or self.notes,
                "scratch": state.get("scratch") or self.scratch,
                "screen": state.get("screen") or {},      # E43a; GET /api/screen is the live one
                "questions": state.get("questions") or [],  # E68: registered questions, newest first
                "prefs": prefs.load(), "now": time.time()}


hub = Hub()


# ----------------------------------------------------------------------------- blocks

def index_summary(index: Index | None) -> dict:
    if index is None:
        return {}
    s = index.stats()
    out = {"files": s.files, "spreadsheets": s.spreadsheets, "documents": s.documents, "tables": s.tables,
           "chunks": s.chunks, "topics": s.topics, "errors": len(s.errors or {})}
    deep = deep_stats(index)
    if deep is not None:
        out["deep"] = deep
    return out


def deep_stats(index: Index | None = None) -> dict | None:
    """index.deep_stats() once the deep pass exists, plus whether it is switched on."""
    index = index if index is not None else hub.index
    fn = getattr(index, "deep_stats", None)
    enabled = bool(prefs.load().get("deep_index", True))
    if not callable(fn):
        return {"state": "off", "enabled": enabled} if not enabled else None
    try:
        got = fn()
    except Exception as e:  # noqa: BLE001
        return {"state": "error", "error": str(e)[:120], "enabled": enabled}
    if not isinstance(got, dict):
        return {"enabled": enabled}
    out = dict(got)
    out["enabled"] = enabled
    if not enabled and out.get("state") in (None, "idle", "", "off"):
        out["state"] = "off"
    return out


def context_files(folder: Path) -> list[dict]:
    """Everything in the folder, joined with what the index made of it."""
    rows: dict[str, dict] = {}
    con = getattr(hub.index, "con", None)
    if con is not None and getattr(hub.index, "folder", None) == folder:
        try:
            for r in con.execute("SELECT path, size, kind, n_tables, n_chunks, error FROM _files"):
                rows[r["path"]] = dict(r)
        except Exception as e:  # noqa: BLE001
            log.warning("file list: %s", e)
    errors = dict(getattr(hub.index.stats(), "errors", {}) or {}) if hub.index else {}

    out: list[dict] = []
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except OSError as e:
        log.warning("context folder unreadable: %s", e)
        return out
    for p in entries:
        if not p.is_file() or p.name.startswith((".", "~$")):
            continue
        row = rows.get(p.name, {})
        try:
            st = p.stat()
            size, mtime = st.st_size, st.st_mtime
        except OSError:
            size, mtime = 0, 0.0
        err = row.get("error") or errors.get(p.name)
        out.append({"name": p.name,
                    "kind": (row.get("kind") or p.suffix.lstrip(".")).lower(),
                    "size": size, "mtime": mtime,
                    "indexed": bool(row) and not err,
                    "error": err,
                    "tables": int(row.get("n_tables") or 0),
                    "chunks": int(row.get("n_chunks") or 0)})
    return out


def context_block(full: bool = True) -> dict:
    """full=True walks the folder; the compact form reuses that count (status ticks at 1 Hz)."""
    folder = prefs.context_dir()
    block = {"dir": str(folder), "own_dir": prefs.is_own_context(folder)}
    if full:
        files = context_files(folder)
        block["files"] = files
        hub.files_count = len(files)
    block["files_count"] = hub.files_count
    return block


def context_msg() -> dict:
    return {"type": "context", **context_block(full=True)}


def llm_block() -> dict:
    """What the page needs to show the Model section, whatever shape brain.llm has."""
    llm = getattr(hub.brain, "llm", None)
    live: dict = {}
    if llm is not None:
        for attr in ("state", "models"):
            fn = getattr(llm, attr, None)
            if callable(fn):
                try:
                    got = fn()
                except Exception:  # noqa: BLE001
                    got = None
                if isinstance(got, dict) and got:
                    live = got
                    break
    saved = prefs.llm_provider()
    live_provider = str(live.get("provider") or getattr(llm, "provider", "") or "").lower()
    # normally the same thing; if the chosen one did not load, show what was chosen and say why
    provider = saved if hub.provider_error else (live_provider or saved)
    if provider not in prefs.PROVIDERS:
        provider = "anthropic"

    models = prefs.models_for(provider)
    for k in prefs.MODEL_KEYS:
        if live.get(k) is not None:
            models[k] = str(live[k])
    answer = getattr(llm, "answer_model", None)
    if answer:
        models["answer"] = str(answer)
    if llm is None and hub.brain is not None:          # brain still owns the models itself
        for k, v in (hub.brain.state().get("models") or {}).items():
            if k in prefs.MODEL_KEYS and v:
                models[k] = str(v)

    defaults = live.get("defaults") if isinstance(live.get("defaults"), dict) else prefs.provider_defaults(provider)
    ks = prefs.key_status(provider)
    block = {"provider": provider, "answer_model": models.get("answer"), "models": models,
             "defaults": {k: str(v) for k, v in defaults.items() if k in prefs.MODEL_KEYS},
             "key_set": ks["set"], "key_hint": ks["hint"],
             "key_ok": hub.key_ok, "key_message": hub.key_msg}
    live = hub.brain.state() if callable(getattr(hub.brain, "state", None)) else {}
    block["live_choice"] = str(live.get("live_choice") or prefs.load().get("live_choice", "auto"))
    block["live_provider"] = str(live.get("live_provider") or provider)
    block["decider"] = decider_block()
    if hub.provider_error:
        block["error"] = hub.provider_error
    if provider == "claude_code":                  # no key: the CLI's own login is the credential
        login = login_snapshot()
        block.update(login=login, key_set=False, key_hint="", hint=CLAUDE_CODE_HINT,
                     key_ok=login["ok"], key_message=login["message"])
    return block


def decider_block() -> dict:
    """The typed-decision gate (docs/research-next.md, bet 6): Jev when a key is set and the
    gate is wired, Haiku otherwise. Cached like the login probe because validating is a call."""
    dec = getattr(hub.brain, "decider", None)
    ks = prefs.key_status("jev")
    if dec is None:
        if ks["set"]:
            return {"kind": "haiku", "ok": True, "key_set": True, "key_hint": ks["hint"],
                    "message": "key saved, using Haiku until the gate lands", "latency_ms": None}
        return {"kind": "haiku", "ok": True, "key_set": False, "key_hint": "",
                "message": "no key, using Haiku", "latency_ms": None}
    kind = str(getattr(dec, "kind", None) or getattr(dec, "provider", None)
               or ("jev" if ks["set"] else "haiku"))
    probe = decider_snapshot()
    try:
        from presage.llm import laya as _laya
        laya_ok = bool(_laya.available())
    except Exception:  # noqa: BLE001
        laya_ok = False
    return {"kind": str(probe.get("kind") or kind), "ok": bool(probe.get("ok")),
            "key_set": ks["set"], "key_hint": ks["hint"],
            "message": str(probe.get("message") or ""),
            "latency_ms": probe.get("latency_ms"),
            "choice": str(getattr(hub.brain, "decider_choice", None) or prefs.load().get("decider", "auto")),
            "laya_available": laya_ok, "intent": intent_block()}


def intent_block() -> dict:
    """The lever pick and the warm-card check (E21a), which are their own choice now: the local
    E21 checkpoint beats Jev on both, and the gate's checkpoint answers neither. No probe: the
    only thing that can be wrong is that the weights are not on this machine, and `why_not` says
    so in one line."""
    out = {"kind": str(getattr(hub.brain, "intent_kind", None) or "off"),
           "choice": str(getattr(hub.brain, "intent_choice", None) or prefs.load().get("intent", "auto")),
           "local_available": False, "dir": None, "why_not": None}
    try:
        from presage.llm import intent_local
        out.update(local_available=intent_local.available(), why_not=intent_local.why_not())
        d = intent_local.weights_dir()
        out["dir"] = d.name if d else None
    except Exception as e:  # noqa: BLE001  the local checkpoint is optional
        out["why_not"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def rebuild_decider() -> None:
    """Pick the gate up again after a key change: JevDecider reads JEV_API_KEY when constructed."""
    if hub.brain is None:
        return
    fn = getattr(hub.brain, "set_decider", None)
    if callable(fn):
        try:
            fn(prefs.load().get("decider", "auto"))
            return
        except Exception as e:  # noqa: BLE001
            log.warning("set_decider(): %s", e)
    try:
        from presage.llm.jev import make_decider
        hub.brain.decider = make_decider(getattr(hub.brain, "llm", None))
    except Exception as e:  # noqa: BLE001  the gate is optional
        log.warning("decider rebuild: %s", e)


def decider_snapshot() -> dict:
    """Cached decider check; a stale one schedules a refresh and returns what it has."""
    fresh = hub.decider is not None and time.time() - hub.decider_at < LOGIN_TTL_S
    if not fresh and hub.loop is not None and (hub._decider_job is None or hub._decider_job.done()):
        hub._decider_job = asyncio.run_coroutine_threadsafe(refresh_decider(), hub.loop)
    return hub.decider or {"kind": None, "ok": False, "message": "Checking…", "latency_ms": None}


async def refresh_decider() -> dict:
    dec = getattr(hub.brain, "decider", None)
    fn = getattr(dec, "validate_key", None)
    out = {"kind": str(getattr(dec, "kind", "") or getattr(dec, "provider", "") or ""),
           "ok": False, "message": "", "latency_ms": None}
    if callable(fn):
        t0 = time.time()
        try:
            res = fn()
            got = (await res) if inspect.isawaitable(res) else res
        except Exception as e:  # noqa: BLE001
            got = (False, short_error(e))
        if isinstance(got, dict):
            out.update({k: got[k] for k in ("kind", "ok", "message", "latency_ms") if k in got})
        elif isinstance(got, (tuple, list)) and len(got) == 2:
            out["ok"], out["message"] = bool(got[0]), str(got[1] or "")
        else:
            out["ok"] = bool(got)
        # the gate reports "ok (203 ms)"; keep one number, in latency_ms, so the page
        # renders "Jev · ok (203 ms)" instead of printing the milliseconds twice
        took = re.search(r"\s*\((\d+)\s*ms\)\s*$", out["message"] or "")
        if took:
            out["message"] = out["message"][:took.start()].strip()
            if out.get("latency_ms") is None:
                out["latency_ms"] = int(took.group(1))
        if out.get("latency_ms") is None and out["ok"]:
            out["latency_ms"] = int((time.time() - t0) * 1000)
        if not out.get("kind"):
            out["kind"] = "jev" if out["ok"] else "haiku"
        if not out["message"]:
            out["message"] = "ok" if out["ok"] else "no key, using Haiku"
    hub.decider, hub.decider_at = out, time.time()
    return out


def settings_view() -> dict:
    """brain's settings, plus any Tune key brain does not own yet (web_mode, mode)."""
    out = dict(hub.brain.settings) if hub.brain else {}
    for k, v in (prefs.load().get("tune") or {}).items():
        if k in prefs.TUNE_KEYS:
            out.setdefault(k, v)
    out.setdefault("web_mode", "auto")
    out.setdefault("mode", prefs.DEFAULT_MODE)
    return out


def modes_list() -> list[dict]:
    """brain.modes() when it is there, otherwise MODES. Always {id,label,hint}."""
    fn = getattr(hub.brain, "modes", None)
    got = None
    if callable(fn):
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001
            log.warning("brain.modes(): %s", e)
    elif isinstance(fn, (list, tuple)):
        got = fn
    if not got:
        return [dict(m) for m in MODES]
    out = []
    for m in got:
        if isinstance(m, dict):
            mid = str(m.get("id") or m.get("mode") or "").strip()
            if not mid:
                continue
            out.append({"id": mid, "label": str(m.get("label") or mid.title()),
                        "hint": str(m.get("hint") or "")})
        elif isinstance(m, (list, tuple)) and m:
            mid = str(m[0])
            out.append({"id": mid, "label": str(m[1]) if len(m) > 1 else mid.title(),
                        "hint": str(m[2]) if len(m) > 2 else ""})
        elif isinstance(m, str):
            out.append({"id": m, "label": m.title(), "hint": ""})
    return out or [dict(m) for m in MODES]


# ------------------------------------------------------------------ claude_code login

def login_snapshot() -> dict:
    """The cached login, refreshed in the background once it is more than a minute old."""
    fresh = hub.login is not None and time.time() - hub.login_at < LOGIN_TTL_S
    if not fresh and hub.loop is not None and (hub._login_job is None or hub._login_job.done()):
        hub._login_job = asyncio.run_coroutine_threadsafe(refresh_login(), hub.loop)
    return hub.login or {"ok": False, "account": None, "message": "Checking…"}


async def refresh_login() -> dict:
    hub.login = await claude_code_login()
    hub.login_at = time.time()
    return hub.login


async def claude_code_login() -> dict:
    """brain.llm.validate_key() when the provider layer has claude_code, else probe the CLI."""
    llm = getattr(hub.brain, "llm", None)
    if str(getattr(llm, "provider", "")) == "claude_code":
        fn = getattr(llm, "validate_key", None)
        if callable(fn):
            try:
                res = fn()
                return _login_from((await res) if inspect.isawaitable(res) else res, llm)
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "account": None, "message": short_error(e)}
    return await asyncio.to_thread(_cli_login)


def _login_from(got, llm) -> dict:
    if isinstance(got, dict):
        return {"ok": bool(got.get("ok")), "account": got.get("account") or _claude_account(),
                "message": str(got.get("message") or "")}
    ok, msg = got if isinstance(got, (tuple, list)) and len(got) == 2 else (bool(got), "")
    account = getattr(llm, "account", None) or _claude_account()
    return {"ok": bool(ok), "account": account,
            "message": str(msg or (f"Signed in as {account}" if ok and account else
                                   "Signed in" if ok else "Not logged in"))}


def _claude_account() -> str | None:
    """The email the CLI is logged in as. Reads no tokens."""
    try:
        data = json.loads((Path.home() / ".claude.json").read_text())
        acct = data.get("oauthAccount") or {}
        return acct.get("emailAddress") or acct.get("displayName") or None
    except Exception:  # noqa: BLE001
        return None


def _cli_login() -> dict:
    """Fallback probe, used until presage.llm.claude_code publishes validate_key()."""
    exe = shutil.which("claude")
    if not exe:
        return {"ok": False, "account": None, "message": "claude CLI not installed"}
    account = _claude_account()
    signed_in = f"Signed in as {account}" if account else "Signed in"
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")}   # a key must not mask the login
    try:
        done = subprocess.run([exe, "auth", "status", "--json"], capture_output=True, text=True,
                              timeout=20, env=env)
        if json.loads(done.stdout.strip() or "{}").get("loggedIn"):
            return {"ok": True, "account": account, "message": signed_in}
    except Exception as e:  # noqa: BLE001
        log.debug("claude auth status: %s", e)
    if _has_oauth_credential():
        return {"ok": True, "account": account, "message": signed_in}
    if account:
        return {"ok": False, "account": account, "message": "Claude Code session expired"}
    return {"ok": False, "account": None, "message": "Not logged in"}


def _has_oauth_credential() -> bool:
    """Is there a Claude subscription credential to refresh from? Existence only, never the value."""
    try:
        creds = json.loads((Path.home() / ".claude" / ".credentials.json").read_text())
        if (creds.get("claudeAiOauth") or {}).get("refreshToken"):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:                       # on macOS the live credential usually lives in the keychain
        done = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials"],
                              capture_output=True, text=True, timeout=5)
        return done.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def audio_block() -> dict:
    return hub.audio.state() if hub.audio else {"state": "stopped", "mic_state": "unknown",
                                                "mic_device": None, "mic_uid": None,
                                                "output_device": None, "error": None}


async def push_status() -> None:
    hub.refresh_status()
    await hub.broadcast({"type": "status", **hub.status, "now": time.time()})


def on_audio_change() -> None:
    """AudioSession calls this whenever its state moves."""
    if hub.loop is None:
        return
    asyncio.run_coroutine_threadsafe(push_status(), hub.loop)


# ----------------------------------------------------------------------------- http

@app.get("/")
async def ui() -> FileResponse:
    return FileResponse(UI_FILE)


@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse(hub.hello())


@app.post("/api/ask")
async def api_ask(request: Request) -> JSONResponse:
    body = await _json(request)
    if hub.brain:
        await hub.brain.ask(str(body.get("text", "")))
    return JSONResponse({"ok": True})


# ------------------------------------------------------------------ screen (E43)

@app.get("/api/screen")
async def api_screen() -> JSONResponse:
    """What the presenter has on screen: which slide of which deck, which sheet of which
    workbook, and the indexed file it maps to. See docs/ui-protocol.md.

    A read of the poller's last tick, so it never blocks and never raises: with no poller
    running it answers `status: "unavailable", reason: "not_started"`, which is the same
    shape as a locked screen or a missing grant. Nothing else in the API changes.
    """
    return JSONResponse(screen_block())


@app.post("/api/drift")
async def api_drift(request: Request) -> JSONResponse:
    """Every figure on the page on screen that no longer matches the cell it came from (E63).

    Takes nothing and checks whatever is up; `{"file": ..., "loc": "slide 4"}` checks a named
    page instead, which is how the bench and a page with its own picker ask. It is the same
    check the page change runs, so it answers in a few milliseconds with no model call, but it
    is not the same decision: this reports the whole list and raises no card, so neither the
    once-a-session rule nor the per-cell cooldown applies.
    """
    body = await _json(request)
    if hub.index is None:
        return _err("no folder is open")
    try:
        from presage import drift as driftmod
    except Exception as e:  # noqa: BLE001
        return _err(f"drift unavailable: {type(e).__name__}: {e}")
    state = {"file": body.get("file"), "loc": body.get("loc")}
    if not state["file"]:
        state = screenmod.snapshot()
        if state.get("status") != "ok" or not state.get("file"):
            return JSONResponse({"screen": state, "checked": False, "drifts": [],
                                 "reason": state.get("reason") or "no_document"})
    try:
        rep = await asyncio.to_thread(driftmod.check, hub.index, state)
    except Exception as e:  # noqa: BLE001
        log.warning("drift check failed: %s", e)
        return _err(f"drift failed: {type(e).__name__}: {str(e)[:200]}")
    out = rep.as_dict()
    out["checked"] = True
    out["seen"] = bool(hub.brain and (out["file"], out["head"]) in hub.brain._drift_seen)  # noqa: SLF001
    return JSONResponse(out)


def screen_block() -> dict:
    w = screenmod.watcher()
    out = screenmod.snapshot()
    out["running"] = bool(w and w._thread and w._thread.is_alive())    # noqa: SLF001
    out["interval_s"] = w.interval if w else screenmod.INTERVAL_S
    return out


def start_screen() -> None:
    """Start the 1 Hz poller unless PRESAGE_SCREEN=0. It is a daemon thread of its own and
    costs 0.2-0.6% of a core; the transcript thread and the event loop never touch it.

    It runs with the audio session (E43a), not with the process: nobody is presenting to a copilot
    that is not listening, and a poller that ticks through a whole idle afternoon is reading window
    titles for nothing. Starting it here also hands the brain its reader, so `Brain._evidence` can
    pack the page on screen first.
    """
    if os.environ.get("PRESAGE_SCREEN", "1") in ("0", "off", "false", "no"):
        log.info("screen poller off (PRESAGE_SCREEN=0)")
        return
    try:
        w = screenmod.start(hub.index)
        w.on_change = on_screen_change          # E63: a new page gets checked against the model
    except Exception as e:  # noqa: BLE001
        log.warning("screen poller: %s", e)


def on_screen_change(state: dict) -> None:
    """Called on the poller thread when the reading changes. Hands it to the loop and returns.

    The poller must never wait on the brain: its whole budget is a millisecond a tick. The drift
    check itself then hops again to a worker thread inside `Brain.on_screen_change`, so neither
    the poll thread nor the event loop does the SQL.
    """
    if hub.loop is None or hub.brain is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(hub.brain.on_screen_change(state), hub.loop)
    except Exception as e:  # noqa: BLE001
        log.debug("screen change dispatch: %s", e)


def stop_screen() -> None:
    """Stop the poller with the audio session. `Brain.screen_state` stays wired: `snapshot()` then
    answers the not-started stub, which the brain reads as no screen."""
    try:
        screenmod.stop()
    except Exception as e:  # noqa: BLE001
        log.warning("screen poller stop: %s", e)


# ------------------------------------------------------------------ audio

@app.post("/api/audio/start")
async def api_audio_start(request: Request) -> JSONResponse:
    body = await _json(request)
    if hub.audio is None:
        return _err("audio is not available in this process")
    chosen = "mic_uid" in body                     # "" means the system default, on purpose
    uid = str(body.get("mic_uid") or "").strip() or None
    if chosen:
        prefs.update(mic_uid=uid, auto_start_audio=True)
    else:
        prefs.update(auto_start_audio=True)
    asyncio.create_task(_start_audio(uid or ("" if chosen else None)))
    await asyncio.sleep(0.05)
    return JSONResponse({"audio": audio_block()})


async def _start_audio(uid: str | None) -> None:
    """uid: a device uid, "" for the system default, None for whatever was saved."""
    await hub.audio.start(uid)
    start_screen()                # E43a: the screen poller lives and dies with the session
    await push_status()


@app.post("/api/audio/stop")
async def api_audio_stop() -> JSONResponse:
    if hub.audio is None:
        return _err("audio is not available in this process")
    prefs.update(auto_start_audio=False)
    await hub.audio.stop()
    stop_screen()
    await push_status()
    return JSONResponse({"audio": audio_block()})


@app.get("/api/audio/devices")
async def api_audio_devices() -> JSONResponse:
    devices = await list_devices()
    inputs = [d for d in devices if d.get("in")]
    outputs = [d for d in devices if d.get("out")]
    hub.devices = inputs
    default_out = next((d.get("name") for d in outputs if d.get("default_output")), None)
    if default_out and hub.audio is not None:
        hub.audio.output_device = default_out
    await hub.broadcast({"type": "devices", "inputs": inputs, "output_device": default_out})
    return JSONResponse({"inputs": inputs, "outputs": outputs, "output_device": default_out,
                         "mic_uid": (hub.audio.mic_uid if hub.audio else None)})


async def list_devices() -> list[dict]:
    """presage.audio.list_devices() in a thread. [] until that function exists."""
    from presage import audio as audio_mod
    fn = getattr(audio_mod, "list_devices", None)
    if fn is None:
        return []
    try:
        devices = await asyncio.wait_for(asyncio.to_thread(fn), 4)
    except Exception as e:  # noqa: BLE001
        log.warning("list_devices: %s", e)
        return []
    out = []
    for d in devices or []:
        d = dict(d)
        if "bluetooth" in str(d.get("transport", "")).lower():
            d["warning"] = BLUETOOTH_WARNING
        out.append(d)
    return out


# ------------------------------------------------------------------ context files

@app.post("/api/context/dir")
async def api_context_dir(request: Request) -> JSONResponse:
    body = await _json(request)
    raw = str(body.get("path") or "").strip()
    if not raw:
        return _err("no path")
    folder = Path(raw).expanduser()
    if not folder.is_dir():
        return _err(f"not a folder: {folder}")
    await use_context_dir(folder.resolve())
    return JSONResponse({"context": context_block(full=True)})


async def use_context_dir(folder: Path) -> None:
    prefs.update(context_dir=str(folder))
    stop_watcher()
    if hub.index is not None:
        try:
            stats = await asyncio.to_thread(hub.index.build, folder)
            log.info("indexed %s: %s", folder, stats)
        except Exception as e:  # noqa: BLE001
            log.error("index %s failed: %s", folder, e)
    if hub.brain is not None:
        hub.brain._system_text = None
        asyncio.create_task(hub.brain.prewarm())
    hub.status["index"] = index_summary(hub.index)
    schedule_deep()
    start_watcher(folder)
    await hub.broadcast(context_msg())
    await push_status()


@app.post("/api/context/browse")
async def api_context_browse() -> JSONResponse:
    code, out, err = await osascript('POSIX path of (choose folder with prompt "Choose the context folder")')
    if code != 0 or not out:
        if _cancelled(err, code):
            return JSONResponse({"cancelled": True})
        return _err(err or "could not open the folder picker")
    folder = Path(out).expanduser()
    if not folder.is_dir():
        return _err(f"not a folder: {folder}")
    await use_context_dir(folder.resolve())
    return JSONResponse({"context": context_block(full=True)})


@app.post("/api/context/add")
async def api_context_add() -> JSONResponse:
    script = ('set picked to choose file with prompt "Add files to presage" with multiple selections allowed\n'
              'set out to ""\n'
              'repeat with f in picked\n'
              '  set out to out & POSIX path of f & linefeed\n'
              'end repeat\n'
              'return out')
    code, out, err = await osascript(script)
    if code != 0 or not out.strip():
        if _cancelled(err, code):
            return JSONResponse({"cancelled": True})
        return _err(err or "could not open the file picker")
    folder = prefs.context_dir()
    folder.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    for line in out.splitlines():
        src = Path(line.strip())
        if not line.strip() or not src.is_file():
            continue
        dst = folder / src.name
        if dst.exists() and dst.resolve() == src.resolve():
            added.append(dst.name)
            continue
        n = 2
        while dst.exists():
            dst = folder / f"{src.stem} {n}{src.suffix}"
            n += 1
        try:
            await asyncio.to_thread(shutil.copy2, src, dst)
            added.append(dst.name)
        except Exception as e:  # noqa: BLE001
            log.warning("copy %s: %s", src.name, e)
    await reindex()
    return JSONResponse({"added": added, "context": context_block(full=True)})


@app.delete("/api/context/file")
async def api_context_delete(request: Request) -> JSONResponse:
    body = await _json(request)
    name = str(body.get("name") or "").strip()
    folder = prefs.context_dir()
    if not prefs.is_own_context(folder):
        return _err("presage does not delete files outside its own context folder")
    if not name or "/" in name or name.startswith("."):
        return _err("bad name")
    target = folder / name
    if not target.is_file():
        return _err(f"no such file: {name}")
    try:
        where = await asyncio.to_thread(trash, target)
    except Exception as e:  # noqa: BLE001
        return _err(f"could not remove {name}: {e}")
    log.info("removed %s (moved to %s)", name, where)
    await reindex()
    return JSONResponse({"removed": name, "trashed_to": str(where), "context": context_block(full=True)})


def trash(path: Path) -> Path:
    """Move to ~/.Trash rather than unlink: a mis-click stays recoverable."""
    bin_dir = Path.home() / ".Trash"
    if not bin_dir.is_dir():
        path.unlink()
        return path
    dst = bin_dir / path.name
    n = 2
    while dst.exists():
        dst = bin_dir / f"{path.stem} {n}{path.suffix}"
        n += 1
    shutil.move(str(path), str(dst))
    return dst


@app.post("/api/context/reveal")
async def api_context_reveal() -> JSONResponse:
    folder = prefs.context_dir()
    try:
        await asyncio.to_thread(subprocess.run, ["open", "-R", str(folder)],
                                capture_output=True, timeout=5)
    except Exception as e:  # noqa: BLE001
        return _err(str(e))
    return JSONResponse({"ok": True, "dir": str(folder)})


@app.post("/api/context/reindex")
async def api_context_reindex() -> JSONResponse:
    await reindex()
    return JSONResponse({"context": context_block(full=True), "index": hub.status.get("index", {})})


def make_llm_json():
    """(prompt, schema, model) -> dict, from brain.classify (main provider, Gemini fallback) or
    brain.llm.classify whatever shape it has."""
    fn = getattr(hub.brain, "classify", None)
    if not callable(fn):
        llm = getattr(hub.brain, "llm", None)
        fn = getattr(llm, "classify", None)
    if not callable(fn):
        return None
    takes_model = True
    try:
        params = inspect.signature(fn).parameters
        takes_model = "model" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        pass

    async def llm_json(prompt, schema, model=None):
        res = fn(prompt, schema, model=model) if (takes_model and model) else fn(prompt, schema)
        return (await res) if inspect.isawaitable(res) else res

    return llm_json


def cancel_deep() -> None:
    task, hub.deep_task = hub.deep_task, None
    if task is not None and not task.done():
        task.cancel()


def schedule_deep(force: bool = False, delay: float = DEEP_DEBOUNCE_S) -> None:
    """Start the deep pass after the fast build. A folder change cancels the run in flight."""
    if hub.index is None or not callable(getattr(hub.index, "deep_build", None)):
        return
    if not force and not prefs.load().get("deep_index", True):
        return
    cancel_deep()
    hub.deep_task = asyncio.create_task(deep_runner(delay))


async def deep_runner(delay: float) -> None:
    try:
        await asyncio.sleep(delay)
        # the fast build reopens the store; a deep pass that starts inside that window sees no
        # connection and ends as "index is not built" (2026-09-30, three files added in a row)
        for _ in range(240):
            ix = hub.index
            if ix is not None and getattr(ix, "folder", None) is not None and getattr(ix, "con", None) is not None:
                break
            await asyncio.sleep(0.5)
        llm_json = make_llm_json()
        if llm_json is None:
            log.info("deep index: no model client yet, skipping")
            return
        models = (llm_block().get("models") or {})
        t0 = time.time()
        log.info("deep index starting on %s", prefs.context_dir())
        await hub.index.deep_build(llm_json=llm_json, on_progress=on_deep_progress,
                                   model_fast=models.get("hedge") or models.get("classifier"),
                                   model_good=models.get("answer"))
        log.info("deep index finished in %.1fs: %s", time.time() - t0, deep_stats())
    except asyncio.CancelledError:
        log.info("deep index cancelled")
        raise
    except Exception as e:  # noqa: BLE001  a failed deep pass must not touch the live path
        log.warning("deep index failed: %s", e)
    finally:
        hub.deep_push_at = 0.0
        if hub.loop is not None:
            hub.loop.create_task(push_status())


def on_deep_progress(*_args, **_kwargs) -> None:
    """Called from the pass, possibly off the loop. Throttled to two frames a second."""
    now = time.time()
    if now - hub.deep_push_at < DEEP_PUSH_EVERY_S:
        return
    hub.deep_push_at = now
    if hub.loop is not None:
        asyncio.run_coroutine_threadsafe(push_status(), hub.loop)


@app.post("/api/context/deep")
async def api_context_deep(request: Request) -> JSONResponse:
    """Run the deep pass now, or switch the automatic one on and off."""
    body = await _json(request)
    if "enabled" in body:
        on = str(body.get("enabled")).lower() not in ("false", "0", "none", "")
        prefs.update(deep_index=on)
        if not on:
            cancel_deep()
        else:
            schedule_deep(force=True, delay=0.0)
    else:
        schedule_deep(force=True, delay=0.0)
    hub.status["index"] = index_summary(hub.index)      # the page renders from this reply
    await push_status()
    return JSONResponse({"index": hub.status["index"], "prefs": prefs.load()})


async def reindex() -> None:
    folder = prefs.context_dir()
    if hub.index is not None:
        try:
            stats = await asyncio.to_thread(hub.index.build, folder)
            log.info("re-indexed %s: %s", folder, stats)
        except Exception as e:  # noqa: BLE001
            log.error("re-index failed: %s", e)
    hub.status["index"] = index_summary(hub.index)
    if hub.brain is not None:
        hub.brain._system_text = None
    schedule_deep()
    await hub.broadcast(context_msg())
    await push_status()


# ------------------------------------------------------------------ model / provider

@app.post("/api/llm")
async def api_llm(request: Request) -> JSONResponse:
    body = await _json(request)
    provider = str(body.get("provider") or prefs.llm_provider()).lower()
    if provider not in prefs.PROVIDERS:
        return _err(f"unknown provider {provider!r}")

    key = body.get("api_key")
    key = str(key).strip() if key else ""
    if key and provider in prefs.ENV_VAR:
        prefs.set_key(provider, key)          # the key itself is never logged or echoed
    elif provider == "claude_code":
        key = ""                              # billed to the subscription the CLI is logged into

    models = body.get("models")
    models = {k: str(v).strip() for k, v in models.items()
              if k in prefs.MODEL_KEYS and v is not None} if isinstance(models, dict) else None
    if models is not None:
        defaults = prefs.provider_defaults(provider)
        models = {k: v for k, v in models.items()
                  if v != defaults.get(k) and (v or k == "hedge")} or None
    prefs.update(llm_provider=provider, models=models)

    hub.key_ok, hub.key_msg, hub.provider_error = None, "", ""
    setter = getattr(hub.brain, "set_provider", None)
    if callable(setter):
        try:
            # no key in the body means "keep using the saved one", not "run without a key"
            res = setter(provider, key or prefs.api_key(provider), prefs.models_for(provider))
            if inspect.isawaitable(res):
                await res
        except Exception as e:  # noqa: BLE001  saved, but the brain is still on the old one
            hub.provider_error = short_error(e)
            log.warning("provider %s did not load: %s", provider, hub.provider_error)
    else:
        hub.provider_error = f"presage.brain cannot switch provider yet; saved {provider}"
        log.warning("brain.set_provider() is not there yet: saved %s, not switched live", provider)

    apply_live_choice()                   # routing is relative to the main provider, so redo it
    if provider == "claude_code":
        login = await refresh_login()
        hub.key_ok, hub.key_msg = login["ok"], login["message"]
    else:
        ok, msg = await validate_key(provider)
        hub.key_ok, hub.key_msg = ok, msg
    await push_status()
    return JSONResponse({"llm": llm_block()})


@app.post("/api/decider")
async def api_decider(request: Request) -> JSONResponse:
    """Save the Jev key (env JEV_API_KEY) and re-check the gate. Nothing else is persisted."""
    body = await _json(request)
    key = body.get("api_key")
    key = str(key).strip() if key else ""
    if key:
        prefs.set_key("jev", key)          # never logged or echoed
    elif body.get("clear"):
        prefs.set_key("jev", "")
    choice = body.get("choice")
    if choice is not None:
        choice = str(choice).lower()
        if choice not in prefs.DECIDER_CHOICES:
            return _err(f"unknown decider {choice!r}")
        prefs.update(decider=choice)
    intent = body.get("intent")            # E21a: the lever pick and the warm card, chosen apart
    if intent is not None:
        intent = str(intent).lower()
        if intent not in prefs.INTENT_CHOICES:
            return _err(f"unknown intent {intent!r}")
        prefs.update(intent=intent)
        if hub.brain is not None:
            hub.brain.intent_choice = intent
    hub.decider, hub.decider_at = None, 0.0

    async def later() -> None:             # the rebuild loads models: never on the request
        try:
            await asyncio.to_thread(rebuild_decider)
            if getattr(hub.brain, "decider", None) is not None:
                await refresh_decider()
        except Exception as e:  # noqa: BLE001
            log.warning("decider rebuild: %s", e)
        await push_status()

    asyncio.create_task(later())
    return JSONResponse({"llm": llm_block(), "rebuilding": True})


@app.post("/api/llm/live")
async def api_llm_live(request: Request) -> JSONResponse:
    """Which provider answers live. Speculative drafts stay on the main provider."""
    body = await _json(request)
    choice = str(body.get("choice") or "auto").lower()
    if choice not in prefs.LIVE_CHOICES:
        return _err(f"unknown live choice {choice!r}")
    prefs.update(live_choice=choice)
    apply_live_choice(choice)
    await push_status()
    return JSONResponse({"llm": llm_block()})


def apply_live_choice(choice: str | None = None, retry: bool = True) -> None:
    """Tell brain where live answers go. `auto` reads ANTHROPIC_API_KEY out of the environment,
    which the claude_code provider hides while it spawns a session, so a call that lands in that
    window resolves to the main provider; retry once when that is what happened."""
    fn = getattr(hub.brain, "set_live_provider", None)
    if not callable(fn):
        return
    want = choice or prefs.load().get("live_choice", "auto")
    try:
        got = fn(want) or want
    except Exception as e:  # noqa: BLE001
        log.warning("live provider %s: %s", want, e)
        return
    log.info("live answers -> %s", got)
    if (retry and want == "auto" and got == prefs.llm_provider() == "claude_code"
            and prefs.api_key("anthropic") and hub.loop is not None):
        hub.loop.call_later(1.5, lambda: apply_live_choice("auto", retry=False))


@app.post("/api/llm/test")
async def api_llm_test(request: Request) -> JSONResponse:
    body = await _json(request)
    provider = str(body.get("provider") or prefs.llm_provider()).lower()
    if provider == "claude_code":
        login = await refresh_login()
        hub.key_ok, hub.key_msg = login["ok"], login["message"]
    else:
        ok, msg = await validate_key(provider)
        hub.key_ok, hub.key_msg = ok, msg
    await push_status()
    return JSONResponse({"llm": llm_block()})


async def validate_key(provider: str) -> tuple[bool, str]:
    """brain.llm.validate_key() when the provider layer is there, else a models.list ping."""
    fn = getattr(getattr(hub.brain, "llm", None), "validate_key", None)
    if callable(fn):
        try:
            res = fn()
            ok, msg = (await res) if inspect.isawaitable(res) else res
            msg = str(msg or "").strip()
            if ok and msg.lower() in ("", "ok", "true", "valid", "fine"):
                msg = "Key OK"                      # the page prints this as it comes
            return bool(ok), msg or "Key rejected"
        except Exception as e:  # noqa: BLE001
            return False, short_error(e)
    return await _validate_fallback(provider)


async def _validate_fallback(provider: str) -> tuple[bool, str]:
    key = prefs.api_key(provider)
    if not key:
        return False, "No API key"
    try:
        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=key)
            await asyncio.wait_for(asyncio.to_thread(lambda: next(iter(client.models.list()), None)), 15)
        else:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=key, timeout=12.0, max_retries=0)
            await client.models.list(limit=1)
        return True, "Key OK"
    except Exception as e:  # noqa: BLE001
        return False, short_error(e)


def short_error(e: Exception | str) -> str:
    """One readable line. Never includes the key: SDKs do not put it in messages."""
    s = str(e)
    m = re.search(r"['\"]message['\"]:\s*['\"]([^'\"]+)['\"]", s)
    if m:
        code = re.match(r"\D*(\d{3})\b", s)
        return f"{m.group(1)}" + (f" ({code.group(1)})" if code else "")
    s = s.strip().splitlines()[0] if s.strip() else type(e).__name__
    return s[:160]


# ------------------------------------------------------------------ intervention log (E12)

def judged_line(msg: dict) -> str:
    """The far-side line this card answers, out of the transcript the hub already holds.

    That one line and the question it became are the only transcript the log is allowed to keep,
    so nothing here walks the deque for context.
    """
    utt = msg.get("utt")
    if utt is None:
        return ""
    for row in reversed(hub.transcript):
        if row.get("source") == "system" and row.get("utt") == utt and row.get("final"):
            return str(row.get("text") or "")
    return ""


def log_card(msg: dict) -> None:
    """One answer frame is one intervention. Never raises: the log must not cost a card."""
    try:
        feedback.record({**msg, "utt_text": judged_line(msg)})
    except Exception as e:  # noqa: BLE001
        log.debug("intervention log: %s", e)


@app.post("/api/feedback")
async def api_feedback(request: Request) -> JSONResponse:
    """`{id, verdict}` with verdict in used | ignored | wrong | good. Writes through to the
    append-only log and echoes a `feedback` frame so every open page reflects the mark."""
    body = await _json(request)
    cid = str(body.get("id") or "").strip()
    if not cid:
        return _err("id is required")
    try:
        row = feedback.mark(cid, body.get("verdict"), body.get("note"))
    except ValueError as e:
        return _err(str(e))
    await hub.broadcast({"type": "feedback", "id": cid, "verdict": row["verdict"], "ts": row["ts"]})
    digest.recorder().mark(cid, row["verdict"])
    return JSONResponse({"ok": True, "id": cid, "verdict": row["verdict"], "known": row["known"],
                         "log": feedback.stats()})


# ------------------------------------------------------------------ digest (E44)

@app.post("/api/digest")
async def api_digest(request: Request) -> JSONResponse:
    """The post-call digest: one model call over the whole call, plus its Markdown.

    Nothing is streamed and nothing is broadcast: this is a document the consultant reads after
    the call, not a card. It runs on the brain's own provider, so it bills the same subscription
    the call did, and it writes `~/.cache/presage/digests/<started_at>.md` unless `save` is false.
    """
    body = await _json(request)
    rec = digest.recorder()
    if not rec.lines:
        return _err("nothing to digest yet: no transcript this session")
    llm = getattr(hub.brain, "llm", None) if hub.brain else None
    started = (hub.audio.state().get("started_at") if hub.audio else None) or rec.started_at
    rec.started_at = started or rec.started_at
    try:
        out = await digest.build(llm, rec, model=body.get("model") or None,
                                 title=str(body.get("title") or ""),
                                 log_rows=feedback.read(feedback.default_log().path()))
    except Exception as e:  # noqa: BLE001
        log.warning("digest failed: %s", e)
        return _err(f"digest failed: {type(e).__name__}: {str(e)[:200]}")
    path = None
    if body.get("save", True):
        try:
            path = str(digest.save(out["markdown"], out["meta"].get("started_at")))
        except Exception as e:  # noqa: BLE001
            log.warning("digest write failed: %s", e)
    return JSONResponse({"digest": out["digest"], "markdown": out["markdown"],
                         "meta": out["meta"], "path": path})


# ------------------------------------------------------------------ quit

@app.post("/api/quit")
async def api_quit() -> JSONResponse:
    log.info("quit requested from the page")
    await hub.broadcast({"type": "status", "quitting": True, "now": time.time()})
    if hub.audio is not None:
        try:
            await asyncio.wait_for(hub.audio.stop(), 3)
        except Exception as e:  # noqa: BLE001
            log.warning("stop on quit: %s", e)
    stop_watcher()
    screenmod.stop()
    cancel_deep()
    loop = asyncio.get_running_loop()
    loop.call_later(0.4, os._exit, 0)
    return JSONResponse({"quitting": True})


# ----------------------------------------------------------------------------- ws

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    hub.clients.add(ws)
    await ws.send_text(json.dumps(hub.hello(), default=str))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")
            brain = hub.brain
            if kind == "ask" and brain:
                asyncio.create_task(brain.ask(str(msg.get("text", ""))))
            elif kind == "set" and brain:
                key, value = str(msg.get("key")), msg.get("value")
                try:
                    brain.set_setting(key, value)
                except Exception as e:  # noqa: BLE001  a bad value must not drop the socket
                    log.warning("set %s: %s", key, e)
                if key in prefs.TUNE_KEYS and key not in brain.settings:
                    prefs.update(tune={key: value})     # brain does not own this one yet
                else:
                    save_tune_later()
                await hub.broadcast({"type": "settings", "values": settings_view()})
            elif kind == "clear" and brain:
                brain.clear()
                hub.transcript.clear()
                await hub.broadcast({"type": "cleared"})
            elif kind == "show" and brain:
                cid = msg.get("id")
                for c in brain.candidates.values():
                    if c.id == cid and c.answer:
                        from dataclasses import replace
                        aid = new_id("a")
                        shown = replace(c.answer, id=aid, source="prefetch", ts=time.time(), latency_ms=0)
                        brain.answers[aid] = shown
                        c.status = "shown"
                        await hub.broadcast(shown.msg())
                        await hub.broadcast(c.msg())
            elif kind == "ping":
                await ws.send_text(json.dumps({"type": "pong", "now": time.time()}))
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


def save_tune_later(delay: float = 1.0) -> None:
    """A slider drag is a burst of sets; write settings.json once it stops."""
    if hub._save_tune and not hub._save_tune.done():
        hub._save_tune.cancel()

    async def later() -> None:
        await asyncio.sleep(delay)
        if hub.brain:
            view = settings_view()
            prefs.update(tune={k: view[k] for k in prefs.TUNE_KEYS if k in view})

    hub._save_tune = asyncio.create_task(later())


# ----------------------------------------------------------------------------- pipeline

UTTERANCE_FIELDS = set(getattr(Utterance, "__dataclass_fields__", {}))


def utterance(tid: str, source: str, t, now: float, utt, retracts: bool) -> Utterance:
    """Build brain's Utterance, passing utt/retracts only once brain has those fields."""
    kw = {"id": tid, "source": source, "text": t.text, "final": bool(t.is_final), "ts": now}
    if "utt" in UTTERANCE_FIELDS:
        kw["utt"] = utt
    if "retracts" in UTTERANCE_FIELDS:
        kw["retracts"] = retracts
    if "question_forming" in UTTERANCE_FIELDS:
        kw["question_forming"] = bool(getattr(t, "question_forming", False))
    return Utterance(**kw)


_ECHO_WINDOW_S = 12.0
_ECHO_MIN_WORDS = 4


def _echo_words(text: str) -> list[str]:
    import re as _re
    return [w for w in _re.findall(r"[a-z0-9']+", (text or "").lower()) if len(w) >= 3]


def echo_of_system(text: str, now: float) -> bool:
    """A mic line that repeats what the far side just said (the client's voice reaching the mic
    through the speakers, so every line shows twice, Jackson 2026-09-30). True when at least
    _ECHO_MIN_WORDS of the mic line's words sit inside a system line heard in the last window at
    80% coverage or more; a short reply the consultant gives in the same words is not caught."""
    words = _echo_words(text)
    if len(words) < _ECHO_MIN_WORDS:
        return False
    recent = [(ts, ws) for ts, ws in hub.recent_system if now - ts <= _ECHO_WINDOW_S]
    hub.recent_system = recent[-40:]
    bag = set(words)
    heard: set[str] = set()
    for _ts, sys_words in recent:              # one mic line can straddle two far-side lines
        heard.update(sys_words)
    if len(heard) < _ECHO_MIN_WORDS:
        return False
    return len(bag & heard) / len(bag) >= 0.8


async def on_transcript(t) -> None:
    """STT transcripts -> UI + brain.

    The transcript id is the line identity on the page. When the provider numbers utterances
    (`utt`, monotonic per source) the id is (source, utt), so an eager final that gets retracted
    and its continuation land on the same line. Without `utt` it falls back to one open id per
    source, which is what providers that never retract need."""
    source = t.source
    utt = getattr(t, "utt", None)
    retracts = bool(getattr(t, "retracts", False))
    if utt is None:
        tid = hub.open_utterance.get(source)
        if tid is None:
            tid = new_id("t")
            hub.open_utterance[source] = tid
    else:
        tid = f"t{source[0]}{utt}"
    now = time.time()
    if source == "system":
        ws = _echo_words(t.text)
        if ws:
            hub.recent_system.append((now, ws))
    elif source == "mic" and echo_of_system(t.text, now):
        hub.echo_dropped += 1
        if hub.echo_dropped % 20 == 1:
            log.info("echo dropped from the mic (%d so far): %r", hub.echo_dropped, t.text[:60])
        # a partial of this line may have gone out before the far side's words arrived: take it
        # back; the page ignores the frame when it never showed the line
        hub.echo_ids[tid] = True
        await hub.broadcast({"type": "transcript", "id": tid, "source": source, "text": "", "final": True,
                             "t0": getattr(t, "t0", now), "t1": getattr(t, "t1", now), "ts": now,
                             "utt": utt, "retracts": True, "dropped": True})
        if len(hub.echo_ids) > 200:
            for k in list(hub.echo_ids)[:100]:
                hub.echo_ids.pop(k, None)
        if t.is_final and utt is None:
            hub.open_utterance.pop(source, None)
        return
    msg = {"type": "transcript", "id": tid, "source": source, "text": t.text, "final": bool(t.is_final),
           "t0": getattr(t, "t0", now), "t1": getattr(t, "t1", now), "ts": now}
    if utt is not None:
        msg["utt"] = utt
        msg["retracts"] = retracts
    unstable = getattr(t, "unstable", None)
    if unstable:
        msg["unstable"] = unstable          # the tail the transcriber may still revise; the page dims it
    if hub.audio is not None and source == "system":
        try:
            sp = hub.audio.speaker_for(source, float(getattr(t, "t0", 0.0)), float(getattr(t, "t1", 0.0)))
        except Exception:  # noqa: BLE001
            sp = None
        if sp:
            msg["speaker"] = sp             # "A", "B", ... stable for the call; "Me" if the far side carried Me
    await hub.broadcast(msg)
    if t.is_final and utt is None:
        hub.open_utterance.pop(source, None)
    if hub.brain:
        asyncio.create_task(hub.brain.on_transcript(utterance(tid, source, t, now, utt, retracts)))


async def status_ticker() -> None:
    while True:
        hub.status["levels"] = {k: round(v, 3) for k, v in hub.levels.items()}
        hub.levels = {k: v * 0.5 for k, v in hub.levels.items()}
        if hub.index:
            hub.status["index"] = index_summary(hub.index)
        if hub.clients:
            await push_status()
        await asyncio.sleep(1.0)


def start_watcher(folder: Path) -> None:
    stop_watcher()
    if hub.index is None:
        return
    try:
        hub.observer = hub.index.watch(folder, on_change=on_context_change)
    except Exception as e:  # noqa: BLE001
        log.warning("watch unavailable: %s", e)


def stop_watcher() -> None:
    obs, hub.observer = hub.observer, None
    if obs is None:
        return
    try:
        stop = getattr(obs, "_presage_stop", None)
        if stop is not None:
            stop.set()
        obs.stop()
    except Exception as e:  # noqa: BLE001
        log.warning("stopping watcher: %s", e)


def on_context_change(stats) -> None:
    """Called on the watchdog thread after it has already re-indexed."""
    log.info("context changed: %s", stats)
    if hub.loop is None:
        return
    asyncio.run_coroutine_threadsafe(after_context_change(), hub.loop)


async def after_context_change() -> None:
    hub.status["index"] = index_summary(hub.index)
    if hub.brain:
        hub.brain._system_text = None
        asyncio.create_task(hub.brain.prewarm())
    schedule_deep()
    await hub.broadcast(context_msg())
    await push_status()


# ----------------------------------------------------------------------------- startup

@app.on_event("startup")
async def startup() -> None:
    hub.loop = asyncio.get_running_loop()
    saved = prefs.load()
    folder = prefs.context_dir()
    folder.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    index = Index()
    stats = index.build(folder)
    hub.index = index
    hub.status["index"] = index_summary(index)
    log.info("indexed %s in %.1fs: %s", folder, time.time() - t0, stats)

    hub.brain = Brain(index, hub.broadcast)
    for key, value in (saved.get("tune") or {}).items():
        try:
            hub.brain.set_setting(key, value)
        except Exception as e:  # noqa: BLE001
            log.warning("saved setting %s=%r: %s", key, value, e)

    hub.audio = AudioSession(on_transcript=on_transcript, on_change=on_audio_change, levels=hub.levels)
    hub.audio.mic_uid = saved.get("mic_uid") or None

    feedback.install_gate_watch(lambda: hub.brain)     # E12: gate verdict + tier, no brain.py hook
    log.info("intervention log -> %s", feedback.stats()["file"])

    asyncio.create_task(hub.brain.housekeeping())
    asyncio.create_task(status_ticker())
    start_watcher(folder)
    # E43a: the reader, not the thread. `snapshot()` is a dict read of the poller's last tick and
    # answers a not-started stub when there is no poller, so this is safe to wire before one runs;
    # the poller itself starts with the audio session, in `_start_audio`.
    hub.brain.screen_state = screenmod.snapshot

    provider = prefs.llm_provider()
    setter = getattr(hub.brain, "set_provider", None)
    if callable(setter) and (provider != "anthropic" or saved.get("models")):
        try:
            res = setter(provider, prefs.api_key(provider), prefs.models_for(provider))
            if inspect.isawaitable(res):
                await res
        except Exception as e:  # noqa: BLE001
            hub.provider_error = short_error(e)
            log.warning("provider %s: %s", provider, hub.provider_error)

    apply_live_choice(saved.get("live_choice"))
    if hub.brain is not None and callable(getattr(hub.brain, "set_decider", None)):
        try:
            hub.brain.set_decider(saved.get("decider", "auto"))
        except Exception as e:  # noqa: BLE001
            log.warning("decider: %s", e)
    if provider == "claude_code":
        asyncio.create_task(refresh_login())
        asyncio.create_task(hub.brain.prewarm())
        asyncio.create_task(hub.brain.keepalive())
    elif prefs.api_key(provider):
        asyncio.create_task(check_key(provider))
        asyncio.create_task(hub.brain.prewarm())
        asyncio.create_task(hub.brain.keepalive())
    else:
        log.warning("no API key for %s: set one in Settings on the page", provider)

    asyncio.create_task(refresh_devices())
    schedule_deep()

    if os.environ.get("PRESAGE_NO_AUDIO"):
        log.info("PRESAGE_NO_AUDIO set: UI + manual ask only")
    elif saved.get("auto_start_audio", True):
        asyncio.create_task(_start_audio(hub.audio.mic_uid))
    else:
        log.info("auto start off: start listening from the page")
    hub.refresh_status()

    if os.environ.get("PRESAGE_OPEN_BROWSER"):
        asyncio.create_task(open_browser())


async def check_key(provider: str) -> None:
    ok, msg = await validate_key(provider)
    hub.key_ok, hub.key_msg = ok, msg
    if not ok:
        log.warning("%s key: %s", provider, msg)
    await push_status()


async def refresh_devices() -> None:
    devices = await list_devices()
    hub.devices = [d for d in devices if d.get("in")]
    out = next((d.get("name") for d in devices if d.get("out") and d.get("default_output")), None)
    if out and hub.audio is not None:
        hub.audio.output_device = out
    if hub.devices:
        await hub.broadcast({"type": "devices", "inputs": hub.devices, "output_device": out})


async def open_browser() -> None:
    await asyncio.sleep(0.4)
    with contextlib.suppress(Exception):
        await asyncio.to_thread(subprocess.run, ["open", f"http://localhost:{PORT}"],
                                capture_output=True, timeout=5)


# ----------------------------------------------------------------------------- helpers

async def _json(request: Request) -> dict:
    out = dict(request.query_params)
    try:
        body = await request.json()
        if isinstance(body, dict):
            out.update(body)
    except Exception:  # noqa: BLE001  no body, or not JSON
        pass
    return out


def _err(message: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=code)


def _cancelled(stderr: str, code: int = 1) -> bool:
    """Cancel raises -128. A dialog that went away without a word counts as cancelled too."""
    if "-128" in stderr or "user cancel" in stderr.lower():
        return True
    return code != 0 and not stderr.strip()


async def osascript(script: str, timeout: float = 600) -> tuple[int, str, str]:
    def run():
        return subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    try:
        p = await asyncio.to_thread(run)
    except subprocess.TimeoutExpired:
        return 1, "", "timed out"
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def lan_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> None:
    import uvicorn
    print(f"\n  presage  →  http://localhost:{PORT}   (phone / second screen: http://{lan_ip()}:{PORT})\n"
          f"  context  →  {prefs.context_dir()}\n", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
