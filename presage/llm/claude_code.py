"""Claude Code provider: answers billed to the user's Claude subscription, not to the API.

The Claude Agent SDK drives the `claude` CLI as a child process, and the CLI bills whatever
credential it finds. That makes the environment the security-critical part of this file:

    ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN visible to the child = billed to the API.

`ClaudeAgentOptions.env` merges *over* `os.environ` in the SDK's subprocess transport, so it can
override a variable but cannot delete one. The only way to make the child not see the key is to
remove it from `os.environ` for the duration of the spawn, which is what `api_key_hidden()` does,
under a lock so two spawns cannot interleave. `child_env_preview()` shows exactly what the child
would get, and the smoke test spawns a fake CLI and asserts the key is absent from its env.

Shape of one answer: a pooled persistent `ClaudeSDKClient`, our `sql` and `search` tools exposed
as an in-process MCP server, `output_format={"type": "json_schema", ...}` for the answer, and
`include_partial_messages=True` so the `answer` field streams to the screen like the other
providers.

Two settings are not optional here, both measured (docs/research-llm.md):
`strict_mcp_config=True` stops the CLI loading the user's own MCP servers, whose tool definitions
were adding 127k-267k tokens to every single request and blowing the 200k window; and thinking is
disabled, which took one Haiku answer from 64 s to 11 s.

Sources: https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan
and https://code.claude.com/docs/en/authentication
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient,
                              CLIConnectionError, CLINotFoundError, ProcessError, ResultError,
                              ResultMessage, ServerToolResultBlock, ServerToolUseBlock, StreamEvent,
                              TextBlock, ToolResultBlock, ToolUseBlock, UserMessage,
                              create_sdk_mcp_server, tool)

from presage.llm.base import (PARTIAL_EVERY_S, DraftDropped, LLM, OnPartial, RunTool, ToolSpec,
                              collect_web_urls, fallback_answer, parse_json, partial_field)

log = logging.getLogger("presage.llm.claude_code")

MCP_NAME = "presage"
PREFIX = f"mcp__{MCP_NAME}__"
POOL_SIZE = int(os.environ.get("PRESAGE_CC_POOL", "6"))
REUSE = int(os.environ.get("PRESAGE_CC_REUSE", "8"))     # queries per session before recycling
# A session is also recycled once a turn on it carried more than this many cached tokens, because
# the size of the conversation, not the number of queries, is what costs time: measured over 89
# answers (E42), first text is 0.77 s at the median while a turn stays under 60k cached tokens and
# 3.03 s between 60k and 80k. The query count cannot stand in for it -- one what-if with three tool
# rounds grows the conversation by more than four lookups do. Against REUSE 8 alone this halves the
# median (first text 2.08 s to 1.30 s, total 3,715 ms to 2,460 ms); it costs spawns, so the tail is
# worse (p90 3.09 s to 5.56 s) until the pool is refilled, which is what the prewarm is for.
# 0 turns it off.
MAX_CACHED = int(os.environ.get("PRESAGE_CC_MAX_CACHED", "60000"))
MAX_TURNS = int(os.environ.get("PRESAGE_CC_TURNS", "10"))
FILE_TURNS = 4               # files-only answers: at most 3 tool rounds, like the API provider
WARM_EVERY_S = float(os.environ.get("PRESAGE_CC_WARM_EVERY", "240"))   # re-write the prefix on this timer
MAX_WARM_SHAPES = 8          # ceiling on how many (model, schema, tools, web) prefixes we keep hot

# Spawn-ahead (E42a). Recycling a child is only cheap if the next question finds another one
# already up: E42 measured a capped call at 0.78 s of first text when it found a pooled session
# and 3.17 s when it had to spawn, and 13 of 30 calls had to spawn. So the pool now keeps its own
# children coming: one spare per shape, started in the background the moment a call takes the
# spare out (which is while that call is still running) and again when a child is recycled or dies.
SPARES = int(os.environ.get("PRESAGE_CC_SPARES", "1"))          # idle children parked per shape; 0 turns it off
# Ceiling on live CLI children: two per model for the answer and the hedge running at once, plus a
# spare for each of the three shapes E42 counted per model. Measured on this Mac (E42a), a child
# parked in the pool is 87 MB of RSS at the median and one that has just answered is 241 MB, so
# eight of them is 0.7 GB idle and about 2 GB at their busiest, on 16 GB. The selftest reached 7.
MAX_CHILDREN = int(os.environ.get("PRESAGE_CC_CHILDREN", "8"))
SPARE_IDLE_S = float(os.environ.get("PRESAGE_CC_SPARE_IDLE", "600"))   # close an idle child after this
REAP_EVERY_S = 30.0          # how often the reaper looks; it stops itself once the pool is empty
# How long a call waits for a spawn-ahead that is already running before connecting its own child.
# What it is waiting for is a child that has already done a turn, which E42 priced at 2.39 s of
# first text (0.78 s against 3.17 s), so a wait shorter than that still pays; past it the call is
# better off spawning, and the background child is parked as the spare when it lands. Unbounded is
# wrong: before this cap, one selftest call sat here for 2.59 s.
AHEAD_WAIT_S = 2.0
CLOSE_WAIT_S = 30.0          # how long close() waits for background spawns, so none outlives the process

# Memory (E52a). `MAX_CHILDREN` counts processes, which is the wrong unit when the question is
# whether the OS is about to kill one: E52's 8-draft block ran six children against a Mac with
# about 95 MB of free pages and the kernel killed five of them mid-answer. So the ceiling stays and
# a floor goes under it, read from the machine rather than assumed.
#
# The floor is 2.5 GB of available memory (free + inactive + speculative + purgeable, which is what
# `vm_stat` reports and what the kernel can hand out without swapping anonymous pages). Where it
# comes from: E42a priced a child at 87 MB parked and 246 MB just after answering, so the eight
# children this pool already allows are ~2.0 GB at their busiest; E41 measured the server with the
# transcriber and the gate in it peaking at 2.05 GB of RSS, and E21a measured 6.1 GB resident plus
# Metal with both Core ML packages loaded on this 17 GB Mac. 2.5 GB is therefore "room for the
# children we have already agreed to, and nothing borrowed from the transcriber".
#
# Two thresholds, not one, so the pool does not spawn and shed in a loop: below `MEM_FLOOR_MB` no
# *optional* child is started (spawn-ahead only -- a call that needs a child still gets one, because
# refusing that is how an answer is lost), and below `MEM_SHED_MB`, or under real kernel pressure,
# idle children are closed on the next reaper tick down to `PRESSURE_KEEP`, which is the warm child
# E52's reserve exists to hand a live answer.
MEM_FLOOR_MB = float(os.environ.get("PRESAGE_CC_MEM_FLOOR_MB", "2560"))   # no spawn-ahead below this
MEM_SHED_MB = float(os.environ.get("PRESAGE_CC_MEM_SHED_MB", "1536"))     # shed idle children below this
# `kern.memorystatus_level` is the kernel's own headroom gauge in percent, the number
# `memory_pressure -Q` prints. It is read for free through sysctl and is the second half of the
# shed test, because it sees the compressor and the file cache that `vm_stat`'s page counts do not.
PRESSURE_LEVEL = int(os.environ.get("PRESAGE_CC_PRESSURE_LEVEL", "15"))
MEM_EVERY_S = 5.0            # how stale a memory reading may be before it is taken again
PRESSURE_KEEP = 1            # idle children kept when shedding: the live lane's warm child
MEM_CHECK = os.environ.get("PRESAGE_CC_MEM_CHECK", "1").lower() not in ("0", "false", "off", "no")
WARM_PROMPT = ("Warm-up ping. Do not call any tool and do not look anything up. Reply with the JSON "
               "object only: answer \"ready\", every other field empty, null or \"none\".")
EFFORT = os.environ.get("PRESAGE_EFFORT", "")   # empty = no effort field (docs/experiments/E2-landscape.md, proposal 2)
WEB_TOOLS = ["WebSearch", "WebFetch"]
THINKING = os.environ.get("PRESAGE_CC_THINKING", "0").lower() in ("1", "true", "on")
BILLING_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
LOGIN_HINT = "Run `claude` once and log in with your Claude account"

_SPAWN_LOCK = asyncio.Lock()


def supports_effort(model: str) -> bool:
    return "haiku" not in model


def digest(text: str) -> str:
    """Short content hash. Logged per call so a prefix that moved between two answers is
    visible in the record rather than inferred from a cache_creation number."""
    return hashlib.sha1((text or "").encode()).hexdigest()[:12]


def cli_path() -> str | None:
    explicit = os.environ.get("PRESAGE_CLAUDE_CLI")
    if explicit:
        return explicit if Path(explicit).exists() else None
    found = shutil.which("claude")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "claude"
    return str(local) if local.exists() else None


# ----------------------------------------------------------------------------- memory (E52a)

_MEM: dict = {"at": 0.0, "available_mb": None, "level": None}
_PAGES = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
_PSUTIL: Any = None          # None = not looked for yet, False = not installed


def _sysctl_int(name: str) -> int | None:
    """One integer sysctl through libc, so the pressure gauge costs no subprocess."""
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("c"))
        size, buf = ctypes.c_size_t(8), ctypes.create_string_buffer(8)
        if lib.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
        return int.from_bytes(buf.raw[:size.value], "little")
    except Exception as e:  # noqa: BLE001
        log.debug("sysctl %s unavailable: %s", name, e)
        return None


def _vm_stat_mb() -> float | None:
    """Available memory in MB from `vm_stat`: free + inactive + speculative + purgeable.

    Free pages alone are not the number -- macOS keeps them near zero by design, which is why
    E52's "95 MB free" reading looked survivable right up to the point the kernel started killing
    children. Inactive and speculative pages are the ones it can actually hand out.
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout
    except Exception as e:  # noqa: BLE001
        log.debug("vm_stat unavailable: %s", e)
        return None
    page, total = 4096, 0.0
    for line in out.splitlines():
        if "page size of" in line:
            with contextlib.suppress(Exception):
                page = int(line.split("page size of")[1].split()[0])
        name, _, val = line.partition(":")
        if name.strip() in _PAGES:
            with contextlib.suppress(ValueError):
                total += int(val.strip().rstrip(".")) * page
    return total / 1e6 if total else None


def mem_state(max_age: float = MEM_EVERY_S) -> dict:
    """Cached {available_mb, level}. `psutil` when it is installed, `vm_stat` otherwise.

    Cached because it is read on the call path: `vm_stat` is a 6.7 ms subprocess on this Mac
    (measured), and the reaper refreshes the reading on its own tick, so a spawn almost always
    reads one that is already there.
    """
    if not MEM_CHECK:
        return {"available_mb": None, "level": None}
    now = time.time()
    if now - _MEM["at"] < max_age:
        return _MEM
    global _PSUTIL
    if _PSUTIL is None:
        try:
            import psutil                               # optional; not installed on this Mac
            _PSUTIL = psutil
        except Exception:  # noqa: BLE001
            _PSUTIL = False                             # don't search sys.path again every 5 s
    mb = None
    try:
        mb = _PSUTIL.virtual_memory().available / 1e6 if _PSUTIL else _vm_stat_mb()
    except Exception as e:  # noqa: BLE001
        log.debug("memory reading failed: %s", e)
    _MEM.update({"at": now, "available_mb": mb, "level": _sysctl_int("kern.memorystatus_level")})
    return _MEM


def under_pressure(st: dict) -> bool:
    """Is the machine tight enough that idle children should go? Either gauge is enough."""
    mb, level = st.get("available_mb"), st.get("level")
    return (mb is not None and mb < MEM_SHED_MB) or (level is not None and level <= PRESSURE_LEVEL)


def child_died(e: BaseException) -> bool:
    """Did this failure mean the CLI child is gone, rather than the model saying something?

    A child the OS killed surfaces two ways: `ProcessError` with the signal's exit code when it
    dies while we are reading its output, and `CLIConnectionError` ("Cannot write to terminated
    process") when a call picks a killed child out of the pool and writes to it. `ResultError` is
    excluded on purpose -- the CLI ran, reported a result and exited, so retrying it would just
    repeat whatever the model said.
    """
    if isinstance(e, (ResultError, CLINotFoundError)):
        return False
    if isinstance(e, (ProcessError, CLIConnectionError, BrokenPipeError, ConnectionResetError)):
        return True
    if type(e).__name__ in ("BrokenResourceError", "ClosedResourceError", "EndOfStream"):
        return True     # anyio, straight off a stream whose far end died
    return isinstance(e, RuntimeError) and "no result message" in str(e)


@contextlib.contextmanager
def api_key_hidden():
    """Remove the API credentials from os.environ for the length of a spawn.

    The SDK builds the child env as {**os.environ, **options.env}, so options.env alone cannot
    unset anything. Popping is what actually keeps the key out of the child; it is restored in
    `finally`, and `_SPAWN_LOCK` keeps the window to one spawn at a time.
    """
    saved = {k: os.environ.pop(k) for k in BILLING_VARS if k in os.environ}
    try:
        yield saved
    finally:
        os.environ.update(saved)


def child_env_preview(options_env: dict | None = None) -> dict:
    """What the child would see, built the same way the SDK builds it. Used by the smoke test."""
    with api_key_hidden():
        return {**os.environ, "CLAUDE_CODE_ENTRYPOINT": "sdk-py", **(options_env or {})}


# ----------------------------------------------------------------------------- one session

@dataclass(eq=False)
class _Session:
    """One connected CLI child. One query at a time; the tool handlers read `run_tool` from here,
    so two sessions never share a callback.

    `eq=False` so a session is its own identity: the pool keeps them in sets and lists and two
    children with the same field values are still two processes.
    """
    key: str
    model: str
    client: Any
    uses: int = 0
    run_tool: RunTool | None = None
    stderr: list[str] = field(default_factory=list)
    warmed: bool = False        # a request has gone through this child, so its prefix is written
    warm_at: float = 0.0        # when that last happened; the prompt cache goes cold on its own
    cached: int = 0             # cached tokens the last turn on this child carried: its size
    answers: int = 0            # real turns, not warm-ups: see _spent
    idle_at: float = 0.0        # parked in the pool at this time; the reaper closes stale ones
    on_path: bool = False       # this child was spawned by a call that was waiting for it
    lane: str = "bg"            # which lane took the slot this child is holding (E52)
    closed: bool = False

    def call(self, name: str, args: Any) -> str:
        if self.run_tool is None:
            return json.dumps({"error": "no tool callback bound"})
        bare = name[len(PREFIX):] if name.startswith(PREFIX) else name
        return self.run_tool(bare, dict(args) if isinstance(args, dict) else {})

    @property
    def proc(self) -> Any:
        """The CLI child itself, through the SDK's transport. Private attributes, so every read
        is defensive: a version that moves them costs the liveness check, not the answer."""
        return getattr(getattr(self.client, "_transport", None), "_process", None)

    @property
    def pid(self) -> int | None:
        return getattr(self.proc, "pid", None)

    def alive(self) -> bool:
        """False only when the child's exit has actually been seen (E52a).

        This is what stops a killed child being handed to the next question: the OS killed five
        of them in E52's 8-draft block and the call that picked one up raised. Unknowable counts
        as alive -- the retry in `answer()` covers the case this check cannot see.
        """
        p = self.proc
        return True if p is None else getattr(p, "returncode", None) is None

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.client.disconnect()


def _spent(s: _Session) -> bool:
    """Is this child too far into its conversation to hand out again?

    Two limits, either of which recycles it: `REUSE` queries, and `MAX_CACHED` cached tokens on
    its last turn. The second is the one that matters (E39: past ~80k cached tokens first text
    is about 3x worse), and the first cannot stand in for it, because a what-if with three tool
    rounds grows the conversation by more than four lookups do. Recycling is nearly free on this
    account: the prompt cache is account-wide, so a fresh child reads the prefix instead of
    rewriting it, and the pool has a spare waiting.

    The token cap only applies once the child has answered something. A child that has only been
    warmed carries the prefix and nothing else, which is the floor every call pays anyway (~35k
    tokens on the big test folder), so recycling it buys nothing and a cap set below the prefix would
    throw every fresh child away the moment it was warmed. At the shipped 60k this never fires.
    """
    return s.uses >= REUSE or (MAX_CACHED > 0 and s.answers > 0 and s.cached > MAX_CACHED)


# --------------------------------------------------------------------------- the priority lane (E52)
#
# E51 timed the first answering card at 5,265 ms p50 and put 92% of it in one place: the live answer
# call waiting its turn. A call that starts with nothing else running writes its first text in
# 1.86 s; one that starts with six to eight others running takes 8.12 s. The speculative path fires
# 38 drafts per two-minute clip against 8 questions, into a pool of 6, and the credited live call
# starts with a median of 4 other calls already in flight. So what a live answer needs is not a
# bigger pool but fewer neighbours, and a place in the queue ahead of the guesses.
#
# Three rules, in `_Lane` so the pool's admission control stays one object:
#   1. RESERVE of the pool's slots are for live answers only. A draft never takes the last two, so
#      an answer and its hedge find a slot without queueing behind anything.
#   2. A live waiter jumps the queue. Live waiters always form a prefix of it, so a draft is never
#      served ahead of an answer somebody actually asked for.
#   3. At most MAX_DRAFTS drafts run at once however much of the pool is free, because E51's slope
#      starts below the pool size: six `claude` children on one Mac also compete for CPU and for
#      one account's prompt cache, which a semaphore counting slots cannot see.
#
# A draft still waiting when a live call arrives to a full pool is dropped rather than queued. It
# is a guess at a question nobody has asked, and by the time a slot reached it the utterance it was
# guessing about would be over. It holds no slot, so dropping it frees nothing at that instant;
# what it buys is the next slot going to the live call, and one fewer child racing it once it runs.
#
# PRESAGE_CC_LANES=0 turns all three off and the gate becomes a plain FIFO semaphore of POOL_SIZE,
# which is exactly what E51 measured. That is the "before" arm of the contention bench.

LIVE, DRAFT, BG = "live", "draft", "bg"
LANES = os.environ.get("PRESAGE_CC_LANES", "1").lower() not in ("0", "false", "off", "no")
RESERVE = int(os.environ.get("PRESAGE_CC_RESERVE", "2"))        # slots only a live call may take
MAX_DRAFTS = int(os.environ.get("PRESAGE_CC_MAX_DRAFTS", "2"))  # drafts running at once; 0 = no cap
# E52a: which parked child a live call is handed. 0 restores E52's newest-first rule, which is the
# "before" arm of that measurement.
LIVE_LEAST_USED = os.environ.get("PRESAGE_CC_LIVE_LRU", "1").lower() not in ("0", "false", "off", "no")
RETRIES = max(0, int(os.environ.get("PRESAGE_CC_RETRIES", "1")))  # retries after a child dies mid-answer


class _Lane:
    """Admission control for the session pool.

    `_Lane(n)` with no other argument is a plain FIFO semaphore of `n`, byte-for-byte the old
    behaviour; the reserve, the draft cap and the queue jump are what E52 adds on top.
    """

    def __init__(self, size: int, reserve: int = 0, max_drafts: int = 0, jump: bool = False) -> None:
        self.size, self.reserve, self.max_drafts, self.jump = size, reserve, max_drafts, jump
        self.held = 0                    # slots taken, any lane
        self.drafts = 0                  # of those, drafts
        self._waiters: list[list] = []   # [lane, future], arrival order, live waiters first
        self.stats = {"dropped": 0, "live_waited": 0, "live_jumped": 0, "draft_waited": 0,
                      "bg_waited": 0, "draft_capped": 0, "live_max_neighbours": 0}

    # -- admission ----------------------------------------------------------

    def free(self) -> int:
        return self.size - self.held

    def _ok(self, lane: str) -> bool:
        if lane == LIVE:
            return self.free() > 0
        if self.free() <= self.reserve:
            return False
        return not (lane == DRAFT and self.max_drafts and self.drafts >= self.max_drafts)

    async def acquire(self, lane: str) -> None:
        """Take a slot, waiting for one if the lane's rules do not allow it yet."""
        if lane == LIVE:
            self.stats["live_max_neighbours"] = max(self.stats["live_max_neighbours"], self.held)
        fast = self._ok(lane)
        if fast and self._waiters:
            # a plain semaphore is FIFO. The live lane is the one E52 lets past a queue, and only
            # past waiters that are not themselves live.
            fast = self.jump and lane == LIVE and not any(w[0] == LIVE for w in self._waiters)
        if fast:
            self._take(lane)
            return
        if lane == LIVE:
            self.stats["live_waited"] += 1
            self._drop_oldest_draft()
        else:
            self.stats["draft_waited" if lane == DRAFT else "bg_waited"] += 1
            if lane == DRAFT and self.max_drafts and self.drafts >= self.max_drafts:
                self.stats["draft_capped"] += 1
        entry = [lane, asyncio.get_running_loop().create_future()]
        self._insert(entry)
        try:
            await entry[1]
        except BaseException:
            with contextlib.suppress(ValueError):
                self._waiters.remove(entry)
            fut = entry[1]
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self.release(lane)      # woken and cancelled in the same tick: hand the slot back
            raise

    def _insert(self, entry: list) -> None:
        """Live waiters form a prefix of the queue; everything else keeps arrival order."""
        if self.jump and entry[0] == LIVE:
            at = next((i for i, w in enumerate(self._waiters) if w[0] != LIVE), None)
            if at is not None:
                self._waiters.insert(at, entry)
                self.stats["live_jumped"] += 1
                return
        self._waiters.append(entry)

    def _take(self, lane: str) -> None:
        self.held += 1
        if lane == DRAFT:
            self.drafts += 1

    def release(self, lane: str) -> None:
        self.held -= 1
        if lane == DRAFT:
            self.drafts -= 1
        self._pump()

    def _pump(self) -> None:
        """Hand the free slots out, front of the queue first. A waiter its lane's rules still
        block is stepped over rather than blocking the ones behind it; a live waiter can never be
        stepped over, because `_ok(LIVE)` fails only when the pool has nothing free at all."""
        i = 0
        while i < len(self._waiters):
            lane, fut = self._waiters[i]
            if fut.done():                      # cancelled while it queued
                del self._waiters[i]
                continue
            if not self._ok(lane):
                i += 1
                continue
            del self._waiters[i]
            self._take(lane)
            fut.set_result(None)

    def _drop_oldest_draft(self) -> None:
        """A live answer is waiting on a full pool: the draft that has queued longest gives up."""
        if not self.jump:
            return
        for i, (lane, fut) in enumerate(self._waiters):
            if lane == DRAFT and not fut.done():
                del self._waiters[i]
                fut.set_exception(DraftDropped("dropped at the pool for a live answer"))
                self.stats["dropped"] += 1
                return

    def state(self) -> dict:
        return {**self.stats, "size": self.size, "held": self.held, "drafts": self.drafts,
                "waiting": len(self._waiters), "reserve": self.reserve,
                "max_drafts": self.max_drafts, "on": self.jump}


class ClaudeCodeLLM(LLM):
    provider = "claude_code"

    @classmethod
    def default_models(cls) -> dict:
        return {"answer": "claude-sonnet-5",
                "hedge": "claude-haiku-4-5",
                "classifier": "claude-haiku-4-5"}

    def __init__(self, api_key: str | None = None, models: dict | None = None) -> None:
        super().__init__(api_key, models)          # api_key is ignored: the CLI owns the credential
        self._idle: dict[str, list[_Session]] = {}
        self._lock = asyncio.Lock()
        # E52: admission control, not a bare semaphore. Off (PRESAGE_CC_LANES=0) it is one.
        self._lane = _Lane(POOL_SIZE, reserve=RESERVE if LANES else 0,
                           max_drafts=MAX_DRAFTS if LANES else 0, jump=LANES)
        self._cwd = Path(tempfile.gettempdir()) / "presage-claude-code"
        self._auth: dict | None = None      # filled by validate_key / auth_status
        self._shapes: dict[str, tuple] = {}  # call shapes seen, so prewarm knows what to keep hot
        self._specs: dict[str, tuple] = {}   # pool key -> connect args, so a spare can be built for it
        self.spares = SPARES                 # per instance, so a second provider can be built without them
        # E52a, per instance for the same reason: both arms of the A/B run in one process.
        self.live_least_used = LIVE_LEAST_USED
        self._counts = {"child_retries": 0, "child_deaths": 0, "answers_lost": 0,
                        "mem_blocked": 0, "shed_pressure": 0}
        self._live: set[_Session] = set()    # every connected child, idle or in flight
        self._pending: dict[str, asyncio.Task] = {}   # spawn-ahead in flight, one per shape
        self._ahead_lock = asyncio.Semaphore(1)       # one background warm at a time
        self._tasks: set[asyncio.Task] = set()
        self._reaper: asyncio.Task | None = None
        self._closed = False

    # ------------------------------------------------------------------ options

    @staticmethod
    def _key(model: str, system: str, tools: list[ToolSpec], schema: dict | None, web: bool) -> str:
        h = hashlib.sha1()
        for part in (model, system or "", json.dumps(schema, sort_keys=True) if schema else "",
                     ",".join(t.name for t in tools), str(web)):
            h.update(part.encode())
        return h.hexdigest()[:16]

    def _server(self, session: _Session, tools: list[ToolSpec]):
        made = []
        for spec in tools:
            async def handler(args: Any, _name: str = spec.name, _s: _Session = session) -> dict:
                out = await asyncio.to_thread(_s.call, _name, args)
                return {"content": [{"type": "text", "text": out}]}
            made.append(tool(spec.name, spec.description, spec.input_schema)(handler))
        return create_sdk_mcp_server(name=MCP_NAME, version="1.0.0", tools=made)

    def _options(self, session: _Session, model: str, system: str, tools: list[ToolSpec],
                 schema: dict | None, web: bool) -> ClaudeAgentOptions:
        builtin = list(WEB_TOOLS) if web else []
        allowed = [PREFIX + t.name for t in tools] + builtin
        kw: dict[str, Any] = dict(
            model=model,
            system_prompt=system or None,        # a plain string, never the claude_code preset
            setting_sources=[],                  # no user/project settings, no CLAUDE.md
            strict_mcp_config=True,              # ONLY our MCP server: see the note below
            tools=builtin,                       # [] disables every built-in tool
            allowed_tools=allowed,               # ours run without a permission prompt
            permission_mode="bypassPermissions",
            max_turns=MAX_TURNS if web else min(MAX_TURNS, FILE_TURNS),   # 3 tool rounds + the answer
            include_partial_messages=True,
            cwd=str(self._cwd),
            env={},                              # see api_key_hidden(): env cannot unset
            stderr=session.stderr.append,
        )
        if not THINKING:
            # measured: thinking on costs ~6k tokens and ~54 s per answer on Haiku 4.5
            kw["thinking"] = {"type": "disabled"}
        if tools:
            kw["mcp_servers"] = {MCP_NAME: self._server(session, tools)}
        if schema is not None:
            kw["output_format"] = {"type": "json_schema", "schema": schema}
        if EFFORT and supports_effort(model):
            kw["effort"] = EFFORT
        path = cli_path()
        if path:
            kw["cli_path"] = path
        return ClaudeAgentOptions(**kw)

    # ------------------------------------------------------------------ pool

    async def _acquire(self, model: str, system: str, tools: list[ToolSpec], schema: dict | None,
                       web: bool, refill: bool = True, lane: str = BG) -> _Session:
        """A session for this shape, from the pool if one is parked there.

        `refill=True` starts the replacement as soon as the spare is taken, so it is warm before
        this call has finished rather than after the next question has arrived. A call that finds
        the pool empty gives a spawn-ahead that is already running `AHEAD_WAIT_S` to land before
        connecting its own child, because that one started earlier and is already warm.

        `lane` is what E52 added: "live" goes first, keeps its own reserved slots and is the only
        lane that may take the last warm child; "draft" is capped and droppable; "bg" (the
        classifier, the prewarm) queues in arrival order.
        """
        key = self._key(model, system, tools, schema, web)
        self._specs[key] = (model, system, tools, schema, web)
        await self._lane.acquire(lane)
        try:
            s = await self._take(key, lane)
            if s is not None:
                if refill:
                    self._ahead(key)
                s.on_path = False
                s.lane = lane
                return s
            # a draft that left the last warm child alone has nothing to wait for: the spawn-ahead
            # it would wait on is building exactly the child it is not allowed to take
            s = await self._wait_ahead(key, lane) if not self._keeps_one(lane) else None
            if s is None and self._keeps_one(lane) and len(self._live) >= MAX_CHILDREN:
                # at the ceiling a draft takes the last warm child after all, because the
                # alternative is a child over the cap, and the cap is what keeps eight `claude`
                # processes from becoming twelve
                s = await self._take(key, LIVE)
            if s is not None:
                s.on_path = False
                s.lane = lane
                return s
            s = await self._connect(key, model, system, tools, schema, web)
            s.on_path = True
            s.lane = lane
            return s
        except BaseException:
            self._lane.release(lane)
            raise

    def _keeps_one(self, lane: str) -> bool:
        """E52: a draft leaves the last warm child of a shape alone.

        Only a draft. The classifier runs on its own model, so it is never in the answer's pool at
        all, and the prewarm's whole job is to take that child, rewrite its prefix and give it
        straight back; making either of them spawn beside the spare would buy a live answer
        nothing and cost a CLI child every keepalive.
        """
        return self._lane.jump and lane == DRAFT

    async def _take(self, key: str, lane: str = LIVE) -> _Session | None:
        """The newest parked child for this shape, discarding any that is spent.

        E52: a draft never takes the last one. The reserve that matters to a live answer is a
        child that is already up and already carries the prefix, which E42 priced at 0.78 s of
        first text against 3.17 s on a spawn; a free semaphore slot with no child behind it is
        worth nothing. A draft that finds only one parked child leaves it and connects its own,
        which costs the draft a spawn and costs the answer nothing.
        """
        keep = 1 if self._keeps_one(lane) else 0
        async with self._lock:
            pool = self._idle.get(key) or []
            while len(pool) > keep:
                # E52a: a live answer takes the *least-used* child, not the newest one. Holding a
                # child back for live answers means live answers keep landing on that same child,
                # and E52 measured it: the contended live calls ran on a child that had already
                # answered 4 questions at the median against 1 on the arm with no reserve. What
                # that costs is E42's curve -- 0.77 s of first text while a turn stays under 60k
                # cached tokens, 3.03 s between 60k and 80k.
                i = (min(range(len(pool)), key=lambda j: pool[j].uses)
                     if lane == LIVE and self.live_least_used else len(pool) - 1)
                s = pool.pop(i)
                if not s.alive():
                    self._counts["child_deaths"] += 1
                    log.warning("dropping a %s child that exited while parked (pid %s)", s.model, s.pid)
                elif not _spent(s):
                    return s
                self._shut(s)
        return None

    async def _connect(self, key: str, model: str, system: str, tools: list[ToolSpec],
                       schema: dict | None, web: bool) -> _Session:
        self._cwd.mkdir(parents=True, exist_ok=True)
        session = _Session(key=key, model=model, client=None)
        session.client = ClaudeSDKClient(options=self._options(session, model, system, tools, schema, web))
        t = time.time()
        async with _SPAWN_LOCK:
            with api_key_hidden():
                await session.client.connect()
        self._live.add(session)
        log.info("claude_code session up in %.1fs (%s, %d live)", time.time() - t, model, len(self._live))
        return session

    async def _release(self, s: _Session, ok: bool) -> None:
        s.run_tool = None
        s.uses += 1
        lane = s.lane
        try:
            if ok and not _spent(s) and await self._park(s):
                return
            self._shut(s)
            self._ahead(s.key)     # recycled or died: build its replacement now, not on the next question
        finally:
            self._lane.release(lane)

    # ------------------------------------------------------------------ spawn-ahead

    async def _park(self, s: _Session) -> bool:
        """Put a child in the idle pool and start the reaper that will eventually close it."""
        s.idle_at = time.time()
        async with self._lock:
            if self._closed or s.closed:
                return False
            self._idle.setdefault(s.key, []).append(s)
        if self._reaper is None or self._reaper.done():
            self._reaper = self._track(asyncio.create_task(self._reap()))
        return True

    def _shut(self, s: _Session) -> None:
        """Close a child and free its slot. Safe to call holding the pool lock: it takes no lock."""
        if s.closed:
            return
        s.closed = True
        self._live.discard(s)
        self._track(asyncio.create_task(s.close()))

    def _track(self, t: asyncio.Task) -> asyncio.Task:
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    def _idle_n(self, key: str) -> int:
        return len(self._idle.get(key) or [])

    def _ahead(self, key: str) -> None:
        """Schedule a warm spare for this shape, if the policy has room for one."""
        if self._closed or self.spares <= 0 or key not in self._specs or key in self._pending:
            return
        if self._idle_n(key) >= self.spares:
            return
        if len(self._live) + len(self._pending) >= MAX_CHILDREN and not self._make_room(key):
            log.debug("spawn-ahead skipped: %d children alive", len(self._live))
            return
        if not self._mem_ok():
            return
        self._pending[key] = self._track(asyncio.create_task(self._spawn_ahead(key)))

    def _mem_ok(self) -> bool:
        """Is there room on the machine for one more optional child? (E52a)

        Only the spare asks. A call that has to have a child gets one whatever this says, because
        the alternative is the answer nobody gets; what the floor stops is the pool adding
        processes nobody has asked for while the machine is already near the kill line.
        """
        mb = mem_state().get("available_mb")
        if mb is None or mb >= MEM_FLOOR_MB:
            return True
        self._counts["mem_blocked"] += 1
        log.info("spawn-ahead skipped: %.0f MB available, floor %.0f MB (%d children alive)",
                 mb, MEM_FLOOR_MB, len(self._live))
        return False

    def _make_room(self, key: str) -> bool:
        """At the ceiling, close the stalest idle child of another shape. Without this, spares of
        a shape nothing asks for any more (a mode switch and a re-index both move every key) would
        hold the slots for the full idle window."""
        best: tuple[str, _Session] | None = None
        for k, pool in self._idle.items():
            for s in pool:
                if k != key and (best is None or s.idle_at < best[1].idle_at):
                    best = (k, s)
        if best is None:
            return False
        self._idle[best[0]].remove(best[1])
        self._shut(best[1])
        return True

    async def _spawn_ahead(self, key: str) -> None:
        """Connect a child for `key`, write its prefix with the one-token warm request, park it.

        Serialised on `_ahead_lock` and outside the pool semaphore: it is one small extra request,
        off the critical path, and never more than one at a time, because a second concurrent
        request is what made E39's hedge slow the primary down by 26%.
        """
        try:
            async with self._ahead_lock:
                if self._closed or self._idle_n(key) >= self.spares:
                    return
                if len(self._live) >= MAX_CHILDREN and not self._make_room(key):
                    return
                if not self._mem_ok():      # the queue behind this lock can be seconds long
                    return
                model, system, tools, schema, web = self._specs[key]
                t = time.time()
                s = await self._connect(key, model, system, tools, schema, web)
                try:
                    if schema is not None:
                        await self._run(s, WARM_PROMPT, model, self.record(model, self.provider),
                                        None, time.time(), real=False)
                    if self._closed or self._idle_n(key) >= self.spares or not await self._park(s):
                        self._shut(s)
                        return
                except BaseException:
                    self._shut(s)
                    raise
                log.info("spawn-ahead parked a warm %s child in %.1fs (%d live)",
                         model, time.time() - t, len(self._live))
        except Exception as e:  # noqa: BLE001
            log.warning("spawn-ahead failed: %s", e)
        finally:
            self._pending.pop(key, None)

    async def _wait_ahead(self, key: str, lane: str = LIVE) -> _Session | None:
        """Give a spawn-ahead that is already running a moment to land, rather than starting a
        child behind it. Only a moment: past `AHEAD_WAIT_S` a call is better off spawning its own,
        and the background one is parked as the spare when it finishes."""
        t = self._pending.get(key)
        if t is None:
            return None
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(t), AHEAD_WAIT_S)
        return await self._take(key, lane)

    async def _reap(self) -> None:
        """Close children nothing has asked for in `SPARE_IDLE_S`, then stop, so an idle presage
        holds neither CLI processes nor a timer."""
        while not self._closed:
            await asyncio.sleep(REAP_EVERY_S)
            now, left = time.time(), 0
            # E52a: one memory reading per tick, which also refreshes the cache the spawn path
            # reads, so a spawn rarely pays for `vm_stat` itself.
            tight = under_pressure(await asyncio.to_thread(mem_state, 0.0))
            async with self._lock:
                for k, pool in list(self._idle.items()):
                    keep = [s for s in pool if now - s.idle_at < SPARE_IDLE_S and s.alive()]
                    for s in pool:
                        if s not in keep:
                            log.info("closing a %s child idle for %.0fs (alive=%s)",
                                     s.model, now - s.idle_at, s.alive())
                            self._shut(s)
                    self._idle[k] = keep
                    left += len(keep)
                if tight and left > PRESSURE_KEEP:
                    left = self._shed(left)
            if not left and not self._pending:
                return

    def _shed(self, left: int) -> int:
        """Under memory pressure, close parked children down to `PRESSURE_KEEP`, stalest first.

        A parked child is 87 MB that nobody is waiting on (E42a); one warm child stays, because
        that is the thing E52's reserve exists to hand a live answer, and a call that needs
        another one can still spawn it. Called holding `self._lock`; `_shut` takes no lock.
        """
        parked = sorted(((s, k) for k, pool in self._idle.items() for s in pool),
                        key=lambda p: p[0].idle_at)
        st = mem_state()
        for s, k in parked[:max(0, left - PRESSURE_KEEP)]:
            log.warning("memory pressure (%s MB available, level %s): closing an idle %s child",
                        round(st.get("available_mb") or -1), st.get("level"), s.model)
            self._idle[k].remove(s)
            self._shut(s)
            self._counts["shed_pressure"] += 1
            left -= 1
        return left

    async def close(self) -> None:
        """Shut every child down, including the ones a spawn-ahead is still connecting: a CLI
        child that outlives the process is an orphan nobody reaps."""
        self._closed = True
        if self._reaper is not None:
            self._reaper.cancel()
            self._reaper = None
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=CLOSE_WAIT_S)
        async with self._lock:
            self._idle = {}
        for s in list(self._live):
            self._live.discard(s)
            s.closed = True
            await s.close()
        pending = [t for t in self._tasks if not t.done()]
        if pending:                       # the disconnects _shut fired on the way through
            await asyncio.wait(pending, timeout=CLOSE_WAIT_S)

    # ------------------------------------------------------------------ answer

    async def answer(self, model: str, system: str, user: str, tools: list[ToolSpec], schema: dict,
                     run_tool: RunTool, on_partial: OnPartial | None = None,
                     web: bool = False, lane: str = BG) -> tuple[dict, dict]:
        t0 = time.time()
        rec = self.record(model, self.provider)
        rec["sys_len"], rec["sys_hash"] = len(system or ""), digest(system)
        rec["schema_hash"] = digest(json.dumps(schema, sort_keys=True)) if schema else ""
        self._note_shape(model, tools, schema, web)
        rec["lane"], rec["in_flight"] = lane, self._lane.held      # E52: who else was running
        # E52a: one retry when the CLI child dies under the call. The OS kills a `claude` child
        # under memory pressure (E52: five of them, `exit code: -9`) and it can also be picked out
        # of the pool already dead; before this, `answer()` raised either way and the question was
        # answered by the fallback provider or not at all. The retry is bounded at one, only for a
        # failure that means the process is gone (`child_died`), and it starts on a child the pool
        # has re-checked, because `_release` shuts the dead one and `_take` drops any other corpse.
        # A retry re-streams from the beginning, which the page handles: a partial frame replaces
        # the card's text rather than appending to it.
        data, last = None, None
        for attempt in range(RETRIES + 1):
            s = await self._acquire(model, system, tools, schema, web, lane=lane)
            rec["session"], rec["session_uses"], rec["session_warm"] = s.key, s.uses, s.warmed
            rec["acquire_s"] = round(time.time() - t0, 2)
            rec["spawned"] = s.on_path      # did this call have to wait for a child of its own?
            rec["pid"] = s.pid
            ok = False
            try:
                s.run_tool = run_tool
                data = await self._run(s, user, model, rec, on_partial, t0)
                ok = True
            except Exception as e:  # noqa: BLE001
                if attempt >= RETRIES or not child_died(e):
                    self._counts["answers_lost"] += 1 if child_died(e) else 0
                    raise
                last = e
            finally:
                await self._release(s, ok)
            if ok:
                break
            self._counts["child_retries"] += 1
            rec["retries"] = rec.get("retries", 0) + 1
            log.warning("claude_code child (pid %s) died mid-answer: %s; retrying on a fresh child",
                        rec.get("pid"), str(last)[:160])
            for k in ("first_event_s", "first_figure_s"):
                rec.pop(k, None)
            rec["first_text_s"], rec["tool_rounds"] = 0.0, 0   # the attempt that lands owns these
        rec["total_s"] = round(time.time() - t0, 2)
        data.setdefault("citations", [])
        return data, rec

    async def _run(self, s: _Session, user: str, model: str, rec: dict,
                   on_partial: OnPartial | None, t0: float, real: bool = True) -> dict:
        bufs: dict[int, str] = {}
        web_ids: set[str] = set()
        buf, texts = "", []
        last_emit, last_sent = 0.0, None
        result: Any = None
        s.warmed, s.warm_at = True, time.time()
        s.answers += 1 if real else 0     # a warm-up does not count against the token cap: _spent
        await s.client.query(user)
        async for msg in s.client.receive_response():
            if isinstance(msg, StreamEvent):
                ev = msg.event or {}
                if ev.get("type") != "content_block_delta":
                    continue
                delta = ev.get("delta") or {}
                kind = delta.get("type")
                # with output_format=json_schema the answer is a tool_use block, so the JSON
                # arrives as input_json_delta; plain prose still arrives as text_delta
                piece = (delta.get("text") if kind == "text_delta"
                         else delta.get("partial_json") if kind == "input_json_delta" else None)
                if not piece:
                    continue
                if not rec.get("first_event_s"):
                    rec["first_event_s"] = round(time.time() - t0, 2)
                i = ev.get("index", 0)
                bufs[i] = bufs.get(i, "") + piece
                text = partial_field(bufs[i])
                if text is None:
                    continue                      # a `sql` tool call's arguments, not the answer
                buf = bufs[i]
                if not rec["first_text_s"]:
                    rec["first_text_s"] = round(time.time() - t0, 2)
                if on_partial is None:
                    continue
                # E52: `figure` is the third key of the answer schema, so it closes while the
                # object is still being written. Sending it with the partial puts the number on
                # the card about half a second before the settled frame instead of with it. Read
                # closed, so a half-written number never reaches the screen; the answer text it
                # rides with already states the same figure in words, so the card shows nothing it
                # was not already showing unverified. `citations`, `say` and the verifier's
                # verdict still wait for the settled frame.
                fig = partial_field(bufs[i], "figure", closed=True)
                if fig is not None and not rec.get("first_figure_s"):
                    rec["first_figure_s"] = round(time.time() - t0, 2)
                now = time.time()
                # the figure closes once, near the end of the object, so it goes out the instant it
                # does rather than waiting on the throttle, which is the whole saving here
                fresh = fig is not None and (last_sent is None or last_sent[1] != fig)
                if (fresh or now - last_emit >= PARTIAL_EVERY_S) and (text, fig) != last_sent:
                    last_emit, last_sent = now, (text, fig)
                    await on_partial(text, model, {"figure": fig} if fig else None)
            elif isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        texts.append(b.text)
                    elif isinstance(b, ToolUseBlock):
                        rec["tool_rounds"] += 1
                        if b.name in WEB_TOOLS:
                            web_ids.add(b.id)       # its result is a plain text blob of links
                    elif isinstance(b, (ServerToolUseBlock, ServerToolResultBlock)):
                        self._web_urls(rec, b)
            elif isinstance(msg, UserMessage):
                for b in msg.content if isinstance(msg.content, list) else []:
                    if isinstance(b, ToolResultBlock) and b.tool_use_id in web_ids:
                        self._web_urls(rec, b)
            elif isinstance(msg, ResultMessage):
                result = msg
        if result is None:
            raise RuntimeError("claude_code: no result message")
        self._account(rec, result)
        s.cached = int(rec.get("cache_read") or 0) + int(rec.get("cache_write") or 0)
        if result.is_error:
            raise RuntimeError(f"claude_code: {str(result.result)[:200]}")
        data = None
        if isinstance(result.structured_output, dict):
            data = result.structured_output
        return data or parse_json(result.result or "") or parse_json("".join(texts)) \
            or parse_json(buf) or fallback_answer("".join(texts) or buf, user)

    @staticmethod
    def _web_urls(rec: dict, block: Any) -> None:
        known = {u["url"] for u in rec["web_urls"]}
        for u in collect_web_urls(getattr(block, "content", None) or getattr(block, "input", None)):
            if u["url"] not in known:
                known.add(u["url"])
                rec["web_urls"].append(u)

    def _account(self, rec: dict, result: Any) -> None:
        u = result.usage if isinstance(result.usage, dict) else {}
        rec["in"] += int(u.get("input_tokens") or 0)
        rec["out"] += int(u.get("output_tokens") or 0)
        rec["cache_read"] += int(u.get("cache_read_input_tokens") or 0)
        rec["cache_write"] += int(u.get("cache_creation_input_tokens") or 0)
        rec["thinking"] += int((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
        rec["stop"] = result.stop_reason or result.subtype
        # `total_cost_usd` is what the tokens would have cost at list price (model_usage reports
        # costBasis "list"), NOT a charge: on a subscription there is no per-token bill. So the
        # only honest signal for who paid is which credential the CLI is logged in with.
        rec["cost_usd_list"] = result.total_cost_usd
        rec["subscription"] = (self._auth or {}).get("authMethod") == "claude.ai" if self._auth else None
        rec["turns"] = result.num_turns

    def pool_stats(self) -> dict:
        """What the lane has done and what it is holding (E52), for `Brain.stats`.

        E52a adds what the machine did to the pool: `child_retries` is answers that survived a
        killed child, `answers_lost` is the ones that did not, `mem_blocked` and `shed_pressure`
        are the floor refusing a spare and the reaper closing one.
        """
        st = mem_state()
        return {**self._lane.state(), **self._counts, "children": len(self._live),
                "idle": sum(len(v) for v in self._idle.values()), "max_children": MAX_CHILDREN,
                "available_mb": round(st.get("available_mb") or 0) or None,
                "mem_level": st.get("level"), "floor_mb": MEM_FLOOR_MB}

    # ------------------------------------------------------------------ classifier

    async def classify(self, prompt: str, schema: dict, model: str | None = None) -> dict:
        model = model or self.classifier_model
        rec = self.record(model, self.provider)
        s = await self._acquire(model, "", [], schema, False)
        ok = False
        try:
            s.run_tool = None
            data = await self._run(s, prompt, model, rec, None, time.time())
            ok = True
            return data if isinstance(data, dict) else {}
        finally:
            await self._release(s, ok)

    # ------------------------------------------------------------------ warmup + key

    def _note_shape(self, model: str, tools: list[ToolSpec], schema: dict | None, web: bool) -> None:
        """Remember a call shape so prewarm can keep its prefix hot."""
        if schema is None or len(self._shapes) >= MAX_WARM_SHAPES:
            return
        names = tuple(t.name for t in tools)
        self._shapes.setdefault(self._key(model, "", tools, schema, web), (model, names, schema, web))

    def _warm_shapes(self, tools: list[ToolSpec]) -> list[tuple]:
        """The (model, tools, schema, web) shapes to write into the cache.

        Shapes seen on this run come first, because those are certain. The list is then topped up
        with the brain's own answer schema, so the first question of a call does not pay the write
        for a shape nothing has asked for yet. Since E42 that is one schema for every answer, so
        the top-up is one shape per model: the answer model and the hedge model, with the tools a
        call offers. The import is lazy and optional, so a provider built outside the brain simply
        warms less.

        Two shapes are deliberately not warmed up front, because a call may never use them and a
        warm costs a request: the what-if pre-run (no tools) and the web escalation (the built-in
        web tools are part of the prefix, so it cannot share one). `_note_shape` records either
        the first time it happens, and the keepalive warms it from then on.
        """
        out, seen = [], set()
        for m, names, sch, web in self._shapes.values():
            tl = [t for t in tools if t.name in names]
            out.append((m, tl, sch, web))
            seen.add(self._key(m, "", tl, sch, web))
        try:
            from presage.brain import answer_schema
        except Exception as e:  # noqa: BLE001
            log.debug("no answer schemas to prewarm: %s", e)
            return out[:MAX_WARM_SHAPES]
        models = [self.answer_model] + ([self.hedge_model] if self.hedge_model
                                        and self.hedge_model != self.answer_model else [])
        # one schema covers the live answer, the speculative draft and the derived answer (E42),
        # so the shape a call fires first is (model, tools, that schema)
        shapes = [(tools, answer_schema())]
        for m in models:
            for tl, sch in shapes:
                k = self._key(m, "", tl, sch, False)
                if k not in seen:
                    seen.add(k)
                    out.append((m, tl, sch, False))
        return out[:MAX_WARM_SHAPES]

    async def prewarm(self, system: str, tools: list[ToolSpec]) -> None:
        """Spawn a child per call shape and write its prefix into the prompt cache.

        Two things this used to get wrong, both measured in docs/experiments/E39-cache-hedge.md.
        It warmed `schema=None`, which is a different pool key from every real answer, so the
        session it left idle was never the one that answered. And it sent no request at all, so
        nothing was ever written to the cache: a cache entry is created by a response, not by a
        connection. The first question of a call therefore paid a full 25-41k token write, worth
        2.5-5.8 s. One throwaway request per shape moves that off the critical path, and
        `WARM_EVERY_S` re-writes it on the keepalive timer so a quiet stretch cannot let the
        entry expire underneath the next question.

        Each warmed child is released straight back into the pool, so after a prewarm every shape
        it covers has its spare parked and the first question of a call takes a warm session
        instead of connecting one (E42a).
        """
        async def warm(model: str, tls: list[ToolSpec], schema: dict | None, web: bool,
                       prefix: str | None = None) -> None:
            prefix = system if prefix is None else prefix
            t = time.time()
            s, ok = None, False
            try:
                # refill=False: the child this warms is released straight back into the pool, so
                # it is the spare. Asking for a second one here would only spawn and discard it.
                s = await self._acquire(model, prefix, tls, schema, web, refill=False)
                stale = time.time() - s.warm_at > WARM_EVERY_S
                if schema is not None and (not s.warmed or stale):
                    await self._run(s, WARM_PROMPT, model, self.record(model, self.provider),
                                    None, time.time(), real=False)
                    log.info("prewarm %s wrote %d-char prefix in %.1fs (tools=%d, web=%s)",
                             model, len(prefix or ""), time.time() - t, len(tls), web)
                ok = True
            except Exception as e:  # noqa: BLE001
                log.warning("prewarm %s failed: %s", model, e)
            finally:
                if s is not None:
                    await self._release(s, ok)

        jobs = [warm(m, tl, sch, web) for m, tl, sch, web in self._warm_shapes(tools)]
        if not jobs:                                   # nothing known: at least spawn the children
            jobs = [warm(self.answer_model, tools, None, False)]
            if self.hedge_model and self.hedge_model != self.answer_model:
                jobs.append(warm(self.hedge_model, tools, None, False))
        if self.classifier_model not in (self.answer_model, self.hedge_model):
            jobs.append(warm(self.classifier_model, [], None, False, prefix=""))
        await asyncio.gather(*jobs)

    async def validate_key(self) -> tuple[bool, str]:
        """`claude auth status --json`, with the API key hidden so it reports the credential the
        answers will actually use."""
        status = await self.auth_status()
        if "error" in status:
            return False, status["error"]
        if not status.get("loggedIn"):
            return False, LOGIN_HINT
        who = status.get("email") or status.get("account") or status.get("authMethod") or "claude account"
        plan = status.get("subscriptionType")
        return True, f"ok ({who}{', ' + plan if plan else ''})"

    async def auth_status(self) -> dict:
        """Cached `claude auth status --json`. Also tells `_account` who is paying."""
        if self._auth is not None:
            return self._auth
        path = cli_path()
        if not path:
            return {"error": "claude CLI not found (install it, or set PRESAGE_CLAUDE_CLI)"}
        try:
            async with _SPAWN_LOCK:
                with api_key_hidden():
                    proc = await asyncio.create_subprocess_exec(
                        path, "auth", "status", "--json",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                        stdin=asyncio.subprocess.DEVNULL, cwd=tempfile.gettempdir())
                out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            return {"error": "claude auth status timed out"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {str(e)[:100]}"}
        status = parse_json(out.decode("utf-8", "replace")) or {}
        if not status:
            return {"error": err.decode("utf-8", "replace")[:120] or "claude auth status returned nothing"}
        self._auth = status
        return status
