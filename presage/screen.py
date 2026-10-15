"""What the presenter has on screen: which slide of which deck, which sheet of which workbook.

A 1 Hz poller on its own thread. It answers one question -- "what document, what page" --
so the brain can resolve "this number", "the figure on this slide", "what's this based on"
without the presenter naming anything, and so E16's figure -> cell links for the slide that
is up can be packed into the evidence first.

    from presage import screen
    w = screen.start(index)          # background thread, ~0.2% of a core idle
    w.snapshot()                     # {"app": "Preview", "document": "Plan_Memo.pdf",
                                     #  "page": 3, "pages": 4, "file": "Plan_Memo.pdf",
                                     #  "loc": "page 3", "status": "ok", "via": "ax-title", ...}

Two signals, cheapest first, both measured on this Mac (docs/experiments/E43-screen.md):

1. **ctypes, every tick, 1.7 ms of CPU.** `CGWindowListCopyWindowInfo` gives the on-screen
   windows front to back with their owner and title; `AXUIElementCopyAttributeValue` on the
   front app's focused window gives `AXTitle` and `AXDocument`. Preview puts the page index
   in its AX title ("Memo.pdf - Page 3 of 4") where the CoreGraphics title has only the file
   name, and `AXDocument` is a file:// URL, so the open document maps to an indexed file by
   path rather than by guessing at names. No subprocess, no Apple Event, nothing to build.

2. **One long-lived `osascript -l JavaScript` helper, only while an app that needs it is in
   front, 5.6 ms of CPU a tick.** Keynote and PowerPoint only tell you the current slide over
   Apple Events, Excel and Numbers only tell you the active sheet, and Chrome and Safari only
   tell you the tab URL. Spawning `osascript` per tick costs 48 ms of CPU (4.8% of a core at
   1 Hz); one helper that blocks on stdin and answers one JSON line a tick costs 5.6 ms. The
   helper is started when such an app comes to the front and killed 20 s after it leaves, so
   a call spent in Zoom pays nothing for it.

What breaks it, and what it reports instead (it never raises and never stops polling):

  * **screen locked**: `CGSSessionScreenIsLocked` is 1 and macOS stubs every window tree.
    status "unavailable", reason "locked". (This is what blocked E37.)
  * **no Accessibility grant**: AX is skipped entirely, and the CoreGraphics window title
    carries on working, so the document still resolves and only the page index is lost.
    reason "no_accessibility" when that is the only thing missing.
  * **no Screen Recording grant either**: `kCGWindowName` comes back empty for every window,
    so only the app name is left. reason "no_window_titles".
  * **no Automation grant for a scriptable app**: the helper answers `{"err": ...}` once, the
    app is marked for cheap signals only for 60 s, and nothing prompts again in that window.
  * **a wedged app**: every AX call is bounded by `AXUIElementSetMessagingTimeout` (0.4 s) and
    every helper reply by a 2.5 s read timeout, after which the helper is killed and respawned.

Grants, which are three different rows in System Settings > Privacy & Security:
  Accessibility (page index, AX titles) - Screen Recording (window titles at all) -
  Automation (Keynote/PowerPoint/Excel/Numbers/Chrome/Safari). presage already asks for the
  second for audio. macOS never prompts for the first. The grant attaches to the app that
  launched the process, so under ~/Applications/Presage.app the row says Presage and from a
  terminal it says the terminal, and re-running make_app.sh invalidates it (E37).

CLI:
    uv run python -m presage.screen --once        one snapshot, JSON
    uv run python -m presage.screen --watch 20    20 s of change lines, with CPU at the end
    uv run python -m presage.screen --walk        the E43 fixture: open a document, step it,
                                                  and print the per-step latency table
"""
from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import difflib
import json
import logging
import os
import pathlib
import plistlib
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger("presage.screen")

INTERVAL_S = 1.0          # the poll period the brief asks for
HELPER_IDLE_S = 20.0      # kill the Apple Events helper this long after its app leaves the front
HELPER_READ_S = 2.5       # a reply slower than this means the app is wedged: respawn
DENIED_S = 60.0           # after an Automation refusal, do not ask that app again for this long
AX_TIMEOUT_S = 0.4        # bound on one AX round trip to another process
MATCH_FLOOR = 0.62        # below this a document name is not claimed to be an indexed file


# --------------------------------------------------------------------------- CoreFoundation

def _lib(name: str, fallback: str = "") -> ctypes.CDLL | None:
    try:
        return ctypes.cdll.LoadLibrary(ctypes.util.find_library(name) or fallback)
    except OSError as e:  # noqa: BLE001 - a Mac without the framework is not a crash
        log.debug("cannot load %s: %s", name, e)
        return None


_cf = _lib("CoreFoundation")
_cg = _lib("CoreGraphics")
_ax = _lib("ApplicationServices")
_proc = _lib("proc", "/usr/lib/libproc.dylib")
MAC = bool(_cf and _cg)

UTF8 = 0x08000100
PLIST_XML = 100
KCG_ON_SCREEN, KCG_NO_DESKTOP = 1, 16

if MAC:
    _cf.CFRelease.argtypes = [ctypes.c_void_p]
    _cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
    _cf.CFGetTypeID.restype = ctypes.c_ulong
    _cf.CFStringGetTypeID.restype = ctypes.c_ulong
    _cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    _cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    _cf.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _cf.CFStringGetCStringPtr.restype = ctypes.c_char_p
    _cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
    _cf.CFStringGetLength.restype = ctypes.c_long
    _cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    _cf.CFStringGetCString.restype = ctypes.c_bool
    _cf.CFPropertyListCreateData.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_void_p]
    _cf.CFPropertyListCreateData.restype = ctypes.c_void_p
    _cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
    _cf.CFDataGetLength.restype = ctypes.c_long
    _cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    _cf.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_ubyte)
    _cg.CGWindowListCopyWindowInfo.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    _cg.CGWindowListCopyWindowInfo.restype = ctypes.c_void_p
    _cg.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p

if _ax:
    _ax.AXIsProcessTrusted.restype = ctypes.c_bool
    _ax.AXUIElementCreateApplication.argtypes = [ctypes.c_int]
    _ax.AXUIElementCreateApplication.restype = ctypes.c_void_p
    _ax.AXUIElementCopyAttributeValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_void_p)]
    _ax.AXUIElementCopyAttributeValue.restype = ctypes.c_int
    _ax.AXUIElementSetMessagingTimeout.argtypes = [ctypes.c_void_p, ctypes.c_float]
    _ax.AXUIElementSetMessagingTimeout.restype = ctypes.c_int

if _proc:
    _proc.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    _proc.proc_pidpath.restype = ctypes.c_int


def _cfstr(s: str) -> Any:
    return _cf.CFStringCreateWithCString(None, s.encode("utf-8"), UTF8)


def _pystr(ref: Any) -> str | None:
    """A CFStringRef as a str. The fast pointer path misses for non-ASCII, hence the copy."""
    if not ref:
        return None
    p = _cf.CFStringGetCStringPtr(ref, UTF8)
    if p:
        return p.decode("utf-8", "replace")
    n = _cf.CFStringGetLength(ref) * 4 + 8
    buf = ctypes.create_string_buffer(n)
    if _cf.CFStringGetCString(ref, buf, n, UTF8):
        return buf.value.decode("utf-8", "replace")
    return None


def _plist(ref: Any) -> Any:
    """Any plist-shaped CF object as Python, in one CoreFoundation call plus one parse.

    Walking a CFArray of CFDictionaries field by field through ctypes is a page of type
    checks; serialising the whole thing to XML and handing it to plistlib is two calls and
    is what makes the 1 Hz tick cost 1 ms instead of 10.
    """
    if not ref:
        return None
    data = _cf.CFPropertyListCreateData(None, ref, PLIST_XML, 0, None)
    if not data:
        return None
    try:
        raw = ctypes.string_at(_cf.CFDataGetBytePtr(data), _cf.CFDataGetLength(data))
        return plistlib.loads(raw)
    except Exception as e:  # noqa: BLE001
        log.debug("plist decode: %s", e)
        return None
    finally:
        _cf.CFRelease(data)


def window_list() -> list[dict]:
    """On-screen windows, front to back. Titles need the Screen Recording grant; everything
    else (owner name, pid, layer) does not."""
    if not MAC:
        return []
    ref = _cg.CGWindowListCopyWindowInfo(KCG_ON_SCREEN | KCG_NO_DESKTOP, 0)
    try:
        return _plist(ref) or []
    finally:
        if ref:
            _cf.CFRelease(ref)


def screen_locked() -> bool:
    """True while the login window is up. macOS then stubs every accessibility tree (E37)."""
    if not MAC:
        return False
    ref = _cg.CGSessionCopyCurrentDictionary()
    try:
        d = _plist(ref) or {}
    finally:
        if ref:
            _cf.CFRelease(ref)
    return bool(d.get("CGSSessionScreenIsLocked")) or d.get("kCGSSessionOnConsoleKey") is False


def ax_trusted() -> bool:
    return bool(_ax and _ax.AXIsProcessTrusted())


def _ax_attr(el: Any, name: str) -> Any:
    out = ctypes.c_void_p()
    key = _cfstr(name)
    try:
        err = _ax.AXUIElementCopyAttributeValue(el, key, ctypes.byref(out))
    finally:
        _cf.CFRelease(key)
    return out.value if err == 0 else None


def _ax_str(el: Any, name: str) -> str | None:
    ref = _ax_attr(el, name)
    if not ref:
        return None
    try:
        if _cf.CFGetTypeID(ref) != _cf.CFStringGetTypeID():
            return None
        return _pystr(ref)
    finally:
        _cf.CFRelease(ref)


def ax_window(pid: int) -> tuple[str | None, str | None]:
    """(title, document path) of that process's focused window, or (None, None).

    Falls back to AXMainWindow, because an app that is not frontmost has no focused window;
    the deck beside Zoom is exactly that case.
    """
    if not _ax or pid <= 0:
        return None, None
    app = _ax.AXUIElementCreateApplication(int(pid))
    if not app:
        return None, None
    try:
        _ax.AXUIElementSetMessagingTimeout(app, AX_TIMEOUT_S)
        win = _ax_attr(app, "AXFocusedWindow") or _ax_attr(app, "AXMainWindow")
        if not win:
            return None, None
        try:
            title = _ax_str(win, "AXTitle")
            doc = _ax_str(win, "AXDocument")
            return title, _unfile(doc)
        finally:
            _cf.CFRelease(win)
    finally:
        _cf.CFRelease(app)


def _unfile(url: str | None) -> str | None:
    """A file:// URL as a path, an absolute path as itself, anything else as None.

    `AXDocument` on a browser window is the page's URL, and `PurePath(url).name` on that is
    "edit?gid=796678265#gid=796678265", which would be shown to the presenter as the name of
    the document they are looking at. Only a real path is a document.
    """
    if not url:
        return None
    if url.startswith("file://"):
        from urllib.parse import unquote, urlparse
        return unquote(urlparse(url).path) or None
    return url if url.startswith("/") else None


_BUNDLE_CACHE: dict[int, tuple[str, str]] = {}    # pid -> (bundle id, executable path)


def bundle_of(pid: int) -> str:
    """The bundle identifier for a pid, cached. Name matching is not safe: this Mac's Keynote
    is installed as "Keynote Creator Studio.app" and still answers to com.apple.Keynote."""
    if pid in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[pid][0]
    bid, path = "", ""
    if _proc and pid > 0:
        buf = ctypes.create_string_buffer(4096)
        if _proc.proc_pidpath(int(pid), buf, 4096) > 0:
            path = buf.value.decode("utf-8", "replace")
            # Not `x.app`: an app that self-updates runs from a code-sign clone whose bundle is
            # /private/var/folders/.../com.google.Chrome.code_sign_clone/.../Google Chrome.app.bundle,
            # which is why Chrome resolved to nothing until this stopped checking the suffix.
            app = path.split("/Contents/MacOS/")[0]
            try:
                info = plistlib.loads(pathlib.Path(app, "Contents", "Info.plist").read_bytes())
                bid = str(info.get("CFBundleIdentifier") or "")
            except Exception as e:  # noqa: BLE001
                log.debug("Info.plist for %s: %s", app, e)
    if len(_BUNDLE_CACHE) > 400:
        _BUNDLE_CACHE.clear()
    _BUNDLE_CACHE[pid] = (bid.lower(), path)
    return bid.lower()


# --------------------------------------------------------------------------- the app table

@dataclass(frozen=True)
class App:
    key: str            # short name used in the state and by the helper
    kind: str           # "deck" | "sheet" | "doc" | "browser"
    script: str = ""    # the JXA probe's app key, "" when cheap signals are enough
    label: str = ""     # the name to show


APPS: dict[str, App] = {
    "com.apple.keynote":            App("keynote", "deck", "keynote", "Keynote"),
    "com.apple.iwork.keynote":      App("keynote", "deck", "keynote", "Keynote"),
    "com.microsoft.powerpoint":     App("powerpoint", "deck", "powerpoint", "PowerPoint"),
    "com.microsoft.excel":          App("excel", "sheet", "excel", "Excel"),
    "com.apple.numbers":            App("numbers", "sheet", "numbers", "Numbers"),
    "com.apple.iwork.numbers":      App("numbers", "sheet", "numbers", "Numbers"),
    "com.apple.preview":            App("preview", "doc", "", "Preview"),
    "com.apple.safari":             App("safari", "browser", "safari", "Safari"),
    "com.google.chrome":            App("chrome", "browser", "chrome", "Chrome"),
    "com.google.chrome.beta":       App("chrome", "browser", "chrome", "Chrome"),
    "com.google.chrome.canary":     App("chrome", "browser", "chrome", "Chrome"),
    "com.brave.browser":            App("brave", "browser", "brave", "Brave"),
    "com.microsoft.edgemac":        App("edge", "browser", "edge", "Edge"),
    "com.adobe.acrobat.pro":        App("acrobat", "doc", "", "Acrobat"),
    "com.adobe.reader":             App("acrobat", "doc", "", "Acrobat Reader"),
    "com.readdle.pdfexpert-mac":    App("pdfexpert", "doc", "", "PDF Expert"),
    "net.sourceforge.skim-app.skim": App("skim", "doc", "", "Skim"),
    "org.libreoffice.script":       App("libreoffice", "doc", "", "LibreOffice"),
    "com.microsoft.word":           App("word", "doc", "", "Word"),
    "com.apple.pages":              App("pages", "doc", "", "Pages"),
    "com.apple.iwork.pages":        App("pages", "doc", "", "Pages"),
}

# Window-server processes that own a layer-0 window without being an app the user is looking at.
SKIP_OWNERS = {"Window Server", "Dock", "Control Center", "Notification Center", "Spotlight",
               "loginwindow", "Screenshot", "SystemUIServer"}

# "Memo.pdf - Page 3 of 4" (Preview, and the same shape in Skim and PDF Expert). English only:
# a localized Preview says "Seite 3 von 4" and falls through to the document with no page.
PAGE_IN_TITLE = re.compile(r"^(?P<doc>.*?)\s*[-‐-―]\s*Page\s+(?P<page>\d+)"
                           r"(?:\s+of\s+(?P<pages>\d+))?\s*$", re.I)
# Chrome and Safari put the tab title in the window title; the web app is its suffix.
WEB_SUFFIX = re.compile(r"\s*[-‐-―]\s*(Google Slides|Google Sheets|Google Docs|"
                        r"Microsoft PowerPoint|Excel|Canva)\s*$", re.I)
WEB_KIND = {"google slides": "deck", "canva": "deck", "microsoft powerpoint": "deck",
            "google sheets": "sheet", "excel": "sheet", "google docs": "doc"}
# Google keeps the open slide in the fragment: .../edit#slide=id.g31f0b2c4_0_5
SLIDE_FRAGMENT = re.compile(r"[#&]slide=id\.(?P<id>[\w-]+)")
GID_FRAGMENT = re.compile(r"[#&]gid=(?P<gid>\d+)")


# --------------------------------------------------------------------------- name matching

_COPY = re.compile(r"\s*(?:\(\d+\)|copy(?:\s+\d+)?|final|draft|v\d+(?:\.\d+)?)\s*$", re.I)


def _norm(name: str) -> str:
    stem = re.sub(r"\.(pptx?|keynote?|key|pdf|xlsx?m?|numbers|docx?|csv|md|txt)$", "", name or "", flags=re.I)
    stem = _COPY.sub("", stem)
    return re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()


def match_file(document: str | None, candidates: list[str]) -> tuple[str | None, float]:
    """The indexed file a document on screen is, and how sure that is.

    Exact name wins, then containment, then a token overlap blended with a character ratio.
    Below MATCH_FLOOR nothing is claimed: a wrong file is worse than no file, because the
    evidence packer would then put another deck's cells in front of the model.
    """
    if not document or not candidates:
        return None, 0.0
    want = _norm(pathlib.PurePath(document).name)
    if not want:
        return None, 0.0
    best, score = None, 0.0
    for cand in candidates:
        got = _norm(cand)
        if not got:
            continue
        if got == want:
            return cand, 1.0
        if want in got or got in want:
            s = 0.90 * min(len(want), len(got)) / max(len(want), len(got)) + 0.10
        else:
            a, b = set(want.split()), set(got.split())
            jac = len(a & b) / len(a | b) if (a | b) else 0.0
            s = 0.5 * jac + 0.5 * difflib.SequenceMatcher(None, want, got).ratio()
        if s > score:
            best, score = cand, s
    return (best, round(score, 3)) if score >= MATCH_FLOOR else (None, round(score, 3))


# --------------------------------------------------------------------------- Apple Events helper

# One long-lived osascript process. It blocks on stdin, answers one JSON line per request and
# exits when the pipe closes, so the Apple Event cost is paid once a tick instead of a process
# start once a tick: 5.6 ms of CPU against 48 ms (E43). Everything is wrapped, because an app
# that is not running, not scriptable or not authorised must be a null field, not a stack trace.
HELPER_JS = r"""
ObjC.import('Foundation');
function say(o) {
  var s = JSON.stringify(o) + "\n";
  var d = $.NSString.alloc.initWithUTF8String(s).dataUsingEncoding($.NSUTF8StringEncoding);
  $.NSFileHandle.fileHandleWithStandardOutput.writeData(d);
}
function ok(f) { try { var v = f(); return (v === undefined) ? null : v; } catch (e) { return null; } }
function probe(req) {
  var r = {"for": req.app};
  var a = req.app;
  if (a === "keynote") {
    var k = Application("Keynote");
    var d = k.documents[0];
    r.doc = ok(function () { return d.name(); });
    r.path = ok(function () { return d.file().toString(); });
    r.pages = ok(function () { return d.slides.length; });
    r.page = ok(function () { return d.currentSlide.slideNumber(); });
    r.playing = ok(function () { return k.playing(); });
    r.title = ok(function () { return d.currentSlide.defaultTitleItem.objectText(); });
  } else if (a === "powerpoint") {
    var p = Application("Microsoft PowerPoint");
    var pres = p.activePresentation;
    r.doc = ok(function () { return pres.name(); });
    r.path = ok(function () { return pres.fullName(); });
    r.pages = ok(function () { return pres.slides.length; });
    r.page = ok(function () { return p.slideShowWindows[0].view.slide.slideIndex(); });
    if (r.page === null) {
      r.page = ok(function () { return p.documentWindows[0].selection.slideRange.slideIndex(); });
    } else { r.playing = true; }
  } else if (a === "excel") {
    var x = Application("Microsoft Excel");
    r.doc = ok(function () { return x.activeWorkbook.name(); });
    r.path = ok(function () { return x.activeWorkbook.fullName(); });
    r.sheet = ok(function () { return x.activeSheet.name(); });
    r.cell = ok(function () { return x.activeCell.getAddress({ rowAbsolute: false, columnAbsolute: false }); });
    r.pages = ok(function () { return x.activeWorkbook.worksheets.length; });
  } else if (a === "numbers") {
    var n = Application("Numbers");
    var nd = n.documents[0];
    r.doc = ok(function () { return nd.name(); });
    r.path = ok(function () { return nd.file().toString(); });
    r.sheet = ok(function () { return nd.activeSheet.name(); });
    r.pages = ok(function () { return nd.sheets.length; });
  } else if (a === "chrome" || a === "brave" || a === "edge") {
    var names = { chrome: "Google Chrome", brave: "Brave Browser", edge: "Microsoft Edge" };
    var c = Application(names[a]);
    var t = c.windows[0].activeTab;
    r.url = ok(function () { return t.url(); });
    r.doc = ok(function () { return t.title(); });
  } else if (a === "safari") {
    var s = Application("Safari");
    var st = s.windows[0].currentTab;
    r.url = ok(function () { return st.url(); });
    r.doc = ok(function () { return st.name(); });
  }
  return r;
}
var inh = $.NSFileHandle.fileHandleWithStandardInput;
while (true) {
  var data = null;
  try { data = inh.availableData; } catch (e) { break; }
  if (!data || data.length === 0) break;
  var line = ObjC.unwrap($.NSString.alloc.initWithDataEncoding(data, $.NSUTF8StringEncoding)) || "";
  line = line.split("\n")[0];
  if (!line || line === "quit") break;
  var req = null;
  try { req = JSON.parse(line); } catch (e) { req = null; }
  if (!req) { say({ err: "bad request" }); continue; }
  var out;
  try { out = probe(req); } catch (e) { out = { "for": req.app, err: String(e) }; }
  say(out);
}
"""


class _Helper:
    """The osascript process, started on demand and killed when nothing needs it."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.started = 0.0
        self.used = 0.0
        self.fails = 0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> bool:
        if self.alive():
            return True
        self.stop()
        try:
            self.proc = subprocess.Popen(
                ["/usr/bin/osascript", "-l", "JavaScript", "-e", HELPER_JS],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            log.debug("screen helper start: %s", e)
            self.proc = None
            return False
        self.started = self.used = time.time()
        return True

    def ask(self, app: str) -> dict:
        """One request, one reply. A reply slower than HELPER_READ_S means the target app is
        wedged (a modal sheet, a spinning beachball), so the helper is killed rather than
        left holding a blocked Apple Event for the rest of the call."""
        if not self.start():
            return {"err": "no helper"}
        self.used = time.time()
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        try:
            self.proc.stdin.write(json.dumps({"app": app}).encode() + b"\n")
            self.proc.stdin.flush()
        except Exception as e:  # noqa: BLE001
            self.stop()
            return {"err": f"write: {e}"}
        got: list[bytes] = []
        done = threading.Event()

        def read() -> None:
            try:
                got.append(self.proc.stdout.readline())   # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                got.append(b"")
            done.set()

        t = threading.Thread(target=read, daemon=True)
        t.start()
        if not done.wait(HELPER_READ_S):
            self.stop()          # the reader thread dies with the pipe
            self.fails += 1
            return {"err": "timeout"}
        if not got or not got[0]:
            self.stop()
            return {"err": "helper exited"}
        try:
            return json.loads(got[0])
        except ValueError:
            return {"err": "bad reply"}

    def idle_stop(self, now: float) -> None:
        if self.alive() and now - self.used > HELPER_IDLE_S:
            self.stop()

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            p.terminate()
            p.wait(timeout=1.5)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- the state

@dataclass
class ScreenState:
    status: str = "unavailable"        # "ok" | "unavailable"
    reason: str | None = "not_started"  # locked | no_accessibility | no_window_titles | no_document | ...
    app: str | None = None              # "PowerPoint", "Preview", "Chrome"
    app_key: str | None = None          # "powerpoint", "preview", "chrome"
    kind: str | None = None             # "deck" | "sheet" | "doc" | "browser"
    document: str | None = None         # the file name as the app shows it
    path: str | None = None             # the full path, when the app gives one
    page: int | None = None             # 1-based slide or page index
    pages: int | None = None            # total, when known
    sheet: str | None = None            # active sheet, for a workbook
    cell: str | None = None             # active cell, for a workbook
    title: str | None = None            # the slide's or page's title, from the index when cheap
    presenting: bool = False            # a slideshow is running, not the editor
    file: str | None = None             # the indexed file this document is
    file_score: float = 0.0
    loc: str | None = None              # the chunk citation head: "slide 4" / "page 3"
    via: str | None = None              # which signal answered: ax-title | applescript | cg-title
    focused: bool = False               # the document app is also the frontmost app
    url: str | None = None              # the tab, for a browser
    ts: float = 0.0
    tick_ms: float = 0.0                # how long this tick took
    grants: dict = field(default_factory=dict)

    def msg(self) -> dict:
        d = asdict(self)
        d["age"] = round(max(0.0, time.time() - self.ts), 3) if self.ts else None
        return d


class ScreenWatcher:
    """The poller. Its own daemon thread: never the transcript thread, never the event loop.

    `snapshot()` is a plain dict read of the last tick and does no I/O, so the brain can call
    it inside the answer path without inheriting any of the failure modes above.
    """

    def __init__(self, index: Any = None, interval: float = INTERVAL_S, detail: bool = True) -> None:
        self.index = index
        self.interval = max(0.2, float(interval))
        self.detail = detail
        self._state = ScreenState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._helper = _Helper()
        self._denied: dict[str, float] = {}       # app key -> when Automation refused it
        self._files: list[str] = []
        self._files_version: Any = object()
        self.ticks = 0
        self.cpu_s = 0.0          # this process, including the poll thread
        self.child_cpu_s = 0.0    # the osascript helper: its CPU is ours too
        self.on_change = None                      # optional callback(dict), called off-thread

    # -------------------------------------------------------------- lifecycle

    def start(self) -> "ScreenWatcher":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="presage-screen", daemon=True)
        self._thread.start()
        atexit.register(self.stop)
        return self

    def stop(self) -> None:
        self._stop.set()
        self._helper.stop()

    def snapshot(self) -> dict:
        with self._lock:
            return self._state.msg()

    def state(self) -> ScreenState:
        with self._lock:
            return self._state

    # -------------------------------------------------------------- the loop

    @staticmethod
    def _child_cpu() -> float:
        import resource
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return r.ru_utime + r.ru_stime

    def _run(self) -> None:
        while not self._stop.is_set():
            t0, c0, k0 = time.perf_counter(), time.process_time(), self._child_cpu()
            try:
                new = self.tick()
            except Exception as e:  # noqa: BLE001 - a poller that dies is worse than a wrong field
                log.debug("screen tick: %s", e)
                new = ScreenState(status="unavailable", reason=f"error: {type(e).__name__}", ts=time.time())
            new.tick_ms = round((time.perf_counter() - t0) * 1000, 2)
            self.ticks += 1
            self.cpu_s += time.process_time() - c0
            self.child_cpu_s += max(0.0, self._child_cpu() - k0)
            with self._lock:
                old, self._state = self._state, new
            if self.on_change and self._changed(old, new):
                try:
                    self.on_change(new.msg())
                except Exception as e:  # noqa: BLE001
                    log.debug("screen on_change: %s", e)
            self._helper.idle_stop(time.time())
            self._stop.wait(self.interval)
        self._helper.stop()

    @staticmethod
    def _changed(a: ScreenState, b: ScreenState) -> bool:
        keys = ("status", "reason", "app_key", "document", "page", "sheet", "file", "presenting")
        return any(getattr(a, k) != getattr(b, k) for k in keys)

    # -------------------------------------------------------------- one tick

    def tick(self) -> ScreenState:
        now = time.time()
        if not MAC:
            return ScreenState(status="unavailable", reason="not_macos", ts=now)
        if screen_locked():
            self._helper.stop()
            return ScreenState(status="unavailable", reason="locked", ts=now,
                               grants={"accessibility": ax_trusted(), "locked": True})
        wins = window_list()
        if not wins:
            return ScreenState(status="unavailable", reason="no_windows", ts=now)

        # Front to back. The document window is the first one that is showing a document, which
        # is not always the frontmost window: the deck sits beside Zoom, and the presenter can be
        # reading their notes. A browser window only counts as a document when its tab title says
        # so ("… - Google Slides"), otherwise a mail tab in front would hide the deck behind it.
        front, doc_win, any_doc = None, None, None
        titled = 0
        for w in wins:
            if w.get("kCGWindowLayer") != 0:
                continue
            owner = str(w.get("kCGWindowOwnerName") or "")
            if owner in SKIP_OWNERS:
                continue
            name = str(w.get("kCGWindowName") or "")
            if name:
                titled += 1
            if front is None:
                front = w
            app = APPS.get(bundle_of(int(w.get("kCGWindowOwnerPID") or 0)))
            if app is None:
                continue
            if any_doc is None:
                any_doc = w
            if doc_win is None and (app.kind != "browser" or WEB_SUFFIX.search(name)):
                doc_win = w
        doc_win = doc_win or any_doc
        if doc_win is None:
            reason = "no_document" if titled or front else "no_window_titles"
            return ScreenState(status="unavailable", reason=reason, ts=now,
                               app=str((front or {}).get("kCGWindowOwnerName") or "") or None,
                               grants=self._grants(titled))

        pid = int(doc_win.get("kCGWindowOwnerPID") or 0)
        app = APPS[bundle_of(pid)]
        st = ScreenState(status="ok", reason=None, ts=now, app=app.label, app_key=app.key,
                         kind=app.kind, focused=doc_win is front,
                         document=str(doc_win.get("kCGWindowName") or "") or None,
                         via="cg-title", grants=self._grants(titled))

        trusted = ax_trusted()
        if trusted:
            ax_title, ax_doc = ax_window(pid)
            if ax_doc:
                st.path, st.document, st.via = ax_doc, pathlib.PurePath(ax_doc).name, "ax-document"
            if ax_title:
                m = PAGE_IN_TITLE.match(ax_title)
                if m:
                    st.page = int(m.group("page"))
                    st.pages = int(m.group("pages")) if m.group("pages") else None
                    st.via = "ax-title"
                    st.document = st.document or m.group("doc") or None
                elif not st.document:
                    st.document = ax_title
        elif st.via == "cg-title":
            st.reason = "no_accessibility"

        if app.script and self.detail and now - self._denied.get(app.key, 0.0) > DENIED_S:
            self._detail(st, app)

        if app.kind == "browser":
            self._web(st)
        self._resolve(st)
        return st

    def _grants(self, titled: int) -> dict:
        return {"accessibility": ax_trusted(), "window_titles": bool(titled), "locked": False}

    def _detail(self, st: ScreenState, app: App) -> None:
        got = self._helper.ask(app.script)
        err = got.get("err")
        if err:
            # -1743 is "not authorised to send Apple Events": stop asking for a minute so a
            # refused grant costs one prompt, not one a second.
            self._denied[app.key] = time.time()
            log.debug("screen detail %s: %s", app.key, err)
            return
        if got.get("doc"):
            st.document = str(got["doc"])
        if got.get("path"):
            st.path = _unfile(str(got["path"])) or st.path
        for src, dst in (("page", "page"), ("pages", "pages")):
            v = got.get(src)
            if isinstance(v, (int, float)) and v > 0:
                setattr(st, dst, int(v))
        for key in ("sheet", "cell", "url", "title"):
            if got.get(key):
                setattr(st, key, str(got[key])[:120])
        if got.get("playing"):
            st.presenting = True
        if any(got.get(k) for k in ("page", "sheet", "url")):
            st.via = "applescript"

    def _web(self, st: ScreenState) -> None:
        """A browser tab: the web app from the title's suffix, the slide from the fragment."""
        name = st.document or ""
        m = WEB_SUFFIX.search(name)
        if m:
            st.kind = WEB_KIND.get(m.group(1).lower(), "doc")
            st.document = name[:m.start()].strip() or None
        url = st.url or ""
        if st.kind == "deck":
            f = SLIDE_FRAGMENT.search(url)
            if f:
                st.title = st.title or f"slide id {f.group('id')}"
                # Google numbers the slides of a deck it created p, p1, p2 ...; a deck imported
                # from pptx gets opaque g-ids and there is no index in the URL at all.
                num = re.fullmatch(r"p(\d*)", f.group("id"))
                if num:
                    st.page = int(num.group(1) or 0) + 1
            st.presenting = "/present" in url
        elif st.kind == "sheet":
            g = GID_FRAGMENT.search(url)
            if g:
                st.sheet = st.sheet or f"gid {g.group('gid')}"

    # -------------------------------------------------------------- the index

    def _candidates(self) -> list[str]:
        version = getattr(self.index, "version", None)
        if self._files and version == self._files_version:
            return self._files
        files: list[str] = []
        con = getattr(self.index, "con", None)
        if con is not None:
            try:
                files = [r[0] for r in con.execute("SELECT path FROM _files") if r[0]]
            except Exception as e:  # noqa: BLE001
                log.debug("screen file list: %s", e)
        self._files, self._files_version = files, version
        return files

    def _resolve(self, st: ScreenState) -> None:
        """Name the indexed file and the chunk citation the brain would join on."""
        if st.page:
            st.loc = f"{'slide' if st.kind == 'deck' else 'page'} {st.page}"
        files = self._candidates()
        if not files or not (st.document or st.path):
            return
        if st.path:
            base = pathlib.PurePath(st.path).name
            if base in files:
                st.file, st.file_score = base, 1.0
        if not st.file:
            st.file, st.file_score = match_file(st.document or st.path, files)
        if st.file and st.loc and not st.title:
            st.title = self._chunk_title(st.file, st.loc)

    def _chunk_title(self, file: str, loc: str) -> str | None:
        """The slide's own title, out of the chunk citation the indexer already wrote
        ("slide 4 'Risks and asks'"). Free: the chunk list is in memory."""
        chunks = getattr(self.index, "chunks", None) or []
        head = loc + " "
        for ch in chunks:
            if getattr(ch, "file", None) != file:
                continue
            cloc = str(getattr(ch, "loc", ""))
            if cloc == loc or cloc.startswith(head):
                m = re.search(r"'(?P<t>[^']+)'", cloc)
                return m.group("t") if m else (getattr(ch, "heading", "") or None)
        return None


# --------------------------------------------------------------------------- module singleton

_watcher: ScreenWatcher | None = None


def start(index: Any = None, interval: float = INTERVAL_S, detail: bool = True) -> ScreenWatcher:
    """Start (or re-point) the one poller this process runs."""
    global _watcher
    if _watcher is None:
        _watcher = ScreenWatcher(index, interval=interval, detail=detail)
        _watcher.start()
    else:
        _watcher.index = index if index is not None else _watcher.index
    return _watcher


def stop() -> None:
    global _watcher
    if _watcher is not None:
        _watcher.stop()
        _watcher = None


def watcher() -> ScreenWatcher | None:
    return _watcher


def snapshot() -> dict:
    """The current screen state, or the not-started stub. Never raises, never blocks."""
    return _watcher.snapshot() if _watcher is not None else ScreenState().msg()


def probe_once(index: Any = None) -> dict:
    """One snapshot without a thread: for the CLI, a smoke test, or a caller that wants a
    single reading and no background cost."""
    w = ScreenWatcher(index)
    try:
        st = w.tick()
        st.tick_ms = 0.0
        return st.msg()
    finally:
        w.stop()


# --------------------------------------------------------------------------- CLI / fixture

def selftest() -> int:
    """The parts that are pure functions of a string, pinned against real captures.

    Every title and URL below was read off this Mac on 2026-09-23, so a macOS update that
    changes Preview's title or a Google URL shape fails here rather than in a call.
    """
    cases: list[tuple[str, Any, Any]] = []

    def eq(name: str, got: Any, want: Any) -> None:
        cases.append((name, got, want))

    # Preview / Skim / PDF Expert put the page in the AX title; CoreGraphics shows only the name.
    m = PAGE_IN_TITLE.match("Plan_Memo.pdf – Page 1 of 4")
    eq("preview page", (m.group("doc"), m.group("page"), m.group("pages")) if m else None,
       ("Plan_Memo.pdf", "1", "4"))
    eq("preview no page", PAGE_IN_TITLE.match("Strategy_Deck"), None)
    m2 = PAGE_IN_TITLE.match("Report.pdf - Page 12")
    eq("page no total", (m2.group("page"), m2.group("pages")) if m2 else None, ("12", None))

    # name -> indexed file
    files = ["Strategy_Deck.pptx", "Acme_Financials.xlsx", "Plan_Model (1).xlsx",
             "Plan_Memo.pdf", "Board_Memo.docx"]
    eq("exact", match_file("Strategy_Deck.pptx", files)[0], "Strategy_Deck.pptx")
    eq("no extension", match_file("Strategy_Deck", files)[0], "Strategy_Deck.pptx")
    eq("copy suffix", match_file("Plan_Model.xlsx", files)[0], "Plan_Model (1).xlsx")
    eq("full path", match_file("/Users/x/Plan_Memo.pdf", files)[0],
       "Plan_Memo.pdf")
    eq("unrelated", match_file("Vendor Data Inventory", files)[0], None)
    eq("empty", match_file("", files)[0], None)
    eq("no index", match_file("Strategy_Deck.pptx", [])[0], None)

    # a browser tab
    w = ScreenWatcher()
    st = ScreenState(document="Platform Options v2.pptx - Google Slides",
                     url="https://docs.google.com/presentation/d/AAA/edit?slide=id.g3fb_2_88#slide=id.g3fb_2_88",
                     kind="browser")
    w._web(st)                                                     # noqa: SLF001
    eq("slides kind", st.kind, "deck")
    eq("slides doc", st.document, "Platform Options v2.pptx")
    eq("slides id", st.title, "slide id g3fb_2_88")
    eq("slides has no index", st.page, None)     # Google's object ids are not slide numbers
    st2 = ScreenState(document="Vendor Data Inventory - Google Sheets", kind="browser",
                      url="https://docs.google.com/spreadsheets/d/AAA/edit?gid=79#gid=796678265")
    w._web(st2)                                                    # noqa: SLF001
    eq("sheets kind", st2.kind, "sheet")
    eq("sheets gid", st2.sheet, "gid 796678265")
    st3 = ScreenState(document="Deck - Google Slides", kind="browser",
                      url="https://docs.google.com/presentation/d/AAA/present#slide=id.p3")
    w._web(st3)                                                    # noqa: SLF001
    eq("present flag", st3.presenting, True)
    eq("p-id index", st3.page, 4)

    # the degraded paths are state, not exceptions
    eq("snapshot with no watcher", ScreenState().msg()["status"], "unavailable")
    eq("reason with no watcher", ScreenState().msg()["reason"], "not_started")

    bad = 0
    for name, got, want in cases:
        if got != want:
            bad += 1
            print(f"  FAIL {name}: {got!r} != {want!r}")
    print(f"{len(cases) - bad}/{len(cases)} cases pass")
    return 1 if bad else 0


def _cli(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="presage.screen", description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="the string parsers, against real captures")
    ap.add_argument("--once", action="store_true", help="one snapshot as JSON")
    ap.add_argument("--watch", nargs="?", const=20, type=float, metavar="SECONDS",
                    help="print every change for N seconds, then the CPU cost")
    ap.add_argument("--walk", nargs="?", const="auto", metavar="FILE",
                    help="the E43 fixture: open a document, step it, time every step")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--app", default="", help="with --walk: preview | powerpoint | keynote")
    ap.add_argument("--context", default="", help="index this folder first, to resolve the file")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    index = None
    if args.context:
        from presage.ingest import Index
        index = Index()
        index.build(pathlib.Path(args.context).expanduser())

    if args.walk:
        return walk(args.walk, app=args.app, steps=args.steps, index=index)
    if args.watch:
        w = ScreenWatcher(index).start()
        shown = ("status", "reason", "app", "document", "page", "pages", "sheet", "file", "loc", "via")
        last = None
        end = time.time() + float(args.watch)
        try:
            while time.time() < end:
                s = w.snapshot()
                key = (s["status"], s["reason"], s["app_key"], s["document"], s["page"], s["sheet"])
                if key != last:
                    last = key
                    line = json.dumps({k: s[k] for k in shown}, default=str)
                    print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            w.stop()
        print("\n" + cpu_line(w))
        return 0

    print(json.dumps(probe_once(index), indent=2, default=str))
    return 0


# Page Down everywhere. The arrow keys also step a deck, but in PowerPoint's editor they move
# a selected shape instead, which would edit the file the walk opened.
def cpu_line(w: "ScreenWatcher") -> str:
    """What the poller cost, counting the osascript helper's CPU as ours, because it is."""
    if not w.ticks:
        return "no ticks"
    own, kid = w.cpu_s / w.ticks * 1000, w.child_cpu_s / w.ticks * 1000
    pct = (w.cpu_s + w.child_cpu_s) / (w.ticks * w.interval) * 100
    return (f"{w.ticks} ticks at {w.interval:g} s: {own:.2f} ms CPU here + {kid:.2f} ms in the "
            f"helper = {pct:.2f}% of a core")


WALK_KEY = {"preview": 121, "powerpoint": 121, "keynote": 121}


def walk(target: str = "auto", app: str = "", steps: int = 6, index: Any = None) -> int:
    """Open a document, step through it, and report how long the poller took to see each step.

    This is the E43 measurement and the fixture Jackson can re-run in one command:

        uv run python -m presage.screen --walk
        uv run python -m presage.screen --walk sample_context/Strategy_Deck.pptx --app powerpoint

    It sends key events through System Events, which needs the Accessibility grant; without it
    the walk still opens the document and reports what the poller can see standing still.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    if target in ("auto", ""):
        target = str(root / "context" / "Plan_Memo.pdf")
    path = pathlib.Path(target).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        print(f"no such file: {path}")
        return 2
    app = app or ("powerpoint" if path.suffix.lower() in (".pptx", ".ppt")
                  else "keynote" if path.suffix.lower() == ".key" else "preview")
    opener = {"preview": "Preview", "powerpoint": "Microsoft PowerPoint", "keynote": "Keynote"}[app]

    # A copy, never the original: the brief's rule for the big test folder, and the same rule keeps
    # a stray key event from editing the sample deck.
    import shutil
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="e43-")) / path.name
    shutil.copy2(path, tmp)

    print(f"opening {tmp.name} in {opener}")
    subprocess.run(["/usr/bin/open", "-a", opener, str(tmp)], check=False)
    w = ScreenWatcher(index, interval=INTERVAL_S).start()
    deadline = time.time() + 40
    while time.time() < deadline:
        s = w.snapshot()
        if s["status"] == "ok" and s["app_key"] == app and s["document"]:
            break
        time.sleep(0.25)
    ready = w.snapshot()
    print(f"poller sees: {ready['app']} / {ready['document']} / page {ready['page']} "
          f"via {ready['via']} (file {ready['file']!r} @ {ready['file_score']})")
    if ready["status"] != "ok":
        print(f"not measurable: {ready['reason']}")
        w.stop()
        return 1

    code = WALK_KEY[app]
    rows: list[dict] = []
    for i in range(steps):
        before = w.snapshot()
        saw, seen_at, t0, presses = None, None, 0.0, 0
        # Preview scrolls continuously, so one Page Down can move less than a page. A key that
        # did not move the document is a fixture artefact, not a poller miss: press again and
        # time from the press that actually turned the page.
        while presses < 4 and saw is None:
            t0 = time.time()
            presses += 1
            subprocess.run(["/usr/bin/osascript", "-e",
                            f'tell application "{opener}" to activate',
                            "-e", f'tell application "System Events" to key code {code}'],
                           capture_output=True)
            while time.time() - t0 < 2.5:
                s = w.snapshot()
                if s.get("page") and s["page"] != before.get("page"):
                    saw, seen_at = s, time.time()
                    break
                time.sleep(0.02)
        rows.append({"step": i + 1, "from": before.get("page"), "to": (saw or {}).get("page"),
                     "ms": round((seen_at - t0) * 1000) if seen_at else None, "presses": presses,
                     "loc": (saw or {}).get("loc"), "title": (saw or {}).get("title")})
        print(f"  step {i + 1}: {rows[-1]['from']} -> {rows[-1]['to']} in "
              f"{rows[-1]['ms']} ms ({presses} key{'s' if presses > 1 else ''})   "
              f"{rows[-1]['loc'] or ''} {rows[-1]['title'] or ''}")
        if saw and saw.get("pages") and saw["page"] >= saw["pages"]:
            print(f"  (last page of {saw['pages']})")
            break
        time.sleep(0.4)

    got = [r["ms"] for r in rows if r["ms"] is not None]
    got.sort()
    print(f"\n{len(got)}/{len(rows)} steps seen. "
          + (f"median {got[len(got) // 2]} ms, max {got[-1]} ms, "
             f"under 1500 ms: {sum(1 for m in got if m <= 1500)}/{len(got)}" if got else "none seen"))
    print(cpu_line(w))
    w.stop()
    out = {"app": app, "document": tmp.name, "rows": rows, "ticks": w.ticks,
           "cpu_ms_per_tick": round(w.cpu_s / max(1, w.ticks) * 1000, 3),
           "child_cpu_ms_per_tick": round(w.child_cpu_s / max(1, w.ticks) * 1000, 3)}
    dest = pathlib.Path(os.environ.get("E43_OUT") or (tmp.parent / "walk.json"))
    dest.write_text(json.dumps(out, indent=2))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=os.environ.get("PRESAGE_LOG", "WARNING"))
    raise SystemExit(_cli(sys.argv[1:]))
