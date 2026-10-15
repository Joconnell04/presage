"""Stub server for the presage UI.

Serves ui/ at / and replays a scripted fake call over /ws so the real client
path (WebSocket, hello, reconnect, ask/set/clear/show) is exercised with no
capture, STT or index behind it. The real server implements the same protocol;
see docs/ui-protocol.md.

    python -m presage.server_stub          # 0.0.0.0:8765

A real call instead of the script: PRESAGE_STUB_REPLAY names a simulator recording
(docs/experiments/sim/*/frames.jsonl) and the stub replays its frames at their recorded offsets.
PRESAGE_STUB_AT=90 (or ?at=90 on the page URL) lands everything before 90 s at once, so a
screenshot starts mid-call (?rec=big-02-model switches to another recording in
docs/experiments/sim), and
PRESAGE_STUB_SPEED=4 (or ?speed=4) plays the rest four times faster. Asks and chip clicks still get the
stub's canned replies.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"

HOST = "0.0.0.0"
PORT = int(os.environ.get("PRESAGE_PORT", "8765"))
LOOP_S = 62.0
REPLAY = os.environ.get("PRESAGE_STUB_REPLAY") or None
REPLAY_AT = float(os.environ.get("PRESAGE_STUB_AT") or 0)
REPLAY_SPEED = max(0.1, float(os.environ.get("PRESAGE_STUB_SPEED") or 1))

F_FIN = "Acme_Financials.xlsx"
F_DECK = "Strategy_Deck.pptx"
F_BOARD = "Board_Notes_Q3.docx"
F_COH = "Cohort_Retention.csv"
F_PRICE = "Pricing_Model.xlsx"
F_ORG = "Org_Plan.pptx"
F_MKT = "Market_Sizing.xlsx"

MODES = [
    {"id": "call", "label": "Call", "hint": "Answers from your files appear here."},
    {"id": "sales", "label": "Sales", "hint": "Objection handling and talk tracks appear here."},
    {"id": "interview", "label": "Interview", "hint": "Suggested answers to their questions appear here."},
    {"id": "pitch", "label": "Pitch", "hint": "Answers and tough-question responses appear here."},
    {"id": "notes", "label": "Notes", "hint": "Notes build as the call goes on."},
    {"id": "quiet", "label": "Quiet", "hint": "Nothing automatic. Ask a question below."},
]

DEFAULT_SETTINGS = {
    "mode": "call",
    "web_mode": "auto",
    "topic_threshold": 0.35,
    "question_threshold": 0.5,
    "prefetch_cooldown_s": 8,
    "max_warm": 6,
    "answer_ttl_s": 90,
}

SETTING_RANGE = {
    "topic_threshold": (0.0, 1.0),
    "question_threshold": (0.0, 1.0),
    "prefetch_cooldown_s": (0.0, 30.0),
    "max_warm": (1.0, 12.0),
    "answer_ttl_s": (10.0, 600.0),
}


# The control surface, faked. Same shapes as presage/server.py; see docs/ui-protocol.md.
AUDIO = {"state": "running", "mic_state": "ok", "mic_device": "MacBook Pro Microphone",
         "mic_uid": "BuiltInMicrophoneDevice", "output_device": "MacBook Pro Speakers",
         "error": None}
MODELS = {"answer": "claude-sonnet-5", "hedge": "claude-haiku-4-5-20251001",
          "classifier": "claude-haiku-4-5-20251001"}
LLM = {"provider": "anthropic", "answer_model": MODELS["answer"], "models": dict(MODELS),
       "defaults": dict(MODELS), "key_set": True, "key_hint": "…4Q7B", "key_ok": True,
       "key_message": "Key OK",
       "decider": {"kind": "haiku", "choice": "auto", "ok": True, "key_set": False, "key_hint": "",
                   "message": "no key, using Haiku", "latency_ms": None, "laya_available": True,
                   "intent": {"kind": "local", "choice": "auto", "local_available": True,
                              "dir": "presage-gate-multi", "why_not": None}},
       "live_choice": "same", "live_provider": "anthropic",
       "login": {"ok": True, "account": "you@example.com", "message": "Signed in as you@example.com"},
       "hint": "Run `claude` once in Terminal and log in with your Claude account"}
FILES = [
    {"name": "Acme_Financials.xlsx", "kind": "xlsx", "size": 48123, "mtime": 0,
     "indexed": True, "error": None, "tables": 5, "chunks": 0},
    {"name": "Board_Memo.docx", "kind": "docx", "size": 21004, "mtime": 0,
     "indexed": True, "error": None, "tables": 0, "chunks": 14},
    {"name": "Cohort_Retention.csv", "kind": "csv", "size": 9120, "mtime": 0,
     "indexed": True, "error": None, "tables": 1, "chunks": 0},
    {"name": "Market_Report.pdf", "kind": "pdf", "size": 883121, "mtime": 0,
     "indexed": True, "error": None, "tables": 0, "chunks": 31},
    {"name": "Strategy_Deck.pptx", "kind": "pptx", "size": 402881, "mtime": 0,
     "indexed": True, "error": None, "tables": 0, "chunks": 22},
    {"name": "notes.md", "kind": "md", "size": 1204, "mtime": 0,
     "indexed": True, "error": None, "tables": 0, "chunks": 3},
]
CONTEXT = {"dir": str(ROOT / "context"), "own_dir": True, "files": FILES, "files_count": len(FILES)}
DEVICES = [
    {"uid": "BuiltInMicrophoneDevice", "name": "MacBook Pro Microphone", "in": 1, "out": 0,
     "rate": 48000, "transport": "builtin", "default_input": True, "default_output": False},
    {"uid": "AirPodsPro:2c-9f", "name": "AirPods Pro", "in": 1, "out": 1, "rate": 24000,
     "transport": "bluetooth", "default_input": False, "default_output": False,
     "warning": "Bluetooth mics can drop out; prefer the built-in mic"},
    {"uid": "USB:MV7", "name": "Shure MV7", "in": 1, "out": 0, "rate": 48000,
     "transport": "usb", "default_input": False, "default_output": False},
]
PREFS = {"context_dir": None, "mic_uid": None, "llm_provider": "anthropic", "models": None,
         "tune": {}, "auto_start_audio": True}

CLIENTS: set = set()


async def broadcast(msg: dict) -> None:
    for c in list(CLIENTS):
        c.put(msg)


def context_msg() -> dict:
    CONTEXT["files_count"] = len(CONTEXT["files"])
    return {"type": "context", **CONTEXT}


def status(t: float, mic: float = 0.18, sysl: float = 0.06) -> dict:
    return {
        "type": "status",
        "now": t,
        "mic": AUDIO["state"] == "running",
        "system": AUDIO["state"] == "running",
        "stt": "local:parakeet",
        "index": {"files": 7, "tables": 5, "chunks": 120, "deep": dict(DEEP)},
        "listening": AUDIO["state"] == "running",
        "me_speaking": SPEAKING["on"],
        "levels": {"mic": round(mic, 3), "system": round(sysl, 3)},
        "audio": dict(AUDIO),
        "llm": dict(LLM),
        "context": {k: CONTEXT[k] for k in ("dir", "own_dir", "files_count")},
    }


def line(i: str, src: str, text: str, t0: float, t1: float, final: bool = True,
         retracts: bool = False, unstable: str = "", speaker: str | None = None) -> dict:
    """utt is the per-source utterance number; the real server keys the line id on (source, utt)
    so an eager final and the continuation that retracts it land on the same row. `ts` is the
    server's wall clock at send time and `unstable` the tail an interim may still rewrite, as the
    real server sends them; `speaker` is the far-side label."""
    utt = int("".join(c for c in i if c.isdigit()) or 0)
    msg = {"type": "transcript", "id": i, "source": src, "text": text,
           "final": final, "t0": t0, "t1": t1, "ts": t1, "utt": utt, "retracts": retracts}
    if unstable:
        msg["unstable"] = unstable
    if speaker and src == "system":
        msg["speaker"] = speaker
    return msg


def answer(i, q, a, fig, cites, source, latency, in_files, ts, web=None, answered_from=None,
           say=None, cached=False, working=None, verdict=None) -> dict:
    """web is [(title, url)]; answered_from defaults from in_files for older-style calls.
    `working` is the arithmetic behind a derived figure and renders under the say-line."""
    return {"type": "answer", "id": i, "question": q, "answer": a, "figure": fig,
            "citations": [{"file": f, "loc": l} for f, l in cites],
            "source": source, "latency_ms": latency, "in_files": in_files, "ts": ts,
            "answered_from": answered_from or ("files" if in_files else "none"),
            "web_citations": [{"title": t, "url": u} for t, u in (web or [])],
            "say": say, "cached": cached, "working": working,
            "derived": answered_from == "derived",
            "mode": "sales" if answered_from == "model" else "call",
            **({"verdict": verdict} if verdict else {})}


# --- scratch paper: the fee lever, its outputs, and the rows the page lists -----------------
FEE_BASE, FEE_Y5 = 0.0085, 8_148_348.58        # Pricing_Model.xlsx › Fee Model!B48 and G57
SCRATCH: list[dict] = []
_RATE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")


def money(v: float) -> str:
    return f"${v / 1e6:.2f}M" if v >= 1e6 else f"${v:,.0f}"


def whatif(rate: float, ts: float) -> tuple[dict, dict]:
    """One what-if on the GMV fee: the derived answer card and the scratch frame after it."""
    y5 = FEE_Y5 * rate / FEE_BASE
    pretty = f"{rate * 100:g}%"
    a = answer(f"w{pretty}", f"what if we charged {pretty} of gmv",
               f"At your {pretty}, year 5 revenue is {money(y5)}, "
               f"{money(y5 - FEE_Y5)} above the 0.85% in the model.",
               money(y5), [(F_PRICE, "Fee Model › B48"), (F_PRICE, "Summary Scenarios › G57")],
               "prefetch" if abs(rate - 0.01) < 1e-9 else "live",
               140 if abs(rate - 0.01) < 1e-9 else 2180, True, ts,
               answered_from="derived", say=f"{money(y5)} at {pretty}",
               working=f"Fee Model B48 0.85% → {pretty}, model recalculated",
               cached=abs(rate - 0.01) < 1e-9)
    item = {"label": "Platform fee", "value": pretty, "output": f"Year 5 revenue {money(y5)}",
            "working": a["working"], "ts": ts, "aid": a["id"]}
    SCRATCH[:] = [s for s in SCRATCH if s["value"] != pretty][-4:] + [item]
    return a, {"type": "scratch", "items": list(SCRATCH)}


SPEAKING = {"on": False}          # stub-wide: the script toggles it so holding can be seen
DEEP = {"state": "idle", "done": 0, "total": 40, "qa": 0, "summaries": 0,
        "chunks_contextualized": 0, "numbers": 0, "definitions": 0, "error": None,
        "started": None, "finished": None, "enabled": True}


def deep(state: str, done: int, qa: int):
    """A step of the background deep pass; the page reads it off status.index.deep."""
    def build(t: float) -> dict:
        DEEP.update(state=state, done=done, qa=qa, summaries=min(done, 12),
                    started=DEEP["started"] or t,
                    finished=t if state in ("ready", "error") else None)
        return status(t)
    return build


def speaking(on: bool):
    def build(t: float) -> dict:
        SPEAKING["on"] = on
        return status(t)
    return build


def notes(ts: float, n: int, summary, actions, questions) -> dict:
    return {"type": "notes", "summary": summary, "actions": actions, "questions": questions,
            "ts": ts, "utterances": n}


def question(aid: str, text: str, who: str, line_id: str | None, status: str, ts: float,
             via: str = "heuristic", policy: str = "answer") -> dict:
    """E68: the brain registered a question before any answer exists. `aid` is the card that will
    answer it; the frame is re-sent with the same id whenever `status` moves."""
    return {"type": "question", "id": "q" + aid[1:], "aid": aid, "text": text, "who": who,
            "source": "mic" if who == "Me" else "system", "line": line_id, "kind": "question",
            "policy": policy, "via": via, "sure": True, "status": status, "ts": ts}


def cand(i, topic, st, preview, ts, question=None, figure=None) -> dict:
    """`topic` is what the real brain sends: up to three workbook labels joined with " · ", some
    of them long. `question` is its guess at the question a ready chip answers."""
    return {"type": "candidate", "id": i, "topic": topic, "status": st,
            "preview": preview, "ts": ts, "question": question, "figure": figure}


SCREEN = {"status": "ok", "reason": None, "app": "PowerPoint", "app_key": "powerpoint", "kind": "deck",
          "document": "Strategy_Deck.pptx", "page": 4, "pages": 12, "sheet": None, "cell": None,
          "title": "FY26 plan by region", "presenting": False, "file": "Strategy_Deck.pptx",
          "file_score": 1.0, "loc": "slide 4", "via": "applescript", "focused": False, "url": None,
          "age": 0.3, "running": True, "interval_s": 1.0}


def hello(t: float, settings: dict) -> dict:
    return {
        "type": "hello",
        "now": t,
        "settings": dict(settings),
        "status": status(t),
        "audio": dict(AUDIO),
        "llm": dict(LLM),
        "context": {k: v for k, v in context_msg().items() if k != "type"},
        "devices": list(DEVICES),
        "prefs": dict(PREFS),
        "screen": {} if REPLAY else dict(SCREEN, ts=t),
        "questions": [],
        "transcript": [
            line("h1", "system", "Right, thanks for making the time. I think we said forty minutes.", t - 96, t - 92, speaker="A"),
            line("h2", "mic", "Forty is fine. I've got the model and the board pack open.", t - 91, t - 87),
            line("h3", "system", "Good. Let's start with where the quarter landed, then the plan.", t - 86, t - 82, speaker="A"),
            line("h4", "mic", "Sure.", t - 81, t - 80),
            line("h5", "system", "How many net new logos did we close in Q3?", t - 70, t - 66, speaker="B"),
            line("h6", "mic", "Thirty-one, and the mix shifted up-market.", t - 63, t - 60),
            line("h7", "system", "That's better than I had in my head. And churn?", t - 52, t - 48, speaker="A"),
            line("h8", "mic", "Logo churn held at two point one percent.", t - 45, t - 42),
            line("h9", "system", "Do we have a read on the Latin America pilot anywhere?", t - 40, t - 36, speaker="B"),
            line("h10", "mic", "Not in what I've loaded. I'll follow up after the call.", t - 33, t - 30),
            line("h11", "system", "Fine. Remind me where ARR ended the quarter?", t - 24, t - 21, speaker="A"),
            line("h12", "mic", "Eighteen point six, up nine on Q2.", t - 17, t - 15),
            line("h13", "system", "Good. So before we get into the plan for next year…", t - 14, t - 11, speaker="A"),
        ],
        "scratch": list(SCRATCH) or [
            {"label": "Platform fee", "value": "1%", "output": "Year 5 revenue $9.59M",
             "working": "Fee Model B48 0.85% → 1%, model recalculated", "ts": t - 6, "aid": "a3d"},
        ],
        "answers": [
            # recalculated on the client's own model rather than read out of a cell
            answer("a3d", "what if we charged 1% of gmv",
                   "At your 1%, year 5 revenue is $9.59M, $1.44M above the 0.85% in the model.",
                   "$9.59M", [(F_PRICE, "Fee Model › B48"), (F_PRICE, "Summary Scenarios › G57")],
                   "live", 1840, True, t - 6, answered_from="derived",
                   say="$9.59M at one percent",
                   working="Fee Model B48 0.85% → 1%, model recalculated"),
            # answered from the web after the files came up empty: in_files stays False for
            # older clients, answered_from/web_citations are what the current page reads.
            answer("a3w", "what are competitors charging for the enterprise tier",
                   "Public list prices run $40-60 per seat; Acme's enterprise tier is $48.",
                   "$40-60", [(F_PRICE, "Tiers › C7")], "live", 2600, False, t - 12,
                   web=[("Notion pricing", "https://www.notion.com/pricing"),
                        ("Figma enterprise plans", "https://www.figma.com/pricing/"),
                        ("", "https://slack.com/pricing")],
                   answered_from="web", say="Market is $40-60, we are $48"),
            answer("a3b", "remind me where arr ended the quarter",
                   "ARR closed Q3 at $18.6M, up 9% QoQ and 2% ahead of plan.",
                   "$18.6M", [(F_FIN, "ARR › D5"), (F_BOARD, "p. 2")], "prefetch", 730, True, t - 21,
                   say="Up 9% on Q2, ahead of plan"),
            answer("a3", "do we have a read on the latin america pilot",
                   "No LatAm pilot data in the loaded files. Closest is the EMEA pilot readout.",
                   None, [], "live", 1900, False, t - 36),
            answer("a2", "and churn",
                   "Q3 logo churn was 2.1%, flat on Q2 and inside the 2.5% board tolerance.",
                   "2.1%", [(F_COH, "Q3 › D18"), (F_BOARD, "p. 3")], "prefetch", 620, True, t - 48,
                   say="Flat on Q2, inside tolerance"),
            answer("a1", "how many net new logos did we close in q3",
                   "31 net new logos in Q3, with 9 above $100k ACV.",
                   "31", [(F_FIN, "Bookings › C24")], "prefetch", 710, True, t - 66),
        ],
        "candidates": [
            cand("c1", "Next-year plan · FY26 revenue target by segment, base case · Breakeven quarter",
                 "warming", "", t - 12),
            cand("c2", "Q3 revenue · Revenue, quarterly, FY25 (recognised, excluding one-off services) · Plan variance",
                 "ready", "Q3 FY25 revenue $4.2M, +8% QoQ", t - 9, "What did revenue look like in Q3?", "$4.2M"),
        ],
        "modes": MODES,
        "notes": notes(t - 8, 24,
                       ["Q3 landed at $4.2M, 3% over plan",
                        "Enterprise mix up 5 points on upsell",
                        "Churn flat at 2.1%, inside board tolerance"],
                       ["Send CAC payback by end of day",
                        "Circulate the updated board pack"],
                       ["LatAm pilot read", "FY26 hiring phasing"]),
    }


# (seconds into the loop, builder(now) -> message)
SCRIPT = [
    (0.4, lambda t: line("m1", "system", "", t, t + 1, final=False, unstable="…I want to close", speaker="A")),
    (0.9, lambda t: line("m1", "system", "I want to close out", t - 0.5, t + 0.5, final=False,
                         unstable="the quarter", speaker="A")),
    (1.4, lambda t: line("m1", "system", "I want to close out the quarter properly.", t - 1, t + 0.4, speaker="A")),
    (1.0, deep("running", 6, 14)),
    (2.2, lambda t: cand("c3", "Enterprise mix", "warming", "", t)),
    (3.0, lambda t: line("m2", "system", "So what did", t, t + 1, final=False, unstable="revenue", speaker="B")),
    (4.2, lambda t: line("m2", "system", "So what did revenue actually look like in Q3?", t - 1.2, t + 0.3, speaker="B")),
    (4.25, lambda t: question("a4", "So what did revenue actually look like in Q3?", "Them", "m2", "heard", t)),
    (4.9, lambda t: answer("a4", "so what did revenue actually look like in q3",
                           "Q3 FY25 revenue was $4.2M, up 8% QoQ and 3% over plan.", "$4.2M",
                           [(F_FIN, "P&L › B12"), (F_DECK, "slide 4")], "prefetch", 120, True, t,
                           say="Eight percent up, three over plan", cached=True)),
    (4.95, lambda t: question("a4", "So what did revenue actually look like in Q3?", "Them", "m2", "answered", t)),
    (5.2, lambda t: cand("c2", "Q3 revenue", "expired", "", t)),
    (6.0, lambda t: line("m3", "mic", "Four point two, eight percent up on Q2 and three over plan.", t - 1.4, t)),
    (6.0, deep("running", 18, 44)),
    (8.0, lambda t: cand("c4", "Net revenue retention", "warming", "", t)),
    (9.0, lambda t: line("m4", "system", "And how much of that came from the enterprise tier?", t - 2.6, t)),
    (10.2, lambda t: cand("c3", "Enterprise mix · Segment revenue by tier (enterprise, mid-market, SMB), Q3 FY25 · Upsell count",
                          "ready", "Enterprise 61% of Q3 revenue, +5 pts", t,
                          "How much of Q3 revenue came from enterprise?", "61%")),
    (11.4, lambda t: answer("a5", "how much of that came from the enterprise tier",
                            "Enterprise was 61% of Q3 revenue, up 5 points, driven by four upsells.", "61%",
                            [(F_FIN, "Segments › F9")], "live", 2140, True, t,
                            say="Sixty-one percent, up five points",
                            verdict={"ok": True, "level": "unverified", "problems": [],
                                     "reason": "No cited cell holds 61%"})),
    (12.0, lambda t: cand("c3", "Enterprise mix", "expired", "", t)),
    (13.5, lambda t: cand("c4", "Net revenue retention", "ready", "NRR 112%, up from 106% in Q2", t)),
    (11.0, deep("running", 31, 78)),
    (14.5, deep("ready", 40, 96)),
    (14.5, lambda t: cand("c5", "Gross margin", "warming", "", t)),
    (15.5, lambda t: line("m5", "mic", "Mostly upsell, not new logos.", t - 2, t)),
    (17.5, lambda t: line("m6", "system", "So what's net revenue retention looking like now?", t - 2.8, t)),
    (18.2, lambda t: answer("a6", "what's net revenue retention looking like now",  # noqa: E501
                            "NRR is 112%, up from 106% in Q2, with expansion concentrated in the top decile.",
                            "112%", [(F_COH, "NRR › H7"), (F_BOARD, "p. 6")], "prefetch", 90, True, t,
                            say="NRR is 112, up six points", cached=True)),
    (18.8, lambda t: cand("c4", "Net revenue retention", "expired", "", t)),
    (20.0, lambda t: cand("c6", "Headcount plan", "warming", "", t)),
    (21.5, lambda t: cand("c5", "Gross margin", "ready", "Gross margin 68.4%, +120 bps QoQ", t)),
    (23.0, lambda t: line("m7", "system", "Do you know what our CAC payback is at the moment?", t - 3, t)),
    (23.05, lambda t: question("a7", "Do you know what our CAC payback is at the moment?", "Them", "m7", "heard", t)),
    (23.4, speaking(True)),
    # E68: nothing came back in time, so the slot says so, under the id the answer will use
    (24.2, lambda t: {**answer("a7", "do you know what our cac payback is at the moment",
                               "Couldn't answer: nothing came back in 6 s. Still looking.", None, [], "live",
                               6004, False, t, say="No answer yet"),
                      "no_answer": {"reason": "slow", "after_ms": 6004, "detail": ""}, "who": "Them"}),
    (24.25, lambda t: question("a7", "Do you know what our CAC payback is at the moment?", "Them", "m7", "no-answer", t)),
    (24.6, lambda t: answer("a7", "do you know what our cac payback is at the moment",
                            "CAC payback is not in the loaded files. Pricing_Model has blended CAC but no payback.",
                            None, [(F_PRICE, "CAC › B4")], "live", 2380, False, t)),
    (24.65, lambda t: question("a7", "Do you know what our CAC payback is at the moment?", "Them", "m7", "answered", t)),
    # the web escalation: same id, so the card is replaced in place and the page animates the swap
    (27.0, speaking(False)),
    (27.4, lambda t: answer("a7", "do you know what our cac payback is at the moment",
                            "Median B2B SaaS CAC payback is 18 months; Acme's blended CAC is $11.4k.",
                            "18 mo", [(F_PRICE, "CAC › B4")], "live", 3100, False, t,
                            web=[("SaaS Capital 2025 benchmarks", "https://www.saas-capital.com/benchmarks"),
                                 ("", "https://openviewpartners.com/benchmarks/")],
                            answered_from="web")),
    (26.5, lambda t: line("m8", "mic", "I don't have payback in front of me. I'll send it through today.", t - 3.4, t)),
    # notes update in place, and one answer the model suggested rather than found
    (28.0, lambda t: notes(t, 31,
                           ["Q3 landed at $4.2M, 3% over plan",
                            "Enterprise mix up 5 points on upsell",
                            "Churn flat at 2.1%, inside board tolerance",
                            "CAC payback not in the pack"],
                           ["Send CAC payback by end of day",
                            "Circulate the updated board pack",
                            "Confirm FY26 hiring phasing"],
                           ["LatAm pilot read", "Pricing review date"])),
    (29.2, lambda t: answer("a11", "they think we are too expensive",
                            "Anchor on the 3.1x pipeline coverage, then offer the annual plan at 12% off.",
                            None, [], "manual", 1500, False, t, answered_from="model")),
    (29.0, lambda t: cand("c7", "Pipeline coverage", "warming", "", t)),
    (30.0, lambda t: line("m9", "system", "Alright. And the margin", t - 2, t)),
    # the live answer streams before it settles: `partial` frames carry the growing sentence and the figure
    (30.2, lambda t: {**answer("a8", "and the margin picture", "Gross margin was 68.4%, up 120 bps", "68.4%",
                               [], "prefetch", None, True, t), "partial": True}),
    (30.45, lambda t: {**answer("a8", "and the margin picture",
                                "Gross margin was 68.4%, up 120 bps QoQ as hosting costs fell", "68.4%",
                                [], "prefetch", None, True, t), "partial": True}),
    (30.6, lambda t: line("m9", "system", "Alright. And the margin picture?", t - 2.6, t, retracts=True)),
    (30.7, lambda t: answer("a8", "and the margin picture",
                            "Gross margin was 68.4%, up 120 bps QoQ as hosting costs fell with the migration.",
                            "68.4%", [(F_FIN, "P&L › B31"), (F_DECK, "slide 11")], "prefetch", 680, True, t)),
    (31.3, lambda t: cand("c5", "Gross margin", "expired", "", t)),
    (33.0, lambda t: cand("c6", "Headcount plan · Opex by function, FY26 plan (R&D, sales, G&A), including contractors converting in H2 · Hiring phasing",
                          "ready", "FY26 plan adds 34 heads, 19 in R&D", t, "How many people does the FY26 plan add?", "34")),
    # the transcriber's words repaired from the workbook's own vocabulary (E46): `heard` keeps the original
    (34.0, lambda t: {**line("m10", "mic", "Hosting came down a lot after the migration, margin's about seventy.", t - 2.6, t),
                      "heard": "Hosting came down a lot after the my gration, margin's about seventy."}),
    # the consultant said a figure the files contradict: a correction, not an answer
    (34.1, lambda t: {**answer("s7e1", "said about seventy percent", "said about seventy percent; gross margin is 68.4%",
                               "68.4%", [(F_FIN, "P&L › B31")], "live", 1, True, t,
                               say="Margin is 68.4%, not 70",
                               working="said about seventy percent; Gross margin, Q3 is 68.4% in P&L B31"),
                      "answered_from": "correction",
                      "verdict": {"ok": True, "level": "spoken", "spoken": "contradicts",
                                  "said": "about seventy percent", "actual": "68.4%"}}),
    (36.0, lambda t: cand("c7", "Pipeline coverage", "ready", "Q4 pipeline coverage 3.1x against plan", t)),
    (37.0, lambda t: line("m11", "system", "Let's talk about what the plan assumes for hiring.", t - 3, t)),
    (41.6, lambda t: answer("a9", "what does the FY26 plan assume for hiring",
                            "FY26 adds 34 heads: 19 R&D, 9 sales, 6 G&A, weighted to the first half.", "34",
                            [(F_ORG, "slide 7"), (F_FIN, "Opex › C40")], "manual", 1460, True, t)),
    (43.5, lambda t: line("m12", "mic", "Thirty-four heads, most of it in R&D in the first half.", t - 3, t)),
    (46.0, lambda t: cand("c8", "TAM by segment", "warming", "", t)),
    (47.0, lambda t: line("m13", "system", "And how much pipeline do we have against that?", t - 2.6, t)),
    (47.7, lambda t: answer("a10", "how much pipeline do we have against that",
                            "Q4 pipeline coverage is 3.1x plan, versus 2.7x at the same point last year.",
                            "3.1x", [(F_FIN, "Pipeline › D7")], "prefetch", 760, True, t)),
    (48.3, lambda t: cand("c7", "Pipeline coverage", "expired", "", t)),
    # a slide on screen quotes a number the model has moved past: a drift card, not an answer
    (48.8, lambda t: {**answer("d19a", "slide 4 says $4.0M", "Slide 4 says $4.0M; the model's P&L!B12 is $4.2M",
                               "$4.2M", [(F_FIN, "P&L › B12"), (F_DECK, "slide 4")], "live", 4, True, t,
                               say="Model says $4.2M, not $4.0M",
                               working="Strategy_Deck.pptx › slide 4 prints $4.0M; Acme_Financials.xlsx › P&L!B12 "
                                       "held 4,012,000 when the deck was linked and holds 4,208,000 now"),
                      "answered_from": "drift", "verdict": {"ok": True, "level": "drift", "slide": "$4.0M"}}),
    (50.0, lambda t: cand("c8", "TAM by segment", "ready", "Serviceable market $1.8B, mid-market $740M", t)),
    (51.0, lambda t: line("m14", "mic", "Three point one times, which is ahead of where we were last year.", t - 3.4, t)),
    # the scratch-paper exchange: a what-if off the grid, then a follow-up that moves the same lever
    (44.4, lambda t: line("m12b", "system", "And what if we charged one percent of GMV instead of the 0.85?", t - 3, t)),
    (45.1, lambda t: whatif(0.01, t)[0]),
    (45.2, lambda t: whatif(0.01, t)[1]),
    (52.2, lambda t: line("m14c", "system", "And at 1.25%?", t - 1.4, t)),
    (52.9, lambda t: whatif(0.0125, t)[0]),
    (53.0, lambda t: whatif(0.0125, t)[1]),
    (54.0, lambda t: line("m15", "system", "Good. Let's park the rest for the board session.", t - 3, t)),
    (56.5, lambda t: cand("c9", "Board agenda", "warming", "", t)),
    (57.5, lambda t: line("m16", "mic", "Works for me. I'll circulate the updated pack beforehand.", t - 3, t)),
    # capture restarted (a mic switch), so the utterance counter did too and "m1" comes back for a
    # new line; the page must add a row, not overwrite the one from 58 s ago
    (59.0, lambda t: line("m1", "system", "Sorry, I switched headsets. Can you hear me?", t - 2, t, speaker="A")),
]

# What a {"type":"show","id":...} promotion returns for each warm candidate.
SHOW = {
    "c1": ("Next-year plan", "FY26 plan targets $21.4M revenue, 27% growth, breakeven in Q3.",
           "$21.4M", [(F_DECK, "slide 14")]),
    "c2": ("Q3 revenue", "Q3 FY25 revenue was $4.2M, up 8% QoQ and 3% over plan.",
           "$4.2M", [(F_FIN, "P&L › B12")]),
    "c3": ("Enterprise mix", "Enterprise was 61% of Q3 revenue, up 5 points, driven by four upsells.",
           "61%", [(F_FIN, "Segments › F9")]),
    "c4": ("Net revenue retention", "NRR is 112%, up from 106% in Q2.",
           "112%", [(F_COH, "NRR › H7")]),
    "c5": ("Gross margin", "Gross margin was 68.4%, up 120 bps QoQ.",
           "68.4%", [(F_FIN, "P&L › B31")]),
    "c6": ("Headcount plan", "FY26 adds 34 heads: 19 R&D, 9 sales, 6 G&A, weighted to the first half.",
           "34", [(F_ORG, "slide 7")]),
    "c7": ("Pipeline coverage", "Q4 pipeline coverage is 3.1x plan, versus 2.7x last year.",
           "3.1x", [(F_FIN, "Pipeline › D7")]),
    "c8": ("TAM by segment", "Serviceable market is $1.8B, of which mid-market is $740M.",
           "$1.8B", [(F_MKT, "Sizing › E11")]),
    "c9": ("Board agenda", "Board session covers the FY26 plan, pricing, and the EMEA hiring case.",
           None, [(F_BOARD, "p. 1")]),
}

# Canned replies for {"type":"ask"}; the fallback is the "not in files" card.
ASK = [
    (("revenue", "q3", "top line"),
     ("Q3 FY25 revenue was $4.2M, up 8% QoQ and 3% over plan.", "$4.2M",
      [(F_FIN, "P&L › B12"), (F_DECK, "slide 4")])),
    (("churn", "retention", "nrr"),
     ("NRR is 112% and logo churn is 2.1%, both better than plan.", "112%",
      [(F_COH, "NRR › H7")])),
    (("margin", "gross"),
     ("Gross margin was 68.4%, up 120 bps QoQ after the hosting migration.", "68.4%",
      [(F_FIN, "P&L › B31")])),
    (("head", "hiring", "org"),
     ("FY26 adds 34 heads: 19 R&D, 9 sales, 6 G&A, weighted to the first half.", "34",
      [(F_ORG, "slide 7")])),
    (("pipeline", "coverage", "q4"),
     ("Q4 pipeline coverage is 3.1x plan, versus 2.7x at the same point last year.", "3.1x",
      [(F_FIN, "Pipeline › D7")])),
    (("tam", "market", "sizing"),
     ("Serviceable market is $1.8B, of which mid-market is $740M.", "$1.8B",
      [(F_MKT, "Sizing › E11")])),
]


class Conn:
    """One browser. A queue keeps the replay and the reply path off each other."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.out: asyncio.Queue[dict] = asyncio.Queue()
        self.settings = dict(DEFAULT_SETTINGS)
        self.seq = 0

    def put(self, msg: dict) -> None:
        self.out.put_nowait(msg)

    def next_id(self, prefix: str) -> str:
        self.seq += 1
        return f"{prefix}{self.seq}"


async def writer(c: Conn) -> None:
    while True:
        msg = await c.out.get()
        await c.ws.send_text(json.dumps(msg))


async def replay(c: Conn) -> None:
    while True:
        start = time.time()
        c.put(hello(start, c.settings))
        for at, build in SCRIPT:
            await asyncio.sleep(max(0.0, start + at - time.time()))
            c.put(build(time.time()))
        await asyncio.sleep(max(0.0, start + LOOP_S - time.time()))


async def replay_file(c: Conn) -> None:
    """A simulator recording, frame by frame: the hello first, the rest at their recorded offsets
    from it. Status frames carry the recording's own `now`, so ages on the page stay right at any
    speed."""
    path = Path(REPLAY)
    rec = c.ws.query_params.get("rec") or ""        # ?rec=big-02-model picks another recording
    if re.fullmatch(r"[a-z0-9-]+", rec) and (ROOT / "docs/experiments/sim" / rec / "frames.jsonl").is_file():
        path = ROOT / "docs/experiments/sim" / rec / "frames.jsonl"
    if not path.is_absolute():
        path = ROOT / path
    frames = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    t0 = frames[0].get("_ts") or 0.0
    try:                                   # the page passes its own ?at= on the socket URL
        skip = float(c.ws.query_params.get("at") or REPLAY_AT)
    except ValueError:
        skip = REPLAY_AT
    try:
        speed = max(0.1, float(c.ws.query_params.get("speed") or REPLAY_SPEED))
    except ValueError:
        speed = REPLAY_SPEED
    start = time.time()
    for m in frames:
        at = float(m.pop("_ts", t0) or t0) - t0
        wait = (at - skip) / speed
        if wait > 0:
            await asyncio.sleep(max(0.0, start + wait - time.time()))
        c.put(m)
    await asyncio.Event().wait()           # the recording is over; keep the socket open


async def levels(c: Conn) -> None:
    while True:
        await asyncio.sleep(0.7)
        c.put(status(time.time(), random.random() * 0.55, random.random() * 0.4))


def handle_ask(c: Conn, text: str, aid: str) -> None:
    low = (text or "").lower()
    if any(k in low for k in ("what if", "instead", "suppose", "gmv fee")) or _RATE.search(low):
        m = _RATE.search(low)
        rate = float(m.group(1)) / 100 if m else 0.01
        a, frame = whatif(rate, time.time())
        c.put({**a, "id": aid, "question": text, "source": "manual", "cached": False, "who": "Me"})
        c.put(frame)
        c.put(question(aid, text, "Me", None, "answered", time.time(), via="typed"))
        return
    for keys, (body, fig, cites) in ASK:
        if any(k in low for k in keys):
            c.put({**answer(aid, text, body, fig, cites, "manual", random.randint(700, 1600), True,
                            time.time()), "who": "Me"})
            c.put(question(aid, text, "Me", None, "answered", time.time(), via="typed"))
            return
    c.put({**answer(aid, text, "Nothing in the loaded files answers that directly. Try naming a file or a metric.",
                    None, [], "manual", random.randint(900, 1800), False, time.time()), "who": "Me"})
    c.put(question(aid, text, "Me", None, "answered", time.time(), via="typed"))


def handle_show(c: Conn, cid: str) -> None:
    topic, body, fig, cites = SHOW.get(
        cid, ("Topic", "No detail cached for this topic yet.", None, []))
    now = time.time()
    c.put(answer(c.next_id("s"), topic.lower(), body, fig, cites, "prefetch", 120, True, now))
    c.put(cand(cid, topic, "expired", "", now))


def handle_set(c: Conn, key: str, value) -> None:
    if key not in DEFAULT_SETTINGS:
        return
    if key in ("web_mode", "mode"):              # the settings that are words, not numbers
        allowed = ("auto", "off") if key == "web_mode" else tuple(m["id"] for m in MODES)
        if str(value) in allowed:
            c.settings[key] = str(value)
            c.put({"type": "settings", "values": dict(c.settings)})
        return
    lo, hi = SETTING_RANGE[key]
    try:
        v = min(hi, max(lo, float(value)))
    except (TypeError, ValueError):
        return
    if isinstance(DEFAULT_SETTINGS[key], int):
        v = int(round(v))
    c.settings[key] = v
    c.put({"type": "settings", "values": dict(c.settings)})


async def reader(c: Conn) -> None:
    while True:
        raw = await c.ws.receive_text()
        try:
            m = json.loads(raw)
        except json.JSONDecodeError:
            continue
        kind = m.get("type")
        if kind == "ask":
            aid = c.next_id("t")
            c.put(question(aid, m.get("text", ""), "Me", None, "heard", time.time(), via="typed"))
            await asyncio.sleep(random.uniform(0.7, 1.4))
            handle_ask(c, m.get("text", ""), aid)
        elif kind == "show":
            handle_show(c, m.get("id", ""))
        elif kind == "set":
            handle_set(c, m.get("key", ""), m.get("value"))
        elif kind == "clear":
            c.put({"type": "cleared"})


app = FastAPI(title="presage stub")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/api/state")
async def stub_state():
    return hello(time.time(), DEFAULT_SETTINGS)


# ----------------------------------------------------------------- control surface

@app.post("/api/audio/start")
async def stub_audio_start(body: dict | None = None):
    uid = (body or {}).get("mic_uid") or None
    AUDIO.update(state="running", mic_state="ok", mic_uid=uid or "BuiltInMicrophoneDevice",
                 mic_device=next((d["name"] for d in DEVICES if d["uid"] == uid),
                                 "MacBook Pro Microphone"), error=None)
    return {"audio": dict(AUDIO)}


@app.post("/api/audio/stop")
async def stub_audio_stop():
    AUDIO.update(state="stopped", mic_state="unknown")
    return {"audio": dict(AUDIO)}


@app.get("/api/audio/devices")
async def stub_devices():
    await broadcast({"type": "devices", "inputs": DEVICES, "output_device": AUDIO["output_device"]})
    return {"inputs": DEVICES, "outputs": [], "output_device": AUDIO["output_device"],
            "mic_uid": AUDIO["mic_uid"]}


@app.post("/api/context/deep")
async def stub_deep(body: dict | None = None):
    if "enabled" in (body or {}):
        DEEP["enabled"] = bool((body or {}).get("enabled"))
        DEEP["state"] = "running" if DEEP["enabled"] else "off"
    else:
        DEEP.update(state="running", done=0, qa=0)
    await broadcast(status(time.time()))
    return {"index": {"files": 7, "tables": 5, "chunks": 120, "deep": dict(DEEP)},
            "prefs": dict(PREFS, deep_index=DEEP["enabled"])}


@app.post("/api/context/reindex")
async def stub_reindex():
    await broadcast(context_msg())
    return {"context": CONTEXT}


@app.post("/api/context/browse")
@app.post("/api/context/add")
@app.post("/api/context/reveal")
async def stub_picker():
    return {"cancelled": True}          # no native dialogs from the stub


@app.post("/api/context/dir")
async def stub_context_dir(body: dict):
    CONTEXT["dir"] = str(body.get("path") or CONTEXT["dir"])
    CONTEXT["own_dir"] = CONTEXT["dir"].endswith("/context")
    await broadcast(context_msg())
    return {"context": CONTEXT}


@app.delete("/api/context/file")
async def stub_context_delete(body: dict):
    if not CONTEXT["own_dir"]:
        return JSONResponse({"error": "presage does not delete files outside its own context folder"},
                            status_code=400)
    name = str(body.get("name") or "")
    CONTEXT["files"] = [f for f in CONTEXT["files"] if f["name"] != name]
    await broadcast(context_msg())
    return {"removed": name, "context": CONTEXT}


@app.post("/api/llm")
async def stub_llm(body: dict):
    LLM["provider"] = str(body.get("provider") or LLM["provider"])
    LLM["key_set"] = LLM["provider"] != "claude_code"
    if isinstance(body.get("models"), dict):
        LLM["models"].update({k: v for k, v in body["models"].items() if v})
        LLM["answer_model"] = LLM["models"].get("answer")
    if body.get("api_key"):
        LLM.update(key_set=True, key_hint="…" + str(body["api_key"])[-4:])
    LLM.update(key_ok=True, key_message="Key OK")
    return {"llm": dict(LLM)}


@app.post("/api/decider")
async def stub_decider(body: dict | None = None):
    body = body or {}
    if body.get("api_key"):
        LLM["decider"] = dict(LLM["decider"], kind="jev", ok=True, key_set=True, key_hint="…4Q7B",
                              message="ok", latency_ms=127)
    if body.get("intent") in ("auto", "local", "jev", "off"):
        LLM["decider"] = dict(LLM["decider"], intent=dict(LLM["decider"]["intent"], choice=body["intent"],
                              kind="local" if body["intent"] == "auto" else body["intent"]))
    if body.get("choice") in ("auto", "jev", "laya", "haiku"):
        kind = {"auto": "pregate+laya", "jev": "jev", "laya": "laya", "haiku": "haiku"}[body["choice"]]
        LLM["decider"] = dict(LLM["decider"], choice=body["choice"], kind=kind, ok=True, message="ok",
                              latency_ms={"jev": 183, "haiku": 833}.get(body["choice"], 71))
    return {"llm": dict(LLM)}


@app.get("/api/screen")
async def stub_screen():
    return {} if REPLAY else dict(SCREEN, ts=time.time())     # a recording has no screen of the stub's


@app.post("/api/llm/live")
async def stub_llm_live(body: dict):
    choice = str(body.get("choice") or "auto")
    LLM["live_choice"] = choice
    LLM["live_provider"] = LLM["provider"] if choice in ("auto", "same") else choice
    return {"llm": dict(LLM)}


@app.post("/api/feedback")
async def stub_feedback(body: dict):
    """E12: the verdict on a card, echoed to every page as a `feedback` frame."""
    cid, verdict = str(body.get("id") or ""), str(body.get("verdict") or "")
    if not cid or verdict not in ("used", "ignored", "wrong", "good"):
        return JSONResponse({"error": "id and a verdict of used, ignored, wrong or good"}, status_code=400)
    await broadcast({"type": "feedback", "id": cid, "verdict": verdict, "ts": time.time()})
    return {"ok": True, "id": cid, "verdict": verdict, "known": True}


@app.post("/api/llm/test")
async def stub_llm_test():
    return {"llm": dict(LLM)}


@app.post("/api/quit")
async def stub_quit():
    await broadcast({"type": "status", "quitting": True, "now": time.time()})
    asyncio.get_running_loop().call_later(0.4, lambda: os._exit(0))
    return {"quitting": True}


@app.get("/")
async def index():
    return FileResponse(UI_DIR / "index.html",
                        headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    c = Conn(ws)
    CLIENTS.add(c)
    feed = (replay_file(c),) if REPLAY else (replay(c), levels(c))
    tasks = [asyncio.create_task(t) for t in (writer(c), *feed, reader(c))]
    try:
        await asyncio.gather(*tasks)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        CLIENTS.discard(c)
        for t in tasks:
            t.cancel()


app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
