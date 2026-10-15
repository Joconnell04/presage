# presage UI protocol

One WebSocket at `ws://<host>/ws`, JSON text frames, one message per frame. The
page is `ui/index.html`, served from the same origin. `presage/server_stub.py`
implements everything below and replays a scripted call; the real server should
be a drop-in replacement for it.

Frames the client does not recognise are ignored, so the server can add fields
without breaking the page.

This file is the contract. How the page draws each frame (sizes, colours, where things sit) is in
`docs/design/README.md`; where a rendering note here disagrees with it, the design doc is current.

## Server to client

### `hello`
Sent once per connection, before anything else, and again on every reconnect.
The client throws away all of its state and rebuilds from it.

```json
{"type":"hello",
 "now": 1758196800.12,
 "settings": {"topic_threshold":0.35,"question_threshold":0.5,
              "prefetch_cooldown_s":8,"max_warm":6,"answer_ttl_s":90},
 "status": { ...same body as a status message... },
 "transcript": [ ...last 50 transcript messages, oldest first... ],
 "answers":    [ ...recent answers, NEWEST FIRST... ],
 "candidates": [ ...warm candidates... ],
 "audio":   { ...same as status.audio... },
 "llm":     { ...same as status.llm... },
 "context": { ...same as a context message... },
 "devices": [ ...same as devices.inputs... ],
 "modes":   [{"id":"call","label":"Call","hint":"Answers from your files appear here."}, ...],
 "notes":   { ...same as a notes message... } | null,
 "scratch": [ ...same items as a scratch message, oldest first... ],
 "screen":  { ...same body as GET /api/screen, or {} ... },
 "questions": [ ...the last 10 question frames, NEWEST FIRST... ],
 "prefs":   { ...settings.json, see "Settings that persist"... }}
```

`questions` (E68) is the last ten registered questions as `question` frames (below), newest first,
each at its current `status`, so a page that reconnects while a question is still waiting for its
card can draw the waiting slot again. `[]` on a server older than E68.

`screen` (E43a) is one reading of what the presenter has on screen, taken at
send time, so a page can render "Slide 4 of 6 · Strategy_Deck.pptx" on connect
without polling. It is `{}` when nothing resolved, when the poller is not
running, or when the reading is older than `screen_ttl_s`; every other field is
the one `GET /api/screen` documents below, and that route is still what a page
polls to keep it live. There is no `screen` frame on the WebSocket.

`modes` is the mode roster, in switcher order, from `brain.modes()`. The page
renders the labels and uses `hint` as the empty state of the answers column, so
none of that copy lives in the page.

`settings` is the five Tune values and nothing else: it is what the sliders read
and what `set` writes. The persisted file is `prefs`, which is a superset and is
not interchangeable with it.

`now` is an addition to the brief: unix seconds on the server's clock at send
time. The client stores `offset = now - clientNow` and renders every age as
`(clientNow + offset) - ts`, so ages stay right when the browser's clock is off
or the page is on a phone. If `now` is missing the offset is 0 and `ts` is
assumed to be on the client's clock.

`answers` must be newest first. The first entry becomes the big card.

### `transcript`
```json
{"type":"transcript","id":"ts7","source":"system","text":"...",
 "final":false,"t0":1712.3,"t1":1714.8,"utt":7,"retracts":false}
```
`source` is `mic` (labelled **Me**, accent colour) or `system` (labelled
**Them**). Re-sending the same `id` replaces that line in place, so an interim
becomes its final without a new row. Interim lines render dimmed and italic.

`utt` is the provider's utterance number, monotonic per source, and the server
builds the id from (source, utt) when it is there: `ts7` for system utterance 7,
`tm7` for the mic's. `retracts: true` says this frame withdraws an eager final
that was already sent for that utterance, which lands on the same row because
the id is the same. Providers that never retract can leave both out, and the
server falls back to one open id per source, which is what it did before.

`t0`/`t1` are seconds and only need to be consistent with each other. The
timestamp column shows `m:ss` relative to the smallest `t0` the client has seen,
which makes it a call clock rather than a wall clock.

The client keeps the last 220 lines. It auto-scrolls to the bottom unless the
user has scrolled up, in which case a `↓ Latest` pill appears.


`ts` (optional): the server's wall clock, unix seconds, when the frame was sent. The page takes
`ts - (t1 - t0)` as the line's start for the `m:ss` column when it is there, because `t0`/`t1`
restart at 0 whenever capture restarts. It also uses `ts` to spot a line id that came back for a
new utterance (see `docs/frontend-asks.md`, "Transcript ids must survive a capture restart").

`unstable` (optional, interims only): the tail the transcriber may still rewrite. `text` is the
committed prefix and `text + " " + unstable` the whole current hypothesis. The page renders the
tail dimmed and italic; nothing should act on it.

`heard` (optional): what the recogniser actually wrote, when `presage/stt/lexicon.py` changed it
(E46). `text` is the repaired line the brain reads -- "the GMV bracket" -- and `heard` is
"the arm bracket". Absent on every line the corrector left alone, which is most of them. The page
should show `text` and keep `heard` behind a hover or a small "heard as" note: a consultant who
sees a word they did not say wants to know it was the transcriber and not them, and a correction
that redraws the line they are reading is worse than the error. The local provider puts it on the
`Transcript` (`presage.stt.local.HeardTranscript`); putting it on the wire is one line beside
`unstable` in `server.on_transcript`, `msg["heard"] = getattr(t, "heard", "") or None`, and until
that line exists the field never reaches the page.

`speaker` (optional, system lines only): the far-side speaker label from the on-device labeller, `"A"`, `"B"`, ... stable for the life of the call, or `"Me"` when the consultant's own voice leaked into the far side (identified by enrollment from the first 10 s of the mic). Absent until the labeller has settled the span; a later frame for the same `id` may add it. Render as "Them A".

### `candidate`
```json
{"type":"candidate","id":"c9","topic":"Q3 revenue","status":"warming",
 "preview":"Q3 FY25 revenue $4.2M, +8% QoQ","ts":1712.9}
```
`status` is `warming` (amber dot), `ready` (green dot, `preview` shown) or
`expired` (chip removed). Same `id` updates the chip in place. Clicking a
`ready` chip sends `show`; `warming` chips are not clickable.

The client keeps at most `max_warm` chips, dropping the oldest, so the server
does not have to send `expired` for everything it drops.

### `answer`
```json
{"type":"answer","id":"a4","question":"what did revenue look like in q3",
 "answer":"Q3 FY25 revenue was $4.2M, up 8% QoQ and 3% over plan.",
 "figure":"$4.2M",
 "citations":[{"file":"Acme_Financials.xlsx","loc":"P&L › B12"}],
 "say":"Eight percent up, three over plan",
 "source":"prefetch","latency_ms":840,"in_files":true,"ts":1713.4,
 "answered_from":"files","web_citations":[],
 "working":null,"derived":false}
```
- `say` is the line the consultant can speak as it stands, 4 to 8 words. When it
  is there the top card is figure (62px), say-line (34px), then the basis chips,
  and `answer` is hidden until the card is tapped, which expands it in place
  (the **More** button says so). That is the 1.5-second glance in
  `docs/research-next.md`, bet 5. Without `say` the top card falls back to the
  old layout.
- `answer` is the fallback and the detail: 25 words at most. Older cards always
  render it at 18px, whether or not they carry a `say`.
- `figure` is optional. When present it renders above the say-line at 62px on
  the top card. Keep it under about 8 characters or it wraps.
- `citations` render as `file › loc` chips. `loc` is free text; `[]` is fine.
- `cached: true` means the answer came straight out of the deep index rather than a live
  model call; the card adds a small outlined **INSTANT** tag next to the source badge.
- `mode` is the mode id the answer was produced in. Anything other than `call`
  shows as a small grey label in the status line, which is what explains a
  **SUGGESTED** card.
- `source` drives the badge: `prefetch` → green **PREFETCHED**, `live` →
  accent **LIVE**, `manual` → grey **MANUAL**.
- `latency_ms` renders as `0.8 s`.
- `answered_from` is `files`, `web`, `model`, `derived`, `none`, `correction` or
  `drift` (the last two are the two cards below, neither of which is an answer to
  anything anybody asked). `web` adds a
  sky **WEB** badge; `model` adds an amber **SUGGESTED** badge, which means the
  model wrote it rather than reading it out of a file or a page, so it needs no
  citations and gets no red treatment; `derived` adds an orange **DERIVED**
  badge, which means the number was computed by the `whatif` or `calc` tool from
  cells the card cites, not read out of one; `none` adds the red **NOT IN FILES** badge, mutes
  the answer text and puts a red border on the card. It is the field the page reads; `in_files: false` is
  the same signal for messages sent before web search existed, and the page
  falls back to it when `answered_from` is missing.
- `web_citations` is `[{"title","url"}]` and renders as sky chips next to the
  file citations, each a link that opens in a new tab. An entry with no title
  shows its host. `citations` and `web_citations` can both be non-empty.
- `working` is how a derived number was reached, with its inputs, e.g.
  `"Fee Model B48 0.85% → 1%, model recalculated"` or `"$2.60M × 1% = $26,050"`.
  It renders under the say-line in the mono face, small and muted, on every card
  that carries one. `null` or absent on a card that computed nothing.
- `derived: true` is the same signal as `answered_from: "derived"`, carried
  separately so the server can keep its own count; the page reads
  `answered_from`. A derived card that also came out of the deep index keeps its
  **INSTANT** tag next to **DERIVED**.
- Re-sending an existing `id` replaces that card and moves it to the top. That
  is how the web escalation lands: the live answer says `none`, the retry with
  web search re-sends the same `id` with `answered_from: "web"`, and the page
  animates the swap so the eye catches it.

Any `answer` clears the pending spinner in the ask bar, whatever its `source`.
The client keeps the newest 14.

- `partial: true` marks a card that is still being written: the same `id` arrives again, without
  the flag, when the answer settles. Until E52 the only field a partial carried was `answer`,
  growing word by word; `figure` was always `null` and the citations always empty. It now carries
  **`figure` as well**, from the moment the model closes that key, which is a median of about a
  second before the settled frame. A partial `figure` is a whole value or absent, never half a
  number, and it is the same figure the streamed `answer` sentence already states in words, so a
  page that draws the chip early is not showing anything new; what is not there yet is
  `citations`, `say`, `working` and `verdict`, all of which arrive with the settled frame. A page
  that ignores `partial` frames, or reads only `answer` from them, behaves exactly as before.


`who` (E68, optional): who asked the question the card answers, `"Them"` or `"Me"`. `null` on a
card nobody asked for (a correction, a drift card) and on anything sent before E68. A `"Me"` card
answers the consultant's own spoken question: a lookup ("what was the Q3 number again?") or a
check (the consultant asked the client a fact the files hold; see `question.policy`). The page
may want to tag it, since the consultant knows they asked it.

`line` (E68, optional): the transcript line id the card answers (`"ts7"`, `"tm3"`), the same id
the `transcript` frame carries. `utt` stays what it always was, the **client's** utterance number,
so a card answering the consultant carries `utt: null` and only `line`.

`no_answer` (E68, optional): set on a no-answer card, below. Absent or `null` on every answer.

`verdict` (optional): the verifier's check of the card's numbers against the cells and pages it cites: `{ok, level: "cell"|"doc"|"derived"|"web"|"unverified", problems: [str], reason}`. `ok: false` on a live card means a number contradicts its citation (render an Unverified badge; the figure should not be read out); `level: "unverified"` means no cited cell holds the number (an under-citation, a softer chip). Speculative and cached cards that fail are never sent.

#### the spoken correction

```json
{"type":"answer","id":"sddd7a6db","question":"said nine million",
 "figure":"$14.85M","say":"Startup Costs is $14.85M",
 "answer":"said nine million; Startup Costs is $14.85M",
 "citations":[{"file":"Plan_Model.xlsx","loc":"Summary Recommendation › B5"}],
 "working":"said nine million; Startup Costs, Bottom Line Ask is $14.85M in Summary Recommendation B5",
 "answered_from":"correction","source":"live","latency_ms":1,"in_files":true,
 "topic":"Startup Costs","model":"","derived":false,
 "verdict":{"ok":true,"level":"spoken","spoken":"contradicts","said":"nine million",
            "actual":"$14.85M","cell":"Plan_Model.xlsx › Summary Recommendation › B5",
            "how":"metric","ms":1.39,"problems":[],
            "reason":"said nine million; Startup Costs is $14.85M"}}
```

The same frame with one new `answered_from`. Nobody asked anything: the consultant said a
figure out loud that the files contradict, and `presage/spoken.py` caught it in about a
millisecond with no model call (E9b, wired in E9c). It is the only card the Me side produces.

- `answered_from: "correction"` is the whole of the addition. The current page tolerates it
  without a change — `answeredFrom` falls through to the Files dot rather than breaking — but it
  deserves its own quiet treatment: **this is not an answer and should not take the top of the
  answer deck.** The two things worth drawing are `question` and `figure`; `working` belongs
  behind the card. A correction that takes a paragraph to read is one the consultant cannot use
  mid-sentence.
- `question` is the figure that was **said**, not the line it came from: the intervention log
  keeps no mic side (E12), and what went into the room is the card's own content.
- `say` is the correction to speak, three to five words. `working` is one line with the cell in
  it; `answer` repeats the short reason, which is what the row view renders.
- `verdict.level` is `"spoken"` and `verdict.ok` is true, because the figure on the card *is*
  the cell it cites: E23's verifier is skipped for these and the guard's own verdict is carried
  here instead. `verdict.spoken` is `contradicts` (the only verdict that produces a card),
  `said` / `actual` / `cell` are the three things compared, `how` is the tier that resolved the
  referent (`metric`, `lever` or `qa`), and `ms` is what the check cost.
- `id` starts with `s`. One correction per cell per 60 s.
- Everything else in the protocol treats it as an ordinary card: `POST /api/feedback` marks it
  used / ignored / wrong, and it reaches the intervention log as a `card` row carrying
  `answered_from: "correction"`.

#### the slide-drift card

```json
{"type":"answer","id":"d41c9ba2","question":"page 2 says $11.3M",
 "figure":"$14.15M","say":"Model says $14.15M, not $11.3M",
 "answer":"Page 2 says $11.3M; the model's Funding Scenarios!D100 is $14.15M",
 "citations":[{"file":"Plan_Model.xlsx","loc":"Funding Scenarios › D100"},
              {"file":"Plan_Memo.pdf","loc":"page 2"}],
 "working":"Plan_Memo.pdf › page 2 prints $11.3M; Plan_Model.xlsx › Funding Scenarios!D100 (Funding Scenarios Cumulative Capital Required) held 11,319,112.95 when the deck was linked and holds 14,148,891.19 now; link score 0.60; 5 other cells held the same number",
 "answered_from":"drift","source":"live","latency_ms":4,"in_files":true,
 "topic":"Funding Scenarios Cumulative Capital Required","model":"","derived":false,
 "verdict":{"ok":true,"level":"drift","problems":[],
            "reason":"Page 2 says $11.3M; the model's Funding Scenarios!D100 is $14.15M",
            "cell":"Plan_Model.xlsx › Funding Scenarios!D100",
            "was":11319112.95,"now":14148891.19,"slide":"$11.3M",
            "score":0.6014,"n_alt":5,"others":0,"ms":4.31}}
```

The same frame again, with a second new `answered_from`. Nobody asked anything and nobody said
anything: a slide or a memo page came up on screen, and a figure printed on it no longer matches
the model cell it was built from (E16's `links` table says which cell that is, E63 re-reads it).
`presage/drift.py` finds it in a few milliseconds with no model call.

- `answered_from: "drift"` is the whole of the addition, and it wants the same quiet treatment as
  `correction`: **this is not an answer and should not take the top of the answer deck.** The two
  things worth drawing are `question` (what the page claims) and `figure` (what the model says).
- `question` is the page's own claim, `figure` is the model's number **in the page's own units**,
  so `$9.6M` is answered with `$11.98M` and never with `11982000`. `answer` is the whole sentence
  and `say` is the three to six words to speak instead of reading the slide out.
- `citations` are two, in this order: the cell first, because that is where the live number is,
  and the page second. `topic` is the cell's own row label.
- `verdict.level` is `"drift"` and `verdict.ok` is **true**: the figure on the card is the cell.
  What disagrees with it is the page. E23's verifier is skipped for these, exactly as it is for a
  correction, because a card whose content is "no cell holds this number any more" is the one card
  the verifier would always reject. `was` / `now` / `slide` are the three things compared,
  `others` is how many more figures on the same page also drifted (the card shows the most
  confident one), and `n_alt` is how many other cells held that number when the link was built,
  which is how sure the citation is.
- `id` starts with `d`. **One card per page per session**, and one per cell per 60 s on top, so
  stepping back to slide 4 raises nothing and two pages quoting the same stale number raise one
  card between them.
- `latency_ms` is what the check cost, not a model call: 0.7 to 5 ms a page.
- `POST /api/drift` returns the same findings for the page on screen (or for
  `{"file":…, "loc":"slide 4"}`) as a list, without raising a card and without touching either
  cooldown: `{file, head, links, checked, skipped, ms, drifts:[…], checked:true, seen}`. `seen`
  says whether this session already raised that page's card.
- `settings.drift = 0` turns the cards off, the same shape as `screen_bias`. With the screen
  poller off (`PRESAGE_SCREEN=0`) nothing fires, because nothing reports a page change.

#### the no-answer card

```json
{"type":"answer","id":"a9f2c1a7b","question":"what's the total startup cost we're asking for?",
 "answer":"Couldn't answer: nothing came back in 6 s. Still looking.","say":"No answer yet",
 "figure":null,"citations":[],"web_citations":[],"answered_from":"none","in_files":false,
 "source":"live","latency_ms":6004,"model":"","utt":7,"line":"ts7","who":"Them",
 "verdict":{"ok":true,"level":"none","problems":[],"reason":"Couldn't answer: nothing came back in 6 s. Still looking."},
 "no_answer":{"reason":"slow","after_ms":6004,"detail":""}}
```

E68. A registered question (a `question` frame went out for it) is never met with silence. When
nothing has reached the page for it `NO_ANSWER_S` (6 s) after it registered, or the answer call
ends with nothing to show, this card goes up **under the card id the answer would have used**, so
the answer replaces it in place whenever it lands, exactly as a redraw does today.

- It is an ordinary `answer` frame with `answered_from: "none"`, which the current page already
  draws as the red **NOT IN FILES** card. `no_answer` says why, so the page can word it better:

  | `no_answer.reason` | `answer` | `say` | what happened |
  | --- | --- | --- | --- |
  | `slow` | Couldn't answer: nothing came back in 6 s. Still looking. | No answer yet | the call is still running; its card replaces this one if it comes |
  | `none` | Not in the files. | Not in the files | the model searched and found nothing, and said it with no words |
  | `empty` | Couldn't answer: the answer came back empty. | Couldn't answer | the call ended with nothing |
  | `error` | Couldn't answer: `<short error>`. | Couldn't answer | the call failed (this used to be an "Error: …" card claiming `answered_from: "files"`) |
  | `timeout` | Couldn't answer: no answer in 25 s. | Couldn't answer | still nothing on the page `LIVE_TIMEOUT_S` after it registered: the call is given up (a call already streaming text is never cut off) |

  A `slow` card can be followed by an `empty`, `none`, `error` or `timeout` one under the same id
  when the call finishes with nothing; the page should treat each as a redraw.
- `question` is the question as registered, `after_ms` how long after registration the card went
  up, `detail` the error text for `error`.
- `verdict.level` is `"none"` and `verdict.ok` true: there is no figure to check, and E23's
  verifier is skipped, so nothing can drop it.
- It is **not** raised for a consultant's `check` or `heard` question (those end quiet by design),
  for an objection, or for a line the gate read as probably not a question (`question.sure:
  false`). A model's own "Not in files: …" answer is not a no-answer card; it keeps coming as it
  always did.
- A page that draws it as anything other than an answer, e.g. dimmed with the question text
  under it, should switch back to the normal card when the same id arrives without `no_answer`.

### `question`
```json
{"type":"question","id":"q9f2c1a7b","aid":"a9f2c1a7b","text":"what's the total startup cost we're asking for?",
 "who":"Them","source":"system","utt":7,"utts":[7],"line":"ts7","kind":"question","policy":"answer",
 "via":"heuristic","sure":true,"status":"heard","ts":1758196800.1}
```

E68. The moment the brain registers a line as a question, **before any answer exists**, whoever
asked it. Re-sent under the same `id` whenever its `status` changes. The current page ignores it
(unknown frame types are dropped), so it can be adopted whenever the GUI gets to it; the server
stub does not emit it yet.

- `aid` is the id of the `answer` card that will answer it. The page can open a slot for that
  card now ("Heard: …") and fill it when the answer frame with that id arrives. `id` is `aid`
  with its first letter swapped to `q`.
- `who` is `"Them"` (the client) or `"Me"` (the consultant's own spoken question, new in E68);
  `source` is the transcript source it came from (`system` / `mic`). A typed ask registers too,
  with `who: "Me"` and `via: "typed"`.
- `text` is the question as registered: the joined text when two finals were one question, and
  it is re-sent with the longer text when the transcriber's settled final adds words.
- `utt` / `line` are the transcript line it came from (`utt` is per source, so read it with
  `source`; `line` is the transcript frame's own id). `utts` lists every utterance folded into the
  question, `utt` first: two finals a sentence pause split, or a settled final that moved.
- `kind` is the gate's reading: `question`, `request`, `objection` (a what-if is a `question`).
- `policy` is what the brain will do about it:
  - `answer`: a card, and a no-answer card at worst (every client question; a consultant lookup);
  - `check`: the consultant asked the client for a fact; a card only if the files hold it, so the
    consultant can check the client's reply against it, and nothing otherwise (`status` goes to
    `quiet`);
  - `heard`: the consultant asked the client for a view, plan or preference: registered, no card
    will follow. This is the "we heard you ask" signal and nothing more.
- `via` is how it was registered: `heuristic`, `second-opinion` (the pre-gate said no, the tier-2
  gate overruled it), `gate`, `interview`, `me`, `typed`, `joined`, or `repeat`.
- `sure` is false when the line registered on shape alone and the gate doubted it; such a question
  gets its answer card but never a no-answer card.
- `status`: `heard` (registered, nothing on screen yet) → `answered` (the first card for `aid`,
  partial or settled, reached the page) | `no-answer` (the no-answer card went up; a later answer
  still moves it to `answered`) | `quiet` (a `check` the files could not answer). A `repeat` frame
  (the same question again within 20 s) carries the **earlier** question's `aid` and status: the
  card already on screen is its answer.
- `ts` is when it registered (server clock).

### `feedback`
```json
{"type":"feedback","id":"a4","verdict":"wrong","ts":1758196800.1}
```
The verdict now on record for a card, echoed to every open page after any client
posts `POST /api/feedback`, so two screens agree. `verdict` is `used`, `ignored`,
`wrong` or `good`, and a later frame for the same `id` replaces the earlier one.

The page should show the mark on that card and nothing else: no toast, no
reordering, no removal. A card with no frame has no verdict, which is the normal
state for most cards. `hello` does not carry these; a reload forgets what was
marked, and the record lives in the log, not on screen.

### `notes`
```json
{"type":"notes",
 "summary":["Q3 landed at $4.2M, 3% over plan", "..."],
 "actions":["Send CAC payback by end of day"],
 "questions":["LatAm pilot read"],
 "ts":1758196800.1,"utterances":31}
```
The rolling notes card, pinned above the answer stack. At most 5 summary lines,
5 actions and 3 open questions; the page truncates past that. Each frame is the
whole card, not a delta, and replaces the one on screen. `utterances` is how
many final transcript lines it was built from and renders as "31 lines".

Only Notes mode produces these, but the card stays visible in every mode once it
exists, so switching to Call to check a number does not lose the notes. `hello`
carries the last one as `notes`, or `null`. `cleared` drops it.

### `scratch`
```json
{"type":"scratch",
 "items":[{"label":"GMV fee","value":"1%",
           "output":"Year 5 subscriber revenue $9.59M",
           "working":"Fee Model B48 0.85% → 1%, model recalculated",
           "ts":1758196800.1}]}
```
The what-ifs run so far this call, oldest first, at most five. Each frame is the
whole list, not a delta. One is sent after every derived answer: `label` is the
lever that moved, `value` is what it moved to, `output` is the headline number
it produced (the card's `figure` when there is one), `working` is the same
string the card shows.

`aid` (optional, asked for in `docs/frontend-asks.md`): the id of the card that produced the row.
The page hides a row whose card is on screen and no longer derived.

The page lists them in a Scratch card under the Notes card, newest at the top,
as `label  value → output`, with `working` as the row's tooltip. The card is
hidden while the list is empty, which is most calls. `hello` carries the list as
`scratch` (`[]` when there is none) and `cleared` empties it.

### `status`
```json
{"type":"status","mic":true,"system":true,"stt":"local:parakeet",
 "index":{"files":7,"tables":5,"chunks":120},"listening":true,
 "levels":{"mic":0.2,"system":0.05},"now":1758196800.12,"me_speaking":false,
 "audio":{"state":"running","mic_state":"ok","mic_device":"MacBook Pro Microphone",
          "mic_uid":"BuiltInMicrophoneDevice","output_device":"MacBook Pro Speakers",
          "error":null},
 "llm":{"provider":"anthropic","answer_model":"claude-sonnet-5",
        "models":{"answer":"…","hedge":"…","classifier":"…"},
        "defaults":{"answer":"…","hedge":"…","classifier":"…"},
        "key_set":true,"key_hint":"…AB12","key_ok":true,"key_message":"Key OK"},
 "context":{"dir":"/Users/me/presage/context","files_count":7,"own_dir":true}}
```
Every field is optional and merges into what the client already has, so a
levels-only heartbeat is fine. `levels` are 0 to 1 and drive the two meters in
the top bar (hidden on phones). `listening: false` turns the STT dot amber.
`index.tables` and `index.chunks` show in the tooltip on the files chip.

`index.deep` is the background pass that builds the Q&A cache, when this build has one:

```json
"deep":{"state":"running","done":12,"total":40,"qa":96,"summaries":12,
        "chunks_contextualized":220,"numbers":41,"definitions":9,
        "error":null,"started":1758196800.1,"finished":null,"enabled":true}
```

`state` is `idle`, `running`, `ready` or `error`, and `enabled` is the persisted
`deep_index` setting. Settings → Files renders it under the folder path as
"Deep index · 12/40 · 96 questions ready" with a thin bar, "Deep index · ready ·
96 questions" when it finishes, "Deep index · off" with a Turn on link, or the
error in red; the files chip tooltip says the same. Progress frames are
throttled to two a second. The whole row is hidden when the build has no deep
pass at all. `now`
is accepted here too and re-syncs the clock offset.

`me_speaking` is true from the first interim on the mic until that utterance's
final. While it is true the page holds new answer cards and candidate
promotions, and releases them on the first status that says false, or after 4
seconds, whichever comes first; a count shows next to the ANSWER heading
("1 held"). Redraws of a card already on screen are not held, so a streaming
answer keeps updating, and the Notes card is never held. A mic utterance that
never closes is forced false after 25 s.

`audio.state` is `stopped` | `starting` | `running` | `error`, and `error`
carries the message when it is `error`. `audio.mic_state` is `ok` (green Mic
dot), `recovering` (amber), `silent` or `fallback` (red), or `unknown` before the
helper has said anything; the dot's tooltip is the reason, and the Settings
status line reads e.g. "Running · mic fell back to MacBook Pro Microphone".
Anything other than `running` shows the subdued "Not listening" banner with a
Start button on the main page.

`llm.provider` is `anthropic`, `gemini` or `claude_code`. `llm.key_ok` is
`true`, `false` or `null` when the key has not been checked, and `key_hint` is
the last four characters, never the key.

`claude_code` bills the Claude subscription the `claude` CLI is logged into, so
it has no key: `key_set` is false and the block carries

```json
"login":{"ok":true,"account":"you@example.com","message":"Signed in as you@example.com"},
"hint":"Run `claude` once in Terminal and log in with your Claude account"
```

which the Model section shows in place of the key field, with the hint only when
`login.ok` is false.

`llm.live_choice` and `llm.live_provider` are the answer routing: speculative drafts always
run on `provider`, while the live answer for a landed question can run somewhere else.
`live_choice` is what was picked (`auto`, `same`, or a provider name) and `live_provider` is
what that resolves to, which the Model section prints as "Live answers on Anthropic API ·
speculative on Claude Code". `auto` means the API key when the main provider is the Claude Code
subscription and `ANTHROPIC_API_KEY` exists, measured about twice as fast, and `same` otherwise.

`llm.decider` is the typed-decision gate (bet 6), shown as a Decisions row under
the provider:

```json
"decider":{"kind":"jev","ok":true,"key_set":true,"key_hint":"…20c8",
           "message":"ok","latency_ms":362}
```

`kind` is `jev` when the hosted gate is answering and `haiku` when it has fallen
back, which is what the status line reads: "Jev · ok (362 ms)" or "Jev · no key,
using Haiku". The key lives in `.env` as `JEV_API_KEY` and nothing about the
gate is written to settings.json. The server caches the login for 60 s because the probe can
take a second.

`llm.decider.intent` is a second row under it (E21a), for the two typed-intent
questions: which input of the model a spoken what-if means, and whether a warm
card answers the question that just landed. They are a separate choice from the
gate because they have a separate best answer -- a local 421M checkpoint beats
hosted Jev on both -- and because the gate's own checkpoint answers neither.

```json
"intent":{"kind":"local","choice":"auto","local_available":true,
          "dir":"presage-gate-multi","why_not":null}
```

`kind` is what is answering now (`local`, `jev` or `off`), `choice` is what was
asked for (`auto`, `local`, `jev`, `off`), `dir` is the checkpoint directory's
name under `~/.cache/presage/laya`, and `why_not` is the one-line reason the
local path cannot run when `local_available` is false ("torch not installed: uv
sync --extra laya", "no intent checkpoint in ..."). There is no probe and no
latency: the only thing that can be wrong is that the weights are not here.
A status line reads "Intent · local (presage-gate-multi)" or "Intent · Jev".
`POST /api/decider` takes `{"intent": "auto"|"local"|"jev"|"off"}`, saves it to
settings.json and rebuilds; it takes `choice` for the gate in the same body, and
either may be sent alone.

`llm.error` appears when the chosen provider did not load (for
example the provider module is missing); the page shows it in red and keeps
showing the provider that was chosen, not the one still answering.

`context` here is the compact form: the file list only travels in a `context`
message.

`{"type":"status","quitting":true}` is sent by `POST /api/quit` just before the
process exits. The page stops reconnecting and shows a "presage quit" card.

The real server sends a status every second while a client is connected; the
stub sends one every 700 ms.

### `context`
```json
{"type":"context","dir":"/Users/me/presage/context","own_dir":true,"files_count":3,
 "files":[{"name":"Acme_Financials.xlsx","kind":"xlsx","size":48123,"mtime":1758196000.0,
           "indexed":true,"error":null,"tables":5,"chunks":0}]}
```
Sent after anything changes the folder or its contents: the watcher re-indexing,
a browse, an add, a remove, a reindex. It is the whole list, not a delta.
`indexed: false` with `error: null` means presage does not read that file type
(the row says "Skipped"); `error` is the reason indexing failed, shown red.
`own_dir` is true only when the folder is the `context/` folder inside the
checkout, which is the only folder presage will add to or remove from.

### `devices`
```json
{"type":"devices","inputs":[{"uid":"BuiltInMicrophoneDevice","name":"MacBook Pro Microphone",
  "in":1,"out":0,"rate":48000,"transport":"builtin","default_input":true,
  "default_output":false}],"output_device":"MacBook Pro Speakers"}
```
Sent after a device refresh. A Bluetooth input also carries
`"warning":"Bluetooth mics can drop out; prefer the built-in mic"`, which the
page shows in amber under the mic dropdown.

### `settings`
```json
{"type":"settings","values":{"topic_threshold":0.35,"question_threshold":0.5,
 "prefetch_cooldown_s":8,"max_warm":6,"answer_ttl_s":90}}
```
Merges into the client's settings and moves the sliders. The server should echo
this after a `set` so two open pages stay in sync.

### `cleared`
```json
{"type":"cleared"}
```
Client drops all transcript lines, answers and candidates.

## Client to server

| message | sent when |
| --- | --- |
| `{"type":"ask","text":"..."}` | Enter in the ask bar. The bar shows a spinner and a placeholder card until any `answer` arrives. |
| `{"type":"set","key":"max_warm","value":9}` | A Tune slider moves. Fires on every `input` event, so the server should expect a burst while a slider is dragged. |
| `{"type":"clear"}` | Clear button. |
| `{"type":"show","id":"c9"}` | A ready On-deck chip is clicked. The server should reply with an `answer` and expire the candidate. |

The socket is unchanged by the control surface: everything the Settings drawer
does is an HTTP call, and the result comes back to every open page as a
`status`, `context` or `devices` broadcast.

## HTTP API

All local, no auth, JSON in and out. Errors are `{"error":"…"}` with a 4xx.

| method and path | body | does |
| --- | --- | --- |
| `GET /api/state` | | the `hello` payload, for a page that has not opened a socket |
| `POST /api/ask` | `{text}` | same as an `ask` frame |
| `POST /api/audio/start` | `{mic_uid?}` | starts capture and returns `{audio}` right away with `state: "starting"`; the real state arrives on the socket. `mic_uid` is a uid from `/api/audio/devices`, `""` for the system default, absent to reuse the saved one. Saves the choice and sets `auto_start_audio` |
| `POST /api/audio/stop` | | stops the helper and the STT worker (measured 48-79 ms), clears `auto_start_audio` |
| `GET /api/audio/devices` | | `{inputs, outputs, output_device, mic_uid}` and broadcasts `devices` |
| `POST /api/context/dir` | `{path}` | validates, saves, re-indexes, restarts the watcher, broadcasts `context` |
| `POST /api/context/browse` | | native folder picker, then the same as `/api/context/dir`. `{"cancelled":true}` when the user cancels |
| `POST /api/context/add` | | native file picker, copies the chosen files into the context folder, re-indexes, returns `{added:[names]}` |
| `DELETE /api/context/file` | `{name}` | moves that file to `~/.Trash` and re-indexes. 400 with "presage does not delete files outside its own context folder" unless `own_dir` |
| `POST /api/context/reveal` | | `open -R` the folder in Finder |
| `POST /api/context/reindex` | | rebuild the fast index now, which also schedules the deep pass |
| `POST /api/context/deep` | `{enabled?}` | with no body, runs the deep pass now; `{"enabled": false}` saves the setting and cancels the run, `true` saves it and starts one. Returns `{index, prefs}` |
| `POST /api/llm` | `{provider, api_key?, models?}` | saves, rebuilds the client (`brain.set_provider`), validates the key, returns `{llm}`. `provider: "claude_code"` takes no key and is checked against the CLI login instead |
| `POST /api/llm/test` | `{provider?}` | validate only |
| `POST /api/llm/live` | `{choice}` | where live answers run: `auto`, `same`, `anthropic`, `claude_code` or `gemini`. Saves `live_choice`, calls `brain.set_live_provider`, returns `{llm}` |
| `POST /api/decider` | `{api_key?, choice?, intent?}` | saves `JEV_API_KEY`, the gate `choice` and the `intent` choice, rebuilds the gate so it picks the key up, re-checks it, returns `{llm}`. An empty body just re-checks |
| `POST /api/feedback` | `{id, verdict, note?}` | the user's verdict on a card: `used`, `ignored`, `wrong` or `good`. Appends it to the intervention log and broadcasts a `feedback` frame. `{"ok":true,"id","verdict","known"}`; `known: false` means the log has no card with that id, which is not an error |
| `POST /api/digest` | `{save?, title?, model?}` | the post-call digest: one model call over the whole call, returning `{digest, markdown, meta, path}`. See below |
| `GET /api/screen` | | what the presenter has on screen: which slide of which deck, which sheet of which workbook, and the indexed file it maps to. See below |
| `POST /api/quit` | | broadcasts `quitting`, stops audio, exits the process 0.4 s later |

`POST /api/llm` only stores models that differ from the provider's defaults, so
the defaults keep moving with the provider layer. An empty `hedge` is kept, and
means no hedge model.

### `GET /api/screen`

A read of a 1 Hz poller's last tick (`presage/screen.py`, E43). It never blocks
and never errors: everything that can go wrong is a `status`/`reason` pair.
Nothing else in the API or on the socket changed; this is the only new route,
and there is no `screen` frame on the WebSocket, so a page that wants it polls.
Poll it at 1 Hz or slower; `age` says how old the reading is.

```json
{"status": "ok", "reason": null, "app": "PowerPoint", "app_key": "powerpoint",
 "kind": "deck", "document": "Strategy_Deck.pptx", "path": "/Users/…/Strategy_Deck.pptx",
 "page": 4, "pages": 6, "sheet": null, "cell": null,
 "title": "FY26 plan by region", "presenting": false,
 "file": "Strategy_Deck.pptx", "file_score": 1.0, "loc": "slide 4",
 "via": "applescript", "focused": true, "url": null,
 "ts": 1790140762.19, "age": 0.31, "tick_ms": 7.1, "running": true, "interval_s": 1.0,
 "grants": {"accessibility": true, "window_titles": true, "locked": false}}
```

| field | is |
| --- | --- |
| `status` | `"ok"` or `"unavailable"`. Every other field may be `null` when it is not `"ok"` |
| `reason` | why, when not ok: `not_started`, `locked`, `no_document`, `no_windows`, `not_macos`, `error: …`. It also appears **with** `status: "ok"` as `no_accessibility`, which means the document resolved from the window title but the page index could not |
| `app`, `app_key` | the app showing the document: `PowerPoint`/`powerpoint`, `Keynote`, `Preview`, `Excel`, `Numbers`, `Chrome`, `Safari`, `Acrobat`, `PDF Expert`, `Skim`, `LibreOffice`, `Word`, `Pages` |
| `kind` | `deck`, `sheet`, `doc` or `browser`. A browser tab is re-kinded from the tab title, so a Google Slides tab is `deck` |
| `document` | the file name as the app shows it |
| `path` | the full path, when the app gives one (`AXDocument`, or the app's own AppleScript) |
| `page`, `pages` | the 1-based slide or page index and the total. `null` when the app does not publish it |
| `sheet`, `cell` | the active sheet, and the active cell for Excel |
| `title` | the slide's or page's own title, taken from the indexed chunk's citation when the file resolved |
| `presenting` | a slideshow is running rather than the editor |
| `file`, `file_score` | the indexed context file this document is, and how sure. `null` below 0.62; an exact path or name match is 1.0 |
| `loc` | the chunk citation head, `"slide 4"` or `"page 3"`: what `links_for(file, loc)` joins on |
| `via` | which signal answered: `ax-title`, `ax-document`, `applescript` or `cg-title` |
| `focused` | the document app is also the frontmost app; `false` means the deck is beside Zoom, which still resolves |
| `url` | the tab, for a browser |
| `age`, `tick_ms` | seconds since the reading, and how long that tick took |
| `running`, `interval_s` | whether the poller thread is alive, and its period |
| `grants` | `accessibility`, `window_titles` (Screen Recording), `locked` |

A page can show `document`, `page`/`pages` or `sheet`, and `title`. `status:
"unavailable"` should render as nothing rather than an error: a presenter with
no deck open is the normal case, not a fault. `reason: "no_accessibility"` is
the one worth surfacing, with the fix: System Settings › Privacy & Security ›
Accessibility › `+` › Presage. `PRESAGE_SCREEN=0` turns the poller off, and the
route then answers `not_started` with `running: false`.

**The poller runs with the audio session** (E43a), not with the process: it
starts when listening starts and stops when listening stops, so this route
answers `not_started` with `running: false` until the first `POST
/api/audio/start` (which auto-start also goes through). Nobody presents to a
copilot that is not listening, and a poller ticking through an idle afternoon is
reading window titles for nothing.

### `POST /api/digest`

The post-call digest (E44): what the client asked, what was answered on screen
and whether the card was used, what is still open, what the consultant promised,
objections, figures said aloud, next steps. One model call on the brain's own
provider over the whole call, so it takes seconds, not milliseconds: it is a
document to read after the call, not a card. Nothing is streamed and nothing is
broadcast. `400` with `{"error":"nothing to digest yet: no transcript this
session"}` before anything has been said.

```json
{"save": true, "title": "Acme raise", "model": null}
```

`save` (default true) writes the Markdown to
`~/.cache/presage/digests/<started_at>.md`, where `<started_at>` is the audio
session's start as local `YYYY-MM-DDTHH-MM-SS`; `path` in the response is where
it went, or `null` when `save` was false or the write failed.

```json
{"digest": {
   "summary": "One sentence.",
   "questions":  [{"line":"L42","line_id":"ts7","at":"03:41","at_s":221.4,"ts":1790123106.4,
                   "who":"Them A","speaker":"A","quote":"…the line as transcribed…",
                   "question":"What's the total startup cost?",
                   "status":"answered_on_screen","answer":"$16.4M","card":"a9c8d3d46"}],
   "open":       [ …the questions whose status is `unanswered` or `answered_without_figure`… ],
   "promises":   [{"line":"L51", …, "promise":"Send the updated model","due":"this week"}],
   "objections": [{"line":"L60", …, "objection":"…","response":"…","landed":"parked"}],
   "figures":    [{"line":"L44", …, "figure":"$16.4M","about":"total ask","card":"a9c8d3d46"}],
   "next_steps": [{"line":"L70", …, "step":"Send the deck","owner":"me"}],
   "cards":      [{"id":"a9c8d3d46","at":"00:17","question":"…","figure":"$16.4M","say":"…",
                   "answer":"…","source":"live","answered_from":"files","mark":"used",
                   "in_digest":true}],
   "counts":     {"questions":9,"promises":1,"objections":1,"figures":0,"next_steps":2,
                  "open":3,"cards_shown":8,"cards_in_digest":7},
   "dropped":    {"no_line":0,"wrong_side":0,"duplicate":0,"bad_card":0,"empty":0,"not_spoken":0}},
 "markdown": "# Post-call digest — …",
 "meta": {"prompt_tokens": 1188, "lines_total": 18, "cards": 8, "seconds": 11.8,
          "provider": "claude_code", "model": "claude-sonnet-5", "call_s": 119.1,
          "started_at": 1790123089.02, "usage": {"in":2,"out":1276,"cache_read":0,"cache_write":3780}},
 "path": "/Users/…/.cache/presage/digests/2026-09-22T20-24-49.md"}
```

Every item is anchored to a transcript line: `line` is the id the model was
given, `line_id` is the page's own transcript id for that line, and `at`, `ts`,
`who`, `speaker` and `quote` are resolved from it by the server, not written by
the model. An item that cannot be resolved never reaches the response, and
`dropped` counts what went. `status` is one of `answered_on_screen`,
`answered_by_me`, `answered_without_figure`, `unanswered`. `mark` on a card is
the intervention log's verdict (`used` / `ignored` / `wrong` / `good`),
`ignored?` for a card that was replaced with nothing said, else `unmarked`.
`figures` are numbers a person said out loud, listed and not checked; a number
that was only ever on a card is not one.

The page has no digest surface yet; `docs/frontend-asks.md` is where that ask
lives. Until then the endpoint is called by hand and read from the file.

## Settings that persist

`settings.json` at the repo root, written by the page, read at startup:

```json
{"context_dir": null, "mic_uid": null, "llm_provider": "anthropic",
 "models": null,
 "tune": {"topic_threshold": 0.6, "web_mode": "auto", "mode": "call", "…": 0},
 "auto_start_audio": true, "deep_index": true, "live_choice": "auto",
 "decider": "auto", "intent": "auto"}
```

`null` means "use the default": `context_dir` falls back to `PRESAGE_CONTEXT`
and then to `<repo>/context`, `models` to the provider's defaults, `mic_uid` to
the system default input. Anything set here wins over the environment. A Tune
slider writes `tune` one second after it stops moving.

`live_choice` is applied at startup once the provider is built, and again whenever the main
provider changes, because the routing is relative to it.

`deep_index` gates the automatic deep pass. The server schedules one after every
fast build (startup, the watcher, a reindex, a folder change), debounced three
seconds, and cancels the run in flight when the folder changes again.

`llm_provider` is `anthropic`, `gemini` or `claude_code`; the last one has no
key to store, since it uses whatever account the `claude` CLI is logged into.

`decider` is the question gate (`auto`, `jev`, `laya`, `haiku`) and `intent` is
the lever pick and the warm-card check (`auto`, `local`, `jev`, `off`). They are
two keys because they are two decisions: `auto` for the gate is Jev with a key
and the local gate otherwise, while `auto` for `intent` is **local first** --
the E21 checkpoint in `~/.cache/presage/laya/presage-gate-multi` beats Jev on
both questions -- then Jev with a key, then off. `off` means the lever pick
falls back to the word matcher and a warm card in the uncertain cosine band is
dropped rather than confirmed. Both are read at startup and applied by
`brain.set_decider`; `PRESAGE_DECIDER` and `PRESAGE_INTENT` override them for
one run, and `PRESAGE_INTENT_DIR` points the local checkpoint somewhere else.

API keys are not in that file. They live in `.env` next to it
(`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`), mode 600; saving a key rewrites only its
own line and leaves the rest of the file alone. The page only ever sees
`key_set` and the last four characters.

## Settings keys

| key | range | step | slider label |
| --- | --- | --- | --- |
| `topic_threshold` | 0 to 1 | 0.01 | Topic |
| `question_threshold` | 0 to 1 | 0.01 | Question |
| `prefetch_cooldown_s` | 0 to 30 | 1 | Cooldown |
| `max_warm` | 1 to 12 | 1 | Max warm |
| `answer_ttl_s` | 10 to 600 | 10 | Answer TTL |
| `web_mode` | `auto` or `off` | | Web |
| `screen_bias` | 0 or 1 | 1 | (not on a slider) |
| `screen_ttl_s` | 0 to 60 | 1 | (not on a slider) |
| `mode` | one of `modes[].id` | | the top-bar switcher |

`screen_bias` and `screen_ttl_s` are E43a's two, and neither is on the Tune
popover today. `screen_bias: 0` is the off switch for the whole screen prior:
the evidence goes back to what E38a packs and `hello.screen` goes back to `{}`.
`screen_ttl_s` is how old the poller's last tick may be before the brain treats
it as no screen at all, which is what stops a dead poller pinning the evidence
to a slide from ten minutes ago. They ride in the same `settings` message and
the same `set` frame as the sliders, but they are not in `TUNE_KEYS`, so a
change lasts for the run and not across a restart; `PRESAGE_SCREEN=0` is the
setting that persists, because it is the environment.

`mode` is `call` (default), `sales`, `interview`, `pitch`, `notes` or `quiet`.
The page changes it with a normal `set` frame, renders the roster from `modes`,
and the number keys 1 to 6 switch modes when the ask box is not focused and the
Settings drawer is closed. Quiet mode greys the On-deck strip, because nothing
warms in it.

`web_mode` is the one setting that is a word, not a number: `auto` lets a live
answer that found nothing in the files retry with web search, `off` never
leaves the files. It rides in the same `settings` message and the same `set`
frame as the sliders, and the page renders it as a segmented control at the
bottom of the Tune popover.

`max_warm` also caps how many On-deck chips the client keeps.

## Connection

The client dials `/ws` on load and reconnects automatically with backoff from
0.5 s to 5 s, re-rendering from `hello` each time. Messages typed while
disconnected are queued and flushed on reconnect. The top-right indicator is a
green dot plus `Live`, or red plus `reconnecting…`.

## URL parameters

| param | effect |
| --- | --- |
| `?mock=1` | No WebSocket. Runs a scripted 60 s call in the page and loops it. Used for screenshots and for working on the UI with no backend. |
| `?at=34` | With `mock=1`, starts the script 34 s in, with correct ages. Makes screenshots deterministic. |
| `?deck=top` | Moves the On-deck strip above the answer stack. Tried and rejected: it pushes the big answer down and stacks two labels. Left in as a switch. |
| `?settings=1` | Opens the Settings drawer on load. Used for screenshots; ⌘, and the gear button do the same thing. |
| `?theme=dark` | `dark`, `light` or `system` for this load only, without saving it. Settings → Appearance saves the choice in the browser. |
| `?gallery=1` | No socket. Renders every component in every state with the page's own render functions: the component gallery behind `docs/design/`. |
| `?mode=notes` | With `mock=1`, starts the scripted call in that mode. |
| `?at=90` (stub replay) | The page puts its own query string on the socket URL. The stub's replay mode (`PRESAGE_STUB_REPLAY`, a simulator recording) reads `at` from it and lands everything before 90 s at once. The real server ignores it. |

## Notes for the real server

- Send `hello` before anything else on every connection. Nothing else is
  buffered or replayed.
- `id` is the identity for transcript lines, candidates and answers. Reusing one
  is an update; a new one is a new row or card.
- Timestamps only have to be self-consistent. Send `now` in `hello` and ages are
  correct regardless of the client's clock.
- No client-side persistence. A reload shows exactly what `hello` carries, so
  `hello` should carry enough to be useful mid-call.
