# presage design system

The rules behind `ui/index.html`. The page is the implementation: its `:root` block between
`tokens:begin` and `tokens:end` is the source of every value below, `tokens.css` here is a copy
of it, and `/?gallery=1` renders every component in every state with the page's own render
functions. Add `&theme=light` for the light variant.

![The page mid-call, dark](../screenshots/presage-call.png)

## The glance rule

A card is read in 1.5 seconds while the consultant is talking. Everything on the answer side
serves that.

1. **Figure first**, 62 px, at a fixed offset from the card top.
2. **Say-line** under it, 32 px: the 4 to 8 words the consultant can speak as they stand.
3. **Basis last**: citation chips and the status line, pinned to the bottom of the card.
4. **Detail only on More** (↵ or a click): the full 25-word answer and the untruncated question.

The top card has a 300 px minimum height, so the figure, the say-line and the rows below all sit
in the same place card after card. A card with no figure puts the say-line first; a card with no
say-line shows the answer at 24 px. Older answers collapse to 34 px rows (figure, say-line,
provenance, age) so about ten stay visible under the top card.

## Layout

| region | size | holds |
| --- | --- | --- |
| header | 48 px | brand, mode switcher, status group (Mic, System, STT, the slide on screen, files), Tune, Clear, Settings, connection |
| answers column | the rest | Answers label (with the checks flag at its right), top card, rows, On deck strip |
| right rail | `clamp(360px, 31vw, 500px)` | Scratch (only when it has rows), then the Transcript panel with a Notes tab once notes exist |
| footer | 60 px | ask box (⌘K) and the action bar of keys that do something right now |

Notes mode moves the notes into the answers column above the cards, because there they are the
page. Past 1760 px of content (`--max-w`) the page stops growing and centres, header and footer
included, so a row's say-line and its provenance never end up a screen apart. Under 1100 px wide
the columns stack and the meters and action bar hide.

## Colour means state

Colour is never decoration. Grey carries everything else.

| token | means |
| --- | --- |
| `--ok` green | from the files, ready, listening, marked used |
| `--warm` amber | warming, suggested by the model, held while you speak |
| `--bad` red | not in files, unverified figure, error, marked wrong |
| `--web` sky | answered from the web, web citation |
| `--derived` orange | a number the workbook recalculated, the scratch values |
| `--accent` indigo | Me, focus, selection, a pending ask, the active tab |

A filled tag is reserved for the two things the consultant must not read out: **Not in files**
and **Unverified**. Every other provenance is a dot and a word.

## Tokens

Dark is the default. Light is a full second set on `:root[data-theme="light"]`; the choice is
Settings → Appearance (Dark, Light, System), kept in the browser, and applied by a two-line
script before first paint so there is no flash.

| token | dark | light | use |
| --- | --- | --- | --- |
| `--bg` | `#0b0b0d` | `#f4f4f6` | page |
| `--surface-1` | `#111114` | `#ffffff` | panels, chips, inputs, hovered row |
| `--surface-2` | `#16161a` | `#ffffff` | top card, popover |
| `--surface-3` | `#1e1e23` | `#ececf0` | keycaps, hover, meter track |
| `--surface-sel` | `#24242b` | `#e6e8f6` | selected row |
| `--border` | `#222228` | `#e1e1e7` | every 1 px edge |
| `--border-strong` | `#2e2e36` | `#d0d0d9` | top card, popover, hover edges |
| `--divider` | `#1a1a1f` | `#ebebef` | lines inside a panel |
| `--text-1` | `#f5f5f8` | `#0f0f12` | figure, say-line, Me |
| `--text-2` | `#d2d2da` | `#2a2a30` | body, transcript |
| `--text-3` | `#8d8d98` | `#686873` | secondary, meta |
| `--text-4` | `#70707b` | `#7e7e88` | timestamps, placeholders, empty states (4.0:1 and 3.7:1 on `--bg`) |
| `--accent` | `#7c8cff` | `#4f5fe8` | see above |
| `--ok` `--warm` `--bad` `--web` `--derived` | `#5ee0b1` `#f0b35a` `#f4676d` `#5ab6ff` `#ff9151` | `#0a7c59` `#946009` `#c42b41` `#15679f` `#b04a0c` | state; used as text too, so each clears 4.3:1 on the selected row |
| `--*-soft` | 13 to 14% alpha | 9 to 11% alpha | tag and highlight fills |

**Type.** SF Pro through the system stack, SF Mono for working lines. No web fonts.

| token | size / weight | use |
| --- | --- | --- |
| `--fs-figure` | 62 / 700, -0.034em, tabular | top card figure |
| `--fs-say` | 32 / 600, -0.022em | top card say-line |
| `--fs-lede` | 24 / 550 | top card answer when there is no say-line |
| `--fs-detail` | 16 | the answer behind More |
| `--fs-row` | 14.5 / 650, tabular | row figure |
| `--fs-body` | 13.5 | transcript, row say-line |
| `--fs-ui` | 12.5 | buttons, chips, scratch |
| `--fs-sm` | 12 | drawer labels, hints, tabs, file names |
| `--fs-meta` | 11.5 | timing, ages, status |
| `--fs-xs` | 11 | keycaps, timestamps, file counts |
| `--fs-cap` | 10.5 / 650, 0.09em, uppercase | section labels |

Every `font-size` in the page's CSS uses one of these, except tags (10 px), file-kind badges (8.5 px) and the narrow-screen overrides.

**Space** is a 4 px grid (`--s1` 2 px to `--s10` 32 px). **Radii**: 4 keycap-small, 6 tags and
chips inside cards, 8 buttons, inputs and rows, 12 panels, 14 the top card. **Sizes**: bar 48,
row 34, control 28, keycap 20.

**Elevation.** Dark surfaces separate by border and a step in fill, not shadow. `--e1` is the top
card's faint inner highlight and long soft shadow, `--e2` the popover, `--e3` the drawer.

**Motion.** `--d1` 90 ms for hover, `--d2` 160 ms for popovers and the scrim, `--d3` 220 ms for a
card arriving (6 px drop and fade), `--d4` 620 ms for the web swap outline. One curve,
`--ease` `cubic-bezier(.2,.7,.3,1)`. `prefers-reduced-motion` turns all of it off.

## Components

**Top card.** Question (13 px, one line), figure, say-line, working line (mono, orange rule) when
the number was computed, detail when open, basis chips, status line. Status line order:
provenance, verdict tag, timing ("Prefetched 840 ms", "Instant 5 ms", "Live 2.2 s", "Manual"),
mode when not Call, feedback tag, age at the far right. Variants: miss (red-tinted border, muted
text), unverified (figure struck through in red), writing (a `partial` frame: the figure a shade
dimmer, the streamed sentence grey, "Writing" with a pulsing accent dot, no timing yet), heard and
pending (below), still looking and no answer (below), empty (dashed outline with the mode's hint).

**Heard slot (E68).** The moment the server registers a question it will answer, the top card
becomes a slot under the answer's own id: "They asked" or "You asked", a placeholder where the
figure will land, the question itself at 24 px where the say-line will be, and "Heard" with a
pulsing accent dot. The answer replaces it in place. A typed ask shows the same card with
"Asking" until the server registers it. A question the server will not answer (a `check` or
`heard` policy) opens no slot.

**No answer (E68).** A question the server has nothing for yet says why instead of pretending the
files lack it. `slow`: "Still looking" with a pulsing amber dot and "No answer yet" in grey; the
answer can still replace it. `empty`, `error`, `timeout`: a grey "No answer" tag, the reason in its
tooltip. Only `none` gets the red Not in files treatment. An answer to the consultant's own
question carries a small accent "Me" after the provenance.

**Figures.** A bare number from the server ("27835848.58") prints with thousands separators and
no decimals it cannot show ("27,835,849"); anything with a unit stays as sent. Past 14
characters the figure drops to 78% size, past 22 to 58%. The say-line stops at two lines and an
answer at three; More shows the rest.

**Row.** 96 px figure column, say-line (or answer), provenance at the right, age. Hover or
selection swaps the age for U and W keycaps. Selected: `--surface-sel`. Open: the row grows in
place into question, answer, working, basis and timing, on a bordered panel.

**Provenance.** Six-pixel dot plus one word: Files, Web, Derived, Suggested. Not in files is the
red tag.

**Tags.** 20 px, uppercase 10 px. Filled red: Not in files, Unverified. Outlined: Uncited
(grey), Used (green), Wrong (red), Ignored (grey). Filled amber: "1 held".

**Candidate chip.** 28 px, at most 300 px wide (220 while warming), every part clipped inside it.
The server's `topic` joins up to three workbook labels and can run past a thousand characters, so
the chip never prints it whole. Ready: green dot, the figure in bold, then the question the card
would answer. Warming: pulsing amber dot, the first topic in grey, "+2" when more are joined, not
clickable. Shown: accent border and pulsing accent dot until its answer lands. The question, the
answer and every topic are in the tooltip. Ready chips sort first and two with the same figure
and question show once. A new chip rises 3 px into place. The strip fades at the right edge.

**Checks flag.** A correction (the consultant said a figure the files contradict) or a stale
slide (a page on screen quotes a number the model has moved past) is not an answer, so it never
takes the top card. It sits at the right of the Answers label as a 22 px flag: red rule for a
correction, amber for a stale slide, then what was said or printed, an arrow and the model's
figure, "+N" when more are waiting. Click opens a popover with each check: kind, age, the
say-line to speak, the comparison, the working line, the citations, and U / W. Checks are never
held while Me speaks.

**Screen chip.** "Slide 4/12", "Page 3/9" or the sheet name, in the status group, from
`hello.screen` and a 2 s poll of `GET /api/screen`. Nothing open draws nothing. Amber when
Accessibility is missing, with the fix in the tooltip; accent while presenting.

**Transcript line.** `m:ss` stamp, speaker (Me in accent, Them or Them A/B in grey), text.
Interim lines are grey; the unstable tail is dimmer and italic. A line the transcriber's words
were repaired on (`heard`) has a dotted underline; the hover says what was heard. A line the server
registered as a question has a 5 px dot in the gutter left of its stamp: grey while heard, green
once answered, amber when it got a no-answer card. The list fades at its top edge.

**Scratch row.** Lever, value (orange, right-aligned in a fixed column so values line up),
arrow, output (wraps instead of truncating, so the number at the end is never cut). New rows
flash `--derived-soft` once. Tooltip: the working line.

**Notes.** Summary, Actions (amber bullets), Open (accent bullets). Three columns in Notes mode,
stacked in the rail tab. The tab shows an accent dot when notes changed behind it.

**Controls.** Button (28 px; on, primary, danger, "Sure?" confirm, disabled), segmented control
(mode switcher, tabs, Settings choices), keycap, clickable hint, dropdown and text field (30 px,
accent focus ring), popover (Tune), drawer (Settings, 420 px), banner (Not listening, with
Start), meters (3 px bars scaled with `transform`), status dots.

## States

| state | where | how it looks |
| --- | --- | --- |
| warming | candidate chip | amber dot pulsing, grey topic, not clickable |
| ready | candidate chip | green dot, preview shown, click promotes it |
| shown | candidate chip | accent border and dot until the answer arrives and the chip expires |
| expired | candidate chip | removed |
| pending | top card | question, two shimmer bars, "Asking"; the ask box spins |
| writing | top card | a `partial` answer: dimmer figure, grey sentence, pulsing "Writing" |
| heard | top card, row | a registered question waiting for its card: the question as the lede, pulsing "Heard" |
| still looking | top card, row | a `slow` no-answer card: "No answer yet" in grey, pulsing amber "Still looking" |
| check | Answers label | a correction (red) or stale slide (amber) flag; the popover holds the detail |
| error | banner, dots, drawer notes | red dot and the reason; audio errors show the banner with Start |
| web | card, row | sky dot "Web", sky citation chips that open in a new tab; a files miss turning into a web answer flashes a sky outline |
| derived | card, row, scratch | orange dot "Derived", mono working line, the value in Scratch |
| held | Answers label | amber "N held" while Me is speaking; cards land at the pause or after 4 s |

## Keyboard

| key | does |
| --- | --- |
| ⌘K | focus the ask box |
| ↵ | in the ask box: ask. Otherwise: More / Less on the selected card |
| ↑ ↓ | move the selection through the answers |
| U, W | mark the selected card used or wrong (`POST /api/feedback`) |
| Esc | close the drawer or Tune, leave the ask box, or collapse and reselect the top card |
| 1 to 6 | switch mode |
| ⌘, | Settings |

Single keys do nothing while a text field has focus or the drawer is open. The footer always
shows the keys that apply right now, and each hint is clickable.

## Rules for anything new

- Labels three words or fewer; buttons are verbs; no help paragraphs in the chrome. Paths and
  status detail go in a tooltip.
- No `alert`, `confirm` or `prompt`. Destructive actions confirm inline: the button turns into
  "Sure?" for one more click.
- A new colour needs a new state in the table above, or it is grey.
- Check both themes in the gallery before shipping.

## Performance

The page is one file with no dependencies and no build, so first paint needs a single request.

- Status frames arrive every 700 ms (every second from the real server). They go through
  `put()`, which writes to the DOM only when a value changed; meters move with `transform`; the
  drawer's sections redraw only while it is open.
- An answer frame redraws the top card and patches rows by id (`renderRows`): a streaming answer
  that re-sends the same id rebuilds one card, not the list, and does not re-animate.
- A transcript frame inserts or replaces one line. The list is rebuilt only when the call clock's
  origin moves.
- The age ticker runs once a second and only rewrites text that changed.

## Files

| file | what |
| --- | --- |
| `../../ui/index.html` | the page; tokens between `tokens:begin` and `tokens:end` |
| `tokens.css` | copy of that block. Check it: `diff <(sed -n '/tokens:begin/,/tokens:end/p' ui/index.html) <(tail -n +3 docs/design/tokens.css)` |

**Real data.** The scripted mock is tidy by construction. Check anything that renders server text
against a recorded call: `PRESAGE_STUB_REPLAY=<your run>/frames.jsonl uv run
python -m presage.server_stub` (recordings are not shipped in this repo), then open `/?at=110` to land everything before 110 s at once. The
recordings carry thousand-character topics, raw figures, partial answers and late interims.
`?rec=sample-02-churn` switches recording, `?speed=20` replays twenty times faster (a whole call
at 20× produces no long task in the page).

Screenshots are headless Chrome at 1440 × 900 (the gallery at 1440 wide, full height):

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=1440,900 --virtual-time-budget=6000 --screenshot=out.png "http://localhost:8765/?mock=1&at=54"
```
