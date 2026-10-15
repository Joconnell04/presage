"""presage.sim - the call simulator: scripted two-voice calls, clips, and a scored run.

Three steps, each its own module and its own command:

    scripts.py   Sonnet writes a call script from a context folder's answer key. Every
                 client turn carries the outcome presage should produce: a card or no
                 card, which figure, which cells, what kind of line it is.
    render.py    macOS `say` turns the script into two 16 kHz mono wavs (Them and Me) on
                 one timeline, plus the timeline JSON the scorer measures against.
    run.py       starts a server on the folder with the Them wav, records every WebSocket
                 frame, then scores: first card per question, wrong cards, misses, gate
                 decisions, cost.

Everything a later experiment needs to be comparable lives in the script: the same clip,
the same expectations, the same scorer. `docs/experiments/E5-simulator.md` is the baseline.
"""

__all__ = ["scripts", "render", "run", "answer_key"]
