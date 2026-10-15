"""E1: the gate, faster. `docs/experiments/E1-gate-speed.md` is the write-up.

`FastLayaDecider` is a drop-in for `presage.llm.laya.LayaDecider` backed by a Core ML export of the
same weights. Nothing here imports torch at run time; `coreml.py --export` needs it, the decider
does not.
"""
from presage.llm.laya_fast.decider import FastLayaDecider

__all__ = ["FastLayaDecider"]
