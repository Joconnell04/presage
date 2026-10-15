#!/bin/sh
# One command: install deps (first run only), build the audio helper, start the server.
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null 2>&1 || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
# The local gate (Core ML, plus torch for the shapes the fixed package cannot take) ships as extras;
# a plain `uv sync` removes them and the gate silently falls back to the hosted decider.
uv sync -q --extra laya --extra laya-fast
[ -x bin/audiotap ] || { [ -f native/audiotap/build.sh ] && sh native/audiotap/build.sh; }
exec uv run python -m presage.server "$@"
