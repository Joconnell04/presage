#!/bin/bash
# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Presage Listen
# @raycast.mode silent
# Optional parameters:
# @raycast.icon 🟢
# @raycast.packageName Presage
# Documentation:
# @raycast.description Start listening (mic and system audio).
# @raycast.author Jackson O'Connell
PORT="${PRESAGE_PORT:-8766}"
if curl -s --max-time 8 -X POST -H 'Content-Type: application/json' -d '{}' "http://localhost:$PORT/api/audio/start" >/dev/null; then echo "presage: listening"; else echo "presage is not running"; fi
