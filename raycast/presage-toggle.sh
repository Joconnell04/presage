#!/bin/bash
# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Toggle Presage
# @raycast.mode silent
# Optional parameters:
# @raycast.icon 🎙️
# @raycast.packageName Presage
# Documentation:
# @raycast.description Listening on/off. Shows the state it left presage in.
# @raycast.author Jackson O'Connell
PORT="${PRESAGE_PORT:-8766}"
state=$(curl -s --max-time 2 "http://localhost:$PORT/api/state" | python3 -c 'import sys,json; print((json.load(sys.stdin).get("audio") or {}).get("state",""))' 2>/dev/null)
if [ -z "$state" ]; then echo "presage is not running"; exit 0; fi
if [ "$state" = "running" ] || [ "$state" = "starting" ]; then
  curl -s --max-time 5 -X POST "http://localhost:$PORT/api/audio/stop" >/dev/null && echo "presage: paused (not listening)"
else
  curl -s --max-time 8 -X POST -H 'Content-Type: application/json' -d '{}' "http://localhost:$PORT/api/audio/start" >/dev/null && echo "presage: listening"
fi
