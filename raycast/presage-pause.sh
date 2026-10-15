#!/bin/bash
# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Pause Presage
# @raycast.mode silent
# Optional parameters:
# @raycast.icon ⏸️
# @raycast.packageName Presage
# Documentation:
# @raycast.description Stop listening, whatever state it is in. Stays off across relaunches until you press Listen.
# @raycast.author Jackson O'Connell
PORT="${PRESAGE_PORT:-8766}"
if curl -s --max-time 5 -X POST "http://localhost:$PORT/api/audio/stop" >/dev/null; then echo "presage: paused (not listening)"; else echo "presage is not running"; fi
