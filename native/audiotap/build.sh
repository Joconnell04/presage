#!/bin/sh
# Builds presage/bin/audiotap. Needs only CommandLineTools; no Xcode.app, no packages.
#
# Skips the build when the binary is already newer than the source. Each build re-signs
# the binary ad hoc with a new hash, and if you ever grant this binary a TCC permission
# directly (rather than inheriting it from the app that launches it), a rebuild throws
# that grant away. Pass --force to rebuild anyway.
#
# Links into a temp file and renames, so rebuilding while a capture is running replaces
# the file without disturbing the running process's mapped image.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$REPO/bin/audiotap"
SRC="$HERE/main.swift"

if [ "$1" != "--force" ] && [ -f "$OUT" ] && [ "$OUT" -nt "$SRC" ]; then
  echo "up to date: $OUT"
  exit 0
fi

mkdir -p "$REPO/bin"
TMP="$OUT.new.$$"
trap 'rm -f "$TMP"' EXIT
swiftc -O -swift-version 5 \
  -framework CoreAudio -framework AVFoundation -framework Foundation \
  -o "$TMP" "$SRC"
mv -f "$TMP" "$OUT"
trap - EXIT
echo "built $OUT"
