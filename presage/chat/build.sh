#!/bin/sh
# Builds presage/bin/axchat from axchat.swift. Needs only CommandLineTools; no Xcode.app.
#
# Same rules as native/audiotap/build.sh: the build re-signs the binary ad hoc with a new
# hash, so a TCC grant given to this binary directly is thrown away by a rebuild. The
# grant that matters is the one on the app that launches it (~/Applications/Presage.app),
# which a rebuild here does not touch. Pass --force to rebuild anyway.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$REPO/bin/axchat"
SRC="$HERE/axchat.swift"

if [ "$1" != "--force" ] && [ -f "$OUT" ] && [ "$OUT" -nt "$SRC" ]; then
  echo "up to date: $OUT"
  exit 0
fi

mkdir -p "$REPO/bin"
TMP="$OUT.new.$$"
trap 'rm -f "$TMP"' EXIT
swiftc -O -swift-version 5 \
  -framework ApplicationServices -framework AppKit -framework Foundation -framework Quartz \
  -o "$TMP" "$SRC"
mv -f "$TMP" "$OUT"
trap - EXIT
echo "built $OUT"
