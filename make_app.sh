#!/bin/sh
# Builds a double-clickable Presage.app that runs ./run.sh from this checkout and opens the page.
#
# Why an app bundle: macOS attaches the Microphone and Screen & System Audio Recording grants to
# the app that launches the capture, so with this wrapper the grants belong to "Presage" instead
# of Terminal, the mic prompt names Presage, and the app shows up as its own row in System
# Settings. The bundle is signed ad hoc; its identity changes on every rebuild, which throws the
# grants away, so this script does nothing when the app already exists (pass --force to rebuild).
#
#   ./make_app.sh                 -> ~/Applications/Presage.app
#   ./make_app.sh /path/Presage.app
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
FORCE=0
for a in "$@"; do [ "$a" = "--force" ] && FORCE=1; done
APP="$HOME/Applications/Presage.app"
for a in "$@"; do [ "$a" != "--force" ] && APP="$a"; done

if [ -d "$APP" ] && [ "$FORCE" = "0" ]; then
  echo "exists: $APP (use --force to rebuild; rebuilding resets its permission grants)"
  exit 0
fi

mkdir -p "$(dirname "$APP")"


# The launcher is an AppleScript applet rather than a bare shell script: a script inside a bundle
# never gets the "access files in your Downloads folder" prompt (the checkout lives there), it
# just fails with "Operation not permitted", while an applet's `do shell script` runs as a proper
# app process and macOS asks. The applet is Apple's signed binary with our Info.plist and icon.
rm -rf "$APP"
SRC="$(mktemp -t presage-launcher).applescript"
cat > "$SRC" <<'EOF'
do shell script "/bin/sh " & quoted form of (POSIX path of (path to resource "launch.sh"))
EOF
osacompile -o "$APP" "$SRC"
rm -f "$SRC"
cat > "$APP/Contents/Resources/launch.sh" <<EOF
#!/bin/sh
# Written by make_app.sh; the checkout it runs is baked in below.
REPO="$HERE"
LOG="\$HOME/Library/Logs/presage.log"
mkdir -p "\$(dirname "\$LOG")"
export PATH="\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
export PRESAGE_OPEN_BROWSER=1
echo "=== \$(date) launcher start ===" >> "\$LOG"
ls "\$REPO" >/dev/null 2>&1 || { echo "=== \$(date) cannot read \$REPO (Files and Folders permission) ===" >> "\$LOG"; exit 1; }
cd "\$REPO" || exit 1
if pgrep -f "python[0-9.]* -m presage.server" >/dev/null 2>&1; then
  echo "=== \$(date) already running, opening the page ===" >> "\$LOG"
  open "http://localhost:\${PRESAGE_PORT:-8766}"
  exit 0
fi
echo "=== \$(date) launching from \$REPO ===" >> "\$LOG"
nohup ./run.sh >> "\$LOG" 2>&1 &
EOF
chmod +x "$APP/Contents/Resources/launch.sh"
PLIST="$APP/Contents/Info.plist"
setkey () { /usr/libexec/PlistBuddy -c "Set :$1 $3" "$PLIST" >/dev/null 2>&1 || /usr/libexec/PlistBuddy -c "Add :$1 $2 $3" "$PLIST"; }
setkey CFBundleIdentifier string com.github.joconnell04.presage
setkey CFBundleName string Presage
setkey CFBundleDisplayName string Presage
setkey LSUIElement bool true
setkey CFBundleIconFile string Presage
setkey NSMicrophoneUsageDescription string "Presage transcribes your side of the call from the microphone."
setkey NSAudioCaptureUsageDescription string "Presage transcribes the other side of the call from system audio."
rm -f "$APP/Contents/Resources/applet.icns"

# Icon: a dark rounded tile with an accent ring, drawn with stdlib-only PNG writing.
python3 - "$APP/Contents/Resources" <<'EOF'
import math, struct, sys, zlib, subprocess, os, shutil
out = sys.argv[1]
def png(size):
    rows = []
    c = size / 2; r_tile = size * 0.46; rad = size * 0.11; ring = size * 0.20; w = size * 0.045
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            dx, dy = abs(x - c), abs(y - c)
            qx, qy = max(dx - (r_tile - rad), 0), max(dy - (r_tile - rad), 0)
            inside = math.hypot(qx, qy) <= rad
            if not inside:
                row += b"\x00\x00\x00\x00"; continue
            d = math.hypot(x - c, y - c)
            on_ring = abs(d - ring) <= w / 2
            dot = math.hypot(x - c - ring * 0.95, y - c) <= w * 1.1
            if on_ring or dot:
                row += bytes([0x7c, 0x8c, 0xff, 0xff])
            else:
                row += bytes([0x14, 0x14, 0x17, 0xff])
        rows.append(bytes(row))
    raw = b"".join(rows)
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
iconset = os.path.join(out, "Presage.iconset"); os.makedirs(iconset, exist_ok=True)
for s in (16, 32, 64, 128, 256, 512):
    open(os.path.join(iconset, f"icon_{s}x{s}.png"), "wb").write(png(s))
    open(os.path.join(iconset, f"icon_{s//2}x{s//2}@2x.png"), "wb").write(png(s)) if s >= 32 else None
open(os.path.join(iconset, "icon_512x512@2x.png"), "wb").write(png(1024))
subprocess.run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(out, "Presage.icns")], check=True)
shutil.rmtree(iconset)
EOF

codesign --force --deep -s - "$APP" >/dev/null 2>&1 || true
echo "built $APP  (double-click it, or: open \"$APP\")"
echo "first launch: allow the microphone when asked, then add Presage under"
echo "System Settings > Privacy & Security > Screen & System Audio Recording and launch again"
