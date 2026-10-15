// axchat: read a video-call app's chat panel through the macOS Accessibility API.
//
// Built to presage/bin/axchat by presage/chat/ax.py (same pattern as native/audiotap).
// Needs only CommandLineTools. The Accessibility grant belongs to the app that launches
// this binary, so running it under ~/Applications/Presage.app makes the System Settings
// row say "Presage"; run from a terminal and the row says the terminal.
//
//   axchat --list                 every running app with windows, and their window titles
//   axchat --dump <app> [depth]   the AX subtree of that app's windows, for tuning the locator
//   axchat --watch [--app zoom|teams|auto] [--interval 0.5]
//
// --watch prints one JSON object per line on stdout. It does no parsing of its own: it
// finds the chat container, reads every text leaf under it in visual order, and prints
// the whole ordered list whenever it changes. Grouping leaves into messages is done in
// presage/chat/parse.py, because that is the part that needs to change when Zoom moves a
// label, and Python is where it can be tested against a fixture.
//
//   {"event":"leaves","app":"zoom","pid":123,"window":"Zoom Meeting","t":1758196800.12,
//    "ms":3.4,"items":[{"r":"AXStaticText","v":"Walt De Kock","x":40,"y":120}, ...]}
//   {"event":"target","app":"zoom","window":"Chat","path":"AXWindow>AXScrollArea","leaves":14}
//   {"event":"no_target","app":"zoom","reason":"no window with a chat container"}
//   {"event":"locked"}         the screen is locked: macOS stubs every window tree
//   {"event":"untrusted"}      no Accessibility grant; nothing can be read
//
// stderr carries human-readable notes only.

import Foundation
import ApplicationServices
import AppKit

// ----------------------------------------------------------------------------- AX helpers

@inline(__always) func copyAttr(_ el: AXUIElement, _ name: String) -> CFTypeRef? {
    var v: CFTypeRef?
    guard AXUIElementCopyAttributeValue(el, name as CFString, &v) == .success else { return nil }
    return v
}

func axString(_ el: AXUIElement, _ name: String) -> String? {
    guard let v = copyAttr(el, name), CFGetTypeID(v) == CFStringGetTypeID() else { return nil }
    let s = (v as! NSString) as String
    return s.isEmpty ? nil : s
}

func axElements(_ el: AXUIElement, _ name: String) -> [AXUIElement] {
    guard let v = copyAttr(el, name), CFGetTypeID(v) == CFArrayGetTypeID() else { return [] }
    let cf = unsafeBitCast(v, to: CFArray.self)
    var out: [AXUIElement] = []
    out.reserveCapacity(CFArrayGetCount(cf))
    for i in 0..<CFArrayGetCount(cf) {
        guard let p = CFArrayGetValueAtIndex(cf, i) else { continue }
        let e = Unmanaged<AXUIElement>.fromOpaque(p).takeUnretainedValue()
        if CFGetTypeID(e) == AXUIElementGetTypeID() { out.append(e) }
    }
    return out
}

func axPoint(_ el: AXUIElement, _ name: String) -> CGPoint? {
    guard let v = copyAttr(el, name), CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
    let av = unsafeBitCast(v, to: AXValue.self)
    var p = CGPoint.zero
    return AXValueGetValue(av, .cgPoint, &p) ? p : nil
}

/// The text a leaf carries, whichever attribute the app chose to put it in.
func leafText(_ el: AXUIElement) -> String? {
    if let v = copyAttr(el, kAXValueAttribute as String), CFGetTypeID(v) == CFStringGetTypeID() {
        let s = (v as! NSString) as String
        if !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return s }
    }
    for k in [kAXTitleAttribute as String, kAXDescriptionAttribute as String] {
        if let s = axString(el, k), !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return s }
    }
    return nil
}

func children(_ el: AXUIElement) -> [AXUIElement] { axElements(el, kAXChildrenAttribute as String) }
func role(_ el: AXUIElement) -> String { axString(el, kAXRoleAttribute as String) ?? "?" }

// ----------------------------------------------------------------------------- state

func screenLocked() -> Bool {
    guard let d = CGSessionCopyCurrentDictionary() as? [String: Any] else { return false }
    return (d["CGSSessionScreenIsLocked"] as? Int) == 1
}

/// Writes one JSON line. A closed pipe means the parent is gone, so exit rather than
/// keep walking another app's UI forever: SIGTERM to the Python parent does not reach a
/// child, and an orphaned reader is the one failure mode a user would never notice.
func emit(_ obj: [String: Any]) {
    guard let d = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) else { return }
    do {
        try FileHandle.standardOutput.write(contentsOf: d)
        try FileHandle.standardOutput.write(contentsOf: Data("\n".utf8))
    } catch {
        exit(0)
    }
}

/// True once the parent has exited and this process has been reparented to launchd.
func orphaned() -> Bool { getppid() <= 1 }

func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

// ----------------------------------------------------------------------------- targets

struct Target {
    let app: String            // "zoom" | "teams"
    let pid: pid_t
    let window: String
    let container: AXUIElement
    let path: String
}

/// Bundle-id prefixes per app. Zoom runs the in-meeting UI in its own process on some
/// builds (us.zoom.ZoomMeeting), so both are candidates and whichever has the chat
/// container wins.
let BUNDLES: [String: [String]] = [
    "zoom":  ["us.zoom.xos", "us.zoom.zoommeeting", "us.zoom.zoomhybridconf"],
    "teams": ["com.microsoft.teams", "com.microsoft.teams2"],
]

/// Chromium builds its web accessibility tree only once it thinks an assistive client is
/// watching, and the switch is `AXEnhancedUserInterface` set on the application element.
/// Electron apps take `AXManualAccessibility` instead; Teams is a WebView2/Edge build and
/// answers kAXErrorAttributeUnsupported to that one, so both are set and neither is
/// checked. Zoom's own Prism toolkit reads `AXEnhancedUserInterface` too.
func wakeAccessibility(_ ax: AXUIElement) {
    _ = AXUIElementSetAttributeValue(ax, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
    _ = AXUIElementSetAttributeValue(ax, "AXManualAccessibility" as CFString, kCFBooleanTrue)
}

func runningApps(_ want: String) -> [NSRunningApplication] {
    let wanted = want == "auto" ? Array(BUNDLES.keys) : [want]
    var out: [NSRunningApplication] = []
    for app in NSWorkspace.shared.runningApplications {
        let bid = (app.bundleIdentifier ?? "").lowercased()
        for w in wanted where BUNDLES[w]!.contains(where: { bid.hasPrefix($0) }) {
            out.append(app)
            break
        }
    }
    return out
}

func appKind(_ app: NSRunningApplication) -> String {
    let bid = (app.bundleIdentifier ?? "").lowercased()
    for (kind, prefixes) in BUNDLES where prefixes.contains(where: { bid.hasPrefix($0) }) { return kind }
    return "?"
}

let CHAT_WORDS = ["chat", "message", "conversation", "in-meeting"]

func looksLikeChat(_ el: AXUIElement) -> Bool {
    for k in ["AXIdentifier", kAXTitleAttribute as String, kAXDescriptionAttribute as String,
              kAXHelpAttribute as String, kAXRoleDescriptionAttribute as String] {
        guard let s = axString(el, k)?.lowercased() else { continue }
        for w in CHAT_WORDS where s.contains(w) { return true }
    }
    return false
}

let CONTAINER_ROLES: Set<String> = ["AXScrollArea", "AXList", "AXTable", "AXOutline", "AXGroup", "AXWebArea"]

/// Every text leaf under `el`, in visual order (top to bottom, then left to right).
func leaves(_ el: AXUIElement, _ budget: inout Int) -> [(String, String, CGPoint)] {
    if budget <= 0 { return [] }
    budget -= 1
    var out: [(String, String, CGPoint)] = []
    let kids = children(el)
    if kids.isEmpty {
        if let t = leafText(el) {
            out.append((role(el), t, axPoint(el, kAXPositionAttribute as String) ?? .zero))
        }
        return out
    }
    // a container that also carries its own text (Chromium does this) contributes it once
    let r = role(el)
    if r == "AXStaticText" || r == "AXTextArea", let t = leafText(el) {
        out.append((r, t, axPoint(el, kAXPositionAttribute as String) ?? .zero))
        return out
    }
    for c in kids { out.append(contentsOf: leaves(c, &budget)) }
    return out
}

func countLeaves(_ el: AXUIElement, _ budget: inout Int) -> Int {
    var b = budget
    let n = leaves(el, &b).count
    budget = b
    return n
}

/// Best chat container inside one window: the deepest element that names itself chat and
/// holds text, else the container role with the most text leaves.
func findContainer(_ win: AXUIElement) -> (AXUIElement, String)? {
    var best: (AXUIElement, String, Int, Bool)? = nil        // el, path, leaves, named
    var stack: [(AXUIElement, String, Int)] = [(win, role(win), 0)]
    var visited = 0
    while let (el, path, depth) = stack.popLast() {
        visited += 1
        if visited > 4000 || depth > 24 { continue }
        let r = role(el)
        if CONTAINER_ROLES.contains(r) {
            var budget = 1200
            let n = countLeaves(el, &budget)
            let named = looksLikeChat(el)
            if n >= 1 {
                let better: Bool
                if let b = best {
                    // a container that names itself chat beats a bigger anonymous one
                    better = (named && !b.3) || (named == b.3 && n > b.2)
                } else { better = true }
                if better { best = (el, path, n, named) }
            }
        }
        for c in children(el) { stack.append((c, path + ">" + role(c), depth + 1)) }
    }
    guard let b = best else { return nil }
    return (b.0, b.1)
}

func findTargets(_ want: String) -> [Target] {
    var out: [Target] = []
    for app in runningApps(want) {
        let ax = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(ax, 2.0)
        wakeAccessibility(ax)
        let wins = axElements(ax, kAXWindowsAttribute as String)
        for w in wins {
            if role(w) != "AXWindow" { continue }        // a stubbed tree returns the app element
            let title = axString(w, kAXTitleAttribute as String) ?? ""
            guard let (c, path) = findContainer(w) else { continue }
            out.append(Target(app: appKind(app), pid: app.processIdentifier,
                              window: title, container: c, path: path))
        }
    }
    // a window that calls itself Chat first, then the one with the most text
    return out.sorted { a, b in
        let ac = a.window.lowercased().contains("chat"), bc = b.window.lowercased().contains("chat")
        return ac && !bc
    }
}

// ----------------------------------------------------------------------------- commands

let args = CommandLine.arguments
func flag(_ name: String, _ fallback: String) -> String {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return fallback }
    return args[i + 1]
}

if !AXIsProcessTrustedWithOptions([kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: false] as CFDictionary) {
    emit(["event": "untrusted"])
    note("no Accessibility grant: System Settings > Privacy & Security > Accessibility")
    if !args.contains("--watch") { exit(3) }
}

if args.contains("--list") {
    for app in NSWorkspace.shared.runningApplications where app.activationPolicy == .regular {
        let ax = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(ax, 2.0)
        let wins = axElements(ax, kAXWindowsAttribute as String)
        let titles = wins.map { "\(role($0)):\(axString($0, kAXTitleAttribute as String) ?? "")" }
        emit(["event": "app", "name": app.localizedName ?? "?", "bundle": app.bundleIdentifier ?? "",
              "pid": app.processIdentifier, "windows": titles])
    }
    emit(["event": "locked", "value": screenLocked()])
    exit(0)
}

if let i = args.firstIndex(of: "--dump"), i + 1 < args.count {
    let want = args[i + 1].lowercased()
    let maxDepth = args.count > i + 2 ? (Int(args[i + 2]) ?? 14) : 14
    if screenLocked() { note("WARNING: the screen is locked; macOS returns a stub tree for every window") }
    for app in NSWorkspace.shared.runningApplications {
        let name = (app.localizedName ?? "").lowercased(), bid = (app.bundleIdentifier ?? "").lowercased()
        if !name.contains(want) && !bid.contains(want) { continue }
        let ax = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(ax, 5.0)
        wakeAccessibility(ax)
        let wins = axElements(ax, kAXWindowsAttribute as String)
        print("=== \(app.localizedName ?? "?") [\(app.bundleIdentifier ?? "")] pid=\(app.processIdentifier) windows=\(wins.count)")
        for (wi, w) in wins.enumerated() {
            print("--- window \(wi) role=\(role(w)) title=«\(axString(w, kAXTitleAttribute as String) ?? "")»")
            var n = 0
            func walk(_ el: AXUIElement, _ d: Int) {
                if d > maxDepth || n > 3000 { return }
                n += 1
                var line = String(repeating: "  ", count: d) + role(el)
                if let s = axString(el, kAXSubroleAttribute as String) { line += "/" + s }
                if let s = axString(el, "AXIdentifier") { line += " id=«\(s)»" }
                for (k, label) in [(kAXTitleAttribute as String, "title"),
                                   (kAXDescriptionAttribute as String, "desc")] {
                    if let s = axString(el, k) { line += " \(label)=«\(s.prefix(70))»" }
                }
                if let t = copyAttr(el, kAXValueAttribute as String), CFGetTypeID(t) == CFStringGetTypeID() {
                    line += " value=«\(((t as! NSString) as String).prefix(110))»"
                }
                print(line)
                for c in children(el) { walk(c, d + 1) }
            }
            walk(w, 1)
            if let (_, p) = findContainer(w) { print("    >>> container path: \(p)") }
        }
    }
    exit(0)
}

// ----------------------------------------------------------------------------- watch

let interval = Double(flag("--interval", "0.5")) ?? 0.5
let want = flag("--app", "auto")
var lastKey = ""
var lastTargetKey = ""
var wasLocked = false

note("axchat watching app=\(want) every \(interval)s")
signal(SIGPIPE, SIG_IGN)            // a closed stdout is handled in emit(), not by dying mid-write
while true {
    if orphaned() { exit(0) }
    let locked = screenLocked()
    if locked != wasLocked {
        emit(["event": "locked", "value": locked, "t": Date().timeIntervalSince1970])
        wasLocked = locked
    }
    if locked { Thread.sleep(forTimeInterval: max(interval, 2.0)); continue }

    let t0 = Date()
    let targets = findTargets(want)
    guard let target = targets.first else {
        if lastTargetKey != "" {
            emit(["event": "no_target", "t": Date().timeIntervalSince1970])
            lastTargetKey = ""; lastKey = ""
        }
        Thread.sleep(forTimeInterval: max(interval, 1.0))
        continue
    }
    var budget = 1500
    let found = leaves(target.container, &budget)
    let ms = Date().timeIntervalSince(t0) * 1000

    let tkey = "\(target.app)|\(target.pid)|\(target.window)|\(target.path)"
    if tkey != lastTargetKey {
        emit(["event": "target", "app": target.app, "pid": target.pid, "window": target.window,
              "path": target.path, "leaves": found.count, "t": Date().timeIntervalSince1970])
        lastTargetKey = tkey
        lastKey = ""
    }
    let key = found.map { "\($0.0)\u{1}\($0.1)" }.joined(separator: "\u{2}")
    if key != lastKey {
        lastKey = key
        let items: [[String: Any]] = found.map { ["r": $0.0, "v": $0.1,
                                                  "x": Double($0.2.x), "y": Double($0.2.y)] }
        emit(["event": "leaves", "app": target.app, "pid": target.pid, "window": target.window,
              "t": Date().timeIntervalSince1970, "ms": round(ms * 10) / 10, "items": items])
    }
    Thread.sleep(forTimeInterval: interval)
}
