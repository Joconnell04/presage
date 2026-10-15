import { getPreferenceValues, open, showHUD, showToast, Toast } from "@raycast/api";
import { execFile, spawn } from "node:child_process";
import { homedir } from "node:os";

type State = "running" | "starting" | "stopped" | "error" | "";
type Prefs = { port?: string; repo?: string };

function prefs(): { port: string; repo: string } {
  const p = getPreferenceValues<Prefs>();
  return { port: (p.port || "8766").trim(), repo: (p.repo || "~/presage").trim().replace(/^~(?=\/|$)/, homedir()) };
}
function base(): string {
  return `http://localhost:${prefs().port}`;
}

async function state(): Promise<State> {
  try {
    const r = await fetch(`${base()}/api/state`, { signal: AbortSignal.timeout(2000) });
    const j = (await r.json()) as { audio?: { state?: State } };
    return (j.audio && j.audio.state) || "";
  } catch {
    return "";
  }
}

async function post(path: string): Promise<boolean> {
  try {
    const r = await fetch(`${base()}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
      signal: AbortSignal.timeout(8000),
    });
    return r.ok;
  } catch {
    return false;
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitUp(seconds: number): Promise<boolean> {
  const until = Date.now() + seconds * 1000;
  while (Date.now() < until) {
    if (await state()) return true;
    await sleep(2000);
  }
  return false;
}

/** Start the server: the app bundle first (its own microphone and system-audio grants), then
 *  run.sh straight from here when the bundle cannot read the checkout (a Downloads-folder
 *  permission it has not been given yet). */
function serverProcessAlive(): Promise<boolean> {
  return new Promise((resolve) => {
    execFile("/usr/bin/pgrep", ["-f", "python[0-9.]* -m presage.server"], (err, out) => resolve(!err && out.trim().length > 0));
  });
}

async function launch(): Promise<boolean> {
  const { repo, port } = prefs();
  const toast = await showToast({ style: Toast.Style.Animated, title: "Starting presage…" });
  spawn("open", ["-a", `${process.env.HOME}/Applications/Presage.app`], { detached: true, stdio: "ignore" }).unref();
  // the bundle starts the server under its own permissions; a process shows within seconds, the
  // page takes 40 to 90 s (models load), so wait for the process first and never start a second one
  let started = false;
  for (let i = 0; i < 8 && !started; i++) {
    await sleep(2000);
    started = await serverProcessAlive();
  }
  if (started) { const ok = await waitUp(150); toast.hide(); return ok; }
  toast.message = "the app bundle did not start it; running it directly";
  const sh = [
    `export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"`,
    `export PRESAGE_PORT=${port}`,
    `cd '${repo}' || exit 1`,
    `nohup ./run.sh >> "$HOME/Library/Logs/presage.log" 2>&1 &`,
  ].join("; ");
  spawn("/bin/sh", ["-c", sh], { detached: true, stdio: "ignore" }).unref();
  const ok = await waitUp(120);
  toast.hide();
  return ok;
}

export async function ensureUp(): Promise<boolean> {
  if (await state()) return true;
  return launch();
}

export async function launchCommand(): Promise<void> {
  const was = await state();
  const ok = was ? true : await launch();
  if (!ok) return showHUD("presage did not start; see ~/Library/Logs/presage.log");
  await open(`${base()}/`);
  await showHUD(was ? "presage is up" : "presage started");
}

export async function pause(): Promise<void> {
  await showHUD((await post("/api/audio/stop")) ? "presage: paused (not listening)" : "presage is not running");
}

export async function listen(): Promise<void> {
  if (!(await ensureUp())) return showHUD("presage did not start; see ~/Library/Logs/presage.log");
  await showHUD((await post("/api/audio/start")) ? "presage: listening" : "presage: could not start listening");
}

export async function toggle(): Promise<void> {
  const s = await state();
  if (!s) return listen();
  if (s === "running" || s === "starting") return pause();
  return listen();
}

export async function quit(): Promise<void> {
  if (!(await state())) return showHUD("presage is not running");
  await showHUD((await post("/api/quit")) ? "presage: quit" : "presage: could not quit");
}
