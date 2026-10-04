"""
`rps setup` : camera, light and the play zone, once per location.

check    open the camera for a few seconds and say how it is doing (frame rate, light, whether it is
         overexposed). --watch keeps it open and updating, --window shows the video too. This is the
         one Setup command worth running over SSH on a Pi with no screen.
probe    the Setup page's Auto-configure: measure every camera mode and save the best (about 25 s).
zone     set the play zone: drag it on the video if there is a window, or type the numbers.
latency  measure the camera's own delay (mirror or LED) and save it for the Speed card.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from rps.camera import crop_roi, open_source
from rps.cli.common import Ctx, die, prompt
from rps.cli.theme import theme
from rps.config import save_config

ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS = ROOT / "tools"
FPS_GOOD, FPS_MIN = 27, 15           # what the app considers enough to throw in
LIGHT_LO, LIGHT_HI = 60, 200         # the app's "good light" band
CLIPPED_MAX = 5.0                    # percent of the zone that may be pure white
LATENCY_RESULT = re.compile(r"^\[latency\] \w+: median ([\d.]+) ms")


def register(sub):
    sub.set_defaults(action="check")
    sub.add_argument("--video", default=None, metavar="DIR", help="use a recorded session instead of the camera")
    sub.add_argument("--mock-camera", action="store_true", help="use the simulated camera (no webcam needed)")
    sub.add_argument("--seconds", type=float, default=5.0, help="check: how long to measure (default: 5)")
    sub.add_argument("--watch", action="store_true", help="check: keep going until Ctrl-C")
    sub.add_argument("--window", action="store_true", help="check: show the video as well")

    actions = sub.add_subparsers(dest="action", metavar="ACTION", title="actions")
    actions.add_parser("check", help="measure frame rate, light and overexposure")

    probe = actions.add_parser("probe", help="measure every camera mode and save the best one")
    probe.add_argument("--seconds", type=float, default=2.5, help="seconds per camera mode (default: 2.5)")
    probe.add_argument("--index", type=int, default=None, help="only this camera index")
    probe.add_argument("--no-save", action="store_true", help="report the best mode without saving it")

    zone = actions.add_parser("zone", help="set the play zone (the square the app analyses)")
    zone.add_argument("numbers", nargs="*", type=int, metavar="X Y SIZE",
                      help="the numbers directly, instead of dragging on the video")
    zone.add_argument("--margin", type=float, default=None, help="also set roi.record_margin")

    latency = actions.add_parser("latency", help="measure the camera's own delay")
    latency.add_argument("--mode", choices=["screen", "led"], default="screen",
                         help="screen: flash this terminal and watch it in a mirror (the screen's own delay is "
                              "included); led: the robot flashes inside the zone, needs the reference firmware")
    latency.add_argument("--trials", type=int, default=20, help="flashes to time (default: 20)")
    latency.add_argument("--no-save", action="store_true", help="report it without saving")


def run(args, ctx: Ctx) -> int:
    action = getattr(args, "action", None) or "check"
    return {"check": _check, "probe": _probe, "zone": _zone, "latency": _latency}[action](args, ctx)


def _open(args, ctx: Ctx):
    cfg = ctx.load()
    return open_source(cfg.camera, cfg.roi, video=getattr(args, "video", None),
                       mock=getattr(args, "mock_camera", False), realtime=True)


# ---------------------------------------------------------------------------- check
def _check(args, ctx: Ctx) -> int:
    """How good is the camera right now: frame rate, light, and is the zone on the hand."""
    source = _open(args, ctx).start()
    window = "RPS setup - camera check"
    times: List[float] = []
    brightness: List[float] = []
    clipped: List[float] = []
    last = None
    if args.window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    theme.title("Camera check")
    try:
        while True:
            frame = source.read(timeout=2.0)
            if frame is None:
                theme.out(theme.warn("no frame from the camera (2 s timeout)"))
                break
            times.append(frame.t)
            last = frame
            gray = cv2.cvtColor(crop_roi(frame.bgr, source.roi), cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            clipped.append(float(np.mean(gray >= 250) * 100))
            if args.window:
                shown = frame.bgr.copy()
                x, y, size = source.roi
                cv2.rectangle(shown, (x, y), (x + size, y + size), (0, 220, 0), 2)
                cv2.putText(shown, f"{len(times)} frames   light {brightness[-1]:.0f}/255   "
                                   f"overexposed {clipped[-1]:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.imshow(window, shown)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if time.perf_counter() - times[0] >= args.seconds:      # --watch: never stops on time
                break
        if last is None:
            die("The camera gave no frames")
        fps = (len(times) - 1) / max(1e-6, times[-1] - times[0]) if len(times) > 1 else 0.0
        report = _check_report(fps, float(np.mean(brightness)), float(np.mean(clipped)), source, last)
        ctx.emit({"fps": fps, "brightness": np.mean(brightness), "overexposed_percent": np.mean(clipped),
                  "frames": len(times), "ok": report["ok"], "roi": list(source.roi)},
                 text="\n".join(report["lines"]))
        return 0 if report["ok"] else 1
    finally:
        source.stop()
        if args.window:
            cv2.destroyAllWindows()


def _check_report(fps: float, brightness: float, clipped: float, source, frame) -> dict:
    lines: List[str] = []
    ok = True
    if fps >= FPS_GOOD:
        lines.append(theme.ok(f"{fps:.1f} fps: fast enough to throw at"))
    elif fps >= FPS_MIN:
        lines.append(theme.warn(f"{fps:.1f} fps: playable, but every throw costs more camera time. 27+ is better"))
        ok = False
    else:
        lines.append(theme.bad(f"{fps:.1f} fps: too slow to play. Try another USB port, or rps setup probe"))
        ok = False
    if LIGHT_LO <= brightness <= LIGHT_HI:
        lines.append(theme.ok(f"light {brightness:.0f}/255: good"))
    elif brightness < LIGHT_LO:
        lines.append(theme.warn(f"light {brightness:.0f}/255: too dark, the hand tracker will lose fingers"))
        ok = False
    else:
        lines.append(theme.warn(f"light {brightness:.0f}/255: very bright, washed out detail"))
        ok = False
    if clipped <= CLIPPED_MAX:
        lines.append(theme.ok(f"overexposed {clipped:.1f}% of the zone"))
    else:
        lines.append(theme.bad(f"overexposed {clipped:.1f}%: pure white, and white never moves, so nothing is "
                               f"seen there"))
        ok = False
    x, y, size = source.roi
    h, w = frame.bgr.shape[:2]
    if size <= 0 or x + size > w or y + size > h:
        lines.append(theme.bad(f"the play zone ({x}, {y}, {size}) does not fit in this {w}x{h} image"))
        ok = False
    else:
        lines.append(theme.note(f"play zone {x}, {y}, {size} inside a {w}x{h} image "
                                f"({theme.plain(theme.style('rps setup zone', 'pink'))} to move it)"))
    return {"ok": ok, "lines": lines}


# ---------------------------------------------------------------------------- probe
def _probe(args, ctx: Ctx) -> int:
    """The Setup page's Auto-configure: try every mode in this light, keep the best."""
    cmd = [sys.executable, str(TOOLS / "camera_probe.py"), "--config", str(ctx.config),
           "--seconds", str(args.seconds)]
    if args.index is not None:
        cmd += ["--index", str(args.index)]
    if not args.no_save:
        cmd.append("--write-config")
    theme.title("Auto-configure")
    theme.out(theme.hint(f"trying every camera mode in this light, about {int(25 * args.seconds / 2.5)} s"))
    code = _stream(cmd)
    if code == 0 and not args.no_save:
        theme.out(theme.ok(f"camera settings saved to {ctx.config}"))
    return code


# ---------------------------------------------------------------------------- zone
def _zone(args, ctx: Ctx) -> int:
    """The play zone: the only square the app analyses, so frame the hand and not your face."""
    cfg = ctx.load()
    numbers = args.numbers
    if numbers and len(numbers) != 3:
        die("Give the play zone as three numbers: rps setup zone X Y SIZE")
    if not numbers:
        numbers = _drag_zone(args, ctx) if _can_drag(ctx) else None
        if numbers is None and not ctx.interactive:
            die("No screen to drag on and no numbers given: use `rps setup zone X Y SIZE`")
    if numbers is None:
        theme.out(theme.hint(f"the camera is {cfg.camera.width}x{cfg.camera.height}; frame the hand only"))
        numbers = [int(prompt("x", str(cfg.roi.x))), int(prompt("y", str(cfg.roi.y))),
                   int(prompt("size", str(cfg.roi.size)))]
    x, y, size = numbers
    if size <= 0:
        die("The play zone needs a size above zero")
    cfg.roi.x, cfg.roi.y, cfg.roi.size = int(x), int(y), int(size)
    if args.margin is not None:
        cfg.roi.record_margin = float(args.margin)
    save_config(cfg, str(ctx.config), quiet=ctx.json)
    ctx.emit({"roi": {"x": x, "y": y, "size": size}},
             text=theme.ok(f"play zone saved: x={x} y={y} size={size}"))
    return 0


def _can_drag(ctx: Ctx) -> bool:
    """Dragging needs something to drag on and someone to drag: a display and a real terminal."""
    if not ctx.interactive:
        return False
    return sys.platform == "win32" or bool(os.environ.get("DISPLAY"))


def _drag_zone(args, ctx: Ctx) -> Optional[List[int]]:
    """Drag a square on the live video, the same way the Setup page's button works. None if this
    machine cannot open a window, so the caller can ask for the numbers instead."""
    window = "RPS setup - drag the play zone"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error:
        theme.out(theme.warn("this machine cannot open a video window"))
        return None
    source = _open(args, ctx).start()
    picked = None
    try:
        theme.out(theme.hint("drag a square over the hand, then press enter"))
        while picked is None:
            frame = source.read(timeout=2.0)
            if frame is None:
                die("The camera gave no frames, so the zone cannot be dragged. Type the numbers instead.")
            picked = tuple(cv2.selectROI(window, frame.bgr, showCrosshair=True, fromCenter=False))
            if picked == (0, 0, 0, 0):
                picked = None                          # cancelled: offer it again
    finally:
        source.stop()
        cv2.destroyAllWindows()
    x, y, w, h = picked
    return [int(x), int(y), int(min(w, h))]             # the zone is a square


# ---------------------------------------------------------------------------- latency
def _latency(args, ctx: Ctx) -> int:
    """How many milliseconds the camera image lags behind reality, by the screen flash or the LED."""
    cfg = ctx.load()
    if args.mode == "led" and cfg.robot.protocol != "ack":
        die("The LED test needs the reference firmware (robot.protocol=ack): the team firmware has no LED "
            "command. Use --mode screen with a mirror instead.")
    cmd = [sys.executable, str(TOOLS / "latency_test.py"), "--mode", args.mode, "--trials", str(args.trials),
           "--config", str(ctx.config)]
    theme.title(f"Camera delay ({args.mode})")
    theme.out(theme.hint("point a mirror at the camera, so it can see this screen"
                         if args.mode == "screen" else "the robot's LED has to flash inside the play zone"))
    measured: List[float] = []

    def watch(line: str) -> None:
        print(line)
        found = LATENCY_RESULT.search(line)
        if found:
            measured.append(float(found.group(1)))

    code = _stream(cmd, watch)
    if code != 0 or not measured:
        return code or 1
    ms = measured[-1]
    if args.no_save:
        theme.out(f"camera delay {ms:.1f} ms (not saved)")
        return 0
    cfg.latency.camera_latency_ms = round(ms, 1)
    save_config(cfg, str(ctx.config), quiet=ctx.json)
    theme.out(theme.ok(f"camera delay {ms:.1f} ms saved as latency.camera_latency_ms"))
    return 0


# ---------------------------------------------------------------------------- running the app's own tools
def _stream(cmd: List[str], watch=None) -> int:
    """Run one of the tools the app already has, showing its output as it arrives."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in process.stdout:
            line = line.rstrip()
            if watch is None:
                print("  " + theme.hint(line) if line.startswith(" ") else line)
            else:
                watch(line)
    finally:
        process.wait()
    return process.returncode