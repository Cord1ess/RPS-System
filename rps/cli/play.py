"""
`rps play` : play the game from the terminal, with a text HUD.

By default this shows a compact, line-based HUD in the terminal while driving the
same runner that the OpenCV window uses. It tells you whether the hand is inside
the play zone, what the camera sees, which recognizer is running, the last throw,
the current game mode and any warnings. With --no-hud it prints one line per
decision (the same style as the old `play.py`'s console output).

Examples:
    python -m rps play --mock-camera --mock-esp --mode continuous
    python -m rps play --mode guided --headless 300
    python -m rps play --no-hud --mock-camera --mock-esp --headless 120
"""

import sys
import time
from typing import Optional

from rps.cli.common import Ctx, die
from rps.cli.theme import theme
from rps.decision import GESTURE_NAME
from rps.pipeline import RECOGNIZERS
from rps.runner import Runner, RunnerError, decision_line, runner_from_args, summary_lines


def register(sub):
    # Global flags are added by the dispatcher; do not re-add them here
    sub.add_argument("--recognizer", choices=list(RECOGNIZERS), default=None)
    sub.add_argument("--mode", choices=["countdown", "guided", "continuous"], default=None)
    sub.add_argument("--robot-plays", choices=["win", "draw", "lose"], default=None)
    sub.add_argument("--video", default=None, metavar="DIR")
    sub.add_argument("--mock-camera", action="store_true")
    sub.add_argument("--mock-esp", action="store_true")
    sub.add_argument("--no-robot", action="store_true")
    sub.add_argument("--headless", type=int, default=0)
    sub.add_argument("--log", default=None, metavar="PATH")
    sub.add_argument("--no-hud", action="store_true", help="print decisions and notes, no live HUD")
    sub.add_argument("--hud-rate", type=float, default=0.25, help="how often the HUD updates (seconds)")


def run(args, ctx: Ctx) -> int:
    from rps.config import load_config

    cfg = load_config(str(ctx.config), args.set)
    if args.mode:
        cfg.decision.mode = args.mode
    if args.robot_plays:
        cfg.decision.robot_plays = args.robot_plays
    try:
        runner = runner_from_args(cfg, args)
    except RunnerError as e:
        die(str(e))
    for msg in runner.load_messages:
        if not args.no_hud:
            theme.out(theme.hint(f"[play] {msg}"))
        else:
            print(f"[play] {msg}")

    hud_enabled = not args.no_hud and args.headless <= 0 and sys.stdout.isatty()
    if hud_enabled:
        theme.title(f"RPS  {theme.dim(runner.banner)}")
        theme.out(f"{theme.key('recognizer')} {runner.label}")
        theme.out(f"{theme.key('mode')} {cfg.decision.mode}  {theme.key('robot plays to')} {cfg.decision.robot_plays}")
        theme.out(theme.hint("q/ctrl+c to quit  |  r reset counters  |  m switch mode"))
        theme.out()

    last_hud = 0.0
    last_decision = None
    frames = 0

    def on_decision(d):
        nonlocal last_decision
        last_decision = d
        if not hud_enabled:
            print(f"[play] {decision_line(d, runner.cnn)}")

    def on_note(text: str):
        if not hud_enabled:
            print(f"[play] {text}")

    def on_frame(frame, result):
        nonlocal frames, last_hud
        frames += 1
        now = time.time()
        if hud_enabled and now - last_hud >= max(0.05, args.hud_rate):
            last_hud = now
            _draw_hud(runner, result, last_decision, cfg)
        if args.headless and frames >= args.headless:
            return False
        return True

    stats = runner.run(limit=args.headless, on_frame=on_frame if (hud_enabled or args.headless > 0) else None,
                       on_decision=on_decision, on_note=on_note)
    for line in summary_lines(stats):
        if hud_enabled:
            theme.out(line)
        else:
            print(line)
    return 0


def _draw_hud(runner, result, last_decision, cfg) -> None:
    """A compact text HUD that shows whether the hand is inside the play zone,
    the raw recognizer readings, the last throw and the current game state."""
    hand = result.hand
    pose = hand.pose if hand is not None else ""
    inside = hand.present and not hand.skipped if hand is not None else False
    zone_state = theme.ok("hand inside zone") if inside else (theme.warn("hand near/partial") if pose else theme.bad("no hand in zone"))

    engine = runner.pipeline.engine
    snap = engine.snapshot()
    lines = []
    lines.append(f"{zone_state}  {theme.dim('fps')} {runner.log.fps():.1f}")
    lines.append(f"{theme.key('last gesture')} {GESTURE_NAME.get(result.snapshot.last_gesture, '-')}"
                 f"  {theme.key('robot')} {result.snapshot.pose or '-'}")
    if last_decision is not None:
        total = last_decision.software_ms
        if total is not None:
            lines.append(theme.hint(f"last throw -> command {total:.0f} ms ({last_decision.frames} frames)"))
    raw_cnn = raw_mp = "-"
    if result.cnn is not None:
        raw_cnn = f"{GESTURE_NAME[result.cnn[0]]} {result.cnn[1]:.2f}"
    if result.hand is not None and not result.hand.skipped:
        raw_mp = f"{GESTURE_NAME[result.hand.gesture]} {result.hand.confidence:.2f}" if result.hand.present else "no hand"
    lines.append(f"{theme.key('dextra raw')} {raw_cnn}  {theme.key('mediapipe')} {raw_mp}")
    lines.append(f"{theme.key('mode')} {engine.mode}  {theme.key('commits')} {snap.commits}  {theme.key('switches')} {snap.switches}")
    theme.out("  ".join(lines))