"""
Live runtime in an OpenCV window: webcam -> Dextra (motion image) and/or Mediapipe -> decision
-> ESP32. The recognizers are named as in the app: dextra_raw, dextra_tuned, mediapipe, both.
The loop itself is rps.runner, which `python -m rps play` also drives, with a text HUD instead of
this window.

Examples:
    python play.py                                   # config.json settings (recognizer, mode, robot)
    python play.py --recognizer mediapipe --mock-esp # simulated robot on this computer
    python play.py --mode continuous --set robot.host=192.168.4.1
    python play.py --video data/recordings/alice/<session>   # replay a recording in real time
    python play.py --mock-camera --headless 300 --mock-esp   # smoke test without camera or window

Keys: q/ESC quit | m toggle continuous/countdown | r reset counters
"""

import argparse
import sys

import cv2

from rps.config import load_config
from rps.decision import GESTURE_NAME
from rps.hud import draw_banner, draw_card, draw_dvs_preview, draw_hand, draw_roi
from rps.pipeline import RECOGNIZERS
from rps.runner import Runner, RunnerError, decision_line, runner_from_args, summary_lines

WINDOW = "RPS - Dextra + Mediapipe"


def raw_readings(result):
    raw_cnn = raw_mp = None
    if result.cnn is not None:
        raw_cnn = f"{GESTURE_NAME[result.cnn[0]]} {result.cnn[1]:.2f}"
    if result.hand is not None and not result.hand.skipped:
        raw_mp = f"{GESTURE_NAME[result.hand.gesture]} {result.hand.confidence:.2f}" if result.hand.present else "no hand"
    return raw_cnn, raw_mp


def draw_and_read_keys(runner: Runner, window: str, effective: str):
    """The per-frame half of a windowed run: draw everything, then q quits, m switches mode, r resets."""
    def on_frame(frame, result):
        display = frame.bgr.copy()
        draw_roi(display, runner.source.roi)
        draw_hand(display, runner.source.roi, result.hand)
        log = runner.log
        stats = {"fps": log.fps(), "dvs_ms": log.mean("dvs_ms"), "cnn_ms": log.mean("cnn_ms"),
                 "mp_ms": log.mean("mp_ms"), "grab_to_send_ms": log.mean("grab_to_send_ms")}
        draw_banner(display, runner.banner, stats)
        draw_dvs_preview(display, runner.pipeline.last_dvs_frame)
        raw_cnn, raw_mp = raw_readings(result)
        draw_card(display, result.snapshot, raw_cnn, raw_mp,
                  runner.link.stats() if runner.link else None, effective)
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("m"):
            mode = "continuous" if runner.pipeline.engine.mode == "countdown" else "countdown"
            runner.mode_to(mode, frame.t)
            print(f"[play] Mode -> {mode}")
        elif key == ord("r"):
            runner.reset_counters()
        return True

    return on_frame


def main():
    parser = argparse.ArgumentParser(description="RPS v3 live runtime")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--recognizer", choices=list(RECOGNIZERS), default=None)
    parser.add_argument("--mode", choices=["countdown", "guided", "continuous"], default=None,
                        help="guided: rounds follow the beat guide's timing (the beat is only heard in the app)")
    parser.add_argument("--robot-plays", choices=["win", "draw", "lose"], default=None,
                        help="win: beats your throw; draw: copies it; lose: plays what it beats")
    parser.add_argument("--video", default=None, help="Replay a recording directory instead of the camera")
    parser.add_argument("--mock-camera", action="store_true")
    parser.add_argument("--mock-esp", action="store_true", help="Simulated robot on this computer")
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--headless", type=int, default=0, help="Process N frames without a window")
    parser.add_argument("--log", default=None, help="Write a per-frame CSV log here")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.mode:
        cfg.decision.mode = args.mode
    if args.robot_plays:
        cfg.decision.robot_plays = args.robot_plays
    try:
        runner = runner_from_args(cfg, args)
    except RunnerError as e:
        sys.exit(f"[play] {e}")
    for msg in runner.load_messages:
        print(f"[play] {msg}")

    show = args.headless <= 0
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f"[play] {runner.label}, mode {cfg.decision.mode}, robot plays to "
          f"{cfg.decision.robot_plays}. Keys: q quit, m mode, r reset.")

    stats = runner.run(limit=args.headless, on_frame=draw_and_read_keys(runner, WINDOW, runner.effective) if show else None,
                       on_decision=lambda d: print(f"[play] {decision_line(d, runner.cnn)}"),
                       on_note=lambda text: print(f"[play] {text}"))
    if show:
        cv2.destroyAllWindows()
    for line in summary_lines(stats):
        print(line)


if __name__ == "__main__":
    main()