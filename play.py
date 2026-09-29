"""
Live runtime without the app window: webcam -> Dextra (motion image) and/or Mediapipe -> decision
-> ESP32. The recognizers are named as in the app: dextra_raw, dextra_tuned, mediapipe, both.

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
import time

import cv2
import numpy as np

from rps.camera import open_source
from rps.config import load_config
from rps.hud import draw_banner, draw_card, draw_dvs_preview, draw_hand, draw_roi
from rps.decision import GESTURE_NAME
from rps.perf import boost_process
from rps.game import ENDLESS_ROUNDS, BeatSchedule
from rps.pipeline import RECOGNIZERS, Pipeline, load_models, reader_name
from rps.robot_link import MockEsp, RobotLink
from rps.timing import LatencyLog

WINDOW = "RPS - Dextra + Mediapipe"


def build_models(cfg, recognizer):
    """Loads Dextra and/or Mediapipe for the recognizer, degrading gracefully for "both"."""
    try:
        cnn, hand, messages = load_models(cfg, recognizer)
    except RuntimeError as e:
        hint = "" if recognizer == "mediapipe" else " Or play with Mediapipe alone: --recognizer mediapipe"
        sys.exit(f"[play] {e}{hint}")
    for msg in messages:
        print(f"[play] {msg}")
    return cnn, hand


def raw_readings(result):
    raw_cnn = raw_mp = None
    if result.cnn is not None:
        raw_cnn = f"{GESTURE_NAME[result.cnn[0]]} {result.cnn[1]:.2f}"
    if result.hand is not None and not result.hand.skipped:
        raw_mp = f"{GESTURE_NAME[result.hand.gesture]} {result.hand.confidence:.2f}" if result.hand.present else "no hand"
    return raw_cnn, raw_mp


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

    boost_process()   # keep off EcoQoS efficiency cores: ~2x faster, steadier MediaPipe
    cfg = load_config(args.config, args.set)
    if args.mode:
        cfg.decision.mode = args.mode
    if args.robot_plays:
        cfg.decision.robot_plays = args.robot_plays
    recognizer = args.recognizer or cfg.decision.recognizer
    cnn, hand = build_models(cfg, recognizer)
    effective = " + ".join(n for n, on in ((reader_name("cnn", cnn), cnn), ("Mediapipe", hand)) if on)

    robot = "off" if args.no_robot else ("simulated" if args.mock_esp else cfg.robot.mode)
    mock_esp = link = None
    if robot == "simulated":
        mock_esp = MockEsp(port=cfg.robot.port).start()
    if robot != "off":
        link = RobotLink.from_config(cfg.robot, host="127.0.0.1" if mock_esp else None).start()

    source = open_source(cfg.camera, cfg.roi, video=args.video, mock=args.mock_camera, realtime=True)
    pipeline = Pipeline(cfg, cnn, hand, pose_sink=link.send_pose if link else None)
    if cfg.decision.mode == "guided":
        g = cfg.game
        schedule = BeatSchedule(time.perf_counter(), 60.0 / g.beat_bpm, cfg.decision.pumps_before_shoot,
                                g.rounds or ENDLESS_ROUNDS, g.lead_beats, g.gap_beats)
        pipeline.engine.start_guided(schedule, cfg.latency.camera_latency_ms / 1000.0)
    log = LatencyLog(args.log)
    show = args.headless <= 0
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f"[play] {RECOGNIZERS[recognizer]} (running: {effective}), mode {cfg.decision.mode}, robot plays to "
          f"{cfg.decision.robot_plays}. Keys: q quit, m mode, r reset.")

    frames = 0
    laptop_ms = []                          # per decision: first frame showing the throw -> command sent
    try:
        while True:
            frame = source.read(timeout=2.0)
            if frame is None:
                if args.video:
                    break
                print("[play] No frame from camera (timeout): robot to ready until frames return.")
                pipeline.set_mode(pipeline.engine.mode, time.perf_counter())   # restart the game, send READY
                continue
            result = pipeline.step(frame, source.roi)
            log.add(pipeline.log_record(result))
            frames += 1
            d = result.decision
            if d is not None:
                total = d.software_ms
                if total is not None:
                    laptop_ms.append(total)
                print(f"[play] {GESTURE_NAME[d.gesture]} read by {reader_name(d.source, cnn)} -> robot {d.pose}: "
                      f"reading {d.read_ms:.0f} ms ({d.frames} frames) + processing and send "
                      + ("-" if d.process_ms is None else f"{d.process_ms:.0f} ms")
                      + ("" if total is None else f" = {total:.0f} ms on the laptop"))
            if args.headless and frames >= args.headless:
                break
            if not show:
                continue

            display = frame.bgr.copy()
            draw_roi(display, source.roi)
            draw_hand(display, source.roi, result.hand)
            stats = {"fps": log.fps(), "dvs_ms": log.mean("dvs_ms"), "cnn_ms": log.mean("cnn_ms"),
                     "mp_ms": log.mean("mp_ms"), "grab_to_send_ms": log.mean("grab_to_send_ms")}
            draw_banner(display, f"RPS  |  {RECOGNIZERS[recognizer]}", stats)
            draw_dvs_preview(display, pipeline.last_dvs_frame)
            raw_cnn, raw_mp = raw_readings(result)
            draw_card(display, result.snapshot, raw_cnn, raw_mp, link.stats() if link else None, effective)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = "continuous" if pipeline.engine.mode == "countdown" else "countdown"
                pipeline.set_mode(mode, frame.t)
                print(f"[play] Mode -> {mode}")
            elif key == ord("r"):
                pipeline.engine.switches = pipeline.engine.commits = 0
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        if link:
            link.stop()
        if mock_esp:
            mock_esp.stop()
        if hand:
            hand.close()
        log.close()
        if show:
            cv2.destroyAllWindows()
        s = log.summary()
        print("\n================ SESSION SUMMARY ================")
        print(f"  frames: {frames}   camera fps: {s.get('fps', 0):.1f}")
        if laptop_ms:
            print(f"  throw read and sent (laptop): median {float(np.median(laptop_ms)):.0f} ms, "
                  f"fastest {min(laptop_ms):.0f} ms, slowest {max(laptop_ms):.0f} ms over {len(laptop_ms)} throws")
        for key in ("dvs_ms", "cnn_ms", "mp_ms", "total_ms", "grab_to_send_ms"):
            if f"{key}_mean" in s:
                print(f"  {key:16s} mean {s[f'{key}_mean']:6.2f}   p95 {s[f'{key}_p95']:6.2f}")
        snap = pipeline.engine.snapshot()
        print(f"  commits: {snap.commits}   switches: {snap.switches}   hand-path skips: "
              f"{hand.skipped_frames if hand else 0}")
        if mock_esp:
            print(f"  mock ESP pose changes: {[p for _, p in mock_esp.pose_log]}")
        print("=================================================")


if __name__ == "__main__":
    main()
