"""
Live runtime: webcam -> pseudo-DVS -> RoshamboNet -> vote -> decision -> ESP32, with the
MediaPipe hand path as still-hand authority and fallback.

Examples:
    python play.py                                   # fused, countdown mode, ESP at config address
    python play.py --source mediapipe --mock-esp     # no CNN needed; logs poses from a local mock ESP
    python play.py --mode continuous --set robot.host=192.168.4.1
    python play.py --video data/recordings/alice/<session>   # replay a recording in real time
    python play.py --mock-camera --headless 300 --mock-esp   # smoke test without camera or window

Keys: q/ESC quit | m toggle continuous/countdown | r reset counters
"""

import argparse
import sys
import time

import cv2

from rps.camera import open_source
from rps.config import load_config
from rps.hud import draw_banner, draw_card, draw_dvs_preview, draw_hand, draw_roi
from rps.decision import GESTURE_NAME
from rps.perf import boost_process
from rps.pipeline import Pipeline, load_models
from rps.robot_link import MockEsp, RobotLink
from rps.timing import LatencyLog

WINDOW = "RPS v3 - pseudo-DVS + MediaPipe"


def build_models(cfg, source_mode):
    """Loads the CNN and/or MediaPipe according to --source, degrading gracefully."""
    try:
        cnn, hand, messages = load_models(cfg, source_mode)
    except RuntimeError as e:
        sys.exit(f"[play] {e}")
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
    parser.add_argument("--source", choices=["fused", "cnn", "mediapipe"], default=None)
    parser.add_argument("--mode", choices=["continuous", "countdown"], default=None)
    parser.add_argument("--video", default=None, help="Replay a recording directory instead of the camera")
    parser.add_argument("--mock-camera", action="store_true")
    parser.add_argument("--mock-esp", action="store_true", help="Run a mock ESP on 127.0.0.1 and send to it")
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--headless", type=int, default=0, help="Process N frames without a window")
    parser.add_argument("--log", default=None, help="Write a per-frame CSV log here")
    args = parser.parse_args()

    boost_process()   # keep off EcoQoS efficiency cores: ~2x faster, steadier MediaPipe
    cfg = load_config(args.config, args.set)
    if args.mode:
        cfg.decision.mode = args.mode
    source_mode = args.source or cfg.decision.source
    cnn, hand = build_models(cfg, source_mode)
    effective = "fused" if cnn and hand else ("cnn" if cnn else "mediapipe")

    mock_esp = link = None
    if args.mock_esp:
        mock_esp = MockEsp(port=cfg.robot.port).start()
        cfg.robot.host = "127.0.0.1"
    if cfg.robot.enabled and not args.no_robot:
        link = RobotLink.from_config(cfg.robot).start()

    source = open_source(cfg.camera, cfg.roi, video=args.video, mock=args.mock_camera, realtime=True)
    pipeline = Pipeline(cfg, cnn, hand, pose_sink=link.send_pose if link else None)
    log = LatencyLog(args.log)
    show = args.headless <= 0
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f"[play] source={effective} mode={cfg.decision.mode}. Keys: q quit, m mode, r reset.")

    frames = 0
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
            if args.headless and frames >= args.headless:
                break
            if not show:
                continue

            display = frame.bgr.copy()
            draw_roi(display, source.roi)
            draw_hand(display, source.roi, result.hand)
            stats = {"fps": log.fps(), "dvs_ms": log.mean("dvs_ms"), "cnn_ms": log.mean("cnn_ms"),
                     "mp_ms": log.mean("mp_ms"), "grab_to_send_ms": log.mean("grab_to_send_ms")}
            draw_banner(display, "RPS v3  |  pseudo-DVS + RoshamboNet + MediaPipe", stats)
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
