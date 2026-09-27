"""
Records raw training/evaluation sessions (record once, generate DVS frames many times).
The desktop app (app.py -> Record tab) does the same with a form; both use rps/recorder.py.

Saves the play-zone ROI plus a margin as lossless FFV1 video, per-frame capture timestamps,
and metadata, so emulator parameters (C, N, K, ROI) can be re-tuned later without re-recording.

Session types (ROSHAMBO17-style session-level labels):
    show        60-90 s holding ONE gesture the whole time while moving/rotating the hand and
                changing distance. Do not pump: every frame gets the session label.
    throws      countdown throws of ONE gesture ("rock, paper, scissors, shoot" x N).
                Used for training (transition frames) and for evaluation.
    background  no hand in the zone: move body/arm around it, change lights.

Examples:
    python record_session.py --person alice --type show --label rock --duration 75
    python record_session.py --person alice --type throws --label scissors --throws 10 --duration 45
    python record_session.py --person alice --type background --duration 60

Keys during recording: q = stop and keep, x = abort and delete.
"""

import argparse
import time

import cv2

from rps.camera import CameraSource, MockSource, crop_roi
from rps.config import load_config
from rps.dvs_emulator import PseudoDVS
from rps.hud import draw_dvs_preview
from rps.perf import boost_process
from rps.recorder import LABELS, SESSION_TYPES, SessionRecorder


def main():
    parser = argparse.ArgumentParser(description="Record a raw RPS session")
    parser.add_argument("--person", required=True)
    parser.add_argument("--type", required=True, choices=SESSION_TYPES)
    parser.add_argument("--label", choices=LABELS, default=None)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--throws", type=int, default=0, help="Planned throws (throws sessions)")
    parser.add_argument("--hand", default="right", choices=["right", "left", "both"])
    parser.add_argument("--lighting", default="", help="Free text, e.g. 'lamp', 'daylight', 'dim'")
    parser.add_argument("--notes", default="")
    parser.add_argument("--out_root", default="data/recordings")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--mock", action="store_true", help="Record the synthetic mock camera (pipeline tests)")
    parser.add_argument("--no_window", action="store_true")
    args = parser.parse_args()
    if args.type != "background" and args.label is None:
        parser.error("--label is required for show/throws sessions")

    boost_process()
    cfg = load_config(args.config)
    cam = (MockSource(cfg.camera.width, cfg.camera.height, cfg.camera.fps) if args.mock
           else CameraSource(cfg.camera, cfg.roi)).start()
    rec = SessionRecorder(cfg, args.person, args.type, args.label, args.duration, args.throws, args.hand,
                          args.lighting, args.notes, args.out_root, getattr(cam, "info", {"mock": True}))
    dvs = PseudoDVS(cfg.dvs)
    window, show = "Record session", not args.no_window
    if show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"[record] {rec.session_id}: {rec.instruction}")
    aborted, started = False, False
    t_countdown = time.perf_counter()
    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                continue
            remaining = args.countdown - (time.perf_counter() - t_countdown)
            if remaining <= 0:
                if not started:
                    rec.start(frame, cam.roi)
                    started = True
                if not rec.add(frame):
                    break
            dvs_frame, _ = dvs.process(crop_roi(frame.bgr, cam.roi), frame.t)   # the play zone only, as live
            if not show:
                continue
            display = frame.bgr.copy()
            x, y, s = cam.roi
            cv2.rectangle(display, (x, y), (x + s, y + s), (0, 220, 0), 2)
            w = display.shape[1]
            cv2.rectangle(display, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(display, rec.instruction, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if started:
                cv2.circle(display, (18, 38), 7, (0, 0, 255), -1)
                cv2.putText(display, f"REC {rec.elapsed:5.1f}/{args.duration:.0f}s  frames {len(rec.rows)}  "
                            f"dropped {rec.dropped}  missed {rec.missed}", (32, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(display, f"Starting in {remaining:3.1f} s", (10, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            draw_dvs_preview(display, dvs_frame)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("x"):
                aborted = True
                break
    finally:
        cam.stop()
        if show:
            cv2.destroyAllWindows()

    if aborted or not started:
        rec.abort()
        print("[record] Aborted; nothing saved.")
        return
    meta = rec.finish()
    if meta is None:
        print("[record] No frames recorded; nothing saved.")
        return
    print(f"[record] Saved {rec.out_dir}: {meta['frames']} frames, {meta['fps_measured']:.1f} fps, "
          f"{meta['size_mb']} MB, writer dropped {meta['dropped_by_writer']}, reader missed "
          f"{meta['missed_by_reader']}, video check {'OK' if meta['video_check_ok'] else 'MISMATCH'}")


if __name__ == "__main__":
    main()
