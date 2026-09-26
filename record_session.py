"""
Records raw training/evaluation sessions (record once, generate DVS frames many times).

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
import csv
import json
import os
import queue
import shutil
import threading
import time

import cv2
import numpy as np

from rps.camera import CameraSource, MockSource, margin_box
from rps.config import load_config
from rps.dvs_emulator import PseudoDVS
from rps.hud import draw_dvs_preview
from rps.perf import boost_process

INSTRUCTIONS = {
    "show": "Hold {label} the WHOLE time. Move, rotate, change distance. No pumping.",
    "throws": "Countdown throws of {label}: pump x3 then throw, hold ~1 s, relax, repeat.",
    "background": "Keep your hand OUT of the green box. Move body/arm around it.",
}
LABELS = ("rock", "paper", "scissors", "background")


class VideoWriterThread:
    """FFV1 (lossless) writer on its own thread; falls back to a PNG sequence."""

    def __init__(self, out_dir: str, size, fps: float):
        self.path = os.path.join(out_dir, "video.mkv")
        self.writer = cv2.VideoWriter(self.path, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"),
                                      fps, size, True)
        self.png_dir = None
        if not self.writer.isOpened():
            self.writer = None
            self.png_dir = os.path.join(out_dir, "frames")
            os.makedirs(self.png_dir, exist_ok=True)
            print("[record] FFV1 unavailable -> writing a PNG sequence instead.")
        self.size = size
        self.q: "queue.Queue" = queue.Queue(maxsize=90)
        self.written = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._loop, name="writer", daemon=True)
        self._thread.start()

    @property
    def video_file(self) -> str:
        return "video.mkv" if self.writer is not None else "frames"

    def put(self, img: np.ndarray) -> bool:
        assert img.shape[1] == self.size[0] and img.shape[0] == self.size[1], "frame size changed"
        try:
            self.q.put_nowait(img)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _loop(self):
        while True:
            img = self.q.get()
            if img is None:
                break
            if self.writer is not None:
                self.writer.write(img)
            else:
                cv2.imwrite(os.path.join(self.png_dir, f"{self.written:06d}.png"), img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
            self.written += 1

    def close(self):
        self.q.put(None)
        self._thread.join()
        if self.writer is not None:
            self.writer.release()


def count_video_frames(path: str) -> int:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def main():
    parser = argparse.ArgumentParser(description="Record a raw RPS session")
    parser.add_argument("--person", required=True)
    parser.add_argument("--type", required=True, choices=["show", "throws", "background"])
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

    label = "background" if args.type == "background" else args.label
    if label is None:
        parser.error("--label is required for show/throws sessions")

    boost_process()
    cfg = load_config(args.config)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    session_id = f"{stamp}_{args.type}_{label}"
    out_dir = os.path.join(args.out_root, args.person, session_id)
    os.makedirs(out_dir, exist_ok=True)

    cam = (MockSource(cfg.camera.width, cfg.camera.height, cfg.camera.fps) if args.mock
           else CameraSource(cfg.camera, cfg.roi)).start()
    frame = cam.read(timeout=3.0)
    if frame is None:
        cam.stop()
        raise SystemExit("[record] Camera produced no frames.")
    h, w = frame.bgr.shape[:2]
    x0, y0, x1, y1 = margin_box(cam.roi, cfg.roi.record_margin, w, h)
    roi_in_crop = [cam.roi[0] - x0, cam.roi[1] - y0, cam.roi[2]]
    writer = VideoWriterThread(out_dir, (x1 - x0, y1 - y0), float(cfg.camera.fps))
    dvs = PseudoDVS(cfg.dvs)

    window = "Record session"
    show = not args.no_window
    if show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    text = INSTRUCTIONS[args.type].format(label=label.upper())
    rows, t_first, missed, last_cam_id = [], None, 0, None
    aborted = False
    t_start_countdown = time.perf_counter()
    print(f"[record] {session_id}: {text}")
    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                continue
            now = time.perf_counter()
            remaining_cd = args.countdown - (now - t_start_countdown)
            recording = remaining_cd <= 0
            crop = frame.bgr[y0:y1, x0:x1]
            if recording:
                if t_first is None:
                    t_first = frame.t
                if last_cam_id is not None and frame.id > last_cam_id + 1:
                    missed += frame.id - last_cam_id - 1
                last_cam_id = frame.id
                if writer.put(crop.copy()):
                    rows.append((len(rows), round(frame.t - t_first, 6), frame.id))
            dvs_frame, _ = dvs.process(crop[roi_in_crop[1]:roi_in_crop[1] + roi_in_crop[2],
                                            roi_in_crop[0]:roi_in_crop[0] + roi_in_crop[2]], frame.t)
            if not show:
                if recording and frame.t - t_first >= args.duration:
                    break
                continue

            display = frame.bgr.copy()
            x, y, s = cam.roi
            cv2.rectangle(display, (x, y), (x + s, y + s), (0, 220, 0), 2)
            cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), 1)
            cv2.rectangle(display, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(display, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if recording:
                elapsed = frame.t - t_first
                cv2.circle(display, (18, 38), 7, (0, 0, 255), -1)
                cv2.putText(display, f"REC {elapsed:5.1f}/{args.duration:.0f}s  frames {len(rows)}  "
                            f"dropped {writer.dropped}  missed {missed}", (32, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if elapsed >= args.duration:
                    break
            else:
                cv2.putText(display, f"Starting in {remaining_cd:3.1f} s", (10, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            draw_dvs_preview(display, dvs_frame if dvs_frame is not None else None)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("x"):
                aborted = True
                break
    finally:
        cam.stop()
        writer.close()
        if show:
            cv2.destroyAllWindows()

    if aborted or not rows:
        shutil.rmtree(out_dir, ignore_errors=True)
        print("[record] Aborted; nothing saved.")
        return

    with open(os.path.join(out_dir, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "t", "camera_frame_id"])
        wr.writerows(rows)
    n_video = count_video_frames(writer.path) if writer.video_file == "video.mkv" else writer.written
    duration = rows[-1][1]
    meta = {
        "session_id": session_id,
        "person": args.person,
        "type": args.type,
        "label": label,
        "hand": args.hand,
        "lighting": args.lighting,
        "notes": args.notes,
        "throws_planned": args.throws,
        "frames": len(rows),
        "video_frames": n_video,
        "dropped_by_writer": writer.dropped,
        "missed_by_reader": missed,
        "duration_s": duration,
        "fps_measured": (len(rows) - 1) / duration if duration > 0 else 0.0,
        "video_file": writer.video_file,
        "crop_box": [x0, y0, x1, y1],
        "roi_camera": list(cam.roi),
        "roi_in_crop": roi_in_crop,
        "camera": cfg.to_dict()["camera"],
        "camera_info": getattr(cam, "info", {"mock": True}),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    ok = n_video == len(rows)
    print(f"[record] Saved {out_dir}: {len(rows)} frames, {meta['fps_measured']:.1f} fps, "
          f"writer dropped {writer.dropped}, reader missed {missed}, video check {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print(f"[record] WARNING: video has {n_video} frames but the CSV has {len(rows)} rows.")


if __name__ == "__main__":
    main()
