"""
Raw session recording shared by record_session.py (CLI) and the desktop app.

A session stores the play-zone ROI plus a margin as lossless FFV1 video (PNG-sequence fallback),
per-frame capture timestamps (frames.csv) and metadata (meta.json), so pseudo-DVS parameters can
be re-tuned later without re-recording.
"""

import csv
import json
import os
import queue
import shutil
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np

from rps.camera import Frame, Roi, margin_box
from rps.config import Config

LABELS = ("rock", "paper", "scissors", "background")
SESSION_TYPES = ("show", "throws", "background")
INSTRUCTIONS = {
    "show": "Hold {label} the WHOLE time. Move, rotate, change distance. No pumping.",
    "throws": "Countdown throws of {label}: pump x3 then throw, hold ~1 s, relax, repeat.",
    "background": "Keep your hand OUT of the green box. Move body/arm around it.",
}
# Recommended per-person protocol: (type, label) -> sessions
PROTOCOL = [("show", "rock"), ("show", "paper"), ("show", "scissors"),
            ("throws", "rock"), ("throws", "paper"), ("throws", "scissors"), ("background", "background")]


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


def video_frame_size(path: str) -> Optional[tuple]:
    """(width, height) of the first decoded frame, or None."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    ok, img = cap.read()
    cap.release()
    return (img.shape[1], img.shape[0]) if ok else None


def even_span(lo: int, hi: int, keep_lo: int, keep_hi: int, limit: int) -> tuple:
    """
    [lo, hi) adjusted to an even length that still contains [keep_lo, keep_hi) and stays inside
    [0, limit). The video encoder silently drops the last row/column of odd-sized frames, which
    would cut into the play zone.
    """
    if (hi - lo) % 2 == 0:
        return lo, hi
    if hi > keep_hi:
        return lo, hi - 1
    if lo < keep_lo:
        return lo + 1, hi
    return (lo, hi + 1) if hi < limit else (lo - 1, hi)


def folder_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 1e6


class SessionRecorder:
    """start(frame, roi) -> add(frame) per frame until it returns False -> finish() or abort()."""

    def __init__(self, cfg: Config, person: str, kind: str, label: Optional[str], duration_s: float,
                 throws: int = 0, hand: str = "right", lighting: str = "", notes: str = "",
                 out_root: str = "data/recordings", camera_info: Optional[Dict] = None):
        if kind not in SESSION_TYPES:
            raise ValueError(f"session type must be one of {SESSION_TYPES}")
        label = "background" if kind == "background" else label
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if not person.strip():
            raise ValueError("person is required")
        self.cfg = cfg
        self.person, self.kind, self.label = person.strip(), kind, label
        self.duration_s, self.hand = duration_s, hand
        self.throws = throws if kind == "throws" else 0
        self.lighting, self.notes = lighting, notes
        self.camera_info = camera_info or {}
        self.session_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{label}"
        self.out_dir = os.path.join(out_root, self.person, self.session_id)
        self.writer: Optional[VideoWriterThread] = None
        self.rows = []
        self.t_first: Optional[float] = None
        self.missed = 0
        self._last_id: Optional[int] = None

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[self.kind].format(label=self.label.upper())

    def start(self, frame: Frame, roi: Roi):
        h, w = frame.bgr.shape[:2]
        x0, y0, x1, y1 = margin_box(roi, self.cfg.roi.record_margin, w, h)
        x0, x1 = even_span(x0, x1, roi[0], roi[0] + roi[2], w)
        y0, y1 = even_span(y0, y1, roi[1], roi[1] + roi[2], h)
        self.crop_box = [x0, y0, x1, y1]
        self.roi_camera = list(roi)
        self.roi_in_crop = [roi[0] - x0, roi[1] - y0, roi[2]]
        os.makedirs(self.out_dir, exist_ok=True)
        self.writer = VideoWriterThread(self.out_dir, (x1 - x0, y1 - y0), float(self.cfg.camera.fps))

    def add(self, frame: Frame) -> bool:
        """Queues one frame; returns False once the duration is reached."""
        x0, y0, x1, y1 = self.crop_box
        if self.t_first is None:
            self.t_first = frame.t
        if self._last_id is not None and frame.id > self._last_id + 1:
            self.missed += frame.id - self._last_id - 1
        self._last_id = frame.id
        if self.writer.put(frame.bgr[y0:y1, x0:x1].copy()):
            self.rows.append((len(self.rows), round(frame.t - self.t_first, 6), frame.id))
        return self.elapsed < self.duration_s

    @property
    def elapsed(self) -> float:
        return self.rows[-1][1] if self.rows else 0.0

    @property
    def dropped(self) -> int:
        return self.writer.dropped if self.writer else 0

    def abort(self):
        if self.writer is not None:
            self.writer.close()
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def finish(self) -> Optional[Dict]:
        """Closes the video, writes frames.csv + meta.json, verifies the frame count."""
        if self.writer is None:
            return None
        self.writer.close()
        if not self.rows:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            return None
        with open(os.path.join(self.out_dir, "frames.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["index", "t", "camera_frame_id"])
            wr.writerows(self.rows)
        is_video = self.writer.video_file == "video.mkv"
        n_video = count_video_frames(self.writer.path) if is_video else self.writer.written
        size_ok = (not is_video) or video_frame_size(self.writer.path) == tuple(self.writer.size)
        duration = self.rows[-1][1]
        meta = {
            "session_id": self.session_id,
            "person": self.person,
            "type": self.kind,
            "label": self.label,
            "hand": self.hand,
            "lighting": self.lighting,
            "notes": self.notes,
            "throws_planned": self.throws,
            "frames": len(self.rows),
            "video_frames": n_video,
            "video_check_ok": n_video == len(self.rows) and size_ok,
            "dropped_by_writer": self.writer.dropped,
            "missed_by_reader": self.missed,
            "duration_s": duration,
            "fps_measured": (len(self.rows) - 1) / duration if duration > 0 else 0.0,
            "video_file": self.writer.video_file,
            "crop_box": self.crop_box,
            "roi_camera": self.roi_camera,
            "roi_in_crop": self.roi_in_crop,
            "camera": self.cfg.to_dict()["camera"],
            "camera_info": self.camera_info,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta["size_mb"] = round(folder_size_mb(self.out_dir), 1)
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta


def list_recordings(root: str = "data/recordings"):
    """All recordings under root as a list of meta dicts (with 'dir' added)."""
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, _, files in os.walk(root):
        if "meta.json" in files:
            try:
                with open(os.path.join(dirpath, "meta.json"), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["dir"] = dirpath
                meta.setdefault("size_mb", round(folder_size_mb(dirpath), 1))
                out.append(meta)
            except (OSError, ValueError):
                continue
    return sorted(out, key=lambda m: (m.get("person", ""), m.get("session_id", "")))
