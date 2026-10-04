"""
Camera thread for the desktop app.

One worker owns the frame source (webcam, mock scene or a recording). Each frame is handed to the
active tab's processor *in this thread* (DVS emulation, CNN, MediaPipe, recording), and only the
newest result is published; the UI pulls it with take_latest(), so a slow repaint can never make
the pipeline fall behind the camera.
"""

import threading
import time
import traceback
from collections import deque
from typing import Callable, Optional

from PySide6.QtCore import QThread, Signal

from rps.camera import clamp_roi, open_source

STALL_S = 1.5      # no frame for this long from a live source: report "stalled"


class CameraWorker(QThread):
    frame_ready = Signal()
    state_changed = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.kind = "camera"            # "camera" | "mock" | "video"
        self.video_path: Optional[str] = None
        self.source = None
        self._processor: Optional[Callable] = None
        self._lock = threading.Lock()
        self._latest = None
        self._stop = False
        self.fps = 0.0
        self.info = {}

    def set_processor(self, fn: Optional[Callable]):
        with self._lock:
            self._processor = fn

    def set_roi(self, x: int, y: int, size: int):
        src = self.source
        if src is not None:
            frame_w = self.info.get("width", self.state.cfg.camera.width)
            frame_h = self.info.get("height", self.state.cfg.camera.height)
            src.roi = clamp_roi((x, y, size), frame_w, frame_h)

    def take_latest(self):
        with self._lock:
            payload, self._latest = self._latest, None
        return payload

    def begin(self, kind: str, video_path: Optional[str] = None):
        """Starts the thread. The stop flag is cleared here, not in run(), so a stop() issued right
        after begin() can never be lost."""
        self.kind, self.video_path = kind, video_path
        self._stop = False
        self.start()

    def run(self):
        cfg = self.state.cfg
        try:
            src = open_source(cfg.camera, cfg.roi, video=self.video_path if self.kind == "video" else None,
                              mock=self.kind == "mock", realtime=True)
        except Exception as e:  # camera busy, privacy settings, missing file...
            self.state_changed.emit(f"error: {e}")
            return
        self.source = src
        self.info = dict(getattr(src, "info", {}) or {})
        self.state_changed.emit("running")
        stamps = deque(maxlen=30)
        last_frame, stalled = time.perf_counter(), False
        try:
            while not self._stop:
                frame = src.read(timeout=0.5)
                if frame is None:
                    if self.kind == "video":
                        break
                    if not stalled and time.perf_counter() - last_frame > STALL_S:
                        stalled = True                      # e.g. another app took the webcam
                        self.state_changed.emit("stalled")
                    continue
                last_frame = time.perf_counter()
                if stalled:
                    stalled = False
                    self.state_changed.emit("running")
                stamps.append(time.perf_counter())
                if len(stamps) > 1:
                    self.fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                if "width" not in self.info:
                    self.info["height"], self.info["width"] = frame.bgr.shape[:2]
                with self._lock:
                    proc = self._processor
                try:
                    payload = proc(frame, src) if proc is not None else {"display": frame.bgr}
                except Exception:
                    payload = {"display": frame.bgr, "error": traceback.format_exc()}
                payload["fps"] = self.fps              # new images per second (driver repeats are skipped)
                payload["roi"] = src.roi
                payload["repeats"] = getattr(src, "repeats", 0)
                with self._lock:
                    self._latest = payload
                self.frame_ready.emit()
        finally:
            src.stop()
            self.source = None
            self.state_changed.emit("stopped")

    @property
    def stopping(self) -> bool:
        return self.isRunning() and self._stop

    def request_stop(self):
        """Asks the thread to stop and returns at once: releasing a webcam takes ~0.5 s, which would
        freeze the window. state_changed("stopped") and finished follow."""
        self._stop = True

    def stop(self):
        """Stops and waits (for when the camera must be free right away, e.g. before a camera tool)."""
        self._stop = True
        self.wait(4000)

    def apply_camera_settings(self):
        """Exposure and colour balance changed: the running webcam takes them without reopening."""
        src = self.source
        if src is not None and hasattr(src, "request_apply"):
            src.request_apply()
