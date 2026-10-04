"""
Frame sources for the v3 pipeline.

- CameraSource: threaded webcam grabber with latest-frame semantics, locked exposure and
  white balance, and a perf_counter timestamp + frame id on every frame. Windows (Media Foundation,
  DirectShow) and Linux (V4L2, e.g. a USB webcam on a Raspberry Pi); or an IP camera (camera.url,
  rtsp:// or http:// MJPEG) through FFmpeg with its buffering turned off.
- VideoFileSource: replays a recording made by record_session.py with its recorded timestamps
  (re-stamped on the perf_counter clock when replayed in real time, as the app does).
- MockSource: synthetic moving-blob scene for tests and hardware-free smoke runs.

Every source exposes `roi` = (x, y, size): the square play zone in its own frame coordinates.

Driver repeats: some webcam drivers (seen with MSMF in low light) hand over each image twice, the
copy ~2 ms after the original, so "30 fps" is really ~17 new images a second and every frame-count
threshold (still frames, Mediapipe repeats) is met by copies. Both the webcam and replay skip them.
"""

import csv
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from rps.config import CameraConfig, RoiConfig

Roi = Tuple[int, int, int]

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY}
WINDOWS_ONLY = ("msmf", "dshow")
# An IP camera through FFmpeg: no input buffering, so the newest image is the one read (low delay).
# TCP keeps images whole: a torn UDP image would look like movement to the Dextra view.
FFMPEG_LOW_DELAY = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|reorder_queue_size;0"


def backend_name(name: str, platform: str = sys.platform) -> str:
    """The capture driver to use here: the Windows drivers do not exist on Linux, which uses V4L2."""
    if platform != "win32" and name in WINDOWS_ONLY:
        return "v4l2"
    return name if name in BACKENDS else "any"


def fourcc_for(backend: str, fourcc: str) -> str:
    """V4L2 calls uncompressed YUY2 'YUYV'."""
    return "YUYV" if backend == "v4l2" and fourcc.upper() == "YUY2" else fourcc


def set_manual_exposure(cap, backend: str, log2_seconds: float):
    """Locks exposure. The setting is in DirectShow's unit (log2 seconds: -6 = 15.6 ms); V4L2 takes
    100 microsecond steps and 1 = manual, DirectShow/Media Foundation 0.25 = manual."""
    if backend == "v4l2":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, round(2.0 ** log2_seconds * 10000.0))
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, log2_seconds)


def set_auto_exposure(cap, backend: str):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if backend == "v4l2" else 0.75)   # V4L2: aperture priority
REPEAT_MAX_S = 0.02     # a byte-identical image this soon after the last is the driver's copy, not a new exposure


def is_repeat(img: np.ndarray, t: float, prev: Optional[np.ndarray], prev_t: Optional[float]) -> bool:
    """True for a driver's copy of the previous image. A covered lens still gives new (black) images at the
    frame rate, so it is never mistaken for one."""
    return prev is not None and t - prev_t < REPEAT_MAX_S and np.array_equal(img, prev)


@dataclass
class Frame:
    id: int
    t: float            # capture time in seconds (perf_counter for live and real-time replay, else recorded)
    bgr: np.ndarray
    live: bool = False  # True when t is on the perf_counter clock (webcam, real-time replay): latencies count


def clamp_roi(roi: Roi, width: int, height: int) -> Roi:
    """Keeps a square ROI inside the frame, shrinking it if necessary."""
    x, y, size = roi
    size = int(max(16, min(size, width, height)))
    x = int(min(max(0, x), width - size))
    y = int(min(max(0, y), height - size))
    return x, y, size


def crop_roi(img: np.ndarray, roi: Roi) -> np.ndarray:
    x, y, size = roi
    return img[y:y + size, x:x + size]


def margin_box(roi: Roi, margin: float, width: int, height: int) -> Tuple[int, int, int, int]:
    """ROI grown by `margin` on every side, clipped to the frame: (x0, y0, x1, y1)."""
    x, y, size = roi
    pad = int(round(size * margin))
    return max(0, x - pad), max(0, y - pad), min(width, x + size + pad), min(height, y + size + pad)


class CameraSource:
    """Threaded webcam capture. read() returns the newest frame not yet returned."""

    def __init__(self, cam: CameraConfig, roi: RoiConfig):
        self.cfg = cam
        self._roi_cfg = roi
        self.cap: Optional[cv2.VideoCapture] = None
        self.roi: Roi = (roi.x, roi.y, roi.size)
        self.info: Dict[str, float] = {}
        self._latest: Optional[Frame] = None
        self._last_returned = -1
        self._cond = threading.Condition()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.dropped_reads = 0
        self.repeats = 0                # driver repeats skipped (see is_repeat)
        self.backend = backend_name(cam.backend)
        self._apply_pending = False     # exposure / colour balance changed: apply between two frames

    def open(self) -> "CameraSource":
        if self.cfg.url:
            return self._open_url()
        self.backend = backend_name(self.cfg.backend)
        print(f"[camera] Opening camera {self.cfg.index} (backend={self.backend})...")
        cap = cv2.VideoCapture(self.cfg.index, BACKENDS[self.backend])
        if not cap.isOpened():
            hint = ("Check Windows camera privacy settings and close other apps using it (OBS, Teams, browser)."
                    if sys.platform == "win32" else
                    "Check that it is plugged in (ls /dev/video*) and that no other program uses it.")
            raise RuntimeError(f"Could not open camera {self.cfg.index}. {hint}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_for(self.backend, self.cfg.fourcc)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.apply_locks()
        return self._describe()

    def _open_url(self) -> "CameraSource":
        """An IP camera. Its resolution, frame rate and exposure are set in the camera's own settings."""
        self.backend = "ffmpeg"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_LOW_DELAY
        print(f"[camera] Opening IP camera {self.cfg.url}...")
        cap = cv2.VideoCapture(self.cfg.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open the IP camera at {self.cfg.url}. Check the address (open it in a "
                               f"browser or VLC) and that the laptop or Pi is on its network.")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        return self._describe()

    def _describe(self) -> "CameraSource":
        cap = self.cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.roi = clamp_roi((self._roi_cfg.x, self._roi_cfg.y, self._roi_cfg.size), w, h)
        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.info = {
            "width": w,
            "height": h,
            "fps_reported": cap.get(cv2.CAP_PROP_FPS),
            "fourcc": "".join(chr((fourcc_val >> (8 * i)) & 0xFF) for i in range(4)),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "wb_temperature": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
            "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
            "backend": self.backend,
        }
        print(f"[camera] Opened: {self.info}")
        return self

    def apply_locks(self):
        """
        Locks exposure and white balance when configured (a DVS has no auto-gain). The FHD
        Camera pins gain at minimum in manual mode, so only lock in a well-lit play zone.
        """
        cap = self.cap
        self._locked = False
        if self.backend == "ffmpeg":                    # an IP camera: set in its own settings
            return
        if self.cfg.lock_exposure:
            set_manual_exposure(cap, self.backend, self.cfg.exposure)
            self._locked = True
        if self.cfg.lock_white_balance:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.cfg.wb_temperature)
            self._locked = True

    LIVE_KEYS = ("lock_exposure", "exposure", "lock_white_balance", "wb_temperature", "mirror")

    def request_apply(self):
        """Applies the exposure and colour balance settings to the running camera at the next frame (about
        35-55 ms on the demo laptop's camera; no reopening). Mirroring already applies to every frame."""
        self._apply_pending = True

    def _apply_live(self):
        cap = self.cap
        if cap is None or self.backend == "ffmpeg":       # an IP camera: set in its own settings
            return
        if self.cfg.lock_exposure:
            set_manual_exposure(cap, self.backend, self.cfg.exposure)
        else:
            set_auto_exposure(cap, self.backend)
        if self.cfg.lock_white_balance:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.cfg.wb_temperature)
        else:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        self._locked = self.cfg.lock_exposure or self.cfg.lock_white_balance

    def restore_auto(self):
        """Returns the driver to automatic exposure/white balance so other apps look normal."""
        if self.cap is None or not getattr(self, "_locked", False):
            return
        set_auto_exposure(self.cap, self.backend)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    def start(self) -> "CameraSource":
        if self.cap is None:
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        frame_id = 0
        prev, prev_t = None, None
        while self._running:
            if self._apply_pending:                         # set by the UI; applied here, between reads
                self._apply_pending = False
                self._apply_live()
            ok, img = self.cap.read()
            t = time.perf_counter()
            if not ok or img is None:
                self.dropped_reads += 1
                time.sleep(0.005)
                continue
            if is_repeat(img, t, prev, prev_t):
                self.repeats += 1
                continue
            prev, prev_t = img, t
            if self.cfg.mirror:
                img = cv2.flip(img, 1)
            with self._cond:
                self._latest = Frame(frame_id, t, img, live=True)
                self._cond.notify_all()
            frame_id += 1

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._latest is None or self._latest.id <= self._last_returned:
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or not self._running:
                    return None
                self._cond.wait(remaining)
            frame = self._latest
            self._last_returned = frame.id
            return frame

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            if self.cfg.restore_auto_on_exit:
                self.restore_auto()
            self.cap.release()
            self.cap = None
        print("[camera] Released.")


class VideoFileSource:
    """Replays a record_session.py recording (video.mkv + frames.csv + meta.json)."""

    def __init__(self, recording_dir: str, realtime: bool = False):
        self.dir = recording_dir
        with open(os.path.join(recording_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.times = []
        with open(os.path.join(recording_dir, "frames.csv"), "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.times.append(float(row["t"]))
        video_path = os.path.join(recording_dir, self.meta.get("video_file", "video.mkv"))
        if os.path.isdir(video_path):
            self.cap = None
            self.png_files = sorted(os.listdir(video_path))
        else:
            self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open recording video: {video_path}")
            self.png_files = None
        self.roi: Roi = tuple(self.meta["roi_in_crop"])
        self.realtime = realtime
        self._idx = 0
        self._t0_wall = None
        self._prev, self._prev_t = None, None
        self.repeats = 0           # driver repeats skipped

    def __len__(self):
        return len(self.times)

    def start(self) -> "VideoFileSource":
        return self

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        while True:                # skips the driver's repeats, as the webcam does live
            if self._idx >= len(self.times):
                return None
            if self.cap is not None:
                ok, img = self.cap.read()
                if not ok:
                    return None
            else:
                img = cv2.imread(os.path.join(self.dir, self.meta["video_file"], self.png_files[self._idx]))
            t = self.times[self._idx]
            if not is_repeat(img, t, self._prev, self._prev_t):
                break
            self.repeats += 1
            self._idx += 1
        self._prev, self._prev_t = img, t
        if self.realtime:          # paced like a camera and stamped on its clock (perf_counter), as the beat is
            if self._t0_wall is None:
                self._t0_wall = time.perf_counter() - t
            t = self._t0_wall + t
            delay = t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        frame = Frame(self._idx, t, img, live=self.realtime)
        self._idx += 1
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()


class MockSource:
    """
    Synthetic scene: textured background with a bright blob that pumps up and down (1.5 Hz),
    then holds still. Produces real pseudo-events without a camera.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0,
                 roi: Optional[Roi] = None, realtime: bool = True, seed: int = 0):
        self.width, self.height, self.fps = width, height, fps
        size = min(width, height) * 3 // 4
        self.roi: Roi = roi or ((width - size) // 2, (height - size) // 2, size)
        self.realtime = realtime
        rng = np.random.default_rng(seed)
        bg = rng.integers(60, 120, (height // 8, width // 8), dtype=np.uint8)
        self.background = cv2.resize(bg, (width, height), interpolation=cv2.INTER_NEAREST)
        self._idx = 0
        self._t0 = None

    def start(self) -> "MockSource":
        self._t0 = time.perf_counter()
        return self

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        t = self._idx / self.fps
        scene_t = t
        if self.realtime:          # stamped on the camera clock (perf_counter), as a webcam frame is
            t = self._t0 + scene_t
            delay = t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        img = cv2.cvtColor(self.background, cv2.COLOR_GRAY2BGR)
        x, y, size = self.roi
        cycle = scene_t % 4.0
        # 0-2 s: pump (1.5 Hz vertical motion), 2-4 s: hold still
        offset = 0.18 * size * np.sin(2 * np.pi * 1.5 * cycle) if cycle < 2.0 else 0.0
        cx, cy = x + size // 2, int(y + size // 2 + offset)
        cv2.ellipse(img, (cx, cy), (size // 6, size // 5), 0, 0, 360, (210, 200, 190), -1)
        frame = Frame(self._idx, t, img, live=self.realtime)
        self._idx += 1
        return frame

    def stop(self):
        pass


def open_source(cam: CameraConfig, roi: RoiConfig, video: Optional[str] = None,
                mock: bool = False, realtime: bool = True):
    """Factory used by play.py / replay tools."""
    if video:
        return VideoFileSource(video, realtime=realtime).start()
    if mock:
        return MockSource(cam.width, cam.height, cam.fps, realtime=realtime).start()
    return CameraSource(cam, roi).start()
