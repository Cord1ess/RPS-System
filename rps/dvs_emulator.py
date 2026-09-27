"""
Pseudo-DVS: emulates a Dynamic Vision Sensor from webcam frames, producing Dextra-style
constant-event-count 64x64 frames.

A DVS pixel fires an event each time its log intensity moves by a contrast threshold C away
from the level it last fired at. We emulate that per pixel (v2e/ESIM style) on a 128x128 ROI:

    L      = log(I + offset)                      (lookup table on uint8 gray)
    Lref  += g                                    (global gain correction, g = robust median dL)
    delta  = L - Lref
    n_on   = floor(delta / C),  n_off = floor(-delta / C)
    Lref  += (n_on - n_off) * C

Events pass a 3x3 background-activity filter, are 2x2-binned into a 64x64 histogram, and are
accumulated across webcam frames until N events are collected (activity-driven: a still scene
produces no frames at all, so the robot keeps its last decision exactly like Dextra).
Overshoot is binomially thinned to N, counts are clipped at K and scaled by 255/K
(Dextra producer.py normalization). When motion stops, a partial frame holding at least
N * flush_min_fraction events is flushed so the settled hand shape is not lost.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from rps.config import DvsConfig


@dataclass
class DvsFrame:
    image: np.ndarray       # (frame_size, frame_size) uint8, Dextra-normalized counts
    counts: np.ndarray      # (frame_size, frame_size) int, event counts after thinning, before clipping
    n_events: int           # events accumulated before thinning
    t_start: float
    t_end: float
    flushed: bool           # emitted early because motion stopped


@dataclass
class DvsStats:
    events: int                     # filtered events produced by this webcam frame
    event_map: np.ndarray           # (frame_size, frame_size) int32 binned events of this webcam frame
    centroid: Optional[Tuple[float, float]]   # (x, y) in [0, 1] ROI coords, None if too few events
    gain_offset: float              # global log-intensity correction applied this frame
    global_reset: bool
    accumulated: int                # events waiting in the accumulator after this frame


class PseudoDVS:
    def __init__(self, cfg: DvsConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.lut = np.log(np.arange(256, dtype=np.float32) + np.float32(cfg.log_offset)).astype(np.float32)
        self.bin = cfg.sensor_size // cfg.frame_size
        self.reset()

    def reset(self):
        self.last_sensor: Optional[np.ndarray] = None
        self.l_ref: Optional[np.ndarray] = None
        self.l_prev: Optional[np.ndarray] = None
        self.acc = np.zeros((self.cfg.frame_size, self.cfg.frame_size), dtype=np.int64)
        self.acc_total = 0
        self.acc_t_start: Optional[float] = None
        self.still_run = 0

    # ------------------------------------------------------------------ helpers
    def to_sensor(self, roi_img: np.ndarray) -> np.ndarray:
        """ROI (BGR or gray, any size) -> sensor_size x sensor_size uint8 gray."""
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if roi_img.ndim == 3 else roi_img
        s = self.cfg.sensor_size
        if gray.shape != (s, s):
            gray = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA)
        return gray

    def _gain_offset(self, d_l: np.ndarray) -> float:
        """Robust global log-intensity shift between consecutive frames (exposure/gain drift)."""
        g = float(np.median(d_l))
        near = np.abs(d_l - g) < self.cfg.contrast_threshold
        if near.mean() > 0.2:
            g = float(np.median(d_l[near]))
        return g

    def _emit(self, t: float, flushed: bool) -> DvsFrame:
        n, total = self.cfg.event_count, self.acc_total
        counts = self.acc
        if total > n:
            counts = self.rng.binomial(counts, n / total)
        k = self.cfg.clip_count
        img = (np.minimum(counts, k) * (255.0 / k)).astype(np.uint8)
        frame = DvsFrame(img, counts.astype(np.int32), int(total),
                         self.acc_t_start if self.acc_t_start is not None else t, t, flushed)
        self._clear_accumulator()
        return frame

    def _clear_accumulator(self):
        self.acc[:] = 0
        self.acc_total = 0
        self.acc_t_start = None

    # ------------------------------------------------------------------ main step
    def process(self, roi_img: np.ndarray, t: float) -> Tuple[Optional[DvsFrame], DvsStats]:
        """Feeds one webcam ROI image; returns (emitted frame or None, per-frame stats)."""
        cfg = self.cfg
        fs = cfg.frame_size
        gray = self.to_sensor(roi_img)
        self.last_sensor = gray
        l_now = self.lut[gray]

        if self.l_ref is None:
            self.l_ref = l_now.copy()
            self.l_prev = l_now
            return None, DvsStats(0, np.zeros((fs, fs), np.int32), None, 0.0, False, 0)

        g = self._gain_offset(l_now - self.l_prev)
        self.l_prev = l_now
        self.l_ref += g

        delta = l_now - self.l_ref
        c = cfg.contrast_threshold
        n_on = np.floor(np.maximum(delta, 0.0) / c)
        n_off = np.floor(np.maximum(-delta, 0.0) / c)
        self.l_ref += (n_on - n_off) * c
        counts = (n_on + n_off).astype(np.int32)

        fired = counts > 0
        if fired.mean() > cfg.global_reset_fraction:
            # Lighting jump the gain model could not explain: resynchronize, emit nothing.
            self.l_ref = l_now.copy()
            return None, DvsStats(0, np.zeros((fs, fs), np.int32), None, g, True, self.acc_total)

        if cfg.noise_filter and fired.any():
            neighbours = cv2.boxFilter(fired.astype(np.uint8), cv2.CV_16U, (3, 3), normalize=False)
            counts = np.where(neighbours > 1, counts, 0)   # > 1: at least one neighbour besides itself

        b = self.bin
        event_map = counts.reshape(fs, b, fs, b).sum(axis=(1, 3), dtype=np.int32)
        n_events = int(event_map.sum())

        centroid = None
        if n_events >= 20:
            ys, xs = np.indices(event_map.shape)
            centroid = (float((xs * event_map).sum() / n_events + 0.5) / fs,
                        float((ys * event_map).sum() / n_events + 0.5) / fs)

        emitted = None
        if n_events > 0:
            if self.acc_t_start is None:
                self.acc_t_start = t
            self.acc += event_map
            self.acc_total += n_events

        is_still = n_events < cfg.still_events_per_frame
        self.still_run = self.still_run + 1 if is_still else 0

        if self.acc_total >= cfg.event_count:
            emitted = self._emit(t, flushed=False)
        elif self.acc_total > 0:
            too_old = self.acc_t_start is not None and (t - self.acc_t_start) > cfg.max_accumulation_s
            if self.still_run >= cfg.flush_still_frames or too_old:
                if self.acc_total >= cfg.event_count * cfg.flush_min_fraction:
                    emitted = self._emit(t, flushed=True)
                else:
                    self._clear_accumulator()   # stale trickle: never mix it into the next throw

        return emitted, DvsStats(n_events, event_map, centroid, g, False, self.acc_total)


def events_in_box(event_map: np.ndarray, box: Optional[Tuple[float, float, float, float]]) -> int:
    """Sums a per-webcam-frame event map inside a normalized (x0, y0, x1, y1) ROI box."""
    if box is None:
        return int(event_map.sum())
    fs = event_map.shape[0]
    x0, y0, x1, y1 = box
    i0, i1 = int(np.clip(np.floor(y0 * fs), 0, fs)), int(np.clip(np.ceil(y1 * fs), 0, fs))
    j0, j1 = int(np.clip(np.floor(x0 * fs), 0, fs)), int(np.clip(np.ceil(x1 * fs), 0, fs))
    return int(event_map[i0:i1, j0:j1].sum())
