"""
Vertical hand motion in the play zone, for counting countdown pumps.

VerticalMotion: dense optical flow (Farneback) on a small grayscale copy of the play zone gives
the median vertical velocity of the moving pixels. It needs no hand detection, so it keeps
working through motion blur, and costs about 1 ms per frame.

RhythmPumpDetector: turns that velocity into pump and landing events and learns the player's
tempo and stroke size, so thresholds and time windows fit fast and slow players alike.
"""

from typing import Optional

import cv2
import numpy as np


class VerticalMotion:
    def __init__(self, size: int = 64, min_flow_px: float = 0.5, min_moving_fraction: float = 0.01):
        self.size = size
        self.min_flow_px = min_flow_px
        self.min_moving = max(8, int(min_moving_fraction * size * size))
        self.reset()

    def reset(self):
        self.prev: Optional[np.ndarray] = None
        self.prev_t: Optional[float] = None

    def update(self, gray: np.ndarray, t: float) -> Optional[float]:
        """
        Returns vertical velocity in play-zone heights per second (+ = moving down),
        0.0 when nothing moves, None on the first frame.
        """
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        prev, prev_t = self.prev, self.prev_t
        self.prev, self.prev_t = small, t
        if prev is None or t <= prev_t:
            return None
        flow = cv2.calcOpticalFlowFarneback(prev, small, None, 0.5, 3, 11, 3, 5, 1.1, 0)
        fy = flow[..., 1]
        moving = np.hypot(flow[..., 0], fy) > self.min_flow_px
        if int(moving.sum()) < self.min_moving:
            return 0.0
        return float(np.median(fy[moving])) / self.size / (t - prev_t)


class RhythmPumpDetector:
    """
    Events from vertical velocity (+ = down):
      "bottom"  a down stroke followed by an up stroke (one pump). A short pause at the bottom
                is allowed.
      "landed"  a down stroke followed by stillness that lasts longer than a pump pause.
                This is how a throw differs from a pump.

    Adaptation: minimum stroke size and speed are fractions of the player's recent strokes;
    the pause allowed at the bottom and the minimum time between pumps scale with their tempo.
    """

    def __init__(self, min_amplitude: float = 0.06, min_speed: float = 0.25, min_period_s: float = 0.15,
                 history: int = 6):
        self.base_amplitude = min_amplitude
        self.base_speed = min_speed
        self.base_period = min_period_s
        self.history = history
        self.strokes, self.speeds, self.intervals = [], [], []
        self.reset()

    def reset(self, keep_rhythm: bool = True):
        self.direction = 0              # +1 down, -1 up, 0 still
        self.stroke = 0.0               # distance of the current movement run (play-zone heights)
        self.peak = 0.0                 # peak speed of the current run
        self.down_size = 0.0
        self.down_end_t: Optional[float] = None
        self.pending_bottom = False     # down stroke reversed: wait for real upward travel
        self.pending_land_t: Optional[float] = None   # down stroke stopped: pump pause or throw?
        self.last_t: Optional[float] = None
        self.last_bottom_t = -1e9
        if not keep_rhythm:
            self.strokes, self.speeds, self.intervals = [], [], []

    # ------------------------------------------------------------------ learned rhythm
    @property
    def tempo(self) -> Optional[float]:
        """Median seconds between pump bottoms, once two intervals are known."""
        return float(np.median(self.intervals)) if len(self.intervals) >= 2 else None

    @property
    def amplitude_threshold(self) -> float:
        if len(self.strokes) >= 2:
            return max(self.base_amplitude, 0.35 * float(np.median(self.strokes)))
        return self.base_amplitude

    @property
    def speed_threshold(self) -> float:
        if len(self.speeds) >= 2:
            return max(0.6 * self.base_speed, 0.2 * float(np.median(self.speeds)))
        return self.base_speed

    @property
    def land_seconds(self) -> float:
        """Stillness after a down stroke that counts as a landing rather than a pump pause."""
        tempo = self.tempo
        return max(0.10, min(0.35, 0.35 * tempo)) if tempo else 0.15

    def _remember(self, lst, value):
        lst.append(value)
        del lst[:-self.history]

    # ------------------------------------------------------------------ per frame
    def update(self, t: float, vy: Optional[float]) -> Optional[str]:
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        if vy is None:
            return None
        v_on = self.speed_threshold
        if vy > v_on:
            d = 1
        elif vy < -v_on:
            d = -1
        elif abs(vy) < 0.5 * v_on:
            d = 0
        else:
            d = self.direction                      # hysteresis band: keep going

        if d != self.direction:
            resume = self._transition(t, d)
            self.direction = d
            self.stroke, self.peak = resume, 0.0
        if d != 0:
            self.stroke += abs(vy) * dt
            self.peak = max(self.peak, abs(vy))

        if self.pending_bottom and self.direction == -1 and self.stroke >= 0.5 * self.amplitude_threshold:
            return self._confirm_bottom()
        if self.pending_land_t is not None and self.direction == 0 \
                and t - self.pending_land_t >= self.land_seconds:
            self.pending_land_t = None
            return "landed"
        return None

    def _transition(self, t: float, new_dir: int) -> float:
        """Handles the end of the current run; returns the stroke to resume (merged down strokes)."""
        size = self.stroke
        if self.direction != 0 and size >= self.amplitude_threshold:
            self._remember(self.strokes, size)
            self._remember(self.speeds, self.peak)
        if self.direction == 1:                             # a down stroke ended
            if size >= self.amplitude_threshold:
                self.down_size, self.down_end_t = size, t
                if new_dir == -1:
                    self.pending_bottom, self.pending_land_t = True, None
                else:
                    self.pending_land_t = t
        elif self.direction == 0 and self.pending_land_t is not None:
            if new_dir == -1:                               # rose after a short pause: it was a pump
                self.pending_bottom, self.pending_land_t = True, None
            elif new_dir == 1:                              # kept going down: same stroke
                self.pending_land_t = None
                return self.down_size
        elif self.direction == -1:
            self.pending_bottom = False
        return 0.0

    def _confirm_bottom(self) -> Optional[str]:
        self.pending_bottom = False
        t = self.down_end_t
        tempo = self.tempo
        min_gap = max(self.base_period, 0.45 * tempo) if tempo else self.base_period
        if t - self.last_bottom_t < min_gap:
            return None
        if t - self.last_bottom_t < 3.0:
            self._remember(self.intervals, t - self.last_bottom_t)
        self.last_bottom_t = t
        return "bottom"
