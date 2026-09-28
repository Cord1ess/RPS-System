"""
Vertical hand motion in the play zone, for counting countdown pumps.

VerticalMotion: dense optical flow (Farneback) on a small grayscale copy of the play zone gives
the median vertical velocity of the moving pixels. It needs no hand detection, so it keeps
working through motion blur, and costs about 1 ms per frame.

PumpDetector: finds pumps in the hand's height, taken from the tracked wrist when Mediapipe sees
the hand and carried through gaps by that vertical motion. It learns the player's pump size and
tempo, so small and large, fast and slow pumps are all counted.
"""

from typing import Optional

import cv2
import numpy as np

MAX_PUMP_INTERVAL_S = 1.2   # a longer gap between bottoms is a pause (e.g. after a throw), not the pump rhythm


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


class PumpDetector:
    """
    Events from the hand's height y (play-zone heights, + = down):
      "bottom"  a low point followed by a rise of at least rise_threshold: one pump. Checked on a
                real recording: 57 of 57 pumps (the previous speed-based detector caught 48).
      "landed"  a down stroke that stops and stays down longer than a pump turn: the throw.

    rise_threshold is a fraction of the player's own recent swings (never below min_rise), so it
    follows the size of their pumps; the minimum time between pumps and the landing time follow
    their tempo. Using the height rather than the speed matters: fingers opening into scissors move
    up quickly, but the hand does not rise, so a throw is not mistaken for a pump.
    """

    def __init__(self, min_rise: float = 0.02, rise_fraction: float = 0.3, min_period_s: float = 0.12,
                 first_rise: float = 0.05, history: int = 6):
        self.min_rise = min_rise
        self.rise_fraction = rise_fraction
        self.min_period = min_period_s
        self.first_rise = first_rise
        self.history = history
        self.swings, self.intervals = [], []
        self.reset()

    def reset(self, keep_rhythm: bool = True, ignore_until: float = -1e9):
        """
        ignore_until: low points reached before this time are not pumps. Used after a throw: the
        hand is still sinking into it (measured 0.2-0.3 s after the decision), while the next
        round's first pump comes a beat or more later.
        """
        self.ignore_until = ignore_until
        self.y: Optional[float] = None            # current height estimate
        self.last_t: Optional[float] = None
        self.from_wrist = False                   # the last height came from the tracked wrist
        self.trend = 0                            # +1 moving down, -1 moving up, 0 unknown
        self.ext: Optional[float] = None          # lowest (trend >= 0) or highest (trend -1) point so far
        self.ext_t: Optional[float] = None
        self.top: Optional[float] = None          # last confirmed high point
        self.last_turn: Optional[float] = None    # height of the last confirmed high or low point
        self.landed_reported = False
        self.last_bottom_t = -1e9
        if not keep_rhythm:
            self.swings, self.intervals = [], []

    # ------------------------------------------------------------------ learned rhythm
    @property
    def tempo(self) -> Optional[float]:
        """Median seconds between pumps, once two intervals are known."""
        return float(np.median(self.intervals)) if len(self.intervals) >= 2 else None

    @property
    def rise_threshold(self) -> float:
        if len(self.swings) >= 2:
            return max(self.min_rise, self.rise_fraction * float(np.median(self.swings)))
        return max(self.min_rise, self.first_rise)

    @property
    def land_seconds(self) -> float:
        """Stillness at the bottom that means a throw landed rather than a pump turning."""
        tempo = self.tempo
        return max(0.10, min(0.35, 0.4 * tempo)) if tempo else 0.15

    def _remember(self, lst, value):
        lst.append(value)
        del lst[:-self.history]

    # ------------------------------------------------------------------ per frame
    def update(self, t: float, vy: Optional[float], wrist_y: Optional[float] = None) -> Optional[str]:
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        if wrist_y is not None:                       # the tracked wrist is the height when it is seen
            if self.y is not None and not self.from_wrist:
                self._shift(wrist_y - self.y)         # re-found after a gap: move the reference, not the hand
            self.y, self.from_wrist = wrist_y, True
        elif vy is not None:                          # otherwise the play zone's motion carries the height
            self.y = (self.y or 0.0) + vy * dt
            self.from_wrist = False
        if self.y is None:
            return None
        y = self.y
        if self.ext is None:
            self.ext, self.ext_t = y, t
            return None
        rise = self.rise_threshold
        if self.trend >= 0:                           # going down (or not known yet): track the low point
            if y > self.ext:
                self.ext, self.ext_t, self.landed_reported = y, t, False
            elif self.ext - y >= rise:                # it rose enough: that low point was a pump
                bottom_t = self.ext_t
                self._turn(self.ext)
                self.trend, self.ext, self.ext_t = -1, y, t
                return None if bottom_t < self.ignore_until else self._bottom(bottom_t)
            elif (not self.landed_reported and self.top is not None and self.ext - self.top >= rise
                  and t - self.ext_t >= self.land_seconds):
                self.landed_reported = True           # came down and stayed down: the throw landed
                return "landed"
            return None
        if y < self.ext:                              # going up: track the high point
            self.ext, self.ext_t = y, t
        elif y - self.ext >= rise:                    # it fell enough: that was the top of the pump
            self._turn(self.ext)
            self.top = self.ext
            self.trend, self.ext, self.ext_t, self.landed_reported = 1, y, t, False
        return None

    def _turn(self, turn_y: float):
        """A confirmed high or low point: the distance from the previous one is a swing."""
        if self.last_turn is not None:
            self._remember(self.swings, abs(turn_y - self.last_turn))
        self.last_turn = turn_y

    def _shift(self, offset: float):
        for name in ("ext", "top", "last_turn"):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name) + offset)

    def _bottom(self, t: float) -> Optional[str]:
        tempo = self.tempo
        min_gap = max(self.min_period, 0.3 * tempo) if tempo else self.min_period
        gap = t - self.last_bottom_t
        if gap < min_gap:                             # a wobble at the same low point
            return None
        if gap <= MAX_PUMP_INTERVAL_S:
            self._remember(self.intervals, gap)
        self.last_bottom_t = t
        return "bottom"
