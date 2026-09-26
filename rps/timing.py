"""
Per-frame latency and decision log: rolling stats for the HUD, optional CSV for analysis.
"""

import csv
from collections import deque
from typing import Dict, Optional

import numpy as np

FIELDS = [
    "frame_id", "t_capture", "events", "emitted", "dvs_ms", "cnn_ms", "mp_ms", "total_ms",
    "grab_to_send_ms", "cnn_label", "cnn_conf", "mp_present", "mp_gesture", "mp_conf", "mp_skipped",
    "events_in_hand", "pose", "state", "human", "source",
]


class LatencyLog:
    def __init__(self, csv_path: Optional[str] = None, window: int = 100):
        self.window = window
        self.hist: Dict[str, deque] = {}
        self.t_capture = deque(maxlen=window)
        self._file = None
        self._writer = None
        if csv_path:
            self._file = open(csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=FIELDS, extrasaction="ignore")
            self._writer.writeheader()

    def add(self, rec: Dict):
        self.t_capture.append(rec["t_capture"])
        for key in ("dvs_ms", "cnn_ms", "mp_ms", "total_ms", "grab_to_send_ms"):
            v = rec.get(key)
            if v is not None and v == v:
                self.hist.setdefault(key, deque(maxlen=self.window)).append(v)
        if self._writer is not None:
            self._writer.writerow(rec)

    def mean(self, key: str) -> Optional[float]:
        h = self.hist.get(key)
        return float(np.mean(h)) if h else None

    def fps(self) -> float:
        if len(self.t_capture) < 2:
            return 0.0
        span = self.t_capture[-1] - self.t_capture[0]
        return (len(self.t_capture) - 1) / span if span > 0 else 0.0

    def summary(self) -> Dict[str, float]:
        out = {"fps": self.fps()}
        for key, h in self.hist.items():
            arr = np.asarray(h)
            out[f"{key}_mean"] = float(arr.mean())
            out[f"{key}_p95"] = float(np.percentile(arr, 95))
        return out

    def close(self):
        if self._file is not None:
            self._file.close()
