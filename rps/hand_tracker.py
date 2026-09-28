"""
MediaPipe hand path: the still-hand authority and fallback classifier.

HandLandmarker (Tasks API, VIDEO mode) runs on the play-zone ROI plus a margin. Finger curl is
measured from 3D `hand_world_landmarks` joint angles, so fingers pointing at the camera still
read correctly. Each finger has a two-threshold hysteresis so it cannot flicker at the boundary.

Rules (index, middle, ring, pinky; thumb ignored because it is ambiguous in all three symbols):
    scissors = index + middle extended (or the middle hidden behind the index), ring + pinky curled
    paper    = at least 3 of 4 extended
    rock     = at most 1 extended
    else     = unknown (-1)
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from rps.camera import Roi, margin_box
from rps.config import HandConfig

ROCK, PAPER, SCISSORS, BACKGROUND, UNKNOWN = 0, 1, 2, 3, -1

# (wrist, MCP, PIP, DIP, TIP) for index, middle, ring, pinky
FINGER_CHAINS = [(0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
MAX_FLEXION_RAD = np.deg2rad(270.0)   # ~90 deg MCP + ~100 deg PIP + ~80 deg DIP
# Sideways scissors hide the middle finger behind the index, and the tracker then guesses it as
# slightly bent (measured ~0.5). A fist never has a straight index, so index straight + ring and
# pinky bent + middle only this bent is still scissors (held scissors 93% -> 98% correct per frame).
SCISSORS_HIDDEN_MIDDLE_MAX = 0.55
HAND_RULES_VERSION = 2      # bump whenever the rules change, so cached hand-tracker labels are redone


@dataclass
class HandObs:
    present: bool
    gesture: int                    # ROCK/PAPER/SCISSORS, UNKNOWN, or BACKGROUND when no hand
    confidence: float
    curls: Optional[np.ndarray] = None          # (4,) 0 = straight .. 1 = fully curled
    extended: Optional[List[bool]] = None
    wrist_y: Optional[float] = None             # ROI-normalized
    box: Optional[Tuple[float, float, float, float]] = None   # ROI-normalized (x0, y0, x1, y1)
    landmarks: Optional[np.ndarray] = None      # (21, 2) ROI-normalized image coords
    ms: float = 0.0
    skipped: bool = False


NO_HAND = HandObs(present=False, gesture=BACKGROUND, confidence=1.0)


def finger_curls(world: np.ndarray) -> np.ndarray:
    """Total flexion per finger from 3D landmarks (21, 3), normalized to [0, 1]."""
    curls = np.zeros(4, dtype=np.float32)
    for f, chain in enumerate(FINGER_CHAINS):
        pts = world[list(chain)]
        vecs = np.diff(pts, axis=0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        unit = vecs / norms
        cosines = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
        curls[f] = float(np.sum(np.arccos(cosines)) / MAX_FLEXION_RAD)
    return np.clip(curls, 0.0, 1.0)


def classify_curls(curls: Sequence[float], prev_extended: Optional[List[bool]],
                   extend_threshold: float, curl_threshold: float) -> Tuple[int, float, List[bool]]:
    """Applies per-finger hysteresis, then the RPS rules. Returns (gesture, confidence, extended)."""
    mid = 0.5 * (extend_threshold + curl_threshold)
    extended = []
    for i, c in enumerate(curls):
        if c < extend_threshold:
            extended.append(True)
        elif c > curl_threshold:
            extended.append(False)
        else:   # inside the hysteresis band: keep the previous state
            extended.append(prev_extended[i] if prev_extended is not None else c < mid)
    idx, mid_f, ring, pinky = extended
    n_ext = sum(extended)
    if idx and not ring and not pinky and (mid_f or curls[1] < SCISSORS_HIDDEN_MIDDLE_MAX):
        gesture = SCISSORS
    elif n_ext >= 3:
        gesture = PAPER
    elif n_ext <= 1:
        gesture = ROCK
    else:
        gesture = UNKNOWN
    certainty = np.clip(np.abs(np.asarray(curls, dtype=np.float32) - mid) / 0.2, 0.0, 1.0)
    confidence = float(np.mean(certainty)) if gesture != UNKNOWN else 0.0
    return gesture, confidence, extended


def tracker_fingerprint(cfg: HandConfig) -> str:
    """Changes whenever stored Mediapipe readings could differ: model file, settings or rules."""
    import hashlib
    import json
    from dataclasses import asdict
    size = os.path.getsize(cfg.model_path) if os.path.exists(cfg.model_path) else 0
    key = f"{size}|{json.dumps(asdict(cfg), sort_keys=True)}|{HAND_RULES_VERSION}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


class HandTracker:
    def __init__(self, cfg: HandConfig):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(
                f"MediaPipe model '{cfg.model_path}' not found. Download it with:\n"
                "  curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
        self.cfg = cfg
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self._extended: Optional[List[bool]] = None
        self._skip_next = False
        self.skipped_frames = 0

    def process(self, frame_bgr: np.ndarray, roi: Roi, t: float, allow_skip: bool = True) -> HandObs:
        """Runs MediaPipe on the ROI (+ margin) of one frame; ROI-normalized outputs."""
        if allow_skip and self._skip_next:
            self._skip_next = False
            self.skipped_frames += 1
            return HandObs(present=False, gesture=UNKNOWN, confidence=0.0, skipped=True)

        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = margin_box(roi, self.cfg.crop_margin, w, h)
        crop = np.ascontiguousarray(cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        ts_ms = max(int(round(t * 1000.0)), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self.landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=crop), ts_ms)
        ms = (time.perf_counter() - t0) * 1000.0
        self._skip_next = ms > self.cfg.frame_budget_ms

        if not result.hand_landmarks:
            self._extended = None
            return HandObs(present=False, gesture=BACKGROUND, confidence=1.0, ms=ms)

        rx, ry, rsize = roi
        cw, ch = x1 - x0, y1 - y0
        img_lm = np.array([[lm.x, lm.y] for lm in result.hand_landmarks[0]], dtype=np.float32)
        roi_lm = np.empty_like(img_lm)
        roi_lm[:, 0] = (img_lm[:, 0] * cw + x0 - rx) / rsize
        roi_lm[:, 1] = (img_lm[:, 1] * ch + y0 - ry) / rsize
        world = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]], dtype=np.float32)

        curls = finger_curls(world)
        gesture, confidence, self._extended = classify_curls(
            curls, self._extended, self.cfg.extend_threshold, self.cfg.curl_threshold)
        pad = 0.05
        box = (float(roi_lm[:, 0].min() - pad), float(roi_lm[:, 1].min() - pad),
               float(roi_lm[:, 0].max() + pad), float(roi_lm[:, 1].max() + pad))
        return HandObs(present=True, gesture=gesture, confidence=confidence, curls=curls,
                       extended=list(self._extended), wrist_y=float(roi_lm[0, 1]), box=box,
                       landmarks=roi_lm, ms=ms)

    def close(self):
        self.landmarker.close()
