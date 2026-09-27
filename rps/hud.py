"""
On-screen HUD for play.py / record_session.py.

Layout: title banner (top), pseudo-DVS preview drawn BELOW the banner (the v2 HUD painted the
banner over its mask preview), play-zone ROI with MediaPipe landmarks, decision card (bottom).
Raw CNN and MediaPipe readings are shown separately from the committed decision so a
confidence value is never attached to the wrong class.
"""

from typing import Dict, Optional

import cv2
import numpy as np

from rps.camera import Roi
from rps.decision import GESTURE_NAME, Snapshot
from rps.dvs_emulator import DvsFrame
from rps.hand_tracker import HandObs

POSE_NAME = {"R": "ROCK", "P": "PAPER", "S": "SCISSORS", "N": "READY"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H = 50
PREVIEW = 150


def draw_roi(img: np.ndarray, roi: Roi, color=(0, 220, 0)):
    x, y, s = roi
    cv2.rectangle(img, (x, y), (x + s, y + s), color, 2)


def draw_hand(img: np.ndarray, roi: Roi, hand: Optional[HandObs]):
    if hand is None or not hand.present or hand.landmarks is None:
        return
    x, y, s = roi
    for lx, ly in hand.landmarks:
        cv2.circle(img, (int(x + lx * s), int(y + ly * s)), 3, (255, 200, 0), -1)
    if hand.box is not None:
        x0, y0, x1, y1 = hand.box
        cv2.rectangle(img, (int(x + x0 * s), int(y + y0 * s)), (int(x + x1 * s), int(y + y1 * s)),
                      (255, 200, 0), 1)


def draw_dvs_preview(img: np.ndarray, dvs_frame, label: str = "pseudo-DVS"):
    """dvs_frame: a DvsFrame, a 64x64 uint8 array, or None."""
    h, w = img.shape[:2]
    x0, y0 = w - PREVIEW - 10, BANNER_H + 10
    frame = dvs_frame.image if isinstance(dvs_frame, DvsFrame) else dvs_frame
    if frame is not None:
        big = cv2.resize(frame, (PREVIEW, PREVIEW), interpolation=cv2.INTER_NEAREST)
        img[y0:y0 + PREVIEW, x0:x0 + PREVIEW] = cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)
    else:
        img[y0:y0 + PREVIEW, x0:x0 + PREVIEW] = 0
    cv2.rectangle(img, (x0, y0), (x0 + PREVIEW - 1, y0 + PREVIEW - 1), (255, 255, 0), 1)
    cv2.putText(img, label, (x0, y0 + PREVIEW + 16), FONT, 0.45, (255, 255, 0), 1)


def draw_banner(img: np.ndarray, title: str, stats: Dict[str, Optional[float]]):
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, BANNER_H), (20, 20, 20), -1)
    cv2.putText(img, title, (12, 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
    fps = stats.get("fps") or 0.0
    parts = [f"cam {fps:4.1f} fps"]
    for key, name in (("dvs_ms", "dvs"), ("cnn_ms", "cnn"), ("mp_ms", "mp"), ("grab_to_send_ms", "grab->send")):
        v = stats.get(key)
        if v is not None:
            parts.append(f"{name} {v:4.1f}ms")
    color = (0, 255, 0) if fps >= 27 else (0, 165, 255)
    cv2.putText(img, " | ".join(parts), (12, 41), FONT, 0.45, color, 1)


def draw_card(img: np.ndarray, snap: Snapshot, raw_cnn: Optional[str], raw_mp: Optional[str],
              link: Optional[Dict], source_mode: str):
    h, w = img.shape[:2]
    y0 = h - 118
    cv2.rectangle(img, (10, y0), (w - 10, h - 10), (15, 15, 15), -1)
    cv2.rectangle(img, (10, y0), (w - 10, h - 10), (60, 60, 60), 1)
    human = GESTURE_NAME.get(snap.human, "-") if snap.human is not None else "-"
    robot = POSE_NAME.get(snap.pose, snap.pose)
    cv2.putText(img, f"YOU: {human.upper()}", (22, y0 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (220, 220, 220), 1)
    cv2.putText(img, f"ROBOT: {robot}", (260, y0 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 255, 0) if snap.pose != "N" else (150, 150, 150), 2)
    state = snap.state + (f"  pumps {snap.pumps}" if snap.mode == "countdown" else "")
    cv2.putText(img, f"mode {snap.mode} [{source_mode}] | {state} | {snap.reason}", (22, y0 + 56), FONT, 0.45,
                (0, 255, 255), 1)
    cv2.putText(img, f"raw CNN: {raw_cnn or '-'}   raw MP: {raw_mp or '-'}   motion: {'yes' if snap.motion_active else 'no'}",
                (22, y0 + 78), FONT, 0.45, (200, 200, 200), 1)
    if link is None:
        link_txt, color = "ESP: disabled", (150, 150, 150)
    elif link.get("connected"):
        link_txt, color = f"ESP: connected, rtt {link.get('rtt_median_ms') or 0:.1f} ms", (0, 255, 0)
    else:
        link_txt, color = f"ESP: no ack (sent {link.get('sent', 0)})", (0, 0, 255)
    cv2.putText(img, f"commits {snap.commits}  switches {snap.switches}  |  {link_txt}", (22, y0 + 100), FONT, 0.45,
                color, 1)
