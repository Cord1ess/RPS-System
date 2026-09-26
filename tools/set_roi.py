"""
Choose the play-zone ROI: drag a box around where the hand plays (exclude your face and body,
like Dextra's camera framing), press ENTER/SPACE to accept. The box is made square and saved
to config.json.

Usage: python tools/set_roi.py
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rps.camera import CameraSource, clamp_roi  # noqa: E402
from rps.config import load_config, save_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Select the square play-zone ROI")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cam = CameraSource(cfg.camera, cfg.roi).start()
    try:
        frame = None
        t_end = time.perf_counter() + 1.5          # let exposure settle
        while time.perf_counter() < t_end or frame is None:
            frame = cam.read(timeout=2.0) or frame
        img = frame.bgr.copy()
        x, y, s = cam.roi
        cv2.rectangle(img, (x, y), (x + s, y + s), (0, 220, 0), 1)
        cv2.putText(img, "Drag the play zone (hand only), ENTER to accept, c to cancel", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        bx, by, bw, bh = cv2.selectROI("Select play zone", img, showCrosshair=True)
        cv2.destroyAllWindows()
        if bw == 0 or bh == 0:
            print("[set_roi] Cancelled; config unchanged.")
            return
        size = max(bw, bh)
        cx, cy = bx + bw // 2, by + bh // 2
        h, w = img.shape[:2]
        cfg.roi.x, cfg.roi.y, cfg.roi.size = clamp_roi((cx - size // 2, cy - size // 2, size), w, h)
        print(f"[set_roi] ROI = x {cfg.roi.x}, y {cfg.roi.y}, size {cfg.roi.size}")
        save_config(cfg, args.config)
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
