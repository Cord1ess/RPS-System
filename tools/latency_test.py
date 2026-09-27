"""
Measures camera latency: time from a real-world light change to the frame being available in
software (the number to put in config latency.camera_latency_ms).

Modes:
    led     The ESP32 switches its LED (protocol message L,<seq>,<0|1>). Put the LED inside the
            play-zone ROI. Measures UDP + ESP + camera latency (UDP/ESP are a few ms).
    screen  A window flashes black/white; hold a mirror so the webcam sees the window inside the
            play zone. Measures display + camera latency (pessimistic: includes monitor lag).

Usage:
    python tools/latency_test.py --mode led --trials 20
    python tools/latency_test.py --mode screen --trials 20
"""

import argparse
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rps.camera import CameraSource, crop_roi  # noqa: E402
from rps.config import load_config  # noqa: E402
from rps.perf import boost_process  # noqa: E402
from rps.robot_link import RobotLink  # noqa: E402

FLASH_WINDOW = "latency-flash"


def roi_brightness(cam, frame) -> float:
    return float(cv2.cvtColor(crop_roi(frame.bgr, cam.roi), cv2.COLOR_BGR2GRAY).mean())


def level_after(cam, seconds: float) -> float:
    """Mean ROI brightness over the last frames of a settling period."""
    vals, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        f = cam.read(timeout=1.0)
        if f is not None:
            vals.append(roi_brightness(cam, f))
    return float(np.median(vals[-5:])) if vals else 0.0


def main():
    parser = argparse.ArgumentParser(description="Measure camera latency")
    parser.add_argument("--mode", choices=["led", "screen"], required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    boost_process()
    cfg = load_config(args.config)
    cam = CameraSource(cfg.camera, cfg.roi).start()
    link = None
    white = np.full((400, 400, 3), 255, np.uint8)
    black = np.zeros_like(white)

    def stimulus(on: bool) -> float:
        if args.mode == "led":
            link.send_led(on)
            return time.perf_counter()
        cv2.imshow(FLASH_WINDOW, white if on else black)
        cv2.waitKey(1)
        return time.perf_counter()

    try:
        if args.mode == "led":
            link = RobotLink(cfg.robot.host, cfg.robot.port, heartbeat_s=10.0).start()
        else:
            cv2.namedWindow(FLASH_WINDOW, cv2.WINDOW_NORMAL)
        stimulus(False)
        off = level_after(cam, 1.0)
        stimulus(True)
        on = level_after(cam, 1.0)
        if abs(on - off) < 5:
            raise SystemExit(f"[latency] Stimulus not visible in the ROI (off {off:.1f}, on {on:.1f}). "
                             f"Place the LED / mirror image inside the play zone.")
        threshold = 0.5 * (on + off)
        rising = on > off
        print(f"[latency] off level {off:.1f}, on level {on:.1f}, threshold {threshold:.1f}")

        latencies = []
        period = 1.0 / max(cfg.camera.fps, 1)
        for trial in range(args.trials):
            stimulus(False)
            level_after(cam, 0.5)
            # level_after() returns just after a frame arrived; without a random wait the stimulus would
            # always land at the same point of the frame cycle and every trial would read the same
            # whole number of frame periods instead of the average latency.
            time.sleep(random.uniform(0.0, period))
            t_on = stimulus(True)
            t_end = t_on + 1.0
            while time.perf_counter() < t_end:
                f = cam.read(timeout=1.0)
                if f is None or f.t < t_on:
                    continue
                b = roi_brightness(cam, f)
                if (b > threshold) == rising:
                    latencies.append((f.t - t_on) * 1000.0)
                    print(f"  trial {trial + 1:2d}: {latencies[-1]:6.1f} ms")
                    break
        if latencies:
            arr = np.array(latencies)
            print(f"\n[latency] {args.mode}: median {np.median(arr):.1f} ms, mean {arr.mean():.1f} ms, "
                  f"min {arr.min():.1f}, max {arr.max():.1f} ({len(arr)} trials)")
            print(f"          Set latency.camera_latency_ms to about {np.median(arr):.0f} in config.json"
                  + (" (minus your monitor's lag for screen mode)" if args.mode == "screen" else ""))
    finally:
        if link:
            link.send_led(False)
            link.stop(send_ready=False)
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
