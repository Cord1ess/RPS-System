"""
Camera probe: measures what the webcam really delivers before any data is recorded.

Run it in the actual play lighting. For each backend it measures auto exposure and several
locked exposures, reporting mean fps (dropped frames count against it), frame-interval jitter,
brightness and mains-flicker ripple, then recommends the best usable combination.

Findings on the demo laptop's "FHD Camera" (2026-09-27, dim room at night):
- The camera has no gain control; locking exposure also pins gain at its minimum, so locked
  exposure needs a well-lit play zone (a lamp) or frames come out black.
- Auto exposure in dim light lengthens frames: DirectShow drops to ~16 fps.

The driver is returned to automatic exposure when the probe finishes.

Usage:
    python tools/camera_probe.py
    python tools/camera_probe.py --write-config      # store the recommended mode in config.json
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rps.config import load_config, save_config  # noqa: E402

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
LOCKED_EXPOSURES = (-5, -6, -7)
MIN_FPS = 28.0
MIN_BRIGHTNESS = 60.0


def open_mode(index, backend, width=640, height=480, fps=30, fourcc="YUY2"):
    cap = cv2.VideoCapture(index, BACKENDS[backend])
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def measure(cap, seconds=2.5, warmup=1.0):
    """Returns dict with mean fps, p90 interval, brightness and flicker ripple."""
    t_end = time.perf_counter() + warmup
    while time.perf_counter() < t_end:
        cap.read()
    stamps, means = [], []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        ok, img = cap.read()
        if ok:
            stamps.append(time.perf_counter())
            means.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()))
    if len(stamps) < 3:
        return {"fps": 0.0, "p90_ms": float("nan"), "brightness": 0.0, "ripple": float("nan")}
    dt = np.diff(stamps) * 1000.0
    return {
        "fps": 1000.0 / float(np.mean(dt)),
        "p90_ms": float(np.percentile(dt, 90)),
        "brightness": float(np.mean(means)),
        "ripple": flicker_ripple(np.array(means)),
    }


def flicker_ripple(means: np.ndarray) -> float:
    """Frame-mean ripple (% of mean) after removing slow drift. >1% suggests mains flicker."""
    if len(means) < 10 or np.mean(means) < 5:
        return float("nan")
    trend = np.convolve(np.pad(means, 4, mode="edge"), np.ones(9) / 9.0, mode="valid")
    return float(np.std(means - trend) / np.mean(means) * 100.0)


def usable(m) -> bool:
    return m["fps"] >= MIN_FPS and m["brightness"] >= MIN_BRIGHTNESS


def main():
    parser = argparse.ArgumentParser(description="Probe webcam modes, exposure locks and lighting")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=2.5)
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = cfg.camera.index if args.index is None else args.index

    print("=" * 86)
    print("  CAMERA PROBE  (YUY2 640x480; keep the play zone lit exactly as it will be during play)")
    print("=" * 86)
    rows = []  # (backend, locked, exposure, metrics)
    for backend in ("dshow", "msmf"):
        cap = open_mode(index, backend)
        if cap is None:
            print(f"  {backend}: cannot open camera")
            continue
        try:
            m = measure(cap, args.seconds)
            rows.append((backend, False, None, m))
            print(f"  {backend:5s} auto exposure     : {m['fps']:5.1f} fps (p90 dt {m['p90_ms']:5.1f} ms), "
                  f"brightness {m['brightness']:6.1f}")
            if backend == "dshow":   # MSMF ignores exposure writes on this driver
                for exp in LOCKED_EXPOSURES:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
                    m = measure(cap, args.seconds)
                    rows.append((backend, True, exp, m))
                    print(f"  {backend:5s} locked {exp:+d} ({1000 * 2.0 ** exp:4.1f} ms): {m['fps']:5.1f} fps "
                          f"(p90 dt {m['p90_ms']:5.1f} ms), brightness {m['brightness']:6.1f}, "
                          f"flicker ripple {m['ripple']:4.2f}%")
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        finally:
            cap.release()

    # Preference: locked DirectShow (clean 30 fps, stable image) at the shortest usable exposure,
    # then the fastest usable auto mode, then whatever is fastest (with a lighting warning).
    locked_ok = [r for r in rows if r[1] and usable(r[3]) and not (r[3]["ripple"] > 1.0)]
    auto_ok = [r for r in rows if not r[1] and usable(r[3])]
    if locked_ok:
        best = min(locked_ok, key=lambda r: r[2])
    elif auto_ok:
        best = max(auto_ok, key=lambda r: r[3]["fps"])
    else:
        best = max((r for r in rows if r[3]["brightness"] >= MIN_BRIGHTNESS), key=lambda r: r[3]["fps"],
                   default=max(rows, key=lambda r: r[3]["fps"]) if rows else None)
    if best is None:
        print("\n[!] Camera could not be opened on any backend.")
        return

    backend, locked, exp, m = best
    print("\n" + "-" * 86)
    print(f"[*] Recommended: backend={backend}, exposure={'locked ' + str(exp) if locked else 'auto'} "
          f"-> {m['fps']:.1f} fps, brightness {m['brightness']:.0f}")
    if not usable(m):
        print("[!] Not good enough for play: needs >= 28 fps at brightness >= 60.")
        print("    The camera has no gain control, so the fix is light: put a lamp on the play zone")
        print("    (flicker-free LED preferred) and rerun this probe.")
    elif not locked:
        print("    Auto exposure is usable; the DVS emulator's global-gain correction absorbs slow AE drift.")
        print("    More light on the play zone would allow a locked exposure (cleaner events, less blur).")

    if args.write_config:
        cfg.camera.backend = backend
        cfg.camera.fourcc = "YUY2"
        cfg.camera.width, cfg.camera.height = 640, 480
        cfg.camera.lock_exposure = bool(locked)
        cfg.camera.lock_white_balance = bool(locked)
        if locked:
            cfg.camera.exposure = float(exp)
        save_config(cfg, args.config)
    print("[camera_probe] Driver left in automatic exposure mode.")


if __name__ == "__main__":
    main()
