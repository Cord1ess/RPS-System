"""
Speed benchmark of the running part (no training): how long each stage takes per camera frame on
this computer, for each recognizer, on a recording. Run it on the laptop and on the Raspberry Pi
with the same recording to compare them.

The pipeline runs as fast as it can (not in real time) and Mediapipe runs on every frame, so its
real cost is measured. A camera sends a frame every 33 ms (30 fps): a stage total above that means
frames are dropped live (Mediapipe then skips frames and decisions come a frame or two later).

Usage:
    python tools/benchmark.py                                  # the first recording found, 600 frames
    python tools/benchmark.py --recording data/recordings/<person>/<session> --frames 900
    python tools/benchmark.py --recognizers both,dextra_tuned --json results.json
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rps.camera import VideoFileSource  # noqa: E402
from rps.config import load_config  # noqa: E402
from rps.perf import boost_process  # noqa: E402
from rps.pipeline import RECOGNIZERS, Pipeline, load_models, missing_reason  # noqa: E402

FRAME_BUDGET_MS = 1000.0 / 30.0
WARMUP_FRAMES = 30


def machine() -> dict:
    """What this computer is and whether it is running at full speed."""
    info = {"system": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "python": platform.python_version(), "cores": os.cpu_count(), "cpu": platform.processor() or ""}
    model = "/proc/device-tree/model"
    if os.path.exists(model):                                   # Raspberry Pi
        with open(model, "rb") as f:
            info["cpu"] = f.read().decode(errors="replace").strip("\x00 \n")
    governor = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(governor):
        with open(governor) as f:
            info["cpu_governor"] = f.read().strip()
    try:                                                        # Pi: under-voltage or heat throttling
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        info["pi_throttled"] = out.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class PowerStatus(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                            ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                            ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
            status = PowerStatus()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                info["on_battery"] = status.ACLineStatus == 0
        except (OSError, AttributeError):
            pass
    return info


def find_recording() -> str:
    found = sorted(glob.glob("data/recordings/*/*_throws_*")) or sorted(glob.glob("data/recordings/*/*"))
    if not found:
        raise SystemExit("No recording found: record one on the Record page, or pass --recording.")
    return found[0]


def stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"median": None, "p90": None, "n": 0}
    return {"median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2), "n": int(len(v))}


def run(cfg, recording: str, recognizer: str, max_frames: int) -> dict:
    cnn, hand, messages = load_models(cfg, recognizer)
    pipeline = Pipeline(cfg, cnn, hand, allow_mp_skip=False)
    src = VideoFileSource(recording)
    rows = []
    try:
        while len(rows) < max_frames + WARMUP_FRAMES:
            frame = src.read()
            if frame is None:
                if not rows:
                    break
                src.stop()
                src = VideoFileSource(recording)                # loop the recording until enough frames
                pipeline = Pipeline(cfg, cnn, hand, allow_mp_skip=False)
                continue
            t0 = time.perf_counter()
            r = pipeline.step(frame, src.roi)
            rows.append({"dvs": r.dvs_ms, "cnn": r.cnn_ms if r.cnn is not None else None,
                         "mp": r.mp_ms if r.hand is not None else None,
                         "total": (time.perf_counter() - t0) * 1000.0})
    finally:
        src.stop()
        if hand is not None:
            hand.close()
    rows = rows[WARMUP_FRAMES:]
    total = stats(r["total"] for r in rows)
    return {"recognizer": recognizer, "frames": len(rows), "messages": messages,
            "dvs_ms": stats(r["dvs"] for r in rows), "dextra_ms": stats(r["cnn"] for r in rows),
            "mediapipe_ms": stats(r["mp"] for r in rows), "total_ms": total,
            "keeps_up_at_30fps": total["p90"] is not None and total["p90"] <= FRAME_BUDGET_MS}


def main():
    parser = argparse.ArgumentParser(description="Per-stage speed of the running pipeline on this computer")
    parser.add_argument("--recording", default=None)
    parser.add_argument("--recognizers", default="dextra_tuned,mediapipe,both")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--json", default=None, help="Also write the results here")
    args = parser.parse_args()
    boost_process()                         # as the app does: fast cores on Windows laptops (no-op elsewhere)
    cfg = load_config(args.config)
    recording = args.recording or find_recording()
    info = machine()
    print("=" * 78)
    print(f"  Benchmark on {info['cpu'] or info['machine']} ({info['cores']} cores, {info['system']}, "
          f"Python {info['python']})")
    if info.get("on_battery"):
        print("  [!] On battery: Windows slows the CPU. Plug in for the real laptop numbers.")
    if info.get("cpu_governor") not in (None, "performance", "ondemand", "schedutil"):
        print(f"  [!] CPU governor '{info['cpu_governor']}' may slow the CPU.")
    if info.get("pi_throttled") and not info["pi_throttled"].endswith("=0x0"):
        print(f"  [!] Pi throttling reported ({info['pi_throttled']}): check the power supply and cooling.")
    print(f"  Recording: {recording}")
    print("=" * 78)
    results = []
    for recognizer in [r.strip() for r in args.recognizers.split(",") if r.strip()]:
        why = missing_reason(cfg, recognizer) if recognizer != "both" else None
        if recognizer not in RECOGNIZERS or why:
            print(f"  {recognizer}: skipped ({why or 'unknown recognizer'})")
            continue
        res = run(cfg, recording, recognizer, args.frames)
        results.append(res)
        print(f"\n  {RECOGNIZERS[recognizer]}  ({res['frames']} frames)")
        for name, key in (("Dextra view", "dvs_ms"), ("Dextra", "dextra_ms"), ("Mediapipe", "mediapipe_ms"),
                          ("Total per frame", "total_ms")):
            s = res[key]
            if s["n"]:
                print(f"    {name:16s} median {s['median']:6.1f} ms   p90 {s['p90']:6.1f} ms")
        verdict = "keeps up with 30 fps" if res["keeps_up_at_30fps"] else \
            "slower than 30 fps: Mediapipe will skip frames live"
        print(f"    -> {verdict} (budget {FRAME_BUDGET_MS:.0f} ms per frame)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"machine": info, "recording": recording, "results": results}, f, indent=2)
        print(f"\n  Saved {args.json}")


if __name__ == "__main__":
    main()
