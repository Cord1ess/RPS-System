"""
Measures how well a pretrained CNN (default: Dextra's) recognises OUR camera's gestures, and finds
the input settings under which it transfers best. No training involved.

Record a few labelled sessions first (Record tab or record_session.py): about 30 s "show" session
per gesture while moving the hand, plus a background session. Every recording is replayed
through the pseudo-DVS emulator for each combination of:
    contrast threshold C x event count N x hand zoom (central crop of the play zone)
and every emitted frame is classified under all 8 rotations/flips. The report ranks settings by
balanced accuracy over the gestures, shows where mistakes go, and writes contact sheets that put
OUR frames next to Dextra's own frames for each class.

Usage:
    python tools/dextra_transfer.py --recordings data/recordings/alice
    python tools/dextra_transfer.py --recordings data/recordings --event_counts 1500,3000 --zooms 1.0,0.8
"""

import argparse
import copy
import glob
import itertools
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CLASS_NAMES, SYMBOL_TO_LABEL  # noqa: E402
from rps.camera import VideoFileSource  # noqa: E402
from rps.cnn import GestureCNN, orient_batch, predict_batch  # noqa: E402
from rps.config import load_config  # noqa: E402
from rps.dvs_emulator import PseudoDVS  # noqa: E402
from rps.perf import boost_process  # noqa: E402

NAMES = [c.split("_", 1)[1] for c in CLASS_NAMES]
ORIENTS = [(r, f) for f in (False, True) for r in (0, 90, 180, 270)]


def floats(text):
    return [float(v) for v in text.split(",") if v]


def load_sensor_frames(rec_dir, zoom, sensor_size):
    """ROI crops (optionally zoomed to the centre) resized to the emulator's sensor grid."""
    src = VideoFileSource(rec_dir)
    out, times = [], []
    while True:
        frame = src.read()
        if frame is None:
            break
        x, y, s = src.roi
        z = int(round(s * zoom))
        off = (s - z) // 2
        crop = frame.bgr[y + off:y + off + z, x + off:x + off + z]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(gray, (sensor_size, sensor_size), interpolation=cv2.INTER_AREA))
        times.append(frame.t)
    src.stop()
    return out, times


def contact_sheet(ours, theirs, title, path, n=24):
    def tiles(frames, k):
        if len(frames) == 0:
            return []
        pick = np.random.default_rng(0).choice(len(frames), min(k, len(frames)), replace=False)
        return [cv2.applyColorMap(cv2.resize(frames[i], (96, 96), interpolation=cv2.INTER_NEAREST),
                                  cv2.COLORMAP_INFERNO) for i in pick]

    def row(ts, label, cols=12):
        ts = ts + [np.zeros((96, 96, 3), np.uint8)] * (-len(ts) % cols)
        ts = [cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60)) for t in ts]
        rows = [np.hstack(ts[i:i + cols]) for i in range(0, len(ts), cols)] if ts else []
        bar = np.full((24, 100 * cols, 3), 30, np.uint8)
        cv2.putText(bar, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return [bar] + rows

    parts = row(tiles(ours, n), f"{title}: OUR camera (as the CNN sees it)") + \
        row(tiles(theirs, 12), f"{title}: Dextra's own frames")
    cv2.imwrite(path, np.vstack(parts))


def main():
    parser = argparse.ArgumentParser(description="Transfer test of a pretrained CNN on our recordings")
    parser.add_argument("--recordings", default="data/recordings")
    parser.add_argument("--model", default="models/dextra_roshambo.pth")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--contrasts", default="0.15,0.2,0.3")
    parser.add_argument("--event_counts", default="750,1500,3000,5000")
    parser.add_argument("--zooms", default="1.0,0.8,0.65")
    parser.add_argument("--dextra_samples", default="models/dextra/sample_frames")
    parser.add_argument("--out", default="data/analysis/dextra_transfer")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    boost_process()
    cfg = load_config(args.config)
    cnn = GestureCNN(args.model)
    recs = []
    for meta_path in sorted(glob.glob(os.path.join(args.recordings, "**", "meta.json"), recursive=True)):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["type"] in ("show", "background"):
            recs.append((os.path.dirname(meta_path), SYMBOL_TO_LABEL[meta["label"]], meta))
    if not recs:
        raise SystemExit("No 'show' or 'background' recordings found. Record ~30 s per gesture (moving the "
                         "hand) plus a background session in the Record tab, then rerun.")
    present = sorted({lab for _, lab, _ in recs})
    print(f"[transfer] {len(recs)} recordings; classes present: {[NAMES[c] for c in present]}")
    if not all(c in present for c in (0, 1, 2)):
        print("[transfer] NOTE: record all three gestures for a complete comparison.")

    contrasts, counts, zooms = floats(args.contrasts), [int(v) for v in floats(args.event_counts)], floats(args.zooms)
    results = []
    frames_by_setting = {}
    for zoom in zooms:
        sensor = [(load_sensor_frames(d, zoom, cfg.dvs.sensor_size), lab) for d, lab, _ in recs]
        for c, n in itertools.product(contrasts, counts):
            dvs_cfg = copy.deepcopy(cfg.dvs)
            dvs_cfg.contrast_threshold, dvs_cfg.event_count = c, n
            frames, labels = [], []
            for (seq, times), lab in sensor:
                dvs = PseudoDVS(dvs_cfg, seed=1)
                for img, t in zip(seq, times):
                    emitted, _ = dvs.process(img, t)
                    if emitted is not None:
                        frames.append(emitted.image)
                        labels.append(lab)
            if not frames:
                continue
            frames, labels = np.stack(frames), np.array(labels)
            frames_by_setting[(c, n, zoom)] = (frames, labels)
            for rot, flip in ORIENTS:
                pred = predict_batch(cnn, frames, rot, flip).argmax(1)
                acc = {k: float(np.mean(pred[labels == k] == k)) for k in present}
                gestures = [acc[k] for k in (0, 1, 2) if k in acc]
                conf = np.zeros((4, 4), int)
                np.add.at(conf, (labels, pred), 1)
                results.append({"C": c, "N": n, "zoom": zoom, "rotate": rot, "flip": flip, "acc": acc,
                                "balanced": float(np.mean(gestures)) if gestures else 0.0,
                                "frames": len(labels), "confusion": conf})
            print(f"[transfer] zoom {zoom:.2f} C {c:.2f} N {n:5d}: {len(labels)} DVS frames")

    results.sort(key=lambda r: r["balanced"], reverse=True)
    print("\n" + "=" * 100)
    print(f"{'rank':>4} {'zoom':>5} {'C':>5} {'N':>5} {'rot':>4} {'flip':>5} | " +
          "".join(f"{NAMES[k]:>11}" for k in present) + f" | {'balanced':>9} {'frames':>7}")
    print("-" * 100)
    for i, r in enumerate(results[:args.top]):
        print(f"{i + 1:4d} {r['zoom']:5.2f} {r['C']:5.2f} {r['N']:5d} {r['rotate']:4d} {str(r['flip']):>5} | " +
              "".join(f"{r['acc'][k] * 100:10.0f}%" for k in present) + f" | {r['balanced'] * 100:8.1f}% {r['frames']:7d}")

    current = [r for r in results if r["C"] == cfg.dvs.contrast_threshold and r["N"] == cfg.dvs.event_count
               and r["zoom"] == 1.0 and r["rotate"] == cfg.cnn.rotate and r["flip"] == cfg.cnn.flip]
    best = results[0]
    if current:
        c0 = current[0]
        print(f"\nCurrent settings (zoom 1.00, C {c0['C']}, N {c0['N']}, rotate {c0['rotate']}, flip {c0['flip']}): "
              f"balanced {c0['balanced'] * 100:.1f}% ({', '.join(f'{NAMES[k]} {v * 100:.0f}%' for k, v in c0['acc'].items())})")
    print(f"\nBest: zoom {best['zoom']:.2f}, C {best['C']}, N {best['N']}, rotate {best['rotate']}, flip {best['flip']}"
          f" -> balanced {best['balanced'] * 100:.1f}%")
    print("Confusion matrix at best (rows = what you showed, cols = what the CNN said):")
    print(f"{'':>12}" + "".join(f"{NAMES[k]:>11}" for k in range(4)))
    for k in present:
        print(f"{NAMES[k]:>12}" + "".join(f"{v:11d}" for v in best["confusion"][k]))
    zoom_note = "" if best["zoom"] == 1.0 else \
        f"  and shrink the play zone to about {best['zoom'] * 100:.0f}% of its size around the hand (Setup tab)"
    print(f"\nTo use it live: rotate {best['rotate']}, flip {best['flip']}, dvs.event_count {best['N']}, "
          f"dvs.contrast_threshold {best['C']}{zoom_note}.")
    # machine-readable line for the desktop app (Evaluate tab)
    print("@@BEST " + json.dumps({"rotate": best["rotate"], "flip": best["flip"], "event_count": best["N"],
                                  "contrast_threshold": best["C"], "zoom": best["zoom"],
                                  "balanced": best["balanced"],
                                  "per_gesture": {NAMES[k]: v for k, v in best["acc"].items()},
                                  "current_balanced": current[0]["balanced"] if current else None}), flush=True)

    os.makedirs(args.out, exist_ok=True)
    theirs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.dextra_samples, "*.png"))):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        theirs[cnn.predict(img)[0]].append(img)       # samples are already in Dextra's orientation
    frames, labels = frames_by_setting[(best["C"], best["N"], best["zoom"])]
    oriented = orient_batch(frames, best["rotate"], best["flip"])
    for k in present:
        path = os.path.join(args.out, f"{NAMES[k]}_ours_vs_dextra.png")
        contact_sheet(oriented[labels == k], np.array(theirs.get(k, [])), NAMES[k], path)
        print(f"[transfer] sheet: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
