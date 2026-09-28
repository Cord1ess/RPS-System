"""
Builds pseudo-DVS training frames from raw recordings (replaces clean_dataset.py).

For every recording in data/recordings/<person>/<session>/ and every (event count N, frame skip)
variant, the ROI video is replayed through PseudoDVS exactly as at runtime; each emitted 64x64
frame is labelled from the webcam frame at which it was emitted:

    show        session label; frames where MediaPipe CONFIDENTLY sees a different gesture are
                dropped (e.g. a fist while re-gripping during a paper session)
    background  label 3 (background)
    throws      MediaPipe per-frame label with look-ahead: confident frames keep their gesture
                (pump fists are genuinely "rock"); unsure/blurred frames take the next confident
                gesture within --lookahead seconds (this is what teaches early commits);
                otherwise excluded

Multi-N generation (ROSHAMBO17 used 0.5k/1k/2k events) uses real binomial thinning in the
emulator, so it is a genuine sparsity augmentation. Frame skip 2 simulates an fps drop.

Output: data/frames/<person>/<session>__N<N>_s<skip>.npz  (frames, labels, t, flushed, n_events)
        data/frames/index.json

Usage:
    python build_dataset.py
    python build_dataset.py --event_counts 750,1500,3000 --frame_skips 1,2 --no_mp
"""

import argparse
import copy
import glob
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from model import CLASS_NAMES, SYMBOL_TO_LABEL
from rps.camera import VideoFileSource, crop_roi
from rps.config import load_config
from rps.dvs_emulator import PseudoDVS
from rps.hand_tracker import tracker_fingerprint
from rps.perf import boost_process

EXCLUDE = -1
MIN_HAND_VISIBLE = 0.6      # warn when Mediapipe finds the hand in fewer frames than this


def find_recordings(root: str) -> List[str]:
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "meta.json"), recursive=True))


def mediapipe_track(rec_dir: str, hand_cfg) -> Dict[str, np.ndarray]:
    """Per-webcam-frame Mediapipe readings, cached in the recording folder."""
    cache = os.path.join(rec_dir, "mp_cache.npz")
    fp = tracker_fingerprint(hand_cfg)
    if os.path.exists(cache):
        data = np.load(cache)
        if str(data["fingerprint"]) == fp:
            return {k: data[k] for k in ("present", "gesture", "confidence", "wrist_y")}
    from rps.hand_tracker import HandTracker
    tracker = HandTracker(hand_cfg)
    src = VideoFileSource(rec_dir)
    present, gesture, conf, wrist = [], [], [], []
    while True:
        frame = src.read()
        if frame is None:
            break
        obs = tracker.process(frame.bgr, src.roi, frame.t, allow_skip=False)
        present.append(obs.present)
        gesture.append(obs.gesture)
        conf.append(obs.confidence)
        wrist.append(obs.wrist_y if obs.wrist_y is not None else np.nan)
    src.stop()
    tracker.close()
    out = {"present": np.array(present, bool), "gesture": np.array(gesture, np.int8),
           "confidence": np.array(conf, np.float32), "wrist_y": np.array(wrist, np.float32)}
    np.savez_compressed(cache, fingerprint=fp, **out)
    return out


def frame_labels(meta: Dict, times: np.ndarray, mp: Dict[str, np.ndarray], clean_conf: float,
                 lookahead_s: float) -> np.ndarray:
    """Label per webcam frame (EXCLUDE = do not train on it)."""
    n = len(times)
    kind = meta["type"]
    session_label = SYMBOL_TO_LABEL[meta["label"]]
    if kind == "background":
        return np.full(n, SYMBOL_TO_LABEL["background"], np.int8)
    if kind == "show":
        labels = np.full(n, session_label, np.int8)
        if mp is not None:
            contradict = (mp["confidence"] >= clean_conf) & (mp["gesture"] >= 0) & \
                         (mp["gesture"] <= 2) & (mp["gesture"] != session_label)
            labels[contradict] = EXCLUDE
        return labels
    # throws
    if mp is None:
        return np.full(n, session_label, np.int8)
    confident = (mp["confidence"] >= clean_conf) & (mp["gesture"] >= 0) & (mp["gesture"] <= 2)
    labels = np.full(n, EXCLUDE, np.int8)
    labels[confident] = mp["gesture"][confident]
    next_label, next_t = EXCLUDE, np.inf
    for i in range(n - 1, -1, -1):     # look-ahead fill for unsure / blurred frames
        if confident[i]:
            next_label, next_t = mp["gesture"][i], times[i]
        elif next_label != EXCLUDE and next_t - times[i] <= lookahead_s:
            labels[i] = next_label
    return labels


def build_variants(rec_dir: str, dvs_cfgs: Dict, labels: np.ndarray, seeds: Dict) -> Dict:
    """
    One decoding pass feeding every (N, skip) emulator variant. Returns
    {(N, skip): {"frames", "labels", "t", "flushed", "n_events"}}.
    """
    src = VideoFileSource(rec_dir)
    emulators = {key: PseudoDVS(c, seed=seeds[key]) for key, c in dvs_cfgs.items()}
    acc = {key: {"frames": [], "labels": [], "t": [], "flushed": [], "n_events": []} for key in dvs_cfgs}
    idx = 0
    while True:
        frame = src.read()
        if frame is None:
            break
        sensor = None
        for (n, skip), dvs in emulators.items():
            if idx % skip:
                continue
            if sensor is None:
                sensor = dvs.to_sensor(crop_roi(frame.bgr, src.roi))   # shared 128x128 gray
            emitted, _ = dvs.process(sensor, frame.t)
            if emitted is not None and labels[idx] != EXCLUDE:
                a = acc[(n, skip)]
                a["frames"].append(emitted.image)
                a["labels"].append(labels[idx])
                a["t"].append(frame.t)
                a["flushed"].append(emitted.flushed)
                a["n_events"].append(emitted.n_events)
        idx += 1
    src.stop()
    out = {}
    for key, a in acc.items():
        fs = dvs_cfgs[key].frame_size
        out[key] = {
            "frames": np.array(a["frames"], np.uint8).reshape(-1, fs, fs),
            "labels": np.array(a["labels"], np.int8),
            "t": np.array(a["t"], np.float32),
            "flushed": np.array(a["flushed"], bool),
            "n_events": np.array(a["n_events"], np.int32),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Recordings -> pseudo-DVS training frames")
    parser.add_argument("--recordings", default="data/recordings")
    parser.add_argument("--out", default="data/frames")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--event_counts", default="750,1500,3000")
    parser.add_argument("--frame_skips", default="1,2")
    parser.add_argument("--no_mp", action="store_true", help="Skip MediaPipe label cleaning/look-ahead")
    parser.add_argument("--clean_conf", type=float, default=0.75)
    parser.add_argument("--lookahead", type=float, default=0.30)
    args = parser.parse_args()

    boost_process()
    cfg = load_config(args.config)
    event_counts = [int(x) for x in args.event_counts.split(",")]
    skips = [int(x) for x in args.frame_skips.split(",")]
    recordings = find_recordings(args.recordings)
    if not recordings:
        raise SystemExit(f"No recordings found under {args.recordings}. Use record_session.py first.")

    os.makedirs(args.out, exist_ok=True)
    index = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dvs": cfg.to_dict()["dvs"],
             "event_counts": event_counts, "frame_skips": skips, "files": []}
    totals = defaultdict(Counter)
    warnings = []
    t0 = time.time()
    for r_i, rec_dir in enumerate(recordings):
        with open(os.path.join(rec_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        name = f"{meta['person']}/{meta['session_id']}"
        if args.no_mp and meta["type"] == "throws":
            # Without Mediapipe the pumping fists would all be labelled with the thrown gesture.
            warnings.append(f"{name}: skipped; countdown throws need Mediapipe to tell pumps from the throw")
            continue
        src = VideoFileSource(rec_dir)
        times = np.array(src.times, np.float64)
        src.stop()
        mp = None
        if not args.no_mp and meta["type"] in ("show", "throws"):
            mp = mediapipe_track(rec_dir, cfg.hand)
            seen = float(np.mean(mp["present"])) if len(mp["present"]) else 0.0
            if seen < MIN_HAND_VISIBLE:
                warnings.append(f"{name}: the hand was found in only {seen:.0%} of frames (bad light or framing); "
                                f"most of this session is left out")
        labels = frame_labels(meta, times, mp, args.clean_conf, args.lookahead)
        excluded = int((labels == EXCLUDE).sum())
        if len(times) and excluded > 0.5 * len(times):
            warnings.append(f"{name}: {excluded} of {len(times)} frames left out by the label check")
        print(f"[build] ({r_i + 1}/{len(recordings)}) {name}: "
              f"{len(times)} webcam frames, {excluded} excluded by labelling")

        person_dir = os.path.join(args.out, meta["person"])
        os.makedirs(person_dir, exist_ok=True)
        dvs_cfgs, seeds = {}, {}
        for n in event_counts:
            for skip in skips:
                dvs_cfgs[(n, skip)] = copy.deepcopy(cfg.dvs)
                dvs_cfgs[(n, skip)].event_count = n
                seeds[(n, skip)] = int(hashlib.md5(f"{meta['session_id']}|{n}|{skip}".encode()).hexdigest()[:8], 16)
        variants = build_variants(rec_dir, dvs_cfgs, labels, seeds)
        for (n, skip), data in variants.items():
            name = f"{meta['session_id']}__N{n}_s{skip}.npz"
            path = os.path.join(person_dir, name)
            np.savez_compressed(path, **data)
            counts = Counter(int(x) for x in data["labels"])
            for c, k in counts.items():
                totals[meta["person"]][c] += k
            index["files"].append({"path": os.path.relpath(path, args.out).replace("\\", "/"),
                                   "person": meta["person"], "session": meta["session_id"],
                                   "type": meta["type"], "label": meta["label"], "event_count": n,
                                   "frame_skip": skip, "frames": int(len(data["labels"])),
                                   "flushed": int(data["flushed"].sum())})
            print(f"          N={n:5d} skip={skip}: {len(data['labels']):5d} frames "
                  f"({int(data['flushed'].sum())} flushed) {dict(sorted(counts.items()))}")

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\n[build] Done in {time.time() - t0:.0f}s. Frames per person and class:")
    for person, counts in sorted(totals.items()):
        row = "  ".join(f"{CLASS_NAMES[c]}={counts.get(c, 0)}" for c in range(4))
        print(f"  {person:12s} {row}")
    for w in warnings:
        print(f"[build] WARNING: {w}")


if __name__ == "__main__":
    main()
