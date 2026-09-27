"""
Offline evaluation of the full live pipeline (rps.pipeline.Pipeline) on recorded sessions.

Each recording is replayed with its recorded timestamps through exactly the code play.py runs.
Sessions are scored by type:
    show        continuous mode: hold accuracy (fraction of time the committed gesture is the
                session label), switches per minute (should be ~0), per-frame CNN/MediaPipe accuracy
    throws      countdown mode: throw accuracy (commits == label), commits vs planned throws,
                commit time relative to the hand coming to rest (negative = anticipated), and the
                projected time the robot pose is visible = that + camera latency + servo transition
    background  continuous mode: false commits per minute

Sources compared: mediapipe, cnn, fused. The report ends with the go/no-go rule from the plan:
if fused does not beat MediaPipe-only on held-out people, ship MediaPipe-only.

Usage:
    python replay_eval.py --recordings data/recordings/bob          # held-out person
    python replay_eval.py --recordings data/recordings --sources mediapipe
    python replay_eval.py --recordings data/recordings/bob --sweep vote.k=1,2,3
"""

import argparse
import copy
import csv
import glob
import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from model import SYMBOL_TO_LABEL
from rps.camera import VideoFileSource
from rps.config import apply_overrides, load_config
from rps.hand_tracker import PAPER, SCISSORS
from rps.perf import boost_process
from rps.pipeline import Pipeline

LAND_BEFORE_S, LAND_AFTER_S = 1.5, 0.4     # where to look for the throw's landing around a decision
LAND_PEAK_FRAMES = 3                      # a wrist low point must be the lowest within this many frames


class ReplayHand:
    """Serves MediaPipe results computed once per recording (keyed by frame timestamp)."""

    def __init__(self, observations: Dict[float, object]):
        self.obs = observations
        self.skipped_frames = 0

    def process(self, frame_bgr, roi, t, allow_skip=True):
        return self.obs[t]

    def close(self):
        pass


def find_recordings(root: str) -> List[str]:
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "meta.json"), recursive=True))


def track_hand(rec_dir: str, cfg) -> ReplayHand:
    from rps.hand_tracker import HandTracker
    tracker = HandTracker(cfg.hand)
    src = VideoFileSource(rec_dir)
    obs = {}
    while True:
        frame = src.read()
        if frame is None:
            break
        obs[frame.t] = tracker.process(frame.bgr, src.roi, frame.t, allow_skip=False)
    src.stop()
    tracker.close()
    return ReplayHand(obs)


def replay(rec_dir: str, meta: Dict, cfg, cnn, hand: Optional[ReplayHand]) -> Dict:
    cfg = copy.deepcopy(cfg)
    cfg.decision.mode = "countdown" if meta["type"] == "throws" else "continuous"
    pipeline = Pipeline(cfg, cnn, hand, pose_sink=None, allow_mp_skip=False)
    src = VideoFileSource(rec_dir)
    rows, commits = [], []
    last_commits = 0
    while True:
        frame = src.read()
        if frame is None:
            break
        r = pipeline.step(frame, src.roi)
        s = r.snapshot
        rows.append({"t": frame.t, "events": r.dvs_stats.events, "human": s.human, "pose": s.pose,
                     "state": s.state, "pumps": s.pumps, "vy": r.vy,
                     "cnn": r.cnn[0] if r.cnn is not None else None,
                     "mp": r.hand.gesture if r.hand is not None and r.hand.present else None,
                     "mp_conf": r.hand.confidence if r.hand is not None else 0.0,
                     "wrist_y": r.hand.wrist_y if r.hand is not None and r.hand.present else None})
        if s.commits > last_commits:
            commits.append({"t": frame.t, "gesture": s.human, "source": s.source, "pose": s.pose})
            last_commits = s.commits
    src.stop()
    return {"rows": rows, "commits": commits, "switches": pipeline.engine.switches}


def last_pump_index(rows: List[Dict], ci: int) -> int:
    """Frame where the round's last pump was counted, for the decision at frame ci."""
    t_c, j = rows[ci]["t"], ci
    while j > 0 and rows[j - 1].get("pumps") == rows[ci].get("pumps") and t_c - rows[j - 1]["t"] <= LAND_BEFORE_S:
        j -= 1
    return j


def landing_time(rows: List[Dict], ci: int) -> Optional[float]:
    """
    End of the throw's down stroke, for the decision at frame ci: the first low point of the
    tracked wrist (row "ref_wrist") after the round's last pump, else the play-zone motion turning
    from down to up. Players often go straight into the next round, so waiting for the hand to come
    to rest would miss most throws; and the throw does not always land lower than the pumps.
    """
    t_c = rows[ci]["t"]
    idx = [i for i in range(last_pump_index(rows, ci) + 1, len(rows)) if rows[i]["t"] <= t_c + LAND_AFTER_S]
    if not idx:
        return None
    k = LAND_PEAK_FRAMES
    if sum(rows[i].get("ref_wrist") is not None for i in idx) >= 0.5 * len(idx):
        for i in idx:
            y = rows[i]["ref_wrist"]
            if y is None or i + k >= len(rows):
                continue
            near = [r["ref_wrist"] for r in rows[i - k:i + k + 1] if r.get("ref_wrist") is not None]
            if y >= max(near) and y > min(near):                   # a real low point (image y grows downward)
                return rows[i]["t"]
        return None
    for i in idx[:-1]:
        if rows[i]["vy"] is not None and rows[i]["vy"] > 0 and (rows[i + 1]["vy"] or 0.0) <= 0:
            return rows[i + 1]["t"]
    return None


def throw_time(rows: List[Dict], ci: int, label: int) -> Optional[float]:
    """
    When the human's throw became visible, for the decision at frame ci. Paper/scissors: the first
    frame after the round's last pump in which the hand tracker (row "ref_gesture", the same for every
    method being scored) shows that gesture. Rock looks like the pumping fist, so for rock (or when
    the shape was never seen) it is the end of the throw's down stroke.
    """
    if label in (PAPER, SCISSORS):
        end = rows[ci]["t"] + LAND_AFTER_S
        for i in range(last_pump_index(rows, ci) + 1, len(rows)):
            if rows[i]["t"] > end:
                break
            if rows[i].get("ref_gesture") == label:
                return rows[i]["t"]
    return landing_time(rows, ci)


def add_reference(rows: List[Dict], ref: Optional["ReplayHand"]):
    """Adds the hand tracker's view of each frame, used as the common timing reference."""
    for r in rows:
        o = ref.obs.get(r["t"]) if ref is not None else None
        r["ref_gesture"] = o.gesture if o is not None and o.present else None
        r["ref_wrist"] = o.wrist_y if o is not None and o.present else None


def score(meta: Dict, res: Dict, cfg) -> Dict:
    rows, commits = res["rows"], res["commits"]
    label = SYMBOL_TO_LABEL[meta["label"]]
    duration = max(rows[-1]["t"] - rows[0]["t"], 1e-6) if rows else 1e-6
    out = {"session": meta["session_id"], "person": meta["person"], "type": meta["type"],
           "label": meta["label"], "duration_s": round(duration, 1), "commits": len(commits)}
    cnn = [r["cnn"] for r in rows if r["cnn"] is not None]
    if cnn:
        out["cnn_frame_acc"] = float(np.mean(np.array(cnn) == label))

    if meta["type"] == "show":
        mp = [r["mp"] for r in rows if r["mp"] is not None and r["mp_conf"] >= cfg.decision.mp_min_confidence]
        if mp:
            out["mp_frame_acc"] = float(np.mean(np.array(mp) == label))
        if commits:
            t0 = commits[0]["t"]
            held = [r["human"] == label for r in rows if r["t"] >= t0]
            out["hold_acc"] = float(np.mean(held))
            out["first_commit_s"] = round(t0 - rows[0]["t"], 2)
        else:
            out["hold_acc"] = 0.0
        out["switches_per_min"] = res["switches"] / duration * 60.0
    elif meta["type"] == "background":
        out["false_commits_per_min"] = len(commits) / duration * 60.0
    else:
        correct = [c for c in commits if c["gesture"] == label]
        out["throws_planned"] = meta.get("throws_planned", 0)
        out["throw_acc"] = len(correct) / len(commits) if commits else 0.0
        latencies, visible = [], []
        times = [r["t"] for r in rows]
        for c in commits:
            t_throw = throw_time(rows, times.index(c["t"]), label)
            if t_throw is None:
                continue
            lat_ms = (c["t"] - t_throw) * 1000.0
            latencies.append(lat_ms)
            servo = cfg.latency.servo_transition_ms.get(f"N>{c['pose']}", 150.0)
            visible.append(lat_ms + cfg.latency.camera_latency_ms + servo)
        if latencies:
            out["commit_vs_throw_ms_median"] = float(np.median(latencies))
            out["visible_ms_median"] = float(np.median(visible))
            out["visible_ms_p90"] = float(np.percentile(visible, 90))
            out["latencies_ms"], out["visible_all_ms"] = latencies, visible   # pooled in aggregate()
    return out


def aggregate(scores: List[Dict]) -> Dict:
    def mean(key, kind):
        vals = [s[key] for s in scores if s["type"] == kind and key in s]
        return float(np.mean(vals)) if vals else None

    throws = [s for s in scores if s["type"] == "throws"]

    def pooled_median(key):
        vals = [v for s in throws for v in s.get(key, [])]
        return float(np.median(vals)) if vals else None

    all_commits = sum(s["commits"] for s in throws)
    correct = sum(round(s.get("throw_acc", 0) * s["commits"]) for s in throws)
    return {
        "show_hold_acc": mean("hold_acc", "show"),
        "show_switches_per_min": mean("switches_per_min", "show"),
        "show_cnn_frame_acc": mean("cnn_frame_acc", "show"),
        "show_mp_frame_acc": mean("mp_frame_acc", "show"),
        "throw_acc": correct / all_commits if all_commits else None,
        "throw_commits": all_commits,
        "throws_planned": sum(s.get("throws_planned", 0) for s in throws),
        # every decision counts once: a session with a single stray decision must not weigh as much
        # as a session with 28 throws
        "commit_vs_throw_ms": pooled_median("latencies_ms"),
        "visible_ms": pooled_median("visible_all_ms"),
        "bg_false_commits_per_min": mean("false_commits_per_min", "background"),
    }


def fmt(v, pct=False, digits=1):
    if v is None:
        return "-"
    return f"{v * 100:.{digits}f}%" if pct else f"{v:.{digits}f}"


def print_table(results: Dict[str, Dict]):
    cols = [("show_hold_acc", "hold acc", True), ("show_switches_per_min", "switch/min", False),
            ("show_cnn_frame_acc", "cnn frame", True), ("show_mp_frame_acc", "mp frame", True),
            ("throw_acc", "throw acc", True), ("throw_commits", "commits", False),
            ("throws_planned", "planned", False), ("commit_vs_throw_ms", "commit-throw ms", False),
            ("visible_ms", "visible ms", False), ("bg_false_commits_per_min", "bg false/min", False)]
    print(f"{'source':<22}" + "".join(f"{name:>15}" for _, name, _ in cols))
    for source, agg in results.items():
        print(f"{source:<22}" + "".join(f"{fmt(agg[k], pct):>15}" for k, _, pct in cols))


def go_no_go(results: Dict[str, Dict]):
    fused, mp = results.get("fused"), results.get("mediapipe")
    if not fused or not mp:
        return
    evaluable = any(fused[k] is not None and mp[k] is not None
                    for k in ("throw_acc", "show_hold_acc", "visible_ms"))
    if not evaluable or (not fused["throw_commits"] and not mp["throw_commits"]
                         and not fused["show_hold_acc"] and not mp["show_hold_acc"]):
        print("\nGO/NO-GO: not enough evaluable sessions (need held-out show/throws recordings "
              "with commits) - no verdict.")
        return

    def better_or_equal(a, b, higher=True, tol=0.0):
        if a is None or b is None:
            return True
        return a >= b - tol if higher else a <= b + tol
    ok = (better_or_equal(fused["throw_acc"], mp["throw_acc"]) and
          better_or_equal(fused["show_hold_acc"], mp["show_hold_acc"], tol=0.01) and
          better_or_equal(fused["show_switches_per_min"], mp["show_switches_per_min"], higher=False, tol=0.5) and
          better_or_equal(fused["visible_ms"], mp["visible_ms"], higher=False))
    print("\nGO/NO-GO: " + ("fused pipeline beats MediaPipe-only -> ship FUSED" if ok else
                            "fused does NOT beat MediaPipe-only on held-out data -> ship MEDIAPIPE-ONLY"))


def hand_for(rec_dir: str, cfg, hand_cache: Dict) -> ReplayHand:
    """Hand-tracker results per recording, cached per hand-tracker setting (sweeps may change them)."""
    key = (rec_dir, json.dumps(asdict(cfg.hand), sort_keys=True))
    if key not in hand_cache:
        hand_cache[key] = track_hand(rec_dir, cfg)
    return hand_cache[key]


def evaluate(recordings, cfg, sources, cnn, hand_cache, csv_rows=None, tag="") -> Dict[str, Dict]:
    results = {}
    for source in sources:
        scores = []
        for rec_dir in recordings:
            with open(os.path.join(rec_dir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            hand = hand_for(rec_dir, cfg, hand_cache) if source in ("fused", "mediapipe") else None
            res = replay(rec_dir, meta, cfg, cnn if source in ("fused", "cnn") else None, hand)
            ref = hand
            if ref is None and meta["type"] == "throws":      # same timing reference for every method
                try:
                    ref = hand_for(rec_dir, cfg, hand_cache)
                except Exception as e:                       # hand tracker unavailable: motion-based timing
                    print(f"[replay] hand tracker unavailable for timing ({e}); using motion only")
            add_reference(res["rows"], ref)
            s = score(meta, res, cfg)
            scores.append(s)
            if csv_rows is not None:
                flat = {k: v for k, v in s.items() if not isinstance(v, list)}   # per-decision lists stay out
                csv_rows.append({"tag": tag, "source": source, **flat})
        results[source] = aggregate(scores)
        # machine-readable line for the desktop app (Evaluate tab)
        print("@@RESULT " + json.dumps({"setting": tag, "source": source, **results[source]}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Replay recordings through the live pipeline")
    parser.add_argument("--recordings", default="data/recordings")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--sources", default="mediapipe,cnn,fused")
    parser.add_argument("--sweep", default=None, metavar="SECTION.KEY=V1,V2,...")
    parser.add_argument("--csv", default=None, help="Write per-recording scores here")
    args = parser.parse_args()

    boost_process()
    cfg = load_config(args.config, args.set)
    recordings = find_recordings(args.recordings)
    if not recordings:
        raise SystemExit(f"No recordings under {args.recordings}")
    sources = [s for s in args.sources.split(",") if s]
    cnn = None
    if any(s in ("fused", "cnn") for s in sources):
        if os.path.exists(cfg.cnn.model_path):
            from rps.cnn import GestureCNN
            cnn = GestureCNN(cfg.cnn.model_path, cfg.cnn.threads, cfg.cnn.rotate, cfg.cnn.flip)
            for w in cnn.check_dvs(cfg.dvs):
                print(f"[replay] WARNING: {w}")
        else:
            print(f"[replay] No motion model file at {cfg.cnn.model_path}: scoring the hand tracker only.")
            sources = [s for s in sources if s == "mediapipe"]
    print(f"[replay] {len(recordings)} recordings, sources {sources}")

    hand_cache, csv_rows = {}, []
    if args.sweep:
        path, values = args.sweep.split("=", 1)
        section, key = path.split(".", 1)
        for value in values.split(","):
            cfg_v = copy.deepcopy(cfg)
            apply_overrides(cfg_v, {section: {key: value}})
            print(f"\n=== {path} = {value} ===")
            print_table(evaluate(recordings, cfg_v, sources, cnn, hand_cache, csv_rows, tag=f"{path}={value}"))
    else:
        results = evaluate(recordings, cfg, sources, cnn, hand_cache, csv_rows)
        print()
        print_table(results)
        go_no_go(results)

    if args.csv and csv_rows:
        keys = sorted({k for row in csv_rows for k in row})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(csv_rows)
        print(f"[replay] Per-recording scores written to {args.csv}")


if __name__ == "__main__":
    main()
