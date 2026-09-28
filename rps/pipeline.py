"""
Per-frame processing order shared by play.py (live) and replay_eval.py (offline), so that
replayed results match live behaviour:

    ROI -> PseudoDVS -> [frame emitted?] -> CNN -> decision.on_motion -> send pose (immediately)
        -> MediaPipe (same frame) -> decision.on_hand -> send pose if changed
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from rps.camera import Frame, Roi, crop_roi
from rps.config import Config
from rps.decision import DecisionEngine, MotionObs, Snapshot
from rps.dvs_emulator import DvsFrame, DvsStats, PseudoDVS, events_in_box
from rps.hand_tracker import HandObs
from rps.motion import VerticalMotion


@dataclass
class StepResult:
    frame: Frame
    dvs_frame: Optional[DvsFrame]
    dvs_stats: DvsStats
    cnn: Optional[Tuple[int, float, np.ndarray]]      # (label, confidence, probabilities)
    hand: Optional[HandObs]
    events_in_hand: Optional[int]
    poses: List[Tuple[str, float]] = field(default_factory=list)   # (pose, perf_counter when sent)
    snapshot: Optional[Snapshot] = None
    dvs_ms: float = 0.0
    cnn_ms: float = 0.0
    mp_ms: float = 0.0
    total_ms: float = 0.0
    vy: Optional[float] = None


# The four ways to read the hand, by the names used everywhere in the app.
RECOGNIZERS = {
    "dextra_raw": "Dextra Raw",
    "dextra_tuned": "Dextra Tuned",
    "mediapipe": "Mediapipe",
    "both": "Both (Dextra Tuned + Mediapipe)",
}


def recognizer_parts(cfg: Config, recognizer: str) -> Tuple[Optional[str], bool]:
    """(Dextra model file or None, whether Mediapipe runs) for a recognizer name."""
    if recognizer not in RECOGNIZERS:
        raise ValueError(f"Unknown recognizer '{recognizer}'")
    model = {"dextra_raw": cfg.cnn.raw_model, "dextra_tuned": cfg.cnn.tuned_model,
             "both": cfg.cnn.tuned_model}.get(recognizer)
    return model, recognizer in ("mediapipe", "both")


def missing_reason(cfg: Config, recognizer: str) -> Optional[str]:
    """Why a recognizer cannot run as named (its model file is missing), or None."""
    import os
    model, _ = recognizer_parts(cfg, recognizer)
    if model is None or os.path.exists(model):
        return None
    if model == cfg.cnn.raw_model:
        return "Dextra Raw is not downloaded yet (Bot tuning or Play Debug: Download Dextra)."
    return "Dextra Tuned does not exist yet: tune Dextra on the Train page."


def load_models(cfg: Config, recognizer: str) -> Tuple[object, object, List[str]]:
    """
    Loads Dextra (raw or tuned) and/or Mediapipe for a recognizer. "Both" still runs on Mediapipe
    alone when Dextra Tuned is missing. Returns (cnn, hand, messages); raises RuntimeError when the
    chosen recognizer cannot run at all.
    """
    import os
    cnn = hand = None
    messages: List[str] = []
    model, use_mp = recognizer_parts(cfg, recognizer)
    dextra = "Dextra Raw" if model == cfg.cnn.raw_model else "Dextra Tuned"
    if model is not None:
        if os.path.exists(model):
            from rps.cnn import GestureCNN
            try:
                cnn = GestureCNN(model, cfg.cnn.threads, cfg.cnn.rotate, cfg.cnn.flip)
                messages += [f"Note: {w}" for w in cnn.check_dvs(cfg.dvs)]
            except Exception as e:   # wrong or damaged file
                if not use_mp:
                    raise RuntimeError(f"{dextra} ({model}) could not be loaded: {e}")
                messages.append(f"{dextra} ({model}) could not be loaded ({e}): playing with Mediapipe only.")
        elif not use_mp:
            raise RuntimeError(missing_reason(cfg, recognizer))
        else:
            messages.append(f"{missing_reason(cfg, recognizer)} Playing with Mediapipe only.")
    if use_mp and cfg.hand.enabled:
        try:
            from rps.hand_tracker import HandTracker
            hand = HandTracker(cfg.hand)
        except Exception as e:  # MediaPipe missing or model file absent
            if cnn is None:
                raise RuntimeError(f"Mediapipe could not start: {e}")
            messages.append(f"Mediapipe could not start ({e}): playing with {dextra} only.")
    if cnn is None and hand is None:
        raise RuntimeError("Neither Dextra nor Mediapipe is available.")
    return cnn, hand, messages


def reader_name(source: str, cnn) -> str:
    """Display name of the reader that made a decision ("cnn" / "mp" in the engine)."""
    if source == "mp":
        return "Mediapipe"
    if source == "cnn":
        return "Dextra Tuned" if getattr(cnn, "meta", {}).get("tuned_from") else "Dextra Raw"
    return ""


class Pipeline:
    def __init__(self, cfg: Config, cnn=None, hand=None,
                 pose_sink: Optional[Callable[[str], None]] = None, allow_mp_skip: bool = True):
        self.cfg = cfg
        self.cnn = cnn
        self.hand = hand
        self.pose_sink = pose_sink
        self.allow_mp_skip = allow_mp_skip
        self.dvs = PseudoDVS(cfg.dvs)
        self.motion = VerticalMotion()
        self.engine = DecisionEngine(cfg.decision, cfg.vote, use_cnn=cnn is not None, use_mp=hand is not None)
        self.last_dvs_frame: Optional[DvsFrame] = None

    def _emit_pose(self, pose: Optional[str], result: StepResult):
        if pose is None:
            return
        if self.pose_sink is not None:
            self.pose_sink(pose)
        result.poses.append((pose, time.perf_counter()))

    def set_mode(self, mode: str, t: float):
        pose = self.engine.set_mode(mode, t)
        if pose is not None and self.pose_sink is not None:
            self.pose_sink(pose)

    def step(self, frame: Frame, roi: Roi) -> StepResult:
        t0 = time.perf_counter()
        roi_img = crop_roi(frame.bgr, roi)
        dvs_frame, stats = self.dvs.process(roi_img, frame.t)
        t1 = time.perf_counter()
        if dvs_frame is not None:
            self.last_dvs_frame = dvs_frame

        cnn_out = None
        cnn_ms = 0.0
        if dvs_frame is not None and self.cnn is not None:
            label, conf, probs, cnn_ms = self.cnn.predict(dvs_frame.image)
            cnn_out = (label, conf, probs)
        result = StepResult(frame, dvs_frame, stats, cnn_out, None, None, dvs_ms=(t1 - t0) * 1000.0,
                            cnn_ms=cnn_ms)

        centroid_y = stats.centroid[1] if stats.centroid is not None else None
        vy = self.motion.update(self.dvs.last_sensor, frame.t)
        result.vy = vy
        motion = MotionObs(frame.id, frame.t, stats.events, centroid_y,
                           (cnn_out[0], cnn_out[1]) if cnn_out is not None else None, vy)
        self._emit_pose(self.engine.on_motion(motion), result)

        if self.hand is not None:
            hand = self.hand.process(frame.bgr, roi, frame.t, allow_skip=self.allow_mp_skip)
            box = hand.box if hand.present else None
            result.hand = hand
            result.events_in_hand = events_in_box(stats.event_map, box)
            result.mp_ms = hand.ms
            self._emit_pose(self.engine.on_hand(frame.t, hand, result.events_in_hand), result)

        result.snapshot = self.engine.snapshot()
        result.total_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def log_record(self, r: StepResult) -> dict:
        snap = r.snapshot
        sent = r.poses[0][1] if (r.poses and r.frame.live) else None   # only meaningful on the camera clock
        return {
            "frame_id": r.frame.id,
            "t_capture": r.frame.t,
            "events": r.dvs_stats.events,
            "emitted": int(r.dvs_frame is not None),
            "dvs_ms": round(r.dvs_ms, 3),
            "cnn_ms": round(r.cnn_ms, 3) if r.cnn is not None else None,
            "mp_ms": round(r.mp_ms, 3) if r.hand is not None and not r.hand.skipped else None,
            "total_ms": round(r.total_ms, 3),
            "grab_to_send_ms": round((sent - r.frame.t) * 1000.0, 3) if sent is not None else None,
            "cnn_label": r.cnn[0] if r.cnn is not None else None,
            "cnn_conf": round(r.cnn[1], 4) if r.cnn is not None else None,
            "mp_present": int(r.hand.present) if r.hand is not None else None,
            "mp_gesture": r.hand.gesture if r.hand is not None else None,
            "mp_conf": round(r.hand.confidence, 4) if r.hand is not None else None,
            "mp_skipped": int(r.hand.skipped) if r.hand is not None else None,
            "events_in_hand": r.events_in_hand,
            "pose": snap.pose,
            "state": snap.state,
            "human": snap.human,
            "source": snap.source,
        }
