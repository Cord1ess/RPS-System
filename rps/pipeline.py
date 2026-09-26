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


class Pipeline:
    def __init__(self, cfg: Config, cnn=None, hand=None,
                 pose_sink: Optional[Callable[[str], None]] = None, allow_mp_skip: bool = True):
        self.cfg = cfg
        self.cnn = cnn
        self.hand = hand
        self.pose_sink = pose_sink
        self.allow_mp_skip = allow_mp_skip
        self.dvs = PseudoDVS(cfg.dvs)
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
        motion = MotionObs(frame.id, frame.t, stats.events, centroid_y,
                           (cnn_out[0], cnn_out[1]) if cnn_out is not None else None)
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
        sent = r.poses[0][1] if r.poses else None
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
