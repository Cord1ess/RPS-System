"""
Central configuration for the v3 pipeline.

Every tunable lives here as a dataclass default. `config.json` (optional) overrides
any subset of fields, and CLI flags of the form `--set section.key=value` override both.
"""

import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List

DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class CameraConfig:
    # Defaults suit a dim room; run tools/camera_probe.py --write-config in the play lighting.
    # The FHD Camera has no gain control: a locked exposure only works with a well-lit play zone.
    index: int = 0
    backend: str = "msmf"           # "dshow" | "msmf" | "any"
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "YUY2"            # uncompressed: no MJPG block artifacts that would create false events
    mirror: bool = True
    lock_exposure: bool = False
    exposure: float = -6.0          # DirectShow log2 seconds: -6 = 15.6 ms, -7 = 7.8 ms
    lock_white_balance: bool = False
    wb_temperature: int = 4500
    restore_auto_on_exit: bool = True


@dataclass
class RoiConfig:
    # Square play zone in (mirrored) camera pixels. Frame the hand only, not the face.
    x: int = 140
    y: int = 60
    size: int = 360
    record_margin: float = 0.25     # recordings keep this much extra context around the ROI


@dataclass
class DvsConfig:
    sensor_size: int = 128          # ROI is resized to sensor_size x sensor_size (DVS128 geometry)
    frame_size: int = 64            # CNN input size (2x2 binning)
    log_offset: float = 8.0         # L = log(I + offset); damps dark-pixel noise
    contrast_threshold: float = 0.20
    event_count: int = 1500         # N: constant events per emitted frame (Dextra DVS128 value)
    clip_count: int = 16            # K: per-pixel count clip (Dextra EVENT_COUNT_CLIP_VALUE)
    noise_filter: bool = True       # drop events without a firing 3x3 neighbour
    global_reset_fraction: float = 0.60
    flush_min_fraction: float = 0.33    # flush a partial frame on motion stop if >= N * this
    still_events_per_frame: int = 40    # below this a webcam frame counts as "still"
    flush_still_frames: int = 2
    max_accumulation_s: float = 0.5


@dataclass
class CnnConfig:
    model_path: str = "models/motion_cnn_v3.pth"   # or models/dextra_roshambo.pth (tools/import_dextra.py)
    threads: int = 1
    rotate: int = 0                 # rotate DVS frames CCW before the CNN: 0 | 90 | 180 | 270
    flip: bool = False              # mirror DVS frames left-right before the CNN


@dataclass
class HandConfig:
    model_path: str = "models/hand_landmarker.task"
    enabled: bool = True
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    crop_margin: float = 0.25       # MediaPipe sees the ROI plus this margin
    extend_threshold: float = 0.30  # finger curl below this -> extended
    curl_threshold: float = 0.45    # finger curl above this -> curled (hysteresis band between)
    frame_budget_ms: float = 28.0   # skip the next frame if MediaPipe took longer than this


@dataclass
class VoteConfig:
    method: str = "sequence"        # "sequence" (Dextra default) | "majority"
    k: int = 2                      # identical predictions in a row
    min_confidence: float = 0.70
    max_gap_s: float = 0.25         # a streak breaks if predictions are further apart than this
    majority_window: int = 5


@dataclass
class DecisionConfig:
    mode: str = "countdown"         # "continuous" | "countdown"
    source: str = "fused"           # "fused" | "cnn" | "mediapipe"
    active_events_per_frame: int = 100   # webcam-frame events above this -> motion active
    still_events_per_frame: int = 40     # webcam-frame events below this -> still
    still_frames: int = 2           # consecutive still frames -> motion stopped ("settled")
    mp_stable_frames: int = 3       # MediaPipe gesture must repeat this many frames
    mp_min_confidence: float = 0.60
    mp_still_events_in_hand: int = 25    # MediaPipe is still-hand authority only below this
    mp_after_cnn_commit_s: float = 0.15  # ... and only this long after the last CNN commit
    switch_dead_time_s: float = 0.15     # applies to MediaPipe-driven switches and A->B->A flips
    idle_timeout_s: float = 1.0
    idle_action: str = "ready"      # "ready" | "hold"
    # Countdown mode
    pumps_before_shoot: int = 3
    pump_source: str = "flow"            # "flow" (optical flow in the play zone) | "mp" (tracked wrist)
    pump_miss_tolerance: int = 1         # a throw that lands on the beat may come this many pumps early
    pump_min_amplitude: float = 0.06     # smallest stroke before the player's own size is learned (zone heights)
    pump_min_period_s: float = 0.15
    shoot_window_s: float = 1.2
    rock_min_shoot_s: float = 0.15       # rock may commit on settle only after this long in SHOOT
    rock_settle_fallback_s: float = 0.35  # ... and after the final downstroke, or this long if y was lost
    hold_min_s: float = 1.0
    hold_max_s: float = 4.0
    correction_s: float = 0.0            # MediaPipe may correct a commit within this window (0 = off)


@dataclass
class RobotConfig:
    enabled: bool = True
    host: str = "192.168.4.1"       # ESP32 soft-AP default address
    port: int = 4210
    heartbeat_s: float = 0.10
    ack_timeout_s: float = 0.5


@dataclass
class LatencyConfig:
    # Used by replay_eval.py to project when the robot pose becomes visible.
    camera_latency_ms: float = 50.0      # replace with the mirror-test measurement
    servo_transition_ms: Dict[str, float] = field(default_factory=lambda: {
        "N>R": 150.0, "N>P": 150.0, "N>S": 150.0,
        "R>P": 150.0, "R>S": 150.0, "P>R": 150.0,
        "P>S": 150.0, "S>R": 150.0, "S>P": 150.0,
    })


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    dvs: DvsConfig = field(default_factory=DvsConfig)
    cnn: CnnConfig = field(default_factory=CnnConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    vote: VoteConfig = field(default_factory=VoteConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce(current: Any, value: Any) -> Any:
    """Converts a JSON/CLI value to the type of the existing default."""
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, dict) and isinstance(value, str):
        return json.loads(value)
    return value


def apply_overrides(cfg: Config, overrides: Dict[str, Dict[str, Any]]) -> Config:
    """Applies {section: {key: value}} onto cfg, rejecting unknown keys loudly."""
    for section, values in overrides.items():
        if not hasattr(cfg, section):
            raise KeyError(f"Unknown config section '{section}'")
        sec = getattr(cfg, section)
        valid = {f.name for f in fields(sec)}
        for key, value in values.items():
            if key not in valid:
                raise KeyError(f"Unknown config key '{section}.{key}'")
            setattr(sec, key, _coerce(getattr(sec, key), value))
    return cfg


def parse_set_args(items: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parses ['decision.mode=continuous', 'vote.k=3'] into nested overrides."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in items or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"Expected section.key=value, got '{item}'")
        path, value = item.split("=", 1)
        section, key = path.split(".", 1)
        out.setdefault(section, {})[key] = value
    return out


def load_config(path: str = DEFAULT_CONFIG_PATH, set_args: List[str] = None) -> Config:
    """Defaults <- config.json (if present) <- --set overrides."""
    cfg = Config()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            apply_overrides(cfg, json.load(f))
    if set_args:
        apply_overrides(cfg, parse_set_args(set_args))
    return cfg


def save_config(cfg: Config, path: str = DEFAULT_CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"[config] Saved configuration to {os.path.abspath(path)}")
