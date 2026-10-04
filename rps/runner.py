"""
The live loop with no window attached: camera (or a replay) -> Dextra and/or Mediapipe -> decision
-> robot. Everything that shows what is happening goes through hooks, so this one runtime drives
both the OpenCV window in play.py and the text HUD on the command line.

The recognizers are named as in the app: dextra_raw, dextra_tuned, mediapipe, both.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from rps.camera import open_source
from rps.decision import GESTURE_NAME
from rps.game import ENDLESS_ROUNDS, BeatSchedule
from rps.perf import boost_process
from rps.pipeline import RECOGNIZERS, Pipeline, load_models, reader_name
from rps.robot_link import MockEsp, RobotLink
from rps.timing import LatencyLog


class RunnerError(RuntimeError):
    """Something the person running it has to fix: no model, no recognizer, no camera."""


@dataclass
class RunStats:
    """What a run did, for the summary at the end."""
    frames: int = 0
    laptop_ms: List[float] = field(default_factory=list)   # per decision: first frame showing the throw -> command sent
    log_summary: dict = field(default_factory=dict)
    commits: int = 0
    switches: int = 0
    hand_skipped: int = 0
    mock: bool = False
    mock_poses: List[str] = field(default_factory=list)


def decision_line(d, cnn) -> str:
    """One throw, the way the runtime has always reported it."""
    total = d.software_ms
    return (f"{GESTURE_NAME[d.gesture]} read by {reader_name(d.source, cnn)} -> robot {d.pose}: "
            f"reading {d.read_ms:.0f} ms ({d.frames} frames) + processing and send "
            + ("-" if d.process_ms is None else f"{d.process_ms:.0f} ms")
            + ("" if total is None else f" = {total:.0f} ms on the laptop"))


def summary_lines(stats: RunStats) -> List[str]:
    """The end-of-run summary, so the window and the terminal report a run the same way."""
    s = stats.log_summary
    lines = ["", "================ SESSION SUMMARY ================",
             f"  frames: {stats.frames}   camera fps: {s.get('fps', 0):.1f}"]
    if stats.laptop_ms:
        lines.append(f"  throw read and sent (laptop): median {float(np.median(stats.laptop_ms)):.0f} ms, "
                     f"fastest {min(stats.laptop_ms):.0f} ms, slowest {max(stats.laptop_ms):.0f} ms "
                     f"over {len(stats.laptop_ms)} throws")
    for key in ("dvs_ms", "cnn_ms", "mp_ms", "total_ms", "grab_to_send_ms"):
        if f"{key}_mean" in s:
            lines.append(f"  {key:16s} mean {s[f'{key}_mean']:6.2f}   p95 {s[f'{key}_p95']:6.2f}")
    lines.append(f"  commits: {stats.commits}   switches: {stats.switches}   hand-path skips: {stats.hand_skipped}")
    if stats.mock:
        lines.append(f"  mock ESP pose changes: {stats.mock_poses}")
    lines.append("=================================================")
    return lines


class Runner:
    """Camera -> recognizers -> decision -> robot, ready to run. `on_frame`, `on_decision` and
    `on_note` are how a front-end watches: on_frame returning False stops the run."""

    def __init__(self, cfg, recognizer: Optional[str] = None, video: Optional[str] = None,
                 mock_camera: bool = False, no_robot: bool = False, mock_esp: bool = False,
                 log_path: Optional[str] = None):
        boost_process()          # keep off EcoQoS efficiency cores: ~2x faster, steadier MediaPipe
        self.cfg = cfg
        self.video = video
        self.recognizer = recognizer or cfg.decision.recognizer
        self.cnn, self.hand, self.load_messages = self._load_models()
        self.effective = " + ".join(n for n, on in ((reader_name("cnn", self.cnn), self.cnn),
                                                    ("Mediapipe", self.hand)) if on)
        self.robot = "off" if no_robot else ("simulated" if mock_esp else cfg.robot.mode)
        self.mock = MockEsp(port=cfg.robot.port).start() if self.robot == "simulated" else None
        self.link = (RobotLink.from_config(cfg.robot, host="127.0.0.1" if self.mock else None).start()
                     if self.robot != "off" else None)
        self.source = open_source(cfg.camera, cfg.roi, video=video, mock=mock_camera, realtime=True)
        self.pipeline = Pipeline(cfg, self.cnn, self.hand, pose_sink=self.link.send_pose if self.link else None)
        if cfg.decision.mode == "guided":
            g = cfg.game
            schedule = BeatSchedule(time.perf_counter(), 60.0 / g.beat_bpm, cfg.decision.pumps_before_shoot,
                                    g.rounds or ENDLESS_ROUNDS, g.lead_beats, g.gap_beats)
            self.pipeline.engine.start_guided(schedule, cfg.latency.camera_latency_ms / 1000.0)
        self.log = LatencyLog(log_path)

    def _load_models(self):
        """Dextra and/or Mediapipe for the recognizer, degrading gracefully for "both"."""
        try:
            cnn, hand, messages = load_models(self.cfg, self.recognizer)
        except RuntimeError as e:
            hint = "" if self.recognizer == "mediapipe" else " Or play with Mediapipe alone: --recognizer mediapipe"
            raise RunnerError(f"{e}{hint}") from None
        return cnn, hand, messages

    @property
    def label(self) -> str:
        return f"{RECOGNIZERS[self.recognizer]} (running: {self.effective})"

    @property
    def banner(self) -> str:
        """The one-line title the window puts on the video."""
        return f"RPS  |  {RECOGNIZERS[self.recognizer]}"

    def run(self, limit: int = 0, on_frame: Optional[Callable] = None,
            on_decision: Optional[Callable] = None, on_note: Optional[Callable] = None) -> RunStats:
        """Run until stopped. `limit` frames ends it early (a headless smoke test); on_frame is
        called per frame to draw and read keys, and stops the run by returning False."""
        stats = RunStats()
        note = on_note if on_note is not None else (lambda _text: None)
        try:
            while True:
                frame = self.source.read(timeout=2.0)
                if frame is None:
                    if self.video:
                        break
                    note("No frame from camera (timeout): robot to ready until frames return.")
                    self.pipeline.set_mode(self.pipeline.engine.mode, time.perf_counter())   # restart the game, send READY
                    continue
                result = self.pipeline.step(frame, self.source.roi)
                self.log.add(self.pipeline.log_record(result))
                stats.frames += 1
                d = result.decision
                if d is not None:
                    if d.software_ms is not None:
                        stats.laptop_ms.append(d.software_ms)
                    if on_decision is not None:
                        on_decision(d)
                if limit and stats.frames >= limit:
                    break
                if on_frame is not None and on_frame(frame, result) is False:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._close(stats)
        return stats

    def _close(self, stats: RunStats) -> RunStats:
        self.source.stop()
        if self.link:
            self.link.stop()
        if self.mock:
            self.mock.stop()
        if self.hand:
            self.hand.close()
        self.log.close()
        stats.log_summary = self.log.summary()
        snap = self.pipeline.engine.snapshot()
        stats.commits, stats.switches = snap.commits, snap.switches
        stats.hand_skipped = self.hand.skipped_frames if self.hand else 0
        stats.mock = self.mock is not None
        stats.mock_poses = [p for _, p in self.mock.pose_log] if self.mock else []
        return stats

    def mode_to(self, mode: str, t: float) -> None:
        self.pipeline.set_mode(mode, t)

    def reset_counters(self) -> None:
        self.pipeline.engine.switches = self.pipeline.engine.commits = 0


def runner_from_args(cfg, args) -> Runner:
    """A Runner from parsed arguments, for play.py and `rps play` alike."""
    return Runner(cfg, recognizer=args.recognizer, video=args.video, mock_camera=args.mock_camera,
                  no_robot=args.no_robot, mock_esp=args.mock_esp, log_path=args.log)