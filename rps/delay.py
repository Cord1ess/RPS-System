"""
Where the time goes, per throw: the delays from the throw appearing to the robot's hand having moved,
in three kinds of time:

    Camera | Waiting for camera images | Computing + UDP send ‖ Wi-Fi | Robot hand moves
    hardware      camera-paced                the laptop       sent   hardware

- Computing + UDP send (measured on every throw): the deciding camera image arriving -> the UDP
  command leaving. This is the laptop's own work.
- Waiting for camera images (measured): first image showing the throw -> the image it was decided
  on. A throw is confirmed on 2-3 images, so this is set by the camera's frame rate (33 ms per image
  at 30 fps). Rock waits for the hand to stop, because the pumping fist looks the same.
- Hardware, from the settings, each labelled with where its number comes from: the camera delay
  test (Setup), the Wi-Fi ping and the move times (Bot tuning).

This is the model behind the Speed card. It has no Qt in it: the desktop app paints it as a stacked
bar (rps.ui.delay_view) and the command line prints it as a bar of blocks with the same colours and
the same numbers.
"""

from dataclasses import dataclass
from typing import List, Optional

from rps.decision import GESTURE_NAME, ROCK
from rps.pipeline import reader_name

COMPUTE, WAIT, HARDWARE = "compute", "wait", "hardware"
COLORS = {COMPUTE: ["#3fb96b"], WAIT: ["#e0a030"], HARDWARE: ["#5b6470", "#7a8390", "#9aa3ad"]}
HISTORY = 10
POSE_GESTURE = {"R": "rock", "P": "paper", "S": "scissors", "N": "ready"}


@dataclass
class Segment:
    name: str
    ms: Optional[float]                     # None: not measured
    kind: str                               # COMPUTE | WAIT | HARDWARE
    source: str                             # where the number comes from


@dataclass
class ThrowDelay:
    """One throw's delays, ready to show."""
    gesture: int
    pose: str
    reader: str
    frames: int
    segments: List[Segment]

    def ms(self, name: str) -> Optional[float]:
        return next(s.ms for s in self.segments if s.name == name)

    @property
    def compute_ms(self) -> Optional[float]:
        return self.ms("Computing + UDP send")

    @property
    def to_command_ms(self) -> Optional[float]:
        """First camera image showing the throw -> command sent."""
        compute = self.compute_ms
        return None if compute is None else self.ms("Waiting for camera images") + compute

    @property
    def hardware_ms(self) -> float:
        return sum(s.ms for s in self.segments if s.kind == HARDWARE and s.ms is not None)

    def known_segments(self) -> List[Segment]:
        return [s for s in self.segments if s.ms is not None and s.ms > 0]

    def sent_after_ms(self) -> Optional[float]:
        """Time from the throw appearing to the command leaving: where the bar's marker goes."""
        total = self.to_command_ms
        if total is None:
            return None
        camera = self.ms("Camera") or 0.0
        return camera + total

    def headline(self) -> str:
        """The one-line summary both front-ends print above the bar."""
        compute, total = self.compute_ms, self.to_command_ms
        if compute is None:
            return "Read (the robot already showed it: nothing sent)"
        return f"Computed and sent in {compute:.0f} ms  ·  throw -> command {total:.0f} ms"

    def detail(self) -> str:
        """The breakdown under the bar: what was measured, and what is still a guess."""
        seg = {s.name: s for s in self.segments}
        hw = [f"{s.name.lower()} {'?' if s.ms is None else f'{s.ms:.0f} ms'}"
              for s in self.segments if s.kind == HARDWARE]
        wait = self.ms("Waiting for camera images")
        return (f"{GESTURE_NAME[self.gesture].capitalize()}, read by {self.reader}. Laptop: "
                + ("nothing to send" if self.compute_ms is None else f"{self.compute_ms:.0f} ms computing and sending")
                + f"; waiting {wait:.0f} ms ({seg['Waiting for camera images'].source.split(': ', 1)[1]}). "
                f"Hardware: {', '.join(hw)} = ~{self.hardware_ms:.0f} ms"
                + ("" if all(s.ms is not None for s in self.segments) else " (? = not measured yet)") + ".")


def robot_move_ms(cfg, robot_from: str, pose: str) -> Optional[float]:
    """The hand's move time for this change, from the move times on Bot tuning (0: already there)."""
    if robot_from == pose:
        return 0.0
    return cfg.latency.servo_transition_ms.get(f"{robot_from}>{pose}")


def throw_delay(timing, robot_from: str, cfg, cnn=None) -> ThrowDelay:
    """A pipeline DecisionTiming plus the hardware delays from the settings."""
    lat = cfg.latency
    more = max(0, timing.frames - 1)
    pace = f" at {timing.read_ms / more:.0f} ms each" if more else ""
    wait = f"{more} more camera image{'s' if more != 1 else ''}{pace} to be sure"
    if timing.gesture == ROCK:
        wait += "; rock waits for the hand to stop (the pumping fist looks the same)"
    work = f"Dextra view {timing.dvs_ms:.0f} ms"
    if timing.cnn_ms:
        work += f", Dextra {timing.cnn_ms:.0f} ms"
    if timing.mp_ms:
        work += f", Mediapipe {timing.mp_ms:.0f} ms"
    return ThrowDelay(timing.gesture, timing.pose, reader_name(timing.source, cnn) or "?", timing.frames, [
        Segment("Camera", lat.camera_latency_ms, HARDWARE, "camera delay test on Setup"),
        Segment("Waiting for camera images", timing.read_ms, WAIT, f"measured: {wait}"),
        Segment("Computing + UDP send", timing.process_ms, COMPUTE, f"measured: {work}, then the UDP send"),
        Segment("Wi-Fi", lat.network_ms or None, HARDWARE,
                "ping on Bot tuning" if lat.network_ms else "not measured: Bot tuning > Measure Wi-Fi delay"),
        Segment("Robot hand moves", robot_move_ms(cfg, robot_from, timing.pose), HARDWARE,
                "already showing it" if robot_from == timing.pose else "move times on Bot tuning"),
    ])
