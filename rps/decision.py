"""
Decision engine: fuses CNN (motion) and MediaPipe (still hand) evidence and runs the game.

Pure logic, no I/O: the pipeline feeds it per-frame observations in a fixed order
(on_motion first, so the CNN path is never delayed; then on_hand for the same frame).
Both calls return the new robot pose ("R", "P", "S" or "N" = ready) when it changes, else None.

Fusion rules
- While motion is active the CNN is the authority: a SequenceVote commit switches immediately.
- MediaPipe may commit as the still-hand authority only when its frame has (almost) no events
  inside the hand box and was captured after the last CNN commit plus a margin, so a late,
  blurred mid-throw fist cannot overwrite a correct paper/scissors commit.
- Hysteresis (dead time) applies to MediaPipe-driven switches and to A->B->A flips only.
- A background vote never changes the command (Dextra behaviour).

Game modes
- continuous: the robot always shows the counter to the current decision.
- countdown: IDLE -> ARMED (count pumps) -> SHOOT (paper/scissors commit on vote, rock commits
  on settle) -> HOLD. The pumping fist is ignored because it is also "rock".
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from model import COUNTER_MOVES
from rps.config import DecisionConfig, VoteConfig
from rps.hand_tracker import BACKGROUND, PAPER, ROCK, SCISSORS, HandObs
from rps.voting import make_voter

READY = "N"
POSE_LETTER = {"rock": "R", "paper": "P", "scissors": "S"}
COUNTER_POSE = {g: POSE_LETTER[COUNTER_MOVES[g]["ai_symbol"]] for g in (ROCK, PAPER, SCISSORS)}
GESTURE_NAME = {ROCK: "rock", PAPER: "paper", SCISSORS: "scissors", BACKGROUND: "background", -1: "unknown"}

IDLE, ARMED, SHOOT, HOLD = "IDLE", "ARMED", "SHOOT", "HOLD"


@dataclass
class MotionObs:
    frame_id: int
    t: float
    events: int                                 # filtered events of this webcam frame
    centroid_y: Optional[float] = None          # event centroid, ROI-normalized
    cnn: Optional[Tuple[int, float]] = None     # (label, confidence) if a DVS frame was classified


@dataclass
class Snapshot:
    mode: str
    state: str
    pose: str
    human: Optional[int]
    source: str = ""
    motion_active: bool = False
    pumps: int = 0
    switches: int = 0
    commits: int = 0
    cnn_vote: Optional[int] = None
    mp_gesture: Optional[int] = None
    reason: str = ""


class PumpDetector:
    """Counts downstroke bottoms (image y maxima) with amplitude hysteresis and a min period."""

    def __init__(self, min_amplitude: float, min_period_s: float):
        self.min_amplitude = min_amplitude
        self.min_period_s = min_period_s
        self.reset()

    def reset(self):
        self.extreme: Optional[float] = None
        self.extreme_t = 0.0
        self.direction = 0            # +1 moving down (y grows), -1 moving up
        self.last_bottom_t = -1e9

    def update(self, t: float, y: Optional[float]) -> bool:
        """Returns True when a bottom is confirmed (the hand rose again by min_amplitude)."""
        if y is None:
            return False
        if self.extreme is None:
            self.extreme, self.extreme_t = y, t
            return False
        if self.direction >= 0:
            if y > self.extreme:
                self.extreme, self.extreme_t = y, t
                self.direction = 1
            elif self.direction == 1 and self.extreme - y >= self.min_amplitude:
                bottom_t = self.extreme_t
                self.direction, self.extreme, self.extreme_t = -1, y, t
                if bottom_t - self.last_bottom_t >= self.min_period_s:
                    self.last_bottom_t = bottom_t
                    return True
            elif self.direction == 0 and self.extreme - y >= self.min_amplitude:
                self.direction, self.extreme, self.extreme_t = -1, y, t
        else:
            if y < self.extreme:
                self.extreme, self.extreme_t = y, t
            elif y - self.extreme >= self.min_amplitude:
                self.direction, self.extreme, self.extreme_t = 1, y, t
        return False


class DecisionEngine:
    def __init__(self, cfg: DecisionConfig, vote_cfg: VoteConfig, use_cnn: bool, use_mp: bool):
        self.cfg = cfg
        self.use_cnn = use_cnn
        self.use_mp = use_mp
        self.voter = make_voter(vote_cfg)
        self.pump = PumpDetector(cfg.pump_min_amplitude, cfg.pump_min_period_s)
        self.pump_from_mp = use_mp and cfg.pump_source in ("auto", "mp")
        self.mode = cfg.mode
        self.pose = READY
        self.human: Optional[int] = None
        self.prev_human: Optional[int] = None
        self.last_switch_t = -1e9
        self.last_cnn_commit_t = -1e9
        self.commit_t = -1e9
        self.last_source = ""
        self.switches = 0
        self.commits = 0
        # motion
        self.motion_active = False
        self.still_run = 0
        self.last_motion_t = -1e9
        self.settled_t = -1e9
        # evidence
        self.cnn_vote: Optional[int] = None
        self.cnn_rock_t = -1e9
        self.mp_gesture: Optional[int] = None
        self.mp_count = 0
        self.last_hand_t = -1e9
        self.mp_present = False
        # countdown
        self.state = IDLE
        self.pumps = 0
        self.shoot_t = -1e9
        self.shoot_descended = False
        self.reason = ""

    # ---------------------------------------------------------------- public API
    def set_mode(self, mode: str, t: float = 0.0) -> Optional[str]:
        self.mode = mode
        self.state = IDLE
        self.pumps = 0
        self.pump.reset()
        self.voter.reset()
        self.human = None
        self.reason = f"mode -> {mode}"
        return self._set_pose(READY)

    def snapshot(self) -> Snapshot:
        return Snapshot(self.mode, self.state if self.mode == "countdown" else "LIVE", self.pose, self.human,
                        self.last_source, self.motion_active, self.pumps, self.switches, self.commits,
                        self.cnn_vote, self.mp_gesture, self.reason)

    def on_motion(self, obs: MotionObs) -> Optional[str]:
        t = obs.t
        settled = self._update_motion(obs)
        vote = None
        if obs.cnn is not None and self.use_cnn:
            vote = self.voter.update(obs.cnn[0], obs.cnn[1], t)
            self.cnn_vote = vote
            if vote == ROCK:
                self.cnn_rock_t = t
        if self.mode == "continuous":
            return self._continuous_motion(t, vote)
        return self._countdown_motion(t, vote, settled, obs.centroid_y)

    def on_hand(self, t: float, hand: Optional[HandObs], events_in_hand: Optional[int]) -> Optional[str]:
        if hand is None or hand.skipped or not self.use_mp:
            return None
        self._update_mp(t, hand)
        eligible = self._mp_eligible(t, events_in_hand)
        if self.mode == "continuous":
            return self._continuous_hand(t, eligible)
        return self._countdown_hand(t, hand, eligible)

    # ---------------------------------------------------------------- evidence tracking
    def _update_motion(self, obs: MotionObs) -> bool:
        """Updates motion activity; returns True on the frame where motion settles."""
        cfg = self.cfg
        if obs.events >= cfg.active_events_per_frame:
            self.motion_active = True
            self.still_run = 0
            self.last_motion_t = obs.t
            return False
        if obs.events < cfg.still_events_per_frame:
            self.still_run += 1
            if self.motion_active and self.still_run >= cfg.still_frames:
                self.motion_active = False
                self.settled_t = obs.t
                return True
        return False

    def _update_mp(self, t: float, hand: HandObs):
        self.mp_present = hand.present
        if hand.present:
            self.last_hand_t = t
        g = hand.gesture if hand.present and hand.confidence >= self.cfg.mp_min_confidence else None
        if g is not None and g in (ROCK, PAPER, SCISSORS):
            self.mp_count = self.mp_count + 1 if g == self.mp_gesture else 1
            self.mp_gesture = g
        else:
            self.mp_gesture, self.mp_count = None, 0

    def _mp_stable(self) -> bool:
        return self.mp_gesture is not None and self.mp_count >= self.cfg.mp_stable_frames

    def _mp_eligible(self, t: float, events_in_hand: Optional[int]) -> bool:
        if not self._mp_stable():
            return False
        if not self.use_cnn:
            return True     # MediaPipe-only: it is the sole authority
        still = events_in_hand is None or events_in_hand <= self.cfg.mp_still_events_in_hand
        return still and t >= self.last_cnn_commit_t + self.cfg.mp_after_cnn_commit_s

    def _hand_present(self, t: float) -> bool:
        if self.use_mp:
            return (t - self.last_hand_t) < 0.3
        return (t - self.last_motion_t) < self.cfg.idle_timeout_s

    def _idle(self, t: float) -> bool:
        recent_hand = (t - self.last_hand_t) < self.cfg.idle_timeout_s if self.use_mp else False
        return not recent_hand and (t - self.last_motion_t) >= self.cfg.idle_timeout_s

    # ---------------------------------------------------------------- commits
    def _set_pose(self, pose: str) -> Optional[str]:
        if pose == self.pose:
            return None
        self.pose = pose
        return pose

    def _commit(self, gesture: int, source: str, t: float, force: bool = False) -> Optional[str]:
        if gesture == self.human and not force:
            return None
        dead = self.cfg.switch_dead_time_s
        if not force and self.human is not None:
            if source == "mp" and t - self.last_switch_t < dead:
                return None
            if gesture == self.prev_human and t - self.last_switch_t < dead:
                return None
        if self.human is not None and gesture != self.human:
            self.switches += 1
        self.prev_human, self.human = self.human, gesture
        self.last_switch_t = self.commit_t = t
        self.last_source = source
        self.commits += 1
        if source == "cnn":
            self.last_cnn_commit_t = t
        self.reason = f"{GESTURE_NAME[gesture]} via {source}"
        return self._set_pose(COUNTER_POSE[gesture])

    # ---------------------------------------------------------------- continuous mode
    def _continuous_motion(self, t: float, vote: Optional[int]) -> Optional[str]:
        if vote in (ROCK, PAPER, SCISSORS):
            return self._commit(vote, "cnn", t)
        return self._continuous_idle(t)

    def _continuous_hand(self, t: float, eligible: bool) -> Optional[str]:
        if eligible:
            return self._commit(self.mp_gesture, "mp", t)
        return self._continuous_idle(t)

    def _continuous_idle(self, t: float) -> Optional[str]:
        if self.human is not None and self.cfg.idle_action == "ready" and self._idle(t):
            self.human = None
            self.reason = "idle"
            return self._set_pose(READY)
        return None

    # ---------------------------------------------------------------- countdown mode
    def _enter(self, state: str, t: float, reason: str) -> Optional[str]:
        self.state = state
        self.reason = reason
        if state in (IDLE, ARMED):
            self.pumps = 0
            self.pump.reset()
            self.voter.reset()
            self.human = None
            if state == ARMED or self.cfg.idle_action == "ready":
                return self._set_pose(READY)
        if state == SHOOT:
            self.shoot_t = t
            self.shoot_descended = False
            self.voter.reset()
        if state == HOLD:
            self.pump.reset()
        return None

    def _pump_y(self, centroid_y: Optional[float], hand: Optional[HandObs]) -> Optional[float]:
        if self.pump_from_mp:
            return hand.wrist_y if hand is not None and hand.present else None
        return centroid_y

    def _countdown_common(self, t: float, y: Optional[float]) -> Optional[str]:
        """
        Presence, pump counting and timeouts. Called exactly once per frame from whichever
        stream carries the pump position (on_hand for MediaPipe wrist y, on_motion for events).
        """
        cfg = self.cfg
        if self.state == IDLE:
            if self._hand_present(t) or self.motion_active:
                return self._enter(ARMED, t, "hand in play zone")
            return None
        if self.state == ARMED:
            if self._idle(t):
                return self._enter(IDLE, t, "no hand")
            if self.pump.update(t, y):
                self.pumps += 1
                self.reason = f"pump {self.pumps}"
                if self.pumps >= cfg.pumps_before_shoot:
                    return self._enter(SHOOT, t, "shoot!")
            return None
        if self.state == SHOOT:
            self.pump.update(t, y)
            if self.pump.direction == 1:
                self.shoot_descended = True
            if t - self.shoot_t > cfg.shoot_window_s:
                return self._enter(ARMED, t, "shoot window expired")
            return None
        if self.state == HOLD:
            held = t - self.commit_t
            if held >= cfg.hold_max_s:
                return self._enter(ARMED, t, "hold expired")
            bottom = self.pump.update(t, y)
            if held >= cfg.hold_min_s:
                if self._idle(t):
                    return self._enter(IDLE, t, "hand left")
                if bottom:
                    out = self._enter(ARMED, t, "new round")
                    self.pumps = 1
                    return out
        return None

    def _rock_allowed(self, t: float) -> bool:
        """
        Rock is the pumping fist, so it may only commit once the final downstroke has been
        seen (or, if position tracking was lost, after a fallback delay) and the hand settled.
        """
        since = t - self.shoot_t
        if since < self.cfg.rock_min_shoot_s:
            return False
        return self.shoot_descended or since >= self.cfg.rock_settle_fallback_s

    def _shoot_commit(self, gesture: int, source: str, t: float) -> Optional[str]:
        res = self._commit(gesture, source, t, force=True)
        self.state = HOLD
        self.pump.reset()
        return res

    def _countdown_motion(self, t: float, vote: Optional[int], settled: bool,
                          centroid_y: Optional[float]) -> Optional[str]:
        out = None if self.pump_from_mp else self._countdown_common(t, centroid_y)
        if self.state != SHOOT:
            return out
        if vote in (PAPER, SCISSORS):
            return self._shoot_commit(vote, "cnn", t)
        if settled and self._rock_allowed(t):
            if self.cnn_rock_t >= self.shoot_t:
                return self._shoot_commit(ROCK, "cnn", t)
            if self.use_mp and self.mp_gesture == ROCK:
                return self._shoot_commit(ROCK, "mp", t)
        return out

    def _countdown_hand(self, t: float, hand: HandObs, eligible: bool) -> Optional[str]:
        out = self._countdown_common(t, self._pump_y(None, hand)) if self.pump_from_mp else None
        if out is not None:
            return out
        if self.state == SHOOT and self._mp_stable():
            g = self.mp_gesture
            if g in (PAPER, SCISSORS) or (g == ROCK and not self.motion_active and self._rock_allowed(t)):
                return self._shoot_commit(g, "mp", t)
        if self.state == HOLD and self.cfg.correction_s > 0 and eligible:
            if t - self.commit_t <= self.cfg.correction_s and self.mp_gesture != self.human:
                return self._commit(self.mp_gesture, "mp", t, force=True)
        return None
