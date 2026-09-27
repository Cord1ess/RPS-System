"""
Decision engine: fuses motion-model (CNN) and hand-tracker (MediaPipe) evidence and runs the game.

Pure logic, no I/O: the pipeline feeds it per-frame observations in a fixed order
(on_motion first, so the CNN path is never delayed; then on_hand for the same frame).
Both calls return the new robot pose ("R", "P", "S" or "N" = ready) when it changes, else None.

Fusion rules
- While the hand moves the CNN is the authority: a vote commit switches immediately.
- The hand tracker decides only when the hand is still (few events in its box) and after the last
  CNN decision plus a margin, so a late, blurred mid-throw fist cannot overwrite a correct commit.
- Hysteresis (dead time) applies to tracker-driven switches and to A->B->A flips only.
- A background vote never changes the command (Dextra behaviour).

Game modes
- continuous: the robot always shows the counter to the current decision.
- countdown: IDLE -> ARMED (count pumps) -> SHOOT -> HOLD. Pumps come from rps.motion's
  RhythmPumpDetector, which learns the player's tempo. The throw is the stroke that lands (stops)
  instead of reversing; one missed pump is tolerated when the landing comes on the beat.
  Paper/scissors commit as soon as they are seen; rock (the pumping fist) only once the throw has
  landed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from model import COUNTER_MOVES
from rps.config import DecisionConfig, VoteConfig
from rps.hand_tracker import BACKGROUND, PAPER, ROCK, SCISSORS, HandObs
from rps.motion import RhythmPumpDetector
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
    vy: Optional[float] = None                  # vertical hand velocity, play-zone heights/s (+ = down)


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
    tempo: Optional[float] = None               # learned seconds per pump


class DecisionEngine:
    def __init__(self, cfg: DecisionConfig, vote_cfg: VoteConfig, use_cnn: bool, use_mp: bool):
        self.cfg = cfg
        self.use_cnn = use_cnn
        self.use_mp = use_mp
        self.voter = make_voter(vote_cfg)
        self.pump = RhythmPumpDetector(cfg.pump_min_amplitude, min_period_s=cfg.pump_min_period_s)
        self.pump_from_mp = use_mp and cfg.pump_source == "mp"
        self._wrist = None                      # (t, y) for wrist velocity when pump_source == "mp"
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
        # evidence
        self.cnn_vote: Optional[int] = None
        self.last_vote: Optional[int] = None
        self.last_vote_t = -1e9
        self.cnn_rock_t = -1e9
        self.mp_gesture: Optional[int] = None
        self.mp_count = 0
        self.last_hand_t = -1e9
        self.mp_present = False
        # countdown
        self.state = IDLE
        self.pumps = 0
        self.last_bottom_t = -1e9
        self.shoot_t = -1e9
        self.shoot_landed = False
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
                        self.cnn_vote, self.mp_gesture, self.reason, self.pump.tempo)

    def on_motion(self, obs: MotionObs) -> Optional[str]:
        t = obs.t
        self._update_motion(obs)
        vote = None
        if obs.cnn is not None and self.use_cnn:
            vote = self.voter.update(obs.cnn[0], obs.cnn[1], t)
            self.cnn_vote = vote
            if vote in (ROCK, PAPER, SCISSORS):
                self.last_vote, self.last_vote_t = vote, t
            if vote == ROCK:
                self.cnn_rock_t = t
        if self.mode == "continuous":
            if vote in (ROCK, PAPER, SCISSORS):
                return self._commit(vote, "cnn", t)
            return self._continuous_idle(t)
        event = None if self.pump_from_mp else self.pump.update(t, obs.vy)
        return self._countdown(t, event, vote)

    def on_hand(self, t: float, hand: Optional[HandObs], events_in_hand: Optional[int]) -> Optional[str]:
        if hand is None or hand.skipped or not self.use_mp:
            return None
        self._update_mp(t, hand)
        eligible = self._mp_eligible(t, events_in_hand)
        if self.mode == "continuous":
            if eligible:
                return self._commit(self.mp_gesture, "mp", t)
            return self._continuous_idle(t)
        event = self.pump.update(t, self._wrist_velocity(t, hand)) if self.pump_from_mp else None
        out = self._countdown(t, event, None)
        if out is None and self.state == HOLD and self.cfg.correction_s > 0 and eligible \
                and t - self.commit_t <= self.cfg.correction_s and self.mp_gesture != self.human:
            out = self._commit(self.mp_gesture, "mp", t, force=True)
        return out

    # ---------------------------------------------------------------- evidence tracking
    def _update_motion(self, obs: MotionObs):
        cfg = self.cfg
        if obs.events >= cfg.active_events_per_frame:
            self.motion_active = True
            self.still_run = 0
            self.last_motion_t = obs.t
        elif obs.events < cfg.still_events_per_frame:
            self.still_run += 1
            if self.motion_active and self.still_run >= cfg.still_frames:
                self.motion_active = False

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

    def _wrist_velocity(self, t: float, hand: HandObs) -> Optional[float]:
        if not hand.present or hand.wrist_y is None:
            self._wrist = None
            return None
        prev, self._wrist = self._wrist, (t, hand.wrist_y)
        if prev is None or t <= prev[0]:
            return None
        return (hand.wrist_y - prev[1]) / (t - prev[0])

    def _mp_stable(self) -> bool:
        return self.mp_gesture is not None and self.mp_count >= self.cfg.mp_stable_frames

    def _mp_eligible(self, t: float, events_in_hand: Optional[int]) -> bool:
        if not self._mp_stable():
            return False
        if not self.use_cnn:
            return True     # hand tracker only: it is the sole authority
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
            self.pump.reset()                  # keeps the learned rhythm
            self.voter.reset()
            self.human = None
            if state == ARMED or self.cfg.idle_action == "ready":
                return self._set_pose(READY)
        if state == SHOOT:
            self.shoot_t = t
            self.shoot_landed = False
        return None

    def _shoot_window(self) -> float:
        tempo = self.pump.tempo
        return max(self.cfg.shoot_window_s, 3.0 * tempo) if tempo else self.cfg.shoot_window_s

    def _countdown(self, t: float, event: Optional[str], vote: Optional[int]) -> Optional[str]:
        cfg = self.cfg
        if self.state == IDLE:
            if self._hand_present(t) or self.motion_active:
                return self._enter(ARMED, t, "hand in play zone")
            return None

        if self.state == ARMED:
            if self._idle(t):
                return self._enter(IDLE, t, "no hand")
            if event == "bottom":
                self.pumps += 1
                self.last_bottom_t = t
                self.reason = f"pump {self.pumps}"
                if self.pumps >= cfg.pumps_before_shoot:
                    self._enter(SHOOT, t, "throw")
            elif event == "landed" and self.pumps >= max(1, cfg.pumps_before_shoot - cfg.pump_miss_tolerance):
                self._enter(SHOOT, t, "throw landed")
                self.shoot_landed = True
                return self._shoot_decide(t, vote)
            return None

        if self.state == SHOOT:
            if event == "bottom":                  # still pumping: the throw has not come yet
                self.pumps += 1
                self.last_bottom_t = t
                self.shoot_t, self.shoot_landed = t, False
                self.reason = f"pump {self.pumps}"
                return None
            if event == "landed":
                self.shoot_landed = True
            if t - self.shoot_t > self._shoot_window():
                return self._enter(ARMED, t, "no throw seen")
            return self._shoot_decide(t, vote)

        if self.state == HOLD:
            held = t - self.commit_t
            if held >= cfg.hold_max_s:
                return self._enter(ARMED, t, "hold expired")
            if held >= cfg.hold_min_s:
                if self._idle(t):
                    return self._enter(IDLE, t, "hand left")
                if event == "bottom":
                    out = self._enter(ARMED, t, "new round")
                    self.pumps = 1
                    self.last_bottom_t = t
                    return out
        return None

    def _shoot_decide(self, t: float, vote: Optional[int]) -> Optional[str]:
        """Commits the throw as soon as the evidence allows it."""
        if vote in (PAPER, SCISSORS):
            return self._decided(vote, "cnn", t)
        if self.use_mp and self._mp_stable() and self.mp_gesture in (PAPER, SCISSORS):
            return self._decided(self.mp_gesture, "mp", t)
        since = t - self.shoot_t
        fallback = since >= self.cfg.rock_settle_fallback_s and not self.motion_active
        if not (self.shoot_landed or fallback) or since < 0:
            return None
        stroke_start = self.last_bottom_t            # the final stroke began after the last pump
        if self.use_cnn and self.last_vote in (PAPER, SCISSORS) and self.last_vote_t > stroke_start:
            return self._decided(self.last_vote, "cnn", t)
        if self.use_mp and self.mp_gesture is not None:
            return self._decided(self.mp_gesture, "mp", t)
        if self.use_cnn and self.cnn_rock_t > stroke_start:
            return self._decided(ROCK, "cnn", t)
        return None

    def _decided(self, gesture: int, source: str, t: float) -> Optional[str]:
        out = self._commit(gesture, source, t, force=True)
        self.state = HOLD
        self.reason = f"{GESTURE_NAME[gesture]} via {source}"
        return out
