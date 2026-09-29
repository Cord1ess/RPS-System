"""
Decision engine: fuses Dextra (the CNN on the motion image) and Mediapipe evidence and runs the game.

Pure logic, no I/O: the pipeline feeds it per-frame observations in a fixed order
(on_motion first, so Dextra is never delayed; then on_hand for the same frame).
Both calls return the new robot pose ("R", "P", "S" or "N" = ready) when it changes, else None.

Fusion rules ("Both")
- While the hand moves Dextra is the authority: a vote commit switches immediately.
- Mediapipe decides only when the hand is still (few events in its box) and after the last
  Dextra decision plus a margin, so a late, blurred mid-throw fist cannot overwrite a correct commit.
- A steady Mediapipe reading that contradicts a Dextra vote blocks it.
- Hysteresis (dead time) applies to Mediapipe-driven switches and to A->B->A flips only.
- A background vote never changes the command (Dextra behaviour).

What the robot plays (decision.robot_plays): "win" shows the move that beats the player's throw,
"draw" copies it, "lose" shows the move it beats. Every mode below works with each of them.

Every decision records when its throw was first seen on camera (Decision.t_first) and the frame it
was decided on (t_frame), so the app can show how long reading the hand took.

Game modes
- continuous: the robot always answers the current decision.
- countdown: IDLE -> ARMED (count pumps) -> SHOOT -> HOLD. Pumps come from rps.motion's
  PumpDetector, which learns the player's pump size and tempo. An open hand after enough pumps is
  the throw; rock (the pumping fist) commits once the throw has landed.
- guided: the beat guide (rps.game.BeatSchedule) sets when each round's throw is due. ARMED during
  the count-in, SHOOT in a window around the throw beat, HOLD after it; a window that closes
  without a decision is a missed round.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from model import COUNTER_MOVES
from rps.config import DecisionConfig, VoteConfig
from rps.hand_tracker import BACKGROUND, PAPER, ROCK, SCISSORS, HandObs
from rps.motion import PumpDetector
from rps.voting import make_voter

READY = "N"
POSE_LETTER = {"rock": "R", "paper": "P", "scissors": "S"}
GESTURE_NAME = {ROCK: "rock", PAPER: "paper", SCISSORS: "scissors", BACKGROUND: "background", -1: "unknown"}
COUNTER_POSE = {g: POSE_LETTER[COUNTER_MOVES[g]] for g in (ROCK, PAPER, SCISSORS)}      # beats the throw
SAME_POSE = {g: POSE_LETTER[GESTURE_NAME[g]] for g in (ROCK, PAPER, SCISSORS)}           # copies it
LOSING_POSE = {ROCK: "S", PAPER: "R", SCISSORS: "P"}                                     # what the throw beats
ROBOT_POSE = {"win": COUNTER_POSE, "draw": SAME_POSE, "lose": LOSING_POSE}

IDLE, ARMED, SHOOT, HOLD = "IDLE", "ARMED", "SHOOT", "HOLD"
ROBOT, YOU, DRAW = "robot", "you", "draw"
TRACKER_FRESH_S = 0.1    # a Mediapipe reading older than this cannot block a Dextra vote
SETTLE_MIN_S = 0.45      # after a decision, low points this soon (or within 1.2 beats) are the throw settling


def outcome(gesture: int, pose: str) -> str:
    """Who won a round: ROBOT, YOU or DRAW, from the player's throw and the robot's move."""
    if pose == COUNTER_POSE[gesture]:
        return ROBOT
    return DRAW if pose == SAME_POSE[gesture] else YOU


@dataclass
class Decision:
    """One decision and when its evidence appeared, on the camera clock."""
    gesture: int
    pose: str                                   # the robot's move for it
    source: str                                 # "cnn" (Dextra) | "mp" (Mediapipe)
    t_first: float                              # first camera frame that showed the throw (rock: when it landed)
    t_frame: float                              # the camera frame it was decided on


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
    round: int = -1                             # guided: the round being played
    round_results: Optional[dict] = None        # guided: round -> gesture, or None when no throw was seen
    misses: int = 0                             # countdown: rounds that ended without a decision
    decision: Optional[Decision] = None         # the latest decision


class DecisionEngine:
    def __init__(self, cfg: DecisionConfig, vote_cfg: VoteConfig, use_cnn: bool, use_mp: bool):
        self.cfg = cfg
        self.use_cnn = use_cnn
        self.use_mp = use_mp
        self.voter = make_voter(vote_cfg)
        self.pump = PumpDetector(cfg.pump_min_rise, min_period_s=cfg.pump_min_period_s)
        self._vy: Optional[float] = None        # this frame's vertical motion, for the pump detector
        self.mode = cfg.mode
        # guided mode (beat guide): rps.game.BeatSchedule, and the delay from a beat being played to
        # the player's throw on it reaching a camera frame (sound output + camera)
        self.schedule = None
        self.sync_offset_s = 0.0
        self.round = -1
        self.round_results = {}                 # round -> gesture decided, or None when no throw was seen
        self.window_open_t = -1e9
        self.misses = 0                         # countdown rounds that ended without a decision
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
        self.mp_since_t = self.mp_last_t = -1e9      # first and latest image of the current Mediapipe gesture
        self.cnn_run_label: Optional[int] = None
        self.cnn_since = {}                     # label -> first frame of Dextra's latest run of that answer
        self.throw_t: Optional[float] = None    # when the throw landed (the hand stopped at its low point)
        self.last_decision: Optional[Decision] = None
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

    def start_guided(self, schedule, sync_offset_s: float = 0.0) -> Optional[str]:
        """Plays rounds on the beats of `schedule` (rps.game.BeatSchedule)."""
        self.schedule, self.sync_offset_s = schedule, sync_offset_s
        self.round, self.round_results = -1, {}
        return self.set_mode("guided")

    def snapshot(self) -> Snapshot:
        return Snapshot(self.mode, "LIVE" if self.mode == "continuous" else self.state, self.pose, self.human,
                        self.last_source, self.motion_active, self.pumps, self.switches, self.commits,
                        self.cnn_vote, self.mp_gesture, self.reason, self.pump.tempo, self.round,
                        dict(self.round_results), self.misses, self.last_decision)

    def on_motion(self, obs: MotionObs) -> Optional[str]:
        t = obs.t
        self._vy = obs.vy
        self._update_motion(obs)
        vote = None
        if obs.cnn is not None and self.use_cnn:
            if obs.cnn[0] != self.cnn_run_label:          # Dextra starts seeing something new
                self.cnn_run_label = obs.cnn[0]
                self.cnn_since[obs.cnn[0]] = t
            vote = self.voter.update(obs.cnn[0], obs.cnn[1], t)
            if vote is not None and self._tracker_contradicts(vote, t):
                vote = None
            self.cnn_vote = vote
            if vote in (ROCK, PAPER, SCISSORS):
                self.last_vote, self.last_vote_t = vote, t
            if vote == ROCK:
                self.cnn_rock_t = t
        if self.mode == "continuous":
            if vote in (ROCK, PAPER, SCISSORS):
                return self._commit(vote, "cnn", t)
            return self._continuous_idle(t)
        # with Mediapipe the pump detector runs in on_hand, where this frame's wrist height is known
        event = None if self.use_mp else self.pump.update(t, obs.vy)
        return self._rounds(t, event, vote)

    def on_hand(self, t: float, hand: Optional[HandObs], events_in_hand: Optional[int]) -> Optional[str]:
        if hand is None or not self.use_mp:
            return None
        if hand.skipped:                        # Mediapipe skipped this frame: motion alone carries the pumps
            if self.mode == "continuous":
                return None
            return self._rounds(t, self.pump.update(t, self._vy), None)
        self._update_mp(t, hand)
        eligible = self._mp_eligible(t, events_in_hand)
        if self.mode == "continuous":
            if eligible:
                return self._commit(self.mp_gesture, "mp", t)
            return self._continuous_idle(t)
        event = self.pump.update(t, self._vy, hand.wrist_y if hand.present else None)
        out = self._rounds(t, event, None)
        if self.mode != "countdown":
            return out
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
            if g != self.mp_gesture:
                self.mp_since_t = t
            self.mp_gesture, self.mp_last_t = g, t
        else:
            self.mp_gesture = None

    def _mp_stable(self) -> bool:
        """The gesture held for mp_stable_ms: a time, not a frame count, so it means the same at any frame rate."""
        return self.mp_gesture is not None and \
            self.mp_last_t - self.mp_since_t >= self.cfg.mp_stable_ms / 1000.0 - 1e-6

    def _tracker_contradicts(self, vote: int, t: float) -> bool:
        """
        A steady, current hand-tracker reading that disagrees blocks a motion-model vote: on a webcam
        the tracker is far more accurate per frame. In countdown the pumping fist is expected to
        open, so there only a paper/scissors reading can block, and the motion model keeps its head
        start on the throw.
        """
        if not self.use_mp or vote not in (ROCK, PAPER, SCISSORS) or not self._mp_stable():
            return False
        if t - self.last_hand_t > TRACKER_FRESH_S or self.mp_gesture == vote:
            return False
        return self.mode == "continuous" or self.mp_gesture in (PAPER, SCISSORS)

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
        pose = ROBOT_POSE.get(self.cfg.robot_plays, COUNTER_POSE)[gesture]
        self.last_decision = Decision(gesture, pose, source, self._first_seen(gesture, source, t), t)
        return self._set_pose(pose)

    def _first_seen(self, gesture: int, source: str, t: float) -> float:
        """The first camera frame on which the deciding reader showed this throw."""
        since = self.mp_since_t if source == "mp" else self.cnn_since.get(gesture, t)
        # In a round the throw cannot begin before its final stroke: a shape glimpsed during the pumps
        # (a reader briefly misreading the fist) is not the throw.
        if self.mode == "countdown":
            since = max(since, self.pump.last_bottom_t)       # the last pump's low point
        elif self.mode == "guided":
            since = max(since, self.window_open_t)
        if gesture == ROCK and self.mode != "continuous":
            # A fist is shown all through the pumps: the rock throw is there once the hand has landed
            # (or, if the landing was not seen, when the throw was due).
            throw = self.throw_t
            if throw is None:
                s = self.schedule
                if self.mode == "guided" and s is not None and 0 <= self.round < s.rounds:
                    throw = s.shoot_time(self.round) + self.sync_offset_s
                else:
                    throw = self.shoot_t
            since = max(since, throw)
        return min(since, t)

    def _continuous_idle(self, t: float) -> Optional[str]:
        if self.human is not None and self.cfg.idle_action == "ready" and self._idle(t):
            self.human = None
            self.reason = "idle"
            return self._set_pose(READY)
        return None

    # ---------------------------------------------------------------- countdown mode
    def _enter(self, state: str, t: float, reason: str, reset_pump: bool = True) -> Optional[str]:
        self.state = state
        self.reason = reason
        if state in (IDLE, ARMED):
            self.pumps = 0
            if reset_pump:
                self.pump.reset()              # keeps the learned rhythm
            self.voter.reset()
            self.human = None
            if state == ARMED or self.cfg.idle_action == "ready":
                return self._set_pose(READY)
        if state == SHOOT:
            self.shoot_t = t
            self.shoot_landed = False
            self.throw_t = None
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
            enough = self.pumps >= max(1, cfg.pumps_before_shoot - cfg.pump_miss_tolerance)
            if event == "bottom":
                self.pumps += 1
                self.last_bottom_t = t
                self.reason = f"pump {self.pumps}"
                if self.pumps >= cfg.pumps_before_shoot:
                    self._enter(SHOOT, t, "throw")
            elif event == "landed" and enough:
                self._enter(SHOOT, t, "throw landed")
                self.shoot_landed, self.throw_t = True, self.pump.ext_t
                return self._shoot_decide(t, vote)
            elif self.pumps >= max(1, cfg.pumps_before_shoot - cfg.pump_miss_tolerance - 1) \
                    and self._open_hand(vote):
                # Paper/scissors after pumping is the throw itself, whatever beat it came on (players
                # throw on the 3rd or the 4th down stroke). One pump fewer is enough here: the last
                # pump's low point is confirmed only once the hand rises again, and the fingers can
                # already be open by then.
                self._enter(SHOOT, t, "throw")
                return self._shoot_decide(t, vote)
            return None

        if self.state == SHOOT:
            if event == "bottom":                  # still pumping: the throw has not come yet
                self.pumps += 1
                self.last_bottom_t = t
                self.shoot_t, self.shoot_landed, self.throw_t = t, False, None
                self.reason = f"pump {self.pumps}"
                return None
            if event == "landed":
                self.shoot_landed, self.throw_t = True, self.pump.ext_t
            if t - self.shoot_t > self._shoot_window():
                self.misses += 1
                return self._enter(ARMED, t, "no throw seen")
            return self._shoot_decide(t, vote)

        if self.state == HOLD:
            held = t - self.commit_t
            if held >= cfg.hold_max_s:
                return self._enter(ARMED, t, "hold expired")
            if held < cfg.hold_min_s:              # settling after the throw is not part of the next round
                return None
            if self._idle(t):
                return self._enter(IDLE, t, "hand left")
            if event == "bottom":                  # the next round's first pump; the robot keeps its move until now
                out = self._enter(ARMED, t, "new round", reset_pump=False)   # this pump starts the rhythm
                self.pumps = 1
                self.last_bottom_t = t
                return out
        return None

    def _rounds(self, t: float, event: Optional[str], vote: Optional[int]) -> Optional[str]:
        return self._guided(t, event, vote) if self.mode == "guided" else self._countdown(t, event, vote)

    # ---------------------------------------------------------------- guided mode (beat guide)
    def _guided(self, t: float, event: Optional[str], vote: Optional[int]) -> Optional[str]:
        s, cfg = self.schedule, self.cfg
        if s is None:
            return None
        te = t - self.sync_offset_s               # the frame's time on the beat clock
        n = s.round_at(te)
        if n != self.round:                       # a new round's count-in begins: robot to ready
            if 0 <= self.round < s.rounds:
                self.round_results.setdefault(self.round, None)   # ended before its window closed: a miss
            self.round = n
            if not 0 <= n < s.rounds:
                self.state = IDLE if n < 0 else HOLD
                self.reason = "get ready" if n < 0 else "match over"
                return None
            self.state, self.pumps, self.human = ARMED, 0, None
            self.voter.reset()
            self.pump.reset()                     # keeps the learned rhythm
            self.reason = f"round {n + 1}"
            return self._set_pose(READY)
        if not 0 <= n < s.rounds:
            return None
        shoot = s.shoot_time(n)
        if self.state == ARMED:
            if event == "bottom":                 # shown to the player; the beat decides when to throw
                self.pumps += 1
                self.reason = f"pump {self.pumps}"
            if te < shoot - cfg.guided_early_s:
                return None
            self.state, self.shoot_landed, self.throw_t = SHOOT, False, None
            self.window_open_t = self.shoot_t = t
            self.reason = "throw"
        if self.state == SHOOT:
            if event == "landed":
                self.shoot_landed, self.throw_t = True, self.pump.ext_t
            if te > shoot + cfg.guided_late_s:    # the window closed without a readable throw
                self.state, self.reason = HOLD, "no throw seen"
                self.round_results[n] = None
                return None
            return self._guided_decide(t, te - shoot, vote)
        return None

    def _guided_decide(self, t: float, since_beat: float, vote: Optional[int]) -> Optional[str]:
        """Paper/scissors as soon as they show; rock once the throw had time to open and did not."""
        if vote in (PAPER, SCISSORS):
            return self._decided(vote, "cnn", t)
        if self.use_mp and self._mp_stable() and self.mp_gesture in (PAPER, SCISSORS):
            return self._decided(self.mp_gesture, "mp", t)
        if self.use_cnn and self.last_vote in (PAPER, SCISSORS) and self.last_vote_t >= self.window_open_t:
            return self._decided(self.last_vote, "cnn", t)
        if not (self.shoot_landed or since_beat >= self.cfg.guided_rock_after_s):
            return None
        if self.use_mp and self._mp_stable() and self.mp_gesture == ROCK:
            return self._decided(ROCK, "mp", t)
        if self.use_cnn and self.cnn_rock_t >= self.window_open_t:
            return self._decided(ROCK, "cnn", t)
        return None

    def _open_hand(self, vote: Optional[int]) -> bool:
        """Paper or scissors is being shown (a pumping fist never is)."""
        return vote in (PAPER, SCISSORS) or \
            (self.use_mp and self._mp_stable() and self.mp_gesture in (PAPER, SCISSORS))

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
        if self.mode == "guided":
            self.round_results[self.round] = gesture
        # The round is over. The hand is still sinking into the throw: that low point is not the next
        # round's first pump, and the pause after the throw is not a pump interval.
        tempo = self.pump.tempo
        self.pump.reset(ignore_until=t + max(SETTLE_MIN_S, 1.2 * tempo if tempo else 0.0))
        return out
