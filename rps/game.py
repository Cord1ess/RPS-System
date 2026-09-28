"""
Timing of the beat-guided game (the optional audio and visual guide).

Everything is arithmetic on one start time, so the camera thread (which decides throws) and the
UI thread (which plays the sounds and shows the count) agree on every beat without messaging.

Timeline, in beats of beat_s seconds:
    lead_beats soft beats (get ready)
    then, for each round:
        `pumps` accented beats, shown as 3, 2, 1: pump on these
        1 throw beat, shown as SHOOT: throw on this one
        gap_beats soft beats while the result shows
A soft beat keeps playing throughout so the player feels the tempo.
"""

import math
from dataclasses import dataclass

ENDLESS_ROUNDS = 100000     # an endless match (game.rounds = 0) plays until Stop: over 3 days at 150 bpm


@dataclass
class Beat:
    index: int          # beats since the start
    t: float            # when it is played (perf_counter clock)
    kind: str           # "soft" | "pump" | "shoot"
    round: int          # the round it belongs to (-1 before the first); gap beats belong to the round just played
    label: str          # "", "3", "2", "1" or "SHOOT"


class BeatSchedule:
    def __init__(self, t0: float, beat_s: float, pumps: int, rounds: int, lead_beats: int = 4,
                 gap_beats: int = 4):
        if beat_s <= 0 or pumps < 1 or rounds < 1:
            raise ValueError("beat_s > 0, pumps >= 1 and rounds >= 1 are required")
        self.t0, self.beat_s, self.pumps, self.rounds = t0, beat_s, pumps, rounds
        self.lead_beats, self.gap_beats = lead_beats, gap_beats

    @property
    def round_beats(self) -> int:
        return self.pumps + 1 + self.gap_beats

    @property
    def total_beats(self) -> int:
        return self.lead_beats + self.rounds * self.round_beats

    @property
    def end_time(self) -> float:
        return self.t0 + self.total_beats * self.beat_s

    def first_pump_beat(self, n: int) -> int:
        return self.lead_beats + n * self.round_beats

    def shoot_time(self, n: int) -> float:
        return self.t0 + (self.first_pump_beat(n) + self.pumps) * self.beat_s

    def beat(self, k: int) -> Beat:
        t = self.t0 + k * self.beat_s
        if k < self.lead_beats:
            return Beat(k, t, "soft", -1, "")
        n, pos = divmod(k - self.lead_beats, self.round_beats)
        if n >= self.rounds:
            return Beat(k, t, "soft", self.rounds, "")
        if pos < self.pumps:
            return Beat(k, t, "pump", n, str(self.pumps - pos))
        if pos == self.pumps:
            return Beat(k, t, "shoot", n, "SHOOT")
        return Beat(k, t, "soft", n, "")

    def beat_index_at(self, t: float) -> int:
        return math.floor((t - self.t0) / self.beat_s)

    def round_at(self, t: float) -> int:
        """
        The round being played at time t: from half a beat before its first pump beat until half a
        beat before the next round's. -1 before the first round; `rounds` once the match is over.
        """
        k = (t - self.t0) / self.beat_s + 0.5 - self.lead_beats
        if k < 0:
            return -1
        return min(int(k // self.round_beats), self.rounds)
