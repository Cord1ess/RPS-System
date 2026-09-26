"""
Temporal vote filters that decide when a CNN prediction is trusted.

SequenceVote is Dextra's default (consumer.py PREDICTION_VOTING_METHOD="sequence"): k identical
predictions in a row. Here the streak is also time-bounded, because our frames arrive at 30 Hz
or less rather than Dextra's up to 500 Hz, so "k frames" must not silently mean hundreds of ms.
"""

from typing import Optional

from model import MajorityVote
from rps.config import VoteConfig


class SequenceVote:
    def __init__(self, k: int = 2, min_confidence: float = 0.7, max_gap_s: float = 0.25):
        self.k = k
        self.min_confidence = min_confidence
        self.max_gap_s = max_gap_s
        self.reset()

    def reset(self):
        self.label: Optional[int] = None
        self.count = 0
        self.last_t: Optional[float] = None

    def update(self, label: int, confidence: float, t: float) -> Optional[int]:
        """Returns the label once it has been predicted k times in a row, else None."""
        if confidence < self.min_confidence:
            self.reset()
            return None
        continuing = (self.label == label and self.last_t is not None
                      and (t - self.last_t) <= self.max_gap_s)
        self.count = self.count + 1 if continuing else 1
        self.label = label
        self.last_t = t
        return label if self.count >= self.k else None


class TimedMajorityVote:
    """model.MajorityVote with the same confidence gate and a time-based reset."""

    def __init__(self, window_length: int = 5, num_classes: int = 4,
                 min_confidence: float = 0.7, max_gap_s: float = 0.25):
        self.window_length = window_length
        self.num_classes = num_classes
        self.min_confidence = min_confidence
        self.max_gap_s = max_gap_s
        self.reset()

    def reset(self):
        self.voter = MajorityVote(self.window_length, self.num_classes)
        self.last_t: Optional[float] = None

    def update(self, label: int, confidence: float, t: float) -> Optional[int]:
        if self.last_t is not None and (t - self.last_t) > self.max_gap_s:
            self.reset()
        self.last_t = t
        if confidence < self.min_confidence:
            return self.voter.vote()
        return self.voter.new_prediction_and_vote(label)


def make_voter(cfg: VoteConfig, num_classes: int = 4):
    if cfg.method == "majority":
        return TimedMajorityVote(cfg.majority_window, num_classes, cfg.min_confidence, cfg.max_gap_s)
    return SequenceVote(cfg.k, cfg.min_confidence, cfg.max_gap_s)
