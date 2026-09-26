from rps.voting import SequenceVote, TimedMajorityVote


def test_sequence_needs_k_in_a_row():
    v = SequenceVote(k=2, min_confidence=0.7, max_gap_s=0.25)
    assert v.update(1, 0.9, 0.00) is None
    assert v.update(1, 0.9, 0.03) == 1
    assert v.update(1, 0.9, 0.06) == 1


def test_sequence_breaks_on_label_change():
    v = SequenceVote(k=2)
    v.update(1, 0.9, 0.00)
    assert v.update(2, 0.9, 0.03) is None
    assert v.update(2, 0.9, 0.06) == 2


def test_sequence_breaks_on_low_confidence():
    v = SequenceVote(k=2, min_confidence=0.7)
    v.update(0, 0.9, 0.00)
    assert v.update(0, 0.5, 0.03) is None
    assert v.update(0, 0.9, 0.06) is None     # streak restarted
    assert v.update(0, 0.9, 0.09) == 0


def test_sequence_breaks_on_time_gap():
    v = SequenceVote(k=2, max_gap_s=0.25)
    v.update(2, 0.9, 0.00)
    assert v.update(2, 0.9, 0.50) is None      # a still hand in between: no stale streaks


def test_majority_resets_after_gap():
    v = TimedMajorityVote(window_length=5, min_confidence=0.5, max_gap_s=0.25)
    for i in range(3):
        v.update(0, 0.9, i * 0.03)
    assert v.update(0, 0.9, 0.09) == 0
    assert v.update(1, 0.9, 1.0) is None       # window cleared by the gap
