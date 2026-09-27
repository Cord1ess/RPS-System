import numpy as np

from rps.hand_tracker import (PAPER, ROCK, SCISSORS, UNKNOWN, FINGER_CHAINS,
                              classify_curls, finger_curls)

EXT, CURL = 0.30, 0.45


def synthetic_hand(extended):
    """World landmarks (21, 3): straight fingers point +y; curled fingers fold 80 deg per joint."""
    pts = np.zeros((21, 3), np.float32)
    bases = [-0.02, 0.0, 0.02, 0.04]
    for f, chain in enumerate(FINGER_CHAINS):
        _, mcp, pip, dip, tip = chain
        pts[mcp] = (bases[f], 0.08, 0.0)
        bend = 0.0 if extended[f] else np.deg2rad(80)
        cur = pts[mcp].astype(np.float64)
        angle = 0.0
        for j in (pip, dip, tip):
            angle += bend
            d = np.array([0.0, np.cos(angle), -np.sin(angle)])
            cur = cur + 0.025 * d
            pts[j] = cur
    return pts


def classify(extended):
    curls = finger_curls(synthetic_hand(extended))
    gesture, conf, _ = classify_curls(curls, None, EXT, CURL)
    return gesture, conf, curls


def test_fist_is_rock():
    g, conf, curls = classify([False, False, False, False])
    assert g == ROCK and conf > 0.5 and curls.min() > CURL


def test_open_hand_is_paper():
    g, conf, curls = classify([True, True, True, True])
    assert g == PAPER and conf > 0.5 and curls.max() < EXT


def test_v_sign_is_scissors():
    g, _, _ = classify([True, True, False, False])
    assert g == SCISSORS


def test_two_odd_fingers_unknown():
    g, conf, _ = classify([True, False, False, True])
    assert g == UNKNOWN and conf == 0.0


def test_hysteresis_keeps_state_inside_band():
    curls = [0.40, 0.05, 0.9, 0.9]   # index inside the band
    _, _, prev = classify_curls([0.1, 0.05, 0.9, 0.9], None, EXT, CURL)
    g, _, ext = classify_curls(curls, prev, EXT, CURL)
    assert ext[0] is True and g == SCISSORS
    _, _, prev = classify_curls([0.9, 0.05, 0.9, 0.9], None, EXT, CURL)
    g, _, ext = classify_curls(curls, prev, EXT, CURL)
    assert ext[0] is False and g == ROCK


def test_sideways_scissors_with_hidden_middle_finger():
    # index straight, middle hidden behind it (read as slightly bent), ring + pinky bent
    g, _, _ = classify_curls([0.2, 0.5, 0.7, 0.7], None, 0.30, 0.45)
    assert g == SCISSORS
    # a fist stays rock even with a loose middle finger: the index is bent
    g, _, _ = classify_curls([0.7, 0.5, 0.7, 0.7], None, 0.30, 0.45)
    assert g == ROCK
    # a clearly bent middle finger with only the index out is not scissors
    g, _, _ = classify_curls([0.2, 0.8, 0.7, 0.7], None, 0.30, 0.45)
    assert g == ROCK
