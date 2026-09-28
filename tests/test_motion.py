import os

import cv2
import numpy as np
from scipy.signal import find_peaks

from rps.motion import PumpDetector, VerticalMotion

FPS = 30.0


def blob_frame(cy, size=128):
    rng = np.random.default_rng(0)
    bg = cv2.resize(rng.integers(60, 120, (16, 16), dtype=np.uint8), (size, size),
                    interpolation=cv2.INTER_NEAREST)
    cv2.ellipse(bg, (size // 2, int(cy)), (22, 28), 0, 0, 360, 210, -1)
    return bg


def test_vertical_motion_sign_and_stillness():
    vm = VerticalMotion()
    assert vm.update(blob_frame(50), 0.0) is None
    down = [vm.update(blob_frame(50 + 4 * i), i / FPS) for i in range(1, 6)]
    assert all(v > 0.3 for v in down)                 # moving down, in play-zone heights per second
    up = [vm.update(blob_frame(70 - 4 * i), (6 + i) / FPS) for i in range(1, 6)]
    assert all(v < -0.3 for v in up[1:])
    still = [vm.update(blob_frame(50), (12 + i) / FPS) for i in range(3)]
    assert still[-1] == 0.0


def pump_heights(n_pumps, period, throw=True, hold_s=0.8, amplitude=0.3, top=0.3, pause_frames=0):
    """Hand height per frame (+ = down): n pumps (down then up), then a throw that lands and stays."""
    out = [top]
    n = max(4, int(round(period * FPS)))
    half = n // 2
    for _ in range(n_pumps):
        out += [top + amplitude * (k + 1) / half for k in range(half)]
        out += [top + amplitude] * pause_frames
        out += [top + amplitude * (1 - (k + 1) / (n - half)) for k in range(n - half)]
    if throw:
        out += [top + amplitude * (k + 1) / half for k in range(half)] + [top + amplitude] * int(hold_s * FPS)
    return out


def run(detector, heights, as_wrist=True):
    """Feeds heights either as the tracked wrist or as the matching vertical motion (no hand tracker)."""
    events, prev = [], heights[0]
    for i, y in enumerate(heights):
        vy = (y - prev) * FPS
        prev = y
        e = detector.update(i / FPS, vy, y if as_wrist else None)
        if e:
            events.append((round(i / FPS, 3), e))
    return events


def test_three_pumps_then_throw_at_different_tempos_with_and_without_the_wrist():
    for as_wrist in (True, False):
        for period in (0.3, 0.45, 0.7):
            kinds = [e for _, e in run(PumpDetector(), pump_heights(3, period), as_wrist)]
            assert kinds == ["bottom", "bottom", "bottom", "landed"], (as_wrist, period, kinds)


def test_small_pumps_are_counted_once_learned():
    # a player whose pumps are only 6% of the play zone: every pump still counts
    kinds = [e for _, e in run(PumpDetector(), pump_heights(6, 0.4, amplitude=0.06))]
    assert kinds.count("bottom") == 6


def test_pause_at_the_bottom_is_still_a_pump():
    ev = run(PumpDetector(), pump_heights(3, 0.5, pause_frames=2))
    assert [e for _, e in ev] == ["bottom", "bottom", "bottom", "landed"]


def test_tempo_and_size_are_learned():
    det = PumpDetector()
    run(det, pump_heights(4, 0.5, throw=False))
    assert det.tempo is not None and abs(det.tempo - 0.5) < 0.07
    assert abs(det.rise_threshold - 0.3 * 0.3) < 0.02      # a fraction of the 0.3 swings


def test_small_jitter_is_not_a_pump():
    rng = np.random.default_rng(1)
    ev = run(PumpDetector(), list(0.5 + rng.normal(0, 0.004, 90)))
    assert ev == []


def test_the_wrist_coming_back_after_a_gap_is_not_a_pump():
    det = PumpDetector()
    events = []
    for i in range(60):                                    # still hand, tracked
        events.append(det.update(i / FPS, 0.0, 0.6))
    for i in range(60, 75):                                # tracking lost, no motion
        events.append(det.update(i / FPS, 0.0, None))
    for i in range(75, 120):                               # found again, a little lower than before
        events.append(det.update(i / FPS, 0.0, 0.66))
    assert [e for e in events if e] == []


def test_real_recording_pumps():
    """Every pump of a real 45 s session (the wrist's low points that were followed by a rise)."""
    d = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "real_throws_scissors.npz"))
    m, h = d["motion"], d["hand"]
    t, vy = m[:, 0].astype(float), m[:, 3].astype(float)
    wy = np.where(h[:, 1] > 0, h[:, 4], np.nan).astype(float)
    ok = ~np.isnan(wy)
    low, _ = find_peaks(np.interp(t, t[ok], wy[ok]), prominence=0.04)
    det, found = PumpDetector(), []
    for i in range(len(t)):
        if det.update(t[i], None if np.isnan(vy[i]) else float(vy[i]),
                      None if np.isnan(wy[i]) else float(wy[i])) == "bottom":
            found.append(det.last_bottom_t)
    found = np.array(found)
    caught = sum(1 for b in low if np.min(np.abs(found - t[b])) <= 0.15)
    # the low points include the 28 throw landings; the game ignores those after a decision
    assert caught >= len(low) - 1
    assert 0.35 <= det.tempo <= 0.45
