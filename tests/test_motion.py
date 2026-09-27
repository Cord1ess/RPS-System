import cv2
import numpy as np

from rps.motion import RhythmPumpDetector, VerticalMotion

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


def pump_velocity(n_pumps, period, throw=True, hold_s=0.8, amplitude=0.3, pause_frames=0):
    """Velocity samples (zone heights/s) for n pumps (down+up), then a throw that lands and holds."""
    out = []
    n = max(4, int(round(period * FPS)))
    half = n // 2
    speed = amplitude / (half / FPS)
    for _ in range(n_pumps):
        out += [speed] * half + [0.0] * pause_frames + [-speed] * (n - half)
    if throw:
        out += [speed] * half + [0.0] * int(hold_s * FPS)
    return out


def run(detector, samples):
    events = []
    for i, v in enumerate(samples):
        e = detector.update(i / FPS, v)
        if e:
            events.append((round(i / FPS, 3), e))
    return events


def test_three_pumps_then_throw_at_different_tempos():
    for period in (0.3, 0.45, 0.7):
        ev = run(RhythmPumpDetector(), pump_velocity(3, period))
        kinds = [e for _, e in ev]
        assert kinds == ["bottom", "bottom", "bottom", "landed"], (period, kinds)


def test_pause_at_the_bottom_is_still_a_pump():
    ev = run(RhythmPumpDetector(), pump_velocity(3, 0.5, pause_frames=2))
    assert [e for _, e in ev] == ["bottom", "bottom", "bottom", "landed"]


def test_tempo_is_learned():
    det = RhythmPumpDetector()
    run(det, pump_velocity(4, 0.5, throw=False))
    assert det.tempo is not None and abs(det.tempo - 0.5) < 0.07


def test_small_jitter_is_not_a_pump():
    rng = np.random.default_rng(1)
    ev = run(RhythmPumpDetector(), list(rng.normal(0, 0.1, 90)))
    assert ev == []
