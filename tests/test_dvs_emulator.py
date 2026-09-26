import numpy as np
import cv2

from rps.config import DvsConfig
from rps.dvs_emulator import PseudoDVS, events_in_box


def textured_background(seed=0, size=128):
    rng = np.random.default_rng(seed)
    bg = rng.integers(60, 120, (size // 8, size // 8), dtype=np.uint8)
    return cv2.resize(bg, (size, size), interpolation=cv2.INTER_NEAREST)


def scene_with_blob(bg, cx, cy, r=18, level=210):
    img = bg.copy()
    cv2.circle(img, (int(cx), int(cy)), r, level, -1)
    return img


def run(dvs, frames, fps=30.0):
    out = []
    for i, img in enumerate(frames):
        emitted, stats = dvs.process(img, i / fps)
        out.append((emitted, stats))
    return out


def test_still_scene_emits_nothing():
    bg = textured_background()
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(90):   # 3 s of a static scene with sensor noise
        noise = rng.normal(0, 1.5, bg.shape)
        frames.append(np.clip(bg + noise, 0, 255).astype(np.uint8))
    results = run(PseudoDVS(DvsConfig()), frames)
    assert all(e is None for e, _ in results)
    assert sum(s.events for _, s in results) < 50


def test_moving_blob_emits_constant_count_frames():
    cfg = DvsConfig(event_count=400)
    bg = textured_background()
    frames = [scene_with_blob(bg, 30 + 3 * i, 64) for i in range(25)]
    results = run(PseudoDVS(cfg), frames)
    emitted = [e for e, _ in results if e is not None]
    assert len(emitted) >= 3
    for e in emitted:
        assert e.image.shape == (64, 64) and e.image.dtype == np.uint8
        # Dextra normalization: counts clipped at K=16 then scaled by 255/16 -> multiples of 15
        levels = np.unique(e.image)
        assert set((levels.astype(int) * 16 // 255).tolist()) <= set(range(17))
        assert e.image.max() > 0


def test_binomial_thinning_keeps_integer_counts_near_n():
    cfg = DvsConfig(event_count=300)
    dvs = PseudoDVS(cfg, seed=3)
    bg = textured_background()
    dvs.process(bg, 0.0)
    emitted, stats = dvs.process(scene_with_blob(bg, 64, 64, r=30), 1 / 30)
    assert stats.events > cfg.event_count          # one big frame overshoots N
    assert emitted is not None and emitted.n_events == stats.events
    assert emitted.counts.dtype.kind == "i"
    assert abs(int(emitted.counts.sum()) - cfg.event_count) < 0.15 * cfg.event_count


def test_global_brightness_ramp_is_absorbed():
    bg = textured_background().astype(np.float32)
    frames = [np.clip(bg * (1.0 + 0.03 * i), 0, 255).astype(np.uint8) for i in range(20)]
    results = run(PseudoDVS(DvsConfig()), frames)
    assert all(e is None for e, _ in results)
    assert sum(s.events for _, s in results) < 100


def test_motion_stop_flushes_partial_frame():
    # N too large to ever fill, so the only way out is the motion-stop flush
    cfg = DvsConfig(event_count=100_000, flush_min_fraction=0.001, max_accumulation_s=10.0)
    bg = textured_background()
    frames = [scene_with_blob(bg, 30 + 4 * i, 64) for i in range(6)]
    frames += [frames[-1]] * 5    # hand stops
    results = run(PseudoDVS(cfg), frames)
    flushed = [e for e, _ in results if e is not None and e.flushed]
    assert len(flushed) == 1


def test_events_in_box():
    m = np.zeros((64, 64), np.int32)
    m[10:20, 10:20] = 1
    assert events_in_box(m, None) == 100
    assert events_in_box(m, (0.0, 0.0, 0.5, 0.5)) == 100
    assert events_in_box(m, (0.5, 0.5, 1.0, 1.0)) == 0
