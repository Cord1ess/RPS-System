"""Frame sources: driver repeats are skipped, and real-time sources are stamped on the camera clock."""

import csv
import json
import os
import time

import cv2
import numpy as np

from rps.camera import REPEAT_MAX_S, MockSource, VideoFileSource, is_repeat


def image(value: int, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(value)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)       # sensor-noise-like content


def test_a_repeat_is_an_identical_image_arriving_too_soon_for_a_new_exposure():
    a = image(1)
    assert is_repeat(a.copy(), 10.002, a, 10.0)                      # the driver's copy, 2 ms later
    assert not is_repeat(a.copy(), 10.0 + REPEAT_MAX_S + 0.013, a, 10.0)   # e.g. a covered lens at 30 fps
    assert not is_repeat(image(2), 10.002, a, 10.0)                   # a new image
    assert not is_repeat(a, 10.0, None, None)                         # the first frame


def write_recording(folder, frames):
    """A record_session.py-style recording as a PNG sequence: frames = [(t, image), ...]."""
    os.makedirs(os.path.join(folder, "frames"))
    for i, (_t, img) in enumerate(frames):
        cv2.imwrite(os.path.join(folder, "frames", f"{i:05d}.png"), img)
    with open(os.path.join(folder, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "t"])
        for i, (t, _img) in enumerate(frames):
            w.writerow([i, t])
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"video_file": "frames", "roi_in_crop": [0, 0, 32]}, f)


def test_replay_skips_the_driver_repeats_in_a_recording(tmp_path):
    a, b, c = image(1), image(2), image(3)
    # the pattern seen in real recordings: each image, then its copy ~2 ms later, then a ~60 ms gap
    write_recording(str(tmp_path), [(0.0, a), (0.002, a), (0.064, b), (0.066, b), (0.130, c)])
    src = VideoFileSource(str(tmp_path)).start()
    got = []
    while (frame := src.read()) is not None:
        got.append((frame.t, frame.bgr))
    assert [t for t, _ in got] == [0.0, 0.064, 0.130]                 # recorded times, offline
    assert all(np.array_equal(img, want) for (_t, img), want in zip(got, (a, b, c)))
    assert src.repeats == 2


def test_real_time_sources_are_stamped_on_the_camera_clock(tmp_path):
    write_recording(str(tmp_path), [(5.0, image(1)), (5.033, image(2))])
    src = VideoFileSource(str(tmp_path), realtime=True).start()
    first, second = src.read(), src.read()
    now = time.perf_counter()
    assert abs(second.t - now) < 0.05 and abs((second.t - first.t) - 0.033) < 1e-6
    mock = MockSource(realtime=True).start()
    assert abs(mock.read().t - time.perf_counter()) < 0.05
    assert MockSource(realtime=False).start().read().t == 0.0          # offline: scene time
