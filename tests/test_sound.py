"""The beat guide's sounds must be heard on laptop speakers, which play almost nothing below ~250 Hz."""

import numpy as np
import pytest

from rps.sound import RATE, SOUNDS, available_backend, sounds


def share_above(x: np.ndarray, hz: float = 250.0) -> float:
    power = np.abs(np.fft.rfft(x)) ** 2
    return float(power[np.fft.rfftfreq(len(x), 1 / RATE) >= hz].sum() / power.sum())


@pytest.mark.parametrize("sound", list(SOUNDS))
def test_every_beat_is_audible_on_laptop_speakers(sound):
    s = sounds(sound)
    assert set(s) == {"soft", "pump", "shoot"}
    for kind, x in s.items():
        assert share_above(x) >= 0.3, (sound, kind)          # the first steady beat had 0.3 % there: silent
        assert 0.9 <= np.max(np.abs(x)) <= 1.0, (sound, kind)    # full scale: loudness comes from the volumes


def test_a_backend_is_found_and_a_missing_one_is_silent(monkeypatch):
    assert available_backend() in ("sounddevice", "winsound", "pw-play", "aplay", "paplay", "afplay", "none")
    monkeypatch.setattr("rps.sound._sounddevice", lambda: None)
    monkeypatch.setattr("rps.sound._winsound", lambda: None)
    monkeypatch.setattr("rps.sound._player_command", lambda: None)
    assert available_backend() == "none"
