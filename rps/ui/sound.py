"""
Beat guide sounds. They are made here and written to temporary WAV files, so nothing is shipped.
Every sound has most of its loudness above 250 Hz, where laptop speakers play (a pure bass drum
would be silent on them). Three sets, each with:
    soft   the steady beat, on every beat: keeps the tempo (its own volume)
    pump   the count (3, 2, 1): pump on these; the same hit, harder (count volume)
    shoot  a double hit: throw now (count volume)
BeatPlayer plays a rps.game.BeatSchedule on time and reports each beat for the visual count.
"""

import os
import tempfile
import time
import wave

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal

RATE = 44100
STALE_S = 0.1           # a beat the UI reached this late is shown but not played
SOUNDS = {"drum": "Drum", "wood": "Wood block", "beep": "Beep"}
CUE_KINDS = ("pump", "shoot")


def _decay(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / RATE / tau)


def _tone(freq: float, dur: float, tau: float, drop_to: float = None, drop_tau: float = 0.035) -> np.ndarray:
    """A decaying sine; with drop_to its pitch falls from freq to drop_to, like a drum skin."""
    n = int(dur * RATE)
    f = np.full(n, float(freq))
    if drop_to is not None:
        f = drop_to + (freq - drop_to) * np.exp(-np.arange(n) / RATE / drop_tau)
    return np.sin(2 * np.pi * np.cumsum(f) / RATE) * _decay(n, tau)


def _click(ms: float, n_total: int, seed: int = 0) -> np.ndarray:
    """A sharp tick (differenced noise, i.e. high-passed) at the start: the stick or beater."""
    n = int(ms / 1000 * RATE)
    tick = np.diff(np.random.default_rng(seed).uniform(-1, 1, n + 1)) * np.linspace(1, 0, n)
    out = np.zeros(n_total)
    out[:n] = tick
    return out


def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    return peak * x / np.max(np.abs(x))


def drum(accent: bool) -> np.ndarray:
    dur = 0.22
    n = int(dur * RATE)
    body = np.tanh(5.0 * _tone(170, dur, 0.09, drop_to=65))              # kick; saturation adds overtones
    knock = _tone(300, dur, 0.06) + 0.5 * _tone(510, dur, 0.035) + 0.25 * _tone(810, dur, 0.02)   # floor-tom thud
    return _norm(0.5 * body + knock + _click(5, n) * (0.9 if accent else 0.5))


def wood(freq: float) -> np.ndarray:
    dur = 0.12
    return _norm(_tone(freq, dur, 0.03) + 0.5 * _tone(freq * 2.65, dur, 0.012) + 0.4 * _click(2, int(dur * RATE), 3))


def beep(freq: float, dur: float = 0.07) -> np.ndarray:
    n = int(dur * RATE)
    i = np.arange(n)
    ramp = np.minimum(1, i / (0.004 * RATE)) * np.minimum(1, (n - i) / (0.015 * RATE))    # no clicks at the ends
    return _norm(np.sin(2 * np.pi * freq * i / RATE) * ramp)


def _double(hit: np.ndarray, gap_s: float = 0.07, crack: bool = False) -> np.ndarray:
    x = np.concatenate([hit, np.zeros(int(gap_s * RATE)), hit])
    if crack:                                                          # a snare-like crack on the first hit
        m = int(0.12 * RATE)
        x[:m] += 0.5 * np.random.default_rng(1).uniform(-1, 1, m) * _decay(m, 0.03)
    return _norm(x, 0.95)


def sounds(sound: str = "drum") -> dict:
    """Peak-normalised samples per beat kind; loudness comes from the volumes."""
    if sound == "wood":
        return {"soft": wood(900), "pump": wood(900), "shoot": _double(wood(1300))}
    if sound == "beep":
        return {"soft": beep(880), "pump": beep(880), "shoot": _double(beep(1320, 0.06), 0.05)}
    return {"soft": drum(False), "pump": drum(True), "shoot": _double(drum(True), crack=True)}


def write_wav(path: str, samples: np.ndarray):
    data = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())


class BeatSounds:
    """One preloaded sound effect per beat kind, for one sound set."""

    def __init__(self, sound: str):
        from PySide6.QtMultimedia import QSoundEffect
        folder = os.path.join(tempfile.gettempdir(), "rps_beat_sounds")
        os.makedirs(folder, exist_ok=True)
        self.effects = {}
        for kind, samples in sounds(sound).items():
            path = os.path.join(folder, f"{sound}_{kind}.wav")
            write_wav(path, samples)
            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(path))
            self.effects[kind] = effect

    def play(self, kind: str, volume: float):
        effect = self.effects.get(kind)
        if effect is not None and volume > 0:
            effect.setVolume(float(min(max(volume, 0.0), 1.0)))
            effect.play()


class BeatPlayer(QObject):
    """Plays a BeatSchedule; emits every beat (for the on-screen count) as it is played. The volumes
    can change while it plays."""
    beat = Signal(object)            # rps.game.Beat
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(4)
        self.timer.timeout.connect(self._tick)
        self._sets = {}                  # sound -> BeatSounds, or False without audio support
        self._sounds = None
        self.schedule = None
        self.beat_volume, self.cue_volume = 0.6, 1.0
        self.next_k = 0
        self.played = 0                  # sounds started (for tests and diagnostics)

    @property
    def running(self) -> bool:
        return self.schedule is not None

    def start(self, schedule, sound: str = "drum", beat_volume: float = 0.6, cue_volume: float = 1.0):
        if sound not in self._sets:
            try:
                self._sets[sound] = BeatSounds(sound)
            except Exception:                      # no audio support: the count still shows on screen
                self._sets[sound] = False
        self._sounds = self._sets[sound]
        self.schedule, self.next_k, self.played = schedule, 0, 0
        self.set_volumes(beat_volume, cue_volume)
        self.timer.start()

    def set_volumes(self, beat_volume: float, cue_volume: float):
        self.beat_volume, self.cue_volume = beat_volume, cue_volume

    def stop(self):
        self.timer.stop()
        self.schedule = None

    def _tick(self):
        s = self.schedule
        if s is None:
            return
        now = time.perf_counter()
        while self.next_k < s.total_beats and s.beat(self.next_k).t <= now:
            b = s.beat(self.next_k)
            self.next_k += 1
            if self._sounds and now - b.t < STALE_S:
                self._sounds.play(b.kind, self.cue_volume if b.kind in CUE_KINDS else self.beat_volume)
                self.played += 1
            self.beat.emit(b)
        if self.next_k >= s.total_beats and now >= s.end_time:
            self.stop()
            self.finished.emit()
