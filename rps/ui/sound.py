"""
The beat guide's Qt playback: BeatSounds (one preloaded effect per beat kind) and BeatPlayer, which
plays a rps.game.BeatSchedule on time and reports each beat for the visual count. The samples
themselves come from rps.sound, which the command line plays without Qt.
"""

import os
import tempfile
import time

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal

from rps.sound import CUE_KINDS, RATE, SOUNDS, STALE_S, sounds, write_wav


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
