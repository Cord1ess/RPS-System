"""
Beat guide sounds, made here and written to temporary WAV files, so nothing is shipped. Every sound
has most of its energy above 250 Hz, where laptop and TV speakers play (a pure bass drum would be
silent on them). Three kinds, one per role in a round:
    soft   the steady beat, on every beat: keeps the tempo (its own volume)
    pump   the count (3, 2, 1): pump on these; the same hit, harder (count volume)
    shoot  a double hit: throw now (count volume)
The synth is plain numpy, so the desktop app (rps.ui.sound, Qt Multimedia) and the command line
(rps.cli) play exactly the same beats. Loudness comes from the volumes, not from the samples.
"""

import collections
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

RATE = 44100
STALE_S = 0.1           # a beat reached this late is shown (or printed) but not played
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
    """Peak-normalised samples per beat kind."""
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


def wav_folder(sound: str = "drum") -> str:
    """Where the WAVs live (%TEMP%/rps_beat_sounds); written on demand, never shipped."""
    folder = os.path.join(tempfile.gettempdir(), "rps_beat_sounds")
    os.makedirs(folder, exist_ok=True)
    for kind, samples in sounds(sound).items():
        write_wav(os.path.join(folder, f"{sound}_{kind}.wav"), samples)
    return folder


# ------------------------------------------------------------------ playback without Qt
# First backend that exists wins. sounddevice is the only one that can change the volume while the
# match runs, so it is worth installing (`pip install sounddevice`); the others play the files at
# whatever the system volume is. None means silence: the count is still shown on screen / printed.
def _sounddevice():
    try:
        import sounddevice
    except Exception:
        return None
    return sounddevice


def _winsound():
    try:
        import winsound
    except Exception:
        return None
    return winsound


def _player_command():
    """A command that plays a WAV file and returns, for the platforms without a Python module."""
    for name in ("pw-play", "aplay", "paplay", "afplay"):
        path = shutil.which(name)
        if path:
            return path
    return None


def available_backend() -> str:
    """Name of the audio backend that would be used, or "none"."""
    if _sounddevice() is not None:
        return "sounddevice"
    if _winsound() is not None:
        return "winsound"
    return os.path.basename(_player_command()) if _player_command() else "none"


class BeatAudio:
    """
    Plays one sound set (three WAVs) without Qt, for the command line. The desktop app uses Qt
    Multimedia instead (rps.ui.sound.BeatSounds); the samples come from here, so both sound the same.

    Beats can overlap (the double hit on SHOOT follows a count hit), so the sounddevice backend keeps
    one output stream running and feeds it from a queue rather than starting a stream per beat.
    """

    def __init__(self, sound: str = "drum"):
        self.sound = sound
        self.backend = available_backend()
        self.played = 0                       # beats handed to the backend (for tests and diagnostics)
        self._samples = sounds(sound)
        folder = wav_folder(sound)
        self.paths = {kind: os.path.join(folder, f"{sound}_{kind}.wav") for kind in self._samples}
        self._pending = collections.deque()   # [samples, offset] pairs: appended by play(), read by the callback
        self._stream = None
        self._winsound = None
        self._command = None
        if self.backend == "sounddevice":
            sd = _sounddevice()
            try:
                self._stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="float32",
                                               latency="low", callback=self._fill)
                self._stream.start()
            except Exception:                 # no output device, or it cannot be opened
                self._stream = None
                self.backend = "none"
        elif self.backend == "winsound":
            self._winsound = _winsound()
        elif self.backend != "none":
            self._command = _player_command()

    @property
    def volume_ok(self) -> bool:
        """True when this backend can change the volume while playing (sounddevice only)."""
        return self.backend == "sounddevice"

    def play(self, kind: str, volume: float) -> bool:
        """Plays one beat at `volume` (0-1). False when there is no audio, or the volume is zero."""
        if self.backend == "none" or kind not in self.paths or volume <= 0:
            return False
        if self.backend == "sounddevice":
            samples = self._samples[kind] * float(min(volume, 1.0))
            self._pending.append([np.asarray(samples, dtype=np.float32), 0])
            self.played += 1
            return True
        if self.backend == "winsound":
            flags = self._winsound.SND_FILENAME | self._winsound.SND_ASYNC | self._winsound.SND_NODEFAULT
            self._winsound.PlaySound(self.paths[kind], flags)
        else:
            argv = [self._command, "-q", self.paths[kind]] if os.path.basename(self._command) == "aplay" \
                else [self._command, self.paths[kind]]
            subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.played += 1
        return True

    def _fill(self, outdata, frames, time_info, status):        # sounddevice callback, on its own thread
        out = outdata[:, 0] if getattr(outdata, "ndim", 1) > 1 else outdata
        filled = 0
        while filled < frames and self._pending:
            chunk, pos = self._pending[0]
            take = min(frames - filled, len(chunk) - pos)
            out[filled:filled + take] = chunk[pos:pos + take]
            filled += take
            if pos + take >= len(chunk):
                self._pending.popleft()
            else:
                self._pending[0][1] = pos + take
        if filled < frames:
            out[filled:] = 0.0

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
