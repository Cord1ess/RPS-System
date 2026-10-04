"""
Interface size: one scale factor for every font, button, panel and graph (Qt's QT_SCALE_FACTOR), so the
app can be made smaller on a small or low-resolution screen and larger on a big one.

It is chosen per computer (QSettings, not config.json: the laptop and the Pi have different screens)
and applied when the app starts, so changing it restarts the app. "Auto" fits the window to the screen
it opens on: smaller when the screen is smaller than the layout's comfortable size, never larger.
"""

import os
from typing import Dict, Optional

PREF_KEY = "ui_scale"
CHOICES = [("auto", "Auto (fit this screen)"), ("0.6", "60%"), ("0.7", "70%"), ("0.8", "80%"), ("0.9", "90%"),
           ("1.0", "100%"), ("1.1", "110%"), ("1.25", "125%"), ("1.5", "150%")]
DESIGN_W, DESIGN_H = 1440, 900          # the window size the layout is comfortable at, at 100%
MIN_SCALE, MAX_SCALE = 0.5, 2.0
ENV, MARK = "QT_SCALE_FACTOR", "RPS_UI_SCALE"   # MARK: this process's scale was set by the app itself


def parse(choice) -> Optional[float]:
    """A numeric choice as a factor, or None for "auto" (or anything unreadable)."""
    try:
        value = float(str(choice).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if value > 3:                        # "80" meant 80%
        value /= 100.0
    return min(MAX_SCALE, max(MIN_SCALE, value))


def fit_scale(avail_w: int, avail_h: int) -> float:
    """The largest factor up to 100% at which the layout fits a screen area of this size (logical px)."""
    s = min(1.0, avail_w / DESIGN_W, avail_h / DESIGN_H)
    return max(MIN_SCALE, int(s * 20) / 20.0)      # rounded down to 5% steps


def before_qt(choice: str, environ: Dict[str, str] = os.environ) -> bool:
    """Called before QApplication is created. Sets the factor for a numeric choice and returns True when
    "auto" still has to measure the screen (after QApplication exists). A QT_SCALE_FACTOR set outside the
    app wins; one the app set itself (when it restarted for "auto") is kept."""
    if environ.get(MARK):
        return False                     # restarted with the factor already measured
    if environ.get(ENV):
        return False                     # set by the user outside the app
    value = parse(choice)
    if value is None:
        return True
    if abs(value - 1.0) > 1e-6:
        environ[ENV] = f"{value:g}"
        environ[MARK] = environ[ENV]
    return False


def current() -> float:
    try:
        return float(os.environ.get(ENV, "1") or 1)
    except ValueError:
        return 1.0


def clean_env(environ: Dict[str, str] = os.environ) -> Dict[str, str]:
    """The environment for a restart: the app's own scale removed, so the new process chooses again."""
    env = dict(environ)
    if env.pop(MARK, None) is not None:
        env.pop(ENV, None)
    return env
