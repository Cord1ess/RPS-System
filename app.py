"""
RPS v3 desktop control centre.

Tabs: Setup (camera, play zone, ESP link) -> Record -> Dataset -> Train -> Evaluate -> Play,
plus Settings with every config field. The command-line tools remain available; the app runs
the same code (recording, pipeline) in-process and the heavy jobs (build, train, replay,
probe) as background processes with live output.

Usage:
    python app.py
    python app.py --mock            # synthetic camera (no webcam needed)
    python app.py --data-root D:/rps_data
    python app.py --fullscreen --page play   # a stand-alone demo (e.g. a Raspberry Pi at boot)
    python app.py --scale 80                 # interface size for this run (default: Settings > Interface size)
"""

import argparse
import os
import subprocess
import sys

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from rps.perf import boost_process
from rps.ui.common import AppState
from rps.ui import scale
from rps.ui.main_window import MainWindow
from rps.ui.style import apply_theme

APP = os.path.abspath(__file__)


def without_scale(argv):
    """The command-line arguments without --scale, so a restart uses the size chosen in Settings."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "--scale":
            skip = True
        elif not a.startswith("--scale="):
            out.append(a)
    return out


def relaunch(env) -> int:
    """Starts the app again with this environment (the interface size must be set before Qt starts)."""
    cmd = [sys.executable, APP] + sys.argv[1:]
    if sys.platform != "win32":
        os.execve(sys.executable, cmd, env)         # replaces this process
    return subprocess.call(cmd, env=env)            # Windows: the new run, then exit with its code


def main():
    parser = argparse.ArgumentParser(description="RPS v3 desktop control centre")
    parser.add_argument("--config", default=None, help="Default: config.json in the project folder")
    parser.add_argument("--data-root", default=None, help="Folder holding recordings/ and frames/ "
                                                          "(default: data in the project folder)")
    parser.add_argument("--mock", action="store_true", help="Use the synthetic camera by default")
    parser.add_argument("--page", default=None, help="Open on this page, e.g. play, play-debug, setup, bot")
    parser.add_argument("--fullscreen", action="store_true", help="Fill the screen (for a stand-alone demo)")
    parser.add_argument("--scale", default=None, help="Interface size, e.g. 0.8 or 80 (default: Settings > "
                                                      "Interface size, which starts as Auto)")
    args = parser.parse_args()

    # Paths given on the command line are relative to where the app was started; everything else
    # (models/, config.json, data/, the background jobs) is relative to the project folder.
    config = os.path.abspath(args.config) if args.config else "config.json"
    data_root = os.path.abspath(args.data_root) if args.data_root else "data"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    prefs = QSettings("RPS-System", "app")
    measure = scale.before_qt(args.scale or str(prefs.value(scale.PREF_KEY, "auto")))
    boost_process()
    app = QApplication(sys.argv)
    if measure:                                     # Auto: fit the screen this opens on
        area = app.primaryScreen().availableGeometry()
        fit = scale.fit_scale(area.width(), area.height())
        if fit < 1.0:
            env = dict(os.environ)
            env[scale.ENV] = env[scale.MARK] = f"{fit:g}"
            sys.exit(relaunch(env))
    app.setApplicationName("RPS v3")
    apply_theme(app)
    window = MainWindow(AppState(config, data_root), start_kind="mock" if args.mock else "camera")
    window.restart_command = [sys.executable, APP] + without_scale(sys.argv[1:])
    if args.mock:
        window.pages[0].source.setCurrentIndex(1)
    if args.page:
        window.open_page(args.page)
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
