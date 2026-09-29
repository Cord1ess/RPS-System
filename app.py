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
"""

import argparse
import os
import sys

from PySide6.QtWidgets import QApplication

from rps.perf import boost_process
from rps.ui.common import AppState
from rps.ui.main_window import MainWindow
from rps.ui.style import apply_theme


def main():
    parser = argparse.ArgumentParser(description="RPS v3 desktop control centre")
    parser.add_argument("--config", default=None, help="Default: config.json in the project folder")
    parser.add_argument("--data-root", default=None, help="Folder holding recordings/ and frames/ "
                                                          "(default: data in the project folder)")
    parser.add_argument("--mock", action="store_true", help="Use the synthetic camera by default")
    parser.add_argument("--page", default=None, help="Open on this page, e.g. play, play-debug, setup, bot")
    parser.add_argument("--fullscreen", action="store_true", help="Fill the screen (for a stand-alone demo)")
    args = parser.parse_args()

    # Paths given on the command line are relative to where the app was started; everything else
    # (models/, config.json, data/, the background jobs) is relative to the project folder.
    config = os.path.abspath(args.config) if args.config else "config.json"
    data_root = os.path.abspath(args.data_root) if args.data_root else "data"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    boost_process()
    app = QApplication(sys.argv)
    app.setApplicationName("RPS v3")
    apply_theme(app)
    window = MainWindow(AppState(config, data_root), start_kind="mock" if args.mock else "camera")
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
