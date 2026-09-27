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
"""

import argparse
import sys

from PySide6.QtWidgets import QApplication

from rps.perf import boost_process
from rps.ui.common import AppState
from rps.ui.main_window import MainWindow
from rps.ui.style import apply_theme


def main():
    parser = argparse.ArgumentParser(description="RPS v3 desktop control centre")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-root", default="data", help="Folder holding recordings/ and frames/")
    parser.add_argument("--mock", action="store_true", help="Use the synthetic camera by default")
    args = parser.parse_args()

    boost_process()
    app = QApplication(sys.argv)
    app.setApplicationName("RPS v3")
    apply_theme(app)
    window = MainWindow(AppState(args.config, args.data_root), start_kind="mock" if args.mock else "camera")
    if args.mock:
        window.pages[0].source.setCurrentIndex(1)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
