"""Tab base class: camera tabs supply processor() (camera thread) and on_frame() (UI thread)."""

from PySide6.QtWidgets import QWidget


class Tab(QWidget):
    title = "Tab"
    uses_camera = False

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.state = main.state

    def processor(self, frame, src) -> dict:
        """Runs in the camera thread. Must not touch Qt widgets."""
        return {"display": frame.bgr}

    def on_frame(self, payload: dict):
        """Runs in the UI thread with the newest processor result."""

    def on_activated(self):
        pass

    def on_deactivated(self):
        pass

    def shutdown(self):
        pass
