"""
RPS v3 perception package: webcam DVS emulation, RoshamboNet inference,
MediaPipe fallback, decision engine, and ESP32 robot link.
"""

import os

# Windows Media Foundation: its hardware colour conversion makes opening the webcam about three times
# slower (4.2 s -> 1.4 s to the first image on the demo laptop, mostly in applying the format). OpenCV
# reads this once, when cv2 is imported, so it must be set before any `import cv2`. No effect elsewhere.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
