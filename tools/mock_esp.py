"""
Stand-alone mock ESP32: listens on UDP, logs pose changes, acknowledges every message.

Usage (in one terminal):   python tools/mock_esp.py
Then (in another):         python play.py --set robot.host=127.0.0.1
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rps.robot_link import MockEsp  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Mock ESP32 hand controller (UDP)")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    esp = MockEsp(port=args.port, host=args.host).start()
    print(f"[mock_esp] Listening on {args.host}:{args.port}. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        esp.stop()
        print(f"[mock_esp] Pose changes: {len(esp.pose_log)}")


if __name__ == "__main__":
    main()
