"""
UDP link to the ESP32 hand controller.

Protocol (ASCII, one message per datagram, default port 4210):
    PC  -> ESP  P,<seq>,<pose>,<pc_ms>        pose in {R, P, S, N}; N = READY. The pose is what
                                              the ROBOT shows; counter logic stays on the PC.
    PC  -> ESP  L,<seq>,<0|1>                 LED off/on (camera-latency test)
    ESP -> PC   A,<seq>,<esp_ms>,<reset>      acknowledgement; reset = esp_reset_reason()

A new pose is sent immediately on change and re-sent every heartbeat_s (idempotent on the ESP,
which falls back to N after 2 s of silence). Acknowledgements give the round-trip time.
"""

import socket
import threading
import time
from collections import deque
from typing import Dict, Optional

POSES = ("R", "P", "S", "N")


def encode_pose(seq: int, pose: str, pc_ms: int) -> bytes:
    if pose not in POSES:
        raise ValueError(f"Invalid pose '{pose}'")
    return f"P,{seq},{pose},{pc_ms}".encode("ascii")


def encode_led(seq: int, on: bool) -> bytes:
    return f"L,{seq},{1 if on else 0}".encode("ascii")


def encode_ack(seq: int, esp_ms: int, reset_reason: int = 0) -> bytes:
    return f"A,{seq},{esp_ms},{reset_reason}".encode("ascii")


def parse_message(data: bytes) -> Optional[Dict]:
    """Parses any protocol message; returns None for malformed input."""
    try:
        parts = data.decode("ascii").strip().split(",")
        kind = parts[0]
        if kind == "P" and len(parts) == 4 and parts[2] in POSES:
            return {"type": "P", "seq": int(parts[1]), "pose": parts[2], "pc_ms": int(parts[3])}
        if kind == "L" and len(parts) == 3:
            return {"type": "L", "seq": int(parts[1]), "on": parts[2] == "1"}
        if kind == "A" and len(parts) >= 3:
            return {"type": "A", "seq": int(parts[1]), "esp_ms": int(parts[2]),
                    "reset": int(parts[3]) if len(parts) > 3 else 0}
    except (UnicodeDecodeError, ValueError):
        pass
    return None


class RobotLink:
    def __init__(self, host: str, port: int = 4210, heartbeat_s: float = 0.1, ack_timeout_s: float = 0.5):
        self.addr = (host, port)
        self.heartbeat_s = heartbeat_s
        self.ack_timeout_s = ack_timeout_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.settimeout(0.02)
        self._lock = threading.Lock()
        self._seq = 0
        self._pose = "N"
        self._pending: Dict[int, float] = {}
        self._t0 = time.perf_counter()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.rtts_ms = deque(maxlen=200)
        self.sent = 0
        self.acked = 0
        self.last_ack_t = 0.0
        self.last_reset_reason: Optional[int] = None

    def start(self) -> "RobotLink":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="robot_link", daemon=True)
        self._thread.start()
        print(f"[robot_link] Sending to {self.addr[0]}:{self.addr[1]} (heartbeat {self.heartbeat_s * 1000:.0f} ms)")
        return self

    def _now_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000) & 0x7FFFFFFF

    def _send(self, payload: bytes, seq: int):
        try:
            self.sock.sendto(payload, self.addr)
            self._pending[seq] = time.perf_counter()
            self.sent += 1
        except OSError as e:
            print(f"[robot_link] send failed: {e}")

    def send_pose(self, pose: str) -> int:
        """Sends immediately (called from the decision thread the moment a pose changes)."""
        with self._lock:
            self._pose = pose
            self._seq += 1
            seq = self._seq
            self._send(encode_pose(seq, pose, self._now_ms()), seq)
            self._next_heartbeat = time.perf_counter() + self.heartbeat_s
        return seq

    def send_led(self, on: bool) -> int:
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._send(encode_led(seq, on), seq)
        return seq

    @property
    def pose(self) -> str:
        return self._pose

    def _loop(self):
        self._next_heartbeat = time.perf_counter() + self.heartbeat_s
        while self._running:
            try:
                data, _ = self.sock.recvfrom(256)
                msg = parse_message(data)
                if msg and msg["type"] == "A":
                    with self._lock:
                        t_sent = self._pending.pop(msg["seq"], None)
                    if t_sent is not None:
                        self.rtts_ms.append((time.perf_counter() - t_sent) * 1000.0)
                    self.acked += 1
                    self.last_ack_t = time.perf_counter()
                    self.last_reset_reason = msg["reset"]
            except socket.timeout:
                pass
            except OSError:
                # Windows raises ConnectionResetError on ICMP port-unreachable; keep going
                pass
            now = time.perf_counter()
            with self._lock:
                if now >= self._next_heartbeat:
                    self._seq += 1
                    self._send(encode_pose(self._seq, self._pose, self._now_ms()), self._seq)
                    self._next_heartbeat = now + self.heartbeat_s
                stale = [s for s, ts in self._pending.items() if now - ts > self.ack_timeout_s]
                for s in stale:
                    del self._pending[s]

    def connected(self) -> bool:
        return self.acked > 0 and (time.perf_counter() - self.last_ack_t) < 1.0

    def stats(self) -> Dict:
        rtt = sorted(self.rtts_ms)
        return {
            "sent": self.sent,
            "acked": self.acked,
            "connected": self.connected(),
            "rtt_median_ms": rtt[len(rtt) // 2] if rtt else None,
            "rtt_p90_ms": rtt[int(len(rtt) * 0.9)] if rtt else None,
            "last_reset_reason": self.last_reset_reason,
        }

    def stop(self, send_ready: bool = True):
        if send_ready and self._running:
            self.send_pose("N")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self.sock.close()
        print(f"[robot_link] Closed. {self.stats()}")


class MockEsp:
    """In-process ESP32 stand-in: logs pose changes and acknowledges every message."""

    def __init__(self, port: int = 4210, host: str = "127.0.0.1", verbose: bool = True):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.1)
        self.verbose = verbose
        self.pose = "N"
        self.pose_log = []
        self._t0 = time.perf_counter()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "MockEsp":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="mock_esp", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            try:
                data, addr = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                continue
            msg = parse_message(data)
            if msg is None:
                continue
            esp_ms = int((time.perf_counter() - self._t0) * 1000)
            if msg["type"] == "P" and msg["pose"] != self.pose:
                self.pose = msg["pose"]
                self.pose_log.append((time.perf_counter(), self.pose))
                if self.verbose:
                    print(f"[mock_esp] t={esp_ms:7d} ms  seq={msg['seq']:6d}  POSE -> {self.pose}")
            elif msg["type"] == "L" and self.verbose:
                print(f"[mock_esp] t={esp_ms:7d} ms  LED {'ON' if msg['on'] else 'OFF'}")
            self.sock.sendto(encode_ack(msg["seq"], esp_ms, 0), addr)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self.sock.close()
