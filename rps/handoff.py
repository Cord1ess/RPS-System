"""
Robot handover between computers (e.g. the Pi and a laptop) running this app on the same network.

The robot's firmware takes commands from any computer, so two running games would fight over the hand.
When one copy of the app starts driving a real robot, it broadcasts on the local network

    RPS-CTRL:TAKE:<instance id>:<computer name>:<robot address>

and every other copy that is driving the same robot stops (or, with `handoff` off, just notes it).
Starting on the other computer is all it takes to switch; the robot needs no change.

UDP broadcast on port 4299 of the local network only. Windows asks once whether Python may receive on
"private networks": allow it, or this computer cannot be told to hand over (it can still take over).
"""

import socket
import threading
import uuid
from typing import Callable, Optional

PORT = 4299
PREFIX = "RPS-CTRL:TAKE:"


def encode_take(instance: str, computer: str, robot: str) -> bytes:
    return f"{PREFIX}{instance}:{computer}:{robot}".encode("utf-8")


def parse_take(data: bytes) -> Optional[tuple]:
    """(instance, computer, robot address) of a takeover message, or None."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.startswith(PREFIX):
        return None
    parts = text[len(PREFIX):].split(":", 2)
    return tuple(parts) if len(parts) == 3 and all(parts) else None


class Handoff:
    """Listens for other copies of the app taking a robot, and announces this one taking it."""

    def __init__(self, on_taken: Callable[[str, str], None], port: int = PORT,
                 targets: Optional[list] = None, listen: bool = True):
        self.on_taken = on_taken                    # (computer name, robot address), from the listener thread
        self.port = port
        self.instance = uuid.uuid4().hex[:12]
        self.computer = socket.gethostname()
        self.targets = targets or [("255.255.255.255", port)]
        self.listening = False
        self._running = False
        self._sock: Optional[socket.socket] = None
        if listen:
            self._start_listener()

    def _start_listener(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # Pi and laptop copies, Play and Debug
            sock.bind(("", self.port))
            sock.settimeout(0.5)
        except OSError:
            return                                   # port taken: this copy can still announce
        self._sock, self._running, self.listening = sock, True, True
        threading.Thread(target=self._loop, name="handoff", daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                if not self._running:
                    return
                continue
            msg = parse_take(data)
            if msg and msg[0] != self.instance:
                try:
                    self.on_taken(msg[1], msg[2])
                except RuntimeError:                 # the window is closing
                    pass

    def announce(self, robot: str) -> bool:
        """Tells the other copies on this network that this one now drives `robot`. True if sent."""
        sent = False
        targets = list(self.targets)
        parts = robot.split(".")
        if len(parts) == 4 and all(p.isdigit() for p in parts) and self.targets[0][0] == "255.255.255.255":
            targets.append((".".join(parts[:3] + ["255"]), self.port))   # the robot's own /24, whatever the route
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            for target in targets:
                try:
                    sock.sendto(encode_take(self.instance, self.computer, robot), target)
                    sent = True
                except OSError:
                    pass
        finally:
            sock.close()
        return sent

    def stop(self):
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
