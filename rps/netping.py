"""
Wi-Fi delay to the robot, measured with ping (ICMP echo). The ESP32's network stack answers pings
by itself, whatever its firmware does, so this works with the team firmware, which never replies
to moves. One-way delay is taken as half the round trip.

Windows: the IcmpSendEcho API (no administrator rights needed). Linux (Raspberry Pi): the system
ping command.
"""

import re
import socket
import subprocess
import sys
import time
from statistics import median
from typing import List, Optional

SLOW_RTT_MS = 20.0          # a round trip above this on a local network usually means Wi-Fi power saving


def _ping_windows(ip: str, count: int, timeout_s: float, interval_s: float) -> List[Optional[float]]:
    import ctypes
    from ctypes import wintypes

    class IpOptions(ctypes.Structure):
        _fields_ = [("Ttl", ctypes.c_ubyte), ("Tos", ctypes.c_ubyte), ("Flags", ctypes.c_ubyte),
                    ("OptionsSize", ctypes.c_ubyte), ("OptionsData", ctypes.c_void_p)]

    class EchoReply(ctypes.Structure):
        _fields_ = [("Address", ctypes.c_ulong), ("Status", ctypes.c_ulong), ("RoundTripTime", ctypes.c_ulong),
                    ("DataSize", ctypes.c_ushort), ("Reserved", ctypes.c_ushort), ("Data", ctypes.c_void_p),
                    ("Options", IpOptions)]

    iphlpapi = ctypes.WinDLL("iphlpapi")
    iphlpapi.IcmpCreateFile.restype = wintypes.HANDLE
    iphlpapi.IcmpSendEcho.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p, wintypes.WORD,
                                      ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD]
    iphlpapi.IcmpSendEcho.restype = wintypes.DWORD
    iphlpapi.IcmpCloseHandle.argtypes = [wintypes.HANDLE]
    handle = iphlpapi.IcmpCreateFile()
    address = int.from_bytes(socket.inet_aton(ip), "little")      # IPAddr: the address bytes as stored
    payload = b"rps-ping" * 4
    reply = ctypes.create_string_buffer(ctypes.sizeof(EchoReply) + len(payload) + 64)
    out: List[Optional[float]] = []
    try:
        for i in range(count):
            t0 = time.perf_counter()
            n = iphlpapi.IcmpSendEcho(handle, address, payload, len(payload), None, reply, len(reply),
                                      int(timeout_s * 1000))
            rtt = (time.perf_counter() - t0) * 1000.0
            ok = n > 0 and EchoReply.from_buffer(reply).Status == 0
            out.append(rtt if ok else None)
            if i + 1 < count:
                time.sleep(max(0.0, interval_s - rtt / 1000.0))
    finally:
        iphlpapi.IcmpCloseHandle(handle)
    return out


def _ping_linux(ip: str, count: int, timeout_s: float, interval_s: float) -> List[Optional[float]]:
    cmd = ["ping", "-n", "-c", str(count), "-i", f"{max(0.2, interval_s):.1f}", "-W", str(max(1, round(timeout_s))), ip]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=count * (interval_s + timeout_s) + 5,
                         env={"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
    got = {int(seq): float(ms) for seq, ms in re.findall(r"icmp_seq=(\d+)\b.*?time=([\d.]+) ms", run.stdout)}
    if not got and run.returncode not in (0, 1):
        raise OSError(run.stderr.strip() or f"ping failed ({run.returncode})")
    return [got.get(i + 1) for i in range(count)]


def ping_ms(host: str, count: int = 10, timeout_s: float = 1.0, interval_s: float = 0.2) -> List[Optional[float]]:
    """Round-trip times in ms, one per ping (None: no answer). Raises OSError if ping cannot run."""
    ip = socket.gethostbyname(host)
    if sys.platform == "win32":
        return _ping_windows(ip, count, timeout_s, interval_s)
    return _ping_linux(ip, count, timeout_s, interval_s)


def summarize(rtts: List[Optional[float]]) -> dict:
    """Median and slowest round trip, losses, and the one-way delay (half the median round trip)."""
    got = sorted(r for r in rtts if r is not None)
    if not got:
        return {"answered": 0, "sent": len(rtts)}
    med = median(got)
    return {"answered": len(got), "sent": len(rtts), "median_ms": med, "max_ms": got[-1],
            "one_way_ms": med / 2.0, "slow": med > SLOW_RTT_MS or got[-1] > 3 * SLOW_RTT_MS}
