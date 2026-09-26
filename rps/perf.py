"""
Process-level performance settings for Windows laptops with hybrid (P/E-core) CPUs.

Windows 11 applies EcoQoS "power throttling" to processes it considers background work, which
can park them on efficiency cores and roughly double MediaPipe's per-frame time. Opting out of
execution-speed throttling and raising the priority class keeps the perception loop on fast
cores. No-ops on other platforms.
"""

import ctypes
import sys

PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
ProcessPowerThrottling = 4
HIGH_PRIORITY_CLASS = 0x00000080


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def boost_process(high_priority: bool = True) -> bool:
    """Disables EcoQoS throttling (and optionally raises priority). Returns True on success."""
    if sys.platform != "win32":
        return False
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    state = _PowerThrottlingState(
        PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        0,   # StateMask 0 = throttling OFF for the controlled features
    )
    ok = bool(kernel32.SetProcessInformation(handle, ProcessPowerThrottling,
                                             ctypes.byref(state), ctypes.sizeof(state)))
    if high_priority:
        ok = bool(kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)) and ok
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)   # 1 ms timer resolution for sleeps/waits
    except OSError:
        pass
    return ok
