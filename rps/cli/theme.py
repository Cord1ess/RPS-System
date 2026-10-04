"""
How the command line looks: the pink accent, the colours that mean something (green for measured
and fine, amber for waiting on something, red for wrong), and a table that lines up with or without
rich.

rich is used when it is installed, for tables and boxes only. Everything else is coloured here, so
a Raspberry Pi install without rich still gets the same colours, and a pipe to a file gets plain
text. Set NO_COLOR (the usual convention) or pass --no-color to turn colour off.
"""

import os
import shutil
import sys
from math import floor
from typing import Iterable, List, Optional, Sequence

PINK = "#ff5fa2"           # the accent: headings, rules, prompts, keys, the scoreboard
PINK_BRIGHT = "#ffa8cd"    # the accent on its own, with nothing around it
PINK_DIM = "#b34a78"       # the accent where it should not shout: quotes, hints
TEXT = "#e4e7eb"
MUTED = "#9aa3ad"
DARK = "#16191d"
OK = "#3fb96b"
WARN = "#e0a030"
BAD = "#e0524a"
DEXTRA = "#a78bfa"
MEDIAPIPE = "#2cc5c9"

ROLES = {                     # name -> (hex, bold)
    "pink": (PINK, False),
    "pink_bright": (PINK_BRIGHT, False),
    "pink_dim": (PINK_DIM, False),
    "pink_bold": (PINK, True),
    "text": (TEXT, False),
    "muted": (MUTED, False),
    "ok": (OK, False),
    "warn": (WARN, False),
    "bad": (BAD, False),
    "dextra": (DEXTRA, False),
    "mediapipe": (MEDIAPIPE, False),
    "none": (TEXT, False),
}

BULLET = "•"
ARROW = "→"
CHECK = "✓"
CROSS = "✗"
DOT = "·"
BLOCK = "█"


def _rgb(color: str):
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _to_256(color: str) -> int:
    """The xterm-256 colour nearest a hex one, for terminals without 24-bit colour."""
    r, g, b = _rgb(color)
    if abs(r - g) < 8 and abs(g - b) < 8:                     # grey: use the 24-step ramp
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + round((r - 8) / 247 * 23)
    return 16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)


def enable_virtual_terminal() -> bool:
    """Ask Windows' console for ANSI escapes. True if this terminal can show them."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)                    # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False                                       # not a console (piped to a file)
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x4))    # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        return False


def enable_utf8() -> bool:
    """Windows consoles still default to cp1252, where a block or an arrow is a crash. Ask for
    UTF-8, and fall back to '?' for anything the terminal still cannot show."""
    ok = False
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            ok = True
        except (AttributeError, ValueError, OSError):
            continue
    return ok


def color_wanted(force: Optional[bool] = None, stream=None) -> bool:
    """--no-color wins, then NO_COLOR, then whether this is a terminal someone is looking at."""
    if force is False:
        return False
    if os.environ.get("NO_COLOR") or os.environ.get("RPS_NO_COLOR"):
        return False
    if force is True:
        return True
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def depth_wanted(color: bool) -> int:
    """How many bits of colour this terminal really takes: 24, 8, or 0 for none."""
    if not color:
        return 0
    if os.name == "nt" and not os.environ.get("WT_SESSION") and enable_virtual_terminal():
        return 24                                           # Windows Terminal, or conhost with VT on
    term = f"{os.environ.get('COLORTERM', '')} {os.environ.get('TERM', '')}".lower()
    if "truecolor" in term or "24bit" in term:
        return 24
    if "256" in term:
        return 8
    return 24


class Theme:
    """Everything printed by the CLI goes through one of these."""

    def __init__(self, color: Optional[bool] = None, stream=None):
        self.silent = False                    # --json: the machine reads stdout, not a screen
        self.configure(color, stream)

    def configure(self, color: Optional[bool] = None, stream=None) -> "Theme":
        """Work out the colour once: --no-color and NO_COLOR win, then the terminal's own answer."""
        self.color = color_wanted(color, stream)
        self.depth = depth_wanted(self.color)
        self.width = shutil.get_terminal_size((100, 24)).columns
        self.console = None
        if self.color:
            try:                                              # rich, only for tables and boxes
                from rich.console import Console

                self.console = Console(highlight=False, soft_wrap=True)
            except ImportError:
                self.console = None
        return self

    def style(self, text: str, role: str = "text") -> str:
        """Colour some text. `role` is a name ("pink", "ok", "warn", "bad") or a "#rrggbb" of its
        own, so a bar can wear exactly the colours another front-end uses for the same thing."""
        color, bold = (role, False) if role.startswith("#") else ROLES.get(role, ROLES["text"])
        if not self.color or self.depth == 0:
            return text
        r, g, b = _rgb(color)
        if self.depth == 8:
            sgr = f"\x1b[38;5;{_to_256(color)}m"
        else:
            sgr = f"\x1b[38;2;{r};{g};{b}m"
        return f"{sgr}{text}\x1b[0m" if not bold else f"\x1b[1m{sgr}{text}\x1b[0m"

    def out(self, text: str = "") -> None:
        if not self.silent:
            print(text)

    def title(self, text: str) -> None:
        """The app's name, once, at the top of a run."""
        self.out(self.style(text, "pink_bold"))

    def heading(self, text: str) -> None:
        """A section: pink, with a rule under it."""
        self.out()
        self.out(self.style(text.upper(), "pink_bold"))
        self.out(self.style("─" * min(self.width, max(8, len(text) + 4)), "pink_dim"))

    def field(self, text: str) -> None:
        """What is being asked for, before an answer is typed."""
        self.out(self.style(f"{text} ", "pink") + self.style("›", "pink_bright"))

    def rule(self, label: str = "") -> None:
        if not label:
            self.out(self.style("─" * min(self.width, 80), "pink_dim"))
            return
        pad = max(0, min(self.width, 80) - len(label) - 3)
        self.out(self.style(f"─── {label} ", "pink_dim") + self.style("─" * pad, "pink_dim"))

    def key(self, name: str) -> str:
        """A setting's name, as the app writes it: latency.camera_latency_ms"""
        return self.style(name, "pink")

    def val(self, text: str) -> str:
        return self.style(str(text), "text")

    def hint(self, text: str) -> str:
        return self.style(text, "muted")

    def dim(self, text: str) -> str:
        return self.style(text, "muted")

    def ok(self, text: str) -> str:
        return self.style(f"{CHECK} {text}", "ok")

    def warn(self, text: str) -> str:
        return self.style(f"{ARROW} {text}", "warn")

    def bad(self, text: str) -> str:
        return self.style(f"{CROSS} {text}", "bad")

    def note(self, text: str) -> str:
        return self.style(f"{BULLET} {text}", "muted")

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[str]], aligns: Optional[Sequence[str]] = None) -> None:
        """Aligned columns. rich draws them when it is installed; otherwise padded text."""
        rows = [[("" if c is None else str(c)) for c in row] for row in rows]
        if self.console is not None:
            from rich import box
            from rich.table import Table

            table = Table(box=box.SIMPLE_HEAD, header_style=PINK, border_style=PINK_DIM,
                          show_edge=False, pad_edge=False, expand=False)
            for i, head in enumerate(headers):
                table.add_column(head, justify=(aligns[i] if aligns else "left"), no_wrap=(aligns is not None
                                                                                                 and aligns[i] == "right"))
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
            return
        columns = len(headers)
        aligns = list(aligns or ["left"] * columns)
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row[:columns]):
                widths[i] = max(widths[i], len(cell))
        head = "  ".join(self.style(h.upper().ljust(widths[i]), "pink") for i, h in enumerate(headers))
        self.out(head)
        self.out(self.style("─" * min(self.width, sum(widths) + 2 * (columns - 1)), "pink_dim"))
        for row in rows:
            cells = []
            for i in range(columns):
                cell = row[i] if i < len(row) else ""
                cells.append(cell.rjust(widths[i]) if aligns[i] == "right" else cell.ljust(widths[i]))
            self.out("  ".join(cells).rstrip())

    def bar(self, parts: Sequence[tuple], width: int = 40, marker: Optional[float] = None) -> None:
        """A row of blocks: (fraction, role, label) each, `fraction` being 0..1 of the whole. The
        `marker` (also 0..1) is where the command left the laptop, marked under the bar."""
        used = 0
        cells: List[str] = []
        for fraction, role, label in parts:
            if fraction <= 0:
                continue
            n = max(1, floor(width * fraction + 0.5))         # half up, so the blocks add up
            used += n
            cells.append(self.style(BLOCK * n, role) + self.style(f" {label}", "muted"))
        if used < width:
            cells.append(self.style(DOT * (width - used), "muted"))
        self.out(" ".join(cells))
        if marker is not None:
            at = max(0, min(width - 1, floor(width * marker + 0.5)))
            self.out(self.style(" " * at + "↑ " + ARROW + " command sent", "pink_dim"))

    def choices(self, options: Sequence[str], default: Optional[str] = None) -> str:
        """The one-line hint of what a prompt accepts, e.g.  [rock/paper/scissors] (rock)"""
        body = "/".join(options)
        if default:
            body += f" ({default})"
        return self.style(f"  [{body}]", "pink_dim")

    def plain(self, text: str) -> str:
        """Strip the colour back out, for --json and for writing to a file."""
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", text)


theme = Theme()