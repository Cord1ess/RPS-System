"""The command line itself: the parser, the pink theme, and that colour disappears when it should."""

import io
import os
import subprocess
import sys

import pytest

from rps.cli.common import Ctx
from rps.cli.main import build_parser, ctx_from, main
from rps.cli.theme import (ARROW, BAD, BLOCK, DOT, OK, PINK, PINK_BRIGHT, WARN, Theme, _to_256, enable_utf8,
                               theme)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TTY(io.StringIO):
    """A stream that claims to be a terminal, so the colour code paths can be tested."""

    def isatty(self):
        return True


@pytest.fixture(autouse=True)
def _the_global_theme_is_left_as_it_was():
    was = theme.color
    yield
    theme.configure(color=was)


def test_help_lists_every_command():
    from rps.cli.commands import COMMANDS

    text = build_parser().format_help()
    for name, description, _module in COMMANDS:
        assert name in text, name
        assert description[:30] in " ".join(text.split())


def test_no_arguments_prints_the_help_and_succeeds(capsys):
    assert main([]) == 0
    assert "usage: rps" in capsys.readouterr().out


def test_a_command_with_no_module_behind_it_says_so_instead_of_a_traceback(capsys):
    assert main(["doctor"]) == 127          # nothing imports it when only the help is being printed
    assert "rps.cli.doctor" in capsys.readouterr().err


def test_module_entry_points_agree():
    for argv in (["-m", "rps"], ["rps.py"]):
        done = subprocess.run([sys.executable, *argv, "--version"], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith("rps ")


def test_pink_is_the_accent_and_meaning_keeps_its_own_colour():
    plain, colored = Theme(color=False), Theme(color=True, stream=TTY())
    assert plain.style("Speed", "pink") == "Speed"
    assert plain.ok("fine") == "✓ fine"
    assert colored.style("Speed", "pink").endswith("Speed\x1b[0m")
    assert "\x1b[38;2;255;95;162m" in colored.style("Speed", "pink")            # #ff5fa2
    assert "\x1b[38;2;255;168;205m" in colored.style("Speed", "pink_bright")    # #ffa8cd
    assert colored.style("x", "pink_bold").startswith("\x1b[1m")
    for role, color in (("ok", OK), ("warn", WARN), ("bad", BAD)):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        assert f"\x1b[38;2;{r};{g};{b}m" in colored.style("x", role)


def test_truecolor_falls_back_to_256_then_to_nothing(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("COLORTERM", raising=False)
    assert Theme(color=True, stream=TTY()).depth == 8
    assert f"\x1b[38;5;{_to_256(PINK)}m" in Theme(color=True, stream=TTY()).style("x", "pink")
    assert _to_256(PINK) == 211
    assert _to_256("#000000") == 16 and _to_256("#ffffff") == 231
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert Theme(color=True, stream=TTY()).depth == 24
    monkeypatch.setenv("NO_COLOR", "1")
    assert Theme(stream=TTY()).color is False


def test_no_color_flag_and_json_turn_the_colour_off(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    parser = build_parser()
    ctx_from(parser.parse_known_args(["--no-color", "config"])[0])
    assert theme.color is False
    ctx_from(parser.parse_known_args(["--json", "config"])[0])
    assert theme.color is False
    assert Theme().color is False               # not a terminal here either: no colour


def test_a_colour_of_ones_own_can_be_worn_too():
    colored = Theme(color=True, stream=TTY())
    from rps.delay import COLORS, HARDWARE

    grey = COLORS[HARDWARE][1]                                 # the same greys the Speed card draws
    assert colored.style("Wi-Fi", grey) == f"\x1b[38;2;122;131;144mWi-Fi\x1b[0m"


def test_plain_text_can_always_be_recovered():
    colored = Theme(color=True, stream=TTY())
    assert colored.plain(colored.style("Camera 12 ms", "ok")) == "Camera 12 ms"


def test_a_table_lines_up_with_and_without_rich(capsys):
    Theme(color=False).table(["Throw", "Compute", "Read by"],
                             [["1", "31 ms", "Dextra"], ["10", "6 ms", "Mediapipe"]],
                             aligns=["right", "right", "left"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["THROW", "COMPUTE", "READ", "BY"]
    assert lines[1].startswith("─")                       # a pink rule under the headings
    assert lines[2].split() == ["1", "31", "ms", "Dextra"]
    assert lines[3].split() == ["10", "6", "ms", "Mediapipe"]
    rich_theme = Theme(color=True, stream=TTY())
    if rich_theme.console is not None:
        rich_theme.table(["Throw", "Compute"], [["1", "31 ms"]])
        assert "31 ms" in capsys.readouterr().out


def test_the_speed_bar_is_a_row_of_blocks_with_a_marker_for_the_command(capsys):
    Theme(color=False).bar([(0.5, "ok", "laptop"), (0.25, "muted", "hardware")], width=10, marker=0.5)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"{BLOCK * 5} laptop {BLOCK * 3} hardware {DOT * 2}"
    assert lines[1].index(ARROW) == 7 and lines[1].endswith("command sent")   # 5 blocks in, then the arrow


def test_a_bar_never_rounds_away_what_it_is_measuring(capsys):
    Theme(color=False).bar([(1 / 3, "ok", "a"), (1 / 3, "ok", "b"), (1 / 3, "ok", "c")], width=9)
    line = capsys.readouterr().out.splitlines()[0]
    assert line.count(BLOCK) == 9 and DOT not in line


def test_a_console_that_cannot_show_the_box_still_works():
    assert isinstance(enable_utf8(), bool)      # never raises, whatever the streams are


def test_the_data_root_is_where_recordings_and_models_go(tmp_path):
    ctx = Ctx(config=tmp_path / "config.json", data_root=tmp_path / "elsewhere")
    path = ctx.data_dir("recordings", "alice")
    assert path == tmp_path / "elsewhere" / "recordings" / "alice" and path.parent.exists()
    assert Ctx(config=tmp_path / "config.json").data_dir("logs") == tmp_path / "data" / "logs"