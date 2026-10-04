"""
The interactive menu: `rps` with no arguments, or `rps menu`.

Numbers only, so it works over SSH on a Pi with nothing installed but Python. Every entry here is
the same code the typed commands run: the menu builds an argv list and hands it to the real parser,
so nothing can drift between `rps setup zone` and picking a number.

The choices you make are remembered in data/menu_recent.json (the recognizer, the mode, the HUD, the
robot strategy), so the next visit starts where you left off.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

from rps.cli import bot, config_cmd, play, setup
from rps.cli.common import Ctx, confirm, prompt
from rps.cli.theme import CHECK, theme

RECOGNIZERS = ["mediapipe", "dextra_tuned", "dextra_raw", "both"]
MODES = ["countdown", "guided", "continuous"]
ROBOT_PLAYS = ["lose", "win", "draw"]


class Recent:
    """The choices from last time: which recognizer, which mode, HUD or not, robot strategy."""

    DEFAULTS = {"recognizer": "mediapipe", "mode": "countdown", "robot_plays": "lose", "hud": True}

    def __init__(self, ctx: Ctx):
        self.path = ctx.data_dir("menu_recent.json")
        self.data = dict(self.DEFAULTS)
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for key in self.DEFAULTS:
                    if key in saved:
                        self.data[key] = saved[key]
        except (OSError, ValueError):
            pass                                  # no history yet, or it is not ours to read

    def get(self, key: str):
        return self.data.get(key, self.DEFAULTS[key])

    def set(self, key: str, value) -> None:
        self.data[key] = value
        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError:
            pass                                  # a read-only data dir is not worth stopping for

    def get_int(self, key: str) -> Optional[int]:
        try:
            return int(self.get(key))
        except (TypeError, ValueError):
            return None


def _clear() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def _pause() -> None:
    try:
        theme.out(theme.hint("Enter to go back"))
        input(" ")
    except (KeyboardInterrupt, EOFError):
        print()


def _ask(label: str, options: List[str], current: Optional[str] = None) -> Optional[str]:
    """A numbered list of words. Returns the chosen one, or None to go back. Numbers only."""
    if current:
        theme.out(theme.dim(f"  now: {current}"))
    for i, option in enumerate(options, 1):
        mark = theme.ok(CHECK) if option == current else " "
        theme.out(f"  {theme.style(f'[{i}]', 'pink')} {mark} {option}")
    while True:
        theme.field(label)
        answer = input(" ").strip()
        if not answer:
            return None
        if answer in options:                     # typing the word itself works too
            return answer
        try:
            index = int(answer)
        except ValueError:
            theme.out(theme.warn("Type a number, or the name"))
            continue
        if 1 <= index <= len(options):
            return options[index - 1]
        theme.out(theme.warn(f"Choose 1 to {len(options)}"))


def _ask_line(label: str, default: str = "") -> Optional[str]:
    try:
        return prompt(label, default) or None
    except (KeyboardInterrupt, EOFError):
        return None


def _dispatch(argv: List[str], ctx: Ctx) -> None:
    """Run a command exactly as if it had been typed, with the menu's own settings behind it."""
    from rps.cli.main import build_parser, ctx_from, hoist_globals

    parser = build_parser(only=argv[0])
    args = parser.parse_args(hoist_globals(argv, parser.own_flags))
    args.config = str(ctx.config)
    args.data_root = str(ctx.data_root) if ctx.data_root else None
    args.set = list(ctx.set_values)
    if getattr(args, "func", None) is None:
        return
    try:
        args.func(args, ctx)
    except (SystemExit, KeyboardInterrupt):
        pass                                      # a cancelled command is not a broken menu
    except Exception as e:                       # one bad menu choice should not kill the menu
        theme.out(theme.bad(f"{type(e).__name__}: {e}"))


_CANCELLED = object()      # someone typed something impossible: go back, run nothing


def _numbers(label: str, count: int):
    """Ask for a few whole numbers on one line, e.g. the X Y SIZE of a play zone.

    Returns the numbers, None for a blank answer (which means something else, like dragging on the
    video), or _CANCELLED for an answer that makes no sense.
    """
    while True:
        try:
            theme.field(label)
            raw = input(" ").strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not raw:
            return None
        parts = raw.replace(",", " ").split()
        try:
            values = [int(p) for p in parts]
        except ValueError:
            theme.out(theme.warn("Those have to be whole numbers"))
            continue
        if len(values) != count:
            theme.out(theme.warn(f"That is {len(values)} numbers; this one wants {count}"))
            continue
        return values


# --- the menus ------------------------------------------------------------------------------


def show_main(recent: Recent, ctx: Ctx) -> None:
    _clear()
    theme.title("Rock paper scissors")
    theme.out(theme.dim(f"  settings {ctx.config}"))
    theme.out()
    theme.out(f"  {theme.style('[1]', 'pink')} Setup    {theme.dim('camera, play zone, delay')}")
    theme.out(f"  {theme.style('[2]', 'pink')} Bot      {theme.dim('the robot hand, over Wi-Fi')}")
    theme.out(f"  {theme.style('[3]', 'pink')} Config   {theme.dim('every setting in config.json')}")
    theme.out(f"  {theme.style('[4]', 'pink')} Play     {theme.dim(str(recent.get('recognizer')))}")
    theme.out(f"  {theme.style('[5]', 'pink')} Status   {theme.dim('what is set up, at a glance')}")
    theme.out(f"  {theme.style('[6]', 'pink')} Quit")
    theme.out()


def menu_setup(ctx: Ctx) -> None:
    while True:
        _clear()
        theme.title("Setup")
        theme.out()
        choice = _ask("What", ["Check the camera (fps, light, hand in zone)",
                               "Auto-configure the camera (probe)",
                               "Set the play zone (the square to look at)",
                               "Measure the camera delay",
                               "Back"])
        if choice is None or choice == "Back":
            return
        if choice.startswith("Check"):
            _dispatch(["setup", "check"], ctx)
        elif choice.startswith("Auto"):
            _dispatch(["setup", "probe"], ctx)
        elif choice.startswith("Set"):
            numbers = _numbers("Play zone X Y SIZE (blank to drag on the video)", 3)
            if numbers is _CANCELLED:
                continue                            # nonsense typed: ask again, do not run anything
            _dispatch(["setup", "zone"] + [str(n) for n in (numbers or [])], ctx)
        else:
            mode = _ask("Delay by", ["screen", "led"], "screen")
            if mode:
                _dispatch(["setup", "latency", "--mode", mode], ctx)
        _pause()


def menu_bot(ctx: Ctx) -> None:
    while True:
        _clear()
        theme.title("Bot")
        theme.out()
        choice = _ask("What", ["Ping the robot (Wi-Fi delay)",
                               "Move the hand (rock, paper, scissors)",
                               "Show the poses",
                               "Finger angles",
                               "Hand move times",
                               "Back"])
        if choice is None or choice == "Back":
            return
        if choice.startswith("Ping"):
            _dispatch(["bot", "ping"], ctx)
        elif choice.startswith("Move"):
            pose = _ask("Pose", ["test", "ready", "rock", "paper", "scissors"], "test")
            mock = ["--mock"] if confirm("Use a simulated robot on this computer?", False) else []
            if pose:
                _dispatch(["bot", "send", pose] + mock, ctx)
        elif choice.startswith("Show"):
            _dispatch(["bot", "poses"], ctx)
        elif choice.startswith("Finger"):
            angles = _numbers("Three angles, e.g. 90 90 90 (blank to show them)", 3)
            if angles is _CANCELLED:
                continue
            mock = ["--mock"] if confirm("Use a simulated robot on this computer?", False) else []
            _dispatch(["bot", "finger"] + [str(a) for a in (angles or [])] + mock, ctx)
        else:
            pair = _ask_line("One move to change, e.g. N>R=120 (blank to show the table)", "")
            _dispatch(["bot", "moves"] + ([pair] if pair else []), ctx)
        _pause()


def menu_config(ctx: Ctx) -> None:
    while True:
        _clear()
        theme.title("Config")
        theme.out()
        choice = _ask("What", ["Show the settings",
                               "Get one value",
                               "Set values and save",
                               "Put values back to their defaults",
                               "Where the settings file is",
                               "Back"])
        if choice is None or choice == "Back":
            return
        if choice.startswith("Show"):
            _dispatch(["config"], ctx)
        elif choice.startswith("Get"):
            key = _ask_line("Which one, e.g. robot.host", "")
            if key:
                _dispatch(["config", "get", key], ctx)
        elif choice.startswith("Set"):
            pairs = _ask_line("One or more, e.g. decision.mode=continuous", "")
            if pairs:
                _dispatch(["config", "set"] + pairs.split(), ctx)
        elif choice.startswith("Put"):
            keys = _ask_line("One or more, e.g. decision.mode", "")
            if keys:
                _dispatch(["config", "unset"] + keys.split(), ctx)
        else:
            _dispatch(["config", "path"], ctx)
        _pause()


def menu_play(ctx: Ctx, recent: Recent) -> None:
    while True:
        _clear()
        theme.title("Play")
        theme.out()
        theme.out(theme.dim(f"  recognizer {recent.get('recognizer')}   mode {recent.get('mode')}   "
                            f"robot plays to {recent.get('robot_plays')}   "
                            f"{'HUD' if recent.get('hud') else 'no HUD'}"))
        theme.out()
        choice = _ask("What", ["Play with these settings",
                               "Quick test (no camera, no robot)",
                               "Choose the recognizer",
                               "Choose the mode",
                               "Choose what the robot does",
                               "Toggle the live HUD",
                               "Back"])
        if choice is None or choice == "Back":
            return
        if choice.startswith("Play with"):
            # Ctrl-C stops the game and drops back into the menu rather than out of it
            try:
                _dispatch(_play_argv(recent), ctx)
            except KeyboardInterrupt:
                theme.out()
                theme.out(theme.hint("Stopped. Back to the menu."))
            return
        if choice.startswith("Quick"):
            try:
                _dispatch(["play", "--recognizer", "mediapipe", "--mock-camera", "--mock-esp",
                           "--mode", "continuous", "--no-hud", "--headless", "120"], ctx)
            except KeyboardInterrupt:
                theme.out()
            _pause()
            continue
        if choice.startswith("Choose the recognizer"):
            value = _ask("Recognizer", RECOGNIZERS, recent.get("recognizer"))
            if value:
                recent.set("recognizer", value)
            continue
        if choice.startswith("Choose the mode"):
            value = _ask("Mode", MODES, recent.get("mode"))
            if value:
                recent.set("mode", value)
            continue
        if choice.startswith("Choose what"):
            value = _ask("Robot plays to", ROBOT_PLAYS, recent.get("robot_plays"))
            if value:
                recent.set("robot_plays", value)
            continue
        recent.set("hud", not recent.get("hud"))


def _play_argv(recent: Recent) -> List[str]:
    argv = ["play", "--recognizer", str(recent.get("recognizer")), "--mode", str(recent.get("mode")),
            "--robot-plays", str(recent.get("robot_plays"))]
    if not recent.get("hud"):
        argv.append("--no-hud")
    return argv


def menu_status(recent: Recent, ctx: Ctx) -> None:
    """Everything that matters on one screen: is it set up, and does a hand show up in the zone."""
    _clear()
    theme.title("Status")
    theme.out()
    cfg = ctx.load()
    rows = [
        ("settings file", str(ctx.config)),
        ("recognizer", f"{recent.get('recognizer')} (menu) / {cfg.decision.recognizer} (config.json)"),
        ("mode", str(recent.get("mode"))),
        ("robot", f"{cfg.robot.mode} at {cfg.robot.host}:{cfg.robot.port}"),
        ("play zone", f"x {cfg.roi.x}  y {cfg.roi.y}  size {cfg.roi.size}"),
        ("camera", f"index {cfg.camera.index}  {cfg.camera.width}x{cfg.camera.height}"),
        ("hand tracking", "on" if cfg.hand.enabled else "off"),
        ("robot plays to", str(recent.get("robot_plays"))),
    ]
    for label, value in rows:
        theme.out(f"  {theme.key(label)}  {value}")
    theme.out()
    root = Path(ctx.config).parent
    recognizer = str(recent.get("recognizer"))
    tuned = (root / cfg.cnn.tuned_model).exists() if recognizer in ("dextra_tuned", "both") else None
    raw = (root / cfg.cnn.raw_model).exists() if recognizer in ("dextra_raw", "both") else None
    dextra_dir = root / "models" / "dextra"
    wanted = [("dextra raw", raw), ("dextra tuned", tuned), ("dextra weights", dextra_dir)]
    for name, thing in wanted:
        if thing is None:
            continue
        there = thing.exists() if isinstance(thing, Path) else thing
        theme.out(f"  {theme.key(name)}  {'here' if there else theme.warn('missing')}")
    theme.out()
    theme.out(theme.hint("  A hand in the play zone is only known while playing: choose 4) Play,"))
    theme.out(theme.hint("  and the HUD says 'hand inside zone' or 'no hand in zone'."))
    theme.out(theme.hint("  Every setting, with what it does: choose 3) Config."))
    _pause()


def run_menu(ctx: Ctx) -> int:
    recent = Recent(ctx)
    while True:
        show_main(recent, ctx)
        try:
            theme.field("Choose")
            answer = input(" ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            theme.out(theme.ok("Bye"))
            return 0
        if answer in ("6", "q", "quit", "exit"):
            theme.out(theme.ok("Bye"))
            return 0
        if answer == "1":
            menu_setup(ctx)
        elif answer == "2":
            menu_bot(ctx)
        elif answer == "3":
            menu_config(ctx)
        elif answer == "4":
            menu_play(ctx, recent)
        elif answer == "5":
            menu_status(recent, ctx)
        elif answer:
            theme.out(theme.warn("Choose 1 to 6"))


def run(args, ctx: Ctx) -> int:
    if not ctx.interactive:
        from rps.cli.main import build_parser      # asked for the menu with nothing to type into
        build_parser().print_help()               # so say what the menu would have done
        return 0
    return run_menu(ctx)