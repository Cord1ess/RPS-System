"""
`python -m rps` : the whole app from a terminal.

One parser for everything, built from the list in rps/cli/commands.py so adding a command is adding
one module with a register(subparsers) and a run(args, ctx). Global flags come first (--config,
--data-root, --set, --json, --no-color); each command adds its own. The same commands drive the same
settings as the desktop app.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List

from rps.cli.common import Ctx
from rps.cli.theme import enable_utf8, theme


def version() -> str:
    try:
        from importlib.metadata import version as installed

        return installed("rps-system")
    except Exception:
        return "3.3.0 + cli"


def add_global_flags(parser: argparse.ArgumentParser, suppress: bool = False) -> argparse.ArgumentParser:
    """The flags that apply to every command. They are added to each command as well, so both
    `rps --json config` and `rps config --json` work; there the defaults are suppressed, so saying
    nothing keeps whatever was said before the command name."""
    def d(default):
        return {"default": argparse.SUPPRESS} if suppress else {"default": default}

    parser.add_argument("--config", metavar="PATH", help="settings file to read and write (default: config.json)", **d("config.json"))
    parser.add_argument("--data-root", metavar="PATH",
                        help="where recordings, models and logs live (default: next to the settings file)", **d(None))
    parser.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                        help="change a setting for this run only, e.g. --set decision.recognizer=mediapipe", **d([]))
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of a screen of text", **d(False))
    parser.add_argument("--no-color", action="store_true", help="no colour anywhere (same as NO_COLOR=1)", **d(False))
    parser.add_argument("-q", "--quiet", action="store_true", help="only answers, no explanations", **d(False))
    parser.add_argument("-v", "--verbose", action="store_true", help="show the traceback when something breaks", **d(False))
    return parser


def build_parser(only: str = None) -> argparse.ArgumentParser:
    """The whole parser. With `only`, just that command's module is imported and its flags added,
    so `--help` never has to import the commands it is only listing."""
    from rps.cli.commands import COMMANDS

    parser = argparse.ArgumentParser(
        prog="rps",
        description="Rock paper scissors over a webcam and a robot hand: set it up, record, train, play.",
        epilog="Run `rps COMMAND --help` for one command, e.g. `rps play --headless 300`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"rps {version()}")
    add_global_flags(parser)

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands")
    for name, description, module_name in COMMANDS:
        sub = subparsers.add_parser(name, help=description, description=description,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_global_flags(sub, suppress=True)
        if only is not None and only == name:
            module = importlib.import_module(module_name)
            sub.set_defaults(func=getattr(module, "run"))
            register = getattr(module, "register", None)
            if register is not None:
                register(sub)
    return parser


def ctx_from(args: argparse.Namespace) -> Ctx:
    if args.no_color or args.json:
        theme.configure(color=False)
    return Ctx(config=Path(args.config), data_root=Path(args.data_root) if args.data_root else None,
               json=args.json, set_values=list(args.set), quiet=args.quiet)


GLOBAL_FLAGS_WITH_VALUE = ("--config", "--data-root", "--set")
GLOBAL_FLAGS = ("--json", "--no-color", "-q", "--quiet", "-v", "--verbose")


def hoist_globals(argv: List[str]) -> List[str]:
    """argparse wants the global flags before the command name, but people type them wherever they
    think of them (`rps play --json`, `rps config get x --config pi.json`). Move them to the front."""
    flags, rest = [], []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item in GLOBAL_FLAGS_WITH_VALUE and i + 1 < len(argv):
            flags += argv[i:i + 2]
            i += 2
        elif item.startswith(tuple(f"{f}=" for f in GLOBAL_FLAGS_WITH_VALUE)) or item in GLOBAL_FLAGS:
            flags.append(item)
            i += 1
        else:
            rest.append(item)
            i += 1
    return flags + rest


def main(argv: List[str] = None) -> int:
    enable_utf8()
    argv = hoist_globals(list(sys.argv[1:] if argv is None else argv))
    first = build_parser().parse_known_args(argv)      # which command? nothing else needed yet
    try:
        parser = build_parser(only=first[0].command)  # now its own flags
    except ImportError as e:
        print(f"{theme.bad('command unavailable')}: {e}", file=sys.stderr)
        return 127
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    ctx = ctx_from(args)
    try:
        return args.func(args, ctx) or 0
    except KeyboardInterrupt:
        print()
        return 130
    except SystemExit:
        raise
    except Exception as e:                       # a traceback is for bugs; this line is for people
        print(f"{theme.bad(type(e).__name__)}: {e}", file=sys.stderr)
        if args.verbose:
            raise
        return 1