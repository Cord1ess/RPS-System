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
    """The whole parser. With `only`, that one command's module is imported and its flags added, so
    `--help` never has to import the commands it is only listing, and one command that cannot be
    imported does not take the whole listing down with it."""
    from rps.cli.commands import COMMANDS

    parser = argparse.ArgumentParser(
        prog="rps",
        description="Rock paper scissors over a webcam and a robot hand: set it up, record, train, play.",
        epilog="Run `rps COMMAND --help` for one command, e.g. `rps play --headless 300`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"rps {version()}")
    add_global_flags(parser, suppress=False)  # top level
    parser.own_flags = {f for a in parser._actions for f in a.option_strings}

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands")
    parser.own_flags = set()
    for name, description, module_name in COMMANDS:
        sub = subparsers.add_parser(name, help=description, description=description,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_global_flags(sub, suppress=True)
        if only != name:                          # the listing needs the name and the one line only
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            sub.set_defaults(func=_unavailable(module_name, e))
            continue
        sub.set_defaults(func=getattr(module, "run"))
        inherited = {id(action) for action in sub._actions}
        register = getattr(module, "register", None)
        if register is not None:
            register(sub)
        parser.own_flags = {opt for action in sub._actions if id(action) not in inherited
                            for opt in action.option_strings}
    return parser


def _unavailable(module_name: str, error: Exception):
    """A command whose module will not import (an optional dependency, say). Say so in one line
    rather than letting an ImportError reach the traceback handler."""
    def run(args, ctx):
        print(f"{theme.bad('command unavailable')}: {error}", file=sys.stderr)
        return 127
    return run


def ctx_from(args: argparse.Namespace) -> Ctx:
    if args.no_color or args.json:
        theme.configure(color=False)
    theme.silent = args.json                      # --json: stdout carries the object and nothing else
    return Ctx(config=Path(args.config), data_root=Path(args.data_root) if args.data_root else None,
               json=args.json, set_values=list(args.set), quiet=args.quiet)


GLOBAL_FLAGS_WITH_VALUE = ("--config", "--data-root", "--set")
GLOBAL_FLAGS = ("--json", "--no-color", "-q", "--quiet", "-v", "--verbose")


def hoist_globals(argv: List[str], skip: set = frozenset()) -> List[str]:
    """argparse wants the global flags before the command name, but people type them wherever they
    think of them (`rps play --json`, `rps config get x --config pi.json`). Move them to the front,
    except any the command has a flag of its own with the same name (`rps setup zone --set ...`)."""
    flags, rest = [], []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item in GLOBAL_FLAGS_WITH_VALUE and i + 1 < len(argv) and item not in skip:
            flags += argv[i:i + 2]
            i += 2
        elif (item.startswith(tuple(f"{f}=" for f in GLOBAL_FLAGS_WITH_VALUE)) and item not in skip) \
                or (item in GLOBAL_FLAGS and item not in skip):
            flags.append(item)
            i += 1
        else:
            rest.append(item)
            i += 1
    return flags + rest


def main(argv: List[str] = None) -> int:
    enable_utf8()
    argv = list(sys.argv[1:] if argv is None else argv)
    command = build_parser().parse_known_args(argv)[0].command   # which command? nothing else yet
    if not command:
        # no command: the menu if there is someone there to pick from it, the help otherwise
        probe = ctx_from(build_parser().parse_known_args(argv)[0])
        if probe.interactive and not probe.json:
            command, argv = "menu", argv + ["menu"]   # the same path as typing `rps menu`
        else:
            build_parser().print_help()
            return 0
    try:
        parser = build_parser(only=command)           # now its own flags
    except ImportError as e:
        print(f"{theme.bad('command unavailable')}: {e}", file=sys.stderr)
        return 127
    args = parser.parse_args(hoist_globals(argv, parser.own_flags))
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