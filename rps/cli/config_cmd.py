"""
`rps config` : the settings, in the terminal.

Everything the app has in config.json, grouped the way the tabs are, with the same labels and help
text as the app (rps.fields) and the same defaults (rps.config). `show` marks what is not the
default, `set` and `unset` write the file back, `get` prints one bare value so a script can use it,
and `edit` opens the file in whatever editor the machine has.
"""

import difflib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import fields
from typing import Any, Dict, List, Tuple

from rps.cli.common import Ctx, die
from rps.cli.theme import theme
from rps.config import ALLOWED, Config, DEFAULT_CONFIG_PATH, apply_overrides, load_config, parse_set_args, save_config
from rps.fields import CHOICES, SECTION_INFO, help_for, is_advanced, label_for

MARK = "*"          # next to a setting that is not the default


def register(sub):
    sub.add_argument("--section", action="append", default=[], metavar="NAME",
                     help="only this section (camera, roi, dvs, cnn, hand, vote, decision, robot, game, latency)")
    sub.add_argument("--all", action="store_true", help="show the advanced settings too")
    sub.add_argument("--long", action="store_true", help="also print what each setting is for")
    sub.add_argument("--defaults", action="store_true", help="show the defaults instead of this machine's")

    actions = sub.add_subparsers(dest="action", metavar="ACTION")
    get = actions.add_parser("get", help="print one value, nothing else")
    get.add_argument("key", help="section.key, e.g. robot.host")
    setter = actions.add_parser("set", help="change a setting and save it")
    setter.add_argument("pairs", nargs="+", metavar="KEY=VALUE",
                        help="e.g. decision.mode=continuous robot.host=192.168.4.1")
    setter.add_argument("--no-save", action="store_true", help="check it and stop, without writing")
    unset = actions.add_parser("unset", help="put one or more settings back to their defaults")
    unset.add_argument("keys", nargs="+", metavar="KEY", help="e.g. decision.mode")
    actions.add_parser("path", help="print where the settings file is")
    actions.add_parser("edit", help="open the settings file in $EDITOR")


def run(args, ctx: Ctx) -> int:
    action = getattr(args, "action", None)
    if action == "get":
        return _get(args.key, ctx)
    if action == "set":
        return _set(args.pairs, ctx, save=not args.no_save)
    if action == "unset":
        return _unset(args.keys, ctx)
    if action == "path":
        theme.out(str(ctx.config.resolve()))
        return 0
    if action == "edit":
        return _edit(ctx)
    return _show(args, ctx)


def _load(args, ctx: Ctx) -> Config:
    """The settings this command works on: defaults, then the file, then --set for this run."""
    return Config() if getattr(args, "defaults", False) else ctx.load()


def _split(key: str) -> Tuple[str, str]:
    if "." not in key:
        die(f"Expected section.key, got '{key}'. Sections: {', '.join(SECTION_INFO)}")
    section, _, name = key.partition(".")
    return section, name


def _lookup(cfg: Config, section: str, key: str):
    if not hasattr(cfg, section):
        die(f"Unknown section '{section}'. Sections: {', '.join(SECTION_INFO)}")
    valid = {f.name for f in fields(getattr(cfg, section))}
    if key not in valid:
        close = difflib.get_close_matches(key, sorted(valid), n=3)
        die(f"Unknown setting '{section}.{key}'" + (f". Did you mean: {', '.join(close)}?" if close else ""))
    return getattr(getattr(cfg, section), key)


def _get(key: str, ctx: Ctx) -> int:
    section, name = _split(key)
    value = _lookup(ctx.load(), section, name)
    ctx.emit({key: value}, text=_plain(value))
    return 0


def _set(pairs: List[str], ctx: Ctx, save: bool = True) -> int:
    cfg = ctx.load()
    try:
        overrides = parse_set_args(pairs)
    except ValueError as e:
        die(str(e))
    try:
        apply_overrides(cfg, overrides)          # strict: a typo must not be silently saved
    except (KeyError, ValueError) as e:
        die(str(e).strip("'"))
    changed = [(f"{s}.{k}", v) for s, values in overrides.items() for k, v in values.items()]
    if not save:
        theme.warn("Not saving (--no-save)")
    if save:
        save_config(cfg, str(ctx.config), quiet=ctx.json)
    if ctx.json:
        print(json.dumps({k: v for k, v in changed}, indent=2, default=str))
    else:
        for key, value in changed:
            theme.out(f"{theme.key(key)} {theme.val(_plain(value))}")
    return 0


def _unset(keys: List[str], ctx: Ctx) -> int:
    cfg, fresh = ctx.load(), Config()
    for key in keys:
        section, name = _split(key)
        _lookup(cfg, section, name)
        setattr(getattr(cfg, section), name, getattr(getattr(fresh, section), name))
        theme.out(f"{theme.key(f'{section}.{name}')} {theme.val('back to the default')} "
                  f"{theme.hint(f'= {_plain(getattr(getattr(fresh, section), name))}')}")
    save_config(cfg, str(ctx.config), quiet=ctx.json)
    return 0


def _edit(ctx: Ctx) -> int:
    if not ctx.config.exists():
        die(f"No settings file at {ctx.config} yet: run `rps config set` once first")
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor:
        code = subprocess.call(f'{editor} "{ctx.config}"', shell=True)
    elif sys.platform == "win32":
        code = subprocess.call(["notepad", str(ctx.config)])
    elif shutil.which("vi"):
        code = subprocess.call(["vi", str(ctx.config)])
    else:
        die("No $EDITOR and no vi here: edit the file by hand, then run `rps config` again")
    if code != 0:
        die(f"The editor exited with {code}; the settings were left as they were")
    try:
        load_config(str(ctx.config))             # a hand-edited file must still load
    except Exception as e:
        die(f"{ctx.config} did not load after editing: {e}")
    theme.out(theme.ok(f"{ctx.config} is valid"))
    return 0


def _show(args, ctx: Ctx) -> int:
    cfg = _load(args, ctx)
    default = Config()
    wanted = [s for s in args.section] or list(SECTION_INFO)
    for name in wanted:
        if name not in SECTION_INFO:
            die(f"Unknown section '{name}'. Sections: {', '.join(SECTION_INFO)}")
    if ctx.json:
        print(json.dumps({s: getattr(cfg, s).__dict__ for s in wanted}, indent=2, default=str))
        return 0

    theme.title(f"RPS settings {theme.dim(ctx.config)}")
    if not ctx.config.exists() and not args.defaults:
        theme.out(theme.warn("no settings file yet: showing the defaults"))
    for section in wanted:
        label, description = SECTION_INFO[section]
        theme.heading(f"{label}  ({section})")
        theme.out("  " + theme.hint(description))
        for f in fields(getattr(cfg, section)):
            if is_advanced(section, f.name) and not args.all:
                continue
            value = getattr(getattr(cfg, section), f.name)
            was = getattr(getattr(default, section), f.name)
            mark = MARK if value != was else " "
            name = f"{label_for(section, f.name):<26} {_shown(value, section, f.name)}"
            theme.out(f"  {theme.style(name, 'text')} {theme.style(mark, 'pink_bright')}")
            if args.long:
                for line in _help(section, f.name):
                    theme.out("      " + theme.dim(line))
        theme.out()
    theme.out(theme.dim(f"{MARK} not the default; change one with rps config set section.key value"))
    return 0


def _plain(value: Any) -> str:
    """A value as one line: bare for a single thing, JSON when it is a list or a table of moves."""
    return json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)


def _shown(value: Any, section: str, key: str) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, dict):
        return f"{len(value)} entries" if value else "not set"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "empty"
    return str(value)


def _help(section: str, key: str) -> List[str]:
    lines = [help_for(section, key)] if help_for(section, key) else []
    choices = CHOICES.get((section, key))
    if choices:
        lines.append("one of: " + ", ".join(f"{value} = {label}" for value, label in choices))
    allowed = ALLOWED.get((section, key))
    if allowed:
        lines.append("allowed: " + ", ".join(str(a) for a in allowed))
    return [line for line in lines if line]