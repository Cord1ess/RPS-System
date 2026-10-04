"""
The command line's shared bits: the context every command gets (config path, data root, colour,
--json), and small helpers for asking questions and ending with an error.

Pink is the accent; the colours that mean something (ok/warn/bad) keep their own. With --json,
nothing is coloured and every command prints one JSON object instead of a screen full of text.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rps.cli.theme import theme


@dataclass
class Ctx:
    """What every command needs to know about how it was called."""
    config: Path = Path("config.json")
    data_root: Optional[Path] = None
    json: bool = False
    set_values: List[str] = field(default_factory=list)
    quiet: bool = False

    def load(self):
        """The settings, with --set applied."""
        from rps.config import load_config

        return load_config(str(self.config), self.set_values)

    def data_dir(self, *parts: str) -> Path:
        root = Path(self.data_root) if self.data_root else Path(self.config).parent / "data"
        path = root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def emit(self, payload: Dict[str, Any], text=None) -> None:
        """--json: one object on stdout, nothing else. Otherwise whatever the command printed."""
        if self.json:
            print(json.dumps(payload, indent=2, default=str))
        elif text is not None:
            theme.out(text)

    def say(self, text: str = "") -> None:
        if not self.json and not self.quiet:
            theme.out(text)


def die(message: str, code: int = 1):
    """Stop with something readable: a pink cross and one line saying what to do about it."""
    if theme.console is not None:
        theme.console.print(theme.bad(message))
    else:
        print(theme.bad(message), file=sys.stderr)
    raise SystemExit(code)


def confirm(question: str, default: bool = True) -> bool:
    """Ask, with the answer in pink so it stands out from the question."""
    answer = input(f"{theme.field(question)} [y/n] " if default else f"{theme.field(question)} [y/N] ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def prompt(label: str, default: str = "", choices: Optional[List[str]] = None) -> str:
    """One question, one line. `default` is what an empty answer means."""
    hint = theme.choices(choices, default) if choices else (theme.dim(f"  [{default}]") if default else "")
    while True:
        answer = input(f"{theme.field(label)} {hint} ").strip()
        if not answer and default:
            return default
        if answer or not choices:
            return answer
        theme.warn(f"Pick one of: {', '.join(choices)}")


def human_size(path: Path) -> str:
    try:
        mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        return "-"
    return f"{mb:.1f} MB" if mb >= 1 else f"{mb * 1024:.0f} KB"