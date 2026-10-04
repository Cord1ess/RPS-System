## RPS-System CLI Plan (what was done)

- Branch: feature/rps-cli
- Core refactors: moved config fields to rps/fields.py (Qt-free), moved beat synthesis to rps/sound.py with BeatAudio backend, moved delay model to rps/delay.py (shared by UI and CLI), and extracted the live loop into rps/runner.py so play.py becomes a thin OpenCV front-end.
- CLI foundation: pink theme (with 24-bit/256/none, NO_COLOR, UTF-8 on Windows), common context/JSON/interactive helpers, argparse dispatcher that hoists global flags (--config/--set/--json/--no-color) to work before or after the command, and command registry in rps/cli/commands.py.
- Commands implemented so far: config (show/set/get/unset/path/edit with labels from rps.fields, marks non-defaults), setup (check/probe/zone/latency, reusing tools/*), bot (ping/send/poses/sim/finger/moves, quiet link option). 
- Entry points: python -m rps and python rps.py both invoke the CLI.
- Tests: all non-UI tests pass (106-129 passed range depending on environment); CLI-specific tests for new commands pass where environment allows. Original UI tests skipped when PySide6 missing in some runs but installed for final validation (22 passed with offscreen).

## Planned remaining (from original plan)
Commands: record, dataset, train, eval, play (CLI text HUD variant), debug, doctor, bench, models, mock-esp, import-dextra. Plus additive PI.md and documentation updates. No pyproject.toml/console script as specified.
