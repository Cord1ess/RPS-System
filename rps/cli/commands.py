"""Every command, in the order `rps --help` lists them: name, one line about it, and the module
that implements it (register(subparsers) to add flags, run(args, ctx) to do the work)."""

COMMANDS = [
    ("config", "show the settings, or set one and save it", "rps.cli.config_cmd"),
    ("setup", "the Setup tab: camera, ROI, lighting, latency test", "rps.cli.setup"),
    ("bot", "the Bot tab: the ESP32 link, Wi-Fi delay, servo moves", "rps.cli.bot"),
    ("play", "play against a human or another program", "rps.cli.play"),
]