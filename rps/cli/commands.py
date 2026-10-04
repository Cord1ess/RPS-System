"""Every command, in the order `rps --help` lists them: name, one line about it, and the module
that implements it (register(subparsers) to add flags, run(args, ctx) to do the work)."""

COMMANDS = [
    ("doctor", "check the camera, the models, the robot link and the audio", "rps.cli.doctor"),
    ("config", "show the settings, or set one and save it", "rps.cli.config_cmd"),
    ("setup", "the Setup tab: camera, ROI, lighting, latency test", "rps.cli.setup"),
    ("bot", "the Bot tab: the ESP32 link, Wi-Fi delay, servo moves", "rps.cli.bot"),
    ("record", "record throw sessions to a folder", "rps.cli.record"),
    ("dataset", "build and check the training dataset", "rps.cli.dataset"),
    ("train", "train the Dextra motion model from a dataset", "rps.cli.train"),
    ("eval", "score a trained model on held-out recordings", "rps.cli.eval_cmd"),
    ("models", "list, describe and export the models on disk", "rps.cli.models"),
    ("play", "play against a human or another program", "rps.cli.play"),
    ("debug", "the debug views: cameras, hand, DVS, latency", "rps.cli.debug"),
]