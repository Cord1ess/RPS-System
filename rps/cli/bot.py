"""
`rps bot` : the robot, from the terminal.

ping    measure the Wi-Fi delay to the robot and keep it for the Speed card.
send    send one command and report whether the robot took it (the Bot page's buttons).
poses   the poses and the commands they turn into.
finger  the team firmware's three servo channels (ANGLE commands).
sim     a fake robot on this computer, to try things without the hardware.
moves   the hand's move times, used to estimate when its move becomes visible.
"""

import time
from typing import Dict, List

from rps.cli.common import Ctx, die
from rps.cli.theme import theme
from rps.config import save_config
from rps.hud import POSE_NAME
from rps.netping import ping_ms, summarize
from rps.robot_link import (ANGLE_MAX, ANGLE_MIN, ESP_RST_BROWNOUT, FINGER_CHANNELS, POSES, TEXT_COMMAND, MockEsp,
                            RobotLink)

SLOW_MS = 20.0        # the app's own line: slow or uneven for a local network


def register(sub):
    actions = sub.add_subparsers(dest="action", metavar="ACTION")

    ping = actions.add_parser("ping", help="measure the Wi-Fi delay to the robot")
    ping.add_argument("--count", type=int, default=10, help="pings to send (default: 10)")
    ping.add_argument("--no-save", action="store_true", help="report it without keeping it")

    send = actions.add_parser("send", help="send one pose or command to the robot")
    send.add_argument("pose", nargs="?", default="test", help="rock, paper, scissors, ready or test (default: test)")
    send.add_argument("--mock", action="store_true", help="use a simulated robot on this computer")

    actions.add_parser("poses", help="the poses and the commands they turn into")
    actions.add_parser("sim", help="run a fake robot on this computer, for trying things out")

    finger = actions.add_parser("finger", help="the team firmware's three servo channels")
    finger.add_argument("angles", nargs="*", type=int, metavar="ANGLE",
                        help="three angles, e.g. 90 90 90. Without them, show the current ones")
    finger.add_argument("--mock", action="store_true", help="use a simulated robot on this computer")
    finger.add_argument("--save", action="store_true", help="also remember them in robot.finger_angles")

    moves = actions.add_parser("moves", help="show or set the hand's move times (latency.servo_transition_ms)")
    moves.add_argument("pairs", nargs="*", metavar="FROM>TO=MS",
                       help="e.g. N>R=120. Without them, show the table")
    moves.add_argument("--measure", action="store_true",
                       help="time each move over the robot and work the median out")


def run(args, ctx: Ctx) -> int:
    action = getattr(args, "action", None) or "ping"
    return {"ping": _ping, "send": _send, "poses": _poses, "sim": _sim, "finger": _finger, "moves": _moves}[action](
        args, ctx)


def _target(cfg) -> str:
    return "this computer (simulated)" if cfg.robot.mode == "simulated" else f"{cfg.robot.host}:{cfg.robot.port}"


def _link(cfg, mock: bool = False):
    """A connection to the real or the simulated robot, already sending. Quiet: this command says
    what it sent and what came back, in colour."""
    if cfg.robot.mode == "off" and not mock:
        die("The robot is off (robot.mode=off). Turn it on with `rps config set robot.mode=real`")
    spoken: List[str] = []
    link = RobotLink.from_config(cfg.robot, host="127.0.0.1" if mock else None,
                                 on_text=spoken.append, quiet=True).start()
    return link, spoken


def _wait_a_moment(link, seconds: float) -> Dict:
    time.sleep(seconds)
    return link.stats()


# ---------------------------------------------------------------------------- ping
def _ping(args, ctx: Ctx) -> int:
    """Half the ping time is the delay of one command on the way to the robot."""
    cfg = ctx.load()
    host = "127.0.0.1" if cfg.robot.mode == "simulated" else cfg.robot.host
    theme.title("Wi-Fi delay")
    theme.out(theme.hint(f"pinging {host}, {args.count} times"))
    try:
        result = summarize(ping_ms(host, count=args.count))
    except Exception as e:                         # no ping command, bad address, ...
        die(f"The ping could not run: {e}")
    if not result["answered"]:
        theme.out(theme.bad(f"no answer ({result['answered']} of {result['sent']})"))
        theme.out(theme.hint("the robot has to be on, on the same network, at the address in robot.host"))
        ctx.emit({"answered": False, "sent": result["sent"]})
        return 1
    one_way, lost = result["one_way_ms"], result["sent"] - result["answered"]
    theme.out(theme.rule(f"{one_way:.1f} ms one way"))
    theme.out("  round trip   " + theme.val(f"{result['median_ms']:.1f} ms") + "  " + theme.hint("median"))
    theme.out("  slowest      " + theme.val(f"{result['max_ms']:.0f} ms"))
    if lost:
        theme.out(theme.warn(f"{lost} of {result['sent']} lost"))
    if result["slow"] or one_way > SLOW_MS:
        theme.out(theme.warn("slow or uneven for a local network. The usual cause is the ESP32's Wi-Fi power saving, "
                             "which delays each command by up to ~100 ms: add WiFi.setSleep(false); to its setup()"))
    if cfg.robot.mode == "simulated":
        theme.out(theme.note("the simulated robot is this computer, so this is not a Wi-Fi delay: not kept"))
        return 0
    if args.no_save:
        theme.out(theme.dim("not kept (--no-save)"))
        return 0
    cfg.latency.network_ms = round(one_way, 1)
    save_config(cfg, str(ctx.config), quiet=ctx.json)
    theme.out(theme.ok(f"kept as latency.network_ms = {cfg.latency.network_ms} ms"))
    return 0


# ---------------------------------------------------------------------------- send
def _send(args, ctx: Ctx) -> int:
    """One command, then what came back: what the Bot page shows after a click."""
    cfg = ctx.load()
    want = args.pose.strip().lower()
    pose = {"test": "N" if cfg.robot.protocol == "ack" else "P", "ready": "N"}.get(want, want[:1].upper())
    if pose not in POSES:
        die(f"Unknown pose '{args.pose}'. Use rock, paper, scissors, ready or test.")
    mock = args.mock or cfg.robot.mode == "simulated"
    link, spoken = _link(cfg, mock=mock)
    fake = MockEsp(port=cfg.robot.port, verbose=False).start() if mock else None
    theme.title("Send to the robot")
    try:
        theme.out(f"  to          {theme.val(_target(cfg))}  {theme.hint('simulated' if mock else '')}")
        theme.out(f"  firmware    {theme.val(cfg.robot.protocol)}")
        before = link.stats()
        seq = link.send_pose(pose)
        sent = TEXT_COMMAND.get(pose, "(no ready command in this firmware)") if cfg.robot.protocol == "rps_text" \
            else f"P,{seq},{pose}"
        theme.out(f"  sent        {theme.key(sent)}  "
                  f"{theme.dim('team firmware' if cfg.robot.protocol == 'rps_text' else 'reference firmware')}")
        stats = _wait_a_moment(link, 0.3)
        if stats["sent"] == before["sent"]:
            theme.out("  " + theme.bad(f"nothing was sent: {link.send_error or 'the socket refused it'}"))
            theme.out(theme.hint("check the laptop is on the robot's network, and robot.host and robot.port"))
            return 1
        if not stats["replies"]:                   # team firmware: no replies to wait for
            if fake is not None and fake.received == 0:
                theme.out("  " + theme.bad("the simulated robot received nothing"))
                return 1
            theme.out("  " + theme.warn("this firmware does not reply, so delivery cannot be confirmed here"))
            theme.out(theme.hint(f"check the hand shows {POSE_NAME[pose].lower()}"))
            return 0
        if stats["acked"] > before["acked"]:
            rtt = f" · {stats['rtt_median_ms']:.1f} ms round trip" if stats["rtt_median_ms"] else ""
            theme.out("  " + theme.ok(f"the robot took it{rtt}"))
            if stats["last_reset_reason"] == ESP_RST_BROWNOUT:
                theme.out(theme.warn("the robot last restarted because its supply dipped: give the servos their "
                                     "own supply"))
            for line in spoken:
                theme.out(theme.note(f"robot said: {line}"))
            return 0
        theme.out("  " + theme.bad("no reply"))
        theme.out(theme.hint("check: laptop on the robot's Wi-Fi, address and port match the firmware, Windows "
                             "Firewall allows Python"))
        return 1
    finally:
        link.stop(send_ready=False)
        if fake is not None:
            fake.stop()


# ---------------------------------------------------------------------------- poses, sim
def _poses(args, ctx: Ctx) -> int:
    cfg = ctx.load()
    theme.title("Poses")
    rows = []
    for pose in POSES:
        command = TEXT_COMMAND.get(pose, "(none: this firmware has no ready position)")
        rows.append([POSE_NAME[pose].lower(), pose, command if cfg.robot.protocol == "rps_text"
                     else f"P,<seq>,{pose}"])
    theme.table(["pose", "letter", "sent as"], rows, aligns=["left", "left", "left"])
    theme.out(theme.dim(f"robot.protocol = {cfg.robot.protocol}"))
    return 0


def _sim(args, ctx: Ctx) -> int:
    """The fake robot the app also uses, so the whole thing can be tried without hardware."""
    cfg = ctx.load()
    theme.title("Simulated robot")
    theme.out(theme.hint(f"listening on 127.0.0.1:{cfg.robot.port}; Ctrl-C to stop"))
    fake = MockEsp(port=cfg.robot.port, verbose=True).start()
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        theme.out()
        theme.out(theme.ok(f"stopped after {fake.received} command(s)"))
    finally:
        fake.stop()
    return 0


# ---------------------------------------------------------------------------- finger
def _finger(args, ctx: Ctx) -> int:
    """The three servo channels the team firmware drives directly (ANGLE commands)."""
    cfg = ctx.load()
    if cfg.robot.protocol != "rps_text":
        die("Finger tuning needs the team firmware's ANGLE command: "
            "`rps config set robot.protocol=rps_text`")
    if not args.angles:
        theme.title("Finger channels")
        rows = [[name, f"{angle}°"] for (channel, name), angle in
                zip(sorted(FINGER_CHANNELS.items()), list(cfg.robot.finger_angles))]
        theme.table(["channel", "angle"], rows, aligns=["left", "right"])
        theme.out(theme.dim(f"{ANGLE_MIN}° to {ANGLE_MAX}°"))
        return 0
    angles = list(args.angles)
    if len(angles) != len(FINGER_CHANNELS):
        die(f"Give all {len(FINGER_CHANNELS)} angles, e.g. `rps bot finger 90 90 90`")
    if any(a < ANGLE_MIN or a > ANGLE_MAX for a in angles):
        die(f"Angles have to be between {ANGLE_MIN} and {ANGLE_MAX}")
    link, _ = _link(cfg, mock=args.mock)
    try:
        for channel, angle in enumerate(angles):
            link.send_raw(b"A,%d,%d\n" % (channel, angle))
            theme.out(f"  {theme.key(FINGER_CHANNELS[channel])} {theme.val(f'{angle}°')}")
        if args.save:
            cfg.robot.finger_angles = angles
            save_config(cfg, str(ctx.config), quiet=ctx.json)
            theme.out(theme.ok("kept as robot.finger_angles"))
    finally:
        link.stop(send_ready=False)
    return 0


# ---------------------------------------------------------------------------- moves
def _moves(args, ctx: Ctx) -> int:
    """The hand's move times: how long each change of pose takes, used by Evaluate."""
    cfg = ctx.load()
    if args.pairs:
        table = dict(cfg.latency.servo_transition_ms)
        for pair in args.pairs:
            if "=" not in pair or ">" not in pair.split("=", 1)[0]:
                die(f"Expected FROM>TO=MS, got '{pair}'. Example: N>R=120")
            path, ms = pair.split("=", 1)
            table[path] = float(ms)
        cfg.latency.servo_transition_ms = table
        save_config(cfg, str(ctx.config), quiet=ctx.json)
        theme.out(theme.ok(f"{len(table)} move times kept in latency.servo_transition_ms"))
    theme.title("Hand move times")
    rows = [[path.replace(">", " → "), f"{ms:.0f} ms"] for path, ms in cfg.latency.servo_transition_ms.items()]
    theme.table(["change", "time"], rows, aligns=["left", "right"])
    return 0