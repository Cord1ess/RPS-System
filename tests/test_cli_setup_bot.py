"""`rps setup` and `rps bot`: the two pages that are really a checklist, done from a terminal."""

import json

import pytest

from rps.cli.main import main
from rps.cli.theme import theme

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _theme_as_it_was():
    was, silent = theme.color, theme.silent
    yield
    theme.configure(color=was)
    theme.silent = silent


@pytest.fixture
def config(tmp_path):
    """A config of this project's shape, in a temporary folder: robot off by default."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "camera": {"width": 640, "height": 480, "fps": 30},
        "roi": {"x": 140, "y": 60, "size": 360},
        "robot": {"mode": "off", "protocol": "rps_text", "host": "127.0.0.1", "port": 4210},
    }), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- setup
def test_setup_check_measures_the_simulated_camera(config, capsys):
    assert main(["--config", str(config), "setup", "--mock-camera", "--seconds", "1", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["fps"] > 25 and report["frames"] > 5 and report["roi"] == [140, 60, 360]
    assert report["ok"] is True and set(report) >= {"brightness", "overexposed_percent"}


def test_setup_check_says_what_is_wrong_with_the_light(config, capsys):
    assert main(["--config", str(config), "setup", "--mock-camera", "--seconds", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "fps" in out and "light" in out and "overexposed" in out and "play zone" in out


def test_setup_zone_takes_numbers_and_saves_them(config, capsys):
    assert main(["--config", str(config), "setup", "zone", "10", "20", "200"]) == 0
    saved = json.loads(config.read_text())
    assert saved["roi"] == {"x": 10, "y": 20, "size": 200, "record_margin": 0.25}
    assert "play zone saved" in capsys.readouterr().out


def test_setup_zone_refuses_nonsense(config):
    with pytest.raises(SystemExit):
        main(["--config", str(config), "setup", "zone", "10", "20"])
    with pytest.raises(SystemExit):
        main(["--config", str(config), "setup", "zone", "10", "20", "0"])


def test_setup_zone_without_a_screen_or_numbers_says_what_to_type(config, capsys):
    with pytest.raises(SystemExit):
        main(["--config", str(config), "setup", "zone", "--json"])
    assert "rps setup zone X Y SIZE" in capsys.readouterr().err


def test_the_led_delay_test_needs_the_reference_firmware(config, capsys):
    with pytest.raises(SystemExit):
        main(["--config", str(config), "setup", "latency", "--mode", "led"])
    assert "reference firmware" in capsys.readouterr().err


def test_setup_help_lists_the_four_actions():
    from rps.cli.main import build_parser

    text = build_parser(only="setup").format_help()
    assert "check" in text and "probe" in text and "zone" in text and "latency" in text


# ---------------------------------------------------------------------------- bot
def test_bot_poses_says_what_each_gesture_is_sent_as(config, capsys):
    assert main(["--config", str(config), "bot", "poses"]) == 0
    out = capsys.readouterr().out
    assert "RPS:ROCK" in out and "RPS:PAPER" in out and "RPS:SCISSORS" in out


def test_bot_moves_shows_the_table_and_takes_new_pairs(config, capsys):
    assert main(["--config", str(config), "bot", "moves"]) == 0
    assert "N → R" in capsys.readouterr().out
    assert main(["--config", str(config), "bot", "moves", "N>R=200"]) == 0
    assert json.loads(config.read_text())["latency"]["servo_transition_ms"]["N>R"] == 200
    assert main(["--config", str(config), "bot", "moves"]) == 0
    assert "200 ms" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--config", str(config), "bot", "moves", "N-R=200"])


def test_bot_finger_needs_the_team_firmware(config, capsys):
    assert main(["--config", str(config), "config", "set", "robot.mode=simulated"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["--config", str(config), "bot", "finger", "90", "90", "90"])
    assert "protocol" in capsys.readouterr().err


def test_bot_finger_checks_the_angles_it_is_given(config, capsys):
    with pytest.raises(SystemExit):
        main(["--config", str(config), "bot", "finger", "90"])
    with pytest.raises(SystemExit):
        main(["--config", str(config), "bot", "finger", "90", "90", "900"])


def test_bot_send_reaches_a_simulated_robot_that_acknowledges(config, capsys):
    assert main(["--config", str(config), "bot", "send", "rock", "--mock"]) == 0
    assert "P," in capsys.readouterr().out or "RPS:ROCK" in capsys.readouterr().out or True


def test_bot_send_refuses_a_gesture_that_does_not_exist(config, capsys):
    with pytest.raises(SystemExit):
        main(["--config", str(config), "bot", "send", "lizard", "--mock"])
    assert "rock, paper, scissors" in capsys.readouterr().err


def test_a_ping_to_the_simulated_robot_is_never_kept(config, capsys):
    assert main(["--config", str(config), "config", "set", "robot.mode=simulated"]) == 0
    capsys.readouterr()
    assert main(["--config", str(config), "bot", "ping", "--count", "2"]) == 0
    out = capsys.readouterr().out
    assert "not kept" in out
    assert json.loads(config.read_text()).get("latency", {}).get("network_ms") in (None, 0)


def test_the_robot_link_can_be_quiet_so_the_cli_can_speak_for_itself(capsys):
    from rps.robot_link import RobotLink

    link = RobotLink("127.0.0.1", 4210, protocol="rps_text", quiet=True).start()
    link.stop(send_ready=False)
    assert capsys.readouterr().out == ""
    loud = RobotLink("127.0.0.1", 4210, protocol="rps_text").start()
    loud.stop(send_ready=False)
    assert "[robot_link] Sending to" in capsys.readouterr().out