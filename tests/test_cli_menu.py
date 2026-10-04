"""`rps` with no arguments: the menu. Numbers only, so it works over SSH on a Pi with nothing
installed but Python."""

import json

import pytest

from rps.cli import menu
from rps.cli.common import Ctx
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
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"robot": {"mode": "off", "host": "127.0.0.1", "port": 4210}}), encoding="utf-8")
    return path


@pytest.fixture
def ctx(config):
    return Ctx(config=config)


class TTY:
    """Stands in for a terminal, so the menu sees someone there to answer it."""

    def isatty(self):
        return True


@pytest.fixture
def someone_there(monkeypatch):
    monkeypatch.setattr("sys.stdin", TTY())


def answers(script):
    """A stand-in for someone typing: the numbers they would press, then the list runs out loudly
    rather than hanging if the menu asks for more than they expected."""
    todo = list(script)

    def typed(prompt=""):
        assert todo, f"the menu asked for another answer ({prompt!r}); the script ran out"
        return todo.pop(0)

    return typed


def test_no_arguments_and_nobody_there_prints_the_help(capsys, config):
    assert main(["--config", str(config)]) == 0              # not a terminal: nobody to pick a number
    assert "usage: rps" in capsys.readouterr().out


def test_typing_menu_is_the_same_as_typing_nothing_and_a_number(ctx, monkeypatch, someone_there):
    went = []
    monkeypatch.setattr(menu, "run_menu", lambda c: went.append(c.config) or 0)
    assert main(["--config", str(ctx.config), "menu"]) == 0        # the word
    assert main(["--config", str(ctx.config)]) == 0                # and no word at all
    assert went == [ctx.config, ctx.config]


def test_quitting_from_the_main_menu_says_bye(ctx, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", answers(["6"]))
    assert menu.run_menu(ctx) == 0
    assert "Bye" in capsys.readouterr().out


def test_a_number_that_is_not_on_the_menu_does_nothing(ctx, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", answers(["42", "6"]))
    assert menu.run_menu(ctx) == 0
    assert "Choose 1 to 6" in capsys.readouterr().out


@pytest.mark.parametrize("key", ["q", "quit", "exit"])
def test_the_way_out_is_spelled_out_as_well(ctx, monkeypatch, key):
    monkeypatch.setattr("builtins.input", answers([key]))
    assert menu.run_menu(ctx) == 0


def test_the_main_menu_offers_the_five_things_and_quit(capsys, ctx):
    menu.show_main(menu.Recent(ctx), ctx)
    text = capsys.readouterr().out
    for line in ("Setup", "Bot", "Config", "Play", "Status", "Quit"):
        assert line in text


def test_setup_runs_the_same_command_the_keyboard_would(ctx, monkeypatch):
    ran = []
    monkeypatch.setattr(menu, "_dispatch", lambda argv, c: ran.append(argv))
    monkeypatch.setattr("builtins.input", answers(["1", "", "5"]))     # check the camera, then Back
    menu.menu_setup(ctx)
    assert ran == [["setup", "check"]]


def test_the_play_zone_can_be_typed_as_three_numbers(ctx, monkeypatch):
    ran = []
    monkeypatch.setattr(menu, "_dispatch", lambda argv, c: ran.append(argv))
    monkeypatch.setattr("builtins.input", answers(["3", "100 80 300", "", "5"]))
    menu.menu_setup(ctx)
    assert ran == [["setup", "zone", "100", "80", "300"]]


def test_three_numbers_the_wrong_way_round_are_refused(ctx, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(menu, "_dispatch", lambda argv, c: ran.append(argv))
    monkeypatch.setattr("builtins.input", answers(["3", "a b c", "", "", "5"]))
    menu.menu_setup(ctx)
    out = capsys.readouterr().out
    assert ran == [["setup", "zone"]]              # the blank answer means "drag on the video"
    assert "whole numbers" in out


def test_the_wrong_number_of_numbers_is_refused_too(ctx, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(menu, "_dispatch", lambda argv, c: ran.append(argv))
    monkeypatch.setattr("builtins.input", answers(["3", "10 20", "100 80 300", "", "5"]))
    menu.menu_setup(ctx)
    assert ran == [["setup", "zone", "100", "80", "300"]] and "wants 3" in capsys.readouterr().out


def test_the_camera_delay_asks_which_way_to_measure_it(ctx, monkeypatch):
    ran = []
    monkeypatch.setattr(menu, "_dispatch", lambda argv, c: ran.append(argv))
    monkeypatch.setattr("builtins.input", answers(["4", "2", "", "5"]))   # measure it by LED
    menu.menu_setup(ctx)
    assert ran == [["setup", "latency", "--mode", "led"]]


def test_config_get_and_set_go_through_the_real_parser(ctx, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", answers(["2", "robot.host", "", "6"]))
    menu.menu_config(ctx)
    assert "127.0.0.1" in capsys.readouterr().out.splitlines()      # the bare value, on its own line


def test_config_set_writes_the_file(ctx, monkeypatch):
    monkeypatch.setattr("builtins.input", answers(["3", "robot.host=10.0.0.9", "", "6"]))
    menu.menu_config(ctx)
    assert json.loads(ctx.config.read_text())["robot"]["host"] == "10.0.0.9"


def test_bot_poses_is_the_typed_command(ctx, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", answers(["3", "", "6"]))
    menu.menu_bot(ctx)
    assert "RPS:ROCK" in capsys.readouterr().out


def test_a_simulated_robot_can_be_asked_for(ctx, monkeypatch):
    ran = []
    monkeypatch.setattr(menu, "_dispatch", lambda argv, c: ran.append(argv))
    monkeypatch.setattr("builtins.input", answers(["2", "5", "y", "", "6"]))
    menu.menu_bot(ctx)
    assert ran == [["bot", "send", "scissors", "--mock"]]


def test_the_quick_test_needs_no_camera_and_no_robot(ctx, monkeypatch):
    ran = []
    monkeypatch.setattr(menu, "_dispatch", lambda argv, c: ran.append(argv))
    monkeypatch.setattr("builtins.input", answers(["2", "", "7"]))
    menu.menu_play(ctx, menu.Recent(ctx))
    assert ran == [["play", "--recognizer", "mediapipe", "--mock-camera", "--mock-esp",
                    "--mode", "continuous", "--no-hud", "--headless", "120"]]


def test_playing_uses_what_was_picked_and_then_goes_back(ctx, monkeypatch):
    ran = []
    monkeypatch.setattr(menu, "_dispatch", lambda argv, c: ran.append(argv))
    recent = menu.Recent(ctx)
    recent.set("recognizer", "mediapipe")
    recent.set("mode", "continuous")
    recent.set("hud", False)
    monkeypatch.setattr("builtins.input", answers(["1"]))
    menu.menu_play(ctx, recent)
    assert ran == [["play", "--recognizer", "mediapipe", "--mode", "continuous",
                    "--robot-plays", "lose", "--no-hud"]]


def test_the_hud_can_be_turned_off_and_on_again(ctx):
    recent = menu.Recent(ctx)
    recent.set("hud", True)
    monkeypatch_input = answers(["6", "7"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("builtins.input", monkeypatch_input)
        menu.menu_play(ctx, recent)
    assert recent.get("hud") is False


def test_a_word_is_accepted_where_a_number_would_do(ctx, monkeypatch):
    recent = menu.Recent(ctx)
    monkeypatch.setattr("builtins.input", answers(["3", "dextra_raw", "7"]))
    menu.menu_play(ctx, recent)
    assert recent.get("recognizer") == "dextra_raw"


def test_what_was_picked_last_time_is_remembered(ctx):
    recent = menu.Recent(ctx)
    assert recent.get("recognizer") == "mediapipe"          # the first time: the safe default
    recent.set("recognizer", "dextra_tuned")
    recent.set("mode", "guided")
    assert menu.Recent(ctx).get("recognizer") == "dextra_tuned"
    assert menu.Recent(ctx).get("mode") == "guided"


def test_the_remembered_choices_survive_a_file_that_is_not_ours(ctx):
    recent = menu.Recent(ctx)
    recent.path.write_text("not json at all", encoding="utf-8")
    assert menu.Recent(ctx).get("recognizer") == "mediapipe"     # a bad file is ignored, not fatal


def test_the_remembered_choices_are_not_part_of_the_settings(ctx):
    menu.Recent(ctx).set("recognizer", "both")
    assert "menu_recent" not in ctx.config.read_text()


def test_stopping_the_game_with_ctrl_c_lands_back_in_the_menu(ctx, capsys, monkeypatch):
    def interrupt(argv, c):
        raise KeyboardInterrupt
    monkeypatch.setattr(menu, "_dispatch", interrupt)
    monkeypatch.setattr("builtins.input", answers(["4", "1", "6"]))
    assert menu.run_menu(ctx) == 0
    assert "Back to the menu" in capsys.readouterr().out


def test_a_cancelled_command_is_not_reported_as_a_failure(ctx, capsys, monkeypatch):
    import rps.cli.setup as setup_module

    def interrupt(args, c):
        raise KeyboardInterrupt
    monkeypatch.setattr(setup_module, "_check", interrupt)
    monkeypatch.setattr("builtins.input", answers(["1", "1", "", "5", "6"]))
    assert menu.run_menu(ctx) == 0
    assert "KeyboardInterrupt" not in capsys.readouterr().out


def test_status_says_where_everything_is(ctx, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", answers([""]))
    menu.menu_status(menu.Recent(ctx), ctx)
    text = capsys.readouterr().out
    assert str(ctx.config) in text                  # the settings file
    assert "127.0.0.1" in text                      # where the robot is
    assert "play zone" in text                      # and the square the app looks at


def test_the_menu_survives_a_command_that_breaks(ctx, capsys, monkeypatch):
    import rps.cli.setup as setup_module

    def explode(args, c):
        raise RuntimeError("the camera fell over")
    monkeypatch.setattr(setup_module, "_check", explode)         # the real command, made to fail
    monkeypatch.setattr("builtins.input", answers(["1", "1", "", "5", "6"]))
    assert menu.run_menu(ctx) == 0                  # one bad menu entry is not the end of the menu
    assert "the camera fell over" in capsys.readouterr().out