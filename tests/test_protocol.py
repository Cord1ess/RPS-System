import os
import socket
import time

import numpy as np
import pytest

from rps.robot_link import (ESP_RST_BROWNOUT, MockEsp, RobotLink, encode_ack, encode_angle, encode_led, encode_pose,
                            encode_text_pose, parse_message)


def test_round_trip_encoding():
    assert parse_message(encode_pose(7, "S", 1234)) == {"type": "P", "seq": 7, "pose": "S", "pc_ms": 1234}
    assert parse_message(encode_led(8, True)) == {"type": "L", "seq": 8, "on": True}
    assert parse_message(encode_ack(9, 555, 4)) == {"type": "A", "seq": 9, "esp_ms": 555, "reset": 4}


def test_malformed_messages_rejected():
    for bad in (b"", b"P,1,X,5", b"P,a,R,5", b"Z,1,2", b"\xff\xfe"):
        assert parse_message(bad) is None


def test_link_with_mock_esp_acks_and_heartbeats():
    port = 42177
    esp = MockEsp(port=port, verbose=False).start()
    link = RobotLink("127.0.0.1", port, heartbeat_s=0.05).start()
    try:
        link.send_pose("P")
        time.sleep(0.3)
        link.send_pose("R")
        time.sleep(0.2)
        stats = link.stats()
        assert esp.pose == "R"
        assert [p for _, p in esp.pose_log] == ["P", "R"]
        assert stats["acked"] >= 3                  # immediate sends + heartbeats
        assert stats["rtt_median_ms"] is not None and stats["rtt_median_ms"] < 50
    finally:
        link.stop(send_ready=False)
        esp.stop()


def fake_robot(port):
    """A raw UDP socket standing in for the ESP32, so tests can send any reply."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    s.settimeout(1.0)
    return s


def test_bad_pose_is_rejected_without_killing_the_link():
    esp = MockEsp(port=42178, verbose=False).start()
    link = RobotLink("127.0.0.1", 42178, heartbeat_s=0.05).start()
    try:
        with pytest.raises(ValueError):
            link.send_pose("X")
        sent = link.sent
        time.sleep(0.3)
        assert link._thread.is_alive() and link.sent > sent      # heartbeats continue
        assert esp.pose == "N"
    finally:
        link.stop(send_ready=False)
        esp.stop()


def test_acks_count_once_and_only_from_the_robot():
    robot = fake_robot(42179)
    stranger = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    link = RobotLink("127.0.0.1", 42179, heartbeat_s=10.0).start()
    try:
        seq = link.send_pose("R")
        _, addr = robot.recvfrom(256)
        for _ in range(3):                                          # duplicated reply
            robot.sendto(encode_ack(seq, 5000, 1), addr)
        stranger.sendto(encode_ack(seq, 1, 9), addr)                # not our robot
        time.sleep(0.2)
        s = link.stats()
        assert s["acked"] == 1 and s["sent"] == 1 and s["last_reset_reason"] == 1 and s["reboots"] == 0
        seq = link.send_pose("P")
        robot.recvfrom(256)
        robot.sendto(encode_ack(seq, 40, ESP_RST_BROWNOUT), addr)   # its clock restarted: a reboot
        time.sleep(0.2)
        s = link.stats()
        assert s["reboots"] == 1 and s["last_reset_reason"] == ESP_RST_BROWNOUT
    finally:
        link.stop(send_ready=False)
        robot.close()
        stranger.close()


def test_heartbeat_keeps_its_rhythm():
    robot = fake_robot(42180)
    link = RobotLink("127.0.0.1", 42180, heartbeat_s=0.1).start()
    try:
        times = []
        t_end = time.perf_counter() + 1.6
        while time.perf_counter() < t_end:
            try:
                robot.recvfrom(256)
                times.append(time.perf_counter())
            except socket.timeout:
                break
        gaps = np.diff(times)
        assert len(gaps) >= 12 and abs(float(np.mean(gaps)) - 0.1) < 0.012
    finally:
        link.stop(send_ready=False)
        robot.close()


def test_stop_sends_ready_and_closes():
    robot = fake_robot(42181)
    link = RobotLink("127.0.0.1", 42181, heartbeat_s=10.0).start()
    link.send_pose("S")
    robot.recvfrom(256)
    link.stop()
    assert parse_message(robot.recvfrom(256)[0])["pose"] == "N"
    assert link.sock.fileno() == -1
    robot.close()


# ----------------------------------------------------------------------------- team firmware: RPS:<GESTURE>

def received(sock, wait=0.3):
    """Every datagram that arrives within `wait` seconds."""
    out, end = [], time.perf_counter() + wait
    sock.settimeout(0.05)
    while time.perf_counter() < end:
        try:
            out.append(sock.recvfrom(256)[0])
        except socket.timeout:
            pass
    return out


def test_team_firmware_commands():
    assert encode_text_pose("R") == b"RPS:ROCK"
    assert encode_text_pose("P") == b"RPS:PAPER"
    assert encode_text_pose("S") == b"RPS:SCISSORS"
    assert encode_text_pose("N") is None                      # the firmware has no ready command
    with pytest.raises(ValueError):
        encode_text_pose("X")
    assert parse_message(b"RPS:SCISSORS") == {"type": "RPS", "pose": "S"}
    assert parse_message(b"RPS:LIZARD") is None


def test_team_firmware_link_sends_each_move_once_and_nothing_else():
    robot = fake_robot(42182)
    link = RobotLink("127.0.0.1", 42182, protocol="rps_text").start()
    try:
        link.send_pose("R")
        link.send_pose("N")                                   # ready: nothing to send
        link.send_pose("S")
        assert received(robot, 0.5) == [b"RPS:ROCK", b"RPS:SCISSORS"]    # no heartbeat repeats
        with pytest.raises(RuntimeError):
            link.send_led(True)
        s = link.stats()
        assert s["sent"] == 2 and s["replies"] is False and s["acked"] == 0
        link.stop()                                           # no ready command on stop either
        assert received(robot, 0.2) == []
    finally:
        robot.close()


def test_real_throws_reach_the_robot_as_team_commands():
    """The decision engine on the recorded 45 s session, wired to the link: 28 scissors throws -> 28 rocks."""
    from rps.config import DecisionConfig, VoteConfig
    from rps.decision import DecisionEngine, MotionObs
    from rps.hand_tracker import HandObs
    d = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "real_throws_scissors.npz"))
    m, h = d["motion"], d["hand"]
    robot = fake_robot(42183)
    link = RobotLink("127.0.0.1", 42183, protocol="rps_text").start()
    try:
        eng = DecisionEngine(DecisionConfig(mode="countdown"), VoteConfig(), use_cnn=False, use_mp=True)

        def v(x):
            return None if np.isnan(x) else float(x)

        for i in range(len(m)):
            t = float(m[i, 0])
            for pose in (eng.on_motion(MotionObs(i, t, int(m[i, 1]), v(m[i, 2]), None, v(m[i, 3]))),
                         eng.on_hand(t, HandObs(present=bool(h[i, 1]), gesture=int(h[i, 2]), confidence=float(h[i, 3]),
                                                wrist_y=v(h[i, 4]),
                                                box=None if np.isnan(h[i, 5]) else tuple(map(float, h[i, 5:9])),
                                                skipped=bool(h[i, 9])), None if np.isnan(h[i, 10]) else int(h[i, 10]))):
                if pose:
                    link.send_pose(pose)
        got = received(robot, 0.5)
        assert got == [b"RPS:ROCK"] * 28                      # rock beats scissors, once per throw
    finally:
        link.stop()
        robot.close()


def test_link_follows_the_configured_protocol():
    from rps.config import Config, apply_overrides, parse_set_args
    cfg = Config()
    assert cfg.robot.protocol == "rps_text" and cfg.robot.host == "192.168.0.126" and cfg.robot.port == 4210
    with pytest.raises(ValueError):
        apply_overrides(cfg, parse_set_args(["robot.protocol=json"]))
    robot = fake_robot(42184)
    cfg.robot.port = 42184
    link = RobotLink.from_config(cfg.robot, host="127.0.0.1").start()   # never the real robot in tests
    try:
        link.send_pose("P")
        assert received(robot, 0.3) == [b"RPS:PAPER"]
    finally:
        link.stop()
        robot.close()


def test_simulated_robot_understands_team_commands_and_stays_silent():
    esp = MockEsp(port=42185, verbose=False).start()
    link = RobotLink("127.0.0.1", 42185, protocol="rps_text").start()
    try:
        link.send_pose("S")
        link.send_pose("P")
        time.sleep(0.3)
        assert [p for _, p in esp.pose_log] == ["S", "P"] and esp.received == 2
        assert link.stats()["acked"] == 0                     # like the real firmware: no replies
    finally:
        link.stop()
        esp.stop()


def test_finger_angle_command_matches_the_team_tuning_tool():
    assert encode_angle(0, 0) == b"ANGLE:0,0" and encode_angle(2, 180) == b"ANGLE:2,180"
    for channel, angle in ((3, 90), (-1, 90), (1, 181), (1, -5), (1, 90.5), (1, True)):
        with pytest.raises(ValueError):
            encode_angle(channel, angle)
    assert parse_message(b"ANGLE:1,90") == {"type": "ANGLE", "channel": 1, "angle": 90}
    for bad in (b"ANGLE:1,190", b"ANGLE:4,90", b"ANGLE:1", b"ANGLE:a,b", b"ANGLE:1,2,3"):
        assert parse_message(bad) is None


def test_finger_angles_and_moves_share_one_socket_and_replies_come_back():
    esp = MockEsp(port=42186, verbose=False).start()
    said = []
    link = RobotLink("127.0.0.1", 42186, protocol="rps_text", on_text=said.append).start()
    try:
        assert link.send_raw(encode_angle(1, 120)) and link.send_raw(encode_angle(0, 0))
        link.send_pose("S")
        time.sleep(0.3)
        assert said == ["OK ANGLE:1,120", "OK ANGLE:0,0"]              # the confirmations, in order
        assert esp.angles == {1: 120, 0: 0} and esp.pose == "S"
    finally:
        link.stop()
        esp.stop()
    with pytest.raises(RuntimeError):
        RobotLink("127.0.0.1", 42187, protocol="ack").send_raw(b"ANGLE:0,0")     # the reference firmware has none
