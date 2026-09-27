import socket
import time

import numpy as np
import pytest

from rps.robot_link import (ESP_RST_BROWNOUT, MockEsp, RobotLink, encode_ack, encode_led, encode_pose,
                            parse_message)


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
