import time

from rps.robot_link import (MockEsp, RobotLink, encode_ack, encode_led, encode_pose,
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
