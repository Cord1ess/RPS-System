import json
import os
from dataclasses import fields

import numpy as np
import pytest

from rps.camera import MockSource
from rps.config import ALLOWED, Config, apply_overrides, load_config, parse_set_args
from rps.hand_tracker import PAPER
from rps.pipeline import Pipeline, load_models


def test_set_args_coerce_types():
    cfg = Config()
    apply_overrides(cfg, parse_set_args(["vote.k=3", "decision.mode=continuous",
                                         "camera.lock_exposure=true", "dvs.contrast_threshold=0.25"]))
    assert cfg.vote.k == 3 and cfg.decision.mode == "continuous"
    assert cfg.camera.lock_exposure is True and cfg.dvs.contrast_threshold == 0.25


def test_unknown_keys_rejected():
    with pytest.raises(KeyError):
        apply_overrides(Config(), {"vote": {"kk": 1}})
    with pytest.raises(KeyError):
        apply_overrides(Config(), {"nope": {"k": 1}})


def test_invalid_choice_rejected_on_command_line_but_file_still_loads(tmp_path):
    with pytest.raises(ValueError):
        apply_overrides(Config(), parse_set_args(["decision.recognizer=fused"]))
    with pytest.raises(ValueError):
        apply_overrides(Config(), parse_set_args(["cnn.rotate=45"]))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"decision": {"recognizer": "fused", "mode": "continuous", "source": "fused",
                                             "pump_source": "auto", "rock_min_shoot_s": 0.15},
                                "cnn": {"model_path": "models/x.pth"}, "robot": {"enabled": True}}))
    cfg = load_config(str(path))                                       # old file: bad value + retired keys
    assert cfg.decision.recognizer == "both" and cfg.decision.mode == "continuous"


def test_ui_choices_match_the_allowed_values():
    from rps.ui.fields import CHOICES, FIELD_INFO
    for key, allowed in ALLOWED.items():
        assert key in CHOICES, key
        assert sorted(map(str, allowed)) == sorted(str(v) for v, _ in CHOICES[key]), key
    for section in fields(Config):              # every setting has a name and a hover explanation
        for f in fields(getattr(Config(), section.name)):
            assert (section.name, f.name) in FIELD_INFO, (section.name, f.name)
            assert FIELD_INFO[(section.name, f.name)][1], (section.name, f.name)


def test_repo_config_is_valid():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"), encoding="utf-8") as f:
        apply_overrides(Config(), json.load(f), strict=True)


class FakeCNN:
    def __init__(self):
        self.calls = 0

    def predict(self, frame64):
        self.calls += 1
        probs = np.array([0.02, 0.95, 0.02, 0.01], np.float32)
        return PAPER, 0.95, probs, 0.1


def test_pipeline_wiring_motion_to_pose():
    cfg = Config()
    cfg.decision.mode = "continuous"
    sent = []
    cnn = FakeCNN()
    pipeline = Pipeline(cfg, cnn=cnn, hand=None, pose_sink=sent.append)
    source = MockSource(realtime=False).start()
    for _ in range(45):   # 1.5 s of the pumping mock blob
        pipeline.step(source.read(), source.roi)
    assert cnn.calls >= 2                  # the emulator emitted frames for the CNN
    assert sent == ["S"]                   # paper -> robot scissors, sent exactly once
    assert pipeline.engine.snapshot().human == PAPER


def test_fused_runs_on_the_hand_tracker_when_there_is_no_motion_model(tmp_path):
    cfg = Config()
    if not os.path.exists(cfg.hand.model_path):
        pytest.skip("hand tracker model not downloaded")
    cfg.cnn.tuned_model = str(tmp_path / "not_trained_yet.pth")
    cnn, hand, messages = load_models(cfg, "both")
    try:
        assert cnn is None and hand is not None
        assert any("Playing with Mediapipe only" in m for m in messages)
        pipeline = Pipeline(cfg, cnn=None, hand=hand)
        source = MockSource(realtime=False).start()
        for _ in range(15):
            r = pipeline.step(source.read(), source.roi)
        assert r.cnn is None and r.hand is not None and r.snapshot is not None
    finally:
        hand.close()


def test_an_unusable_model_file_falls_back_or_explains(tmp_path):
    import torch
    from model import RoshamboNet
    bare = tmp_path / "old_v2.pth"
    torch.save(RoshamboNet().state_dict(), bare)            # no meta: not made by train.py/import_dextra
    cfg = Config()
    cfg.cnn.tuned_model = str(bare)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        load_models(cfg, "dextra_tuned")
    if not os.path.exists(cfg.hand.model_path):
        return
    cnn, hand, messages = load_models(cfg, "both")
    try:
        assert cnn is None and hand is not None and any("Mediapipe only" in m for m in messages)
    finally:
        hand.close()



def test_each_recognizer_uses_the_right_parts(tmp_path):
    from rps.pipeline import RECOGNIZERS, missing_reason, recognizer_parts
    cfg = Config()
    assert list(RECOGNIZERS.values()) == ["Dextra Raw", "Dextra Tuned", "Mediapipe", "Both (Dextra Tuned + Mediapipe)"]
    assert recognizer_parts(cfg, "dextra_raw") == (cfg.cnn.raw_model, False)
    assert recognizer_parts(cfg, "dextra_tuned") == (cfg.cnn.tuned_model, False)
    assert recognizer_parts(cfg, "mediapipe") == (None, True)
    assert recognizer_parts(cfg, "both") == (cfg.cnn.tuned_model, True)
    cfg.cnn.tuned_model = str(tmp_path / "none.pth")
    assert "Train page" in missing_reason(cfg, "dextra_tuned") and missing_reason(cfg, "mediapipe") is None
    with pytest.raises(RuntimeError, match="Train page"):
        load_models(cfg, "dextra_tuned")
