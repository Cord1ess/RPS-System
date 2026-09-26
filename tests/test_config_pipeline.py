import numpy as np
import pytest

from rps.camera import MockSource
from rps.config import Config, apply_overrides, parse_set_args
from rps.hand_tracker import PAPER
from rps.pipeline import Pipeline


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
