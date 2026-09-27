import numpy as np
import torch

from model import DEXTRA_CLASS_ORDER, DextraRoshamboNet
from rps.cnn import GestureCNN
from rps.hand_tracker import PAPER, ROCK


def forced_checkpoint(path, dextra_class: int):
    net = DextraRoshamboNet()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.fc.bias[dextra_class] = 10.0
    torch.save({"state_dict": net.state_dict(), "meta": {"arch": "dextra", "input_divisor": 256.0,
                                                         "dvs": {"frame_size": 64, "clip_count": 16}}}, path)


def test_dextra_class_order_is_remapped(tmp_path):
    for dextra_name, ours in (("paper", PAPER), ("rock", ROCK)):
        path = tmp_path / f"{dextra_name}.pth"
        forced_checkpoint(path, DEXTRA_CLASS_ORDER.index(dextra_name))
        label, conf, probs, _ = GestureCNN(str(path)).predict(np.zeros((64, 64), np.uint8))
        assert label == ours and conf > 0.99 and abs(probs.sum() - 1.0) < 1e-5


def test_orientation_options(tmp_path):
    path = tmp_path / "m.pth"
    forced_checkpoint(path, 0)
    frame = np.zeros((64, 64), np.uint8)
    frame[0, :10] = 255                                   # a stroke along the top-left edge
    rotated = GestureCNN(str(path), rotate=90).orient(frame)
    flipped = GestureCNN(str(path), flip=True).orient(frame)
    assert rotated[63, 0] == 255 and rotated[0, 0] == 0   # CCW rotation moves the top row to the left column
    assert flipped[0, 63] == 255 and flipped[0, 0] == 0
