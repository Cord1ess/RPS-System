"""train.py end to end on a tiny synthetic image set: tuning Dextra's model and training from scratch."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from rps.cnn import GestureCNN, describe_model, list_models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEXTRA = os.path.join(ROOT, "models", "dextra_roshambo.pth")


def tiny_frames(folder):
    """Two people, three gestures, 20 images each: a bar whose direction encodes the gesture."""
    rng = np.random.default_rng(0)
    files = []
    for person in ("ann", "bob"):
        os.makedirs(os.path.join(folder, person), exist_ok=True)
        frames, labels = [], []
        for label in range(3):
            for _ in range(20):
                img = np.zeros((64, 64), np.uint8)
                c = rng.integers(20, 44)
                if label == 0:
                    img[c - 3:c + 3, 10:54] = 255
                elif label == 1:
                    img[10:54, c - 3:c + 3] = 255
                else:
                    np.fill_diagonal(img[c - 20:, c - 20:], 255)
                frames.append(img)
                labels.append(label)
        path = os.path.join(person, "s__N1500_s1.npz")
        np.savez_compressed(os.path.join(folder, path), frames=np.array(frames), labels=np.array(labels, np.int8),
                            t=np.arange(len(labels), dtype=np.float32), flushed=np.zeros(len(labels), bool),
                            n_events=np.full(len(labels), 1500, np.int32))
        files.append({"path": path, "person": person, "session": "s", "type": "show", "label": "rock",
                      "event_count": 1500, "frame_skip": 1, "frames": len(labels), "flushed": 0})
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump({"dvs": {"frame_size": 64, "clip_count": 16}, "event_counts": [1500], "files": files}, f)


def run_train(tmp_path, *args):
    out = subprocess.run([sys.executable, "train.py", "--frames", str(tmp_path / "frames"), *args],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def test_tuning_dextra_keeps_its_orientation_and_is_listed_as_tuned(tmp_path):
    if not os.path.exists(DEXTRA):
        pytest.skip("Dextra's model not downloaded")
    tiny_frames(tmp_path / "frames")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"cnn": {"rotate": 90, "flip": True}}))
    out_path = tmp_path / "models" / "dextra_tuned.pth"
    log = run_train(tmp_path, "--start_from", "dextra", "--config", str(cfg), "--all", "--epochs", "1",
                    "--output", str(out_path))
    assert "Tuning Dextra's model; images turned 90 degrees, mirrored True" in log
    meta = torch.load(out_path, weights_only=True)["meta"]
    assert meta["arch"] == "dextra" and meta["tuned_from"] and meta["orientation"] == {"rotate": 90, "flip": True}
    cnn = GestureCNN(str(out_path), rotate=0, flip=False)       # the settings passed in must not win
    assert cnn.fixed_orientation and cnn.rotate == 90 and cnn.flip is True and cnn.divisor == 256.0
    assert describe_model(str(out_path))["kind"] == "dextra_tuned"
    # the first three convolution layers were kept as downloaded
    base = torch.load(DEXTRA, weights_only=True)["state_dict"]
    tuned = torch.load(out_path, weights_only=True)["state_dict"]
    assert torch.equal(base["convs.0.weight"], tuned["convs.0.weight"])
    assert not torch.equal(base["fc.weight"], tuned["fc.weight"])


def test_training_from_scratch_is_used_unrotated(tmp_path):
    tiny_frames(tmp_path / "frames")
    out_path = tmp_path / "scratch.pth"
    run_train(tmp_path, "--start_from", "scratch", "--val_person", "bob", "--epochs", "2", "--output", str(out_path))
    cnn = GestureCNN(str(out_path), rotate=90, flip=True)      # Dextra's orientation settings do not apply
    assert cnn.fixed_orientation and cnn.rotate == 0 and cnn.flip is False
    assert describe_model(str(out_path))["kind"] == "scratch"


def test_model_list_names_each_kind(tmp_path):
    if not os.path.exists(DEXTRA):
        pytest.skip("Dextra's model not downloaded")
    import shutil
    shutil.copy(DEXTRA, tmp_path / "dextra_roshambo.pth")
    (tmp_path / "broken.pth").write_bytes(b"not a model")
    names = [m["name"] for m in list_models(str(tmp_path))]
    assert names == ["Dextra, as downloaded"]
