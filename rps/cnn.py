"""
RoshamboNet inference wrapper for 64x64 pseudo-DVS frames.

Checkpoints written by train.py (v3) are dicts {"state_dict", "meta"} where meta records the
DVS emulator parameters used to build the training frames; the runtime warns on mismatch,
because a CNN is only valid for the event statistics it was trained on.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model import CLASS_NAMES, RoshamboNet
from rps.config import DvsConfig

DVS_META_KEYS = ("sensor_size", "frame_size", "log_offset", "contrast_threshold", "clip_count")


class GestureCNN:
    def __init__(self, model_path: str, threads: int = 1):
        torch.set_num_threads(threads)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state, self.meta = ckpt["state_dict"], dict(ckpt.get("meta", {}))
        else:
            state, self.meta = ckpt, {"legacy_v2": True}
        self.model = RoshamboNet(num_classes=len(CLASS_NAMES), pooling="avg")
        self.model.load_state_dict(state)
        self.model.eval()
        dummy = torch.zeros(1, 1, 64, 64)
        with torch.inference_mode():
            for _ in range(10):
                self.model(dummy)
        print(f"[cnn] Loaded {model_path} (meta: {self.meta})")

    def check_dvs(self, dvs: DvsConfig) -> List[str]:
        """Lists emulator settings that differ from what the checkpoint was trained on."""
        if self.meta.get("legacy_v2"):
            return ["checkpoint is a v2 binary-mask model; it was not trained on pseudo-DVS frames"]
        trained = self.meta.get("dvs", {})
        return [f"dvs.{k}: runtime={getattr(dvs, k)} trained={trained[k]}"
                for k in DVS_META_KEYS if k in trained and trained[k] != getattr(dvs, k)]

    def predict(self, frame64: np.ndarray) -> Tuple[int, float, np.ndarray, float]:
        """Returns (label, confidence, probabilities, inference_ms) for one uint8 64x64 frame."""
        x = torch.from_numpy(frame64).float().div_(255.0).view(1, 1, 64, 64)
        t0 = time.perf_counter()
        with torch.inference_mode():
            probs = F.softmax(self.model(x), dim=1)[0].numpy()
        ms = (time.perf_counter() - t0) * 1000.0
        label = int(np.argmax(probs))
        return label, float(probs[label]), probs, ms


def summarize_meta(meta: Dict) -> str:
    dvs = meta.get("dvs", {})
    return ", ".join(f"{k}={dvs[k]}" for k in DVS_META_KEYS if k in dvs)
