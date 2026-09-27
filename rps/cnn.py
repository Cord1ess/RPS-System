"""
RoshamboNet inference wrapper for 64x64 pseudo-DVS frames.

Checkpoints are dicts {"state_dict", "meta"}:
- v3 models from train.py: meta records the DVS emulator parameters used for the training frames,
  and the runtime warns on mismatch (a CNN is only valid for the event statistics it saw).
- meta["arch"] == "dextra": Dextra's pretrained Keras RoshamboNet imported by
  tools/import_dextra.py; its class order (paper, scissors, rock, background) is remapped here.
Anything else (e.g. a bare state_dict from the removed v2 system) is rejected.

cnn.rotate / cnn.flip orient the frame before inference, e.g. to match Dextra's side-on camera
view (fingers pointing left) when the webcam sees fingers pointing up.
"""

import glob
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model import CLASS_NAMES, DEXTRA_TO_OURS, DextraRoshamboNet, RoshamboNet
from rps.config import DvsConfig

DVS_META_KEYS = ("sensor_size", "frame_size", "log_offset", "contrast_threshold", "clip_count")
DVS_NAMES = {"sensor_size": "Motion grid", "frame_size": "Motion image size", "log_offset": "Dark noise damping",
             "contrast_threshold": "Motion sensitivity", "clip_count": "Per-pixel cap"}


class GestureCNN:
    def __init__(self, model_path: str, threads: int = 1, rotate: int = 0, flip: bool = False):
        torch.set_num_threads(threads)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
            raise ValueError(f"{model_path} is not a motion model made by train.py or tools/import_dextra.py")
        state, self.meta = ckpt["state_dict"], dict(ckpt.get("meta", {}))
        self.arch = self.meta.get("arch", "roshambo")
        if self.arch == "dextra":
            self.model = DextraRoshamboNet(num_classes=len(CLASS_NAMES))
            self.order = np.array(DEXTRA_TO_OURS)
        else:
            self.model = RoshamboNet(num_classes=len(CLASS_NAMES), pooling="avg")
            self.order = None
        self.model.load_state_dict(state)
        self.model.eval()
        self.divisor = float(self.meta.get("input_divisor", 255.0))
        # A model tuned on our recordings learned one orientation; it always uses that one.
        fixed = self.meta.get("orientation")
        self.fixed_orientation = fixed is not None
        self.rotate = int(fixed["rotate"] if fixed else rotate) % 360
        self.flip = bool(fixed["flip"] if fixed else flip)
        dummy = torch.zeros(1, 1, 64, 64)
        with torch.inference_mode():
            for _ in range(10):
                self.model(dummy)
        print(f"[cnn] Loaded {model_path} (arch {self.arch}, rotate {self.rotate}, flip {self.flip})")

    def check_dvs(self, dvs: DvsConfig) -> List[str]:
        """Lists emulator settings that differ from what the checkpoint was trained on."""
        trained = self.meta.get("dvs", {})
        return [f"'{DVS_NAMES[k]}' is {getattr(dvs, k)}, but the motion model was trained with {trained[k]}"
                for k in DVS_META_KEYS if k in trained and trained[k] != getattr(dvs, k)]

    def orient(self, frame64: np.ndarray) -> np.ndarray:
        if self.flip:
            frame64 = frame64[:, ::-1]
        if self.rotate:
            frame64 = np.rot90(frame64, k=self.rotate // 90)   # counter-clockwise
        return np.ascontiguousarray(frame64)

    def predict(self, frame64: np.ndarray) -> Tuple[int, float, np.ndarray, float]:
        """Returns (label, confidence, probabilities, inference_ms) in CLASS_NAMES order."""
        x = torch.from_numpy(self.orient(frame64)).float().div_(self.divisor).view(1, 1, 64, 64)
        t0 = time.perf_counter()
        with torch.inference_mode():
            probs = F.softmax(self.model(x), dim=1)[0].numpy()
        ms = (time.perf_counter() - t0) * 1000.0
        if self.order is not None:
            probs = probs[self.order]
        label = int(np.argmax(probs))
        return label, float(probs[label]), probs, ms


def describe_model(path: str) -> Dict:
    """
    What a motion model file is, from its stored details: kind "dextra" (as downloaded),
    "dextra_tuned" (tuned on our recordings), "scratch" (trained from nothing) or "unusable".
    """
    info = {"path": path.replace("\\", "/"), "file": os.path.basename(path), "detail": ""}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        meta = ckpt["meta"] if isinstance(ckpt, dict) and "state_dict" in ckpt else None
    except Exception:
        meta = None
    if meta is None:
        return {**info, "kind": "unusable", "name": f"Unusable file ({info['file']})"}
    people = meta.get("train_persons") or []
    parts = [f"{len(people)} {'person' if len(people) == 1 else 'people'}"] if people else []
    if meta.get("trained"):
        parts.append(str(meta["trained"])[:10])
    if meta.get("val_person") and meta.get("val_balanced_acc", -1) >= 0:
        parts.append(f"{meta['val_balanced_acc'] * 100:.0f}% on {meta['val_person']}")
    info["detail"] = ", ".join(parts)
    if meta.get("arch") == "dextra" and not meta.get("tuned_from"):
        return {**info, "kind": "dextra", "name": "Dextra, as downloaded"}
    if meta.get("arch") == "dextra":
        return {**info, "kind": "dextra_tuned", "name": "Dextra, tuned on your recordings"}
    return {**info, "kind": "scratch", "name": "Your network, trained from scratch"}


def list_models(folder: str = "models") -> List[Dict]:
    """Every usable motion model file: Dextra as downloaded first, then tuned (newest first), then scratch."""
    models = [describe_model(p) for p in glob.glob(os.path.join(folder, "*.pth"))]
    models = [m for m in models if m["kind"] != "unusable"]
    rank = {"dextra": 0, "dextra_tuned": 1, "scratch": 2}
    models.sort(key=lambda m: (rank[m["kind"]], -os.path.getmtime(m["path"])))
    for kind in ("dextra_tuned", "scratch"):          # tell several files of one kind apart
        same = [m for m in models if m["kind"] == kind]
        if len(same) > 1:
            for m in same:
                m["name"] += f" ({m['file']})"
    return models


def orient_batch(frames: np.ndarray, rotate: int, flip: bool) -> np.ndarray:
    """GestureCNN.orient for a (B, 64, 64) stack."""
    if flip:
        frames = frames[:, :, ::-1]
    if rotate % 360:
        frames = np.rot90(frames, k=(rotate % 360) // 90, axes=(1, 2))
    return np.ascontiguousarray(frames)


def predict_batch(cnn: "GestureCNN", frames: np.ndarray, rotate: int = 0, flip: bool = False,
                  batch: int = 512) -> np.ndarray:
    """Probabilities (B, 4) in CLASS_NAMES order for a stack of uint8 64x64 frames."""
    frames = orient_batch(frames, rotate, flip)
    out = []
    with torch.inference_mode():
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i:i + batch]).float().div_(cnn.divisor).unsqueeze(1)
            p = F.softmax(cnn.model(x), dim=1).numpy()
            out.append(p[:, cnn.order] if cnn.order is not None else p)
    return np.concatenate(out) if out else np.zeros((0, 4), np.float32)

