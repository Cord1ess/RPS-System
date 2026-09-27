"""
Imports Dextra's pretrained RoshamboNet (SensorsINI/dextra-roshambo-python, trained on the
ROSHAMBO17 DVS dataset: ~20 people, ~5M frames) into a v3 checkpoint usable by play.py.

The Dextra repository publishes quantized numpy weights (integer kernels + power-of-two shifts);
this converts them to float PyTorch weights, checks them on Dextra's own sample frames, and
writes models/dextra_roshambo.pth with meta {"arch": "dextra"} so rps/cnn.py remaps the class
order (Dextra: paper, scissors, rock, background).

Note: the Dextra repository has no license file, so its weights are not redistributed here
(models/dextra* is git-ignored). The ROSHAMBO17 dataset itself is CC BY-SA 4.0.

Usage:
    python tools/import_dextra.py
    python play.py --source cnn --set cnn.model_path=models/dextra_roshambo.pth
"""

import argparse
import glob
import os
import subprocess
import sys
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DEXTRA_CLASS_ORDER, DextraRoshamboNet  # noqa: E402

REPO = "SensorsINI/dextra-roshambo-python"
RAW = f"https://raw.githubusercontent.com/{REPO}/master"
WEIGHT_FILES = [f"layer{i}_{k}.npy" for i in range(1, 7) for k in ("kernel", "bias")] + \
               ["shift_per_layer_kernel.npy", "shift_per_layer_bias.npy"]


def download(dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    for name in WEIGHT_FILES:
        path = os.path.join(dest_dir, name)
        if not os.path.exists(path):
            print(f"[import_dextra] downloading {name}")
            urllib.request.urlretrieve(f"{RAW}/model/numpy_weights/{name}", path)


def convert(weights_dir: str) -> DextraRoshamboNet:
    sk = np.load(os.path.join(weights_dir, "shift_per_layer_kernel.npy"))
    sb = np.load(os.path.join(weights_dir, "shift_per_layer_bias.npy"))
    net = DextraRoshamboNet()
    layers = list(net.convs) + [net.fc]
    with torch.no_grad():
        for i, layer in enumerate(layers):
            k = np.load(os.path.join(weights_dir, f"layer{i + 1}_kernel.npy")).astype(np.float32) / 2.0 ** sk[i]
            b = np.load(os.path.join(weights_dir, f"layer{i + 1}_bias.npy")).astype(np.float32) / 2.0 ** sb[i]
            # Keras layouts: conv (kh, kw, cin, cout), dense (in, out)
            w = k.transpose(3, 2, 0, 1) if k.ndim == 4 else k.T
            layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))
            layer.bias.copy_(torch.from_numpy(b))
    return net.eval()


def check_samples(net: DextraRoshamboNet, sample_dir: str):
    files = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
    if not files:
        print("[import_dextra] no sample frames found; skipping the sanity check")
        return
    imgs = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])
    with torch.inference_mode():
        p = F.softmax(net(torch.from_numpy(imgs).float().div(256.0).unsqueeze(1)), 1).numpy()
    lab = p.argmax(1)
    dist = ", ".join(f"{DEXTRA_CLASS_ORDER[c]} {np.mean(lab == c) * 100:.0f}%" for c in range(4))
    print(f"[import_dextra] {len(files)} Dextra sample frames: mean confidence {p.max(1).mean():.2f}; {dist}")


def main():
    parser = argparse.ArgumentParser(description="Import Dextra's pretrained RoshamboNet")
    parser.add_argument("--weights_dir", default="models/dextra/numpy_weights")
    parser.add_argument("--samples_dir", default="models/dextra/sample_frames")
    parser.add_argument("--output", default="models/dextra_roshambo.pth")
    args = parser.parse_args()

    download(args.weights_dir)
    net = convert(args.weights_dir)
    check_samples(net, args.samples_dir)
    try:
        commit = subprocess.run(["gh", "api", f"repos/{REPO}/commits/master", "--jq", ".sha"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    meta = {
        "arch": "dextra",
        "class_order": DEXTRA_CLASS_ORDER,
        "input_divisor": 256.0,
        "dvs": {"frame_size": 64, "clip_count": 16},
        "source": f"https://github.com/{REPO} model/numpy_weights" + (f" @ {commit[:12]}" if commit else ""),
        "trained_on": "ROSHAMBO17 DVS recordings (Lungu, Corradi, Delbruck, ISCAS 2017)",
    }
    torch.save({"state_dict": net.state_dict(), "meta": meta}, args.output)
    print(f"[import_dextra] saved {os.path.abspath(args.output)}")
    print("[import_dextra] try it live:  python play.py --source cnn --mode continuous "
          f"--set cnn.model_path={args.output}")


if __name__ == "__main__":
    main()
