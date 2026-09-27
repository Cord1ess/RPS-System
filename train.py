"""
Trains RoshamboNet on pseudo-DVS frames produced by build_dataset.py.

Key differences from the v2 trainer (legacy/train_v2.py):
1. Split by PERSON, never by frame. Consecutive frames of one burst are near-duplicates, so a
   per-frame split leaks and inflates accuracy. --val_person holds one person out; --lopo runs
   leave-one-person-out and reports mean +/- std; --all trains on everyone for the final model.
2. Inputs are graded constant-event-count DVS frames (Dextra normalization), not binary masks.
3. Augmentations act on event frames: left/right flip (other hand), small rotation/translation/
   scale, event dropout, injected noise events. Frame-skip and multi-N variants come from
   build_dataset.py.
4. Class weights from class frequencies; model selection by balanced accuracy.
5. The checkpoint stores the DVS emulator parameters so the runtime can warn on mismatch.

Usage:
    python train.py --val_person alice
    python train.py --lopo
    python train.py --all --epochs 30
"""

import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from model import CLASS_NAMES, RoshamboNet


def load_frames(frames_dir: str, event_counts: List[int] = None) -> Tuple[List[Dict], Dict]:
    index_path = os.path.join(frames_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{index_path} not found. Run build_dataset.py first.")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    parts = []
    for entry in index["files"]:
        if event_counts and entry["event_count"] not in event_counts:
            continue
        data = np.load(os.path.join(frames_dir, entry["path"]))
        if len(data["labels"]) == 0:
            continue
        parts.append({**entry, "x": data["frames"], "y": data["labels"].astype(np.int64)})
    if not parts:
        raise ValueError("No frames matched the selection.")
    return parts, index


def stack(parts: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    return np.concatenate([p["x"] for p in parts]), np.concatenate([p["y"] for p in parts])


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 1, 64, 64) in [0, 1] (counts / K)."""
    b = x.shape[0]
    flip = torch.rand(b) < 0.5
    x = torch.where(flip.view(b, 1, 1, 1), x.flip(-1), x)
    ang = (torch.rand(b) * 2 - 1) * np.deg2rad(15.0)
    scale = 1.0 + (torch.rand(b) * 2 - 1) * 0.10
    shift = (torch.rand(b, 2) * 2 - 1) * 0.16          # 8 % of width in [-1, 1] coordinates
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    drop_p = torch.rand(b, 1, 1, 1) * 0.2
    x = x * (torch.rand_like(x) >= drop_p)
    noise_p = torch.rand(b, 1, 1, 1) * 0.005
    noise = (torch.rand_like(x) < noise_p).float() * torch.randint(1, 3, x.shape).float() / 16.0
    return torch.clamp(x + noise, 0.0, 1.0)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().div_(255.0).unsqueeze(1)


def predict(model: nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            preds.append(model(to_tensor(x[i:i + batch])).argmax(1).numpy())
    return np.concatenate(preds) if preds else np.zeros(0, np.int64)


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    recalls = [np.mean(p[y == c] == c) for c in np.unique(y)]
    return float(np.mean(recalls)) if recalls else 0.0


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CLASS_NAMES)).astype(np.float64)
    w = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1))
    return torch.tensor(np.clip(w, 0.25, 4.0), dtype=torch.float32)


def train_fold(train_x, train_y, val_x, val_y, args, tag: str):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = RoshamboNet(num_classes=len(CLASS_NAMES), pooling="avg", dropout=0.1)
    weights = class_weights(train_y)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    print(f"[train:{tag}] train {len(train_y)} frames {dict(sorted(Counter(train_y.tolist()).items()))} | "
          f"val {len(val_y) if val_y is not None else 0} | class weights {np.round(weights.numpy(), 2).tolist()}")

    history = {"train_loss": [], "train_acc": [], "val_bal_acc": []}
    best_state, best_bal = None, -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_y))
        run_loss = correct = seen = 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb, yb = augment(to_tensor(train_x[idx])), torch.from_numpy(train_y[idx])
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(idx)
        scheduler.step()
        history["train_loss"].append(run_loss / max(seen, 1))
        history["train_acc"].append(correct / max(seen, 1))
        msg = f"Epoch [{epoch:02d}/{args.epochs}] loss {history['train_loss'][-1]:.4f} acc {history['train_acc'][-1] * 100:5.1f}%"
        if val_y is not None and len(val_y):
            bal = balanced_accuracy(val_y, predict(model, val_x))
            history["val_bal_acc"].append(bal)
            msg += f" || val balanced acc {bal * 100:5.1f}%"
            if bal > best_bal:
                best_bal, best_state = bal, {k: v.clone() for k, v in model.state_dict().items()}
                msg += " <-- BEST"
        print(msg)
    if best_state is None:   # --all: keep the final epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history, best_bal


def report(model, parts_val: List[Dict], plot_path: str, history: Dict):
    x, y = stack(parts_val)
    p = predict(model, x)
    present = sorted(set(y.tolist()) | set(p.tolist()))
    print("\n--- Held-out classification report (best checkpoint) ---")
    print(classification_report(y, p, labels=present, target_names=[CLASS_NAMES[i] for i in present],
                                zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y, p, labels=present))
    print("\nBalanced accuracy by event count N and session type:")
    for key in ("event_count", "type"):
        for val in sorted({part[key] for part in parts_val}):
            sub = [part for part in parts_val if part[key] == val]
            xs, ys = stack(sub)
            print(f"  {key}={val}: {balanced_accuracy(ys, predict(model, xs)) * 100:5.1f}% ({len(ys)} frames)")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], color="crimson", label="Train loss")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.grid(True, alpha=0.3); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], color="crimson", label="Train acc")
    if history["val_bal_acc"]:
        plt.plot(history["val_bal_acc"], color="royalblue", linestyle="--", label="Held-out person (balanced)")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[train] Plot saved to {os.path.abspath(plot_path)}")


def save_checkpoint(model, path: str, index: Dict, args, persons: List[str], val_person, val_bal):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = {
        "dvs": index["dvs"],
        "event_counts": args.event_counts_list or index["event_counts"],
        "train_persons": persons,
        "val_person": val_person or "",
        "val_balanced_acc": float(val_bal) if val_bal is not None else -1.0,
        "classes": list(CLASS_NAMES),
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    print(f"[train] Checkpoint saved to {os.path.abspath(path)}")


def main():
    parser = argparse.ArgumentParser(description="Train RoshamboNet on pseudo-DVS frames")
    parser.add_argument("--frames", default="data/frames")
    parser.add_argument("--val_person", default=None)
    parser.add_argument("--lopo", action="store_true", help="Leave-one-person-out evaluation")
    parser.add_argument("--all", action="store_true", help="Train on every person (final model)")
    parser.add_argument("--event_counts", default="", help="Restrict to these N, e.g. 1500")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/motion_cnn_v3.pth")
    parser.add_argument("--metrics_plot", default="models/training_metrics_v3.png")
    args = parser.parse_args()
    args.event_counts_list = [int(v) for v in args.event_counts.split(",") if v]

    parts, index = load_frames(args.frames, args.event_counts_list)
    persons = sorted({p["person"] for p in parts})
    print(f"[train] {sum(len(p['y']) for p in parts)} frames from {len(persons)} people: {persons}")

    if args.lopo:
        if len(persons) < 2:
            raise SystemExit("--lopo needs at least 2 people.")
        scores = {}
        for person in persons:
            tr = [p for p in parts if p["person"] != person]
            va = [p for p in parts if p["person"] == person]
            (tx, ty), (vx, vy) = stack(tr), stack(va)
            _, _, best = train_fold(tx, ty, vx, vy, args, tag=f"holdout={person}")
            scores[person] = best
        print("\n=========== LEAVE-ONE-PERSON-OUT ===========")
        for person, s in scores.items():
            print(f"  {person:12s} balanced acc {s * 100:5.1f}%")
        vals = np.array(list(scores.values()))
        print(f"  mean {vals.mean() * 100:5.1f}% +/- {vals.std() * 100:4.1f}%")
        return

    if args.all:
        tx, ty = stack(parts)
        model, history, _ = train_fold(tx, ty, None, None, args, tag="all")
        save_checkpoint(model, args.output, index, args, persons, None, None)
        return

    val_person = args.val_person
    if val_person is None:
        if len(persons) < 2:
            raise SystemExit("Only one person recorded: record more people, or pass --all "
                             "(no honest held-out estimate is possible with one person).")
        val_person = persons[-1]
        print(f"[train] No --val_person given; holding out '{val_person}'.")
    if val_person not in persons:
        raise SystemExit(f"--val_person '{val_person}' not found; people: {persons}")
    tr = [p for p in parts if p["person"] != val_person]
    va = [p for p in parts if p["person"] == val_person]
    (tx, ty), (vx, vy) = stack(tr), stack(va)
    model, history, best = train_fold(tx, ty, vx, vy, args, tag=f"holdout={val_person}")
    save_checkpoint(model, args.output, index, args, [p for p in persons if p != val_person], val_person, best)
    report(model, va, args.metrics_plot, history)


if __name__ == "__main__":
    main()
