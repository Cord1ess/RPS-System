# Legacy v2 pipeline (reference only)

The original frame-differencing system, kept for comparison. It is superseded by the v3 pipeline in the repo root.

| File | Role in v2 | Replaced in v3 by |
|---|---|---|
| `pseudo_event_producer.py` | Single frame difference + threshold 18 → binary 64×64 mask | `rps/dvs_emulator.py` |
| `collect_data.py` | 25-frame burst recorder that saves pre-processed masks | `record_session.py` (raw lossless video) |
| `clean_dataset.py` | Heuristic quarantine of empty, duplicate and "fist-like paper" frames | `build_dataset.py` (MediaPipe label cleaning) |
| `train_v2.py` | Per-frame random split, torchvision augmentations | `train.py` (split by person) |
| `live_predict.py` | CNN on every frame + 5-frame majority vote | `play.py` + `rps/decision.py` |
| `dataset_v2/` | ~3.3k real masks from one person plus 730 synthetic `sample_*` blobs, recorded on an older laptop at 10–19 fps | — |
| `motion_model_v2.pth` | v2 weights (binary-mask inputs; not valid for pseudo-DVS frames) | `models/motion_cnn_v3.pth` |

Why v2 flickered: a still hand produces a blank mask. Blank masks were never in training, so the CNN guessed on every frame. Dextra's event camera instead produces *no frames* when nothing moves, so its last decision holds.

Running v2 from the repo root needs `torchvision` (`uv pip install torchvision`):

```bash
python legacy/live_predict.py --model legacy/motion_model_v2.pth
python legacy/train_v2.py --dataset_dir legacy/dataset_v2 --output_model legacy/motion_model_v2.pth
```
