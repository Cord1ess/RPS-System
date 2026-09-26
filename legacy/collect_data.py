"""
Dataset Collection Utility for Rock-Paper-Scissors Pseudo-Event System.
Captures 64x64 binary motion event masks into:
    dataset/
      0_rock/
      1_paper/
      2_scissors/
      3_background/
Triggers bursts on keyboard events:
  'r' -> 0_rock
  'p' -> 1_paper
  's' -> 2_scissors
  'b' -> 3_background
  'q' or ESC -> Quit
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from pseudo_event_producer import PseudoEventProducer

DATASET_ROOT = "dataset"
CLASSES = {
    "r": ("0_rock", 0),
    "p": ("1_paper", 1),
    "s": ("2_scissors", 2),
    "b": ("3_background", 3),
}
CLASS_NAMES = ["0_rock", "1_paper", "2_scissors", "3_background"]


def init_dataset_dirs(base_path: str = DATASET_ROOT):
    """Creates the dataset directory structure."""
    for cname in CLASS_NAMES:
        dir_path = os.path.join(base_path, cname)
        os.makedirs(dir_path, exist_ok=True)
    print(f"[collect_data] Initialized dataset directory at: {os.path.abspath(base_path)}")


def get_class_counts(base_path: str = DATASET_ROOT) -> dict:
    """Returns the number of existing images per class."""
    counts = {}
    for cname in CLASS_NAMES:
        dir_path = os.path.join(base_path, cname)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(".png") or f.endswith(".jpg")]
            counts[cname] = len(files)
        else:
            counts[cname] = 0
    return counts


def generate_synthetic_samples(base_path: str = DATASET_ROOT, samples_per_class: int = 150):
    """
    Generates realistic 64x64 motion masks for rock, paper, scissors, and background
    to populate the dataset instantly for development, benchmarking, or automated verification.
    """
    init_dataset_dirs(base_path)
    print(f"[collect_data] Generating {samples_per_class} synthetic samples per class...")

    for cname in CLASS_NAMES:
        target_dir = os.path.join(base_path, cname)
        for i in range(samples_per_class):
            mask = np.zeros((64, 64), dtype=np.uint8)
            cx, cy = np.random.randint(28, 36), np.random.randint(28, 36)

            if cname == "0_rock":
                # Rock: Compact fist shape (dense circular/oval cluster)
                radius_x = np.random.randint(12, 18)
                radius_y = np.random.randint(12, 18)
                angle = np.random.randint(-20, 20)
                cv2.ellipse(mask, (cx, cy), (radius_x, radius_y), angle, 0, 360, 255, -1)
                # Motion trail outer noise
                if np.random.rand() > 0.5:
                    cv2.circle(mask, (cx + np.random.randint(-5, 5), cy - radius_y - 2), 4, 255, -1)

            elif cname == "1_paper":
                # Paper: Wide open palm with spread out finger contours
                cv2.ellipse(mask, (cx, cy + 4), (16, 14), 0, 0, 360, 255, -1)
                # 5 fingers pointing outwards
                angles = [-60, -30, 0, 30, 60]
                for ang in angles:
                    rad = np.radians(ang + np.random.randint(-5, 5))
                    fx = int(cx + 24 * np.sin(rad))
                    fy = int(cy - 24 * np.cos(rad))
                    cv2.line(mask, (cx, cy), (fx, fy), 255, thickness=np.random.randint(3, 5))

            elif cname == "2_scissors":
                # Scissors: V-shape with 2 extended fingers and closed rest
                cv2.ellipse(mask, (cx, cy + 8), (14, 12), 0, 0, 360, 255, -1)
                # Index and middle finger (V-split)
                for ang in [-22, 22]:
                    rad = np.radians(ang + np.random.randint(-4, 4))
                    fx = int(cx + 26 * np.sin(rad))
                    fy = int(cy - 26 * np.cos(rad))
                    cv2.line(mask, (cx, cy), (fx, fy), 255, thickness=4)

            elif cname == "3_background":
                # Background: Random low-density motion noise or empty frames
                if np.random.rand() > 0.3:
                    num_spots = np.random.randint(1, 8)
                    for _ in range(num_spots):
                        rx, ry = np.random.randint(5, 59), np.random.randint(5, 59)
                        cv2.circle(mask, (rx, ry), np.random.randint(1, 3), 255, -1)

            # Add subtle motion blur / threshold perturbation
            if np.random.rand() > 0.5:
                kernel = np.ones((2, 2), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            filename = f"sample_{int(time.time()*1000)}_{i:04d}.png"
            cv2.imwrite(os.path.join(target_dir, filename), mask)

    print("[collect_data] Synthetic dataset populated successfully!")
    print(get_class_counts(base_path))


def run_collection(args):
    """Main interactive collection loop."""
    init_dataset_dirs(args.dataset_dir)
    
    producer = PseudoEventProducer(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        threshold_value=args.threshold,
        mock_mode=args.mock,
    )

    burst_active = False
    burst_class = None
    burst_counter = 0
    burst_total = args.burst_size

    print("\n========================================================")
    print("  ROCK-PAPER-SCISSORS PSEUDO-EVENT DATA COLLECTOR       ")
    print("========================================================")
    print("  Trigger Keys:")
    print("    [R] -> Record burst for 0_rock")
    print("    [P] -> Record burst for 1_paper")
    print("    [S] -> Record burst for 2_scissors")
    print("    [B] -> Record burst for 3_background")
    print("    [Q] / ESC -> Save & Quit")
    print("========================================================\n")

    counts = get_class_counts(args.dataset_dir)
    preview_window = "Pseudo-Event Dataset Collector"
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, raw_frame, mask_64 = producer.get_motion_mask()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            # Handle active recording burst
            if burst_active and burst_counter < burst_total:
                cname = burst_class
                nz_pixels = int(np.count_nonzero(mask_64))
                # For gestures, avoid capturing empty frames before motion starts or after it finishes
                is_valid_motion = (cname == "3_background") or (nz_pixels >= args.min_motion_pixels)
                if is_valid_motion:
                    filename = f"{cname}_{int(time.time()*1000)}_{burst_counter:03d}.png"
                    save_path = os.path.join(args.dataset_dir, cname, filename)
                    cv2.imwrite(save_path, mask_64)
                    burst_counter += 1
                    counts[cname] += 1

                    if burst_counter >= burst_total:
                        burst_active = False
                        print(f"[REC DONE] Finished burst for {cname}. Total: {counts[cname]}")

            # Prepare visual display
            display = raw_frame.copy()
            dh, dw, _ = display.shape

            # Overlay ROI box in center
            min_dim = min(dh, dw)
            sy = (dh - min_dim) // 2
            sx = (dw - min_dim) // 2
            cv2.rectangle(display, (sx, sy), (sx + min_dim, sy + min_dim), (0, 255, 0), 2)

            # Render 64x64 mask preview enlarged in top-right corner
            mask_large = cv2.resize(mask_64, (160, 160), interpolation=cv2.INTER_NEAREST)
            mask_bgr = cv2.cvtColor(mask_large, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(mask_bgr, (0, 0), (159, 159), (0, 255, 255), 2)
            display[10:170, dw - 170 : dw - 10] = mask_bgr
            cv2.putText(display, "64x64 Mask", (dw - 165, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Status Overlay
            y_offset = 30
            cv2.putText(display, "DATASET COLLECTOR", (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            
            y_offset += 30
            for idx, cname in enumerate(CLASS_NAMES):
                txt = f"[{cname[2].upper()}] {cname}: {counts.get(cname, 0)} frames"
                color = (0, 255, 0) if burst_active and burst_class == cname else (200, 200, 200)
                cv2.putText(display, txt, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y_offset += 24

            if burst_active:
                cv2.circle(display, (dw // 2 - 100, 40), 12, (0, 0, 255), -1)
                cv2.putText(display, f"REC [{burst_class}] {burst_counter}/{burst_total}",
                            (dw // 2 - 75, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(preview_window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), 27]:  # 'q' or ESC
                break
            elif chr(key).lower() in CLASSES and not burst_active:
                char_key = chr(key).lower()
                burst_class = CLASSES[char_key][0]
                burst_active = True
                burst_counter = 0
                print(f"[REC START] Recording {burst_total} frames for {burst_class}...")

    finally:
        producer.release()
        cv2.destroyAllWindows()
        print("\n[collect_data] Final Dataset Statistics:")
        final_counts = get_class_counts(args.dataset_dir)
        for k, v in final_counts.items():
            print(f"  {k}: {v} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect 64x64 motion masks for Rock-Paper-Scissors")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--threshold", type=int, default=18, help="Binary motion threshold value (15-20 recommended)")
    parser.add_argument("--min_motion_pixels", type=int, default=15, help="Minimum active pixels to record a gesture frame (filters empty frames)")
    parser.add_argument("--burst_size", type=int, default=25, help="Number of frames per burst")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_ROOT, help="Output dataset directory")
    parser.add_argument("--mock", action="store_true", help="Run in mock/simulated camera mode")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset and exit")
    parser.add_argument("--synthetic_count", type=int, default=150, help="Samples per class for synthetic generation")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_samples(args.dataset_dir, args.synthetic_count)
    else:
        run_collection(args)
