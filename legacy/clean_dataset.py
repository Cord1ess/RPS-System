#!/usr/bin/env python3
"""
clean_dataset.py - Dataset Sanitization & Quality Filter for Pseudo-Event Roshambo

Automated cleaning utility for 64x64 pseudo-event motion mask datasets:
1. Removes corrupted or unreadable images.
2. Quarantines empty motion frames (near-zero active white pixels).
3. Quarantines duplicate sequential frames that provide no new feature value.
4. Identifies and quarantines transition frames in '1_paper' that contain
   closed fists or partial motion masks resembling '0_rock'.
5. Moves all flagged images into a mirror directory structure './dataset_quarantine'
   for safe inspection, with a complete JSON manifest and terminal summary table.
"""

import os
import sys
import time
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

DEFAULT_DATASET_DIR = "./dataset"
DEFAULT_QUARANTINE_DIR = "./dataset_quarantine"
CLASSES = ["0_rock", "1_paper", "2_scissors", "3_background"]


def parse_filename_sort_key(filename: str) -> Tuple:
    """
    Extracts numerical timestamps and burst counters for natural chronological sorting.
    Example: '1_paper_1789526959852_005.png' -> (1789526959852, 5, '1_paper...')
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    numeric_parts = []
    for p in parts:
        if p.isdigit():
            numeric_parts.append(int(p))
    return tuple(numeric_parts) if numeric_parts else (stem,)


def calculate_dynamic_thresholds(dataset_dir: str) -> Dict[str, Any]:
    """
    Analyzes '0_rock' and '1_paper' distributions to derive data-driven
    thresholds for separating compact fists / transitions from genuine open palm paper masks.
    """
    paper_dir = os.path.join(dataset_dir, "1_paper")
    rock_dir = os.path.join(dataset_dir, "0_rock")

    paper_nz, paper_bbox_areas, paper_widths, paper_heights = [], [], [], []
    rock_nz, rock_bbox_areas = [], []

    if os.path.exists(paper_dir):
        for f in os.listdir(paper_dir):
            if not f.lower().endswith((".png", ".jpg")):
                continue
            img = cv2.imread(os.path.join(paper_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            nz = cv2.countNonZero(img)
            if nz >= 15:
                paper_nz.append(nz)
                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    _, _, w, h = cv2.boundingRect(c)
                    paper_widths.append(w)
                    paper_heights.append(h)
                    paper_bbox_areas.append(w * h)

    if os.path.exists(rock_dir):
        for f in os.listdir(rock_dir):
            if not f.lower().endswith((".png", ".jpg")):
                continue
            img = cv2.imread(os.path.join(rock_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            nz = cv2.countNonZero(img)
            if nz >= 15:
                rock_nz.append(nz)
                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    _, _, w, h = cv2.boundingRect(c)
                    rock_bbox_areas.append(w * h)

    # Dynamic calculation based on percentiles and inter-class boundary
    if paper_nz and rock_nz:
        paper_p20_nz = float(np.percentile(paper_nz, 20))
        rock_med_nz = float(np.median(rock_nz))
        # Paper motion threshold is positioned safely above rock's lower quartile
        dyn_min_nz = max(60, int(paper_p20_nz))

        paper_p15_w = int(np.percentile(paper_widths, 15)) if paper_widths else 12
        paper_p15_h = int(np.percentile(paper_heights, 15)) if paper_heights else 14
        paper_p15_area = int(np.percentile(paper_bbox_areas, 15)) if paper_bbox_areas else 140

        return {
            "paper_min_motion": dyn_min_nz,
            "paper_min_width": max(10, paper_p15_w),
            "paper_min_height": max(12, paper_p15_h),
            "paper_min_bbox_area": max(120, paper_p15_area),
            "source": "dynamic_percentile_analysis",
            "paper_median_nz": float(np.median(paper_nz)),
            "rock_median_nz": rock_med_nz,
        }

    # Fallback to robust static heuristics if distributions cannot be estimated
    return {
        "paper_min_motion": 75,
        "paper_min_width": 11,
        "paper_min_height": 13,
        "paper_min_bbox_area": 130,
        "source": "default_static_heuristics",
    }


def analyze_image_for_quarantine(
    img_path: str,
    class_name: str,
    prev_valid_img: Optional[np.ndarray],
    seen_hashes: set,
    config: argparse.Namespace,
) -> Tuple[bool, str, Dict[str, Any], Optional[np.ndarray]]:
    """
    Evaluates an image against sanitization rules.
    Returns:
        (is_quarantine, reason, metrics_dict, current_valid_img)
    """
    metrics: Dict[str, Any] = {
        "file": os.path.basename(img_path),
        "class": class_name,
    }

    # 1. File existence & basic header check
    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
        return True, "CORRUPTED_EMPTY_FILE", metrics, None

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True, "CORRUPTED_UNREADABLE", metrics, None

    if img.shape != (64, 64):
        metrics["shape"] = list(img.shape)
        return True, f"CORRUPTED_BAD_SHAPE_{img.shape}", metrics, None

    # Compute active white motion pixels (255)
    nz_pixels = int(cv2.countNonZero(img))
    metrics["nonzero_pixels"] = nz_pixels

    # 2. Empty Motion Check
    min_pixels = config.min_background_pixels if class_name == "3_background" else config.min_motion_pixels
    if nz_pixels < min_pixels:
        return True, f"EMPTY_MOTION (nz={nz_pixels} < {min_pixels})", metrics, None

    # 3. Duplicate Frame Check
    # Exact content hash
    img_bytes = img.tobytes()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    if img_hash in seen_hashes:
        metrics["hash"] = img_hash
        return True, "DUPLICATE_IDENTICAL_HASH", metrics, None
    seen_hashes.add(img_hash)

    # Sequential frame difference check
    if prev_valid_img is not None:
        diff = cv2.absdiff(img, prev_valid_img)
        diff_nz = int(cv2.countNonZero(diff))
        metrics["seq_pixel_diff"] = diff_nz

        # If difference is negligible, frame adds no new temporal/spatial feature value
        if diff_nz <= config.duplicate_pixel_diff:
            return True, f"DUPLICATE_SEQUENTIAL (diff_nz={diff_nz} <= {config.duplicate_pixel_diff})", metrics, None

    # 4. Class-Specific Heuristics for 1_paper
    if class_name == "1_paper":
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return True, "PAPER_NO_CONTOURS", metrics, None

        # Take largest contour (primary hand motion cluster)
        primary_contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(primary_contour))
        x, y, w, h = cv2.boundingRect(primary_contour)
        bbox_area = int(w * h)

        metrics["bbox_w"] = w
        metrics["bbox_h"] = h
        metrics["bbox_area"] = bbox_area
        metrics["contour_area"] = contour_area

        # Active pixel threshold test
        if nz_pixels < config.paper_min_motion:
            reason = f"PAPER_LOW_MOTION (nz={nz_pixels} < {config.paper_min_motion})"
            return True, reason, metrics, None

        # Compact fist bounding box dimension test:
        # Open hand (Paper) must have spread fingers extending across both width and height.
        is_compact_fist = (
            w < config.paper_min_width
            or h < config.paper_min_height
            or bbox_area < config.paper_min_bbox_area
        )

        if is_compact_fist:
            reason = (
                f"PAPER_COMPACT_FIST "
                f"(w={w}<{config.paper_min_width} or h={h}<{config.paper_min_height} "
                f"or area={bbox_area}<{config.paper_min_bbox_area})"
            )
            return True, reason, metrics, None

    return False, "RETAINED", metrics, img


def clean_dataset(config: argparse.Namespace) -> Dict[str, Any]:
    """
    Executes dataset scanning, quarantine movement, and reporting.
    """
    dataset_dir = os.path.abspath(config.dataset_dir)
    quarantine_dir = os.path.abspath(config.quarantine_dir)

    print("=" * 80)
    print("      ROSHAMBO PSEUDO-EVENT DATASET CLEANING & QUALITY SANITIZER       ")
    print("=" * 80)
    print(f"Dataset Path    : {dataset_dir}")
    print(f"Quarantine Path : {quarantine_dir}")
    print(f"Execution Mode  : {'[DRY RUN - No files will be moved]' if config.dry_run else '[LIVE QUARANTINE]'}")
    print("-" * 80)

    # Dynamic Threshold Resolution
    if config.dynamic:
        dyn_params = calculate_dynamic_thresholds(dataset_dir)
        print(f"[*] Dynamic heuristic analysis active (source: {dyn_params['source']}):")
        print(f"    - Paper Min Motion (nz) : {dyn_params['paper_min_motion']}")
        print(f"    - Paper Min Width (px)  : {dyn_params['paper_min_width']}")
        print(f"    - Paper Min Height (px) : {dyn_params['paper_min_height']}")
        print(f"    - Paper Min BBox Area   : {dyn_params['paper_min_bbox_area']}")
        if "paper_median_nz" in dyn_params:
            print(f"    - Paper Median nz: {dyn_params['paper_median_nz']:.1f} | Rock Median nz: {dyn_params['rock_median_nz']:.1f}")
        # Apply unless explicitly overridden via CLI flags
        if not config.custom_paper_motion:
            config.paper_min_motion = dyn_params["paper_min_motion"]
        if not config.custom_paper_width:
            config.paper_min_width = dyn_params["paper_min_width"]
        if not config.custom_paper_height:
            config.paper_min_height = dyn_params["paper_min_height"]
        if not config.custom_paper_area:
            config.paper_min_bbox_area = dyn_params["paper_min_bbox_area"]
    else:
        print("[*] Static threshold heuristics active:")
        print(f"    - Paper Min Motion (nz) : {config.paper_min_motion}")
        print(f"    - Paper Min Width (px)  : {config.paper_min_width}")
        print(f"    - Paper Min Height (px) : {config.paper_min_height}")
        print(f"    - Paper Min BBox Area   : {config.paper_min_bbox_area}")

    print(f"    - Min Motion (Gestures) : {config.min_motion_pixels}")
    print(f"    - Min Motion (BG)       : {config.min_background_pixels}")
    print(f"    - Duplicate Diff Thresh : <= {config.duplicate_pixel_diff} pixels")
    print("-" * 80)

    summary_stats: Dict[str, Dict[str, Any]] = {}
    quarantine_records: List[Dict[str, Any]] = []

    for cname in config.classes:
        class_src_dir = os.path.join(dataset_dir, cname)
        class_dst_dir = os.path.join(quarantine_dir, cname)

        if not os.path.exists(class_src_dir):
            print(f"[!] Warning: Class directory not found: {class_src_dir}")
            continue

        if not config.dry_run:
            os.makedirs(class_dst_dir, exist_ok=True)

        all_files = [
            f for f in os.listdir(class_src_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        # Sort chronologically by timestamp & burst sequence
        all_files.sort(key=parse_filename_sort_key)

        total_files = len(all_files)
        retained_count = 0
        quarantined_count = 0
        reasons_breakdown = {
            "corrupted": 0,
            "empty": 0,
            "duplicate": 0,
            "paper_fist": 0,
        }

        prev_valid_img: Optional[np.ndarray] = None
        seen_hashes = set()

        for filename in all_files:
            src_path = os.path.join(class_src_dir, filename)
            dst_path = os.path.join(class_dst_dir, filename)

            is_quarantine, reason, metrics, valid_img = analyze_image_for_quarantine(
                src_path, cname, prev_valid_img, seen_hashes, config
            )

            if is_quarantine:
                quarantined_count += 1
                if "CORRUPTED" in reason:
                    reasons_breakdown["corrupted"] += 1
                elif "EMPTY_MOTION" in reason:
                    reasons_breakdown["empty"] += 1
                elif "DUPLICATE" in reason:
                    reasons_breakdown["duplicate"] += 1
                elif "PAPER" in reason:
                    reasons_breakdown["paper_fist"] += 1

                quarantine_records.append({
                    "class": cname,
                    "filename": filename,
                    "src_path": src_path,
                    "dst_path": dst_path,
                    "reason": reason,
                    "metrics": metrics,
                })

                if config.verbose:
                    print(f"  [QUARANTINE] {cname}/{filename} -> {reason}")

                if not config.dry_run:
                    shutil.move(src_path, dst_path)
            else:
                retained_count += 1
                prev_valid_img = valid_img

        reduction_pct = (quarantined_count / total_files * 100.0) if total_files > 0 else 0.0
        summary_stats[cname] = {
            "total": total_files,
            "retained": retained_count,
            "quarantined": quarantined_count,
            "reduction_pct": reduction_pct,
            "reasons": reasons_breakdown,
        }

    # Save manifest if not dry-run and files were quarantined
    if not config.dry_run and quarantine_records:
        os.makedirs(quarantine_dir, exist_ok=True)
        manifest_path = os.path.join(quarantine_dir, "quarantine_manifest.json")
        manifest_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_dir": dataset_dir,
            "quarantine_dir": quarantine_dir,
            "config": {
                "dynamic": config.dynamic,
                "min_motion_pixels": config.min_motion_pixels,
                "min_background_pixels": config.min_background_pixels,
                "paper_min_motion": config.paper_min_motion,
                "paper_min_width": config.paper_min_width,
                "paper_min_height": config.paper_min_height,
                "paper_min_bbox_area": config.paper_min_bbox_area,
                "duplicate_pixel_diff": config.duplicate_pixel_diff,
            },
            "summary": summary_stats,
            "quarantined_files": quarantine_records,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"\n[+] Quarantine manifest saved to: {manifest_path}")

    # Render Summary Table
    print_summary_table(summary_stats)
    return summary_stats


def print_summary_table(summary_stats: Dict[str, Dict[str, Any]]):
    """
    Renders an informative ASCII summary table in the terminal.
    """
    print("\n" + "=" * 90)
    print(f"{'DATASET SANITIZATION SUMMARY REPORT':^90}")
    print("=" * 90)

    header = (
        f"| {'Class Name':<14} | {'Total':>7} | {'Retained':>9} | "
        f"{'Quarantined':>11} | {'Reduction':>9} | {'Breakdown (Corrupt/Empty/Dup/Fist)':<27} |"
    )
    separator = "+" + "-" * 16 + "+" + "-" * 9 + "+" + "-" * 11 + "+" + "-" * 13 + "+" + "-" * 11 + "+" + "-" * 30 + "+"

    print(separator)
    print(header)
    print(separator)

    tot_all = 0
    ret_all = 0
    quar_all = 0

    for cname, data in summary_stats.items():
        total = data["total"]
        retained = data["retained"]
        quar = data["quarantined"]
        red = data["reduction_pct"]
        r = data["reasons"]

        tot_all += total
        ret_all += retained
        quar_all += quar

        breakdown_str = f"C:{r['corrupted']} | E:{r['empty']} | D:{r['duplicate']} | F:{r['paper_fist']}"
        print(
            f"| {cname:<14} | {total:>7} | {retained:>9} | {quar:>11} | "
            f"{red:>8.2f}% | {breakdown_str:<28} |"
        )

    print(separator)
    total_red = (quar_all / tot_all * 100.0) if tot_all > 0 else 0.0
    print(
        f"| {'TOTAL OVERALL':<14} | {tot_all:>7} | {ret_all:>9} | {quar_all:>11} | "
        f"{total_red:>8.2f}% | {'--':<28} |"
    )
    print(separator)
    print("Legend: C=Corrupted/Unreadable, E=Empty Motion, D=Duplicate Sequential, F=Paper Fist/Low Motion")
    print("=" * 90 + "\n")


def restore_quarantine(config: argparse.Namespace):
    """
    Restores all quarantined images back into the dataset directory.
    """
    dataset_dir = os.path.abspath(config.dataset_dir)
    quarantine_dir = os.path.abspath(config.quarantine_dir)

    if not os.path.exists(quarantine_dir):
        print(f"[-] Quarantine directory does not exist: {quarantine_dir}")
        return

    print("=" * 80)
    print("              RESTORING QUARANTINED IMAGES TO DATASET               ")
    print("=" * 80)
    print(f"From: {quarantine_dir}")
    print(f"To  : {dataset_dir}")
    print("-" * 80)

    restored_count = 0
    for cname in config.classes:
        c_quar_dir = os.path.join(quarantine_dir, cname)
        c_data_dir = os.path.join(dataset_dir, cname)

        if not os.path.exists(c_quar_dir):
            continue

        os.makedirs(c_data_dir, exist_ok=True)
        files = [f for f in os.listdir(c_quar_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for f in files:
            src = os.path.join(c_quar_dir, f)
            dst = os.path.join(c_data_dir, f)
            shutil.move(src, dst)
            restored_count += 1

        print(f"[+] Restored {len(files)} files for class '{cname}'")

    # Remove empty subfolders in quarantine
    manifest_file = os.path.join(quarantine_dir, "quarantine_manifest.json")
    if os.path.exists(manifest_file):
        os.remove(manifest_file)

    print(f"\n[+] Total files restored: {restored_count}")
    print("[+] Dataset restored to original state successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clean & sanitize 64x64 pseudo-event motion mask dataset for Rock-Paper-Scissors."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Path to root dataset directory (default: ./dataset)",
    )
    parser.add_argument(
        "--quarantine_dir",
        type=str,
        default=DEFAULT_QUARANTINE_DIR,
        help="Path to quarantine directory (default: ./dataset_quarantine)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Simulate cleaning and display report without moving files",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Move quarantined files back from ./dataset_quarantine to ./dataset",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Enable dynamic percentile-based thresholding for Paper (default: enabled)",
    )
    parser.add_argument(
        "--no_dynamic",
        dest="dynamic",
        action="store_false",
        help="Disable dynamic thresholding and use strict manual thresholds",
    )
    parser.add_argument(
        "--min_motion_pixels",
        type=int,
        default=15,
        help="Minimum active white pixels for gestures (filters empty frames, default: 15)",
    )
    parser.add_argument(
        "--min_background_pixels",
        type=int,
        default=5,
        help="Minimum active white pixels for background class (default: 5)",
    )
    parser.add_argument(
        "--paper_min_motion",
        type=int,
        default=75,
        help="Minimum active white pixels for 1_paper (default: 75)",
    )
    parser.add_argument(
        "--paper_min_width",
        type=int,
        default=11,
        help="Minimum bounding box width for 1_paper (default: 11)",
    )
    parser.add_argument(
        "--paper_min_height",
        type=int,
        default=13,
        help="Minimum bounding box height for 1_paper (default: 13)",
    )
    parser.add_argument(
        "--paper_min_bbox_area",
        type=int,
        default=130,
        help="Minimum bounding box area (w*h) for 1_paper (default: 130)",
    )
    parser.add_argument(
        "--duplicate_pixel_diff",
        type=int,
        default=5,
        help="Max pixel difference count to consider sequential frame a duplicate (default: 5)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose log of every quarantined file",
    )

    args = parser.parse_args()
    args.classes = CLASSES

    # Track if user explicitly specified custom paper thresholds on CLI
    args.custom_paper_motion = "--paper_min_motion" in sys.argv
    args.custom_paper_width = "--paper_min_width" in sys.argv
    args.custom_paper_height = "--paper_min_height" in sys.argv
    args.custom_paper_area = "--paper_min_bbox_area" in sys.argv

    if args.restore:
        restore_quarantine(args)
    else:
        clean_dataset(args)


if __name__ == "__main__":
    main()
