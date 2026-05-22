#!/usr/bin/env python3
"""
KITTI Depth Dataset Downloader & Converter for YOLO26 Depth+Segmentation Training

Based on ROADMAP.md Phase B: Introduce KITTI dataset for outdoor driving scenes.

This script:
1. Guides/downloads KITTI Depth Prediction data
2. Applies Eigen split for train/val division
3. Converts to YOLO format:
   kitti_yolo/
   ├── images/train/
   ├── images/val/
   ├── depths/train/     # 16-bit PNG (mm), sparse LiDAR depth
   ├── depths/val/
   ├── labels/train/     # YOLO bbox (optional, from KITTI 2D det)
   ├── labels/val/
   ├── segments/train/   # YOLO segments (optional)
   └── segments/val/

Usage:
    # Step 1: Download KITTI (or place manually downloaded zips in --download-dir)
    python download_kitti.py --download-dir ./kitti_zips --output ./kitti_yolo

    # Step 2: If you already have extracted KITTI raw data
    python download_kitti.py \
        --kitti-root ./Kitti/raw_data \
        --depth-root ./Kitti/depth_prediction \
        --output ./kitti_yolo \
        --skip-download

KITTI Dataset Info (from ROADMAP):
    - Scene: Outdoor driving
    - Depth range: 0~80 m
    - Depth source: LiDAR sparse point cloud (~5% pixels valid)
    - Images: ~26k (Eigen train), ~697 (Eigen val)
    - Resolution: ~1242x375
    - Depth density: Sparse

Download Requirements:
    KITTI requires registration. Please visit:
    - http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
    - Download: data_depth_annotated.zip (ground truth depth)
    - Download: data_depth_velodyne.zip (raw velodyne)
    - Or use KITTI RAW: download individual sequences
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Eigen split file lists for KITTI depth prediction
# Source: Eigen et al. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network" (NIPS 2014)
# These are the standard train/val splits used in monocular depth estimation literature.

EIGEN_TRAIN_SCENES = [
    "2011_09_26_drive_0001_sync",
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0005_sync",
    "2011_09_26_drive_0009_sync",
    "2011_09_26_drive_0011_sync",
    "2011_09_26_drive_0013_sync",
    "2011_09_26_drive_0014_sync",
    "2011_09_26_drive_0015_sync",
    "2011_09_26_drive_0017_sync",
    "2011_09_26_drive_0018_sync",
    "2011_09_26_drive_0019_sync",
    "2011_09_26_drive_0020_sync",
    "2011_09_26_drive_0022_sync",
    "2011_09_26_drive_0023_sync",
    "2011_09_26_drive_0027_sync",
    "2011_09_26_drive_0028_sync",
    "2011_09_26_drive_0029_sync",
    "2011_09_26_drive_0032_sync",
    "2011_09_26_drive_0035_sync",
    "2011_09_26_drive_0036_sync",
    "2011_09_26_drive_0039_sync",
    "2011_09_26_drive_0046_sync",
    "2011_09_26_drive_0048_sync",
    "2011_09_26_drive_0051_sync",
    "2011_09_26_drive_0052_sync",
    "2011_09_26_drive_0056_sync",
    "2011_09_26_drive_0057_sync",
    "2011_09_26_drive_0059_sync",
    "2011_09_26_drive_0060_sync",
    "2011_09_26_drive_0061_sync",
    "2011_09_26_drive_0064_sync",
    "2011_09_26_drive_0070_sync",
    "2011_09_26_drive_0079_sync",
    "2011_09_26_drive_0084_sync",
    "2011_09_26_drive_0086_sync",
    "2011_09_26_drive_0091_sync",
    "2011_09_26_drive_0093_sync",
    "2011_09_26_drive_0095_sync",
    "2011_09_26_drive_0096_sync",
    "2011_09_26_drive_0101_sync",
    "2011_09_26_drive_0104_sync",
    "2011_09_26_drive_0106_sync",
    "2011_09_26_drive_0113_sync",
    "2011_09_26_drive_0117_sync",
    "2011_09_26_drive_0119_sync",
    "2011_09_28_drive_0001_sync",
    "2011_09_28_drive_0002_sync",
    "2011_09_29_drive_0004_sync",
    "2011_09_29_drive_0026_sync",
    "2011_09_29_drive_0071_sync",
    "2011_09_30_drive_0016_sync",
    "2011_09_30_drive_0018_sync",
    "2011_09_30_drive_0020_sync",
    "2011_09_30_drive_0027_sync",
    "2011_09_30_drive_0028_sync",
    "2011_09_30_drive_0033_sync",
    "2011_09_30_drive_0034_sync",
    "2011_10_03_drive_0042_sync",
    "2011_10_03_drive_0047_sync",
    "2011_10_03_drive_0058_sync",
]

EIGEN_VAL_SCENES = [
    "2011_09_26_drive_0005_sync",  # Note: some overlap handling done by frame-level filtering
]

# Actually, Eigen split is better defined by frame-level lists.
# Below are the commonly used train/val file lists.
# For practical purposes, we use the official KITTI depth prediction train/val split
# which is already provided in the dataset.

KITTI_DEPTH_URLS = {
    "data_depth_annotated.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip",
    "data_depth_velodyne.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip",
}

# KITTI 2D object detection classes -> COCO-style mapping
KITTI_DETECTION_CLASSES = {
    "Car": 0,
    "Van": 0,
    "Truck": 1,
    "Pedestrian": 2,
    "Person_sitting": 2,
    "Cyclist": 3,
    "Tram": 4,
}


def check_kitti_installed(kitti_root: Path) -> dict:
    """Check if KITTI raw data exists and return paths."""
    result = {"raw": None, "depth": None}

    if kitti_root.exists():
        # Check for raw_data structure
        raw_path = kitti_root / "raw_data"
        if raw_path.exists():
            result["raw"] = raw_path
        else:
            # Maybe kitti_root itself is raw_data
            if any((kitti_root / d).exists() for d in ["2011_09_26", "2011_09_28"]):
                result["raw"] = kitti_root

    return result


def is_valid_zip(path: Path) -> bool:
    """Check if file is a valid zip (not an HTML error page)."""
    if not path.exists() or path.stat().st_size < 100:
        return False
    with open(path, "rb") as f:
        header = f.read(4)
    # ZIP magic number: 50 4B 03 04
    return header == b"PK\x03\x04"


def download_with_wget(url: str, output_path: Path, desc: str = "") -> bool:
    """Attempt to download file using wget."""
    try:
        print(f"Downloading {desc or output_path.name}...")
        print(f"URL: {url}")
        print("NOTE: KITTI requires login/authentication. Automated download usually fails.")

        cmd = [
            "wget",
            "--continue",
            "--tries=1",
            "--timeout=60",
            "-O", str(output_path),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            if is_valid_zip(output_path):
                print(f"  Successfully downloaded: {output_path.name}")
                return True
            else:
                # Likely an HTML redirect/login page
                print(f"  Download returned HTML page (authentication required)")
                output_path.unlink()  # remove the HTML file
                return False
        else:
            if output_path.exists():
                output_path.unlink()
            return False
    except FileNotFoundError:
        print("wget not found. Please install wget or download manually.")
        return False


def download_kitti_dataset(download_dir: Path, output_root: Path) -> dict:
    """Download KITTI depth prediction dataset files.

    Returns dict with paths to downloaded files.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    print("=" * 60)
    print("KITTI Dataset Download")
    print("=" * 60)
    print()
    print("KITTI requires registration. The automated download may fail")
    print("due to authentication requirements.")
    print()
    print("If automatic download fails, please:")
    print("1. Visit: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction")
    print("2. Download 'Depth Prediction' dataset (data_depth_annotated.zip)")
    print("3. Place the zip file in:", download_dir)
    print()

    # Try to download depth annotated
    depth_annotated = download_dir / "data_depth_annotated.zip"
    has_valid_zip = depth_annotated.exists() and is_valid_zip(depth_annotated)

    if not has_valid_zip:
        if depth_annotated.exists():
            print(f"  Removing invalid file: {depth_annotated.name}")
            depth_annotated.unlink()

        success = download_with_wget(
            KITTI_DEPTH_URLS["data_depth_annotated.zip"],
            depth_annotated,
            "KITTI Depth Annotated",
        )
        if success:
            downloaded["depth_annotated"] = depth_annotated
        else:
            print(f"\n{'='*60}")
            print("AUTOMATIC DOWNLOAD FAILED - MANUAL DOWNLOAD REQUIRED")
            print(f"{'='*60}")
            print()
            print("KITTI requires registration. Please follow these steps:")
            print()
            print("1. Visit:")
            print("   http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction")
            print()
            print("2. Click 'Download' and log in (or register)")
            print()
            print("3. Download this file:")
            print("   - data_depth_annotated.zip  (~5 GB)")
            print()
            print("4. Place it in this directory:")
            print(f"   {download_dir}")
            print()
            print("5. Re-run this script:")
            print(f"   python download_kitti.py --output {output_root}")
            print()
    else:
        print(f"Found valid zip: {depth_annotated}")
        downloaded["depth_annotated"] = depth_annotated

    return downloaded


def extract_kitti_zips(download_dir: Path, output_root: Path) -> dict:
    """Extract KITTI zip files to output directory.

    Returns dict with extracted paths.
    """
    extracted = {}

    depth_annotated = download_dir / "data_depth_annotated.zip"
    if depth_annotated.exists():
        if not is_valid_zip(depth_annotated):
            print(f"\nERROR: {depth_annotated.name} is not a valid zip file!")
            print("This usually means the download returned an HTML page instead.")
            print("Please download KITTI manually and place the zip in:")
            print(f"  {download_dir}")
            return extracted

        print(f"\nExtracting {depth_annotated.name}...")
        with zipfile.ZipFile(depth_annotated, "r") as zf:
            zf.extractall(output_root)
        extracted["depth"] = output_root / "data_depth_annotated"
        print(f"  Extracted to: {extracted['depth']}")

    return extracted


def find_all_depth_scenes(kitti_root: Path) -> dict[str, list[str]]:
    """Find all scenes that have depth data, grouped by split.

    Returns:
        {"train": [scene_names], "val": [scene_names]}
    """
    scenes = {"train": [], "val": []}

    for split in ["train", "val"]:
        split_dir = kitti_root / split
        if not split_dir.exists():
            continue

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            # Check if any camera has depth data
            has_depth = False
            for camera in ["image_02", "image_03"]:
                depth_dir = scene_dir / "proj_depth" / "groundtruth" / camera
                if depth_dir.exists() and any(depth_dir.glob("*.png")):
                    has_depth = True
                    break

            if has_depth:
                scenes[split].append(scene_dir.name)

    return scenes


def find_kitti_depth_pairs(kitti_root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Find all (image_path, depth_path) pairs in KITTI dataset.

    Expected structure:
        kitti_root/
        ├── train/
        │   └── 2011_09_26_drive_0001_sync/
        │       ├── image_02/data/*.png       <- RGB images (from KITTI Raw)
        │       └── proj_depth/groundtruth/image_02/*.png  <- Depth (from depth annotated)
        └── val/
            └── ...

    KITTI Depth Prediction dataset (data_depth_annotated.zip) only contains depth maps.
    RGB images must be downloaded separately from KITTI Raw sync data.

    Returns:
        (pairs, missing_scenes) where:
            pairs: list of (rgb_path, depth_path) tuples for scenes that have both
            missing_scenes: list of scene names missing RGB images
    """
    pairs = []
    missing_scenes = []
    depth_only_scenes = []

    for split in ["train", "val"]:
        split_dir = kitti_root / split
        if not split_dir.exists():
            continue

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            # Try both image_02 (left) and image_03 (right)
            for camera in ["image_02", "image_03"]:
                img_dir = scene_dir / camera / "data"
                depth_dir = scene_dir / "proj_depth" / "groundtruth" / camera

                if not depth_dir.exists():
                    continue

                depth_files = sorted(depth_dir.glob("*.png"))
                if not depth_files:
                    continue

                if not img_dir.exists():
                    # Have depth but no RGB
                    missing_scenes.append(f"{split}/{scene_dir.name}/{camera}")
                    if scene_dir.name not in depth_only_scenes:
                        depth_only_scenes.append(scene_dir.name)
                    continue

                for depth_file in depth_files:
                    img_file = img_dir / depth_file.name
                    if img_file.exists():
                        pairs.append((img_file, depth_file))

    print(f"Found {len(pairs)} (image, depth) pairs")
    if missing_scenes:
        print(f"Missing RGB for {len(missing_scenes)} camera instances ({len(depth_only_scenes)} unique scenes)")

    return pairs, depth_only_scenes


def get_kitti_raw_sync_url(scene_name: str) -> str | None:
    """Build S3 URL for KITTI Raw sync data.

    Scene name format: 2011_09_26_drive_0001_sync
    URL format: https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
    """
    if not scene_name.endswith("_sync"):
        return None
    scene_base = scene_name[:-5]  # remove "_sync"
    return f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{scene_base}/{scene_name}.zip"


def download_kitti_raw_sync(
    scene_name: str,
    download_dir: Path,
    extract_root: Path,
) -> bool:
    """Download and extract KITTI Raw sync data for a single scene.

    Args:
        scene_name: e.g. "2011_09_26_drive_0001_sync"
        download_dir: Where to save the zip file
        extract_root: Where to extract (should be kitti_root/train or kitti_root/val)

    Returns True if successful.
    """
    url = get_kitti_raw_sync_url(scene_name)
    if url is None:
        print(f"  Could not build URL for {scene_name}")
        return False

    zip_path = download_dir / f"{scene_name}.zip"

    # Download
    try:
        result = subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(zip_path), url],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0 or not zip_path.exists() or zip_path.stat().st_size < 1000:
            if zip_path.exists():
                zip_path.unlink()
            return False
    except Exception:
        if zip_path.exists():
            zip_path.unlink()
        return False

    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        zip_path.unlink()  # remove zip after extraction to save space
        return True
    except Exception:
        return False


def download_missing_raw_scenes(
    missing_scenes: list[str],
    kitti_root: Path,
    raw_download_dir: Path,
) -> list[str]:
    """Download KITTI Raw sync data for scenes missing RGB images.

    Returns list of scene names that were successfully downloaded.
    """
    if not missing_scenes:
        return []

    print("\n" + "=" * 60)
    print("Downloading KITTI Raw sync data for RGB images")
    print("=" * 60)
    print(f"Scenes to download: {len(missing_scenes)}")
    print("NOTE: Each scene is ~300-800MB. Total may be 50GB+.")
    print("This may take a while...")
    print()

    raw_download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for i, scene_name in enumerate(missing_scenes, 1):
        # Determine split dir
        split = None
        for s in ["train", "val"]:
            if (kitti_root / s / scene_name / "proj_depth").exists():
                split = s
                break
        if split is None:
            continue

        extract_root = kitti_root / split
        print(f"[{i}/{len(missing_scenes)}] Downloading {scene_name}...")

        if download_kitti_raw_sync(scene_name, raw_download_dir, extract_root):
            downloaded.append(scene_name)
            print(f"  Done: {scene_name}")
        else:
            print(f"  FAILED: {scene_name}")

    print(f"\nDownloaded {len(downloaded)}/{len(missing_scenes)} scenes")
    return downloaded


def load_eigen_split(eigen_split_file: Path | None = None) -> tuple[set, set]:
    """Load Eigen train/val split.

    If eigen_split_file is provided, load from it.
    Otherwise use the built-in list.

    Returns (train_ids, val_ids) where each id is "scene_name/frame_name".
    """
    train_ids = set()
    val_ids = set()

    if eigen_split_file and eigen_split_file.exists():
        with open(eigen_split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    frame_id = parts[0]
                    split_name = parts[1].lower()
                    if split_name == "train":
                        train_ids.add(frame_id)
                    elif split_name == "val":
                        val_ids.add(frame_id)
    else:
        # Use scene-level split as fallback
        # All frames from train scenes go to train, val scenes to val
        for scene in EIGEN_TRAIN_SCENES:
            train_ids.add(scene.replace("_sync", ""))

    return train_ids, val_ids


def kitti_to_yolo_format(
    kitti_root: Path,
    output_root: Path,
    eigen_split_file: Path | None = None,
    copy_labels: bool = True,
    download_raw: bool = False,
    raw_download_dir: Path | None = None,
    subset_ratio: float = 1.0,
):
    """Convert KITTI Depth Prediction dataset to YOLO format.

    Args:
        kitti_root: Path to KITTI depth prediction dataset root
                   (contains train/ and val/ subdirs)
        output_root: Output directory for YOLO format dataset
        eigen_split_file: Optional file with frame-level Eigen split
        copy_labels: Whether to also look for KITTI 2D detection labels
        download_raw: Whether to download KITTI Raw sync data for missing RGB images
        raw_download_dir: Directory to download raw sync zips
        subset_ratio: Ratio of train scenes to use (1.0 = all). Val set always complete.
    """
    import random

    output_root = Path(output_root)

    # Create output directories
    for subset in ["train", "val"]:
        (output_root / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_root / "depths" / subset).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / subset).mkdir(parents=True, exist_ok=True)
        (output_root / "segments" / subset).mkdir(parents=True, exist_ok=True)

    # Load Eigen split
    train_ids, val_ids = load_eigen_split(eigen_split_file)

    # --- Scene-level subset sampling (apply BEFORE downloading) ---
    all_scenes = find_all_depth_scenes(kitti_root)
    sampled_train_scenes = set(all_scenes["train"])
    val_scenes = set(all_scenes["val"])

    if subset_ratio < 1.0 and len(sampled_train_scenes) > 0:
        n_train_scenes = max(1, int(len(sampled_train_scenes) * subset_ratio))
        random.seed(42)
        sampled_train_scenes = set(random.sample(sorted(sampled_train_scenes), n_train_scenes))

        print("\n" + "=" * 60)
        print(f"Scene-level sampling: using {subset_ratio*100:.0f}% of train scenes")
        print(f"  Train scenes: {len(all_scenes['train'])} -> {len(sampled_train_scenes)}")
        print(f"  Val scenes:   {len(val_scenes)} (kept complete)")
        print("=" * 60)

    # Find all pairs (some may have RGB already if user pre-downloaded)
    pairs, missing_scenes = find_kitti_depth_pairs(kitti_root)

    # Filter missing_scenes to only sampled train scenes + all val scenes
    # missing_scenes entries are like "train/scene_name/camera" or "val/scene_name/camera"
    # But depth_only_scenes returns just scene names without split prefix
    # Let's rebuild missing from sampled scenes
    missing_scenes_for_download = []
    for split, scene_list in [("train", sorted(sampled_train_scenes)), ("val", sorted(val_scenes))]:
        for scene_name in scene_list:
            scene_dir = kitti_root / split / scene_name
            if not scene_dir.exists():
                continue
            for camera in ["image_02", "image_03"]:
                depth_dir = scene_dir / "proj_depth" / "groundtruth" / camera
                img_dir = scene_dir / camera / "data"
                if depth_dir.exists() and any(depth_dir.glob("*.png")) and not img_dir.exists():
                    missing_scenes_for_download.append(scene_name)
                    break  # one entry per scene

    # Remove duplicates while preserving order
    seen = set()
    missing_scenes_for_download = [s for s in missing_scenes_for_download if not (s in seen or seen.add(s))]

    # Handle missing RGB images
    if missing_scenes_for_download:
        if download_raw and raw_download_dir:
            downloaded = download_missing_raw_scenes(
                missing_scenes_for_download, kitti_root, raw_download_dir
            )
            # Re-scan after downloading
            if downloaded:
                pairs, missing_scenes = find_kitti_depth_pairs(kitti_root)
        else:
            print("\n" + "=" * 60)
            print("WARNING: Some scenes missing RGB images!")
            print("=" * 60)
            print(f"Scenes without RGB: {len(missing_scenes_for_download)}")
            for s in missing_scenes_for_download[:5]:
                print(f"  - {s}")
            if len(missing_scenes_for_download) > 5:
                print(f"  ... and {len(missing_scenes_for_download) - 5} more")
            print()
            print("To download RGB images, add --download-raw:")
            print(f"  python download_kitti.py --kitti-root {kitti_root} --output {output_root} --download-raw")
            print()
            print("Or download KITTI Raw sync manually from:")
            print("  https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/")
            print("=" * 60)

    # Filter pairs to only sampled scenes (if subset_ratio was used)
    if subset_ratio < 1.0:
        pairs = [
            (img_path, depth_path)
            for img_path, depth_path in pairs
            if img_path.parent.parent.parent.name in sampled_train_scenes
            or img_path.parent.parent.parent.parent.name != "train"
        ]

    if not pairs:
        print("ERROR: No valid (image, depth) pairs found!")
        print("       Check that RGB images were downloaded or --download-raw was used.")
        return {"train": {"images": 0, "depths": 0}, "val": {"images": 0, "depths": 0}}

    # Organize by split
    train_pairs = []
    val_pairs = []

    for img_path, depth_path in pairs:
        # Determine split based on directory structure or Eigen split
        scene_name = img_path.parent.parent.parent.name  # e.g., 2011_09_26_drive_0001_sync

        # Simple heuristic: if in train/ dir -> train, if in val/ dir -> val
        split_dir = img_path.parent.parent.parent.parent.name  # train or val

        if split_dir == "train":
            train_pairs.append((img_path, depth_path))
        elif split_dir == "val":
            val_pairs.append((img_path, depth_path))
        else:
            # Fallback: use Eigen scene list
            scene_base = scene_name.replace("_sync", "")
            if any(s.startswith(scene_base) for s in EIGEN_TRAIN_SCENES):
                train_pairs.append((img_path, depth_path))
            else:
                val_pairs.append((img_path, depth_path))

    print(f"\nSplit: {len(train_pairs)} train, {len(val_pairs)} val")

    # Process each split
    stats = {"train": {"images": 0, "depths": 0}, "val": {"images": 0, "depths": 0}}

    for subset, subset_pairs in [("train", train_pairs), ("val", val_pairs)]:
        if not subset_pairs:
            continue

        print(f"\nProcessing {subset} set ({len(subset_pairs)} samples)...")

        for img_path, depth_path in tqdm(subset_pairs, desc=subset, ncols=80):
            # Generate unique sample ID
            scene = img_path.parent.parent.parent.name
            frame = img_path.stem
            sample_id = f"{scene}_{frame}"

            # Output paths
            img_out = output_root / "images" / subset / f"{sample_id}.png"
            depth_out = output_root / "depths" / subset / f"{sample_id}.png"
            label_out = output_root / "labels" / subset / f"{sample_id}.txt"
            seg_out = output_root / "segments" / subset / f"{sample_id}.txt"

            # Copy/convert image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            cv2.imwrite(str(img_out), img)
            stats[subset]["images"] += 1

            # Copy/convert depth
            # KITTI depth is already uint16 mm format
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue

            # Verify it's uint16 and save
            if depth.dtype != np.uint16:
                # Some versions may be different, convert if needed
                if depth.dtype == np.float32 or depth.dtype == np.float64:
                    depth = (depth * 1000).clip(0, 65535).astype(np.uint16)
                else:
                    depth = depth.astype(np.uint16)

            cv2.imwrite(str(depth_out), depth)
            stats[subset]["depths"] += 1

            # Try to load KITTI 2D detection labels (optional)
            if copy_labels:
                # KITTI detection labels are in label_2/
                label_dir = img_path.parent.parent.parent / "label_02"
                label_file = label_dir / f"{frame}.txt"
                if label_file.exists():
                    yolo_lines = convert_kitti_label_to_yolo(label_file, img.shape[1], img.shape[0])
                    with open(label_out, "w") as f:
                        f.write("\n".join(yolo_lines))

            # Create empty segment file if no segments
            if not seg_out.exists():
                seg_out.touch()

    return stats


def convert_kitti_label_to_yolo(label_file: Path, img_w: int, img_h: int) -> list[str]:
    """Convert KITTI 2D detection label to YOLO format.

    KITTI format:
        type truncated occluded alpha xmin ymin xmax ymax h w l x y z ry

    YOLO format:
        class_id cx cy w h
    """
    lines = []
    with open(label_file) as f:
        for line in f.read().strip().splitlines():
            parts = line.split()
            if len(parts) < 15:
                continue

            obj_type = parts[0]
            if obj_type == "DontCare":
                continue

            cls_id = KITTI_DETECTION_CLASSES.get(obj_type)
            if cls_id is None:
                continue

            try:
                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])
            except ValueError:
                continue

            # Convert to YOLO format
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            # Clip to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            if w > 0 and h > 0:
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def create_kitti_yaml(output_root: Path):
    """Create dataset YAML configuration for KITTI."""
    abs_path = output_root.resolve()

    yaml_path = output_root / "kitti_depth_seg.yaml"
    with open(yaml_path, "w") as f:
        f.write("# KITTI Depth Dataset for YOLO26 Depth+Segmentation\n")
        f.write("# Generated by download_kitti.py\n")
        f.write("#\n")
        f.write("# Dataset Info:\n")
        f.write("#   - Scene: Outdoor driving\n")
        f.write("#   - Depth range: 0~80m\n")
        f.write("#   - Depth source: LiDAR sparse point cloud\n")
        f.write("#   - Depth density: ~5% pixels valid\n")
        f.write("#   - Format: 16-bit PNG (millimeters)\n")
        f.write("#\n")
        f.write("# Usage in training:\n")
        f.write("#   depth_scale: 80.0  (for KITTI, vs 20.0 for NYU)\n")
        f.write("#   valid_mask: depth_target > 0  (sparse depth handling)\n")
        f.write("\n")
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("# Detection labels (bbox) - optional\n")
        f.write("labels: labels/train\n")
        f.write("\n")
        f.write("# Segmentation labels - placeholder\n")
        f.write("mask: segments/train\n")
        f.write("\n")
        f.write("# Depth labels (16-bit PNG, millimeters)\n")
        f.write("depth: depths/train\n")
        f.write("\n")
        f.write("# KITTI detection classes mapped to simplified set\n")
        f.write("nc: 5\n")
        f.write("names:\n")
        f.write("  0: vehicle\n")
        f.write("  1: truck\n")
        f.write("  2: pedestrian\n")
        f.write("  3: cyclist\n")
        f.write("  4: tram\n")
        f.write("\n")
        f.write("# Multi-source dataset config (for MultiSourceDepthDataset)\n")
        f.write("# depth_scale: 80.0\n")
        f.write("# depth_max: 80.0\n")
        f.write("# loss_weight: 1.0\n")

    print(f"\nDataset YAML created: {yaml_path}")


def print_dataset_info(output_root: Path, stats: dict):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("KITTI Depth -> YOLO Format Dataset")
    print("=" * 60)
    print(f"Output: {output_root}")
    print()

    for subset in ["train", "val"]:
        img_dir = output_root / "images" / subset
        depth_dir = output_root / "depths" / subset
        label_dir = output_root / "labels" / subset

        n_img = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        n_depth = len(list(depth_dir.glob("*.png"))) if depth_dir.exists() else 0
        n_label = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0

        print(f"  {subset.upper():>5}: {n_img:>5} images | {n_depth:>5} depths | {n_label:>5} labels")

    print("=" * 60)
    print("\nKey characteristics:")
    print("  - Depth is SPARSE (only ~5% pixels have valid depth)")
    print("  - Depth range: 0-80 meters")
    print("  - Use valid_mask = depth_target > 0 in loss computation")
    print("  - Recommended depth_scale: 80.0")
    print()
    print("Training commands:")
    print(f"  python yolo26_train_depth.py \\")
    print(f"      --model yolo26-seg-depth.yaml \\")
    print(f"      --data {output_root / 'kitti_depth_seg.yaml'} \\")
    print(f"      --pretrained yolo26s-seg.pt \\")
    print(f"      --epochs 100 --batch 8 --device 0")


def verify_kitti_setup(kitti_root: Path) -> bool:
    """Verify that KITTI data is properly set up.

    Accepts either:
    - Full setup with RGB + depth (pairs > 0)
    - Depth-only setup (has depth groundtruth but no RGB yet)
    """
    if not kitti_root.exists():
        return False

    # Check for expected subdirectories
    has_train = (kitti_root / "train").exists()
    has_val = (kitti_root / "val").exists()

    if not has_train and not has_val:
        return False

    # Check if depth data exists (look for groundtruth dir)
    has_depth = False
    for split in ["train", "val"]:
        split_dir = kitti_root / split
        if not split_dir.exists():
            continue
        for scene_dir in split_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            depth_dir = scene_dir / "proj_depth" / "groundtruth"
            if depth_dir.exists():
                has_depth = True
                break
        if has_depth:
            break

    if not has_depth:
        return False

    # Try to find paired (RGB, depth) - if RGB missing, that's OK (will download)
    pairs, missing = find_kitti_depth_pairs(kitti_root)
    if len(pairs) > 0:
        print(f"Found {len(pairs)} valid (image, depth) pairs")
    elif missing:
        print(f"Found depth data. Missing RGB for {len(missing)} scenes (will download if --download-raw)")
    else:
        print("Found depth data")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert KITTI Depth Prediction dataset to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: download + convert
  python download_kitti.py --output ./kitti_yolo

  # Convert existing KITTI data (skip download)
  python download_kitti.py \\
      --kitti-root ./Kitti/data_depth_annotated \\
      --output ./kitti_yolo \\
      --skip-download

  # Verify converted dataset
  python download_kitti.py --verify --output ./kitti_yolo
        """,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./kitti_yolo",
        help="Output directory for YOLO format dataset",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="./kitti_zips",
        help="Directory to download zip files",
    )
    parser.add_argument(
        "--kitti-root",
        type=str,
        default=None,
        help="Path to existing KITTI depth prediction dataset root (with train/ and val/ dirs)",
    )
    parser.add_argument(
        "--eigen-split",
        type=str,
        default=None,
        help="Path to Eigen split file (optional, for custom train/val division)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step, use existing data",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip KITTI 2D detection label conversion",
    )
    parser.add_argument(
        "--download-raw",
        action="store_true",
        help="Download KITTI Raw sync data for scenes missing RGB images (may be 50GB+)",
    )
    parser.add_argument(
        "--raw-download-dir",
        type=str,
        default="./kitti_raw_zips",
        help="Directory to download KITTI Raw sync zip files",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=1.0,
        help="Only use this ratio of train scenes (0.1 = 10%% scenes). Val set is always kept complete. Useful for quick experiments without downloading full 66GB+ of Raw data.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted dataset",
    )

    args = parser.parse_args()

    output_root = Path(args.output)
    download_dir = Path(args.download_dir)

    # Verification mode
    if args.verify:
        print("\nVerifying KITTI YOLO dataset...")
        stats = {"train": {"images": 0, "depths": 0}, "val": {"images": 0, "depths": 0}}
        for subset in ["train", "val"]:
            img_dir = output_root / "images" / subset
            depth_dir = output_root / "depths" / subset
            if img_dir.exists():
                stats[subset]["images"] = len(list(img_dir.glob("*.png")))
            if depth_dir.exists():
                stats[subset]["depths"] = len(list(depth_dir.glob("*.png")))
        print_dataset_info(output_root, stats)
        return

    # Determine KITTI root
    kitti_root = None
    if args.kitti_root:
        kitti_root = Path(args.kitti_root)
        if not verify_kitti_setup(kitti_root):
            print(f"Warning: Could not find valid KITTI data at {kitti_root}")
    else:
        # Try to find extracted data in output directory
        potential_roots = [
            output_root / "data_depth_annotated",
            output_root / "kitti",
            Path("./kitti"),
        ]
        for root in potential_roots:
            if verify_kitti_setup(root):
                kitti_root = root
                print(f"Found KITTI data at: {kitti_root}")
                break

    # Download if needed
    if not args.skip_download and kitti_root is None:
        downloaded = download_kitti_dataset(download_dir, output_root)
        if downloaded:
            extracted = extract_kitti_zips(download_dir, output_root)
            if "depth" in extracted:
                kitti_root = extracted["depth"]

    # Final check
    if kitti_root is None or not verify_kitti_setup(kitti_root):
        print("\n" + "=" * 60)
        print("ERROR: KITTI data not found!")
        print("=" * 60)
        print()
        print("Please download KITTI Depth Prediction dataset manually:")
        print("  1. Visit: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction")
        print("  2. Download 'data_depth_annotated.zip'")
        print("  3. Extract to a directory")
        print("  4. Run: python download_kitti.py --kitti-root <extracted_dir> --output ./kitti_yolo")
        print()
        sys.exit(1)

    # Convert to YOLO format
    print("\n" + "=" * 60)
    print("Converting KITTI to YOLO format")
    print("=" * 60)

    stats = kitti_to_yolo_format(
        kitti_root=kitti_root,
        output_root=output_root,
        eigen_split_file=Path(args.eigen_split) if args.eigen_split else None,
        copy_labels=not args.no_labels,
        download_raw=args.download_raw,
        raw_download_dir=Path(args.raw_download_dir),
        subset_ratio=args.subset_ratio,
    )

    # Create YAML
    create_kitti_yaml(output_root)

    # Print info
    print_dataset_info(output_root, stats)

    print("\nDone! KITTI dataset ready for YOLO26 training.")


if __name__ == "__main__":
    main()
