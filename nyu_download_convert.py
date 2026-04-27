#!/usr/bin/env python3
"""
NYU Depth V2 Dataset Downloader and Converter for YOLO26 Depth+Segmentation Training

This script:
1. Downloads NYU Depth V2 dataset (images + depth + labels)
2. Converts segmentation labels from .mat to YOLO format
3. Organizes data into the structure expected by DepthSegmentDataset

Usage:
    # Download and convert
    python nyu_download_convert.py --download --convert --nyu-root /path/to/nyu --output ./nyu_yolo

    # Convert only (if already downloaded)
    python nyu_download_convert.py --convert --nyu-root /path/to/nyu --output ./nyu_yolo --subset train
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2


# NYU Depth V2 official download links (require registration)
NYU_DOWNLOAD_URLS = {
    "train": "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2_labeled.mat",
    "test": "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2_labeled.mat",
}

# NYU class names (40 classes commonly used)
NYU_CLASSES = {
    0: "wall",
    1: "floor",
    2: "cabinet",
    3: "bed",
    4: "chair",
    5: "sofa",
    6: "table",
    7: "tv",
    8: "desk",
    9: "shelf",
    10: "curtain",
    11: "sink",
    12: "toilet",
    13: "bathtub",
    14: "coffee_table",
    15: "counter",
    16: "refrigerator",
    17: "nightstand",
    18: "sink_base",
    19: "door",
    20: "mirror",
    21: " shower_curtain",
    22: "counter_top",
    23: "wall_cabinet",
    24: "furniture",
    25: "picture",
    26: "window",
    27: "shelving",
    28: "blinds",
    29: "blackboard",
    30: "chair",
    31: "board",
    32: "whiteboard",
    33: "door",
    34: "window",
    35: "shelving",
    36: "chair",
    37: "board",
    38: "whiteboard",
    39: "door",
    40: "blinds",
    41: "curtain",
    42: "shelf",
    43: "desk",
    44: "cabinet",
    45: "picture",
    46: "frame",
    47: "tv",
    48: "mirror",
    49: "wall",
    50: "floor",
    51: "ceiling",
    52: "pillow",
    53: "lamp",
    54: "person",
    55: "dog",
    56: "cat",
}


def download_nyu_dataset(output_dir: Path, subset: str = "train"):
    """Download NYU Depth V2 dataset using gdown or direct link.

    Note: NYU dataset requires registration. This script provides
    download methods with fallback options.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading NYU Depth V2 dataset ({subset} set)...")
    print(
        f"Note: You must register at https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"
    )
    print(f"Download the 'labeled dataset' and extract to: {output_dir}")

    # Method 1: Try using gdown if available
    try:
        import gdown

        file_id = "1R1l0M2lL1v4PbNMWtPcDzcBzU6d7NqE"  # Example file ID
        output = output_dir / "nyu_depth_v2_labeled.mat"
        gdown.download(id=file_id, output=str(output), quiet=False)
        return output
    except ImportError:
        pass

    # Method 2: Direct wget/curl (if user has the URL)
    mat_file = output_dir / "nyu_depth_v2_labeled.mat"
    if mat_file.exists():
        print(f"Found existing .mat file: {mat_file}")
        return mat_file

    print(f"\nPlease download manually:")
    print(f"1. Go to https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
    print(f"2. Register and download 'nyu_depth_v2_labeled.mat'")
    print(f"3. Place the file at: {mat_file}")
    print(f"4. Run this script again with --convert flag")

    return None


def convert_nyu_to_yolo(
    nyu_root: str, output_root: str, subset: str = "train", target_classes: int = 40
):
    """Convert NYU Depth V2 dataset to YOLO format.

    Args:
        nyu_root: Path to NYU Depth V2 root directory (contains images/ and labels/)
        output_root: Output directory for YOLO format dataset
        subset: 'train' or 'val' or 'test'
        target_classes: Number of classes to keep (40 for NYU40, 13 for NYU13)
    """
    from scipy.io import loadmat

    nyu_root = Path(nyu_root)
    output_root = Path(output_root)

    # Create output directories
    images_dir = output_root / "images" / subset
    depths_dir = output_root / "depths" / subset
    segments_dir = output_root / "segments" / subset

    for d in [images_dir, depths_dir, segments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find .mat files (labels)
    labels_dir = nyu_root / "labels"
    mat_files = sorted(labels_dir.glob(f"*{subset}*.mat"))

    if not mat_files:
        labels_dir = nyu_root / "labels" / subset
        mat_files = sorted(labels_dir.glob("*.mat"))

    if not mat_files:
        print(f"No .mat files found in {labels_dir}")
        return

    # Get raw images directory
    raw_images_dir = nyu_root / "images" / subset
    if not raw_images_dir.exists():
        raw_images_dir = nyu_root / "images"

    print(f"Found {len(mat_files)} .mat files, converting to YOLO format...")

    # NYU13 simplified class mapping (indoor scene understanding)
    NYU13_CLASSES = {
        0: "wall",
        1: "floor",
        2: "cabinet",
        3: "bed",
        4: "chair",
        5: "sofa",
        6: "table",
        7: "tv",
        8: "furniture",
        9: "objects",
    }

    NYU40_TO_13 = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 4,
        9: 8,
        10: 8,
        11: 9,
        12: 9,
        13: 9,
        14: 6,
        15: 2,
        16: 2,
        17: 3,
        18: 9,
        19: 8,
        20: 8,
        21: 8,
        22: 2,
        23: 2,
        24: 8,
        25: 8,
        26: 8,
        27: 8,
        28: 8,
        29: 8,
        30: 4,
        31: 8,
        32: 8,
        33: 8,
        34: 8,
        35: 8,
        36: 8,
        37: 8,
        38: 8,
        39: 0,
        40: 1,
        41: 1,
        42: 8,
        43: 2,
        44: 2,
        45: 8,
        46: 8,
        47: 7,
        48: 8,
        49: 0,
        50: 1,
        51: 8,
        52: 8,
        53: 8,
        54: 9,
        55: 9,
        56: 9,
    }

    converted_count = 0

    for mat_file in tqdm(mat_files, desc=f"Converting {subset}"):
        scene_id = mat_file.stem.replace("_labels", "")

        # Find corresponding image
        img_file = None
        for ext in [".jpg", ".png", ".jpeg"]:
            potential = raw_images_dir / f"{scene_id}{ext}"
            if potential.exists():
                img_file = potential
                break

        if not img_file:
            # Try without subset in path
            potential = nyu_root / "images" / f"{scene_id}.jpg"
            if potential.exists():
                img_file = potential

        if not img_file:
            print(f"Warning: Image not found for {scene_id}")
            continue

        # Load .mat file
        try:
            mat = loadmat(str(mat_file))
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            continue

        # Get image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        h, w = img.shape[:2]

        # Save image
        img_out = images_dir / f"{scene_id}.jpg"
        if str(img_file) != str(img_out):
            cv2.imwrite(str(img_out), img)

        # Extract and save depth
        depth_data = None
        if "depths" in mat:
            depth_data = mat["depths"]
        elif "depth" in mat:
            depth_data = mat["depth"]

        if depth_data is not None:
            # Convert to 16-bit PNG (millimeters)
            depth_mm = (depth_data * 1000).astype(np.uint16)
            depth_out = depths_dir / f"{scene_id}.png"
            cv2.imwrite(str(depth_out), depth_mm)
        else:
            # Try to find depth as separate file
            depth_png = nyu_root / "depths" / f"{scene_id}.png"
            if depth_png.exists():
                shutil.copy(str(depth_png), str(depths_dir / f"{scene_id}.png"))
            else:
                # Create empty depth placeholder
                depth_zeros = np.zeros((h, w), dtype=np.uint16)
                cv2.imwrite(str(depths_dir / f"{scene_id}.png"), depth_zeros)

        # Extract instance segmentation and convert to YOLO polygon format
        instances = mat.get("labels", None)
        if instances is None:
            instances = mat.get("instance", None)

        seg_out = segments_dir / f"{scene_id}.txt"
        seg_lines = []

        if instances is not None:
            instance_ids = np.unique(instances)

            for inst_id in instance_ids:
                if inst_id <= 0:
                    continue

                # Create binary mask for this instance
                mask = (instances == inst_id).astype(np.uint8)

                # Find contours (polygon boundary)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    contour = contour.squeeze()
                    if contour.ndim == 1:
                        continue

                    # Normalize coordinates to [0, 1]
                    coords = contour / np.array([w, h], dtype=np.float32)
                    coords = coords.flatten()

                    # Map class ID (40 classes -> 13 or use original)
                    orig_cls = int(inst_id) if inst_id < 1000 else int(inst_id // 1000)
                    orig_cls = max(0, min(orig_cls, 255))

                    # Use NYU13 simplified classes
                    mapped_cls = NYU40_TO_13.get(
                        orig_cls, 8
                    )  # Default to "objects" class

                    line = f"{mapped_cls} " + " ".join([f"{c:.6f}" for c in coords])
                    seg_lines.append(line)

        with open(seg_out, "w") as f:
            f.write("\n".join(seg_lines))

        converted_count += 1

    print(f"Successfully converted {converted_count} samples")
    return converted_count


def create_dataset_yaml(output_root: str, nc: int = 13, class_names: list = None):
    """Create dataset YAML configuration file."""
    if class_names is None:
        class_names = [
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "tv",
            "furniture",
            "objects",
        ][:nc]

    yaml_content = f"""# NYU Depth V2 Dataset Configuration for YOLO26 Depth+Segmentation Training
# Generated by nyu_download_convert.py

path: {output_root}
train: images/train
val: images/val
test: images/test

# Segmentation labels
mask: segments/train

# Depth labels (16-bit PNG, millimeters)
depth: depths/train

# Number of classes
nc: {nc}

# Class names
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = Path(output_root).parent / "nyu_depth_seg.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset YAML created: {yaml_path}")
    return yaml_path


def download_with_gdown(file_id: str, output_path: Path, chunk_size: int = 32768):
    """Download file using gdown library."""
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=False)


def verify_dataset(output_root: str, subset: str = "train") -> dict:
    """Verify converted dataset integrity."""
    output_root = Path(output_root)

    images_dir = output_root / "images" / subset
    depths_dir = output_root / "depths" / subset
    segments_dir = output_root / "segments" / subset

    stats = {
        "images": 0,
        "depths": 0,
        "segments": 0,
        "missing_depths": [],
        "empty_segments": [],
    }

    if images_dir.exists():
        stats["images"] = len(list(images_dir.glob("*")))

    if depths_dir.exists():
        stats["depths"] = len(list(depths_dir.glob("*")))
        for seg_file in segments_dir.glob("*.txt") if segments_dir.exists() else []:
            if not (depths_dir / seg_file.stem).with_suffix(".png").exists():
                stats["missing_depths"].append(seg_file.stem)

    if segments_dir.exists():
        seg_files = list(segments_dir.glob("*.txt"))
        stats["segments"] = len(seg_files)
        for seg_file in seg_files:
            with open(seg_file) as f:
                content = f.read().strip()
                if not content:
                    stats["empty_segments"].append(seg_file.stem)

    return stats


def print_dataset_info(output_root: str):
    """Print dataset structure and statistics."""
    output_root = Path(output_root)

    print("\n" + "=" * 60)
    print("NYU Depth V2 -> YOLO Format Dataset")
    print("=" * 60)
    print(f"Output root: {output_root}")
    print()

    for subset in ["train", "val", "test"]:
        images_dir = output_root / "images" / subset
        depths_dir = output_root / "depths" / subset
        segments_dir = output_root / "segments" / subset

        if images_dir.exists():
            n_images = len(list(images_dir.glob("*")))
            n_depths = len(list(depths_dir.glob("*"))) if depths_dir.exists() else 0
            n_segs = (
                len(list(segments_dir.glob("*.txt"))) if segments_dir.exists() else 0
            )

            print(
                f"{subset.upper():>6}: {n_images:>5} images | {n_depths:>5} depths | {n_segs:>5} segments"
            )

    print("=" * 60)
    print("\nDataset is ready for training!")
    print(f"Use with yolo26_train_depth.py:")
    print(
        f"  python yolo26_train_depth.py --model yolo26-seg-depth.yaml --data {output_root.parent / 'nyu_depth_seg.yaml'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="NYU Depth V2 Downloader and YOLO Converter for YOLO26",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and convert (requires manual download of NYU dataset)
  python nyu_download_convert.py --download --convert \\
      --nyu-root ./nyu_raw --output ./nyu_yolo

  # Convert existing NYU dataset
  python nyu_download_convert.py --convert \\
      --nyu-root /path/to/nyu --output ./nyu_yolo --subset train

  # Convert with 40 classes (original NYU classes)
  python nyu_download_convert.py --convert \\
      --nyu-root /path/to/nyu --output ./nyu_yolo --nc 40

  # Verify dataset
  python nyu_download_convert.py --verify --output ./nyu_yolo
        """,
    )

    parser.add_argument("--download", action="store_true", help="Download NYU dataset")
    parser.add_argument("--convert", action="store_true", help="Convert to YOLO format")
    parser.add_argument(
        "--verify", action="store_true", help="Verify converted dataset"
    )

    parser.add_argument(
        "--nyu-root",
        type=str,
        default="./nyu_raw",
        help="NYU Depth V2 root directory (with images/ and labels/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./nyu_yolo",
        help="Output directory for YOLO format dataset",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Subset to convert",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=13,
        help="Number of classes (13 for NYU13, 40 for NYU40)",
    )

    parser.add_argument(
        "--create-yaml",
        action="store_true",
        help="Create dataset YAML configuration file",
    )

    args = parser.parse_args()

    if args.download:
        mat_file = download_nyu_dataset(args.output, args.subset)
        if mat_file is None:
            print("\nPlease download NYU dataset manually and run with --convert")

    if args.convert:
        nyu_root = Path(args.nyu_root)
        output_root = Path(args.output)

        if not nyu_root.exists():
            print(f"Error: NYU root directory not found: {nyu_root}")
            print(
                "Please ensure you have downloaded and extracted the NYU Depth V2 dataset"
            )
            return

        subsets = ["train", "val", "test"] if args.subset == "all" else [args.subset]

        total_converted = 0
        for subset in subsets:
            print(f"\n--- Converting {subset} set ---")
            count = convert_nyu_to_yolo(
                str(nyu_root), str(output_root), subset, args.nc
            )
            total_converted += count if count else 0

        print(f"\nTotal converted: {total_converted} samples")

        if args.create_yaml:
            class_names = [
                "wall",
                "floor",
                "cabinet",
                "bed",
                "chair",
                "sofa",
                "table",
                "tv",
                "furniture",
                "objects",
            ][: args.nc]
            create_dataset_yaml(str(output_root), args.nc, class_names)

        print_dataset_info(str(output_root))

    if args.verify:
        stats = verify_dataset(args.output)
        print("\nDataset verification:")
        print(f"  Images: {stats['images']}")
        print(f"  Depths: {stats['depths']}")
        print(f"  Segments: {stats['segments']}")

        if stats["missing_depths"]:
            print(
                f"\n  Warning: {len(stats['missing_depths'])} samples missing depth files"
            )
        if stats["empty_segments"]:
            print(
                f"  Warning: {len(stats['empty_segments'])} samples have empty segment files"
            )


if __name__ == "__main__":
    main()
