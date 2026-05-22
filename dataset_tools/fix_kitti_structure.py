#!/usr/bin/env python3
"""
Fix KITTI Raw sync directory structure after download.

KITTI Raw sync zips extract with an extra date folder level:
    train/2011_09_26/2011_09_26_drive_0018_sync/image_02/data/

But depth data is at:
    train/2011_09_26_drive_0018_sync/proj_depth/groundtruth/image_02/

This script moves image_02/ and image_03/ from the nested path to the
scene directory that already contains depth data.

Usage:
    python fix_kitti_structure.py --kitti-root ./kitti_yolo
"""

from pathlib import Path
import argparse
import shutil


def fix_kitti_structure(kitti_root: str):
    kitti_root = Path(kitti_root)

    for split in ["train", "val"]:
        split_dir = kitti_root / split
        if not split_dir.exists():
            continue

        # Find date subdirectories (e.g., 2011_09_26, 2011_09_28)
        date_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.endswith("_sync")]

        for date_dir in date_dirs:
            for scene_dir in date_dir.iterdir():
                if not scene_dir.is_dir() or not scene_dir.name.endswith("_sync"):
                    continue

                # Target: the scene directory at split level (where depth data is)
                target_scene = split_dir / scene_dir.name

                if not target_scene.exists():
                    print(f"  Skip: no depth data for {scene_dir.name}")
                    continue

                # Move image_02/ and image_03/ from nested to target
                for camera in ["image_02", "image_03"]:
                    src = scene_dir / camera
                    dst = target_scene / camera

                    if src.exists():
                        if dst.exists():
                            print(f"  Skip: {dst} already exists")
                        else:
                            shutil.move(str(src), str(dst))
                            print(f"  Moved: {src} -> {dst}")

                # Remove empty scene dir
                remaining = list(scene_dir.iterdir())
                if not remaining:
                    scene_dir.rmdir()
                    print(f"  Removed empty: {scene_dir}")

            # Remove empty date dir
            remaining = list(date_dir.iterdir())
            if not remaining:
                date_dir.rmdir()
                print(f"  Removed empty: {date_dir}")

    print("\nDone. Re-run download_kitti.py to convert.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=str, default="./kitti_yolo",
                        help="Path to KITTI dataset root")
    args = parser.parse_args()
    fix_kitti_structure(args.kitti_root)


if __name__ == "__main__":
    main()
