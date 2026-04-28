#!/usr/bin/env python3
"""
NYU Depth V2 .mat Converter for YOLO26 Depth+Segmentation Training

Converts the official NYU Depth V2 labeled .mat file (nyu_depth_v2_labeled.mat)
into YOLO format with images/, depths/, segments/ directory structure.

Usage:
    # Convert from .mat file (MATLAB v7.3 / HDF5)
    python nyu_mat_converter.py \
        --mat-path "D:\\ms download\\nyu_depth_v2_labeled.mat" \
        --output ./nyu_yolo

    # Convert with 13 simplified classes
    python nyu_mat_converter.py \
        --mat-path "D:\\ms download\\nyu_depth_v2_labeled.mat" \
        --output ./nyu_yolo \
        --nc 13
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


# NYU Depth V2 class mapping (semantic labels -> 40 classes)
# The .mat file stores labels as semantic class IDs per pixel.
# We use the official 40-class split from the dataset.

NYU40_CLASS_NAMES = {
    0: "void",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor_mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "fridge",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower_curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "night_stand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "otherstructure",
    39: "otherfurniture",
    40: "otherprop",
}

# NYU13 name-based mapping (maps name index -> NYU13 class)
# Build from names field in .mat file
# IMPORTANT: class 0 is "void" and will be SKIPPED by semantic_to_yolo_labels.
# Do NOT map any valid object to 0.
NYU_NAME_TO_13 = {
    # wall (1)
    "wall": 1,
    "ceiling": 1,
    "pillar": 1,
    "column": 1,
    # floor (2)
    "floor": 2,
    "rug": 2,
    "carpet": 2,
    "mat": 2,
    # cabinet (3) - storage furniture
    "cabinet": 3,
    "counter": 3,
    "dresser": 3,
    "refrigerator": 3,
    "fridge": 3,
    "dishwasher": 3,
    "drawer": 3,
    "night stand": 3,
    "nightstand": 3,
    "bookshelf": 3,
    "bookcase": 3,
    "closet": 3,
    "wardrobe": 3,
    "cupboard": 3,
    "shelf": 3,
    "shelves": 3,
    "rack": 3,
    # bed (4)
    "bed": 4,
    "crib": 4,
    "cradle": 4,
    "bassinet": 4,
    "mattress": 4,
    "headboard": 4,
    "bunk bed": 4,
    # chair (5) - seating furniture
    "chair": 5,
    "sofa": 5,
    "couch": 5,
    "stool": 5,
    "ottoman": 5,
    "bench": 5,
    "bean bag": 5,
    "piano bench": 5,
    # table (6)
    "table": 6,
    "desk": 6,
    "coffee table": 6,
    "countertop": 6,
    # furniture (7) - structural / room elements
    "door": 7,
    "window": 7,
    "blinds": 7,
    "curtain": 7,
    "shower curtain": 7,
    "mirror": 7,
    "bathtub": 7,
    "shower": 7,
    "toilet": 7,
    "sink": 7,
    "faucet": 7,
    "lamp": 7,
    "light": 7,
    "fireplace": 7,
    "stairs": 7,
    "banister": 7,
    "railing": 7,
    "chimney": 7,
    "fire extinguisher": 7,
    "picture": 7,
    "photo": 7,
    "poster": 7,
    "whiteboard": 7,
    "blackboard": 7,
    "chalkboard": 7,
    "board": 7,
    "garage door": 7,
    "door frame": 7,
    "door way": 7,
    "window frame": 7,
    # objects (8) - movable objects / small items
    "book": 8,
    "bottle": 8,
    "paper": 8,
    "cup": 8,
    "glass": 8,
    "bowl": 8,
    "plate": 8,
    "box": 8,
    "bag": 8,
    "toy": 8,
    "computer": 8,
    "laptop": 8,
    "keyboard": 8,
    "mouse": 8,
    "monitor": 8,
    "television": 8,
    "tv": 8,
    "remote": 8,
    "phone": 8,
    "telephone": 8,
    "clock": 8,
    "vase": 8,
    "flower": 8,
    "plant": 8,
    "shoe": 8,
    "clothes": 8,
    "backpack": 8,
    "umbrella": 8,
    "suitcase": 8,
    "keyboard": 8,
}

# For direct label values (if labels are already semantic IDs)
NYU40_TO_13 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 7,
    9: 7,
    10: 7,
    11: 7,
    12: 3,
    13: 7,
    14: 6,
    15: 7,
    16: 7,
    17: 3,
    18: 7,
    19: 7,
    20: 2,
    21: 9,
    22: 0,
    23: 9,
    24: 3,
    25: 9,
    26: 9,
    27: 9,
    28: 9,
    29: 9,
    30: 9,
    31: 9,
    32: 4,
    33: 7,
    34: 7,
    35: 7,
    36: 7,
    37: 9,
    38: 9,
    39: 7,
    40: 9,
}

NYU13_CLASS_NAMES = {
    0: "void",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "table",
    7: "furniture",
    8: "objects",
    9: "other",
}


def build_name_to_nyu13_mapping(names_list: list[str]) -> dict[int, int]:
    """Build mapping from name index to NYU13 class ID.

    Args:
        names_list: List of class names from NYU dataset

    Returns:
        Dict mapping name index (0-893) to NYU13 class (0-9)
    """
    mapping = {}
    for idx, name in enumerate(names_list):
        name_lower = name.lower()
        mapped = None

        # Priority 1: exact keyword match
        for keyword, cls_id in NYU_NAME_TO_13.items():
            if keyword in name_lower:
                mapped = cls_id
                break

        if mapped is None:
            # Priority 2: fallback rules for unmapped names
            if any(k in name_lower for k in
                    ["wall", "ceiling", "pillar", "column"]):
                mapped = 1
            elif any(k in name_lower for k in
                      ["floor", "rug", "carpet", "mat"]):
                mapped = 2
            elif any(k in name_lower for k in
                      ["cabinet", "counter", "dresser", "drawer",
                       "refrigerator", "fridge", "shelf", "rack",
                       "closet", "wardrobe", "cupboard"]):
                mapped = 3
            elif any(k in name_lower for k in
                      ["bed", "crib", "cradle", "bassinet",
                       "mattress", "headboard", "comforter",
                       "blanket", "sheet"]):
                mapped = 4
            elif any(k in name_lower for k in
                      ["chair", "sofa", "couch", "stool",
                       "ottoman", "bench", "seat", "bean bag"]):
                mapped = 5
            elif any(k in name_lower for k in
                      ["table", "desk", "countertop"]):
                mapped = 6
            elif any(k in name_lower for k in
                      ["door", "window", "blind", "curtain",
                       "mirror", "bathtub", "toilet", "sink",
                       "shower", "lamp", "light", "fireplace",
                       "stairs", "banister", "railing", "faucet",
                       "chimney", "vent", "duct", "heater",
                       "radiator", "outlet", "switch"]):
                mapped = 7
            else:
                # Default to objects (8) instead of other (9)
                mapped = 8

        mapping[idx] = mapped

    return mapping


def decode_matlab_string(h5_ref):
    """Decode MATLAB string stored in HDF5 reference."""
    if isinstance(h5_ref, h5py.Reference):
        return ""
    if isinstance(h5_ref, bytes):
        return h5_ref.decode("utf-8", errors="ignore")
    if isinstance(h5_ref, str):
        return h5_ref
    if hasattr(h5_ref, "shape"):
        # MATLAB stores strings as uint16 arrays
        try:
            chars = "".join([chr(c) for c in h5_ref.flatten() if c > 0])
            return chars
        except Exception:
            return str(h5_ref)
    return str(h5_ref)


def load_nyu_mat_v73(mat_path: str) -> dict:
    """Load NYU Depth V2 labeled .mat file (MATLAB v7.3 / HDF5 format).

    Returns dict with keys:
        images:      list of (H, W, 3) uint8 RGB images
        depths:      list of (H, W) float32 depth in meters
        labels:      list of (H, W) uint16 semantic labels (0=void)
        scenes:      list of scene names
        names:       list of class names
    """
    print(f"Loading NYU .mat file (HDF5/v7.3): {mat_path}")

    data = {
        "images": [],
        "depths": [],
        "labels": [],
        "scenes": [],
        "names": [],
    }

    with h5py.File(mat_path, "r") as f:
        keys = list(f.keys())
        print(f"  HDF5 keys: {keys}")

        def get_dataset(key):
            if key in f:
                return f[key]
            for root_key in keys:
                if root_key not in f:
                    continue
                item = f[root_key]
                if isinstance(item, h5py.Group) and key in item:
                    return item[key]
            return None

        # --- Load images: shape detected from actual data ---
        images_ds = get_dataset("images")
        if images_ds is not None:
            print(
                f"  images dataset shape: {images_ds.shape}, dtype: {images_ds.dtype}"
            )
            shape = images_ds.shape
            ndim = len(shape)

            if ndim == 4:
                if shape[0] == 3:
                    # (3, H, W, N)  old MATLAB-HDF5 layout
                    n = shape[-1]
                    h, w = shape[1], shape[2]
                    print(f"  Layout: (3, H, W, N)  -> N={n}, H={h}, W={w}")
                    print(f"  Reading all images into memory...")
                    arr = images_ds[:]  # (3, H, W, N)
                    arr = np.transpose(arr, (3, 1, 2, 0))  # (N, H, W, 3)
                elif shape[1] == 3:
                    # (N, 3, H, W)  your actual layout
                    n = shape[0]
                    h, w = shape[2], shape[3]
                    print(f"  Layout: (N, 3, H, W)  -> N={n}, H={h}, W={w}")
                    print(f"  Reading all images into memory...")
                    arr = images_ds[:]  # (N, 3, H, W)
                    arr = np.transpose(arr, (0, 2, 3, 1))  # (N, H, W, 3)
                else:
                    raise ValueError(f"Unexpected images shape: {shape}")
            else:
                raise ValueError(f"Unexpected images ndim: {ndim}, shape: {shape}")

            arr = np.clip(arr, 0, 255).astype(np.uint8)
            data["images"] = [arr[i] for i in range(n)]
            print(f"  Loaded {n} images")
        else:
            raise ValueError("'images' dataset not found in .mat file")

        # --- Load depths: auto-detect shape ---
        depths_ds = get_dataset("depths")
        if depths_ds is not None:
            print(f"  depths dataset shape: {depths_ds.shape}")
            print(f"  Reading all depths into memory...")
            arr = depths_ds[:]  # load to memory
            if arr.ndim == 3:
                if arr.shape[0] == n:
                    # (N, H, W)
                    data["depths"] = [arr[i].astype(np.float32) for i in range(n)]
                elif arr.shape[-1] == n:
                    # (H, W, N)
                    data["depths"] = [arr[:, :, i].astype(np.float32) for i in range(n)]
                else:
                    raise ValueError(f"Unexpected depths shape: {arr.shape}")
            else:
                raise ValueError(f"Unexpected depths ndim: {arr.ndim}")
            print(f"  Loaded {len(data['depths'])} depths")
        else:
            raise ValueError("'depths' dataset not found in .mat file")

        # --- Load labels: auto-detect shape ---
        labels_ds = get_dataset("labels")
        if labels_ds is not None:
            print(f"  labels dataset shape: {labels_ds.shape}")
            print(f"  Reading all labels into memory...")
            arr = labels_ds[:]
            if arr.ndim == 3:
                if arr.shape[0] == n:
                    data["labels"] = [arr[i].astype(np.uint16) for i in range(n)]
                elif arr.shape[-1] == n:
                    data["labels"] = [arr[:, :, i].astype(np.uint16) for i in range(n)]
                else:
                    raise ValueError(f"Unexpected labels shape: {arr.shape}")
            else:
                raise ValueError(f"Unexpected labels ndim: {arr.ndim}")
            print(f"  Loaded {len(data['labels'])} labels")
        else:
            raise ValueError("'labels' dataset not found in .mat file")

        # --- Load scene names ---
        scenes_ds = get_dataset("scenes")
        if scenes_ds is not None:
            print(f"  scenes dataset shape: {scenes_ds.shape}")
            try:
                # MATLAB cell array stored as object references.
                # Try different indexing strategies.
                shape = scenes_ds.shape
                n_scenes = (
                    shape[0]
                    if shape[0] == n
                    else (shape[1] if len(shape) > 1 and shape[1] == n else shape[0])
                )

                for i in range(n_scenes):
                    # Try (i, 0) for 2D, or (i,) for 1D
                    try:
                        if len(shape) > 1 and shape[1] == n:
                            ref = scenes_ds[0, i]
                        elif len(shape) > 1:
                            ref = scenes_ds[i, 0]
                        else:
                            ref = scenes_ds[i]
                    except (IndexError, ValueError):
                        ref = scenes_ds[i]

                    if isinstance(ref, h5py.Reference):
                        scene_obj = f[ref]
                        scene_str = decode_matlab_string(scene_obj[:])
                        data["scenes"].append(scene_str)
                    else:
                        data["scenes"].append(str(ref))

                # Verify we got enough scenes
                if len(data["scenes"]) != n:
                    print(f"  Warning: expected {n} scenes, got {len(data['scenes'])}")
                    data["scenes"] = [f"scene_{i}" for i in range(n)]
                else:
                    unique_scenes = sorted(set(data["scenes"]))
                    print(
                        f"  Unique scenes: {len(unique_scenes)} (e.g., {unique_scenes[:3]})"
                    )
            except Exception as e:
                print(f"  Warning: could not read scenes: {e}")
                data["scenes"] = [f"scene_{i}" for i in range(n)]
        else:
            data["scenes"] = [f"scene_{i}" for i in range(n)]

        # --- Load class names ---
        names_ds = get_dataset("names")
        if names_ds is not None:
            try:
                # names is (1, 894) with cell array references
                for i in range(names_ds.shape[1]):
                    ref = names_ds[0, i]
                    if isinstance(ref, h5py.Reference):
                        name_obj = f[ref]
                        name_str = decode_matlab_string(name_obj[:])
                        data["names"].append(name_str)
                    else:
                        data["names"].append(str(ref))
                print(f"  Class names: {data['names'][:5]}...")
                print(f"  Total names: {len(data['names'])}")

                # Build name-to-NYU13 mapping
                data["name_to_nyu13"] = build_name_to_nyu13_mapping(data["names"])
                print(f"  Name mapping built: {len(data['name_to_nyu13'])} entries")
            except Exception as e:
                print(f"  Warning: could not read names: {e}")

    print(f"  Loaded {len(data['images'])} samples successfully")
    return data


def extract_train_test_indices(mat_data: dict, train_ratio: float = 0.8) -> tuple:
    """Extract or create train/test split indices.

    If 'scenes' field exists and has enough variety, use scene-based split.
    Otherwise fall back to random split.
    """
    n = len(mat_data["images"])

    scenes = mat_data.get("scenes", [])
    use_scene_split = False

    if scenes and len(scenes) == n:
        unique_scenes = sorted(set(scenes))
        if len(unique_scenes) >= 2:
            n_train_scenes = max(1, int(len(unique_scenes) * train_ratio))
            train_scenes = set(unique_scenes[:n_train_scenes])

            train_indices = [i for i, s in enumerate(scenes) if s in train_scenes]
            test_indices = [i for i, s in enumerate(scenes) if s not in train_scenes]

            print(
                f"Scene-based split: {len(train_scenes)} train scenes, "
                f"{len(unique_scenes) - len(train_scenes)} test scenes"
            )
            use_scene_split = True
        else:
            print(
                f"  Only {len(unique_scenes)} unique scene(s) found, "
                f"falling back to random split"
            )

    if not use_scene_split:
        # Random split
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(n * train_ratio)
        train_indices = indices[:split_idx].tolist()
        test_indices = indices[split_idx:].tolist()
        print(f"Random split: {len(train_indices)} train, {len(test_indices)} test")

    return train_indices, test_indices


def save_image(img: np.ndarray, path: Path):
    """Save RGB image."""
    # img is (H, W, 3) RGB from .mat, save as BGR for OpenCV
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_depth(depth: np.ndarray, path: Path):
    """Save depth map as 16-bit PNG (millimeters).

    NYU depths are in meters (float).
    Convert to uint16 mm for storage.
    """
    # Convert meters to millimeters, clip to valid range
    depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), depth_mm)


def semantic_to_yolo_labels(
    label_map: np.ndarray,
    img_h: int,
    img_w: int,
    class_mapping: dict | None = None,
    min_area: float = 100.0,
) -> tuple[list[str], list[str]]:
    """Convert semantic segmentation label map to YOLO polygon and bbox format.

    For each class, finds connected components and extracts polygon contours
    and bounding boxes.

    Args:
        label_map: (H, W) semantic label array
        img_h, img_w: image dimensions
        class_mapping: dict mapping original class IDs to target class IDs
        min_area: minimum contour area in pixels

    Returns:
        Tuple of (segment_lines, bbox_lines) where:
            segment_lines: List of YOLO polygon format lines
            bbox_lines: List of YOLO bbox format lines "class_id cx cy w h"
    """
    seg_lines = []
    bbox_lines = []
    unique_classes = np.unique(label_map)

    for cls_id in unique_classes:
        if cls_id <= 0:
            continue  # skip void/background

        target_cls = (
            class_mapping.get(int(cls_id), int(cls_id))
            if class_mapping
            else int(cls_id)
        )
        if target_cls < 0:
            continue

        # Binary mask for this class
        mask = (label_map == cls_id).astype(np.uint8)

        # Find connected components (separate instances)
        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        for inst_id in range(1, num_labels):
            area = stats[inst_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            # Get bbox from connected component stats
            x = stats[inst_id, cv2.CC_STAT_LEFT]
            y = stats[inst_id, cv2.CC_STAT_TOP]
            w = stats[inst_id, cv2.CC_STAT_WIDTH]
            h_bbox = stats[inst_id, cv2.CC_STAT_HEIGHT]

            # Normalize bbox to YOLO format (cx, cy, w, h)
            cx = (x + w / 2) / img_w
            cy = (y + h_bbox / 2) / img_h
            nw = w / img_w
            nh = h_bbox / img_h
            bbox_lines.append(f"{target_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            inst_mask = (labels_cc == inst_id).astype(np.uint8)

            # Find contour
            contours, _ = cv2.findContours(
                inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if len(contour) < 3:
                    continue

                # Simplify polygon with epsilon
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) < 3:
                    continue

                # Flatten and normalize
                coords = approx.reshape(-1, 2).astype(np.float32)
                coords[:, 0] /= img_w
                coords[:, 1] /= img_h
                coords = np.clip(coords, 0.0, 1.0)

                # Format: class_id x1 y1 x2 y2 ...
                coord_str = " ".join([f"{c:.6f}" for c in coords.flatten()])
                seg_lines.append(f"{target_cls} {coord_str}")

    return seg_lines, bbox_lines


def convert_single_sample(
    mat_data: dict,
    idx: int,
    output_root: Path,
    subset: str,
    nc: int = 13,
):
    """Convert a single sample from .mat to YOLO format files."""
    # Extract data
    img = mat_data["images"][idx]  # (H, W, 3)
    depth = mat_data["depths"][idx]  # (H, W)
    labels = mat_data["labels"][idx]  # (H, W)

    h, w = img.shape[:2]

    # Create output paths
    sample_id = f"{idx:04d}"
    img_path = output_root / "images" / subset / f"{sample_id}.jpg"
    depth_path = output_root / "depths" / subset / f"{sample_id}.png"
    seg_path = output_root / "segments" / subset / f"{sample_id}.txt"
    label_path = output_root / "labels" / subset / f"{sample_id}.txt"

    # Save image
    save_image(img, img_path)

    # Save depth
    save_depth(depth, depth_path)

    # Convert labels to YOLO polygons and bboxes
    # Use name-based mapping if available (NYU labels are name indices 0-894)
    name_mapping = mat_data.get("name_to_nyu13")
    if name_mapping:
        # Map each pixel's name index to NYU13 class
        # Note: labels=0 is void/background, labels=N corresponds to name index N-1
        mapped_labels = np.zeros_like(labels)
        for name_idx, nyu13_cls in name_mapping.items():
            mapped_labels[labels == (name_idx + 1)] = nyu13_cls
        labels = mapped_labels
        class_mapping = None  # already mapped
    else:
        class_mapping = NYU40_TO_13 if nc == 13 else None

    seg_lines, bbox_lines = semantic_to_yolo_labels(labels, h, w, class_mapping)

    with open(seg_path, "w") as f:
        f.write("\n".join(seg_lines))

    with open(label_path, "w") as f:
        f.write("\n".join(bbox_lines))

    return len(seg_lines)


def convert_nyu_mat_to_yolo(
    mat_path: str,
    output_root: str,
    nc: int = 13,
    train_ratio: float = 0.8,
):
    """Main conversion function.

    Args:
        mat_path: Path to nyu_depth_v2_labeled.mat
        output_root: Output directory for YOLO format dataset
        nc: Number of classes (13 for NYU13, 40 for NYU40)
        train_ratio: Ratio for train/test split
    """
    mat_path = Path(mat_path)
    output_root = Path(output_root)

    if not mat_path.exists():
        raise FileNotFoundError(f"Mat file not found: {mat_path}")

    # Create directories
    for subset in ["train", "test"]:
        (output_root / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_root / "depths" / subset).mkdir(parents=True, exist_ok=True)
        (output_root / "segments" / subset).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / subset).mkdir(parents=True, exist_ok=True)

    # Load .mat (HDF5 v7.3)
    mat_data = load_nyu_mat_v73(str(mat_path))
    n_total = len(mat_data["images"])

    # Split
    train_indices, test_indices = extract_train_test_indices(mat_data, train_ratio)

    # Convert train set
    print(f"\n[Phase 1/3] Converting train set ({len(train_indices)} samples)...")
    train_seg_count = 0
    for i, idx in enumerate(tqdm(train_indices, desc="train", ncols=80)):
        n_segs = convert_single_sample(mat_data, idx, output_root, "train", nc)
        train_seg_count += n_segs
        if (i + 1) % 100 == 0 or (i + 1) == len(train_indices):
            print(
                f"  [train] Processed {i + 1}/{len(train_indices)} images, "
                f"total {train_seg_count} segment objects"
            )
    print(
        f"[Phase 1/3] Train set done: {len(train_indices)} images, "
        f"{train_seg_count} segment objects"
    )

    # Convert test set
    print(f"\n[Phase 2/3] Converting test set ({len(test_indices)} samples)...")
    test_seg_count = 0
    for i, idx in enumerate(tqdm(test_indices, desc="test", ncols=80)):
        n_segs = convert_single_sample(mat_data, idx, output_root, "test", nc)
        test_seg_count += n_segs
        if (i + 1) % 50 == 0 or (i + 1) == len(test_indices):
            print(
                f"  [test] Processed {i + 1}/{len(test_indices)} images, "
                f"total {test_seg_count} segment objects"
            )
    print(
        f"[Phase 2/3] Test set done: {len(test_indices)} images, "
        f"{test_seg_count} segment objects"
    )

    # Get class names
    if nc == 13:
        class_names = [NYU13_CLASS_NAMES[i] for i in range(10)]
        nc = len(class_names)  # actual number of classes = 10 (0-9)
    else:
        class_names = [NYU40_CLASS_NAMES.get(i, f"class_{i}") for i in range(nc + 1)]

    # Create YAML (nc must match len(names))
    create_yaml(output_root, nc, class_names)

    # Print summary
    print_dataset_info(output_root)


def create_yaml(output_root: Path, nc: int, class_names: list[str]):
    """Create dataset YAML configuration."""
    # Use absolute path for reliability
    abs_path = output_root.resolve()

    yaml_path = output_root / "nyu_depth_seg.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"# NYU Depth V2 Dataset for YOLO26 Depth+Segmentation\n")
        f.write(f"# Generated by nyu_mat_converter.py\n")
        f.write(f"path: {abs_path}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/test\n")
        f.write(f"\n")
        f.write(f"# Detection labels (bbox)\n")
        f.write(f"labels: labels/train\n")
        f.write(f"\n")
        f.write(f"# Segmentation labels\n")
        f.write(f"mask: segments/train\n")
        f.write(f"\n")
        f.write(f"# Depth labels (16-bit PNG, millimeters)\n")
        f.write(f"depth: depths/train\n")
        f.write(f"\n")
        f.write(f"# Number of classes\n")
        f.write(f"nc: {nc}\n")
        f.write(f"\n")
        f.write(f"# Class names\n")
        f.write(f"names:\n")
        for i, name in enumerate(class_names[: nc + 1]):
            f.write(f"  {i}: {name}\n")

    print(f"\nDataset YAML created: {yaml_path}")


def print_dataset_info(output_root: Path):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("NYU Depth V2 -> YOLO Format Dataset")
    print("=" * 60)
    print(f"Output: {output_root}")
    print()

    for subset in ["train", "test"]:
        img_dir = output_root / "images" / subset
        depth_dir = output_root / "depths" / subset
        seg_dir = output_root / "segments" / subset

        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_depth = len(list(depth_dir.glob("*"))) if depth_dir.exists() else 0
        n_seg = len(list(seg_dir.glob("*.txt"))) if seg_dir.exists() else 0

        print(
            f"  {subset.upper():>5}: {n_img:>4} images | {n_depth:>4} depths | {n_seg:>4} segments"
        )

    print("=" * 60)

    yaml_path = output_root / "nyu_depth_seg.yaml"
    print(f"\nTraining commands:")
    print(f"  # Segmentation only (yolo26_train.py):")
    print(f"  python yolo26_train.py --model yolo26-seg.yaml --data {yaml_path}")
    print(f"\n  # Depth + Segmentation (yolo26_train_depth.py):")
    print(f"  python yolo26_train_depth.py \\")
    print(
        f"      --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \\"
    )
    print(f"      --data {yaml_path} \\")
    print(f"      --pretrained yolo26s-seg.pt \\")
    print(f"      --epochs 100 --batch 16 --device 0")


def verify_dataset(output_root: str) -> dict:
    """Verify converted dataset integrity."""
    output_root = Path(output_root)
    stats = {"train": {}, "test": {}}

    for subset in ["train", "test"]:
        img_dir = output_root / "images" / subset
        depth_dir = output_root / "depths" / subset
        seg_dir = output_root / "segments" / subset

        img_files = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []
        depth_files = sorted(depth_dir.glob("*.png")) if depth_dir.exists() else []
        seg_files = sorted(seg_dir.glob("*.txt")) if seg_dir.exists() else []

        missing_depth = []
        empty_seg = []

        for img in img_files:
            depth_file = depth_dir / f"{img.stem}.png"
            if not depth_file.exists():
                missing_depth.append(img.stem)

        for seg in seg_files:
            with open(seg) as f:
                if not f.read().strip():
                    empty_seg.append(seg.stem)

        stats[subset] = {
            "images": len(img_files),
            "depths": len(depth_files),
            "segments": len(seg_files),
            "missing_depths": len(missing_depth),
            "empty_segments": len(empty_seg),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert NYU Depth V2 .mat file to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default 13 classes
  python nyu_mat_converter.py \\
      --mat-path "D:\\ms download\\nyu_depth_v2_labeled.mat" \\
      --output ./nyu_yolo

  # Convert with 40 original classes
  python nyu_mat_converter.py \\
      --mat-path "D:\\ms download\\nyu_depth_v2_labeled.mat" \\
      --output ./nyu_yolo \\
      --nc 40

  # Verify converted dataset
  python nyu_mat_converter.py --verify --output ./nyu_yolo
        """,
    )

    parser.add_argument(
        "--mat-path",
        type=str,
        default=None,
        help="Path to nyu_depth_v2_labeled.mat file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./nyu_yolo",
        help="Output directory for YOLO format dataset",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=13,
        choices=[13, 40],
        help="Number of classes (13=NYU13 simplified, 40=NYU40 original)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted dataset",
    )

    args = parser.parse_args()

    if args.verify:
        stats = verify_dataset(args.output)
        print("\nDataset Verification:")
        for subset, s in stats.items():
            print(f"  {subset.upper()}:")
            print(f"    Images:   {s['images']}")
            print(f"    Depths:   {s['depths']}")
            print(f"    Segments: {s['segments']}")
            if s["missing_depths"]:
                print(f"    WARNING: {s['missing_depths']} missing depth files")
            if s["empty_segments"]:
                print(f"    WARNING: {s['empty_segments']} empty segment files")
        return

    if not args.mat_path:
        parser.error("--mat-path is required (unless using --verify)")

    convert_nyu_mat_to_yolo(
        mat_path=args.mat_path,
        output_root=args.output,
        nc=args.nc,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
