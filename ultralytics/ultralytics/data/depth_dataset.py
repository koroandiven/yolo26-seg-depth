# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Depth + Segmentation Dataset for YOLO26 Multi-task Learning

支持 NYU Depth V2 格式数据集:
- images/*.png: RGB 图像
- depths/*.png: 深度图 (16-bit)
- segments/*.txt: YOLO 格式分割标注
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import random
import torch
from pathlib import Path
from typing import Any

from ultralytics.utils import LOCAL_RANK, LOGGER, TQDM
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes

from .augment import Compose, Format, LetterBox, RandomHSV
from .base import BaseDataset
from .utils import get_hash, img2label_paths, load_dataset_cache_file, save_dataset_cache_file


DATASET_CACHE_VERSION = "1.0.3"


class DepthRandomFlip:
    """Horizontal flip that also mirrors the dense depth target.

    The stock ``RandomFlip`` augmentation in Ultralytics only flips the image
    and the instance annotations, leaving ``label["depth"]`` (a dense per-pixel
    target) untouched. That mismatch would teach the depth head the wrong
    correspondence, so we wrap the flip and apply it to depth too.

    Only horizontal flip is supported on purpose: vertical flip would invert
    floor/ceiling geometry which depth cannot reconcile.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return labels

        img = labels["img"]
        instances = labels.pop("instances", None)
        h, w = img.shape[:2]
        if instances is not None:
            instances.convert_bbox(format="xywh")
            w_for_flip = 1 if instances.normalized else w
            instances.fliplr(w_for_flip)
            labels["instances"] = instances

        labels["img"] = np.ascontiguousarray(np.fliplr(img))

        depth = labels.get("depth")
        if depth is not None:
            labels["depth"] = np.ascontiguousarray(np.fliplr(depth))

        return labels


class DepthSafeColorJitter:
    """Color jitter that only affects RGB, safe for depth supervision.

    Adjusts brightness and contrast in-place. Depth, masks, and bboxes
    are left untouched.
    """

    def __init__(self, brightness: float = 0.3, contrast: float = 0.3):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels["img"]
        if img.dtype != np.uint8 or img.shape[-1] != 3:
            return labels

        # Brightness
        if self.brightness > 0:
            beta = random.uniform(-self.brightness, self.brightness)
            img = np.clip(img.astype(np.float32) * (1.0 + beta), 0, 255).astype(np.uint8)

        # Contrast
        if self.contrast > 0:
            alpha = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        labels["img"] = img
        return labels


class DepthSafeGaussianBlur:
    """Random Gaussian blur, safe for depth (only blurs RGB)."""

    def __init__(self, p: float = 0.3, kernel_range: tuple = (3, 7)):
        self.p = p
        self.kernel_range = kernel_range

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return labels
        img = labels["img"]
        if img.dtype != np.uint8:
            return labels
        k = random.choice(range(self.kernel_range[0], self.kernel_range[1] + 1, 2))
        if k < 3:
            return labels
        labels["img"] = cv2.GaussianBlur(img, (k, k), 0)
        return labels


class DepthSafeNoise:
    """Add slight Gaussian noise to RGB, safe for depth supervision."""

    def __init__(self, sigma: float = 5.0 / 255.0, p: float = 0.2):
        self.sigma = sigma
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return labels
        img = labels["img"]
        if img.dtype != np.uint8:
            return labels
        noise = np.random.normal(0, self.sigma * 255, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        labels["img"] = img
        return labels


class DepthSafeCutOut:
    """CutOut augmentation that also masks depth (set to 0).

    This makes the cut region invalid for depth loss via the existing
    valid_mask (depth > 0) mechanism.
    """

    def __init__(self, p: float = 0.2, max_holes: int = 3, max_size: float = 0.1):
        self.p = p
        self.max_holes = max_holes
        self.max_size = max_size

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        depth = labels.get("depth")
        n_holes = random.randint(1, self.max_holes)
        for _ in range(n_holes):
            cw = int(w * random.uniform(0.02, self.max_size))
            ch = int(h * random.uniform(0.02, self.max_size))
            x1 = random.randint(0, max(0, w - cw))
            y1 = random.randint(0, max(0, h - ch))
            img[y1 : y1 + ch, x1 : x1 + cw] = 114  # standard padding gray
            if depth is not None:
                depth[y1 : y1 + ch, x1 : x1 + cw] = 0.0
        labels["img"] = img
        if depth is not None:
            labels["depth"] = depth
        return labels



class DepthSegmentDataset(BaseDataset):
    """Dataset class for Depth + Segmentation multi-task learning.

    Supports NYU Depth V2 format with:
    - RGB images
    - Depth maps (16-bit PNG, values in millimeters)
    - Instance segmentation labels (YOLO format)

    Directory structure:
        dataset/
        ├── images/
        │   ├── train/
        │   │   ├── 0001.jpg
        │   │   └── 0002.jpg
        │   └── val/
        └── depths/
            ├── train/
            │   ├── 0001.png
            │   └── 0002.png
        └── segments/
            ├── train/
            │   ├── 0001.txt
            │   └── 0002.txt
    """

    def __init__(
        self,
        *args,
        data: dict | None = None,
        task: str = "segment",
        depth_max: float = 100.0,
        depth_scale: float = 100.0,
        depth_norm_max: float | None = None,
        source_tag: str = "",
        **kwargs,
    ):
        """Initialize DepthSegmentDataset.

        Args:
            data: Dataset configuration dictionary with paths
            task: Task type, "segment" for depth+segment
            depth_max: Maximum raw depth value in meters (for clipping loaded depth)
            depth_scale: Scale factor to normalize depth targets to [0, depth_scale]
            depth_norm_max: Per-source max depth for normalizing to [0, depth_scale]
                (e.g. NYU 10.0, KITTI 80.0). When set, depth is normalized as
                depth = raw_depth / depth_norm_max * depth_scale, eliminating
                absolute scale differences between sources.
            source_tag: Identifier string for multi-source training (e.g. "nyu", "kitti")
            *args: Additional positional arguments for parent class
            **kwargs: Additional keyword arguments for parent class
        """
        self.use_segments = True
        self.use_keypoints = False
        self.use_obb = False
        self.depth_max = depth_max
        self.depth_scale = depth_scale
        self.depth_norm_max = depth_norm_max
        self.data = data
        self.source_tag = source_tag
        super().__init__(*args, channels=3, **kwargs)

    def get_img_files(self, img_path: str) -> list:
        """Get image file paths from directory."""
        img_path = Path(img_path)
        if not img_path.exists():
            LOGGER.warning(f"Image path does not exist: {img_path}")
            return []
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        files = []
        for ext in extensions:
            files.extend(sorted(img_path.glob(f"*{ext}")))
        return [str(f) for f in files]

    def get_depth_files(self, img_files: list) -> list:
        """Get corresponding depth file paths for image files.

        Assumes depth files are in a 'depths' directory at the same level as 'images'.
        Example: images/train/0001.jpg -> depths/train/0001.png
        """
        depth_files = []
        for img_file in img_files:
            img_path = Path(img_file)
            depth_path = img_path.parent.parent.parent / "depths" / img_path.parent.name / img_path.stem
            depth_path = depth_path.with_suffix(".png")
            if depth_path.exists():
                depth_files.append(str(depth_path))
            else:
                LOGGER.warning(f"Depth file not found: {depth_path}")
                depth_files.append("")
        return depth_files

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """Cache dataset labels, check images and read shapes."""
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."

        results = []
        for im_file, depth_file in zip(self.im_files, self.depth_files):
            lb = self.load_segment_label(im_file)
            depth_exists = Path(depth_file).exists() if depth_file else False
            results.append((im_file, lb, depth_file, depth_exists))

        total = len(results)
        pbar = TQDM(results, desc=desc, total=total)
        for im_file, lb, depth_file, depth_exists, *rest in pbar:
            if depth_file and not depth_exists:
                nm += 1
                continue
            if im_file:
                x["labels"].append(
                    {
                        "im_file": im_file,
                        "depth_file": depth_file if depth_exists else "",
                        "shape": self.get_im_shape(im_file),
                        "cls": lb[:, 0:1] if len(lb) > 0 else np.zeros((0, 1)),
                        "bboxes": lb[:, 1:5] if len(lb) > 0 else np.zeros((0, 4)),
                        "segments": self.load_segments(im_file),
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )
                nf += 1
            else:
                ne += 1
            pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds"
        pbar.close()

        x["hash"] = get_hash(self.label_files + self.im_files + self.depth_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        if x["labels"]:
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def load_segment_label(self, img_file: str) -> np.ndarray:
        """Load YOLO format segmentation label and convert polygons to bboxes."""
        label_file = Path(img_file).parent.parent.parent / "segments" / Path(img_file).parent.name / Path(img_file).stem
        label_file = label_file.with_suffix(".txt")
        if not label_file.exists():
            return np.zeros((0, 5), dtype=np.float32)
        with open(label_file) as f:
            lines = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if not lines:
            return np.zeros((0, 5), dtype=np.float32)

        bboxes = []
        for parts in lines:
            cls_id = float(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) < 6:  # need at least 3 points (6 coords)
                continue
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            bboxes.append([cls_id, cx, cy, w, h])

        if not bboxes:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(bboxes, dtype=np.float32)

    def load_segments(self, img_file: str) -> list:
        """Load segment polygons from YOLO format label file."""
        label_file = Path(img_file).parent.parent.parent / "segments" / Path(img_file).parent.name / Path(img_file).stem
        label_file = label_file.with_suffix(".txt")
        if not label_file.exists():
            return []
        segments = []
        with open(label_file) as f:
            for line in f.read().strip().splitlines():
                parts = line.split()
                if len(parts) < 6:
                    continue
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 != 0:
                    continue
                segment = np.array(coords, dtype=np.float32).reshape(-1, 2)
                segments.append(segment)
        return segments

    def get_im_shape(self, img_file: str) -> tuple:
        """Get image shape (height, width)."""
        img = cv2.imread(img_file)
        if img is None:
            return (640, 640)
        return img.shape[:2]

    def get_labels(self) -> list[dict]:
        """Return list of label dictionaries for training."""
        # Point label files to segments/ directory (polygon labels) instead of labels/ (bbox labels)
        self.label_files = [
            x.replace(f"{os.sep}images{os.sep}", f"{os.sep}segments{os.sep}", 1) for x in self.im_files
        ]
        self.depth_files = self.get_depth_files(self.im_files)

        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files + self.depth_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)

        labels = cache["labels"]
        if not labels:
            raise RuntimeError(f"No valid labels found in {cache_path}")
        [cache.pop(k, None) for k in ("hash", "version", "msgs")]
        self.im_files = [lb["im_file"] for lb in labels]
        self.depth_files = [lb.get("depth_file", "") for lb in labels]
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Build and append transforms to the list."""
        if self.augment:
            # Depth is a dense pixel target. Disable spatial augmentations that do not
            # transform depth identically to the image, but keep safe color jitter
            # and a depth-aware horizontal flip (essential to avoid left/right bias).
            fliplr_p = float(getattr(hyp, "fliplr", 0.5)) if hyp is not None else 0.5
            transforms_list = [
                LetterBox(new_shape=(self.imgsz, self.imgsz)),
                DepthRandomFlip(p=fliplr_p),
                RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                # New safe augmentations for depth (only affect RGB, or sync mask depth)
                DepthSafeColorJitter(brightness=0.4, contrast=0.4),
                DepthSafeGaussianBlur(p=0.3, kernel_range=(3, 7)),
                DepthSafeCutOut(p=0.2, max_holes=3, max_size=0.1),
                DepthSafeNoise(sigma=5.0 / 255.0, p=0.2),
            ]
            transforms = Compose(transforms_list)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=True,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,
            )
        )
        return transforms

    def update_labels_info(self, label: dict) -> dict:
        """Update label format for segmentation task."""
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        segment_resamples = 1000
        if len(segments) > 0:
            max_len = max(len(s[1]) if isinstance(s, list) else len(s) for s in segments)
            segment_resamples = max(segment_resamples, max_len + 1)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)

        label["instances"] = Instances(bboxes, segments, bbox_format=bbox_format, normalized=normalized)
        return label

    def load_depth(self, depth_file: str) -> np.ndarray:
        """Load depth map from PNG file.

        Args:
            depth_file: Path to 16-bit PNG depth file

        Returns:
            Normalized depth map, shape (H, W), range [0, depth_scale]
            (0 indicates missing/invalid depth)
        """
        if not depth_file or not Path(depth_file).exists():
            return np.zeros((self.imgsz, self.imgsz), dtype=np.float32)

        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth is None:
            return np.zeros((self.imgsz, self.imgsz), dtype=np.float32)

        # Convert from mm to meters, clip outliers
        depth = depth.astype(np.float32) / 1000.0
        depth = np.clip(depth, 0, self.depth_max)
        # Per-source depth normalization: eliminate absolute scale differences
        # between sources (e.g. NYU 0-10m vs KITTI 0-80m) by normalizing to
        # [0, depth_scale] using each source's depth_norm_max.
        if self.depth_norm_max is not None and self.depth_norm_max > 0:
            depth = depth / self.depth_norm_max * self.depth_scale
        return depth

    @staticmethod
    def _letterbox_depth(depth: np.ndarray, labels: dict) -> np.ndarray:
        """Apply the same LetterBox geometry as the image transform to a depth map.

        Depth values are dense supervision in meters. Padding is set to 0 so the
        existing valid mask (depth > 0) excludes padded regions from depth loss.
        """
        img = labels["img"]
        shape = depth.shape[:2]
        new_shape = labels.get("rect_shape", labels.get("depth_new_shape", (img.shape[0], img.shape[1])))
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if labels.get("depth_scaleup") is False:
            r = min(r, 1.0)

        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            depth = cv2.resize(depth, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        return cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0).astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        """Get transformed item from the dataset."""
        label = self.get_image_and_label(index)
        depth_file = self.labels[index].get("depth_file", "")
        h0, w0 = label["ori_shape"]

        if depth_file and Path(depth_file).exists():
            depth = self.load_depth(depth_file)
        else:
            depth = np.zeros((h0, w0), dtype=np.float32)

        # Resize depth to match resized image shape from load_image. LetterBox is
        # applied below with the same target shape as the image transform.
        if depth.shape[:2] != label["resized_shape"]:
            depth = cv2.resize(depth, (label["resized_shape"][1], label["resized_shape"][0]), interpolation=cv2.INTER_LINEAR)

        # Store a depth map already aligned to the post-LetterBox image. Standard
        # Ultralytics transforms do not know how to transform dense depth targets.
        label["depth_new_shape"] = (self.imgsz, self.imgsz)
        label["depth_scaleup"] = self.augment
        label["depth"] = self._letterbox_depth(depth, label)

        label = self.transforms(label)
        label.pop("depth_scaleup", None)
        label.pop("depth_new_shape", None)

        # Tag sample with source identifier for multi-source training
        if self.source_tag:
            label["source_tag"] = self.source_tag

        # Remove any None-valued keys to prevent collate_fn issues
        for k in list(label.keys()):
            if label[k] is None:
                del label[k]

        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate data samples into batches."""
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "sem_masks"}:
                value = torch.stack(value, 0)
            elif k == "depth":
                # Defensive: ensure all depths have the same HxW shape
                shapes = {v.shape for v in value}
                if len(shapes) > 1:
                    target_h = max(v.shape[0] for v in value)
                    target_w = max(v.shape[1] for v in value)
                    value = tuple(
                        cv2.copyMakeBorder(v, 0, target_h - v.shape[0], 0, target_w - v.shape[1],
                                          cv2.BORDER_CONSTANT, value=0) if v.shape != (target_h, target_w) else v
                        for v in value
                    )
                value = torch.from_numpy(np.stack(value)).float()
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            elif k == "source_tag":
                # Keep source tags as a list for per-sample reference in loss
                value = list(value)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                if any(v is None for v in value):
                    raise ValueError(f"Key '{k}' contains None values: {value}")
                value = torch.cat(value, 0)
            new_batch[k] = value

        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


def nyu_to_yolo_format(nyu_root: str, output_root: str, subset: str = "train"):
    """Convert NYU Depth V2 dataset to YOLO format.

    Args:
        nyu_root: Path to NYU Depth V2 dataset root (contains 'labels' folder with .mat files)
        output_root: Output directory for YOLO format dataset
        subset: 'train' or 'val'
    """
    from scipy.io import loadmat

    nyu_root = Path(nyu_root)
    output_root = Path(output_root)

    images_dir = output_root / "images" / subset
    depths_dir = output_root / "depths" / subset
    segments_dir = output_root / "segments" / subset

    for d in [images_dir, depths_dir, segments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    labels_dir = nyu_root / "labels"
    raw_images_dir = nyu_root / "images" / subset

    mat_files = sorted(labels_dir.glob(f"*{subset}*.mat"))

    for mat_file in TQDM(mat_files, desc=f"Converting {subset}"):
        scene_id = mat_file.stem.replace("_labels", "")
        img_file = raw_images_dir / f"{scene_id}.jpg"

        if not img_file.exists():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]

        mat = loadmat(str(mat_file))
        instances = mat["labels"] if "labels" in mat else None

        if instances is None:
            continue

        img_out = images_dir / f"{scene_id}.jpg"
        depth_out = depths_dir / f"{scene_id}.png"
        seg_out = segments_dir / f"{scene_id}.txt"

        cv2.imwrite(str(img_out), img)

        if "depths" in mat:
            depth_data = mat["depths"].astype(np.uint16)
            cv2.imwrite(str(depth_out), depth_data)

        instance_ids = np.unique(instances)
        instance_ids = instance_ids[instance_ids > 0]

        seg_lines = []
        for inst_id in instance_ids:
            mask = (instances == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue
                contour = contour.squeeze()
                if contour.ndim == 1:
                    continue
                coords = contour / np.array([w, h], dtype=np.float32)
                coords = coords.flatten()
                line = f"0 " + " ".join([f"{c:.6f}" for c in coords])
                seg_lines.append(line)

        with open(seg_out, "w") as f:
            f.write("\n".join(seg_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert NYU Depth V2 to YOLO format")
    parser.add_argument("--nyu-root", type=str, required=True, help="Path to NYU Depth V2 dataset")
    parser.add_argument("--output-root", type=str, required=True, help="Output directory")
    parser.add_argument("--subset", type=str, default="train", choices=["train", "val"], help="Subset to convert")
    args = parser.parse_args()

    nyu_to_yolo_format(args.nyu_root, args.output_root, args.subset)
