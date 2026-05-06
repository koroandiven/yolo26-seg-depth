#!/usr/bin/env python3
"""
YOLO26 Depth + Segmentation Comprehensive Evaluation Script
同时评估分割头和深度头的性能
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local ultralytics is used instead of site-packages
local_ultralytics = str(Path(__file__).parent / "ultralytics")
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class DepthMetrics:
    """Depth estimation metrics calculator."""

    def __init__(self, max_depth: float = 100.0):
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        self.total_pixels = 0
        self.rmse_sum = 0.0
        self.mae_sum = 0.0
        self.rel_sum = 0.0
        self.log10_sum = 0.0
        self.delta1 = 0.0
        self.delta2 = 0.0
        self.delta3 = 0.0
        self.sq_rel_sum = 0.0
        self.count = 0

    def update(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None):
        """Update metrics with a batch of predictions and targets.

        Args:
            pred: Predicted depth map (H, W) or (B, H, W)
            target: Ground truth depth map (H, W) or (B, H, W)
            mask: Valid pixel mask (H, W) or (B, H, W), optional
        """
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]
            if mask is not None:
                mask = mask[np.newaxis, ...]

        # Clamp predictions and targets
        pred = np.clip(pred, 0.0, self.max_depth)
        target = np.clip(target, 0.0, self.max_depth)

        for i in range(pred.shape[0]):
            p = pred[i]
            t = target[i]

            if mask is not None:
                m = mask[i]
            else:
                # Use valid depth pixels (depth > 0)
                m = t > 0

            if m.sum() == 0:
                continue

            p = p[m]
            t = t[m]

            # Threshold
            p = np.clip(p, 0.01, self.max_depth)
            t = np.clip(t, 0.01, self.max_depth)

            diff = np.abs(p - t)
            ratio = np.maximum(p / t, t / p)

            self.rmse_sum += np.mean(diff ** 2)
            self.mae_sum += np.mean(diff)
            self.rel_sum += np.mean(diff / t)
            self.log10_sum += np.mean(np.abs(np.log10(p) - np.log10(t)))
            self.sq_rel_sum += np.mean(diff ** 2 / t)
            self.delta1 += np.mean(ratio < 1.25)
            self.delta2 += np.mean(ratio < 1.25 ** 2)
            self.delta3 += np.mean(ratio < 1.25 ** 3)
            self.count += 1

    def compute(self) -> dict:
        """Compute and return metrics."""
        if self.count == 0:
            return {}
        return {
            "rmse": np.sqrt(self.rmse_sum / self.count),
            "mae": self.mae_sum / self.count,
            "rel": self.rel_sum / self.count,
            "log10": self.log10_sum / self.count,
            "sq_rel": self.sq_rel_sum / self.count,
            "delta1": self.delta1 / self.count,
            "delta2": self.delta2 / self.count,
            "delta3": self.delta3 / self.count,
        }


@smart_inference_mode()
def evaluate_depth_segmentation(
    model_path: str,
    data: str = "nyu_yolo/nyu_depth_seg.yaml",
    imgsz: int = 640,
    batch: int = 4,
    device: str = "0",
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
    save: bool = True,
    plots: bool = True,
    half: bool = False,
):
    """Comprehensive evaluation of YOLO26 depth+segmentation model.

    Evaluates both segmentation performance (mAP) and depth estimation performance (RMSE, MAE, etc.)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = select_device(device)
    LOGGER.info(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    model.model = model.model.to(device)
    model.model.eval()

    # Load dataset
    from ultralytics.cfg import get_cfg, get_save_dir
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.ops import Profile

    args = get_cfg(overrides={"model": str(model_path), "data": data, "imgsz": imgsz, "batch": batch,
                               "device": device, "conf": conf, "iou": iou, "max_det": max_det,
                               "half": half, "plots": plots, "save": save, "task": "segment"})
    data_dict = check_det_dataset(args.data)
    args.split = "val" if "val" in data_dict else "test"

    # Build dataloader using DepthSegmentDataset
    from ultralytics.data.depth_dataset import DepthSegmentDataset
    from ultralytics.utils.torch_utils import unwrap_model

    gs = max(int(unwrap_model(model.model).stride.max()), 32)
    val_path = data_dict.get(args.split)

    dataset = DepthSegmentDataset(
        img_path=val_path,
        imgsz=imgsz,
        batch_size=batch,
        augment=False,
        hyp=args,
        rect=args.rect,
        cache=args.cache or None,
        single_cls=args.single_cls or False,
        stride=gs,
        pad=0.5,
        prefix="val: ",
        task="segment",
        data=data_dict,
        fraction=1.0,
    )

    dataloader = build_dataloader(dataset, batch, args.workers if device.type != "cpu" else 0, 
                                   shuffle=False, rank=-1)

    # Initialize metrics
    depth_metrics = DepthMetrics(max_depth=100.0)
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils.metrics import SegmentMetrics
    from copy import copy

    seg_validator = SegmentationValidator(dataloader, save_dir=Path("runs/segment/eval_depth"), args=copy(args))
    seg_validator.device = device
    seg_validator.data = data_dict
    seg_validator.args.split = args.split
    seg_validator.init_metrics(unwrap_model(model.model))

    LOGGER.info(f"Starting evaluation on {len(dataset)} images...")

    dt = [Profile(device=device) for _ in range(4)]
    bar = tqdm(dataloader, desc="Evaluating", total=len(dataloader))

    all_depth_preds = []
    all_depth_targets = []

    for batch_i, batch in enumerate(bar):
        # Preprocess
        with dt[0]:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            batch["img"] = (batch["img"].half() if half else batch["img"].float()) / 255

        # Inference
        with dt[1]:
            # Direct forward to get depth output
            raw_preds = model.model(batch["img"], augment=False)

            # For DepthSegment26, eval mode returns (*seg_outputs, depth)
            if isinstance(raw_preds, tuple) and len(raw_preds) > 1:
                seg_preds = raw_preds[:-1]  # All but last are seg outputs
                depth_pred = raw_preds[-1]  # Last is depth
            elif isinstance(raw_preds, dict) and "depth" in raw_preds:
                seg_preds = raw_preds
                depth_pred = raw_preds.pop("depth")
            else:
                seg_preds = raw_preds
                depth_pred = None

        # Postprocess segmentation predictions
        with dt[3]:
            if isinstance(seg_preds, tuple):
                seg_processed = seg_validator.postprocess(seg_preds)
            else:
                # Handle dict or single tensor
                seg_processed = seg_validator.postprocess((seg_preds,))

            seg_validator.update_metrics(seg_processed, batch)

        # Process depth predictions
        if depth_pred is not None and "depth" in batch:
            depth_target = batch["depth"]

            # Move to CPU for numpy processing
            depth_pred = depth_pred.detach().cpu().numpy().squeeze(1)  # (B, H, W)
            depth_target = depth_target.detach().cpu().numpy()  # (B, H, W)

            # Upsample depth to target size if needed
            if depth_pred.shape[1:] != depth_target.shape[1:]:
                depth_pred = np.stack([
                    ops.cv2.resize(d, (depth_target.shape[2], depth_target.shape[1]), interpolation=cv2.INTER_LINEAR)
                    for d in depth_pred
                ])

            depth_metrics.update(depth_pred, depth_target)

            all_depth_preds.append(depth_pred)
            all_depth_targets.append(depth_target)

    # Gather segmentation stats
    seg_validator.gather_stats()
    seg_stats = seg_validator.get_stats()
    seg_validator.finalize_metrics()

    # Compute depth stats
    depth_stats = depth_metrics.compute()

    # Print results
    print("\n" + "=" * 70)
    print("YOLO26 Depth + Segmentation Comprehensive Evaluation Results")
    print("=" * 70)

    print("\n[Segmentation (Mask) Metrics]")
    if hasattr(seg_validator.metrics, "seg"):
        seg = seg_validator.metrics.seg
        print(f"  Precision (P):     {seg.mp:.4f}" if hasattr(seg, "mp") else "  Precision: N/A")
        print(f"  Recall (R):        {seg.mr:.4f}" if hasattr(seg, "mr") else "  Recall: N/A")
        print(f"  mAP@50:            {seg.map50:.4f}" if hasattr(seg, "map50") else "  mAP@50: N/A")
        print(f"  mAP@50-95:         {seg.map:.4f}" if hasattr(seg, "map") else "  mAP@50-95: N/A")

    print("\n[Depth Estimation Metrics]")
    if depth_stats:
        print(f"  RMSE:              {depth_stats['rmse']:.4f} m")
        print(f"  MAE:               {depth_stats['mae']:.4f} m")
        print(f"  REL:               {depth_stats['rel']:.4f}")
        print(f"  Sq.REL:            {depth_stats['sq_rel']:.4f}")
        print(f"  Log10:             {depth_stats['log10']:.4f}")
        print(f"  delta < 1.25:      {depth_stats['delta1']:.4f}")
        print(f"  delta < 1.25^2:    {depth_stats['delta2']:.4f}")
        print(f"  delta < 1.25^3:    {depth_stats['delta3']:.4f}")
    else:
        print("  No depth metrics computed (depth prediction not available)")

    print("\n[Inference Speed]")
    speed = dict(zip(["preprocess", "inference", "loss", "postprocess"],
                     (x.t / len(dataset) * 1e3 for x in dt)))
    for k, v in speed.items():
        print(f"  {k}: {v:.3f} ms")

    print("=" * 70)

    return {
        "segmentation": seg_stats,
        "depth": depth_stats,
        "speed": speed,
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive YOLO26 Depth+Segmentation Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/segment/runs/train_depth/yolo26-seg-depth-exp/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument("--data", type=str, default="nyu_yolo/nyu_depth_seg.yaml", help="Dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    args = parser.parse_args()

    evaluate_depth_segmentation(
        model_path=args.model,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
