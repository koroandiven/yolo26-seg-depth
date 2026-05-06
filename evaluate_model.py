#!/usr/bin/env python3
"""
YOLO26 Depth+Segmentation Model Evaluation Script
评估训练好的 best.pt 模型在验证集上的表现
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local ultralytics is used instead of site-packages
local_ultralytics = str(Path(__file__).parent / "ultralytics")
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def evaluate_depth_metrics(model, data_path: str, imgsz: int = 640, batch: int = 4, device: str = "0"):
    """Evaluate depth prediction quality on the validation set.

    Returns dict with RMSE, MAE, REL, delta thresholds, etc.
    """
    import yaml
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from ultralytics.data.depth_dataset import DepthSegmentDataset
    from ultralytics.utils.torch_utils import select_device

    device = select_device(device, batch)

    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    val_img_path = Path(data_cfg["path"]) / data_cfg.get("val", "images/val")
    val_depth_path = Path(data_cfg["path"]) / data_cfg.get("depth", "depths/val")

    dataset = DepthSegmentDataset(
        img_path=str(val_img_path),
        imgsz=imgsz,
        batch_size=batch,
        augment=False,
        rect=True,
        cache=False,
        single_cls=False,
        stride=max(int(model.model.stride.max()), 32),
        pad=0.5,
        prefix="val: ",
        task="segment",
        data=data_cfg,
        fraction=1.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=0,
        collate_fn=DepthSegmentDataset.collate_fn,
        pin_memory=True,
    )

    model = model.to(device)
    model.eval()

    total_pixels = 0
    sq_err = 0.0
    abs_err = 0.0
    rel_err = 0.0
    log_sq_err = 0.0
    delta_1 = delta_2 = delta_3 = 0
    max_depth = 100.0

    with torch.no_grad():
        for batch_data in loader:
            imgs = batch_data["img"].to(device, dtype=torch.float32) / 255.0
            gt_depth = batch_data.get("depth")
            if gt_depth is None:
                continue
            gt_depth = gt_depth.to(device)

            out = model.predict(imgs, verbose=False)
            # Depth is the last element of the model output tuple
            pred_depth = out[-1] if isinstance(out, (tuple, list)) else None
            if pred_depth is None:
                continue

            if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
                pred_depth = F.interpolate(
                    pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False
                )

            mask = (gt_depth > 0) & (gt_depth <= max_depth)
            if mask.sum() == 0:
                continue

            pred_d = pred_depth.squeeze(1)[mask]
            gt_d = gt_depth[mask]

            diff = pred_d - gt_d
            sq_err += (diff ** 2).sum().item()
            abs_err += diff.abs().sum().item()
            rel_err += (diff.abs() / (gt_d + 1e-6)).sum().item()
            log_sq_err += ((torch.log(pred_d + 1e-6) - torch.log(gt_d + 1e-6)) ** 2).sum().item()

            ratio = torch.max(pred_d / (gt_d + 1e-6), gt_d / (pred_d + 1e-6))
            delta_1 += (ratio < 1.25).sum().item()
            delta_2 += (ratio < 1.25 ** 2).sum().item()
            delta_3 += (ratio < 1.25 ** 3).sum().item()
            total_pixels += mask.sum().item()

    if total_pixels == 0:
        return {}

    rmse = (sq_err / total_pixels) ** 0.5
    mae = abs_err / total_pixels
    rel = rel_err / total_pixels
    log_rmse = (log_sq_err / total_pixels) ** 0.5
    d1 = delta_1 / total_pixels
    d2 = delta_2 / total_pixels
    d3 = delta_3 / total_pixels

    return {
        "rmse": rmse,
        "mae": mae,
        "rel": rel,
        "log_rmse": log_rmse,
        "delta_1.25": d1,
        "delta_1.25^2": d2,
        "delta_1.25^3": d3,
    }


def evaluate_model(
    model_path: str,
    data: str = "nyu_yolo/nyu_depth_seg.yaml",
    imgsz: int = 640,
    batch: int = 4,
    device: str = "0",
    save: bool = True,
    save_json: bool = False,
    plots: bool = True,
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
    half: bool = False,
    dnn: bool = False,
):
    """Evaluate a trained YOLO26 depth+segmentation model.

    Args:
        model_path: Path to trained model weights (.pt file)
        data: Path to dataset YAML configuration
        imgsz: Input image size
        batch: Batch size for validation
        device: Device to run validation on ('cpu', '0', '0,1,2,3', etc.)
        save: Whether to save validation results
        save_json: Whether to save results as JSON
        plots: Whether to generate validation plots
        conf: Confidence threshold
        iou: NMS IoU threshold
        max_det: Maximum detections per image
        half: Whether to use FP16 half-precision
        dnn: Whether to use OpenCV DNN backend

    Returns:
        Validation metrics dictionary
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    LOGGER.info(f"Loading model from {model_path}")
    model = YOLO(str(model_path))

    LOGGER.info(f"Starting validation on dataset: {data}")
    LOGGER.info(f"Image size: {imgsz}, Batch size: {batch}, Device: {device}")

    metrics = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        save=save,
        save_json=save_json,
        plots=plots,
        conf=conf,
        iou=iou,
        max_det=max_det,
        half=half,
        dnn=dnn,
        verbose=True,
    )

    LOGGER.info("Evaluating depth predictions...")
    depth_metrics = evaluate_depth_metrics(model, data, imgsz, batch, device)
    metrics.depth_metrics = depth_metrics

    return metrics


def print_metrics(metrics):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("YOLO26 Depth + Segmentation Evaluation Results")
    print("=" * 60)

    # Box metrics
    print("\n[Detection (Box) Metrics]")
    if hasattr(metrics, "box"):
        box = metrics.box
        print(f"  Precision (P):     {box.mp:.4f}" if hasattr(box, "mp") else "  Precision: N/A")
        print(f"  Recall (R):        {box.mr:.4f}" if hasattr(box, "mr") else "  Recall: N/A")
        print(f"  mAP@50:            {box.map50:.4f}" if hasattr(box, "map50") else "  mAP@50: N/A")
        print(f"  mAP@50-95:         {box.map:.4f}" if hasattr(box, "map") else "  mAP@50-95: N/A")

    # Mask metrics
    print("\n[Segmentation (Mask) Metrics]")
    if hasattr(metrics, "seg"):
        seg = metrics.seg
        print(f"  Precision (P):     {seg.mp:.4f}" if hasattr(seg, "mp") else "  Precision: N/A")
        print(f"  Recall (R):        {seg.mr:.4f}" if hasattr(seg, "mr") else "  Recall: N/A")
        print(f"  mAP@50:            {seg.map50:.4f}" if hasattr(seg, "map50") else "  mAP@50: N/A")
        print(f"  mAP@50-95:         {seg.map:.4f}" if hasattr(seg, "map") else "  mAP@50-95: N/A")

    # Per-class results
    if hasattr(metrics, "results_dict"):
        print("\n[Detailed Results]")
        for k, v in metrics.results_dict.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Depth metrics
    if hasattr(metrics, "depth_metrics") and metrics.depth_metrics:
        print("\n[Depth Estimation Metrics]")
        dm = metrics.depth_metrics
        print(f"  RMSE:              {dm.get('rmse', 0):.4f}")
        print(f"  MAE:               {dm.get('mae', 0):.4f}")
        print(f"  REL:               {dm.get('rel', 0):.4f}")
        print(f"  log RMSE:          {dm.get('log_rmse', 0):.4f}")
        print(f"  delta < 1.25:      {dm.get('delta_1.25', 0):.4f}")
        print(f"  delta < 1.25^2:    {dm.get('delta_1.25^2', 0):.4f}")
        print(f"  delta < 1.25^3:    {dm.get('delta_1.25^3', 0):.4f}")

    # Speed
    if hasattr(metrics, "speed"):
        print("\n[Inference Speed]")
        for k, v in metrics.speed.items():
            print(f"  {k}: {v:.3f} ms")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 Depth+Segmentation Model")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26-seg-depth/runs/segment/runs/train_depth/yolo26-seg-depth-exp/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="nyu_yolo/nyu_depth_seg.yaml",
        help="Path to dataset YAML configuration",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (cpu or cuda device)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate validation plots")
    parser.add_argument("--save-json", action="store_true", help="Save results as JSON")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision")
    args = parser.parse_args()

    metrics = evaluate_model(
        model_path=args.model,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=args.plots,
        save_json=args.save_json,
        half=args.half,
    )

    print_metrics(metrics)

    return metrics


if __name__ == "__main__":
    main()
