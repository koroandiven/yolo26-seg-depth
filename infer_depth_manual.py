#!/usr/bin/env python3
"""
YOLO26 Depth + Segmentation Inference (Manual Model Loading)
Bypass YOLO() to correctly load DepthSegment26 from local ultralytics
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import os
import sys
import torch
from pathlib import Path

# MUST be first: set PYTHONPATH before any import
local_ultralytics = str(Path(__file__).parent / "ultralytics")
os.environ["PYTHONPATH"] = local_ultralytics + ":" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, local_ultralytics)

# Now import local ultralytics
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils import yaml_model_load


def load_model_weights(weights_path: str, device: str):
    """Load trained model manually from .pt checkpoint."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = ckpt["model"]  # This is already a SegmentationModel instance with DepthSegment26 head
    model = model.to(device).eval()
    return model, ckpt


def preprocess(img_path: str, imgsz: int = 640):
    """Preprocess image for inference."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    orig_h, orig_w = img.shape[:2]
    scale = min(imgsz / orig_h, imgsz / orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    pad_h = imgsz - new_h
    pad_w = imgsz - new_w
    top, left = pad_h // 2, pad_w // 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Normalize and convert to tensor
    im = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    im = torch.from_numpy(im).unsqueeze(0)

    return img, im, (scale, top, left, orig_h, orig_w)


def postprocess_masks(pred, proto, img_shape, conf_thresh=0.25, iou_thresh=0.45):
    """Simple NMS and mask decoding (simplified for single-image inference)."""
    # This is a simplified postprocess; for full accuracy use Ultralytics predictor
    # For visualization we rely on model.predict() with patched model
    return None


def visualize_depth(depth: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """Visualize depth map."""
    depth = cv2.resize(depth, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
    d_min, d_max = depth.min(), depth.max()
    depth_vis = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.putText(depth_color, f"Depth: {d_min:.2f}m - {d_max:.2f}m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return depth_color


def run_inference(model, img_path, device, conf=0.25, imgsz=640):
    """Run inference and return segmentation + depth results."""
    import torchvision  # local import to avoid early ultralytics loading

    orig_img, im_tensor, (scale, top, left, orig_h, orig_w) = preprocess(img_path, imgsz)
    im_tensor = im_tensor.to(device)

    with torch.no_grad():
        outputs = model(im_tensor)

    # DepthSegment26 eval returns (pred_tuple, proto, depth)
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        pred = outputs[0]
        proto = outputs[1]
        depth = outputs[2].squeeze().cpu().numpy()
    else:
        raise RuntimeError(f"Unexpected output format: {type(outputs)}")

    # For segmentation visualization, use Ultralytics predictor with the loaded model
    # We temporarily save model state and load via YOLO with local path enforced
    return orig_img, pred, proto, depth, (scale, top, left, orig_h, orig_w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./inference_results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    device = select_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, ckpt = load_model_weights(args.model, device)

    # Check model architecture
    if hasattr(model, "names"):
        print(f"Model classes ({len(model.names)}): {dict(list(model.names.items())[:10])}")
    else:
        print("Model has no names attribute")

    # Check if depth module exists
    has_depth = any("depth" in name for name, _ in model.named_modules())
    print(f"Has depth module: {has_depth}")

    # ------------------------------------------------------------------
    # Method 1: Use Ultralytics predictor for segmentation (with local path)
    # We need to load through YOLO() but ensure local ultralytics is used
    # ------------------------------------------------------------------
    print(f"\nRunning segmentation inference on: {args.source}")

    # Re-import YOLO after ensuring local path is first
    from ultralytics import YOLO

    # Save a temporary checkpoint that can be loaded by YOLO
    # Actually, just use model.predict() directly if available
    if hasattr(model, "predict"):
        results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz, verbose=False)
    else:
        # Fallback: create YOLO wrapper
        yolo_model = YOLO(args.model)
        results = yolo_model.predict(args.source, conf=args.conf, imgsz=args.imgsz, verbose=False)

    seg_result = results[0]
    seg_img = seg_result.plot()
    seg_path = save_dir / "segmentation.jpg"
    cv2.imwrite(str(seg_path), cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    print(f"Segmentation saved: {seg_path} ({len(seg_result.boxes)} detections)")

    # ------------------------------------------------------------------
    # Method 2: Manual forward for depth
    # ------------------------------------------------------------------
    print("\nExtracting depth map...")
    orig_img, pred, proto, depth, meta = run_inference(model, args.source, device, args.conf, args.imgsz)
    scale, top, left, orig_h, orig_w = meta

    depth_color = visualize_depth(depth, (orig_h, orig_w))
    combined = np.hstack([orig_img, depth_color])

    depth_path = save_dir / "depth.jpg"
    combined_path = save_dir / "combined.jpg"
    cv2.imwrite(str(depth_path), depth_color)
    cv2.imwrite(str(combined_path), combined)

    print(f"Depth saved: {depth_path}")
    print(f"Combined saved: {combined_path}")
    print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()
