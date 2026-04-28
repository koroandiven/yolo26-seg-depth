#!/usr/bin/env python3
"""
YOLO26 Depth + Segmentation Inference Script
同时输出分割结果和深度图
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import sys
import torch
from pathlib import Path

# Ensure local ultralytics is used (must be before any ultralytics import)
local_ultralytics = str(Path(__file__).parent / "ultralytics")
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

# Clear cached system ultralytics to force loading local version
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("ultralytics"):
        del sys.modules[mod_name]

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device


def preprocess_image(img_path: str, imgsz: int = 640):
    """Preprocess image to match Ultralytics predictor format."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # LetterBox resize (same as Ultralytics)
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    pad_h = (imgsz - new_h) % 32
    pad_w = (imgsz - new_w) % 32
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR to RGB, HWC to CHW, normalize
    im = padded[:, :, ::-1].transpose(2, 0, 1)  # RGB, CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    im = torch.from_numpy(im).unsqueeze(0)

    return img, im, (scale, top, left)


def visualize_depth(depth: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """Visualize depth map as color image."""
    # Resize to original image size
    depth = cv2.resize(depth, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize to 0-255
    d_min, d_max = depth.min(), depth.max()
    depth_vis = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_vis = (depth_vis * 255).astype(np.uint8)

    # Apply colormap
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Add text with depth range
    text = f"Depth: {d_min:.2f}m - {d_max:.2f}m"
    cv2.putText(depth_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return depth_color


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Depth + Segmentation Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--save-dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="Device: cpu or gpu id")
    args = parser.parse_args()

    device = select_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    model.to(device)
    model.eval()
    print(f"Model classes ({len(model.names)}): {dict(list(model.names.items())[:5])}...")

    # ------------------------------------------------------------------
    # 2. Segmentation inference (using standard predict)
    # ------------------------------------------------------------------
    print(f"\nRunning segmentation inference on: {args.source}")
    seg_results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz, verbose=False, device=device)
    seg_result = seg_results[0]

    # Plot and save segmentation result
    seg_img = seg_result.plot()
    seg_path = save_dir / "segmentation.jpg"
    cv2.imwrite(str(seg_path), cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    print(f"Segmentation saved: {seg_path} ({len(seg_result.boxes)} detections)")

    # ------------------------------------------------------------------
    # 3. Depth inference (manual forward pass)
    # ------------------------------------------------------------------
    print("\nExtracting depth map...")
    orig_img, im_tensor, (scale, pad_top, pad_left) = preprocess_image(args.source, args.imgsz)
    im_tensor = im_tensor.to(device)

    with torch.no_grad():
        outputs = model.model(im_tensor)

    # DepthSegment26 returns (pred, proto, depth) in eval mode
    depth_pred = None
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        depth_pred = outputs[-1]
    elif isinstance(outputs, dict) and "depth" in outputs:
        depth_pred = outputs["depth"]

    if depth_pred is not None:
        depth = depth_pred.squeeze().cpu().numpy()
        depth_color = visualize_depth(depth, orig_img.shape[:2])

        # Side-by-side comparison
        combined = np.hstack([orig_img, depth_color])
        depth_path = save_dir / "depth.jpg"
        combined_path = save_dir / "combined.jpg"

        cv2.imwrite(str(depth_path), depth_color)
        cv2.imwrite(str(combined_path), combined)

        print(f"Depth saved: {depth_path}")
        print(f"Combined saved: {combined_path}")
        print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
    else:
        print("Warning: No depth output found in model forward pass.")
        print(f"Output type: {type(outputs)}, len: {len(outputs) if isinstance(outputs, tuple) else 'N/A'}")

    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()
