#!/usr/bin/env python3
"""
YOLO26 Depth + Segmentation Inference with Mask Guidance

This script performs joint segmentation and depth estimation inference.
The key difference from the basic version is that segmentation masks are
explicitly fed into the depth decoder as spatial guidance, enforcing
object-level depth consistency.
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Ensure local ultralytics (clear cache first)
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("ultralytics"):
        del sys.modules[mod_name]

local_ultralytics = str(Path(__file__).parent / "ultralytics")
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

from ultralytics import YOLO
from ultralytics.nn.modules.head import DepthSegment26, Segment26
from ultralytics.utils.torch_utils import select_device


def generate_mask_guidance_from_results(result, target_shape):
    """Convert segmentation Results to [B, 2, H, W] mask guidance tensor.

    Channel 0: occupancy map (union of all instance masks)
    Channel 1: edge map (Sobel on occupancy)
    """
    H, W = target_shape
    device = result.boxes.data.device if len(result.boxes) > 0 else torch.device("cpu")

    occupancy = torch.zeros(1, 1, H, W, device=device)
    edge = torch.zeros(1, 1, H, W, device=device)

    if result.masks is None or len(result.masks) == 0:
        return torch.cat([occupancy, edge], dim=1)

    # Stack all instance masks (N, H, W)
    masks = result.masks.data  # [N, H, W] on device
    if masks.shape[-2:] != (H, W):
        masks = F.interpolate(
            masks.unsqueeze(1), size=(H, W), mode="nearest"
        ).squeeze(1)

    # Occupancy: union of all instances
    occ = (masks > 0.5).any(dim=0).float()  # [H, W]
    occupancy[0, 0] = occ

    # Edge: Sobel on occupancy
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=device, dtype=torch.float32,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=device, dtype=torch.float32,
    ).view(1, 1, 3, 3)

    occ_exp = occ.unsqueeze(0).unsqueeze(0)
    ex = F.conv2d(occ_exp, sobel_x, padding=1).abs()
    ey = F.conv2d(occ_exp, sobel_y, padding=1).abs()
    edge_map = (ex + ey).squeeze()
    # Suppress Sobel padding artifacts at image boundaries
    edge_map[0, :] = 0
    edge_map[-1, :] = 0
    edge_map[:, 0] = 0
    edge_map[:, -1] = 0
    edge[0, 0] = edge_map

    return torch.cat([occupancy, edge], dim=1)


def patch_depthsegment26_for_inference():
    """Patch DepthSegment26.forward for inference.

    The patched forward does NOT compute depth internally; instead it stores
    features so that depth can be computed externally with mask guidance.
    """

    def _patched_forward(self, x):
        """Return standard Segment outputs; store features for external depth."""
        # Segmentation path
        seg_feat, depth_feat = self.task_attention(x[0])
        x_seg = [seg_feat, *x[1:]]
        outputs = Segment26.forward(self, x_seg)

        # Store features for external depth computation
        self._last_features = [depth_feat, x[1], x[2]]

        # Eval mode: return standard Segment outputs
        return outputs

    DepthSegment26.forward = _patched_forward


def visualize_depth(depth: np.ndarray, orig_shape: tuple) -> np.ndarray:
    depth = cv2.resize(depth, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
    d_min, d_max = depth.min(), depth.max()
    depth_vis = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.putText(depth_color, f"Depth: {d_min:.2f}m - {d_max:.2f}m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return depth_color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./inference_results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--no-mask-guide",
        action="store_true",
        help="Disable mask guidance for depth (fallback to original behavior).",
    )
    args = parser.parse_args()

    # Patch before loading model
    patch_depthsegment26_for_inference()

    device = select_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    model.to(device)

    # Override names to COCO 80
    from ultralytics.nn.tasks import yaml_model_load
    coco_yaml = Path(__file__).parent / "ultralytics" / "ultralytics" / "cfg" / "datasets" / "coco.yaml"
    if coco_yaml.exists():
        coco_data = yaml_model_load(coco_yaml)
        model.model.names = coco_data.get("names", model.model.names)
        print(f"Restored COCO names ({len(model.model.names)} classes)")

    print(f"First 10 names: {dict(list(model.model.names.items())[:10])}")

    # Patch instance forward
    head = model.model.model[-1]
    print(f"Head type: {type(head).__name__}")

    def _instance_forward(x):
        """Patched forward for this instance only."""
        seg_feat, depth_feat = head.task_attention(x[0])
        x_seg = [seg_feat, *x[1:]]
        outputs = Segment26.forward(head, x_seg)
        head._last_features = [depth_feat, x[1], x[2]]
        return outputs

    head.forward = _instance_forward
    print("Patched instance forward for inference")

    # ------------------------------------------------------------------
    # 2. Segmentation inference
    # ------------------------------------------------------------------
    print(f"\nRunning segmentation inference on: {args.source}")
    seg_results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz, verbose=False, device=device)
    seg_result = seg_results[0]

    seg_img = seg_result.plot()
    seg_path = save_dir / "segmentation.jpg"
    cv2.imwrite(str(seg_path), cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    print(f"Segmentation saved: {seg_path} ({len(seg_result.boxes)} detections)")

    # ------------------------------------------------------------------
    # 3. Mask-guided depth inference
    # ------------------------------------------------------------------
    print("\nExtracting depth map with mask guidance...")
    orig_img = cv2.imread(args.source)

    # Preprocess image with same letterbox as model.predict()
    from ultralytics.data.augment import LetterBox
    letterbox_transform = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    im = letterbox_transform(image=orig_img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_tensor = torch.from_numpy(im.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

    # Extract features via backbone + neck
    with torch.no_grad():
        # Run patched forward (stores _last_features)
        _ = model.model(im_tensor)

        # Get stored features
        features = head._last_features  # [depth_feat(P3), P4, P5]

        # Generate mask guidance from segmentation results
        if args.no_mask_guide or seg_result.masks is None or len(seg_result.masks) == 0:
            mask_guidance = None
            print("Mask guidance: DISABLED (no masks or --no-mask-guide)")
        else:
            mask_guidance = generate_mask_guidance_from_results(
                seg_result, target_shape=(args.imgsz // 8, args.imgsz // 8)
            )
            print(f"Mask guidance: ENABLED ({len(seg_result.masks)} instances)")

        # Run mask-guided depth decoder
        depth = head.mask_guided_depth_decoder(features, mask_guidance)
        depth_scale = getattr(head, 'depth_scale', 100.0)
        depth = torch.sigmoid(depth) * depth_scale  # [0, depth_scale] meters
        depth = depth.squeeze().cpu().numpy()

    depth_color = visualize_depth(depth, orig_img.shape[:2])
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
