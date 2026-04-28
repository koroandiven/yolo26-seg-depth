#!/usr/bin/env python3
"""
YOLO26 Depth + Segmentation Inference (Fixed)
Patches DepthSegment26.forward for compatibility with SegmentationPredictor
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import sys
import torch
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


def patch_depthsegment26_for_inference():
    """Patch DepthSegment26.forward to return compatible format for SegmentationPredictor."""
    _original_forward = DepthSegment26.forward

    def _patched_forward(self, x):
        """Return standard Segment outputs; store depth in self._last_depth."""
        outputs = Segment26.forward(self, x)

        # Always compute depth (same as original)
        seg_feat, depth_feat = self.task_attention(x[0])
        depth = self.depth_decoder(x)
        depth = torch.sigmoid(depth) * 100.0
        self._last_depth = depth

        # In training, add depth to outputs dict
        if self.training:
            if isinstance(outputs, dict):
                outputs["depth"] = depth
            return outputs

        # Eval mode: return EXACTLY what Segment26 would return
        # (do NOT append depth - predictor expects standard format)
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

    # Override names to COCO 80 (checkpoint saved with NYU 10 names)
    from ultralytics.nn.tasks import yaml_model_load
    coco_yaml = Path(__file__).parent / "ultralytics" / "ultralytics" / "cfg" / "datasets" / "coco.yaml"
    if coco_yaml.exists():
        coco_data = yaml_model_load(coco_yaml)
        model.model.names = coco_data.get("names", model.model.names)
        print(f"Restored COCO names ({len(model.model.names)} classes)")
    else:
        print(f"Warning: COCO yaml not found at {coco_yaml}, using saved names")

    print(f"First 10 names: {dict(list(model.model.names.items())[:10])}")

    # ------------------------------------------------------------------
    # Patch the loaded INSTANCE's forward (class patch doesn't affect pickle-loaded objects)
    # ------------------------------------------------------------------
    head = model.model.model[-1]
    print(f"Head type: {type(head).__name__}")

    def _instance_forward(x):
        """Patched forward for this instance only."""
        outputs = Segment26.forward(head, x)
        # Compute and store depth, but return standard Segment outputs
        seg_feat, depth_feat = head.task_attention(x[0])
        depth = head.depth_decoder(x)
        depth = torch.sigmoid(depth) * 100.0
        head._last_depth = depth
        return outputs

    head.forward = _instance_forward
    print("Patched instance forward for inference")

    # Debug: check model forward output format
    print("\n--- Debug: checking model forward output ---")
    import torch
    dbg_img = cv2.imread(args.source)
    dbg_im = cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB)
    dbg_im = cv2.resize(dbg_im, (args.imgsz, args.imgsz))
    dbg_im = torch.from_numpy(dbg_im.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        dbg_out = model.model(dbg_im)
    print(f"Forward output type: {type(dbg_out)}")
    if isinstance(dbg_out, tuple):
        print(f"  Tuple length: {len(dbg_out)}")
        for i, o in enumerate(dbg_out):
            print(f"  [{i}] type: {type(o).__name__}, shape: {getattr(o, 'shape', 'N/A')}")
    elif isinstance(dbg_out, dict):
        print(f"  Dict keys: {list(dbg_out.keys())}")
    print("--- End debug ---\n")

    # ------------------------------------------------------------------
    # 2. Segmentation inference (uses patched forward, returns standard format)
    # ------------------------------------------------------------------
    print(f"\nRunning segmentation inference on: {args.source}")
    seg_results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz, verbose=False, device=device)
    seg_result = seg_results[0]

    seg_img = seg_result.plot()
    seg_path = save_dir / "segmentation.jpg"
    cv2.imwrite(str(seg_path), cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    print(f"Segmentation saved: {seg_path} ({len(seg_result.boxes)} detections)")

    # ------------------------------------------------------------------
    # 3. Extract depth from patched forward
    # ------------------------------------------------------------------
    print("\nExtracting depth map...")
    orig_img = cv2.imread(args.source)

    # _last_depth was stored during the predict() call above
    depth_head = model.model.model[-1]
    if hasattr(depth_head, "_last_depth") and depth_head._last_depth is not None:
        depth = depth_head._last_depth.squeeze().cpu().numpy()
    else:
        # Fallback: manual forward pass
        im = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (args.imgsz, args.imgsz))
        im = torch.from_numpy(im.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model.model(im)
        depth = depth_head._last_depth.squeeze().cpu().numpy()

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
