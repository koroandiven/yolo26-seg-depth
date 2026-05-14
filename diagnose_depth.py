#!/usr/bin/env python3
"""Diagnose left/right depth bias and edge sharpness for depth checkpoints.

Compares one or more depth checkpoints by:
1. Left/right asymmetry: mean |D(x) - fliplr(D(fliplr(x)))| over a fixed test set.
   A perfectly symmetric model has 0; a model that always predicts the left side
   deeper has a large value.
2. Column-mean depth profile: average depth as a function of x. A flat profile
   means no systemic left/right bias; a sloped profile reveals the direction.
3. Edge sharpness vs RGB Sobel edges: correlation between |grad(D)| and
   |grad(I)| inside each ground-truth instance. Higher = better boundary
   alignment.
4. Side-by-side visualisation of depth maps for a handful of fixed images
   (with the input flipped vs not), saved as PNGs.

Usage:
    python diagnose_depth.py \
        --ckpts runs/.../exp8/weights/best.pt runs/.../exp14/weights/best.pt \
        --labels exp8 exp14 \
        --images-dir nyu_yolo/images/test \
        --n-images 80 \
        --n-vis 6 \
        --out diagnose_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Local ultralytics
local_ultralytics = str(Path(__file__).parent / "ultralytics")
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

from yolo26_inference import YOLO26Inference


def load_image_for_inference(path: Path, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    """Read image and return BGR original + 640-letterboxed BGR."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    # Letterbox to square keeping aspect ratio
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return img, canvas


def predict_depth(infer: YOLO26Inference, img_bgr: np.ndarray) -> np.ndarray:
    """Run inference and return depth map at the model's native resolution."""
    _ = infer.predict(source=img_bgr, conf=0.25, imgsz=640, verbose=False, save=False)
    head = infer._model.model.model[-1]
    if not hasattr(head, "_last_depth") or head._last_depth is None:
        raise RuntimeError("No depth captured; head not patched?")
    return head._last_depth.squeeze().detach().cpu().numpy()


def compute_metrics(infer: YOLO26Inference, image_paths: list[Path]) -> dict:
    """Compute asymmetry + column profile + edge correlation."""
    asym_vals = []
    col_profile = None
    n_col = 0
    edge_corr = []

    for p in image_paths:
        orig_bgr, lb_bgr = load_image_for_inference(p, 640)
        d_norm = predict_depth(infer, lb_bgr)
        flipped_bgr = np.ascontiguousarray(lb_bgr[:, ::-1, :])
        d_flip_in = predict_depth(infer, flipped_bgr)
        d_flip_back = d_flip_in[:, ::-1]

        # 1. Asymmetry
        asym_vals.append(float(np.mean(np.abs(d_norm - d_flip_back))))

        # 2. Column profile
        col_mean = d_norm.mean(axis=0)  # [W]
        if col_profile is None:
            col_profile = np.zeros_like(col_mean)
        col_profile += col_mean
        n_col += 1

        # 3. Edge correlation
        gray = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Resize depth to image resolution if needed
        if d_norm.shape != gray.shape:
            d_resized = cv2.resize(d_norm, (gray.shape[1], gray.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            d_resized = d_norm
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        img_edge = np.abs(gx) + np.abs(gy)
        dx = cv2.Sobel(d_resized, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(d_resized, cv2.CV_32F, 0, 1, ksize=3)
        d_edge = np.abs(dx) + np.abs(dy)
        # Pearson correlation on flattened arrays (after edge thresholding to
        # focus on actual edges rather than uniform-area noise)
        mask = img_edge > np.percentile(img_edge, 80)  # top-20% strong edges
        if mask.sum() > 100:
            a = img_edge[mask]
            b = d_edge[mask]
            if a.std() > 1e-6 and b.std() > 1e-6:
                edge_corr.append(float(np.corrcoef(a, b)[0, 1]))

    return {
        "asymmetry_mean": float(np.mean(asym_vals)),
        "asymmetry_median": float(np.median(asym_vals)),
        "asymmetry_std": float(np.std(asym_vals)),
        "column_profile": col_profile / n_col,
        "edge_corr_mean": float(np.mean(edge_corr)) if edge_corr else float("nan"),
        "edge_corr_median": float(np.median(edge_corr)) if edge_corr else float("nan"),
        "n": len(image_paths),
    }


def make_side_by_side(infer_list, labels, image_paths, out_dir: Path):
    """For each image: original + per-ckpt (depth, fliplr(depth(fliplr)) overlay)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in image_paths:
        orig_bgr, lb_bgr = load_image_for_inference(p, 640)
        rows = [lb_bgr]
        for infer, label in zip(infer_list, labels):
            d = predict_depth(infer, lb_bgr)
            d_flip_in = predict_depth(infer, np.ascontiguousarray(lb_bgr[:, ::-1, :]))
            d_back = d_flip_in[:, ::-1]
            d_vis = colourise(d)
            d_back_vis = colourise(d_back)
            diff = np.abs(d - d_back)
            diff_vis = colourise(diff, vmin=0.0, vmax=max(diff.max(), 0.5))
            cv2.putText(d_vis, f"{label} D(x)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(d_back_vis, f"{label} flip(D(flip x))", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(diff_vis, f"{label} |diff|", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            row = np.hstack([d_vis, d_back_vis, diff_vis])
            rows.append(row)
        # Resize all rows to same width
        target_w = max(r.shape[1] for r in rows)
        rows = [
            cv2.resize(r, (target_w, int(r.shape[0] * target_w / r.shape[1])))
            for r in rows
        ]
        canvas = np.vstack(rows)
        cv2.imwrite(str(out_dir / f"{p.stem}.png"), canvas)


def colourise(arr: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    a = arr.astype(np.float32)
    if vmin is None:
        vmin = float(a.min())
    if vmax is None:
        vmax = float(a.max())
    a = np.clip((a - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    a = (a * 255).astype(np.uint8)
    return cv2.applyColorMap(a, cv2.COLORMAP_JET)


def save_column_profile_plot(results: dict, labels: list[str], out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for r, lab in zip(results, labels):
        prof = r["column_profile"]
        xs = np.linspace(0, 1, len(prof))
        ax.plot(xs, prof, label=lab)
    ax.set_xlabel("normalised image x (0=left, 1=right)")
    ax.set_ylabel("mean predicted depth")
    ax.set_title("Column-mean depth profile (flat = no left/right bias)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--n-images", type=int, default=80)
    ap.add_argument("--n-vis", type=int, default=6)
    ap.add_argument("--out", default="diagnose_out")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    assert len(args.ckpts) == len(args.labels)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(Path(args.images_dir).glob("*.jpg"))[: args.n_images]
    vis_images = images[:: max(1, len(images) // args.n_vis)][: args.n_vis]
    print(f"Using {len(images)} images for metrics, {len(vis_images)} for visualisation")

    infer_list = []
    results = []
    for ckpt, label in zip(args.ckpts, args.labels):
        print(f"\n=== Loading {label}: {ckpt} ===")
        infer = YOLO26Inference(ckpt, device=args.device, verbose=False)
        infer_list.append(infer)
        print(f"Computing metrics on {len(images)} images...")
        m = compute_metrics(infer, images)
        results.append(m)
        print(
            f"  asym mean={m['asymmetry_mean']:.4f}  median={m['asymmetry_median']:.4f}  std={m['asymmetry_std']:.4f}"
        )
        print(
            f"  edge_corr mean={m['edge_corr_mean']:.4f}  median={m['edge_corr_median']:.4f}"
        )
        print(
            f"  col profile: left={m['column_profile'][:64].mean():.3f}  "
            f"right={m['column_profile'][-64:].mean():.3f}  "
            f"delta(right-left)={m['column_profile'][-64:].mean() - m['column_profile'][:64].mean():+.3f}"
        )

    # Save column plot
    save_column_profile_plot(results, args.labels, out_dir / "column_profile.png")

    # Save visualisations
    print("\nGenerating side-by-side visualisations...")
    make_side_by_side(infer_list, args.labels, vis_images, out_dir / "vis")

    # Save numerical summary
    with open(out_dir / "summary.txt", "w") as f:
        for r, lab in zip(results, args.labels):
            f.write(f"=== {lab} ===\n")
            f.write(f"  n: {r['n']}\n")
            f.write(f"  asymmetry mean   : {r['asymmetry_mean']:.4f}\n")
            f.write(f"  asymmetry median : {r['asymmetry_median']:.4f}\n")
            f.write(f"  asymmetry std    : {r['asymmetry_std']:.4f}\n")
            f.write(f"  edge_corr mean   : {r['edge_corr_mean']:.4f}\n")
            f.write(f"  edge_corr median : {r['edge_corr_median']:.4f}\n")
            cp = r["column_profile"]
            f.write(f"  col left  mean   : {cp[:64].mean():.3f}\n")
            f.write(f"  col right mean   : {cp[-64:].mean():.3f}\n")
            f.write(f"  col delta(R-L)   : {cp[-64:].mean() - cp[:64].mean():+.3f}\n\n")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
