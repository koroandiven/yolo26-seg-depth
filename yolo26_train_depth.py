#!/usr/bin/env python3
"""
YOLO26 Progressive Training Script for Depth + Segmentation Multi-task Learning
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local ultralytics is used instead of site-packages
local_ultralytics = str(Path(__file__).parent / "ultralytics")
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.segment.train import DepthSegmentTrainer
from ultralytics.utils import LOGGER


class ProgressiveFreezeCallback:
    """Progressive training freeze callback - keep seg head frozen, train depth head + backbone.

    Phase 1 (0 ~ freeze_depth_epochs-1): Only train depth head, freeze backbone + seg head.
    Phase 2 (freeze_depth_epochs ~ end): Train depth head + backbone, keep seg head frozen.
    Segmentation loss gradients are optionally blocked to prevent NYU labels from corrupting
    COCO pretrained backbone features.
    """

    def __init__(self, freeze_depth_epochs=50, unfreeze_backbone_at=-1):
        self.freeze_depth_epochs = freeze_depth_epochs
        self.unfreeze_backbone_at = unfreeze_backbone_at
        self.initialized = False
        self._prev_phase = None
        self.freeze_depth_epochs = freeze_depth_epochs
        self.initialized = False
        self._prev_phase = None

    def on_train_epoch_start(self, trainer):
        """Switch freeze strategy at the beginning of each epoch."""
        epoch = trainer.epoch
        model = trainer.model

        # Update freeze strategy whenever phase changes or on first call
        current_phase = 1 if epoch < self.freeze_depth_epochs else 2
        if not self.initialized or current_phase != self._prev_phase:
            self._update_freeze(model, epoch)
            self._print_phase(epoch)
            self.initialized = True
            self._prev_phase = current_phase

    def on_train_batch_start(self, trainer):
        """Re-enforce seg head freeze every batch (trainer.model.train() overwrites it).

        NOTE: We do NOT call seg_head.eval() here because that switches
        DepthSegment26.forward() to inference mode, causing it to return a
        tuple instead of a dict and dropping the 'depth' key.  Instead we only
        freeze BN running stats and keep params frozen.
        """
        from ultralytics.utils.torch_utils import unwrap_model
        import torch.nn as nn

        model = unwrap_model(trainer.model)
        seg_head = model.model[-1] if hasattr(model, "model") else None
        if seg_head is not None:
            # Freeze BN running stats (but keep training mode so forward returns dict)
            for m in seg_head.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()
            # Re-freeze seg head params in case DDP or AMP modifies them.
            # Only depth_branch is trainable; seg_branch stays frozen.
            for n, p in seg_head.named_parameters():
                if "depth" not in n and "task_attention.depth_branch" not in n:
                    p.requires_grad = False
        # Phase 3 (opt-in): unfreeze backbone/neck so depth gradients can
        # adapt shared features. Seg head params stay frozen (loop above);
        # L2-SP + the COCO seg guard keep segmentation capability intact.
        if self.unfreeze_backbone_at >= 0 and trainer.epoch >= self.unfreeze_backbone_at:
            head_prefix = f"model.{len(model.model) - 1}."
            for n, p in model.named_parameters():
                if not n.startswith(head_prefix):
                    p.requires_grad = True

    def _update_freeze(self, model, epoch):
        """Update freeze strategy: keep seg head frozen, only train depth + backbone."""
        from ultralytics.utils.torch_utils import unwrap_model

        model = unwrap_model(model)

        # Identify seg head (last layer = detection/segmentation head)
        seg_head = model.model[-1] if hasattr(model, "model") else None

        # NOTE: Backbone is kept frozen in ALL phases when seg preservation is
        # required.  Unfreezing backbone for depth training inevitably shifts
        # shared features and destroys pretrained segmentation performance.
        # Only depth head (+ depth_branch of task_attention) is trainable.
        for name, param in model.named_parameters():
            if "depth" in name or "task_attention.depth_branch" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Freeze BN running stats in seg head (keep training mode so forward
        # still returns a dict with the 'depth' key), and keep seg params frozen.
        if seg_head is not None:
            import torch.nn as nn

            for m in seg_head.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()
            for n, p in seg_head.named_parameters():
                if "depth" not in n and "task_attention.depth_branch" not in n:
                    p.requires_grad = False

    def _print_phase(self, epoch):
        """Print current training phase.

        Backbone is permanently frozen to preserve pretrained segmentation
        performance; only depth head (+ task_attention.depth_branch) is trained.
        """
        LOGGER.info(
            "Training depth head only (backbone frozen, seg head frozen, "
            "task_attention.seg_branch frozen)"
        )


class CocoSegGuardCallback:
    """Periodic COCO segmentation evaluation during depth training.

    The depth trainer validates on NYU, where the frozen COCO seg head scores
    ~0, so segmentation degradation from (partially) unfrozen training was
    invisible. This guard runs the stock SegmentationValidator on a COCO
    dataset every N epochs, logs mAP and appends coco_guard.csv in the run
    directory. It never raises into training: repeated failures disable it.
    """

    def __init__(self, data: str = "", interval: int = 10, drop_tol: float = 0.05):
        self.data = data
        self.interval = max(int(interval), 1)
        self.drop_tol = float(drop_tol)
        self.baseline = None
        self.failures = 0
        self.csv_path = None

    def on_fit_epoch_end(self, trainer):
        """Run the guard right after the regular epoch-end validation."""
        if not self.data or getattr(trainer, "RANK", -1) not in {-1, 0}:
            return
        is_due = (trainer.epoch + 1) % self.interval == 0 or trainer.epoch == trainer.epochs - 1
        if not is_due:
            return
        try:
            stats = self._validate(trainer)
        except Exception as e:  # never break training because of the guard
            self.failures += 1
            LOGGER.warning(f"COCO seg guard failed ({self.failures}x): {e}")
            if self.failures >= 3:
                LOGGER.warning("COCO seg guard disabled after 3 consecutive failures.")
                self.data = ""
            return
        m50 = float(stats.get("metrics/mAP50(M)", stats.get("metrics/mAP50(B)", 0.0)) or 0.0)
        m95 = float(stats.get("metrics/mAP50-95(M)", 0.0) or 0.0)
        if self.baseline is None:
            self.baseline = m50
        elif m50 < self.baseline - self.drop_tol:
            LOGGER.warning(
                f"COCO seg guard: mAP50(M)={m50:.4f} dropped below baseline "
                f"{self.baseline:.4f} - {self.drop_tol} — the segmentation path is degrading."
            )
        LOGGER.info(f"COCO seg guard (epoch {trainer.epoch + 1}): mAP50(M)={m50:.4f} mAP50-95(M)={m95:.4f}")
        self._write_csv(trainer, m50, m95)

    def _validate(self, trainer):
        """Run the stock segmentation validator on the guard dataset."""
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = get_cfg(
            overrides={
                "task": "segment",
                "data": self.data,
                "imgsz": trainer.args.imgsz,
                "batch": trainer.args.batch,
                "device": trainer.args.device,
                "workers": trainer.args.workers,
                "plots": False,
                "save_json": False,
                "rect": False,
                "conf": 0.001,
                "iou": 0.7,
                "verbose": False,
            }
        )
        validator = SegmentationValidator(
            dataloader=None, save_dir=Path(trainer.save_dir) / "coco_guard", args=args, _callbacks={}
        )
        model = trainer.ema.ema if getattr(trainer, "ema", None) and trainer.ema.ema else trainer.model
        model.eval()
        with torch.no_grad():
            stats = validator(model=model)
        return stats or {}

    def _write_csv(self, trainer, m50, m95):
        try:
            import csv

            save_dir = Path(trainer.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = save_dir / "coco_guard.csv"
            write_header = not self.csv_path.exists()
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "mAP50(M)", "mAP50-95(M)", "baseline_mAP50(M)"])
                writer.writerow([trainer.epoch + 1, f"{m50:.5f}", f"{m95:.5f}", f"{self.baseline:.5f}"])
        except Exception as e:
            LOGGER.warning(f"COCO seg guard could not write csv: {e}")
def train_progressive():
    """Progressive training entry function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo26-seg-depth.yaml")
    parser.add_argument("--data", type=str, default="nyu_yolo/nyu_depth_seg.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--depth-weight", type=float, default=0.5)
    parser.add_argument("--sym-weight", type=float, default=0.0,
                        help="Flip-consistency (symmetry) loss weight for depth; 0 disables it")
    parser.add_argument("--freeze-depth-epochs", type=int, default=50)
    parser.add_argument("--depth-mosaic", type=float, default=0.0,
                        help="Geometry-consistent mosaic probability for depth training; 0 disables (default)")
    parser.add_argument("--unfreeze-backbone-at", type=int, default=-1,
                        help="Epoch from which to unfreeze backbone/neck for depth adaptation; -1 keeps them frozen (default)")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1,
                        help="LR multiplier for unfrozen backbone/neck params (default 0.1)")
    parser.add_argument("--sp-weight", type=float, default=0.0,
                        help="L2-SP weight anchoring unfrozen weights to pretrained values; 0 disables (default)")
    parser.add_argument("--coco-guard-data", type=str, default="",
                        help="Dataset yaml for periodic COCO seg checks during depth training "
                             "(e.g. coco128-seg.yaml or a COCO val yaml); empty disables (default)")
    parser.add_argument("--coco-guard-interval", type=int, default=10,
                        help="Run the COCO seg guard every N epochs (default 10)")
    parser.add_argument("--coco-guard-drop-tol", type=float, default=0.05,
                        help="Warn when COCO mAP50(M) drops this far below the first guard measurement (default 0.05)")
    parser.add_argument("--freeze-seg", action="store_true", default=True,
                        help="Block segmentation loss gradients from flowing back to backbone (default: True)")
    parser.add_argument("--no-freeze-seg", action="store_true",
                        help="Allow segmentation loss gradients to update backbone (NOT recommended when using different segmentation labels)")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to COCO segmentation pretrained weights (e.g., yolo26n-seg.pt). "
             "Required to preserve segmentation performance.",
    )
    parser.add_argument("--project", type=str, default="runs/train_depth")
    parser.add_argument("--name", type=str, default="yolo26-seg-depth-exp")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--fliplr", type=float, default=None,
                        help="Horizontal flip probability for depth augmentation (default: use hyp default 0.5)")
    args = parser.parse_args()

    # Handle --no-freeze-seg
    freeze_seg = not args.no_freeze_seg

    # Separate valid YOLO args from custom args
    # Ultralytics get_cfg() validates all keys, so custom ones must be set after init
    valid_overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
    }
    if args.fliplr is not None:
        valid_overrides["fliplr"] = args.fliplr

    trainer = DepthSegmentTrainer(overrides=valid_overrides)

    # Set custom attributes after init (not validated by get_cfg)
    trainer.depth_weight = args.depth_weight
    trainer.sym_weight = args.sym_weight
    trainer.sp_weight = args.sp_weight
    trainer.unfreeze_backbone_at = args.unfreeze_backbone_at
    trainer.backbone_lr_scale = args.backbone_lr_scale
    trainer.freeze_depth_epochs = args.freeze_depth_epochs
    trainer.pretrained_path = args.pretrained  # path to pretrained .pt file
    trainer.freeze_seg = freeze_seg

    # Re-init criterion with freeze_seg flag
    if hasattr(trainer.model, "criterion") and trainer.model.criterion is not None:
        trainer.model.criterion.freeze_seg = freeze_seg
        trainer.model.criterion.sym_weight = args.sym_weight

    LOGGER.info(
        f"Training config: pretrained={args.pretrained}, "
        f"freeze_depth_epochs={args.freeze_depth_epochs}, "
        f"freeze_seg={freeze_seg}, sym_weight={args.sym_weight}"
    )

    # Register progressive freeze callback (seg head always frozen)
    freeze_callback = ProgressiveFreezeCallback(
        freeze_depth_epochs=args.freeze_depth_epochs,
        unfreeze_backbone_at=args.unfreeze_backbone_at,
    )
    trainer.add_callback("on_train_epoch_start", freeze_callback.on_train_epoch_start)
    trainer.add_callback("on_train_batch_start", freeze_callback.on_train_batch_start)

    guard = CocoSegGuardCallback(
        data=args.coco_guard_data,
        interval=args.coco_guard_interval,
        drop_tol=args.coco_guard_drop_tol,
    )
    trainer.add_callback("on_fit_epoch_end", guard.on_fit_epoch_end)

    # Start training
    trainer.train()


if __name__ == "__main__":
    train_progressive()
