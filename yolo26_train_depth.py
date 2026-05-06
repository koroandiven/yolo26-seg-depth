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

    def __init__(
        self, freeze_depth_epochs=50, use_gradnorm=False
    ):
        self.freeze_depth_epochs = freeze_depth_epochs
        self.use_gradnorm = use_gradnorm
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


def train_progressive():
    """Progressive training entry function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo26-seg-depth.yaml")
    parser.add_argument("--data", type=str, default="nyu_yolo/nyu_depth_seg.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--depth-weight", type=float, default=0.5)
    parser.add_argument("--use-gradnorm", action="store_true")
    parser.add_argument("--freeze-depth-epochs", type=int, default=50)
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

    trainer = DepthSegmentTrainer(overrides=valid_overrides)

    # Set custom attributes after init (not validated by get_cfg)
    trainer.depth_weight = args.depth_weight
    trainer.use_gradnorm = args.use_gradnorm
    trainer.freeze_depth_epochs = args.freeze_depth_epochs
    trainer.pretrained_path = args.pretrained  # path to pretrained .pt file
    trainer.freeze_seg = freeze_seg

    # Re-init criterion with freeze_seg flag
    if hasattr(trainer.model, "criterion") and trainer.model.criterion is not None:
        trainer.model.criterion.freeze_seg = freeze_seg

    LOGGER.info(
        f"Training config: pretrained={args.pretrained}, "
        f"freeze_depth_epochs={args.freeze_depth_epochs}, "
        f"freeze_seg={freeze_seg}"
    )

    # Register progressive freeze callback (seg head always frozen)
    freeze_callback = ProgressiveFreezeCallback(
        freeze_depth_epochs=args.freeze_depth_epochs,
        use_gradnorm=args.use_gradnorm,
    )
    trainer.add_callback("on_train_epoch_start", freeze_callback.on_train_epoch_start)
    trainer.add_callback("on_train_batch_start", freeze_callback.on_train_batch_start)

    # Start training
    trainer.train()


if __name__ == "__main__":
    train_progressive()
