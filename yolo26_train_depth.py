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
    """Progressive training freeze callback - switches freeze strategy at each epoch start."""

    def __init__(
        self, freeze_depth_epochs=50, freeze_seg_epochs=50, use_gradnorm=False
    ):
        self.freeze_depth_epochs = freeze_depth_epochs
        self.freeze_seg_epochs = freeze_seg_epochs
        self.use_gradnorm = use_gradnorm
        self.initialized = False

    def on_train_epoch_start(self, trainer):
        """Switch freeze strategy at the beginning of each epoch."""
        epoch = trainer.epoch
        model = trainer.model

        if not self.initialized or epoch in (
            self.freeze_depth_epochs,
            self.freeze_depth_epochs + self.freeze_seg_epochs,
        ):
            self._update_freeze(model, epoch)
            self._print_phase(epoch)
            self.initialized = True

    def _update_freeze(self, model, epoch):
        """Update freeze strategy."""
        from ultralytics.utils.torch_utils import unwrap_model

        model = unwrap_model(model)

        if epoch < self.freeze_depth_epochs:
            # Phase 1: Only train depth head
            for name, param in model.named_parameters():
                if "depth" not in name:
                    param.requires_grad = False
        elif epoch < self.freeze_depth_epochs + self.freeze_seg_epochs:
            # Phase 2: Unfreeze segmentation head
            for name, param in model.named_parameters():
                if "depth" not in name and "seg" not in name and "cv4" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            # Phase 3: Full unfreeze
            for param in model.parameters():
                param.requires_grad = True

    def _print_phase(self, epoch):
        """Print current training phase."""
        if epoch < self.freeze_depth_epochs:
            phase = 1
            lr = 1e-3
        elif epoch < self.freeze_depth_epochs + self.freeze_seg_epochs:
            phase = 2
            lr = 5e-4
        else:
            phase = 3
            lr = 1e-4
        LOGGER.info(
            f"Phase {phase}: lr={lr}, freeze_depth_epochs={self.freeze_depth_epochs}"
        )


def train_progressive():
    """Progressive training entry function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo26-seg-depth.yaml")
    parser.add_argument("--data", type=str, default="depth-seg.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--depth-weight", type=float, default=0.5)
    parser.add_argument("--use-gradnorm", action="store_true")
    parser.add_argument("--freeze-depth-epochs", type=int, default=50)
    parser.add_argument("--freeze-seg-epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained weights (e.g., yolo26s-seg.pt)",
    )
    parser.add_argument("--project", type=str, default="runs/train_depth")
    parser.add_argument("--name", type=str, default="yolo26-seg-depth-exp")
    args = parser.parse_args()

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
    }

    trainer = DepthSegmentTrainer(overrides=valid_overrides)

    # Set custom attributes after init (not validated by get_cfg)
    trainer.depth_weight = args.depth_weight
    trainer.use_gradnorm = args.use_gradnorm
    trainer.freeze_depth_epochs = args.freeze_depth_epochs
    trainer.freeze_seg_epochs = args.freeze_seg_epochs
    trainer.pretrained_path = args.pretrained  # path to pretrained .pt file

    # Register progressive freeze callback
    freeze_callback = ProgressiveFreezeCallback(
        freeze_depth_epochs=args.freeze_depth_epochs,
        freeze_seg_epochs=args.freeze_seg_epochs,
        use_gradnorm=args.use_gradnorm,
    )
    trainer.add_callback("on_train_epoch_start", freeze_callback.on_train_epoch_start)

    # Start training
    trainer.train()


if __name__ == "__main__":
    train_progressive()
