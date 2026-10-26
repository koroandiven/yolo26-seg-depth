#!/usr/bin/env python3
"""
YOLO26 Reverse Training: train segmentation head, freeze depth head.

Key fix for class ID mismatch (COCO 80 vs Dataset 120):
- Only load backbone+neck weights from yolo26s-seg.pt
- Re-initialize detection/segmentation head from scratch
  (because dataset class IDs (0-79) don't match COCO's class IDs at the same slot)
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path

local_ultralytics = str(Path(__file__).parent / "ultralytics")
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

import torch
from ultralytics.models.yolo.segment.train import DepthSegmentTrainer
from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import SegmentationModel

os.environ["ULTRALYTICS_DISABLE_MLFLOW"] = "1"


def load_backbone_only(model: SegmentationModel, pretrained_path: str, verbose: bool = True):
    """Load only backbone+neck from pretrained weights; randomly init head.

    This avoids the class-ID-mismatch problem when the dataset's class IDs
    (0-79) don't align with COCO's pretrained class IDs at the same slot.
    Detection/segmentation heads will be trained from scratch (~20 epochs).
    """
    import torch.nn as nn

    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    src_sd = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
    dst_sd = model.state_dict()

    HEAD_KEYWORDS = (
        "cv2", "cv3", "cv4", "cv5",
        "proto", "dfl",
        "Segment26", "DepthSegment26",
        "task_attention", "mask_guided_depth_decoder",
    )

    loaded, skipped_head, shape_mismatch = 0, 0, 0
    new_sd = {}
    for k, v in dst_sd.items():
        if any(hk in k for hk in HEAD_KEYWORDS):
            skipped_head += 1
            continue
        if k in src_sd and src_sd[k].shape == v.shape:
            new_sd[k] = src_sd[k]
            loaded += 1
        else:
            shape_mismatch += 1
    dst_sd.update(new_sd)
    model.load_state_dict(dst_sd, strict=False)
    LOGGER.info(f"Loaded backbone+neck: {loaded} tensors; "
                f"Skipped head: {skipped_head}; Shape mismatches: {shape_mismatch}")


class ReverseFreezeCallback:
    """训分割、冻结深度。"""

    def __init__(self, freeze_backbone: bool = False):
        self.freeze_backbone = freeze_backbone
        self.initialized = False

    def on_train_epoch_start(self, trainer):
        if not self.initialized:
            self._apply_freeze(trainer.model)
            self.initialized = True

    def on_train_batch_start(self, trainer):
        self._apply_freeze(trainer.model)

    def _apply_freeze(self, model):
        from ultralytics.utils.torch_utils import unwrap_model
        import torch.nn as nn
        m = unwrap_model(model)
        seg_head = m.model[-1] if hasattr(m, "model") else None

        for name, p in m.named_parameters():
            if "depth" in name or "task_attention.depth_branch" in name:
                p.requires_grad = False                       # depth 全冻结
            elif self.freeze_backbone and not name.startswith("model.22"):
                p.requires_grad = False
            else:
                p.requires_grad = True

        if seg_head is not None:
            for n, mod in seg_head.named_modules():
                if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    if "depth" in n or "task_attention.depth_branch" in n:
                        mod.eval()
                    else:
                        mod.train()

        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        LOGGER.info(f"ReverseFreeze: trainable params = {n_train:,}, "
                    f"freeze_backbone={self.freeze_backbone}")


def train_seg_only():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth-120c.yaml")
    p.add_argument("--data",   default="ultralytics/ultralytics/cfg/datasets/yolo26_seg120_20251121.yaml")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--depth-weight", type=float, default=0.0)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--project", default="runs/train_seg")
    p.add_argument("--name",   default="yolo26-seg-120c-stage1")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pretrained", type=str, default=None,
                   help="Optional. If given, only backbone+neck are loaded (head reinit). "
                        "Omit to train from scratch.")
    p.add_argument("--load-full-head", action="store_true",
                   help="DANGEROUS: load full weights including misaligned head (faster but wrong)")
    # Ultralytics passthrough args
    p.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=None, help="Final LR fraction (cos-lr)")
    p.add_argument("--cos-lr", action="store_true", help="Use cosine LR scheduler")
    p.add_argument("--optimizer", type=str, default=None, help="Optimizer (SGD/Adam/AdamW)")
    p.add_argument("--momentum", type=float, default=None, help="SGD momentum")
    p.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    p.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs")
    p.add_argument("--close-mosaic", type=int, default=None, help="Disable mosaic in last N epochs")
    p.add_argument("--amp", action="store_true", help="Enable AMP (mixed precision)")
    p.add_argument("--fraction", type=float, default=None,
                   help="Fraction of the training set to use (0-1); 1.0 = all (default). "
                        "Useful for smoke tests / quick convergence checks.")
    args = p.parse_args()

    overrides = {
        "model": args.model, "data": args.data,
        "epochs": args.epochs, "batch": args.batch,
        "device": args.device, "imgsz": args.imgsz,
        "workers": args.workers, "project": args.project,
        "name": args.name, "resume": args.resume,
        "patience": args.patience, "fliplr": args.fliplr,
    }
    # Pass through Ultralytics training args if provided
    for opt in ("lr0", "lrf", "cos_lr", "optimizer", "momentum",
                "weight_decay", "warmup_epochs", "close_mosaic", "amp", "fraction"):
        val = getattr(args, opt, None)
        if val is not None and val is not False:
            overrides[opt] = val

    trainer = DepthSegmentTrainer(overrides=overrides)
    trainer.depth_weight = args.depth_weight
    trainer.freeze_seg = False
    trainer.pretrained_path = None  # disable default loading
    trainer.freeze_depth_epochs = 0

    # Build the model through the trainer so it carries the depth attributes
    # (depth_weight / sym_weight / freeze_seg / args) that select the multi-task
    # criterion: with depth_weight missing, the stock 5-item v8SegmentationLoss is
    # used for training while validation uses the 7-item depth loss -> shape mismatch.
    model = trainer.get_model(cfg=args.model, weights=None, verbose=True)

    if hasattr(model, "criterion") and model.criterion is not None:
        model.criterion.freeze_seg = False

    # Load only backbone+neck (avoid class-ID-mismatch in head)
    if args.pretrained is None:
        LOGGER.info("Training from scratch (no pretrained weights)")
    elif not args.load_full_head:
        LOGGER.info(f"Loading backbone+neck only from {args.pretrained} (head will be reinitialized)")
        load_backbone_only(model, args.pretrained)
    else:
        LOGGER.warning("Loading full weights INCLUDING misaligned head — fast but wrong class semantics")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        model.load(ckpt)

    trainer.model = model

    cb = ReverseFreezeCallback(freeze_backbone=args.freeze_backbone)
    trainer.add_callback("on_train_epoch_start", cb.on_train_epoch_start)
    trainer.add_callback("on_train_batch_start", cb.on_train_batch_start)

    LOGGER.info(f"Reverse training (seg-only): nc=120, depth_weight={args.depth_weight}, "
                f"freeze_backbone={args.freeze_backbone}, load_full_head={args.load_full_head}")
    trainer.train()


if __name__ == "__main__":
    train_seg_only()