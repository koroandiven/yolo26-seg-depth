# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.data.depth_dataset import DepthSegmentDataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training based on a segmentation model.

    This trainer specializes in handling segmentation tasks, extending the detection trainer with segmentation-specific
    functionality including model initialization, validation, and visualization.

    Attributes:
        loss_names (tuple[str]): Names of the loss components used during training.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize a SegmentationTrainer object.

        Args:
            cfg (dict): Configuration dictionary with default training settings.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Initialize and return a SegmentationModel with specified configuration and weights.

        Args:
            cfg (dict | str, optional): Model configuration. Can be a dictionary, a path to a YAML file, or None.
            weights (str | Path, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (SegmentationModel): Initialized segmentation model with loaded weights if specified.

        Examples:
            >>> trainer = SegmentationTrainer()
            >>> model = trainer.get_model(cfg="yolo26n-seg.yaml")
            >>> model = trainer.get_model(weights="yolo26n-seg.pt", verbose=False)
        """
        model = SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class DepthSegmentTrainer(SegmentationTrainer):
    """Segmentation + Depth multi-task trainer - supports progressive training and multi-task loss."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)
        self.use_gradnorm = getattr(self.args, "use_gradnorm", False)
        self.depth_weight = getattr(self.args, "depth_weight", 0.5)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Initialize model and replace with multi-task loss function.

        Keep nc from model YAML (80 for COCO) instead of overriding with data YAML.
        This preserves pretrained segmentation weights while training only depth head.
        """
        # Do NOT pass nc=self.data["nc"] to keep COCO 80-class head from pretrained weights
        model = SegmentationModel(cfg, ch=self.data["channels"], verbose=verbose and RANK == -1)

        # Attach hyperparameters (required by v8DetectionLoss)
        model.args = self.args

        # Load pretrained weights if specified (from --pretrained CLI arg)
        pretrained_path = getattr(self, "pretrained_path", None)
        if pretrained_path and Path(pretrained_path).exists():
            LOGGER.info(f"Loading pretrained weights from {pretrained_path}")
            import torch

            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            model.load(ckpt, verbose=verbose)
        elif weights:
            model.load(weights)

        model.depth_weight = self.depth_weight
        model.use_gradnorm = self.use_gradnorm
        return model

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer after freezing seg head to prevent it from being updated."""
        # Freeze seg head permanently (do not train)
        for n, p in model.named_parameters():
            if "seg" in n or "cv3" in n or "cv4" in n or "cv5" in n or "proto" in n:
                p.requires_grad = False
        LOGGER.info("Frozen seg head (cv3/cv4/cv5/proto) - excluded from optimizer")
        return super().build_optimizer(model, name, lr, momentum, decay, iterations)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build DepthSegmentDataset for multi-task training."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return DepthSegmentDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or (mode == "val"),
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task="segment",
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def validate(self):
        """Validate model, ensuring EMA model uses DepthSegmentationLoss."""
        if self.ema and self.ema.ema:
            self.ema.ema.depth_weight = self.depth_weight
            self.ema.ema.use_gradnorm = self.use_gradnorm
            if getattr(self.ema.ema, "criterion", None) is None:
                self.ema.ema.criterion = self.ema.ema.init_criterion()
        return super().validate()

    def get_validator(self):
        """Return multi-task validator."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "depth_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
