# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


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
        """Initialize model and replace with multi-task loss function."""
        model = SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)

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

        from ultralytics.utils.loss import DepthSegmentationLoss

        model.criterion = DepthSegmentationLoss(model, self.depth_weight, self.use_gradnorm)
        return model

    def get_validator(self):
        """Return multi-task validator."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "depth_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
