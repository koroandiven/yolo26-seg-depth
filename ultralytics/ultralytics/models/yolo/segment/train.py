# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.data.depth_dataset import DepthSegmentDataset
from ultralytics.models import yolo
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.metrics import DepthMetric
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DepthSegmentValidator(SegmentationValidator):
    """Validator that keeps enough confusion-matrix rows for COCO-class predictions
    even when the validation dataset has fewer classes (e.g. NYU depth data)."""

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics with COCO-scale confusion matrix and depth metrics."""
        super().init_metrics(model)
        # The model predicts COCO 80 classes but validation data may have fewer.
        # Expand confusion matrix to accommodate all COCO predictions.
        if self.nc < 80:
            self.nc = 80
            from ultralytics.utils.metrics import ConfusionMatrix

            self.confusion_matrix = ConfusionMatrix(
                names={i: str(i) for i in range(self.nc)}, task="detect"
            )
        # Depth estimation metrics
        self.depth_metric = DepthMetric()
        self.source_depth_metrics = {}
        self.source_depth_losses = {}

    def _prepare_pred(self, pred):
        # No-op to keep compatibility; depth preds are handled in update_metrics
        return pred

    def postprocess(self, preds):
        """Post-process predictions and preserve depth for metric update."""
        # Extract depth before NMS/postprocess strips it (preds is ((y, proto), preds_dict))
        self._last_depth_pred = None
        if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict):
            self._last_depth_pred = preds[1].get("depth")
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        """Update segmentation and depth metrics."""
        super().update_metrics(preds, batch)
        # Update depth metrics if depth prediction was captured in postprocess
        depth_pred = getattr(self, "_last_depth_pred", None)
        if depth_pred is not None:
            depth_target = batch.get("depth")
            if depth_target is not None:
                device = depth_pred.device
                depth_target = depth_target.to(device)
                # Handle shape mismatch: depth_pred is (B,1,H,W), depth_target is (B,H,W)
                if depth_pred.dim() == 4 and depth_target.dim() == 3:
                    depth_pred = depth_pred.squeeze(1)
                elif depth_pred.dim() == 4 and depth_target.dim() == 4:
                    depth_target = depth_target.squeeze(1)
                # Resize depth_target to match depth_pred if sizes differ (rect mode)
                if depth_pred.shape != depth_target.shape:
                    import torch.nn.functional as F

                    depth_target = F.interpolate(
                        depth_target.unsqueeze(1),
                        size=depth_pred.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                valid_mask = depth_target > 0
                if valid_mask.any():
                    self.depth_metric.update(depth_pred[valid_mask], depth_target[valid_mask])
                # Per-source depth metrics for multi-source training
                source_tags = batch.get("source_tag")
                if source_tags is not None and valid_mask.any():
                    bsz = depth_pred.shape[0]
                    for b in range(bsz):
                        tag = source_tags[b] if b < len(source_tags) else "unknown"
                        vm_b = valid_mask[b]
                        if vm_b.any():
                            if tag not in self.source_depth_metrics:
                                self.source_depth_metrics[tag] = DepthMetric()
                            self.source_depth_metrics[tag].update(
                                depth_pred[b][vm_b], depth_target[b][vm_b]
                            )

    def get_desc(self):
        """Return description of validation metrics."""
        desc = super().get_desc()
        if hasattr(self, "depth_metric"):
            depth_results = self.depth_metric.compute()
            desc += (
                f" | depth_abs_rel:{depth_results['abs_rel']:.3f}"
                f" depth_rmse:{depth_results['rmse']:.3f}"
                f" depth_d1:{depth_results.get('delta1', 0.0):.3f}"
            )
            # Per-source depth metrics for multi-source training
            for tag, metric in getattr(self, "source_depth_metrics", {}).items():
                src_res = metric.compute()
                desc += f" | {tag}_ar:{src_res['abs_rel']:.3f} {tag}_rm:{src_res['rmse']:.3f}"
        return desc

    def get_depth_source_metrics(self):
        """Return per-source depth metrics dict for logging."""
        results = {}
        for tag, metric in getattr(self, "source_depth_metrics", {}).items():
            src_res = metric.compute()
            results[f"depth_abs_rel_{tag}"] = src_res["abs_rel"]
            results[f"depth_rmse_{tag}"] = src_res["rmse"]
            results[f"depth_delta1_{tag}"] = src_res.get("delta1", 0.0)
        return results

    def get_stats(self):
        """Return validation stats with global and per-source depth metrics."""
        stats = super().get_stats()
        if hasattr(self, "depth_metric"):
            depth_results = self.depth_metric.compute()
            stats.update({
                "depth/abs_rel": depth_results["abs_rel"],
                "depth/rmse": depth_results["rmse"],
                "depth/silog": depth_results["silog"],
                "depth/rmse_log": depth_results["rmse_log"],
                "depth/sq_rel": depth_results["sq_rel"],
                "depth/delta1": depth_results["delta1"],
                "depth/delta2": depth_results["delta2"],
                "depth/delta3": depth_results["delta3"],
            })
        for tag, metric in getattr(self, "source_depth_metrics", {}).items():
            src_res = metric.compute()
            stats[f"depth/{tag}/abs_rel"] = src_res["abs_rel"]
            stats[f"depth/{tag}/rmse"] = src_res["rmse"]
            stats[f"depth/{tag}/delta1"] = src_res["delta1"]
        return stats


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
        self.freeze_seg = getattr(self.args, "freeze_seg", True)

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
        model.freeze_seg = self.freeze_seg
        # Do NOT pre-initialize criterion here - model may still be on CPU.
        # Lazy initialization in BaseModel.loss() ensures correct device.
        return model

    def set_model_attributes(self):
        """Preserve COCO model.names (80 classes) instead of overriding with data YAML names.

        The default BaseTrainer behaviour overwrites model.names with self.data["names"],
        which shrinks the confusion matrix to the validation dataset's class count (e.g.
        NYU 10 classes) and causes IndexError when COCO-pretrained models predict classes
        outside that range.
        """
        pass

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer after freezing seg head detection params, keep depth components trainable."""
        import torch.nn as nn

        seg_head = model.model[-1] if hasattr(model, "model") else None
        if seg_head is not None:
            # Freeze seg detection/segmentation BN running stats to preserve COCO pretraining,
            # but allow depth-branch BN stats to update to the target domain (NYU/KITTI).
            for n, m in seg_head.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    if "depth" in n or "task_attention.depth_branch" in n:
                        m.train()  # allow depth BN running stats to adapt to new domain
                    else:
                        m.eval()   # freeze seg BN running stats
            # Only freeze seg detection/segmentation params (NOT depth components).
            # Keep task_attention.seg_branch frozen to preserve the COCO seg path.
            for n, p in seg_head.named_parameters():
                if "depth" in n or "task_attention.depth_branch" in n:
                    p.requires_grad = True  # depth components always trainable
                else:
                    p.requires_grad = False  # cv2/cv3/cv4/cv5/proto/dfl frozen
        LOGGER.info("Frozen seg head detection params, depth BN adaptive, depth components trainable")
        return super().build_optimizer(model, name, lr, momentum, decay, iterations)

    def _model_train(self):
        """Persist seg-head BN freeze across epochs after base _model_train resets everything."""
        super()._model_train()
        import torch.nn as nn

        seg_head = self.model.model[-1] if hasattr(self.model, "model") else None
        if seg_head is not None:
            for n, m in seg_head.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    if "depth" not in n and "task_attention.depth_branch" not in n:
                        m.eval()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build DepthSegmentDataset for multi-task training.

        Supports multi-source mixing when ``self.data`` contains a ``sources``
        dict (e.g. NYU + KITTI). Each source can have its own ``depth_max``
        for clipping; ``depth_scale`` is unified from the model YAML.
        """
        import torch.utils.data as torch_data
        from pathlib import Path

        # model may still be a string during early init; fall back to stride=32
        try:
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
        except (AttributeError, TypeError):
            gs = 32
        depth_scale = getattr(self.model, "depth_scale", 100.0)

        sources = self.data.get("sources")
        if sources:
            base_path = Path(self.data.get("path", ""))
            datasets = []
            # In multi-source mixing, val images come from datasets with very
            # different aspect ratios (NYU 4:3 indoor vs KITTI ~3:1 outdoor).
            # ``rect`` batching would produce per-batch sizes that differ
            # between sub-datasets, breaking the ConcatDataset collate path
            # (it stacks depth maps of different HxW). Force square LetterBox.
            use_rect = False
            for tag, cfg in sources.items():
                rel_path = cfg.get(mode)
                if not rel_path:
                    continue
                src_path = str((base_path / rel_path).resolve())
                ds = DepthSegmentDataset(
                    img_path=src_path,
                    imgsz=self.args.imgsz,
                    batch_size=batch,
                    augment=mode == "train",
                    hyp=self.args,
                    rect=use_rect,
                    cache=self.args.cache or None,
                    single_cls=self.args.single_cls or False,
                    stride=gs,
                    pad=0.0 if mode == "train" else 0.5,
                    prefix=f"{mode}/{tag}: ",
                    task="segment",
                    data=self.data,
                    fraction=self.args.fraction if mode == "train" else 1.0,
                    depth_scale=depth_scale,
                    depth_max=cfg.get("depth_max", depth_scale),
                    depth_norm_max=cfg.get("depth_norm_max", None),
                    source_tag=tag,
                )
                datasets.append(ds)
            if len(datasets) == 0:
                raise ValueError(f"No valid sources found for mode='{mode}' in data YAML.")
            if len(datasets) == 1:
                return datasets[0]
            concat_ds = torch_data.ConcatDataset(datasets)
            concat_ds.collate_fn = DepthSegmentDataset.collate_fn
            # Compute sample weights for balanced multi-source sampling
            # Strategy: inverse frequency weighting (smaller dataset gets higher weight)
            source_lengths = [len(ds) for ds in datasets]
            total_len = sum(source_lengths)
            weights = []
            for ds in datasets:
                n = len(ds)
                w = total_len / (len(datasets) * n) if n > 0 else 0.0
                weights.extend([w] * n)
            concat_ds.sample_weights = weights
            return concat_ds

        # Single-source fallback
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
            depth_scale=depth_scale,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct dataloader, using WeightedRandomSampler for balanced multi-source training."""
        from ultralytics.data.build import build_dataloader
        import numpy as np

        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False

        # Use WeightedRandomSampler for balanced multi-source training
        sampler = None
        sample_weights = getattr(dataset, "sample_weights", None)
        if mode == "train" and sample_weights is not None and len(sample_weights) == len(dataset):
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
            shuffle = False  # sampler handles shuffling

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            sampler=sampler,
        )

    def validate(self):
        """Validate model, ensuring EMA model uses DepthSegmentationLoss."""
        if self.ema and self.ema.ema:
            self.ema.ema.depth_weight = self.depth_weight
            self.ema.ema.use_gradnorm = self.use_gradnorm
            self.ema.ema.freeze_seg = self.freeze_seg
            self.ema.ema.depth_scale = getattr(self.model, "depth_scale", 20.0)
            if getattr(self.ema.ema, "criterion", None) is None:
                self.ema.ema.criterion = self.ema.ema.init_criterion()
            elif hasattr(self.ema.ema.criterion, "freeze_seg"):
                self.ema.ema.criterion.freeze_seg = self.freeze_seg
        return super().validate()

    def get_validator(self):
        """Return multi-task validator."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "depth_loss"
        return DepthSegmentValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _get_dataset_labels(self):
        """Get labels from dataset, handling ConcatDataset from multi-source mixing."""
        dataset = self.train_loader.dataset if hasattr(self, "train_loader") else None
        if dataset is None:
            return []
        if hasattr(dataset, "datasets"):
            # ConcatDataset: collect labels from all sub-datasets
            labels = []
            for ds in dataset.datasets:
                labels.extend(getattr(ds, "labels", []))
            return labels
        return getattr(dataset, "labels", [])

    def plot_training_labels(self):
        """Create a labeled training plot, supporting ConcatDataset."""
        import numpy as np
        from ultralytics.utils.plotting import plot_labels

        labels = self._get_dataset_labels()
        if not labels:
            return
        boxes = np.concatenate([lb["bboxes"] for lb in labels], 0)
        cls = np.concatenate([lb["cls"] for lb in labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size, supporting ConcatDataset."""
        from ultralytics.utils import override_configs

        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)

        if hasattr(train_dataset, "datasets"):
            labels = []
            for ds in train_dataset.datasets:
                labels.extend(getattr(ds, "labels", []))
            max_num_obj = max(len(label["cls"]) for label in labels) * 4 if labels else 16
            n = len(train_dataset)
        else:
            max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
            n = len(train_dataset)

        del train_dataset
        return super().auto_batch(max_num_obj, dataset_size=n)
