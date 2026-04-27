#!/usr/bin/env python3
"""
YOLO26 Training Script
基于 Ultralytics 框架的 YOLO26 训练器

用法:
    python yolo26_train.py --data coco8.yaml --epochs 100 --device 0
    python yolo26_train.py --model yolo26s-seg.yaml --data my_data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.callbacks import add_integration_callbacks


class YOLO26Trainer:
    """YOLO26 训练器封装类

    提供简洁的 API 接口进行 YOLO26 模型训练

    Attributes:
        model: YOLO 模型实例
        model_path: 模型配置文件路径
        data: 数据集配置文件路径
        device: 训练设备
    """

    def __init__(
        self,
        model: str | Path = "yolo26s-seg.yaml",
        data: str | Path = "coco8-seg.yaml",
        task: str = "segment",
    ):
        """初始化 YOLO26 训练器

        Args:
            model: 模型配置路径 (yaml) 或预训练权重路径 (pt)
            data: 数据集配置文件路径 (yaml)
            task: 任务类型 (detect/segment/pose/obb/classify)
        """
        self.model_path = str(model)
        self.data_path = str(data)
        self.task = task

        self._model: Optional[YOLO] = None
        self._results: Any = None

    def build_model(self) -> YOLO:
        """构建 YOLO 模型

        Returns:
            YOLO: YOLO 模型实例
        """
        self._model = YOLO(self.model_path)
        return self._model

    def train(
        self,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        device: str = "",
        workers: int = 8,
        optimizer: str = "auto",
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: float = 3.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        mosaic: float = 1.0,
        mixup: float = 0.0,
        copy_paste: float = 0.0,
        augment: bool = True,
        flipud: float = 0.0,
        fliplr: float = 0.5,
        close_mosaic: int = 10,
        resume: bool = False,
        amp: bool = True,
        cache: bool = False,
        patience: int = 50,
        save: bool = True,
        save_period: int = -1,
        val: bool = True,
        verbose: bool = True,
        project: str = "runs/train",
        name: str = "yolo26s-seg-exp",
        exist_ok: bool = False,
        pretrained: bool = True,
        deterministic: bool = True,
        single_cls: bool = False,
        rect: bool = False,
        cos_lr: bool = False,
        resume_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """训练 YOLO26 模型

        Args:
            epochs: 训练轮数
            batch: 批次大小
            imgsz: 输入图像尺寸
            device: 训练设备 ('', 'cpu', '0', '0,1,2,3')
            workers: 数据加载线程数
            optimizer: 优化器 ('SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto')
            lr0: 初始学习率
            lrf: 最终学习率因子
            momentum: SGD 动量 / Adam beta1
            weight_decay: 权重衰减
            warmup_epochs: 预热轮数
            warmup_momentum: 预热动量
            warmup_bias_lr: 预热偏置学习率
            box: 边框损失增益
            cls: 分类损失增益
            dfl: DFL 损失增益
            mosaic: Mosaic 增强概率
            mixup: MixUp 增强概率
            copy_paste: Copy-Paste 增强概率
            augment: 是否启用增强
            flipud: 上下翻转概率
            fliplr: 左右翻转概率
            close_mosaic: 在最后 N 轮禁用 mosaic 增强
            resume: 是否从上次中断处继续训练
            amp: 是否启用自动混合精度 (AMP) 训练
            cache: 是否缓存图像 (True/ram/disk)
            patience: 早停耐心值
            save: 是否保存模型
            save_period: 模型保存周期 (-1 仅保存最后和最佳)
            val: 是否在训练期间验证
            verbose: 是否详细输出
            project: 项目保存根目录
            name: 实验名称
            exist_ok: 是否允许已存在同名实验
            pretrained: 是否使用预训练权重
            deterministic: 是否使用确定性操作
            single_cls: 是否将多类数据训练为单类
            rect: 是否使用矩形训练
            cos_lr: 是否使用余弦学习率调度
            resume_path: 从指定路径恢复训练
            **kwargs: 其他训练参数

        Returns:
            Dict: 训练结果字典，包含指标和模型路径

        Example:
            >>> trainer = YOLO26Trainer("yolo26s-seg.yaml", "coco8-seg.yaml")
            >>> results = trainer.train(epochs=100, batch=16, device="0")
            >>> print(results.metrics)
        """
        if self._model is None:
            self.build_model()

        train_args = {
            "data": self.data_path,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "workers": workers,
            "optimizer": optimizer,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "warmup_momentum": warmup_momentum,
            "warmup_bias_lr": warmup_bias_lr,
            "box": box,
            "cls": cls,
            "dfl": dfl,
            "mosaic": mosaic,
            "mixup": mixup,
            "copy_paste": copy_paste,
            "augment": augment,
            "flipud": flipud,
            "fliplr": fliplr,
            "close_mosaic": close_mosaic,
            "resume": resume,
            "amp": amp,
            "cache": cache,
            "patience": patience,
            "save": save,
            "save_period": save_period,
            "val": val,
            "verbose": verbose,
            "project": project,
            "name": name,
            "exist_ok": exist_ok,
            "pretrained": pretrained,
            "deterministic": deterministic,
            "single_cls": single_cls,
            "rect": rect,
            "cos_lr": cos_lr,
        }

        train_args.update(kwargs)

        if resume_path:
            train_args["resume"] = resume_path

        LOGGER.info(f"开始训练 YOLO26 模型...")
        LOGGER.info(f"模型配置: {self.model_path}")
        LOGGER.info(f"数据集配置: {self.data_path}")
        LOGGER.info(
            f"训练参数: epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}"
        )

        self._results = self._model.train(**train_args)

        return self._results

    def export_model(
        self,
        format: str = "onnx",
        imgsz: int | List[int] = 640,
        keras: bool = False,
        optimize: bool = False,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int | None = None,
        workspace: float = 4.0,
        nms: bool = False,
        batch: int = 1,
        device: str = "",
        **kwargs,
    ) -> str:
        """导出训练好的模型

        Args:
            format: 导出格式 (onnx/torchscript/pytorch/engine/coreml/tflite/etc.)
            imgsz: 导出模型输入尺寸
            keras: 是否导出为 Keras 格式
            optimize: 是否优化 TorchScript 模型
            half: 是否导出为 FP16 精度
            int8: 是否导出为 INT8 量化模型
            dynamic: 是否支持动态输入尺寸
            simplify: 是否简化 ONNX 模型
            opset: ONNX opset 版本
            workspace: TensorRT 工作空间大小 (GB)
            nms: 是否在导出模型中添加 NMS
            batch: 导出模型批次大小
            device: 导出设备
            **kwargs: 其他导出参数

        Returns:
            str: 导出模型路径

        Example:
            >>> trainer = YOLO26Trainer("yolo26s-seg.yaml", "coco8-seg.yaml")
            >>> trainer.train(epochs=100)
            >>> trainer.export_model(format="engine", device="0")
        """
        if self._model is None:
            raise ValueError("模型未训练，请先调用 train() 方法")

        export_args = {
            "format": format,
            "imgsz": imgsz,
            "keras": keras,
            "optimize": optimize,
            "half": half,
            "int8": int8,
            "dynamic": dynamic,
            "simplify": simplify,
            "opset": opset,
            "workspace": workspace,
            "nms": nms,
            "batch": batch,
            "device": device,
        }
        export_args.update(kwargs)

        LOGGER.info(f"导出模型为 {format} 格式...")
        export_path = self._model.export(**export_args)
        return export_path

    @property
    def model(self) -> Optional[YOLO]:
        """获取 YOLO 模型实例"""
        return self._model

    @property
    def results(self) -> Any:
        """获取训练结果"""
        return self._results

    @property
    def best_model_path(self) -> str:
        """获取最佳模型路径"""
        if self._results is None:
            return ""
        return str(self._results.save_dir / "weights/best.pt")

    @property
    def last_model_path(self) -> str:
        """获取最后模型路径"""
        if self._results is None:
            return ""
        return str(self._results.save_dir / "weights/last.pt")


def train_with_yaml_config(
    model_yaml: str = "yolo26s-seg.yaml",
    data_yaml: str = "coco8-seg.yaml",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """使用 YAML 配置训练 YOLO26 (便捷函数)

    Args:
        model_yaml: 模型配置文件路径
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 输入图像尺寸
        device: 训练设备
        **kwargs: 其他训练参数

    Returns:
        Dict: 训练结果

    Example:
        >>> results = train_with_yaml_config(
        ...     model_yaml="yolo26s-seg.yaml",
        ...     data_yaml="coco8-seg.yaml",
        ...     epochs=100,
        ...     batch=16,
        ...     device="0"
        ... )
    """
    trainer = YOLO26Trainer(model=model_yaml, data=data_yaml)
    return trainer.train(
        epochs=epochs, batch=batch, imgsz=imgsz, device=device, **kwargs
    )


def train_from_pretrained(
    model_pt: str = "yolo26s-seg.pt",
    data_yaml: str = "coco8-seg.yaml",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """从预训练模型开始训练 (便捷函数)

    Args:
        model_pt: 预训练模型路径
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 输入图像尺寸
        device: 训练设备
        **kwargs: 其他训练参数

    Returns:
        Dict: 训练结果

    Example:
        >>> results = train_from_pretrained(
        ...     model_pt="yolo26s-seg.pt",
        ...     data_yaml="my_data.yaml",
        ...     epochs=50,
        ...     device="0"
        ... )
    """
    trainer = YOLO26Trainer(model=model_pt, data=data_yaml)
    return trainer.train(
        epochs=epochs, batch=batch, imgsz=imgsz, device=device, **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 训练脚本")

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s-seg.yaml",
        help="模型配置文件路径 (yaml) 或预训练权重 (pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco8-seg.yaml",
        help="数据集配置文件路径 (yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批次大小",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像尺寸",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="训练设备 (cpu/0/0,1,2,3)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="项目保存目录",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo26s-seg-exp",
        help="实验名称",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="优化器 (SGD/Adam/AdamW/auto)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="初始学习率",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处继续训练",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="使用预训练权重",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="启用自动混合精度训练",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="早停耐心值",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="模型保存周期",
    )

    args = parser.parse_args()

    trainer = YOLO26Trainer(model=args.model, data=args.data)

    trainer.train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        resume=args.resume,
        pretrained=args.pretrained,
        amp=args.amp,
        patience=args.patience,
        save_period=args.save_period,
    )

    print(f"\n训练完成!")
    print(f"最佳模型: {trainer.best_model_path}")
    print(f"最终模型: {trainer.last_model_path}")
