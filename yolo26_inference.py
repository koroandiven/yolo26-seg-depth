#!/usr/bin/env python3
"""
YOLO26 Inference Library
基于 Ultralytics 框架的 YOLO26 推理库，支持检测和分割模型

作者: YOLO26 Team
版本: 1.0.0
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple, Generator
from PIL import Image

from ultralytics import YOLO
from ultralytics.engine.results import Results


class YOLO26Inference:
    """YOLO26 推理库封装类

    提供简洁的 API 接口进行 YOLO26 目标检测和分割推理

    Attributes:
        model: YOLO 模型实例
        model_path: 模型文件路径
        task: 任务类型 (detect/segment/pose/obb)
        device: 推理设备
        names: 类别名称字典

    Example:
        >>> # 基础用法
        >>> infer = YOLO26Inference("yolo26s-seg.pt")
        >>> results = infer.predict("test.jpg")
        >>> for result in results:
        ...     print(result.boxes.xyxy, result.masks)

        >>> # 检测模型
        >>> infer = YOLO26Inference("yolo26s.pt")
        >>> results = infer.predict("test.jpg", conf=0.5)

        >>> # 批量推理
        >>> infer = YOLO26Inference("yolo26s-seg.pt")
        >>> results = infer.predict_batch(["img1.jpg", "img2.jpg"])

        >>> # 视频推理
        >>> infer = YOLO26Inference("yolo26s.pt")
        >>> for result in infer.predict_video("video.mp4"):
        ...     # 每帧处理
        ...     pass

        >>> # Webcam 实时推理
        >>> infer = YOLO26Inference("yolo26s.pt")
        >>> infer.predict_webcam(0, show=True)
    """

    SUPPORTED_FORMATS = {
        "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        "video": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"],
    }

    def __init__(
        self,
        model_path: str | Path,
        task: str | None = None,
        device: str = "",
        verbose: bool = False,
    ):
        """初始化 YOLO26 推理器

        Args:
            model_path: 模型路径 (.pt, .onnx, .engine 等)
            task: 任务类型，可选: detect, segment, pose, obb。如果为 None，则自动从模型推断
            device: 推理设备，如 'cpu', '0', 'cuda:0'。空字符串则自动选择
            verbose: 是否打印详细信息
        """
        self.model_path = str(model_path)
        self._model = None
        self._device = device
        self._task = task
        self._names = None
        self._load_model(verbose=verbose)

    def _load_model(self, verbose: bool = False) -> None:
        """加载 YOLO 模型"""
        self._model = YOLO(self.model_path)
        if self._task is None:
            self._task = self._model.task
        self._names = self._model.names
        if verbose:
            print(f"模型加载成功: {self.model_path}")
            print(f"任务类型: {self._task}")
            print(f"类别数量: {len(self._names)}")

    @property
    def model(self) -> YOLO:
        """获取 YOLO 模型实例"""
        return self._model

    @property
    def task(self) -> str:
        """获取任务类型"""
        return self._task

    @property
    def names(self) -> Dict[int, str]:
        """获取类别名称"""
        return self._names

    @property
    def device(self) -> torch.device:
        """获取推理设备"""
        return self._model.device if self._model else None

    def predict(
        self,
        source: str | Path | int | Image.Image | np.ndarray | torch.Tensor | List,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int | Tuple[int, int] = 640,
        save: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
        show: bool = False,
        show_labels: bool = True,
        show_conf: bool = True,
        show_boxes: bool = True,
        line_width: int = 2,
        augment: bool = False,
        half: bool = False,
        dnn: bool = False,
        device: str | None = None,
        classes: List[int] | int | None = None,
        retina_masks: bool = False,
        embed: List[int] | None = None,
        verbose: bool = True,
        stream: bool = False,
        project: str | Path = "runs/detect",
        name: str = "predict",
    ) -> List[Results] | Generator[Results, None, None]:
        """目标检测/分割推理

        Args:
            source: 输入源，支持:
                - 图像路径: "image.jpg"
                - 目录路径: "images/"
                - 视频路径: "video.mp4"
                - Webcam ID: 0, 1, ...
                - PIL Image
                - numpy array (H, W, 3)
                - torch tensor (C, H, W) or (B, C, H, W)
                - 路径列表或 glob 模式
            conf: 置信度阈值 [0.0-1.0]
            iou: NMS IoU 阈值 [0.0-1.0]
            imgsz: 推理图片尺寸，可以是 int 或 (height, width)
            save: 是否保存推理结果图片
            save_txt: 是否保存 YOLO txt 格式结果
            save_conf: 是否在 txt 中保存置信度
            save_crop: 是否保存裁剪的目标区域
            show: 是否显示推理结果窗口
            show_labels: 是否在结果上显示标签
            show_conf: 是否在结果上显示置信度
            show_boxes: 是否在结果上显示边界框
            line_width: 绘制线条宽度
            augment: 是否使用 TTA (Test Time Augmentation)
            half: 是否使用 FP16 推理
            dnn: 是否使用 OpenCV DNN 后端
            device: 推理设备，覆盖初始化时的设置
            classes: 只检测特定类别，如 [0, 1, 2] 或 0
            retina_masks: 是否使用高分辨率分割掩码
            embed: 提取特征层索引列表
            verbose: 是否打印详细信息
            stream: 如果为 True，返回生成器；否则返回列表
            project: 保存项目的根目录
            name: 保存子目录名称

        Returns:
            如果 stream=False: List[Results]，推理结果列表
            如果 stream=True: Generator[Results, None, None]，推理结果生成器

        Example:
            >>> infer = YOLO26Inference("yolo26s.pt")
            >>> results = infer.predict("test.jpg", conf=0.5)
            >>> for r in results:
            ...     boxes = r.boxes
            ...     if r.masks is not None:
            ...         masks = r.masks.data
        """
        kwargs = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "save": save,
            "save_txt": save_txt,
            "save_conf": save_conf,
            "save_crop": save_crop,
            "show": show,
            "show_labels": show_labels,
            "show_conf": show_conf,
            "show_boxes": show_boxes,
            "line_width": line_width,
            "augment": augment,
            "half": half,
            "dnn": dnn,
            "retina_masks": retina_masks,
            "embed": embed,
            "verbose": verbose,
            "project": project,
            "name": name,
        }

        if device is not None:
            kwargs["device"] = device
        elif self._device:
            kwargs["device"] = self._device

        if classes is not None:
            kwargs["classes"] = classes

        return self._model.predict(
            source=source,
            stream=stream,
            **kwargs,
        )

    def predict_batch(
        self, sources: List[str | Path | np.ndarray], **kwargs
    ) -> List[Results]:
        """批量推理多张图片

        Args:
            sources: 图片路径或 numpy 数组列表
            **kwargs: predict() 的其他参数

        Returns:
            List[Results]: 所有图片的推理结果列表

        Example:
            >>> infer = YOLO26Inference("yolo26s-seg.pt")
            >>> results = infer.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
        """
        return self.predict(source=sources, stream=False, **kwargs)

    def predict_single(
        self,
        source: str | Path | np.ndarray,
        save_path: str | Path | None = None,
        **kwargs,
    ) -> Results:
        """推理单张图片，返回单个结果

        Args:
            source: 图片路径或 numpy 数组
            save_path: 图像保存路径，为 None 则不保存
            **kwargs: predict() 的其他参数

        Returns:
            Results: 单张图片的推理结果

        Example:
            >>> infer = YOLO26Inference("yolo26s.pt")
            >>> result = infer.predict_single("test.jpg")
            >>> box = result.boxes[0] if len(result.boxes) > 0 else None
        """
        results = self.predict(source=source, stream=False, **kwargs)
        result = results[0] if results else None

        if result is not None and save_path is not None:
            plotted = result.plot()
            if plotted is not None:
                plotted_bgr = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), plotted_bgr)

        return result

    def predict_video(
        self, source: str | Path | int, save_path: str | Path | None = None, **kwargs
    ) -> Generator[Tuple[int, Results], None, None]:
        """视频推理，生成器方式逐帧处理

        Args:
            source: 视频路径或 webcam ID
            save_path: 保存路径，如为 None 则不保存
            **kwargs: predict() 的其他参数

        Yields:
            Tuple[int, Results]: (帧索引, 推理结果)

        Example:
            >>> infer = YOLO26Inference("yolo26s.pt")
            >>> for frame_idx, result in infer.predict_video("video.mp4"):
            ...     if frame_idx % 100 == 0:
            ...         print(f"处理到第 {frame_idx} 帧")
        """
        if save_path is not None:
            kwargs["save"] = True
            kwargs["project"] = str(Path(save_path).parent)
            kwargs["name"] = Path(save_path).stem

        kwargs["stream"] = True

        frame_idx = 0
        for result in self.predict(source=source, **kwargs):
            yield frame_idx, result
            frame_idx += 1

    def predict_webcam(
        self,
        camera_id: int = 0,
        show: bool = True,
        save_path: str | Path | None = None,
        window_name: str = "YOLO26 Webcam",
        **kwargs,
    ) -> None:
        """Webcam 实时推理

        Args:
            camera_id: 摄像头 ID，默认为 0
            show: 是否显示推理结果窗口
            save_path: 视频保存路径，如为 None 则不保存
            window_name: 窗口名称
            **kwargs: predict() 的其他参数

        Example:
            >>> infer = YOLO26Inference("yolo26s.pt")
            >>> infer.predict_webcam(0, show=True, conf=0.5)
        """
        if save_path is not None:
            kwargs["save"] = True
            kwargs["project"] = str(Path(save_path).parent)
            kwargs["name"] = Path(save_path).stem

        kwargs["show"] = show
        kwargs["stream"] = True

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头 {camera_id}")

        try:
            frame_idx = 0
            for result in self.predict(source=camera_id, **kwargs):
                if show:
                    if result.plotted_img is not None:
                        cv2.imshow(window_name, result.plotted_img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                frame_idx += 1
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()

    def predict_directory(
        self,
        directory: str | Path,
        extensions: List[str] | None = None,
        recursive: bool = False,
        **kwargs,
    ) -> Dict[str, Results]:
        """对目录下所有图片进行推理

        Args:
            directory: 图片目录路径
            extensions: 要处理的文件扩展名列表，如 ['.jpg', '.png']
            recursive: 是否递归处理子目录
            **kwargs: predict() 的其他参数

        Returns:
            Dict[str, Results]: 文件路径到推理结果的字典

        Example:
            >>> infer = YOLO26Inference("yolo26s-seg.pt")
            >>> results = infer.predict_directory("images/", recursive=True)
            >>> for path, result in results.items():
            ...     print(f"{path}: {len(result.boxes)} detections")
        """
        directory = Path(directory)
        if extensions is None:
            extensions = self.SUPPORTED_FORMATS["image"]

        if recursive:
            image_paths = []
            for ext in extensions:
                image_paths.extend(directory.rglob(f"*{ext}"))
                image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            image_paths = []
            for ext in extensions:
                image_paths.extend(directory.glob(f"*{ext}"))
                image_paths.extend(directory.glob(f"*{ext.upper()}"))

        image_paths = sorted(set(image_paths))

        all_results = {}
        for img_path in image_paths:
            result = self.predict_single(str(img_path), verbose=False, **kwargs)
            all_results[str(img_path)] = result

        return all_results

    def get_boxes_xyxy(self, result: Results) -> np.ndarray:
        """从结果中提取边界框坐标 (xyxy 格式)

        Args:
            result: 推理结果对象

        Returns:
            np.ndarray: 边界框坐标，形状为 (N, 4)，每行为 [x1, y1, x2, y2]
        """
        if result.boxes is None or len(result.boxes) == 0:
            return np.array([])
        return result.boxes.xyxy.cpu().numpy()

    def get_boxes_xywh(self, result: Results) -> np.ndarray:
        """从结果中提取边界框坐标 (xywh 格式)

        Args:
            result: 推理结果对象

        Returns:
            np.ndarray: 边界框坐标，形状为 (N, 4)，每行为 [x, y, w, h]
        """
        if result.boxes is None or len(result.boxes) == 0:
            return np.array([])
        return result.boxes.xywh.cpu().numpy()

    def get_confidences(self, result: Results) -> np.ndarray:
        """从结果中提取置信度

        Args:
            result: 推理结果对象

        Returns:
            np.ndarray: 置信度数组，形状为 (N,)
        """
        if result.boxes is None or len(result.boxes) == 0:
            return np.array([])
        return result.boxes.conf.cpu().numpy()

    def get_class_ids(self, result: Results) -> np.ndarray:
        """从结果中提取类别 ID

        Args:
            result: 推理结果对象

        Returns:
            np.ndarray: 类别 ID 数组，形状为 (N,)
        """
        if result.boxes is None or len(result.boxes) == 0:
            return np.array([])
        return result.boxes.cls.cpu().numpy().astype(int)

    def get_class_names(self, result: Results) -> List[str]:
        """从结果中提取类别名称

        Args:
            result: 推理结果对象

        Returns:
            List[str]: 类别名称列表
        """
        class_ids = self.get_class_ids(result)
        return [self.names[int(cid)] for cid in class_ids]

    def get_masks(self, result: Results) -> np.ndarray | None:
        """从结果中提取分割掩码

        Args:
            result: 推理结果对象

        Returns:
            np.ndarray | None: 分割掩码，形状为 (N, H, W)，或 None
        """
        if result.masks is None:
            return None
        return result.masks.data.cpu().numpy()

    def get_masks_polygons(self, result: Results) -> List[np.ndarray] | None:
        """从结果中提取分割掩码的多边形坐标

        Args:
            result: 推理结果对象

        Returns:
            List[np.ndarray] | None: 每个掩码的多边形坐标列表
        """
        if result.masks is None:
            return None
        return result.masks.xyn

    def get_depth(self, result: Results) -> np.ndarray | None:
        """从结果中提取深度图

        Args:
            result: 推理结果对象

        Returns:
            np.ndarray | None: 深度图，形状为 (H, W)
        """
        if hasattr(result, "depth") and result.depth is not None:
            return result.depth
        return None

    def to_dict(self, result: Results) -> Dict[str, Any]:
        """将推理结果转换为字典格式

        Args:
            result: 推理结果对象

        Returns:
            Dict: 包含所有检测信息的字典
        """
        output = {
            "boxes_xyxy": self.get_boxes_xyxy(result),
            "boxes_xywh": self.get_boxes_xywh(result),
            "confidences": self.get_confidences(result),
            "class_ids": self.get_class_ids(result),
            "class_names": self.get_class_names(result),
        }

        if self.task == "segment":
            output["masks"] = self.get_masks(result)
            output["masks_polygons"] = self.get_masks_polygons(result)

        # Add depth output if available
        depth = self.get_depth(result)
        if depth is not None:
            output["depth"] = depth

        return output

    def print_summary(self, result: Results) -> None:
        """打印检测结果摘要

        Args:
            result: 推理结果对象
        """
        n_boxes = len(result.boxes) if result.boxes is not None else 0
        n_masks = len(result.masks) if result.masks is not None else 0

        print(f"\n{'=' * 50}")
        print(f"YOLO26 推理结果摘要")
        print(f"{'=' * 50}")
        print(f"任务类型: {self.task}")
        print(f"检测目标数量: {n_boxes}")

        if n_boxes > 0:
            print(f"\n类别分布:")
            class_ids = self.get_class_ids(result)
            class_names = self.get_class_names(result)
            confidences = self.get_confidences(result)

            unique_classes = {}
            for cls_id, cls_name in zip(class_ids, class_names):
                if cls_id not in unique_classes:
                    unique_classes[cls_id] = {
                        "name": cls_name,
                        "count": 0,
                        "avg_conf": 0,
                    }
                unique_classes[cls_id]["count"] += 1
                unique_classes[cls_id]["avg_conf"] += confidences[
                    np.where(class_ids == cls_id)[0][0]
                ]

            for cls_id, info in unique_classes.items():
                avg_conf = info["avg_conf"] / info["count"]
                print(
                    f"  - {info['name']}: {info['count']} 个 (平均置信度: {avg_conf:.3f})"
                )

        if self.task == "segment":
            print(f"分割掩码数量: {n_masks}")

        print(f"{'=' * 50}\n")

    def overlay_masks(
        self,
        image: np.ndarray,
        result: Results,
        alpha: float = 0.5,
        colors: List[Tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """在图像上叠加分割掩码

        Args:
            image: 输入图像 (H, W, 3) BGR 格式
            result: 推理结果对象
            alpha: 掩码透明度
            colors: 每个类别的颜色列表，如为 None 则自动生成

        Returns:
            np.ndarray: 叠加掩码后的图像
        """
        if result.masks is None:
            return image

        masks = self.get_masks(result)
        class_ids = self.get_class_ids(result)

        if colors is None:
            np.random.seed(42)
            colors = [
                (np.random.randint(50, 255) for _ in range(3))
                for _ in range(len(self.names))
            ]
            colors = [tuple(int(c) for c in color) for color in colors]

        output = image.copy()
        h, w = image.shape[:2]

        for mask, cls_id in zip(masks, class_ids):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color = colors[int(cls_id) % len(colors)]

            mask_bool = mask_resized > 0.5
            output[mask_bool] = (
                alpha * np.array(color) + (1 - alpha) * output[mask_bool]
            ).astype(np.uint8)

        return output

    def draw_detections(
        self,
        image: np.ndarray,
        result: Results,
        line_width: int = 2,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        box_color: Tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """在图像上绘制检测结果

        Args:
            image: 输入图像 (H, W, 3) BGR 格式
            result: 推理结果对象
            line_width: 边界框线条宽度
            text_scale: 文本缩放
            text_thickness: 文本厚度
            box_color: 边界框颜色，为 None 则使用类别颜色

        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        output = image.copy()
        boxes = self.get_boxes_xyxy(result)
        class_ids = self.get_class_ids(result)
        class_names = self.get_class_names(result)
        confidences = self.get_confidences(result)

        if box_color is None:
            np.random.seed(42)
            colors = [
                (np.random.randint(50, 255) for _ in range(3))
                for _ in range(len(self.names))
            ]
            colors = [tuple(int(c) for c in color) for color in colors]

        for box, cls_id, cls_name, conf in zip(
            boxes, class_ids, class_names, confidences
        ):
            x1, y1, x2, y2 = map(int, box)
            color = box_color or colors[int(cls_id) % len(colors)]

            cv2.rectangle(output, (x1, y1), (x2, y2), color, line_width)

            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )
            cv2.rectangle(output, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
            )

        return output


class YOLO26Detector(YOLO26Inference):
    """YOLO26 检测模型专用推理类"""

    def __init__(self, model_path: str | Path, device: str = "", verbose: bool = False):
        super().__init__(model_path, task="detect", device=device, verbose=verbose)


class YOLO26Segmentor(YOLO26Inference):
    """YOLO26 分割模型专用推理类"""

    def __init__(self, model_path: str | Path, device: str = "", verbose: bool = False):
        super().__init__(model_path, task="segment", device=device, verbose=verbose)


def create_inference(
    model_path: str | Path,
    task: str | None = None,
    device: str = "",
    verbose: bool = False,
) -> YOLO26Inference:
    """工厂函数：创建 YOLO26 推理器

    Args:
        model_path: 模型路径
        task: 任务类型，可选: detect, segment, pose, obb
        device: 推理设备
        verbose: 是否打印详细信息

    Returns:
        YOLO26Inference: 推理器实例

    Example:
        >>> # 创建检测推理器
        >>> detector = create_inference("yolo26s.pt", task="detect")
        >>>
        >>> # 创建分割推理器
        >>> segmentor = create_inference("yolo26s-seg.pt", task="segment")
    """
    if task == "detect":
        return YOLO26Detector(model_path, device, verbose)
    elif task == "segment":
        return YOLO26Segmentor(model_path, device, verbose)
    else:
        return YOLO26Inference(model_path, task, device, verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO26 推理库演示")
    parser.add_argument("--model", type=str, default="yolo26s-seg.pt", help="模型路径")
    parser.add_argument("--source", type=str, default="000046.jpg", help="输入源")
    parser.add_argument("--conf", type=float, default=0.7, help="置信度阈值")
    parser.add_argument("--device", type=str, default=0, help="推理设备")

    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    infer = YOLO26Inference(args.model, device=args.device, verbose=True)
    print(infer.names)
    print(f"\n对图片进行推理: {args.source}")
    result = infer.predict_single(
        args.source, conf=args.conf, verbose=True, save_path="./result.jpg"
    )

    infer.print_summary(result)

    detections = infer.to_dict(result)
    print(f"检测到 {len(detections['class_ids'])} 个目标")

    if infer.task == "segment" and detections["masks"] is not None:
        print(f"分割掩码形状: {detections['masks'].shape}")
