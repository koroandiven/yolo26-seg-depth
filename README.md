# YOLO26-Seg-Depth

基于 Ultralytics YOLO26 的多任务模型，在 COCO 预训练的实例分割模型基础上新增**单目深度估计**能力。训练时冻结分割路径以保护 COCO 性能，仅训练深度估计分支。

---

## 背景与目标

在 YOLO26 分割模型的共享 backbone/neck 基础上，新增一个深度估计分支，同时满足以下约束：

- **保留 COCO 分割性能**：backbone + 分割头 + `task_attention.seg_branch` 全部冻结
- **新增 NYU Depth V2 深度估计**：仅训练深度分支相关参数
- **解决深度估计中的两个核心问题**：
  1. 左右深度偏置（数据集本身的相机视角不对称导致）
  2. 深度边界模糊（物体边缘处深度过渡不清晰）

## 实验效果

| 实验 | fliplr | asymmetry | 列差 (R−L) | 关键改动 |
|------|--------|-----------|-----------|---------|
| exp14 | 0.5 | **0.431 m** | +0.332 m（右深） | 移除 bias map + FiLM 调制 + soft mask 引导 |
| exp15 | 0.3 | **0.356 m** | −0.344 m（左深） | fliplr 降低 |
| **exp16** | **0.4** | **0.317 m** | **−0.157 m** | **sweet spot，残余偏置最小** |

exp16 在左右对称性上比 exp14 **相对提升 26%**，是三者中残余偏置最小的配置。

## 项目结构

```
yolo26-seg-depth/
├── README.md                       # 本文档
├── plan_exp14.md                   # exp14-16 改进计划与诊断分析
├── yolo26_train_depth.py           # 训练入口脚本
├── yolo26_inference.py             # 推理库（图片/视频/摄像头）
├── diagnose_depth.py               # 左右对称性 + 边界相关性诊断
├── yolo26-seg.yaml                 # 基础分割模型配置
├── yolo26n.pt                      # COCO 预训练权重
├── nyu_yolo/                       # NYU Depth V2 数据集
│   ├── nyu_depth_seg.yaml          # 数据集配置
│   ├── images/                     # RGB 图像
│   ├── depths/                     # 16-bit PNG 深度图 (mm→m)
│   ├── labels/                     # YOLO 检测标签
│   └── segments/                   # 分割标签
├── ultralytics/                    # 修改后的 Ultralytics 算法库
│   └── ultralytics/
│       ├── nn/modules/head.py      # DepthSegment26 / MaskGuidedDepthDecoder
│       ├── nn/tasks.py             # 模型构建与损失衔接
│       ├── utils/loss.py           # 深度损失函数集合
│       ├── models/yolo/segment/train.py  # DepthSegmentTrainer/Validator
│       ├── data/depth_dataset.py   # DepthSegmentDataset（含深度翻转）
│       └── cfg/models/26/yolo26-seg-depth.yaml  # 深度分割模型配置
└── archive/                        # 归档文件（旧文档、测试脚本、样本媒体等）
    ├── plan.md                     # 早期实验计划
    ├── PROJECT.md                  # 项目早期文档
    ├── infer_depth*.py             # 旧版推理脚本
    └── ...
```

## 训练

### 环境准备

```bash
pip install ultralytics torch torchvision opencv-python numpy matplotlib
```

### 快速开始

```bash
python yolo26_train_depth.py \
  --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \
  --data nyu_yolo/nyu_depth_seg.yaml \
  --epochs 80 \
  --batch 8 \
  --pretrained runs/segment/runs/train_depth/yolo26-seg-depth-exp14/weights/best.pt \
  --device 0 \
  --project runs/train_depth \
  --name yolo26-seg-depth-exp16 \
  --depth-weight 0.5 \
  --freeze-depth-epochs 10 \
  --fliplr 0.4
```

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型 YAML 配置文件 | 必填 |
| `--data` | 数据集 YAML 配置文件 | 必填 |
| `--pretrained` | 预训练权重路径（COCO seg 或上一阶段 best.pt） | 必填 |
| `--depth-weight` | 深度损失权重 | 0.5 |
| `--freeze-depth-epochs` | 仅训练 depth head 的轮数 | 50 |
| `--fliplr` | 水平翻转概率（0~1，控制左右对称性） | 0.5 |
| `--freeze-seg` | 阻断分割梯度回传 backbone | True |

### 训练策略

- 冻结 backbone + 分割头 + `seg_branch`，仅深度相关模块可训练
- 分割 head 的 BN 运行统计量保持在 eval 模式，但 forward 仍返回 dict（保证 depth 键存在）

## 推理

### 单图推理

```bash
python yolo26_inference.py \
  --model runs/segment/runs/train_depth/yolo26-seg-depth-exp16/weights/best.pt \
  --source image.jpg \
  --device 0 \
  --save result.jpg
```

### 视频推理

```bash
python yolo26_inference.py \
  --model runs/segment/runs/train_depth/yolo26-seg-depth-exp16/weights/best.pt \
  --source video.mp4 \
  --device 0
```

### Python API

```python
from yolo26_inference import YOLO26Inference

infer = YOLO26Inference("runs/.../best.pt", device="0")
result = infer.predict_single("image.jpg")
depth = infer.get_depth(result)  # numpy array (H, W)
```

## 诊断

诊断脚本用于量化左右对称性和边界相关性：

```bash
python diagnose_depth.py \
  --ckpts runs/.../exp14/weights/best.pt runs/.../exp16/weights/best.pt \
  --labels exp14 exp16 \
  --images-dir nyu_yolo/images/test \
  --n-images 80 \
  --n-vis 6 \
  --out diagnose_out \
  --device 0
```

输出包含：
- **asymmetry**：左右翻转一致性误差（越低越好）
- **column profile**：列均深度曲线（检测左右偏置）
- **edge_corr**：图像边缘与深度边缘的 Pearson 相关性
- **可视化**：`vis/` 目录下的对比图

## 核心改动

| 模块 | 位置 | 说明 |
|------|------|------|
| `DepthSegment26` | `ultralytics/nn/modules/head.py` | 多任务 Head，继承 Segment26，新增深度路径 |
| `MaskGuidedDepthDecoder` | `ultralytics/nn/modules/head.py` | 多尺度深度解码器，FiLM 调制 + soft mask 引导 |
| `DepthRandomFlip` | `ultralytics/data/depth_dataset.py` | 同时翻转图像、实例标签和深度图 |
| `ImageEdgeAlignmentLoss` | `ultralytics/utils/loss.py` | 不依赖 GT mask 的图像边缘-深度边缘对齐损失 |
| `DepthSegmentationLoss` | `ultralytics/utils/loss.py` | 多任务损失包装器，支持冻结 seg 梯度 |
| `DepthSegmentTrainer` | `ultralytics/models/yolo/segment/train.py` | 多任务训练器，支持渐进式冻结 |

## 参考

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [NYU Depth V2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
