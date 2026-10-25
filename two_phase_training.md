# 两阶段训练指南：多类别分割头 → 冻结分割 + 深度头

本文档说明如何在当前代码基线上完成两轮训练：

1. **Phase 1**：用更多类别训练（或微调）分割头，得到一个多类别分割模型；
2. **Phase 2**：冻结该分割头，在其上训练单目深度头（一次推理同时输出分割 + 稠密深度图）。

当前代码已具备 Phase 2 所需的全部机制：分割路径冻结保护、深度质量选模（fitness）、
翻转一致性损失、L2-SP 锚定、分段解冻、COCO/自定义分割护栏、几何一致 Mosaic。
Phase 1 使用标准分割训练流程即可，不需要任何深度相关代码。

---

## Phase 1：多类别分割头训练

### 1. 确定统一类别表（唯一的决策点）

先决定新的类别体系，常见三种来源：

| 方案 | 说明 |
|---|---|
| COCO 80 + NYU 10 合并 | 在 COCO 80 基础上并入 NYU 室内类别（wall/floor/bed/...），共约 88~90 类；需处理与 COCO 已有类别的去重（如 chair/couch/bed 已存在） |
| NYU-40 官方类别 | 只用 NYU 官方 40 类语义，标注转换工作最小 |
| 自有标注 | 在与 NYU 对齐的图片上补充新类别标注 |

把所有训练源的标签按新类别表重映射 id，整理为 YOLO 分割格式（`images/` + `segments/`），
并新建 data yaml：

```yaml
path: /path/to/multi_seg        # 数据集根目录
train: images/train
val: images/val
nc: 90                          # 新类别总数
names: {0: person, 1: bicycle, ..., 40: wall, 41: floor, ...}   # 全量类别表，id 连续
```

### 2. 训练（标准分割流程，以 COCO 预训练为起点）

```bash
yolo segment train \
    model=yolo26-seg.yaml \
    data=multi_seg.yaml \
    pretrained=yolo26n-seg.pt \
    epochs=100 imgsz=640 \
    project=runs/segment name=phase1_multiclass
```

类别数与预训练不一致时，检测/分类头层会自动重初始化，其余权重保留。
**注意训练日志中的 `Transferred N/M items` 一行：N 应接近 M**
（只有因形状变化被重置的头层除外），否则说明预训练权重没有按预期载入。

### 3. Phase 1 产出

`runs/segment/phase1_multiclass/weights/best.pt` 即为 Phase 2 的 `--pretrained`。
记录该模型在新验证集上的 mAP50/mAP50-95，作为后续护栏的参照。

---

## 过渡：一处必要的小改

`ultralytics/ultralytics/models/yolo/segment/train.py` 的
`DepthSegmentValidator.init_metrics` 中，混淆矩阵扩展写死了 80（当初为
"COCO 头在 NYU 上验证"准备）。类别数变化后需要跟随模型头：

```python
# DepthSegmentValidator.init_metrics 内，替换：
#     if self.nc < 80:
#         self.nc = 80
# 为：
model_nc = unwrap_model(model).model[-1].nc
if self.nc < model_nc:
    self.nc = model_nc
```

不改的话：新类别数 > 80 时验证器的混淆矩阵会越界；< 80 时虽不出错但类别名对不上。
（`unwrap_model` 已在本文件导入。）

---

## Phase 2：冻结多类别分割头，训练深度头

### 1. 新建匹配新类别数的深度模型配置（不要改原文件）

```bash
cp ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \
   ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth-nc90.yaml
# 将 nc: 80 改为 nc: 90（与 Phase 1 的类别数一致）
```

### 2. 启动深度训练

```bash
python yolo26_train_depth.py \
    --model yolo26-seg-depth-nc90.yaml \
    --data nyu_yolo/nyu_depth_seg.yaml \
    --pretrained runs/segment/phase1_multiclass/weights/best.pt \
    --name exp20-phase2 \
    --sym-weight 0.1 \
    --coco-guard-data multi_seg.yaml
```

说明：

- **NYU 数据不需要改**。NYU 分割标注的类别 id（0-9）在新类别体系下仍是合法索引；
  seg loss 处于冻结/跳过状态，深度训练用到的 mask 损失（MaskEdge/MaskConsistency）
  只依赖 mask 几何、不依赖类别 id。
- `--coco-guard-data` 应指向 **Phase 1 同类别体系** 的验证集（如 `multi_seg.yaml`）。
  若新体系保留了 COCO 原始 id 不变，也可继续用 COCO 验证集。
- 训练期间的分割指标请查看运行目录下的 `coco_guard.csv`（护栏每 N epoch 测量一次）。

### 3. 关于翻转增强（重要，勿遗漏）

**不需要、也不应把 NYU 数据集离线翻转预处理。** 训练管线内置
`DepthRandomFlip`（`ultralytics/ultralytics/data/depth_dataset.py`），
按 `fliplr` 概率在训练时对 **图像 + 深度图 + 实例标注三者同步做水平翻转**——
翻转由数据管线在每个样本上在线完成，原始数据集保持原样。
垂直翻转被明确禁止（会颠倒地板/天花板几何，深度无法自洽）。

实验结论（exp14~exp16）：水平翻转是压制 Kinect 固定机位"左右深度偏置"的
最有效手段，`--fliplr 0.4` 是已验证的甜点（asymmetry 0.431m → 0.317m，
相对改善 26%；默认 0.5 也可接受）：

```bash
python yolo26_train_depth.py \
    --model yolo26-seg-depth-nc90.yaml \
    --data nyu_yolo/nyu_depth_seg.yaml \
    --pretrained runs/segment/phase1_multiclass/weights/best.pt \
    --name exp20-phase2 \
    --fliplr 0.4 --sym-weight 0.1 \
    --coco-guard-data multi_seg.yaml
```

`--fliplr`（数据层面的翻转）与 `--sym-weight`（损失层面的翻转一致性）互补，
针对同一偏置问题，建议同时开启。

### 4. 基线确认后再放开训练

1. 先纯冻结跑 50~90 epoch，得到 Phase 2 基线；
2. 确认 `coco_guard.csv` 中分割 mAP 稳定、无退化；
3. 再开分段解冻做第二轮消融：

```bash
python yolo26_train_depth.py \
    --model yolo26-seg-depth-nc90.yaml \
    --data nyu_yolo/nyu_depth_seg.yaml \
    --pretrained runs/segment/phase1_multiclass/weights/best.pt \
    --name exp21-phase2-unfreeze \
    --sym-weight 0.1 --depth-mosaic 0.5 \
    --unfreeze-backbone-at 30 --backbone-lr-scale 0.1 --sp-weight 0.01 \
    --coco-guard-data multi_seg.yaml
```

此时 L2-SP 的锚（θ₀）是 Phase 1 的最优分割权重，小学习率（0.1×）+ L2-SP（0.01）
在放开 backbone/neck 的同时把特征拉向分割最优点；护栏负责监控退化。

---

## 实验对比注意事项

- 深度质量横向对比**只在相同代码基线的实验之间进行**。RGB 接线修复
  （train/val/predict/export 四端一致）之前的历史实验（exp5~exp16）的
  val/depth_loss、asymmetry 数值与之后的实验不可直接比较。
- best.pt 的选择已由"分割 mAP 噪声"改为深度质量
  （fitness = depth/delta1 − 0.5 × depth/abs_rel），结果以 results.csv 的
  `depth/*` 列与 `diagnose_depth.py` 的 asymmetry/edge_corr 为准。

## 训练注意事项清单

### 数据与配置

1. **在仓库根目录启动训练**（`python yolo26_train_depth.py ...`）。
   `nyu_yolo/nyu_depth_seg.yaml` 的 `path` 是相对路径；在服务器上训练时
   改成服务器上的实际绝对路径。
2. **深度管线的几何增强只有水平翻转和 `--depth-mosaic`**。标准 hyp 里的
   `degrees/translate/scale/crop` 等空间增强对深度图不生效（被刻意排除，
   避免几何不一致的错误监督），不要指望调它们。
3. **NYU 单源训练的可选调优**：head 输出是 `sigmoid × depth_scale`（默认 100），
   而 NYU 目标只有 0~10m，sigmoid 实际只用到底部 10% 区间。纯 NYU 训练可把
   模型 yaml 的 `depth_scale` 改为 10 换取输出分辨率；**训练与部署必须保持
   同一值**（导出 ONNX 时该值固化在图里）。多源混训（NYU+KITTI）保持 100，
   并记得在 data yaml 里配 `depth_norm_max`（见 `nyu_kitti_mix.yaml`）。

### 训练过程监控

4. **`freeze_seg` 下 box/seg/cls/dfl/sem 损失恒为 0 是预期行为**
   （分割被冻结，新代码直接跳过其计算）。训练是否健康看：
   `train/depth_loss`、`val/depth_loss`、results.csv 的 `depth/*` 指标、
   `coco_guard.csv`，以及线下 `diagnose_depth.py` 的 asymmetry/edge_corr。
5. **显存占用明显高于 stock 分割训练**：稠密深度标签 (B,640,640)、输入图
   stash、sym 损失的额外解码器前向、mosaic 的 2 倍尺寸张量。默认 batch=8
   在 8GB 卡上可能紧张；OOM 时先降 batch，再考虑关闭 `--sym-weight` 或
   `--depth-mosaic`。
6. **护栏读数解读**：首次运行需下载/准备验证数据（coco128-seg 仅 128 张图，
   mAP 绝对值低且抖动大属正常，**只看相对首次测量的跌幅**）；若新类别体系
   重排了 COCO 的 id，护栏数据必须用同体系验证集（正文已述）。
7. **解冻时机与学习率**：lr 按 one-cycle  schedule 随 epoch 衰减，
   `--unfreeze-backbone-at` 越早，backbone 实际得到的绝对 lr 越大
   （lr0 × lf(epoch) × backbone_lr_scale）。30/90 epoch 处解冻约为
   lr0 中段的 0.1 倍，通常安全；再早需同步调小 `--backbone-lr-scale`。

### 产出、恢复与部署

8. **旧实验（exp5~exp16）的 best.pt 是按噪声选出的，不要直接复用其结论**
   或与新实验的指标直接对比。旧 ckpt 可以在新代码下正常加载推理
   （getattr 回退兼容），但：
   - **部署前必须用新代码重新导出 ONNX**——旧 ONNX 把"RGB 分支跑零向量"
     固化进了计算图，RGB 修复对它无效；
   - 重新导出后重跑一次 `diagnose_depth.py`，确认 asymmetry 变化
     （RGB 精修从死分支变为真正参与推理，指标会变）。
9. `--resume` 仅适用于新代码基线上启动的 run；跨代码版本恢复旧实验请新开训练。
10. 若未来启用 `use_log_depth: True`，注意 AMP fp16 下 `exp(raw)` 在
    raw>11 时即溢出（65504 上限）；默认的 sigmoid 线性输出无此问题。

## 附：新增 CLI 参数速查

| 参数 | 默认 | 作用 |
|---|---|---|
| `--fliplr` | hyp 默认 0.5 | 图像+深度+实例同步水平翻转概率；压制左右深度偏置，推荐 0.4 |
| `--sym-weight` | 0.0 | 翻转一致性（对称）损失权重，针对左右深度偏置 |
| `--depth-mosaic` | 0.0 | 几何一致 Mosaic 概率（深度四块拼接 + 2× 降采样） |
| `--unfreeze-backbone-at` | -1 | 从第 N 个 epoch 起解冻 backbone/neck（-1 全程冻结） |
| `--backbone-lr-scale` | 0.1 | 解冻后 backbone/neck 的学习率倍率 |
| `--sp-weight` | 0.0 | L2-SP 权重，把解冻参数锚向 `--pretrained` 权重 |
| `--coco-guard-data` | "" | 训练期分割护栏的数据集 yaml（空 = 关闭） |
| `--coco-guard-interval` | 10 | 护栏评估间隔（epoch） |
| `--coco-guard-drop-tol` | 0.05 | 护栏 mAP50(M) 相对首次测量的允许跌幅 |
