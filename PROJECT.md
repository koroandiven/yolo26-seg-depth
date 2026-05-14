# YOLO26-Seg-Depth 多任务深度估计项目

> 在 YOLO26 分割模型基础上新增单目深度估计能力，同时保持 COCO 分割性能不被 NYU 深度数据污染。

---

## 1. 项目目标

训练一个**多任务模型**，同时完成两项任务：

- **任务一（保持）**：COCO 80 类实例分割（使用预训练权重 `yolo26s-seg.pt`）
- **任务二（新增）**：NYU Depth V2 室内场景单目深度估计（0-10m）

### 核心约束

| 约束 | 原因 |
|------|------|
| Backbone 必须冻结 | 防止 NYU 语义污染 COCO 预训练特征 |
| 分割头必须冻结 | 保持 COCO 分割精度不被破坏 |
| `task_attention.seg_branch` 必须冻结 | 只训练 `depth_branch` |
| 深度输出范围 `depth_scale=20.0` | NYU 室内场景 ~0-10m |
| 使用 `yolo26s-seg.pt` 作为预训练权重 | s scale，与 YAML 配置匹配 |

---

## 2. 实验历程与效果

### 2.1 实验总览

| 实验 | 关键改动 | 效果 | 状态 |
|------|---------|------|------|
| **exp1-4** | 基础深度头 + 多任务损失 | 分割性能下降，深度不稳定 | ❌ 失败 |
| **exp2** | 修复分割头冻结策略（BN eval 替代 `.eval()`） | train/depth_loss 3.96→0.70，**但 val/depth_loss 始终为 0** | ⚠️ 发现关键 bug |
| **exp5** | 修复 val/depth_loss 计算 bug | **val/depth_loss 从 0 → 0.73**，训练正常 | ✅ 验证修复生效 |
| **exp6** | 无 BN DepthConv + SpatialBiasCorrection | 仍有空间偏置 L/R=1.32（左上角大深度） | ⚠️ 偏置未完全消除 |
| **exp12** | 完整网络架构验证 | COCO 图像 max conf **0.946**，分割性能完全保留 | ✅ 验证通过 |
| **exp13** | `decouple_p4p5=False`（当前训练） | 待验证 | 🔄 进行中 |

### 2.2 关键问题与解决

#### 问题 1：分割效果极差（exp1）

**根因**：
- n scale 模型加载了 s scale 预训练权重
- 未正确加载预训练权重
- 分割梯度污染 backbone

**解决**：
- 统一使用 s scale YAML + `yolo26s-seg.pt`
- `ProgressiveFreezeCallback` 用 BN eval 替代 `seg_head.eval()`
- 阻断分割损失梯度回传 backbone

#### 问题 2：val/depth_loss 始终为 0（exp2）

**根因**：`DepthSegment26.forward()` eval 模式返回 `(y, preds_dict)` 元组，但 `DepthSegmentationLoss.loss()` 直接检查 `isinstance(preds, dict)` 为 False，导致 depth_loss 被置 0。

**解决**（3 处修复）：
1. `head.py`：eval 时将 `depth` 注入返回的 `preds_dict`
2. `loss.py`：`DepthSegmentationLoss.loss()` 添加元组解包逻辑
3. `segment/train.py`：`DepthSegmentValidator` 在 `postprocess` 中捕获 `_last_depth_pred`

**验证**：exp5 的 val/depth_loss 从 0 变为 0.73，确认修复生效。

#### 问题 3：空间偏置（exp6）

**现象**：空白图推理时左上角深度大、左深右浅（L/R=1.32）。

**根因分析**：
- `SpatialBiasCorrection.bias_map` 已正则化到零（mean=0.0025），排除 decoder 层面
- `TaskDecouplingAttention.depth_branch` 在 P4/P5 上的训练可能编码了数据集的空间偏置统计
- 随机初始化测试：`decouple_p4p5=True/False` 均无偏置，说明偏置来自训练后的权重

**解决**：
- 添加 `decouple_p4p5` 参数到 YAML 和 `DepthSegment26`
- 设置 `decouple_p4p5: False`，只保留 P3 的 task decoupling
- 预期：消除 P4/P5 depth_branch 的空间偏置，同时降低 box_loss

#### 问题 4：深度估计边缘不一致

**解决**：新增 `MaskEdgeDepthLoss`，在 mask 边缘处施加更高权重，强制深度梯度在物体边界处对齐。

#### 问题 5：恒定深度预测

**解决**：
- `EdgeAwareSmoothnessLoss`：利用图像梯度加权，纹理丰富区域允许更大深度梯度
- `MaskConsistencyDepthLoss`：惩罚 mask 内部深度方差，强制同一物体深度均匀
- 30% mask dropout 训练：防止模型完全依赖 mask 形状作弊

---

## 3. Ultralytics 库改动

### 3.1 新增模块（`nn/modules/head.py`）

| 模块 | 作用 | 关键设计 |
|------|------|---------|
| `TaskDecouplingAttention` | P3 特征解耦为 seg/depth 两路 | `seg_branch` 冻结保持恒等映射，`depth_branch` 可训练 |
| `DepthConv` | 无 BN 的卷积层 | 移除 BatchNorm，防止编码训练数据的空间偏置统计 |
| `SpatialBiasCorrection` | 可学习空间偏置校正 | 低分辨率 bias map 上采样后减去，正则化到零 |
| `DepthResidualBlock` | 深度特征残差块 | 无 BN，4 层堆叠 |
| `MaskGuidedDepthDecoder` | 多尺度深度解码器 | P3/P4/P5 融合 + 弱 mask guidance（30% dropout） |
| `DepthSegment26` | 多任务 Head | 继承 `Segment26`，新增深度路径，支持 `decouple_p4p5` 参数 |

### 3.2 新增损失（`utils/loss.py`）

| 损失类 | 作用 | 公式特点 |
|--------|------|---------|
| `DepthSegmentationLoss` | 多任务损失包装器 | `freeze_seg=True` 时用 `torch.no_grad()` 阻断 seg 梯度 |
| `MultiScaleDepthLoss` | 多尺度深度损失组合 |  scales=(1.0, 0.5)，支持 rect 模式 target resize |
| `SILogLoss` | 尺度不变对数深度损失 | `sqrt(mean(log_diff^2) - alpha*mean(log_diff)^2)` |
| `BerHuLoss` | 反向 Huber 损失 | 对异常值鲁棒，自适应阈值 |
| `MaskConsistencyDepthLoss` | Mask 内深度一致性 | 惩罚 mask 内部深度方差，强制均匀 |
| `EdgeAwareSmoothnessLoss` | 边缘感知平滑 | `exp(-image_gradient)` 加权，允许边缘处大深度梯度 |
| `MaskEdgeDepthLoss` | Mask 边缘深度对齐 | Sobel 边缘检测 + 边缘处深度梯度强化 |

### 3.3 训练/验证改动

| 文件 | 改动 | 效果 |
|------|------|------|
| `nn/tasks.py:loss()` | 添加 `_cached_batch` 传递机制 | Head 可在 forward 中获取 mask 生成 guidance |
| `nn/tasks.py:parse_model()` | 解析 `depth_scale` 和 `decouple_p4p5` | YAML 配置自动生效 |
| `models/yolo/segment/train.py` | `DepthSegmentTrainer` + `DepthSegmentValidator` | 支持多任务训练、冻结策略、深度验证 |
| `data/depth_dataset.py` | `DepthSegmentDataset` | 加载 16-bit PNG 深度图（mm→m），支持 rect 模式 |
| `cfg/models/26/yolo26-seg-depth.yaml` | `depth_scale: 20.0`, `decouple_p4p5: False` | 显式指定深度范围和 decouple 策略 |

### 3.4 关键修复

| Bug | 位置 | 修复 |
|-----|------|------|
| eval 模式 depth 丢失 | `head.py:2111` | `outputs[1]["depth"] = depth` 注入 preds_dict |
| loss 元组解包失败 | `loss.py:1695` | `if isinstance(preds, tuple): preds = preds[1]` |
| validator depth 丢失 | `segment/train.py:37` | `postprocess` 中捕获 `_last_depth_pred` |
| 尺寸不匹配（rect 模式） | `loss.py:1585` | 自动 `F.interpolate` target 到 pred 尺寸 |
| AMP HalfTensor 类型错误 | `loss.py:1375` | Sobel kernel dtype 匹配 `depth.dtype` |

---

## 4. 最新模型训练效果（exp5）

### 4.1 损失趋势

| 指标 | 初始值 | 最终值 | 趋势 |
|------|--------|--------|------|
| train/depth_loss | 3.96 | **0.70** | ↓ 82% |
| val/depth_loss | 1.10 | **0.73** | ↓ 34% |
| train/box_loss | 2.05 | 2.04 | → 持平（冻结保护） |
| val/box_loss | 1.97 | 1.97 | → 持平 |

### 4.2 关键观察

- **验证修复生效**：`val/depth_loss` 从 exp2 的恒为 0 变为正常计算
- **轻微过拟合**：最终 epoch `val/depth_loss (0.73) > train/depth_loss (0.70)`
- **收敛 plateau**：epoch 50 后损失停滞在 ~0.73，继续训练到 150 epochs 收益有限
- **分割保护成功**：box/seg/cls/dfl loss 全部持平，mAP 接近 0 是预期行为（COCO 80 类头在 NYU 10 类数据上验证）

### 4.3 深度估计指标（exp2 checkpoint 修复后验证）

| 指标 | 值 |
|------|-----|
| abs_rel | 0.2579 |
| rmse | 1.0173 m |
| silog | 0.3010 |

> 注：NYU 室内场景深度范围 ~0-10m，RMSE=1.02m 在可接受范围。

---

## 5. 脚本使用教程

### 5.1 训练脚本

**文件**：`yolo26_train_depth.py`

**功能**：启动 YOLO26-Seg-Depth 多任务训练，支持渐进式冻结策略。

```bash
cd /d01/training_data/yolo-other/yolo26-seg-depth

python yolo26_train_depth.py \
  --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \
  --data nyu_yolo/nyu_depth_seg.yaml \
  --epochs 150 \
  --batch 8 \
  --pretrained yolo26s-seg.pt \
  --device 0 \
  --project runs/train_depth \
  --name yolo26-seg-depth-exp13 \
  --depth-weight 0.5 \
  --freeze-depth-epochs 50
```

**关键参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型 YAML 配置文件 | `yolo26-seg-depth.yaml` |
| `--data` | 数据集 YAML 配置文件 | `nyu_yolo/nyu_depth_seg.yaml` |
| `--pretrained` | COCO 分割预训练权重路径 | **必须指定** |
| `--depth-weight` | 深度损失权重 | 0.5 |
| `--freeze-depth-epochs` | 仅训练 depth head 的轮数 | 50 |
| `--freeze-seg` | 阻断分割梯度（默认启用） | True |
| `--no-freeze-seg` | 允许分割梯度更新 backbone | 不推荐 |

**训练策略**：
- Phase 1（0-49 epochs）：仅 depth head 可训练，backbone + seg head 冻结
- Phase 2（50+ epochs）：depth head + backbone 可训练，seg head 仍冻结
- 实际上 backbone 永久冻结（实验证明解冻会导致 COCO 分割置信度从 0.95 → 0.06）

### 5.2 推理脚本

#### 5.2.1 单图推理（带 mask guidance）

**文件**：`infer_depth_fixed.py`

**功能**：对单张图片进行分割 + 深度联合推理，mask guidance 显式注入深度 decoder。

```bash
python infer_depth_fixed.py \
  --model runs/segment/runs/train_depth/yolo26-seg-depth-exp5/weights/best.pt \
  --source test_image.jpg \
  --save-dir ./inference_results \
  --conf 0.25 \
  --device 0
```

**输出**：
- `segmentation.jpg`：分割结果可视化
- `depth.jpg`：深度伪彩色热力图
- `combined.jpg`：原图与深度图拼接

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--no-mask-guide` | 禁用 mask guidance，测试模型是否真正从图像内容推断深度 |
| `--conf` | 分割置信度阈值 |
| `--imgsz` | 推理尺寸，默认 640 |

#### 5.2.2 批量/视频推理

**文件**：`yolo26_inference.py`

**功能**：视频、图片目录、摄像头实时推理。

```bash
# 单图推理
python yolo26_inference.py \
  --model runs/segment/runs/train_depth/yolo26-seg-depth-exp5/weights/best.pt \
  --source test_image.jpg \
  --device 0 \
  --save result.jpg

# 视频推理
python yolo26_inference.py \
  --model best.pt \
  --source video.mp4 \
  --device 0 \
  --save output.mp4

# 摄像头实时推理
python yolo26_inference.py \
  --model best.pt \
  --webcam \
  --camera-id 0 \
  --show-depth \
  --device 0
```

### 5.3 验证脚本

#### 5.3.1 标准验证

```python
from ultralytics import YOLO

model = YOLO("runs/segment/runs/train_depth/yolo26-seg-depth-exp5/weights/best.pt")
results = model.val(
    data="nyu_yolo/nyu_depth_seg.yaml",
    imgsz=640,
    batch=8,
    device="0",
)
```

#### 5.3.2 深度指标单独验证

**文件**：`validate_exp2_final.py`（通用，不限于 exp2）

```bash
python validate_exp2_final.py
```

该脚本会：
1. 加载 checkpoint 并修复 args（dict → SimpleNamespace）
2. 重新初始化 `DepthSegmentationLoss`
3. 运行完整验证并输出 `val/depth_loss` 和深度指标（abs_rel / rmse / silog）

### 5.4 空间偏置检查

**文件**：`check_spatial_bias.py`

```bash
python check_spatial_bias.py \
  --model runs/segment/runs/train_depth/yolo26-seg-depth-exp6/weights/best.pt \
  --device cpu
```

**输出**：
- `SpatialBiasCorrection.bias_map` 统计
- 空白图/噪声图的空间梯度分析
- 左/右、上/下深度比值

### 5.5 模型结构对比

**文件**：`compare_decouple_bias.py`

```bash
python compare_decouple_bias.py
```

对比 `decouple_p4p5=True/False` 在相同随机特征输入下的深度输出差异。

---

## 6. 文件结构

```
yolo26-seg-depth/
├── yolo26_train_depth.py          # 训练入口
├── yolo26_inference.py            # 视频/批量推理
├── infer_depth_fixed.py           # 单图推理（mask guidance）
├── validate_exp2_final.py         # 深度指标验证
├── check_spatial_bias.py          # 空间偏置检测
├── compare_decouple_bias.py       # decouple 对比
├── yolo26-seg-depth.yaml          # 模型配置
├── ARCHITECTURE.md                # 网络架构文档（旧版）
├── nyu_yolo/
│   ├── nyu_depth_seg.yaml         # 数据集配置
│   ├── images/train/              # RGB 图像
│   ├── depths/train/              # 16-bit PNG 深度图（mm）
│   ├── labels/train/              # YOLO 检测标签
│   └── segments/train/            # 分割标签
├── ultralytics/
│   └── ultralytics/
│       ├── nn/modules/head.py     # DepthSegment26 等
│       ├── nn/tasks.py            # BaseModel.loss, parse_model
│       ├── utils/loss.py          # 所有深度损失
│       ├── models/yolo/segment/train.py  # Trainer/Validator
│       ├── data/depth_dataset.py  # DepthSegmentDataset
│       └── cfg/models/26/yolo26-seg-depth.yaml  # 模型 YAML
└── runs/segment/runs/train_depth/
    ├── yolo26-seg-depth-exp2/     # 150 epochs, val/depth_loss=0 (bug)
    ├── yolo26-seg-depth-exp5/     # 150 epochs, val/depth_loss=0.73
    ├── yolo26-seg-depth-exp6/     # 无 BN decoder + SpatialBiasCorrection
    └── yolo26-seg-depth-exp13/    # decouple_p4p5=False (当前训练)
```

---

## 7. 关键决策记录

| 决策 | 原因 | 验证结果 |
|------|------|---------|
| 冻结 backbone + seg_branch | 防止 NYU 语义污染 COCO 特征 | 分割 max conf 保持 0.946 |
| 目标深度保持原始米数 | `/depth_scale` 导致输出完全错误 | exp 验证：原始米数训练正确 |
| 推理时完全禁用 mask_guidance | 测试模型是否真正学到图像→深度映射 | eval 模式 `mask_guidance=None` |
| 30% mask dropout | 强制模型从图像内容学习，不依赖 mask 形状 | 训练正常，深度可学习 |
| Conv → DepthConv（无 BN） | BN 可能编码训练数据的空间偏置统计 | exp6 偏置仍存在，来源在 P4/P5 decouple |
| `decouple_p4p5=False` | P4/P5 task decoupling 导致 box_loss 上升 30% + 空间偏置 | exp13 正在验证 |

---

*文档生成时间: 2026-05-12*
*对应代码版本: yolo26-seg-depth exp5-exp13*
