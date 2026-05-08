# YOLO26-Seg-Depth 网络架构详解

> 本文档详细描述 `yolo26-seg-depth` 多任务模型的网络结构、数据流向、训练策略及推理流程。
> 基于模型配置文件 `yolo26-seg-depth.yaml` 及自定义模块实现。

---

## 1. 整体架构概览

```
输入图像 (B, 3, 640, 640)
    │
    ├─→ [Backbone 0-10]  ──→ 特征提取 (P2/P3/P4/P5)
    │
    ├─→ [Neck/FPN 11-22] ──→ 多尺度特征融合 (P3/P4/P5)
    │
    └─→ [Head 23] DepthSegment26
            │
            ├─→ task_attention(P3) ──→ seg_branch ──→ Segment26 ──→ 分割结果
            │                           (冻结)          (冻结)
            │
            └─→ task_attention(P3) ──→ depth_branch ──→ depth_decoder ──→ 深度图
                                        (可训练)         (可训练)
```

### 1.1 关键设计思想

- **共享 Backbone + Neck**：分割和深度两个任务共享同一套主干网络和特征金字塔，减少参数量和计算量。
- **Task Decoupling Attention**：在 Head 入口处对 P3 特征进行解耦，分为 `seg_feat` 和 `depth_feat` 两路，避免两个任务的特征需求互相冲突。
- **冻结保护机制**：Backbone、Neck、分割头全部冻结，仅训练深度相关模块，确保预训练的 COCO 分割能力不被破坏。

---

## 2. Backbone (Layers 0-10)

Backbone 负责从输入图像中提取多尺度特征。采用 YOLO26 标准结构，由 `Conv`、`C3k2`、`SPPF`、`C2PSA` 等模块组成。

> **Scale 说明**：本文以 `s` scale（width=0.50）为例，对应预训练权重 `yolo26s-seg.pt`。实际通道数已按 width 缩放。

### 2.1 逐层结构

| Layer | Module | Args | 输入通道 | 输出通道 | 输出尺度 | 作用 |
|:-----:|:------:|:----:|:--------:|:--------:|:--------:|:-----|
| 0 | Conv | [64, 3, 2] | 3 | 32 | P1/2 (320×320) | 下采样 ×2，初步特征提取 |
| 1 | Conv | [128, 3, 2] | 32 | 64 | P2/4 (160×160) | 下采样 ×2 |
| 2 | C3k2 | [256, False, 0.25] | 64 | 128 | P2/4 | 跨阶段局部网络，无残差 |
| 3 | Conv | [256, 3, 2] | 128 | 128 | P3/8 (80×80) | 下采样 ×2 |
| 4 | C3k2 | [512, False, 0.25] | 128 | 256 | P3/8 | 跨阶段局部网络 |
| 5 | Conv | [512, 3, 2] | 256 | 256 | P4/16 (40×40) | 下采样 ×2 |
| 6 | C3k2 | [512, True] | 256 | 256 | P4/16 | 跨阶段局部网络，有残差 |
| 7 | Conv | [1024, 3, 2] | 256 | 512 | P5/32 (20×20) | 下采样 ×2 |
| 8 | C3k2 | [1024, True] | 512 | 512 | P5/32 | 跨阶段局部网络 |
| 9 | SPPF | [1024, 5, 3, True] | 512 | 512 | P5/32 | 空间金字塔池化，融合多尺度上下文 |
| 10 | C2PSA | [1024, 1] | 512 | 512 | P5/32 | 通道注意力（PSA）增强 |

### 2.2 关键模块说明

#### Conv
标准卷积 + BatchNorm + SiLU 激活：
```python
Conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
```

#### C3k2
YOLO26 的核心构建块，结合 CSP（Cross Stage Partial）和 Bottleneck：
- `n=2`：2 个 Bottleneck 重复
- `shortcut=False/True`：是否使用残差连接
- `e=0.25`：Bottleneck 中间通道的缩放系数

#### SPPF
快速空间金字塔池化（Spatial Pyramid Pooling - Fast）：
- 并行使用多个不同大小的 MaxPool（内核 5×5）
- 融合局部和全局上下文信息

#### C2PSA
带 PSA（Partial Self-Attention）的 C2f 模块：
- 在通道维度引入自注意力机制
- 增强对长距离依赖的建模能力

---

## 3. Neck / FPN (Layers 11-22)

Neck 采用 **PANet（Path Aggregation Network）** 结构，通过上采样、下采样和横向连接，构建多尺度特征金字塔。

### 3.1 自上而下路径 (Top-Down)

将深层语义特征向浅层传播：

```
P5/32 (512ch) ──→ Upsample ──→ Concat with P4 ──→ C3k2 ──→ 256ch
    │
    └───────────────────────────────────────────────────────→
                                    │
                                    ▼
                              P4/16 (256ch) ──→ Upsample ──→ Concat with P3 ──→ C3k2 ──→ 128ch
                                                    │
                                                    ▼
                                              P3/8 (128ch) [Head 输入]
```

| Layer | Module | Args | 输入 | 输出通道 | 说明 |
|:-----:|:------:|:----:|:----:|:--------:|:-----|
| 11 | Upsample | [None, 2, "nearest"] | 512 | 512 | P5 上采样 ×2 |
| 12 | Concat | [-1, 6] | 512+256 | 768 | 拼接 P4 特征 |
| 13 | C3k2 | [512, True] | 768 | 256 | 融合后输出 |
| 14 | Upsample | [None, 2, "nearest"] | 256 | 256 | 上采样 ×2 |
| 15 | Concat | [-1, 4] | 256+256 | 512 | 拼接 P3 特征 |
| 16 | C3k2 | [128, True] | 512 | **128** | **P3/8-small，Head 输入[0]** |

### 3.2 自下而上路径 (Bottom-Up)

将浅层位置特征向深层传播：

```
P3/8 (128ch) ──→ Conv(s=2) ──→ Concat with P4 ──→ C3k2 ──→ 256ch
    │
    └───────────────────────────────────────────────────────→
                                    │
                                    ▼
                              P4/16 (256ch) ──→ Conv(s=2) ──→ Concat with P5 ──→ C3k2 ──→ 512ch
                                                    │
                                                    ▼
                                              P5/32 (512ch) [Head 输入]
```

| Layer | Module | Args | 输入 | 输出通道 | 说明 |
|:-----:|:------:|:----:|:----:|:--------:|:-----|
| 17 | Conv | [128, 3, 2] | 128 | 128 | P3 下采样 ×2 |
| 18 | Concat | [-1, 13] | 128+256 | 384 | 拼接 Layer 13 输出 |
| 19 | C3k2 | [256, True] | 384 | **256** | **P4/16-medium，Head 输入[1]** |
| 20 | Conv | [256, 3, 2] | 256 | 256 | P4 下采样 ×2 |
| 21 | Concat | [-1, 10] | 256+512 | 768 | 拼接 Layer 10 输出 |
| 22 | C3k2 | [512, True, 0.5, True] | 768 | **512** | **P5/32-large，Head 输入[2]** |

### 3.3 Neck 输出

Neck 向 Head 输出三个尺度的特征图：

| 特征 | 尺度 | 通道 | 感受野 | 检测目标 |
|:----:|:----:|:----:|:------:|:---------|
| P3 | 80×80 | 128 | 小 | 小目标 |
| P4 | 40×40 | 256 | 中 | 中目标 |
| P5 | 20×20 | 512 | 大 | 大目标 |

---

## 4. Head — DepthSegment26 (Layer 23)

Head 是模型的核心创新点，同时负责**实例分割**和**单目深度估计**两个任务。

### 4.1 整体结构

```
                    P3 (128ch)          P4 (256ch)         P5 (512ch)
                      │                   │                  │
                      ▼                   │                  │
            ┌─────────────────┐           │                  │
            │ TaskDecoupling  │           │                  │
            │   Attention     │           │                  │
            │                 │           │                  │
            │  P3 ──┬──→ seg_branch ──→ seg_feat (128ch)    │
            │       │          (冻结)                        │
            │       └──→ depth_branch ─→ depth_feat (128ch)  │
            │                 (可训练)                       │
            └─────────────────┘                              │
                      │                   │                  │
            ┌─────────┘                   │                  │
            ▼                             ▼                  ▼
    ┌───────────────┐           ┌──────────────────┐
    │  Segment26    │           │ MultiScaleDepth  │
    │  (分割头)      │           │    Decoder       │
    │               │           │   (深度头)       │
    │  输入: [seg_feat, P4, P5] │   输入: [depth_feat, P4, P5]│
    │  状态: 冻结    │           │   状态: 可训练   │
    └───────┬───────┘           └────────┬─────────┘
            │                            │
            ▼                            ▼
    ┌───────────────┐           ┌──────────────────┐
    │  boxes +      │           │   Depth Map      │
    │  masks +      │           │   (1, 640, 640)  │
    │  classes +    │           │   [0, 100] meters│
    │  confidences  │           │                  │
    └───────────────┘           └──────────────────┘
```

### 4.2 TaskDecouplingAttention

在 P3 特征上执行任务解耦，将共享特征分成两路：

```python
class TaskDecouplingAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 分割分支：冻结，保持预训练近似恒等映射
        self.seg_branch = nn.Sequential(
            Conv(channels, channels // 2, 3),   # 128→64
            Conv(channels // 2, channels, 3),   # 64→128
            nn.Sigmoid()                         # 输出 [0,1] 注意力权重
        )
        # 深度分支：可训练，学习深度任务适配的特征
        self.depth_branch = nn.Sequential(
            Conv(channels, channels // 2, 3),   # 128→64
            Conv(channels // 2, channels, 3),   # 64→128
            nn.Sigmoid()
        )

    def forward(self, x):
        seg_feat = x * self.seg_branch(x)    # 逐通道注意力
        depth_feat = x * self.depth_branch(x)
        return seg_feat, depth_feat
```

**设计意图**：
- `seg_branch` 在预训练时近似恒等映射（权重接近 0，Sigmoid 输出 ≈ 0.5），冻结后保持这一特性，不破坏原始 P3 特征。
- `depth_branch` 可训练，学习增强/抑制对深度估计有用的特征通道。

### 4.3 Segment26 (分割头)

继承自 Ultralytics 标准 `Segment` 头，负责实例分割预测。

#### 4.3.1 结构组件

| 组件 | 作用 | 输出维度 | 冻结状态 |
|:----:|:-----|:--------:|:--------:|
| `cv2` (×3) | 边界框回归 | 4×reg_max=4 | ❌ 冻结 |
| `cv3` (×3) | 类别分类 | nc=80 | ❌ 冻结 |
| `cv4` (×3) | Mask 系数 | nm=32 | ❌ 冻结 |
| `cv5` (×3) | Mask 辅助 | 1 | ❌ 冻结 |
| `proto` | 原型 Mask 生成器 | nm=32 | ❌ 冻结 |
| `dfl` | Distribution Focal Loss | - | ❌ 冻结 |

#### 4.3.2 单尺度检测分支结构 (以 P3 为例)

```
P3 (128ch)
    │
    ├─→ cv2[0] ──→ Conv(128→32, 3×3) ──→ Conv(32→32, 3×3) ──→ Conv2d(32→4, 1×1)
    │              └── 边界框回归 (xywh)                                              
    │
    ├─→ cv3[0] ──→ Conv(128→32, 3×3) ──→ Conv(32→32, 3×3) ──→ Conv2d(32→80, 1×1)
    │              └── 类别分类 (80 COCO classes)                                      
    │
    └─→ cv4[0] ──→ Conv(128→32, 3×3) ──→ Conv(32→32, 3×3) ──→ Conv2d(32→32, 1×1)
                   └── Mask 系数 (32维)                                               
```

#### 4.3.3 Proto (原型 Mask)

```python
Proto(c1=128, c_=256, c2=32):
    cv1: Conv(128 → 256, 3×3)
    upsample: ×2 (80×80 → 160×160)
    cv2: Conv(256 → 256, 3×3)
    cv3: Conv(256 → 32, 3×3)   # 输出 32 个原型 mask
```

**Mask 解码**：预测框的 32 维 mask 系数与 32 个原型 mask 做线性组合，得到实例 mask。

#### 4.3.4 输出格式

训练时返回字典：
```python
{
    "one2many": {"boxes": ..., "scores": ..., "feats": ...},  # NMS 前
    "one2one":  {"boxes": ..., "scores": ..., "feats": ...},  # 端到端
}
```

推理时返回张量：`(B, N_anchors, 4 + 80 + 32 + 1)`，即每个 anchor 的：
- 4 维边界框 (DFL 解码后)
- 80 维类别分数
- 32 维 mask 系数
- 1 维辅助输出

### 4.4 MultiScaleDepthDecoder (深度头)

多尺度深度估计解码器，从 P3/P4/P5 特征融合生成单张深度图。

#### 4.4.1 结构

```python
class MultiScaleDepthDecoder(nn.Module):
    def __init__(self, ch=[128, 256, 512], c_depth=128):
        # P3 分支
        self.p3_conv1 = Conv(ch[0], c_depth, 3, 2)      # 128→128, 下采样到 40×40
        self.p3_conv2 = Conv(c_depth, c_depth, 3)       # 128→128

        # P4 分支
        self.p4_conv1 = Conv(ch[1], c_depth, 3)         # 256→128
        self.p4_conv2 = Conv(c_depth, c_depth, 3)       # 128→128

        # P5 分支
        self.p5_conv1 = Conv(ch[2], c_depth, 3)         # 512→128
        self.p5_conv2 = Conv(c_depth, c_depth, 3)       # 128→128

        # 上采样融合
        self.up_p4 = nn.Upsample(scale_factor=2)        # 40×40 → 80×80
        self.up_p5 = nn.Upsample(scale_factor=4)        # 20×20 → 80×80

        # 输出头
        self.fusion_conv = Conv(c_depth * 3, c_depth, 3)  # 384→128
        self.depth_up = nn.Sequential(
            nn.Upsample(scale_factor=2),                 # 80×80 → 160×160
            Conv(c_depth, c_depth // 2, 3),              # 128→64
            nn.Upsample(scale_factor=2),                 # 160×160 → 320×320
            Conv(c_depth // 2, c_depth // 4, 3),         # 64→32
            nn.Upsample(scale_factor=2),                 # 320×320 → 640×640
            nn.Conv2d(c_depth // 4, 1, 1)                # 32→1
        )
```

#### 4.4.2 数据流

```
P3/8 (128ch, 80×80) ──→ Conv ──→ Conv ──→ 128ch, 80×80 ──────┐
                                                               ├──→ Concat ──→ Fusion ──→ 上采样 ──→ 深度图
P4/16 (256ch, 40×40) ──→ Conv ──→ Conv ──→ 128ch, 40×40 ──→ Upsample×2 ──→ 80×80 ──┤
                                                                                      │
P5/32 (512ch, 20×20) ──→ Conv ──→ Conv ──→ 128ch, 20×20 ──→ Upsample×4 ──→ 80×80 ──┘
```

#### 4.4.3 输出处理

```python
depth = self.depth_decoder(x)          # (B, 1, 640, 640)
depth = torch.sigmoid(depth) * 100.0   # 归一化到 [0, 100] 米
```

---

## 5. 训练策略与参数冻结

### 5.1 冻结策略

| 模块 | 参数 | 训练状态 | 梯度回传 | 说明 |
|:----:|:----:|:--------:|:--------:|:-----|
| **Backbone 0-10** | 全部 | ❌ 冻结 | ❌ 阻断 | COCO 预训练特征永久保护 |
| **Neck/FPN 11-22** | 全部 | ❌ 冻结 | ❌ 阻断 | COCO 预训练特征永久保护 |
| **task_attention.seg_branch** | Conv 权重 | ❌ 冻结 | ❌ 阻断 | 保持近似恒等映射 |
| **task_attention.depth_branch** | Conv 权重 | ✅ 可训练 | ✅ 允许 | 学习深度适配特征 |
| **Segment26 (cv2/cv3/cv4/cv5)** | 全部 | ❌ 冻结 | ❌ 阻断 | 分割能力完全保留 |
| **proto** | 全部 | ❌ 冻结 | ❌ 阻断 | Mask 原型保持不变 |
| **dfl** | 全部 | ❌ 冻结 | ❌ 阻断 | DFL 分布不变 |
| **MultiScaleDepthDecoder** | 全部 | ✅ 可训练 | ✅ 允许 | 唯一学习目标 |

### 5.2 损失函数

采用 `DepthSegmentationLoss`，联合计算分割损失和深度损失：

```python
# 分割损失 (冻结状态，无梯度回传)
if freeze_seg:
    with torch.no_grad():
        seg_loss = v8SegmentationLoss(seg_preds, batch)

# 深度损失 (可训练)
depth_loss = MultiScaleDepthLoss(depth_pred, depth_target)

# 总损失 (仅 depth_loss 有梯度)
total_loss = depth_weight * depth_loss
```

#### MultiScaleDepthLoss 组成

| 损失项 | 权重 | 说明 |
|:------:|:----:|:-----|
| L1 Loss | 1.0 | 像素级深度差异 |
| SSIM Loss | 1.0 | 结构相似性，保持边缘 |
| Gradient Loss | 1.0 | 深度梯度一致性 |
| Edge-Aware Loss | 0.2 | 边缘处更高权重 |
| Multi-Scale | (1.0, 0.5) | 两个尺度的加权平均 |

### 5.3 训练配置

```yaml
epochs: 150
batch: 8
optimizer: AdamW(lr=0.000714, momentum=0.9)
weight_decay: 0.0005
amp: True  # 自动混合精度
pretrained: yolo26s-seg.pt  # COCO 分割预训练权重
freeze_seg: True            # 阻断分割梯度
```

---

## 6. 推理流程

### 6.1 前向传播

```python
def forward(self, x):
    # 1. Backbone + Neck 提取特征
    x = self.backbone(x)   # P3/P4/P5
    
    # 2. Task Decoupling
    seg_feat, depth_feat = self.task_attention(x[0])
    x_seg = [seg_feat, x[1], x[2]]
    x_depth = [depth_feat, x[1], x[2]]
    
    # 3. 分割头推理 (eval 模式，返回标准格式)
    seg_outputs = Segment26.forward(self, x_seg)
    
    # 4. 深度头推理
    depth = self.depth_decoder(x)
    depth = torch.sigmoid(depth) * 100.0
    
    # 5. 训练/推理分支
    if self.training:
        seg_outputs["depth"] = depth
        return seg_outputs
    else:
        return seg_outputs  # 标准分割输出，depth 存储在 _last_depth
```

### 6.2 分割推理

使用标准 `SegmentationPredictor`：

```python
results = model.predict(image, conf=0.25, task="segment")
# 输出: Results(boxes=..., masks=..., names=80 COCO classes)
```

### 6.3 深度推理

深度图通过 patched forward 存储在 head 实例中：

```python
_ = model.model(image)  # 触发 forward，depth 存储在 head._last_depth
depth_map = head._last_depth  # (1, 640, 640), [0, 100] meters
```

### 6.4 联合推理 (infer_depth_fixed.py)

```bash
python infer_depth_fixed.py \
    --model best.pt \
    --source image.jpg \
    --conf 0.25
```

输出文件：
- `segmentation.jpg`：分割可视化（框 + mask）
- `depth.jpg`：深度伪彩色图
- `combined.jpg`：原图 + 深度图拼接

---

## 7. 模型参数量

### 7.1 整体统计 (s scale)

```
YOLO26-seg-depth summary (fused): 158 layers, 11,032,093 parameters, 0 gradients, 41.2 GFLOPs
```

### 7.2 各模块参数分布

| 模块 | 参数量 | 占比 | 训练参数量 |
|:----:|:------:|:----:|:----------:|
| Backbone (0-10) | ~3.5M | ~32% | 0 (冻结) |
| Neck/FPN (11-22) | ~2.8M | ~25% | 0 (冻结) |
| Segment26 Head | ~3.1M | ~28% | 0 (冻结) |
| TaskDecouplingAttention | ~0.1M | ~1% | ~0.05M (depth_branch) |
| MultiScaleDepthDecoder | ~1.5M | ~14% | ~1.5M (全部) |
| **总计** | **~11.0M** | **100%** | **~1.55M (14%)** |

---

## 8. 关键设计决策 FAQ

### Q1: 为什么 Backbone 必须冻结？

Backbone 是分割和深度两个任务的**共享特征提取器**。COCO 预训练让 Backbone 学到通用的边缘、纹理、形状特征，这些对分割至关重要。如果解冻 Backbone 用于深度训练，特征分布会发生偏移，导致：
- 分割头权重（基于原特征分布训练）不再适用
- 实验验证：exp10 解冻 Backbone 后，COCO 分割置信度从 **0.95 → 0.06**

### Q2: 为什么 seg_branch 也要冻结？

seg_branch 在预训练权重中近似**恒等映射**（Sigmoid 输出 ≈ 0.5，即 `x * 0.5 ≈ x`）。如果允许训练：
- 它会学出对深度有利的注意力模式
- 但这会抑制分割需要的边缘/纹理信息
- 导致 P3 特征质量下降，分割性能退化

### Q3: depth_branch 学到了什么？

depth_branch 从冻结的 COCO 特征中提取深度相关线索：
- **纹理梯度**：远处纹理更密集 → 深度线索
- **相对大小**：已知物体的大小 → 绝对深度
- **遮挡关系**：谁挡住谁 → 相对深度
- **透视几何**：平行线收敛 → 距离感

### Q4: 冻结 Backbone 会导致深度精度不够吗？

理论上会，但实践中：
- COCO 预训练特征包含丰富的几何/纹理信息，对深度估计已有很大帮助
- exp12 的深度 loss 从 **13.74 → 0.78**，证明可学习性良好
- 如果深度精度仍不满足需求，可考虑：
  - 两阶段训练（Stage 2 低学习率微调 Backbone 最后几层）
  - Backbone 插入 Adapter/LoRA 模块
  - 使用带伪深度标签的 COCO 数据联合训练

### Q5: 为什么使用 end2end 模式？

YAML 中 `end2end: True` 启用端到端检测：
- 同时预测 `one2many`（辅助训练）和 `one2one`（推理用）
- `one2one` 分支减少 NMS 后处理依赖
- 提升推理速度和精度

---

## 9. 文件对应关系

| 文件 | 作用 |
|:----:|:-----|
| `yolo26-seg-depth.yaml` | 模型架构配置 |
| `ultralytics/nn/modules/head.py` | DepthSegment26、TaskDecouplingAttention、MultiScaleDepthDecoder 定义 |
| `ultralytics/nn/tasks.py` | SegmentationModel、模型加载逻辑 |
| `ultralytics/utils/loss.py` | DepthSegmentationLoss、MultiScaleDepthLoss |
| `ultralytics/models/yolo/segment/train.py` | DepthSegmentTrainer、DepthSegmentValidator |
| `yolo26_train_depth.py` | 训练入口、ProgressiveFreezeCallback |
| `infer_depth_fixed.py` | 推理入口、forward patch |
| `nyu_yolo/nyu_depth_seg.yaml` | NYU 深度+分割数据集配置 |

---

*文档生成时间: 2026-05-07*
*对应代码版本: yolo26-seg-depth exp12*
