# YOLO26 多任务学习：分割 + 深度估计 改造计划 (改进版)

## 1. 项目概述

### 目标
将 YOLO26 分割模型改造为单主干多任务网络，同时输出：
- 实例分割结果 (masks)
- 稠密深度图 (depth map)

### 技术路线
基于 ultralytics 多任务扩展机制（参考 Pose26 实现模式），新增 DepthSegment26 头部类，实现分割与深度估计的联合学习。

### 核心改进点 (相对于原方案)
1. **多尺度深度融合**：采用 FPN-style 结构融合 P3/8、P4/16、P5/32 特征
2. **任务解耦注意力**：添加 TaskDecouplingAttention 模块减少任务冲突
3. **动态权重机制**：使用梯度归一化 (GradNorm) 自适应调整损失权重
4. **渐进式训练**：分阶段训练策略，先独立训练深度头再联合微调

---

## 2. 现有代码分析

### 2.1 关键文件

| 文件路径 | 说明 |
|---------|------|
| `ultralytics/ultralytics/cfg/models/26/yolo26-seg.yaml` | YOLO26-seg 模型配置 |
| `ultralytics/ultralytics/nn/modules/head.py` | 检测/分割/姿态头部实现 |
| `ultralytics/ultralytics/nn/modules/block.py` | C3k2、Proto26 等模块 |
| `ultralytics/ultralytics/utils/loss.py` | 损失函数定义 |
| `ultralytics/ultralytics/engine/trainer.py` | 训练器引擎 |

### 2.2 现有头部模式 (Pose26 参考)

```python
# ultralytics/nn/modules/head.py - Pose26
class Pose26(Pose):
    def __init__(self, nc=80, kpt_shape=(17, 3), reg_max=16, ch=()):
        super().__init__(nc, kpt_shape, reg_max, ch)
        # 添加额外的关键点预测头
        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
```

---

## 3. 详细实施计划

### Phase 1: 模型头部修改

#### 3.1 新增 TaskDecouplingAttention 模块

**文件**: `ultralytics/ultralytics/nn/modules/head.py`

**新增内容**:
```python
class TaskDecouplingAttention(nn.Module):
    """任务解耦注意力模块 - 减少分割与深度任务的特征冲突"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.seg_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.depth_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seg_att = self.seg_branch(x)
        depth_att = self.depth_branch(x)
        return x * seg_att, x * depth_att
```

#### 3.2 新增多尺度深度融合模块

**文件**: `ultralytics/ultralytics/nn/modules/head.py`

**新增内容**:
```python
class MultiScaleDepthDecoder(nn.Module):
    """多尺度深度融合解码器 - 参照 FPN 架构"""

    def __init__(self, ch, c_depth=64):
        super().__init__()
        c3, c4, c5 = ch[0], ch[1], ch[2]  # P3/8, P4/16, P5/32

        self.seg_conv = nn.Conv2d(c3, c_depth, 1)   # 降维用于分割
        self.depth_p5 = nn.Sequential(
            nn.Conv2d(c5, c_depth, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        self.depth_p4 = nn.Sequential(
            nn.Conv2d(c4, c_depth, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.depth_p3 = nn.Conv2d(c3, c_depth, 1)

        self.fusion = nn.Sequential(
            Conv(c_depth * 3, c_depth, k=3),
            Conv(c_depth, c_depth // 2, k=3),
            nn.Conv2d(c_depth // 2, 1, k=1)
        )
        self.depth_up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, features):
        p3, p4, p5 = features[0], features[1], features[2]

        d_p5 = self.depth_p5(p5)
        d_p4 = self.depth_p4(p4)
        d_p3 = self.depth_p3(p3)

        fused = torch.cat([d_p5, d_p4, d_p3], dim=1)
        depth = self.fusion(fused)
        depth = self.depth_up(depth)
        return depth
```

#### 3.3 新增 DepthSegment26 头部类 (改进版)

**文件**: `ultralytics/ultralytics/nn/modules/head.py`

**新增内容**:
```python
class DepthSegment26(Segment26):
    """YOLO26 Segment + Depth 多任务头部 (改进版)"""

    def __init__(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, nm, npr, reg_max, end2end, ch)

        # 任务解耦注意力
        self.task_attention = TaskDecouplingAttention(ch[0])

        # 多尺度深度融合解码器
        c_depth = max(ch[0] // 4, 64)
        self.depth_decoder = MultiScaleDepthDecoder(ch, c_depth)

        # 深度预测头 (最终 1x1 卷积已在 decoder 中)
        self.depth_head = nn.Conv2d(c_depth // 2, 1, 1)

    def forward(self, x):
        outputs = super().forward(x)  # (boxes, proto, masks)

        # 任务解耦注意力
        seg_feat, depth_feat = self.task_attention(x[0])

        # 多尺度深度预测
        depth = self.depth_decoder(x)  # 使用原始多尺度特征
        depth = torch.sigmoid(depth) * 100.0  # 归一化到 [0, 100] 米

        return (*outputs, depth)
```

#### 3.4 修改 yolo26-seg-depth.yaml 模型配置

**文件**: `ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml`

**新增内容**:
```yaml
# Parameters
nc: 80  # 分割类别数
depth_nc: 1  # 深度通道数

# 主干网络 (保持不变)
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 2, C3k2, [256, True, 2, 0.25]]
  - [-1, 2, C3k2, [512, True, 2, 0.25]]
  - [-1, 2, C3k2, [1024, True, 2, 0.25]]
  # ... (其他层保持与 yolo26-seg.yaml 一致)

# 头部网络
head:
  - [[16, 19, 22], 1, DepthSegment26, [nc, 32, 256, depth_nc]]  # P3/8, P4/16, P5/32
```

---

### Phase 2: 损失函数实现

#### 3.5 新增深度损失类 (改进版)

**文件**: `ultralytics/ultralytics/utils/loss.py`

**新增内容**:
```python
class SILogLoss(nn.Module):
    """Scale-Invariant Log Depth Loss (MiDaS)"""

    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
        n = diff.numel()
        loss = diff.pow(2).sum() / n - (self.alpha / n) * diff.pow(4).sum()
        return self.beta * torch.sqrt(loss + 1e-8)


class BerHuLoss(nn.Module):
    """反向 Huber 损失 (适于深度估计)"""

    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        c = self.threshold * torch.max(diff)
        mask = diff <= c
        loss = torch.where(mask, diff, (diff.pow(2) + c.pow(2)) / (2 * c + 1e-8))
        return loss.mean()


class MultiScaleDepthLoss(nn.Module):
    """多尺度深度损失 - 在不同分辨率下计算损失"""

    def __init__(self, scales=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.silog = SILogLoss()
        self.berhu = BerHuLoss()

    def forward(self, pred, target):
        total_loss = 0
        for scale in self.scales:
            if scale != 1.0:
                pred_s = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                target_s = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                pred_s, target_s = pred, target
            total_loss += self.silog(pred_s, target_s) + 0.5 * self.berhu(pred_s, target_s)
        return total_loss / len(self.scales)
```

#### 3.6 新增动态权重损失类

**文件**: `ultralytics/ultralytics/utils/loss.py`

**新增内容**:
```python
class GradNormLoss(nn.Module):
    """梯度归一化动态权重机制"""

    def __init__(self, model, loss_names, initial_weights, alpha=1.5):
        super().__init__()
        self.model = model
        self.loss_names = loss_names
        self.initial_weights = [w for w in initial_weights]
        self.alpha = alpha
        self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        self.loss_funcs = nn.ModuleDict({name: None for name in loss_names})

    def update_weights(self, losses):
        total_loss = 0
        for i, name in enumerate(self.loss_names):
            total_loss += self.weights[i] * losses[name]

        # 计算梯度
        total_loss.backward(retain_graph=True)

        # 计算各任务梯度范数
        grad_norms = []
        for name in self.loss_names:
            if hasattr(self.model, 'head'):
                param = list(self.model.head.parameters())[0]
            else:
                param = list(self.model.parameters())[0]
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

        # 归一化权重更新
        with torch.no_grad():
            avg_grad = sum(grad_norms) / len(grad_norms)
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] * (grad_norms[i] / avg_grad) ** self.alpha

        return total_loss
```

#### 3.7 新增 DepthSegmentationLoss (改进版)

**文件**: `ultralytics/ultralytics/utils/loss.py`

**新增内容**:
```python
class DepthSegmentationLoss(v8SegmentationLoss):
    """分割+深度联合损失 (改进版 - 支持动态权重)"""

    def __init__(self, model, depth_weight=0.5, use_gradnorm=False):
        super().__init__(model)
        self.use_gradnorm = use_gradnorm
        self.depth_weight = depth_weight
        self.depth_loss = MultiScaleDepthLoss(scales=[1.0, 0.5])
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        if use_gradnorm:
            self.gradnorm = GradNormLoss(
                model,
                loss_names=['seg', 'depth'],
                initial_weights=[1.0, depth_weight],
                alpha=1.5
            )

    def loss(self, preds, batch):
        seg_loss = super().loss(preds, batch)
        seg_loss_val = seg_loss[0]

        depth_pred = preds.get('depth')
        if depth_pred is not None:
            depth_target = batch.get('depth')
            if depth_target is not None:
                d_loss = self.depth_loss(depth_pred, depth_target)

                if self.use_gradnorm:
                    losses = {'seg': seg_loss_val, 'depth': d_loss}
                    total_loss = self.gradnorm.update_weights(losses)
                    return (total_loss, *seg_loss[1:])
                else:
                    total_loss = seg_loss_val + self.depth_weight * d_loss
                    return (total_loss, *seg_loss[1:])

        return seg_loss
```

---

### Phase 3: 渐进式训练策略

#### 3.8 训练阶段划分

**阶段 1: 深度头预训练** (Epochs 1-50)
- 冻结分割头和主干网络
- 仅训练深度解码器
- 使用较大 batch size，固定学习率 1e-3

**阶段 2: 分割头微调** (Epochs 51-100)
- 解冻分割头，深度头保持训练
- 联合学习，分割权重为主 (seg_weight=1.0, depth_weight=0.3)

**阶段 3: 联合微调** (Epochs 101-150)
- 解冻主干网络 (学习率降低 10x)
- 启用 GradNorm 动态权重
- 全面微调 (seg_weight=1.0, depth_weight=0.5)

#### 3.9 训练配置示例

**文件**: `yolo26_train_depth.py`

**新增内容**:
```python
def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo26-seg-depth.yaml')
    parser.add_argument('--data', type=str, default='depth-seg.yaml')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--depth-weight', type=float, default=0.5)
    parser.add_argument('--use-gradnorm', action='store_true')
    parser.add_argument('--freeze-depth-epochs', type=int, default=50)
    parser.add_argument('--freeze-seg-epochs', type=int, default=50)
    return parser.parse_args()


def train_progressive():
    args = parse_train_args()
    model = YOLO(args.model)

    # 阶段 1: 冻结主干和分割头，仅训练深度头
    if epoch < args.freeze_depth_epochs:
        for name, param in model.model.named_parameters():
            if 'depth' not in name:
                param.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.model.parameters()),
            lr=1e-3, weight_decay=0.05
        )

    # 阶段 2: 解冻分割头，联合训练
    elif epoch < args.freeze_depth_epochs + args.freeze_seg_epochs:
        for name, param in model.model.named_parameters():
            if 'depth' in name:
                param.requires_grad = True
            else:
                param.requires_grad = True
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=5e-4)

    # 阶段 3: 全面微调 + GradNorm
    else:
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)
```

---

### Phase 4: 数据集支持

#### 3.10 修改数据集加载器 (支持深度图)

**文件**: `ultralytics/ultralytics/data/dataset.py`

**修改内容**:
- 在 `SegmentationDataset` 类中添加深度图加载
- 支持 `depth` 字段的读取和归一化 (深度图 / 100.0 归一化到 [0,1])
- 数据增强支持深度图的同步增强 (flip, resize, crop)

#### 3.11 新增数据集配置示例

**文件**: `ultralytics/ultralytics/cfg/datasets/depth-seg.yaml`

```yaml
path: ./depth_seg_dataset
train: images/train
val: images/val
mask: segments/train
depth: depth/train
nc: 80
depth_nc: 1
```

---

### Phase 5: 训练器集成

#### 3.12 修改训练器支持多任务

**文件**: `ultralytics/ultralytics/engine/trainer.py`

**修改内容**:
- 在 `Loss` 类初始化时检测 `depth_nc` 参数
- 动态加载 `DepthSegmentationLoss`
- 支持渐进式训练的阶段切换

---

### Phase 6: 验证与推理

#### 3.13 新增深度评估指标

**文件**: `ultralytics/ultralytics/utils/metrics.py`

**新增内容**:
```python
class DepthMetric:
    """深度估计评估指标"""

    def __init__(self):
        self.abs_rel = []
        self.rmse = []
        self.silog = []

    def update(self, pred, target):
        diff = pred - target
        abs_rel = torch.mean(torch.abs(diff) / (target + 1e-8)).item()
        rmse = torch.sqrt(torch.mean(diff.pow(2))).item()
        silog = torch.sqrt(
            torch.mean((torch.log(pred + 1e-8) - torch.log(target + 1e-8)).pow(2)) -
            torch.mean((torch.log(pred + 1e-8) - torch.log(target + 1e-8))).pow(2)
        ).item()
        self.abs_rel.append(abs_rel)
        self.rmse.append(rmse)
        self.silog.append(silog)

    def compute(self):
        return {
            'abs_rel': np.mean(self.abs_rel),
            'rmse': np.mean(self.rmse),
            'silog': np.mean(self.silog)
        }
```

#### 3.14 修改推理类支持深度输出

**文件**: `yolo26_inference.py`

**新增方法**:
```python
def get_depth(self, result):
    """从结果中提取深度图"""
    return result.depth if hasattr(result, 'depth') else None

def predict_depth_segment(self, source, **kwargs):
    """同时预测分割和深度"""
    results = self.model.predict(source, **kwargs)
    for r in results:
        r.depth = getattr(r, 'depth', None)
    return results
```

---

## 4. 文件修改清单

### 4.1 新增文件

| 文件路径 | 说明 |
|---------|------|
| `ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml` | 多任务模型配置 |
| `ultralytics/ultralytics/cfg/datasets/depth-seg.yaml` | 数据集配置示例 |
| `ultralytics/ultralytics/utils/depth_metrics.py` | 深度评估指标 |
| `ultralytics/ultralytics/data/depth_dataset.py` | 深度+分割数据集加载器 |
| `yolo26_train_depth.py` | 渐进式训练脚本 |

### 4.2 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `ultralytics/ultralytics/nn/modules/head.py` | 新增 TaskDecouplingAttention、MultiScaleDepthDecoder、DepthSegment26 (改进版) |
| `ultralytics/ultralytics/utils/loss.py` | 新增 SILogLoss、BerHuLoss、MultiScaleDepthLoss、GradNormLoss、DepthSegmentationLoss (改进版) |
| `ultralytics/ultralytics/engine/trainer.py` | 集成多任务损失、渐进式训练支持 |
| `ultralytics/ultralytics/utils/metrics.py` | 新增 DepthMetric 类 |
| `yolo26_inference.py` | 新增 get_depth() 方法 |

---

## 5. 训练配置示例

### 5.1 命令行训练 (渐进式 + GradNorm)

```bash
python yolo26_train_depth.py \
    --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \
    --data depth-seg.yaml \
    --epochs 150 \
    --batch 16 \
    --device 0 \
    --depth-weight 0.5 \
    --use-gradnorm \
    --freeze-depth-epochs 50 \
    --freeze-seg-epochs 50
```

### 5.2 多GPU训练

```bash
python -m torch.distributed.run --nproc_per_node 4 yolo26_train_depth.py \
    --model yolo26-seg-depth.yaml \
    --data depth-seg.yaml \
    --epochs 150 \
    --batch 64 \
    --device 0,1,2,3 \
    --use-gradnorm
```

---

## 6. 实施时间线

| Phase | 任务 | 预估时间 |
|-------|------|---------|
| Phase 1 | 模型头部修改 (多尺度融合 + 注意力) | 4-5 天 |
| Phase 2 | 损失函数实现 (动态权重 + 多尺度损失) | 3-4 天 |
| Phase 3 | 渐进式训练策略 | 2-3 天 |
| Phase 4 | 数据集支持 | 3-5 天 |
| Phase 5 | 训练器集成 | 2-3 天 |
| Phase 6 | 验证与推理 | 2-3 天 |
| Phase 7 | 调优与测试 | 1-2 周 |

**总预估时间**: 7-9 周

---

## 7. 风险评估与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 深度标签获取困难 | 高 | 使用 MiDaS/BEiT-Large 预训练模型生成伪标签 |
| 多任务训练不稳定 | 中 | 渐进式训练 + GradNorm 动态权重 |
| 内存占用过高 | 中 | 减少 batch size，使用梯度累积 |
| 主任务(分割)性能下降 | 中 | TaskDecouplingAttention + 渐进式解冻 |
| 深度估计精度不足 | 中 | 多尺度融合 + MultiScaleDepthLoss |

---

## 8. 验证方案

### 8.1 深度估计指标
- Abs Rel (Absolute Relative) < 0.15
- RMSE (Root Mean Square Error) < 5.0
- SILog (Scale-Invariant Log) < 0.2

### 8.2 分割指标
- mAP (保持原有性能不下降，< 2% 差异)
- IoU

### 8.3 联合训练验证
- 对比单任务 vs 多任务的分割性能
- 对比单任务 vs 多任务的深度估计性能
- Ablation: 验证 TaskDecouplingAttention、GradNorm、MultiScale 的贡献

### 8.4 消融实验设计

| 实验 | 配置 | 预期结果 |
|------|------|---------|
| Baseline | 分割单任务 | seg mAP=0.45 |
| Single-Scale | 深度单尺度融合 | depth AbsRel=0.18 |
| Multi-Scale | 深度多尺度融合 | depth AbsRel=0.14 |
| +Attention | + TaskDecouplingAttention | depth AbsRel=0.13 |
| +GradNorm | + GradNorm 动态权重 | depth AbsRel=0.12, seg mAP=0.44 |
| Full | 完整渐进式训练 | depth AbsRel=0.11, seg mAP=0.44 |

---

## 9. 核心改进总结

相对原方案的关键改进：

| 改进点 | 原方案 | 改进后 |
|--------|--------|--------|
| 深度特征 | 仅 P3/8 单尺度 | FPN-style P3+P4+P5 多尺度融合 |
| 任务干扰 | 无处理 | TaskDecouplingAttention 解耦 |
| 损失权重 | 固定 0.5 | GradNorm 动态调整 |
| 训练策略 | 直接联合训练 | 渐进式三阶段训练 |
| 深度损失 | 仅 SILog | SILog + BerHu 多尺度损失 |

**预期效果**: 深度估计 AbsRel 从 ~0.18 降至 ~0.11，分割 mAP 保持 0.44 不下降。

---

## 10. 后续优化方向

1. **跨模态注意力**: 引入 cross-attention 增强分割与深度的特征交互
2. **知识蒸馏**: 从单任务教师模型蒸馏到多任务学生模型
3. **实时性优化**: 适配 TensorRT 部署，深度头上采样简化为双线性
4. **自适应分辨率**: 根据输入图像尺度动态调整深度解码器

---

## 11. 数据集准备指南

### 11.1 NYU Depth V2 数据集 (推荐室内场景)

**下载方式**:
1. 访问 https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. 下载 `nyu_depth_v2_labeled.mat` 文件 (~2.8GB)
3. 下载对应图像 `images.zip` (~12GB)

**目录结构**:
```
nyu_v2/
├── labels/           # .mat 文件包含分割标签
│   ├── 0001_labels.mat
│   └── 0002_labels.mat
└── images/          # 原始 RGB 图像
    ├── train/
    │   ├── 0001.jpg
    │   └── 0002.jpg
    └── val/
```

**预处理命令**:
```bash
# 转换训练集
python -m ultralytics.data.depth_dataset \
    --nyu-root /path/to/nyu_v2 \
    --output-root ./depth_seg_dataset \
    --subset train

# 转换验证集
python -m ultralytics.data.depth_dataset \
    --nyu-root /path/to/nyu_v2 \
    --output-root ./depth_seg_dataset \
    --subset val
```

**转换后目录结构**:
```
depth_seg_dataset/
├── images/
│   ├── train/  # RGB 图像 (640x480)
│   └── val/
├── depths/
│   ├── train/  # 16-bit PNG 深度图 (毫米为单位)
│   └── val/
└── segments/
    ├── train/  # YOLO 格式分割标注 (.txt)
    └── val/
```

### 11.2 KITTI 数据集 (室外自动驾驶场景)

**下载方式**:
```bash
# 下载 KITTI 深度标注
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip

# 解压
unzip data_depth_annotated.zip -d kitti/
```

**KITTI 转 YOLO 格式** (需额外处理):
```bash
# 使用 MiDaS 生成伪深度标签
python scripts/convert_kitti_depth.py \
    --kitti-root ./kitti \
    --output ./kitti_yolo \
    --split train
```

### 11.3 快速验证 (合成数据)

如暂无真实数据，可用以下方式生成合成测试数据:

```python
# scripts/generate_synthetic_depth.py
import numpy as np
import cv2
from pathlib import Path

def generate_synthetic_data(output_dir, count=100, img_size=(640, 480)):
    """生成合成深度+分割测试数据"""
    output_dir = Path(output_dir)
    
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'depths' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'segments' / split).mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            # 生成随机 RGB 图像
            img = np.random.randint(0, 255, (*img_size[::-1], 3), dtype=np.uint8)
            cv2.imwrite(str(output_dir / 'images' / split / f'{i:04d}.jpg'), img)
            
            # 生成随机深度图 (0-10米)
            depth = np.random.uniform(0, 10, img_size[::-1]).astype(np.float32)
            depth_png = (depth * 1000).astype(np.uint16)  # 转换为毫米
            cv2.imwrite(str(output_dir / 'depths' / split / f'{i:04d}.png'), depth_png)
            
            # 生成简单分割标注 (YOLO格式)
            with open(output_dir / 'segments' / split / f'{i:04d}.txt', 'w') as f:
                # 随机生成1-3个物体
                for _ in range(np.random.randint(1, 4)):
                    x, y, w, h = np.random.rand(4)
                    coords = [x - w/2, y - h/2, x + w/2, y - h/2, x + w/2, y + h/2, x - w/2, y + h/2]
                    f.write(f'0 ' + ' '.join([f'{c:.6f}' for c in coords]) + '\n')
    
    print(f'Generated {count*2} synthetic samples in {output_dir}')

if __name__ == '__main__':
    generate_synthetic_data('./test_dataset', count=100)
```

### 11.4 数据集验证脚本

```bash
# 检查数据集是否正确加载
python -c "
from ultralytics.data.depth_dataset import DepthSegmentDataset

ds = DepthSegmentDataset(
    img_path='./depth_seg_dataset/images/train',
    data={'names': {0: 'bg', 1: 'fg'}},
    imgsz=640
)
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'Sample keys: {sample.keys()}')
print(f'Depth shape: {sample[\"depth\"].shape}')
print(f'Depth range: [{sample[\"depth\"].min():.2f}, {sample[\"depth\"].max():.2f}]')
"
```

### 11.5 深度标签获取困难解决方案

如果无法获取真实深度标签，可采用以下方案:

| 方案 | 工具 | 精度 | 速度 |
|------|------|------|------|
| MiDaS | PyTorch | 高 | 慢 |
| DepthAnything | HuggingFace | 高 | 中 |
| ZoeDepth | PyTorch | 高 | 中 |
| DPT | Intel | 高 | 慢 |
| Marigold | PyTorch | 最高 | 很慢 |

**使用 MiDaS 生成伪标签**:
```python
# scripts/generate_depth_pseudo_labels.py
import torch
import cv2
from pathlib import Path

def generate_pseudo_depth(model_name, images_dir, output_dir):
    """使用 MiDaS 生成伪深度标签"""
    midas = torch.hub.load('intel-isl/MiDaS', model_name)
    midas.eval()
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in sorted(images_dir.glob('*.jpg')):
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            prediction = midas(img)
        
        depth = prediction.squeeze().cpu().numpy()
        depth = (depth * 1000).astype(np.uint16)  # 转换为毫米
        
        cv2.imwrite(str(output_dir / f'{img_file.stem}.png'), depth)
        print(f'Generated: {img_file.stem}')
```
