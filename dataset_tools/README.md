# YOLO26-Seg-Depth Dataset Tools

本目录包含 YOLO26-Seg-Depth 项目所需的数据集下载、转换与处理脚本。

---

## 脚本清单

| 脚本 | 功能 | 对应数据集 |
|------|------|-----------|
| `download_kitti.py` | 下载 KITTI Depth Prediction + Raw sync RGB，转换为 YOLO 格式 | KITTI |
| `fix_kitti_structure.py` | 修复 KITTI Raw sync 解压后的目录嵌套问题 | KITTI |
| `nyu_mat_converter.py` | 将 NYU Depth V2 `.mat` 文件转换为 YOLO 格式 | NYU Depth V2 |

---

## 1. NYU Depth V2 转换 (`nyu_mat_converter.py`)

### 功能
将官方 NYU Depth V2 `nyu_depth_v2_labeled.mat`（MATLAB v7.3 / HDF5 格式）转换为 YOLO 格式目录结构：

```
nyu_yolo/
├── images/train/     # RGB 图像 (jpg)
├── images/test/      # RGB 图像 (jpg)
├── depths/train/     # 深度图 (16-bit PNG, mm)
├── depths/test/
├── labels/train/     # YOLO bbox 标注
├── labels/test/
├── segments/train/   # YOLO 分割标注
├── segments/test/
└── nyu_depth_seg.yaml
```

### 前置条件
- 下载 `nyu_depth_v2_labeled.mat` (~2.8GB)
  - 来源：https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- Python 依赖：`h5py`, `numpy`, `opencv-python`, `tqdm`

### 用法

```bash
# 基本转换（默认 13 类）
python nyu_mat_converter.py \
    --mat-path /path/to/nyu_depth_v2_labeled.mat \
    --output ./nyu_yolo

# 使用 40 类（官方原始类别）
python nyu_mat_converter.py \
    --mat-path /path/to/nyu_depth_v2_labeled.mat \
    --output ./nyu_yolo \
    --nc 40

# 验证转换结果
python nyu_mat_converter.py --verify --output ./nyu_yolo
```

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mat-path` | `.mat` 文件路径 | 必填 |
| `--output` | 输出目录 | `./nyu_yolo` |
| `--nc` | 类别数（13 或 40） | `13` |
| `--train-ratio` | 训练集比例 | `0.8` |
| `--verify` | 验证数据集完整性 | - |

### NYU 数据特点
- **场景**：室内
- **深度范围**：0~10 m
- **深度密度**：密集（每个像素都有深度值）
- **训练样本**：~1,160（80% split）
- **推荐 depth_scale**：`20.0`

---

## 2. KITTI Depth 下载与转换 (`download_kitti.py`)

### 功能
1. 下载 KITTI Depth Prediction 数据集（深度标注）
2. 下载 KITTI Raw sync 数据（RGB 原始图像）
3. 按场景配对 RGB + Depth
4. 转换为 YOLO 格式目录结构：

```
kitti_yolo_final/
├── images/train/     # RGB 图像 (png)
├── images/val/
├── depths/train/     # 深度图 (16-bit PNG, mm, 稀疏)
├── depths/val/
├── labels/train/     # YOLO bbox 标注（可选）
├── labels/val/
├── segments/train/   # 占位分割标注
├── segments/val/
└── kitti_depth_seg.yaml
```

### 前置条件
- Python 依赖：`numpy`, `opencv-python`, `tqdm`
- 磁盘空间：
  - 完整数据集：~80GB（深度 14GB + Raw sync 66GB）
  - 10% 采样：~10-15GB

### 用法

#### 步骤 1：下载深度数据集

```bash
# 自动下载 KITTI Depth Prediction (data_depth_annotated.zip, ~14GB)
python download_kitti.py --output ./kitti_yolo

# 若自动下载失败（需注册），手动下载后放置到 kitti_zips/ 再运行：
# 下载地址：https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
python download_kitti.py --output ./kitti_yolo
```

#### 步骤 2：下载 RGB 并转换（完整数据集）

```bash
python download_kitti.py \
    --kitti-root ./kitti_yolo/data_depth_annotated \
    --output ./kitti_yolo_final \
    --download-raw \
    --skip-download
```

#### 步骤 2（推荐）：只下载 10% 场景做快速实验

```bash
python download_kitti.py \
    --kitti-root ./kitti_yolo/data_depth_annotated \
    --output ./kitti_yolo_final \
    --download-raw \
    --skip-download \
    --subset-ratio 0.1
```

#### 只转换已有数据（不下载 Raw）

```bash
# 如果已有 Raw sync RGB 数据
python download_kitti.py \
    --kitti-root ./kitti_yolo \
    --output ./kitti_yolo_final \
    --skip-download
```

#### 验证转换结果

```bash
python download_kitti.py --verify --output ./kitti_yolo_final
```

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--kitti-root` | KITTI 数据根目录（含 train/ 和 val/） | 自动查找 |
| `--output` | YOLO 格式输出目录 | `./kitti_yolo` |
| `--download-dir` | depth zip 下载目录 | `./kitti_zips` |
| `--skip-download` | 跳过 depth zip 下载 | - |
| `--download-raw` | 下载 KITTI Raw sync RGB | - |
| `--raw-download-dir` | Raw sync zip 下载目录 | `./kitti_raw_zips` |
| `--subset-ratio` | 采样比例（仅 train 场景） | `1.0` |
| `--no-labels` | 跳过 2D detection label 转换 | - |
| `--verify` | 验证转换结果 | - |

### KITTI 数据特点
- **场景**：室外驾驶
- **深度范围**：0~80 m
- **深度来源**：LiDAR 稀疏点云（~5% 像素有效）
- **深度密度**：稀疏（无效像素值为 0）
- **训练样本**：~26,000（Eigen train split）
- **推荐 depth_scale**：`80.0`
- **Loss 注意**：需用 `valid_mask = depth_target > 0` 过滤无效深度

### 常见问题

#### Q1: KITTI Raw sync 下载后找不到 RGB？
KITTI Raw sync zip 解压后会多一层日期目录，例如：
```
train/2011_09_26/2011_09_26_drive_0018_sync/image_02/data/  <- RGB
train/2011_09_26_drive_0018_sync/proj_depth/groundtruth/     <- Depth
```
需要运行 `fix_kitti_structure.py` 修复目录结构。

#### Q2: 只想下载部分数据做测试？
使用 `--subset-ratio` 参数，例如 `0.1` 只取 10% 的 train 场景（val 集保持完整）：
```bash
python download_kitti.py ... --subset-ratio 0.1 --download-raw
```

---

## 3. KITTI 目录修复 (`fix_kitti_structure.py`)

### 功能
修复 KITTI Raw sync 解压后的目录嵌套问题，将 `image_02/` 和 `image_03/` 从日期子目录移动到场景根目录，使其与深度数据对齐。

### 用法

```bash
python fix_kitti_structure.py --kitti-root ./kitti_yolo
```

### 修复前 vs 修复后

**修复前：**
```
kitti_yolo/train/
├── 2011_09_26/
│   └── 2011_09_26_drive_0018_sync/
│       ├── image_02/data/          <- RGB 在这里
│       └── image_03/data/
└── 2011_09_26_drive_0011_sync/
    └── proj_depth/groundtruth/     <- Depth 在这里
```

**修复后：**
```
kitti_yolo/train/
└── 2011_09_26_drive_0018_sync/
    ├── image_02/data/              <- RGB 移到这里
    ├── image_03/data/
    └── proj_depth/groundtruth/     <- Depth 已在这里
```

---

## 4. 混合训练：NYU + KITTI

根据 ROADMAP.md，建议的多源训练配置：

```yaml
# 在训练脚本中配置多个数据源
sources:
  - path: ./nyu_yolo
    depth_scale: 20.0
    depth_max: 10.0
    weight: 1.0
  - path: ./kitti_yolo_final
    depth_scale: 80.0
    depth_max: 80.0
    weight: 1.0
```

### 关键差异

| 特性 | NYU | KITTI |
|------|-----|-------|
| 场景 | 室内 | 室外驾驶 |
| 深度范围 | 0~10 m | 0~80 m |
| 深度密度 | 密集 | 稀疏 (~5%) |
| depth_scale | 20.0 | 80.0 |
| 图像尺寸 | 640x480 | ~1242x375 |

### 训练命令示例

```bash
# NYU 单数据集训练
python yolo26_train_depth.py \
    --model yolo26-seg-depth.yaml \
    --data ./nyu_yolo/nyu_depth_seg.yaml \
    --pretrained yolo26s-seg.pt \
    --epochs 100 --batch 16 --device 0

# KITTI 单数据集训练
python yolo26_train_depth.py \
    --model yolo26-seg-depth.yaml \
    --data ./kitti_yolo_final/kitti_depth_seg.yaml \
    --pretrained yolo26s-seg.pt \
    --epochs 100 --batch 8 --device 0
```

---

## 附录：KITTI 数据下载直链

若自动下载失败，可手动下载：

| 文件 | 大小 | 直链 |
|------|------|------|
| data_depth_annotated.zip | ~14 GB | https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip |
| data_depth_velodyne.zip | ~5 GB | https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip |
| Raw sync（按场景） | ~300-800MB/场景 | https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/ |

Raw sync 场景命名格式：
```
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
```

---

## 附录：NYU 数据下载

| 文件 | 大小 | 链接 |
|------|------|------|
| nyu_depth_v2_labeled.mat | ~2.8 GB | https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html |

---

*本 README 对应 YOLO26-Seg-Depth 项目的 ROADMAP.md Phase B（KITTI 引入）与 NYU 数据准备阶段。*
