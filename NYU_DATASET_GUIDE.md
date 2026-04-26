# NYU Depth V2 Dataset Preparation Guide

## 下载 NYU 数据集

1. 访问 https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. 注册并下载 "NYU Depth V2 Labeled" 数据集 (约 3GB)
3. 下载内容包括:
   - `nyu_depth_v2_labeled.mat` - 包含图像、深度、分割标签
   - images.zip - RGB 图像 (可选, mat文件中已有)

## 目录结构

下载后，解压到如下结构:

```
nyu_raw/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── val/
│       ├── 0001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── 0001_labels.mat
    │   └── ...
    └── val/
        ├── 0001_labels.mat
        └── ...
```

## 运行转换脚本

```bash
# 转换训练集
python nyu_download_convert.py --convert \
    --nyu-root ./nyu_raw \
    --output ./nyu_yolo \
    --subset train

# 转换验证集
python nyu_download_convert.py --convert \
    --nyu-root ./nyu_raw \
    --output ./nyu_yolo \
    --subset val

# 创建数据集配置 YAML
python nyu_download_convert.py --convert \
    --nyu-root ./nyu_raw \
    --output ./nyu_yolo \
    --subset all \
    --create-yaml
```

## 转换后的数据格式

```
nyu_yolo/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── val/
├── depths/           # 深度图 (16-bit PNG, 毫米单位)
│   ├── train/
│   └── val/
├── segments/         # YOLO 分割标注 (.txt)
│   ├── train/
│   └── val/
└── nyu_depth_seg.yaml  # 数据集配置
```

## 使用 yolo26_train.py 进行分割训练 (无深度)

如果只训练分割，不需要深度:

```bash
python yolo26_train.py \
    --model yolo26-seg.yaml \
    --data nyu_yolo/nyu_depth_seg.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0
```

## 使用 yolo26_train_depth.py 进行深度+分割联合训练

需要训练深度+分割双任务:

```bash
python yolo26_train_depth.py \
    --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \
    --data nyu_yolo/nyu_depth_seg.yaml \
    --pretrained yolo26s-seg.pt \
    --epochs 150 \
    --batch 16 \
    --device 0 \
    --depth-weight 0.5 \
    --freeze-depth-epochs 50 \
    --freeze-seg-epochs 50 \
    --use-gradnorm
```

## 使用 COCO 预训练权重 + NYU 微调

```bash
python yolo26_train_depth.py \
    --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \
    --data nyu_yolo/nyu_depth_seg.yaml \
    --pretrained yolo26s-seg.pt \
    --epochs 100 \
    --batch 16 \
    --device 0 \
    --depth-weight 0.3 \
    --freeze-depth-epochs 30 \
    --freeze-seg-epochs 30 \
    --use-gradnorm
```

## 数据集统计

| 子集 | 图像数 | 深度图 | 分割标注 |
|------|--------|--------|----------|
| train | ~800 | ✅ | ✅ |
| val | ~400 | ✅ | ✅ |

NYU 数据集规模较小，建议:
1. 使用 `--pretrained` 加载 COCO 预训练权重
2. 启用强数据增强 (mosaic=1.0, mixup=0.2)
3. 考虑引入 KITTI 数据扩充


python yolo26_train_depth.py --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml --data nyu_yolo/nyu_depth_seg.yaml --pretrained yolo26s-seg.pt --epochs 100 --batch 2 --imgsz 480 --device 0 --depth-weight 0.5 --freeze-depth-epochs 30 --freeze-seg-epochs 20 --workers 2 --project runs/train_nyu_depth --name exp1