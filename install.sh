#!/bin/bash
# YOLO26 环境安装脚本
# 使用方法: bash yolo26_install.sh

echo "=========================================="
echo "YOLO26 环境安装"
echo "=========================================="

# 激活 conda 环境
eval "$(/hpc2ssd/softwares/anaconda3/bin/conda shell.bash hook)"

# 如果环境名作为参数传入
if [ -n "$1" ]; then
    conda activate "$1"
else
    conda activate kk_lm
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo "Python 版本: $(python --version)"

# 安装 ultralytics
echo ""
echo "安装 ultralytics..."
pip install ultralytics -q

# 检查安装是否成功
if pip show ultralytics > /dev/null 2>&1; then
    echo "✓ ultralytics 安装成功"
    pip show ultralytics | grep Version
else
    echo "✗ ultralytics 安装失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "运行推理示例:"
echo "  python yolo26_inference.py --source test.jpg --model yolo26/yolo26s-seg.pt"