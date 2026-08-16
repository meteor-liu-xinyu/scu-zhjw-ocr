# SCU 教务处验证码识别

SCU教务处（zhjw）验证码的 CNN 识别模型。

## 验证码特征

- 尺寸：180×60 RGB
- 字符数：4 位
- 字符集：`2345678abcdefgmnpwxy`（20 类，去除了 0/1/9/o/l/q/t/u/v 等易混淆字符）
- 干扰：黑色短线 + 背景渐变

## 模型架构

4 层 CNN + SE 注意力 + AdaptiveAvgPool 保留水平位置信息：

```
Input: 1×32×64 (灰度, [0,1] 归一化)
  → Conv3×3(1→24)  + BN + ReLU + MaxPool  →  24×16×32
  → Conv3×3(24→40) + BN + ReLU + MaxPool  →  40×8×16
  → Conv3×3(40→64) + BN + ReLU + MaxPool  →  64×4×8
  → Conv3×3(64→64) + BN + ReLU + MaxPool  →  64×2×4
  → SE 注意力（通道重标定，+580 参数）
  → AdaptiveAvgPool(1,4) → 保留4个字符位置
  → Flatten → 256
  → Linear(256→120) + ReLU + Dropout(0.3)
  → Linear(120→80)   ← 4位 × 20类
```

参数量约 **110K**。

## 性能

| 指标 | 值 |
|------|:--:|
| 测试集整图准确率 | **99.6%** |
| 测试集单字符准确率 | **99.90%** |
| 模型总参数量 | **110,244** |
| 导出 .scuocr 大小（fp32） | **431 KB** |
| 导出 .scuocr 大小（int8） | **107 KB**（无损） |
| 导出 .scuocr 大小（混合精度） | **88 KB**（99.20%） |

> 注：模型训练于公开数据集（10,000 张，80% 训练 / 10% 验证 / 10% 测试）。
> int8 量化（BN 折叠 + 对称量化）实测准确率无损（99.60% → 99.60%）。
> **最终方案：混合精度（Conv int8 + FC/SE int4）88 KB，精度 99.20%**，
> 是大小与精度的最优平衡。曾尝试深度可分离卷积、知识蒸馏、通道剪枝、
> 剪枝稀疏存储等方案，均无法在保持 99%+ 精度的同时超越混合精度。

## 预处理流水线

```
原图 180×60 BGR
  → 裁剪中间区域 (40:140, 5:55) → 100×50
  → 黑线检测 (RGB < 130) → 填固定背景色 (225,222,222)
  → 灰度化: gray = 0.299R + 0.587G + 0.114B
  → 反色: result = 1 - gray/255
  → 缩放到 64×32 (INTER_AREA)
```

训练时附加数据增强：几何变换（平移 ±1px、缩放 ±5%、旋转 ±2°）+ 弹性形变 + 亮度/对比度扰动 + 笔画粗细扰动 + 高斯噪声 + 随机局部遮挡。

## 相关仓库

- **部署包**（浏览器端 CNN 推理 npm 包，内置 int8 权重）：[@scu-plus/zhjw-captcha-ocr](https://github.com/meteor-liu-xinyu/zhjw-captcha-ocr)

## 项目结构

```
scu-zhjw-captcha/
├── model.py             # CaptchaCNN 模型定义
├── preprocess.py        # 预处理流水线 + ZhjwCaptchaDataset + 数据增强
├── train.py             # 训练脚本（含 TensorBoard、余弦退火、断点续训）
├── quantize.py          # int8 量化（BN 折叠 + 对称量化）
├── quantize_mixed.py    # 混合精度量化（Conv int8 + FC/SE int4）
├── export.py            # 导出 .scuocr 格式（fp32/int8/混合精度）
├── requirements.txt     # Python 依赖
├── zhjw-model.scuocr        # fp32 模型文件
├── zhjw-model.mixed.scuocr  # 混合精度模型文件（最终方案，88 KB）
├── data/                # 训练数据（gitignore）
│   ├── IMAGES/          # 验证码图片
│   ├── IMAGES.zip       # 图片压缩包
│   └── label.csv        # 标签文件
├── checkpoints/         # 模型权重（gitignore）
│   ├── best.pt          # 最佳模型（fp32，混合精度来源）
│   └── best.mixed.pt    # 混合精度模型（93.5 KB）
└── runs/                # TensorBoard 日志
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

从 [SCU_OAA-website-Captcha-training-set](https://github.com/SunnyHaze/SCU_OAA-website-Captcha-training-set) 下载 `IMAGES.zip` 和 `label.csv` 到 `data/` 目录。

### 训练

```bash
# 默认训练（200 epochs，带数据增强）
python train.py

# 自定义参数
python train.py --epochs 100 --batch-size 64 --lr 3e-4

# 断点续训
python train.py --resume checkpoints/latest.pt

# 仅测试（使用 model 权重）
python train.py --test-only checkpoints/best.pt
```

### 导出

```bash
# fp32 导出（version=1）
python export.py checkpoints/best.pt -o zhjw-model.scuocr

# int8 量化导出（version=2，BN 折叠 + 对称量化，约 1/4 大小，无损）
python export.py checkpoints/best.pt --int8 -o zhjw-model.int8.scuocr
```