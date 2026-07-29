# SCU 教务处验证码识别

SCU教务处（zhjw）验证码的 CNN 识别模型。

## 验证码特征

- 尺寸：180×60 RGB
- 字符数：4 位
- 字符集：`2345678abcdefgmnpwxy`（20 类，去除了 0/1/9/o/l/q/t/u/v 等易混淆字符）
- 干扰：黑色短线 + 背景渐变

## 模型架构

4 层 CNN + AdaptiveAvgPool 保留水平位置信息：

```
Input: 1×32×64 (灰度, [0,1] 归一化)
  → Conv3×3(1→24)  + BN + ReLU + MaxPool  →  24×16×32
  → Conv3×3(24→40) + BN + ReLU + MaxPool  →  40×8×16
  → Conv3×3(40→64) + BN + ReLU + MaxPool  →  64×4×8
  → Conv3×3(64→64) + BN + ReLU + MaxPool  →  64×2×4
  → AdaptiveAvgPool(1,4) → 保留4个字符位置
  → Flatten → 256
  → Linear(256→120) + ReLU + Dropout(0.3)
  → Linear(120→80)   ← 4位 × 20类
```

参数量约 **110K**。

## 性能

| 指标 | 值 |
|------|:--:|
| 测试集整图准确率 | **99.0%** |
| 测试集单字符准确率 | **99.75%** |
| 模型总参数量 | **110,240** |
| 导出 .scuocr 大小 | **431 KB** |

> 注：模型训练于公开数据集（10,000 张，80% 训练 / 10% 验证 / 10% 测试）。

## 预处理流水线

```
原图 180×60 BGR
  → 裁剪中间区域 (40:140, 5:55) → 100×50
  → 黑线检测 (RGB < 130) → 填固定背景色 (225,222,222)
  → 灰度化: gray = 0.299R + 0.587G + 0.114B
  → 反色: result = 1 - gray/255
  → 缩放到 64×32 (INTER_AREA)
```

训练时附加数据增强：随机平移 ±1px、缩放 ±5%、旋转 ±2°。

## 项目结构

```
scu-zhjw-captcha/
├── model.py             # CaptchaCNN 模型定义
├── preprocess.py        # 预处理流水线 + ZhjwCaptchaDataset + 数据增强
├── train.py             # 训练脚本（含 TensorBoard、余弦退火、断点续训）
├── export.py            # 导出 .scuocr 格式（供 scu-plus 插件使用）
├── requirements.txt     # Python 依赖
├── zhjw-model.scuocr    # 导出的模型文件（直接供插件加载）
├── data/                # 训练数据（gitignore）
│   ├── IMAGES/          # 验证码图片
│   ├── IMAGES.zip       # 图片压缩包
│   └── label.csv        # 标签文件
├── checkpoints/         # 模型权重（gitignore）
│   ├── best.pt          # 最佳模型
│   └── latest.pt        # 最新模型
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

# 仅测试
python train.py --test-only checkpoints/best.pt
```