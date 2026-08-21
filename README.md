# SCU 教务处验证码识别

SCU教务处（zhjw）验证码的 CNN 识别模型。

## 验证码特征

- 尺寸：180×60 RGB
- 字符数：4 位
- 字符集：`2345678abcdefgmnpwxy`（20 类，去除了 0/1/9/o/l/q/t/u/v 等易混淆字符）
- 干扰：黑色短线 + 背景渐变

## 模型版本（两档部署 + 一档备选）

| 版本 | 架构 | 导出文件 | 大小 | 测试集整图准确率 | 适用场景 |
|------|------|----------|:----:|:---------------:|----------|
| **窄版 66KB** | (20,32,48,48)/fc96 | `zhjw-model.scuocr` | 66 KB | **99.00%** | **默认主力**（体积/精度/速度平衡）✅ |
| 原版 107KB | (24,40,64,64)/fc120 | `zhjw-model.int8.scuocr` | 107 KB | **99.60%** | 精度优先（回退选项） |
| 极窄 33KB | (20,32,48,48)/fc96 + QAT int4 | `zhjw-model.qat.scuocr`（可选导出） | 33 KB | 97.20% | 极致压缩，接受掉点 |

> 三者均部署为 **int8 v2 `.scuocr`** 格式，与浏览器 npm 包兼容
> （窄版需同步修改 `@scu-plus/zhjw-captcha-ocr` 中 `src/model.ts` 写死的通道数）。

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

宽度可通过 `--widths w1,w2,w3,w4` / `--fc-width` 调整（窄版为 `20,32,48,48` + fc96）。

## 性能

| 指标 | 原版 | 窄版 |
|------|:----:|:----:|
| 测试集整图准确率 | **99.60%** | **99.00%** |
| 测试集单字符准确率 | **99.90%** | **99.75%** |
| 模型总参数量 | 110,244 | 67,372 |
| 导出 .scuocr 大小（int8） | **107 KB** | **66 KB** |
| int8 量化 | 无损（99.60%） | 无损（99.00%） |
| 单张推理速度（CPU） | 0.680 ms | **0.627 ms** |

> 注：模型训练于公开数据集（10,000 张，80% 训练 / 10% 验证 / 10% 测试）。
> int8 量化（BN 折叠 + 对称量化）实测准确率无损。
> 速度数据来自 `tmp/bench_speed.py`（CPU 单张延迟，批量下窄版优势更大）。

### 已探索/已排除的方案（避免重复劳动）

- ❌ **剪枝**：结构性通道剪枝直接崩（<1%）；非结构化剪枝不减少文件大小
- ❌ **深度可分离卷积**：参数量 -55%，精度保不住 99%+
- ❌ **知识蒸馏**：4 种配置全部大幅劣化（~50%），小数据集软标签噪声大
- ❌ **EMA 权重平均**：实测反而 -1.8%（97.5% vs 99.3%）
- ❌ **混合精度 int4（QAT per-channel）**：33KB / 97.20%（原版 99.20%），掉点超 2%
- ❌ **强增强 / SGDR / 大 batch**：均不如原版弱增强 + 单程 cosine
- ✅ **窄架构 + fc 加宽**：`(20,32,48,48)/fc96` 减 38% 体积仅掉 0.6%（**最终方案**）

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
├── model.py             # CaptchaCNN 模型定义（支持 --widths / --bn-momentum）
├── preprocess.py        # 预处理流水线 + ZhjwCaptchaDataset + 数据增强（default/strong 两档）
├── train.py             # 训练脚本（含 TensorBoard、余弦退火、断点续训、SGDR/早停可选）
├── train_qat.py         # QAT 量化感知训练（per-channel int4/int8，--eval-quant/--export）
├── quantize.py          # int8 量化（BN 折叠 + 对称量化）+ 窄模型 FoldedCaptchaCNN
├── quantize_mixed.py    # 混合精度量化（支持 per-channel，--widths）
├── export.py            # 导出 .scuocr 格式（fp32/int8/混合精度，支持 per-channel）
├── requirements.txt     # Python 依赖
├── zhjw-model.scuocr            # 窄版 int8 模型（66 KB，99.00%，默认主力）
├── zhjw-model.int8.scuocr       # 原版 int8 模型（107 KB，99.60%）
├── data/                # 训练数据（gitignore）
│   ├── IMAGES/          # 验证码图片
│   ├── IMAGES.zip       # 图片压缩包
│   └── label.csv        # 标签文件
├── checkpoints/         # 模型权重（gitignore）
│   ├── best.pt          # 原版最佳模型（fp32，int8 无损）
│   ├── best.narrow.pt   # 窄版最佳模型（(20,32,48,48)/fc96）
│   └── (实验权重已归档至 backups/)
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
# 默认训练（200 epochs，原版弱增强，当前最优配置）
python train.py

# 窄版（66KB 模型，500 轮收敛到 99%）
python train.py --widths 20,32,48,48 --fc-width 96 \
  --epochs 500 --batch-size 64 --lr 3e-4 --weight-decay 5e-5

# 断点续训
python train.py --resume checkpoints/latest.pt

# 仅测试（使用 model 权重）
python train.py --test-only checkpoints/best.pt

# QAT int4 微调（窄版导出 33KB）
python train_qat.py --resume checkpoints/best.narrow.pt --widths 20,32,48,48 --fc-width 96 \
  --epochs 40 --batch-size 128 --lr 2e-4 --eval-quant --export
```

### 导出

```bash
# 窄版 int8 导出（默认主力 66KB）
python export.py checkpoints/best.narrow.pt --int8 -o zhjw-model.scuocr

# 原版 int8 导出（107KB）
python export.py checkpoints/best.pt --int8 -o zhjw-model.int8.scuocr

# fp32 导出（version=1）
python export.py checkpoints/best.narrow.pt -o zhjw-model.fp32.scuocr

# QAT int4 导出（version=3，per-channel，33KB 备选）
python export.py --mixed checkpoints/qat/best.qat-int4.pt -o zhjw-model.qat.scuocr
```