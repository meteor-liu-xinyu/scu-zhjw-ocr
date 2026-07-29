# SCU 教务处验证码识别

四川大学教务处（zhjw）验证码的 CNN 识别模型，专为 **scu-plus** 浏览器插件部署优化。

## 验证码特征

- 尺寸：180×60 RGB
- 字符数：4 位
- 字符集：`23456789abcdefgmnpwxy`（20 类，去除了 0/1/o/l/q/t/u/v 等易混淆字符）
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

参数量约 **110K**（~66KB 模型大小），与 scu-plus 的 CaptchaModelLite 同级，适合浏览器插件部署。

## 性能

| 指标 | 值 |
|------|:--:|
| 验证集整图准确率 | **99.5%** |
| 全连接层参数量 | 40,520 |
| 卷积层参数量 | 69,144 |
| 模型总参数量 | **110,240** |
| 导出 .scuocr 大小 | **431 KB** |

> 注：实际网站实测准确率约 91%，通过简化预处理 + 数据增强（平移/缩放/旋转）后重新训练可进一步提升。

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
├── backups/             # 旧模型备份（gitignore）
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

### 导出模型

```bash
# 导出为 .scuocr 格式（供 scu-plus 插件使用）
python export.py checkpoints/best.pt -o zhjw-model.scuocr
```

导出的 `.scuocr` 文件为自定义二进制格式，包含：
- 8 字节 magic (`SCUOCRLT`)
- 版本号 + tensor 数量
- 各 tensor 的名称、形状、float32 数据

注意：模型中 Conv 层使用 `bias=False`，但导出时会自动零填充 bias 张量，以兼容 scu-plus 插件的 BN 折叠逻辑。

## 预处理流水线

```
原图 180×60 BGR
  → 裁剪中间区域 (40:140, 5:55) → 100×50
  → 黑线检测 (RGB < 130) 填固定背景色 RGB(225,222,222)
  → 灰度化 + 反色: gray = cvtColor(RGB→Gray)/255, result = 1 - gray
  → 缩放到 64×32 (INTER_AREA)
```

对比旧版（移除了动态背景色计算、G 通道、brighten、固定校正曲线），简化后泛化性能更好。

## 部署到 SCU Plus

### 需要的文件

将以下文件放入 scu-plus 项目的 `assets/` 目录：

| 文件 | 说明 | 大小 |
|------|------|:----:|
| `zhjw-model.scuocr` | 训练好的模型权重（二进制格式） | ~431 KB |

插件前端需要配套的 **预处理逻辑**（已在 `model.ts` 中实现）：
1. 裁剪 `[40:140, 5:55]`
2. 黑线检测：RGB < 130 → 填 `(225,222,222)`
3. 灰度化 `RGB→Gray`
4. 反色 `1 - gray/255`
5. 缩放到 64×32

### 导出命令

```bash
python export.py checkpoints/best.pt -o assets/zhjw-model.scuocr
```
```

## 部署目标

模型导出为 `.scuocr` 格式后，集成到 [scu-plus](https://github.com/The-Brotherhood-of-SCU/scu-plus) 浏览器插件中，在浏览器端通过 TypeScript 实现前向推理，无需 PyTorch 运行时。
