# SCU 教务处验证码识别

四川大学教务处（zhjw）验证码的 CNN 识别模型，专为 scu-plus 浏览器插件部署优化。

## 验证码特征

- 尺寸：180×60 RGB
- 字符数：4 位
- 字符集：`23456789abcdefgmnpwxy`（20 类）
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

参数量约 **110K**，与 scu-plus 的 CaptchaModelLite 同级，适合浏览器插件部署。

## 项目结构

```
scu-zhjw-captcha/
├── model.py             # CaptchaCNN 模型定义
├── preprocess.py        # 预处理流水线 + ZhjwCaptchaDataset
├── train.py             # 训练脚本（含 TensorBoard、余弦退火、断点续训）
├── export.py            # 导出 .scuocr 格式（供 scu-plus 插件使用）
├── requirements.txt     # Python 依赖
├── data/                # 训练数据（gitignore）
│   ├── IMAGES/          # 验证码图片
│   ├── IMAGES.zip       # 图片压缩包
│   └── label.csv        # 标签文件
├── checkpoints/         # 模型权重（gitignore）
│   ├── best.pt          # 最佳模型
│   └── latest.pt        # 最新模型
├── backups/             # 旧代码备份（gitignore）
└── tmp/                 # 参考项目（gitignore）
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
# 默认训练（200 epochs）
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
python export.py checkpoints/best.pt -o model.scuocr
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
  → 黑线检测 (RGB<130) 并填背景色
  → G通道反色 (1 - G/255)
  → 调亮 (brighten, 阈值0.3)
  → 减去预计算列校正曲线（补偿背景渐变）
  → 再次调亮 + 低值剪切
  → 缩放到 64×32
```

## 部署目标

模型导出为 `.scuocr` 格式后，集成到 [scu-plus](https://github.com/The-Brotherhood-of-SCU/scu-plus) 浏览器插件中，在浏览器端通过 TypeScript 实现前向推理，无需 PyTorch 运行时。
