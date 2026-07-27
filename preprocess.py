"""
验证码预处理：裁剪 → 去黑线 → G 反色 → 调亮 → 固定校正 → 再调亮。

流水线：
  1. 裁剪中间区域 (40:140, 5:55) → 100×50
  2. 计算背景色（白底 r,g,b>200 均值 ≈ RGB(228,228,226)）
  3. 黑线检测（r,g,b<130）并填背景色
  4. G/255 → 1 - G/255（黑底白字）
  5. brighten(0.3)：调亮白字（增强对比）
  6. 减去预计算列校正曲线（固定背景渐变补偿）
  7. re-brighten(0.10)：再次调亮白字 + clip(<0.08→0)
  8. 缩放到模型输入尺寸（在 Dataset 中完成）

输出形状 (50, 100)，float32 范围 [0, 1]。
数据集加载：从 data/ 目录读取 IMAGES/ + label.csv。
"""

import os
import zipfile
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import INPUT_H, INPUT_W, INPUT_C, CHARSET

# ── 裁剪参数 ────────────────────────────────────────────────────────────
_CROP_X1 = 40
_CROP_X2 = 140
_CROP_Y1 = 5
_CROP_Y2 = 55

# ── 固定校正曲线（预计算：500 张 brighten 后全图列均值 - 右背景目标）────
# 在全图 180×60 上算列均值，取右侧列 (140:178) 均值为背景目标，再切片到裁剪区 [40:140]
_BG_TARGET = 0.065200
_CORRECTION = np.array([
    0.136650, 0.140565, 0.140974, 0.141100, 0.139719, 0.144202, 0.156741, 0.153118, 0.160793, 0.168243,
    0.181943, 0.194972, 0.217015, 0.234975, 0.266573, 0.284089, 0.305024, 0.310572, 0.318055, 0.305905,
    0.306157, 0.290574, 0.292807, 0.295320, 0.301726, 0.303680, 0.313717, 0.305262, 0.298572, 0.286149,
    0.281810, 0.272559, 0.271628, 0.275085, 0.287481, 0.300872, 0.306269, 0.297088, 0.298043, 0.292781,
    0.284954, 0.278746, 0.285313, 0.287081, 0.286769, 0.300760, 0.303747, 0.302743, 0.289216, 0.272951,
    0.258800, 0.257800, 0.267395, 0.265605, 0.286212, 0.297500, 0.283114, 0.288412, 0.282010, 0.264512,
    0.260676, 0.272118, 0.279063, 0.286477, 0.289157, 0.288302, 0.276864, 0.265861, 0.247242, 0.237360,
    0.239036, 0.242310, 0.239622, 0.256232, 0.261821, 0.260663, 0.252387, 0.251900, 0.242533, 0.243439,
    0.238571, 0.245479, 0.249336, 0.256124, 0.252138, 0.244173, 0.225730, 0.194886, 0.156793, 0.126954,
    0.103580, 0.081222, 0.070322, 0.058305, 0.052665, 0.046066, 0.037536, 0.036454, 0.030858, 0.026419,
], dtype=np.float64)

# ── 预处理常数 ──────────────────────────────────────────────────────────
_BLACK_THRESH = 130       # 黑线检测阈值（RGB 三通道均低于此值视为黑线）
_WHITE_THRESH = 200       # 白底检测阈值（用于计算背景色）
_BRIGHT_THRESH = 0.3      # 第一次调亮白字阈值（1-G/255 空间）
_REBRIGHT_THRESH = 0.10   # 第二次调亮白字阈值
_BRIGHT_S = 0.3            # 调亮曲线幂指数（<1 使亮部更亮）


def _get_background_color(rgb: np.ndarray) -> tuple[int, int, int]:
    """计算图片中白底区域的平均颜色。"""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    white = (r > _WHITE_THRESH) & (g > _WHITE_THRESH) & (b > _WHITE_THRESH)
    if white.sum() == 0:
        return (255, 255, 255)
    bg = rgb[white].mean(axis=0)
    return (int(bg[0]), int(bg[1]), int(bg[2]))


def _brighten(img: np.ndarray, thr: float, s: float = _BRIGHT_S) -> np.ndarray:
    """调亮图中高于阈值的像素（piecewise power mapping）。"""
    result = img.copy()
    mask = img > thr
    t = (img[mask] - thr) / (1.0 - thr)
    result[mask] = thr + (1.0 - thr) * (t ** s)
    return result


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    预处理流水线（固定校正版）：
      1. 裁剪中间区域 (40:140, 5:55) → 100×50
      2. 黑线检测（r,g,b<130）并填背景色
      3. G/255 → 1 - G/255（黑底白字）
      4. brighten(0.3)：调亮白字
      5. 减去预计算列校正曲线（固定背景渐变补偿）
      6. re-brighten(0.10)：再次调亮白字 + clip(<0.08→0)

    Args:
        image_bgr: OpenCV BGR 图像 (H, W, 3)

    Returns:
        (50, 100) float32 数组，范围 [0, 1]
    """
    # 1. 裁剪中间区域
    crop = image_bgr[_CROP_Y1:_CROP_Y2, _CROP_X1:_CROP_X2]

    # 2. BGR → RGB
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # 3. 计算背景色
    bg_r, bg_g, bg_b = _get_background_color(rgb)

    # 4. 黑线检测
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    dark = (r < _BLACK_THRESH) & (g < _BLACK_THRESH) & (b < _BLACK_THRESH)

    # 5. 黑线区域填背景色
    cleaned = rgb.copy()
    cleaned[dark] = [bg_r, bg_g, bg_b]

    # 6. G 通道反色：1 - G/255
    g_float = cleaned[:, :, 1].astype(np.float32) / 255.0
    inv_g = 1.0 - g_float

    # 7. 调亮白字
    bright = _brighten(inv_g, _BRIGHT_THRESH, _BRIGHT_S)

    # 8. 减去预计算列校正曲线
    flat = np.clip(bright - _CORRECTION, 0.0, 1.0)

    # 9. 再次调亮白字 + 低值剪切
    result = _brighten(flat, _REBRIGHT_THRESH, _BRIGHT_S)
    result[result < 0.08] = 0.0

    return result


# ── 数据集 ──────────────────────────────────────────────────────────────

class ZhjwCaptchaDataset(Dataset):
    """
    教务处验证码数据集。

    目录结构：
        data/
        ├── IMAGES/          # 解压后的图片文件夹
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ...
        └── label.csv        # 标签文件：filename,label
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._load_data()
        self.char_to_idx = {c: i for i, c in enumerate(CHARSET)}

    def _load_data(self):
        images_dir = os.path.join(self.data_dir, "IMAGES")
        label_csv = os.path.join(self.data_dir, "label.csv")

        # 如果 IMAGES 目录不存在，尝试解压 IMAGES.zip
        if not os.path.isdir(images_dir):
            zip_path = os.path.join(self.data_dir, "IMAGES.zip")
            if os.path.isfile(zip_path):
                print(f"[dataset] 解压 {zip_path} → {images_dir}")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.data_dir)

        # 读取 label.csv
        if not os.path.isfile(label_csv):
            raise FileNotFoundError(
                f"未找到 {label_csv}。请从 "
                "https://github.com/SunnyHaze/SCU_OAA-website-Captcha-training-set "
                "下载 IMAGES.zip 和 label.csv 到 data/ 目录"
            )

        self.samples: list[tuple[str, str]] = []
        with open(label_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                fname, label = row[0].strip(), row[1].strip()
                if not fname or not label:
                    continue
                # 验证 label 是否在字符集中
                valid = all(c in CHARSET for c in label)
                if not valid:
                    print(f"  [warn] 跳过 {fname}: label '{label}' 含无效字符")
                    continue
                self.samples.append((fname, label))

        print(f"[dataset] 已加载 {len(self.samples)} 个样本")

    def __len__(self) -> int:
        return len(self.samples)

    def _find_image(self, fname: str) -> str:
        """查找图片文件，自动补全扩展名。"""
        img_dir = os.path.join(self.data_dir, "IMAGES")
        # 直接路径
        for ext in ["", ".jpg", ".jpeg", ".png"]:
            path = os.path.join(img_dir, fname + ext)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"未找到图片: {img_dir}/{fname}")

    def __getitem__(self, idx: int):
        fname, label = self.samples[idx]
        img_path = self._find_image(fname)

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")

        # 预处理
        img = preprocess_image(image_bgr)  # (50, 100)

        # 缩放到模型输入尺寸（保持 2:1 宽高比）
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)  # (32, 64)

        # 标签编码
        label_indices = [self.char_to_idx[c] for c in label]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return torch.from_numpy(img[np.newaxis, :, :].astype(np.float32)), label_tensor


# ── 工具函数 ────────────────────────────────────────────────────────────

def decode_label(indices: torch.Tensor) -> str:
    """将模型输出解码为字符串。"""
    return "".join(CHARSET[i] for i in indices)


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """DataLoader collate 函数。"""
    images, labels = zip(*batch)
    return torch.stack(images, 0), torch.stack(labels, 0)


# ── 测试 ────────────────────────────────────────────────────────────────

def visualize_preprocess(image_path: str):
    """可视化预处理效果：原始 → 去黑线(原始分辨率) → 模型输入。"""
    import matplotlib.pyplot as plt

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取: {image_path}")
        return

    # 原始 BGR → RGB（显示用）
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 预处理后的模型输入 (1, 26, 80)
    chw = preprocess_image(img)

    # 在原始分辨率上的去黑线 G 通道结果
    r_orig = img[:,:,2].astype(np.float32)
    g_orig = img[:,:,1].astype(np.float32)
    b_orig = img[:,:,0].astype(np.float32)
    dark = (r_orig < _BLACK_THRESH) & (g_orig < _BLACK_THRESH) & (b_orig < _BLACK_THRESH)
    bg_r, bg_g, bg_b = _get_background_color(rgb)
    rgb_clean = rgb.copy()
    rgb_clean[dark] = [bg_r, bg_g, bg_b]
    gc = rgb_clean[:,:,1].astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"原始 ({img.shape[1]}×{img.shape[0]})")
    axes[0].axis("off")
    axes[1].imshow(gc, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"去黑线 G 通道(原始分辨率)\n暗区={dark.sum()}/{dark.size}")
    axes[1].axis("off")
    axes[2].imshow(chw[0], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"模型输入 ({INPUT_W}×{INPUT_H})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_preprocess(sys.argv[1])
    else:
        print("用法: python preprocess.py <图片路径>")
        print("例:   python preprocess.py data/IMAGES/000001.jpg")
