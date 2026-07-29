"""
验证码预处理：裁剪 → 去黑线 → 灰度反色 → 缩放。

流水线：
  1. 裁剪中间区域 (40:140, 5:55) → 100×50
  2. 黑线检测（RGB 三通道均 < 130）→ 填充固定背景色 RGB(225,222,222)
  3. 灰度化 + 反色: gray = cvtColor(RGB→Gray)/255, result = 1 - gray
  4. 缩放到 64×32 (INTER_AREA)

输出形状 (32, 64)，float32 范围 [0, 1]。
"""

import os
import zipfile
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import INPUT_H, INPUT_W, INPUT_C, CHARSET

# ── 固定参数 ────────────────────────────────────────────────────────────
_CROP_X1 = 40
_CROP_X2 = 140
_CROP_Y1 = 5
_CROP_Y2 = 55

_BLACK_THRESH = 130               # 黑线检测阈值（RGB 三通道均低于此值视为黑线）
_BG_COLOR = np.array([225, 222, 222], dtype=np.uint8)  # 固定背景填充色 RGB


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    预处理流水线：
      1. 裁剪中间区域 (40:140, 5:55) → 100×50
      2. 黑线检测（RGB < 130）→ 填充固定背景色 RGB(225,222,222)
      3. 灰度化 + 反色: gray = cvtColor(RGB→Gray)/255, result = 1 - gray
      4. 缩放到 64×32 (INTER_AREA)

    Args:
        image_bgr: OpenCV BGR 图像 (H, W, 3)

    Returns:
        (32, 64) float32 数组，范围 [0, 1]
    """
    # 1. 裁剪中间区域
    crop = image_bgr[_CROP_Y1:_CROP_Y2, _CROP_X1:_CROP_X2]

    # 2. BGR → RGB
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # 3. 黑线检测
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    dark = (r < _BLACK_THRESH) & (g < _BLACK_THRESH) & (b < _BLACK_THRESH)

    # 4. 黑线区域填固定背景色
    cleaned = rgb.copy()
    cleaned[dark] = _BG_COLOR

    # 5. 灰度化 + 反色
    gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    inverted = 1.0 - gray

    # 6. 缩放到模型输入尺寸
    result = cv2.resize(inverted, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)

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

    def __init__(self, data_dir: str, augment: bool = False):
        self.data_dir = data_dir
        self.augment = augment
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

        # 预处理（已在 preprocess_image 中缩放到 64×32）
        img = preprocess_image(image_bgr)  # (32, 64)

        # 训练时数据增强
        if self.augment:
            img = _augment(img)

        # 标签编码
        label_indices = [self.char_to_idx[c] for c in label]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return torch.from_numpy(img[np.newaxis, :, :].astype(np.float32)), label_tensor


# ── 数据增强 ────────────────────────────────────────────────────────────

def _augment(img: np.ndarray) -> np.ndarray:
    """
    训练时数据增强：随机微小平移、缩放、旋转。
    输入/输出都是 (32, 64) float32 [0, 1]。
    """
    h, w = img.shape
    angle = np.random.uniform(-2.0, 2.0)
    scale = np.random.uniform(0.95, 1.05)
    dx = np.random.uniform(-1.0, 1.0)
    dy = np.random.uniform(-1.0, 1.0)

    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    mat[0, 2] += dx
    mat[1, 2] += dy

    aug = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return aug


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

    # 裁剪区域
    crop = img[_CROP_Y1:_CROP_Y2, _CROP_X1:_CROP_X2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # 去黑线中间结果
    dark = (crop_rgb[:,:,0] < _BLACK_THRESH) & (crop_rgb[:,:,1] < _BLACK_THRESH) & (crop_rgb[:,:,2] < _BLACK_THRESH)
    cleaned = crop_rgb.copy()
    cleaned[dark] = _BG_COLOR

    # 预处理后的模型输入
    processed = preprocess_image(img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"原始 ({img.shape[1]}×{img.shape[0]})")
    axes[0].axis("off")
    axes[1].imshow(cleaned)
    axes[1].set_title(f"去黑线后 (裁剪区)\n黑线={dark.sum()}px")
    axes[1].axis("off")
    axes[2].imshow(processed, cmap="gray", vmin=0, vmax=1)
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
