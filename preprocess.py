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
# 裁剪窗预设。实测墨迹包络 x∈[32,150]、y∈[10,57]（全量 10000 张，redness>40）：
#   "current" x40..139 y5..54 —— 历史默认，横向切到 1.67% 的图、纵向切到 15.37%
#   "wide"    x32..149 y3..57 —— 横向只剩 1 张、纵向 1.20%，基本不切
# ⚠ 换窗会改变输入分布，**已有权重全部作废**，必须重训。
CROP_PRESETS = {
    "current": (40, 140, 5, 55),
    "wide": (32, 150, 3, 58),
}
_CROP_PRESET = "current"
_CROP_X1, _CROP_X2, _CROP_Y1, _CROP_Y2 = CROP_PRESETS[_CROP_PRESET]


def set_crop(preset: str) -> None:
    """切换裁剪窗预设（全局）。"""
    global _CROP_PRESET, _CROP_X1, _CROP_X2, _CROP_Y1, _CROP_Y2
    if preset not in CROP_PRESETS:
        raise ValueError(f"未知裁剪预设 {preset!r}，可选 {list(CROP_PRESETS)}")
    _CROP_PRESET = preset
    _CROP_X1, _CROP_X2, _CROP_Y1, _CROP_Y2 = CROP_PRESETS[preset]


def get_crop() -> tuple[int, int, int, int]:
    return _CROP_X1, _CROP_X2, _CROP_Y1, _CROP_Y2


_BLACK_THRESH = 130               # 黑线检测阈值（RGB 三通道均低于此值视为黑线）
_BG_COLOR = np.array([225, 222, 222], dtype=np.uint8)  # 固定背景填充色 RGB


def preprocess_image(image_bgr: np.ndarray, crop: str | None = None) -> np.ndarray:
    """
    预处理流水线：
      1. 裁剪中间区域（默认 x40:140, y5:55 → 100×50；可用 crop="wide" 切到 x32:150 y3:57）
      2. 黑线检测（RGB < 130）→ 填充固定背景色 RGB(225,222,222)
      3. 灰度化 + 反色: gray = cvtColor(RGB→Gray)/255, result = 1 - gray
      4. 缩放到 64×32 (INTER_AREA)

    Args:
        image_bgr: OpenCV BGR 图像 (H, W, 3)
        crop: 裁剪预设名（None = 用全局当前预设）

    Returns:
        (32, 64) float32 数组，范围 [0, 1]
    """
    # 1. 裁剪中间区域
    if crop is None:
        x1, x2, y1, y2 = get_crop()
    else:
        x1, x2, y1, y2 = CROP_PRESETS[crop]
    crop_img = image_bgr[y1:y2, x1:x2]

    # 2. BGR → RGB
    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

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

    def __init__(self, data_dir: str, augment: bool = False,
                 aug_profile: str = "default", crop: str | None = None,
                 crop_jitter: int = 0):
        """
        Args:
            data_dir: 数据目录（含 IMAGES/ 与 label.csv）
            augment: 是否施加 _augment
            aug_profile: 增强档位（default/strong/shift/shift_strong）
            crop: 裁剪预设（None = 用全局当前预设）
            crop_jitter: 裁剪窗随机偏移上限（原图像素）。模拟「版式整体平移」，
                **只应在训练集上开启**；验证/测试集必须保持 0，否则评估被污染。
        """
        self.data_dir = data_dir
        self.augment = augment
        self.aug_profile = aug_profile
        self.crop = crop
        self.crop_jitter = crop_jitter
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

    def _jittered_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        """把裁剪窗整体随机平移，模拟版式偏移（新露出区域用边缘像素补齐）。

        为什么必须在**原图**上平移而不是平移 64×32 输入：
        平移模型输入要填腾出的区域，任何填充值都会造出人为强度台阶；
        在原图上平移再裁剪，新进窗口的是真实像素，背景渐变连续。
        """
        if self.crop_jitter <= 0:
            return image_bgr
        H, W = image_bgr.shape[:2]
        dx = np.random.randint(-self.crop_jitter, self.crop_jitter + 1)
        dy = np.random.randint(-self.crop_jitter, self.crop_jitter + 1)
        if dx == 0 and dy == 0:
            return image_bgr
        pad_x = (abs(dx), 0) if dx > 0 else (0, abs(dx))
        pad_y = (abs(dy), 0) if dy > 0 else (0, abs(dy))
        p = np.pad(image_bgr, (pad_y, pad_x, (0, 0)), mode="edge")
        return p[:H, :W] if (dx > 0 or dy > 0) else p[-H:, -W:]

    def __getitem__(self, idx: int):
        fname, label = self.samples[idx]
        img_path = self._find_image(fname)

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")

        # 训练时：先做裁剪窗抖动（原图域），再走标准预处理
        if self.augment and self.crop_jitter > 0:
            image_bgr = self._jittered_crop(image_bgr)

        # 预处理（已在 preprocess_image 中缩放到 64×32）
        img = preprocess_image(image_bgr, crop=self.crop)  # (32, 64)

        # 训练时数据增强（注意必须把 profile 传下去，
        # 否则 aug_profile 只是个装饰品——本项目踩过这个坑）
        if self.augment:
            img = _augment(img, profile=self.aug_profile)

        # 标签编码
        label_indices = [self.char_to_idx[c] for c in label]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return torch.from_numpy(img[np.newaxis, :, :].astype(np.float32)), label_tensor


# ── 数据增强 ────────────────────────────────────────────────────────────

def _elastic_distort(img: np.ndarray, alpha: float = 0.8, sigma: float = 0.5) -> np.ndarray:
    """弹性形变：模拟验证码字符扭曲。alpha 控制强度，sigma 控制平滑度。"""
    h, w = img.shape
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def _random_occlusion(img: np.ndarray, max_h: int = 3, max_w: int = 6) -> np.ndarray:
    """随机遮挡小块（模拟黑线残留）。反色后背景为 0，故用 0 填充。"""
    h, w = img.shape
    out = img.copy()
    for _ in range(np.random.randint(1, 3)):
        bw = np.random.randint(1, max_w)
        bh = np.random.randint(1, max_h)
        x = np.random.randint(0, w - bw)
        y = np.random.randint(0, h - bh)
        out[y:y + bh, x:x + bw] = 0.0
    return out


# 增强档位参数表。
# ⚠ 关于 shift / shift_strong：本模型的平移容忍度极小（实测原图平移 4px 掉 1.9 点、
#   6px 掉 14 点、8px 掉 45 点），根因是训练时平移只有 ±1px，而旋转/缩放绕图心做、
#   几乎不改变字符的绝对位置 → 训练集里字符永远出现在同一个绝对位置，
#   模型于是把位置背了下来。shift 档就是为补这个洞。
#   单位是**模型输入像素（64×32）**：1 模型px ≈ 1.5625 原图px。
#   - "default"/"strong" 保持原行为（borderValue=0），不做任何改变，向后兼容；
#   - "shift"/"shift_strong" 用 BORDER_REPLICATE 铺背景：反色图里背景≈0.12，
#     用 0 填充会在边缘造出一条黑带，形似粗笔画，是错的填充方式。
_AUG_PROFILES = {
    "default": dict(angle_r=2.0, scale_r=(0.95, 1.05), shift_r=1.0,
                    p_elastic=0.3, p_thick=0.2, p_noise=0.2, p_occ=0.2,
                    alpha_r=(0.92, 1.08), beta_r=(-0.04, 0.04), noise_std=0.015,
                    replicate=False),
    "strong": dict(angle_r=5.0, scale_r=(0.90, 1.10), shift_r=1.0,
                   p_elastic=0.5, p_thick=0.4, p_noise=0.4, p_occ=0.4,
                   alpha_r=(0.85, 1.15), beta_r=(-0.06, 0.06), noise_std=0.03,
                   replicate=False),
    # ±4 模型px ≈ ±6.25 原图px，覆盖实测 4~6px 的失效边界
    "shift": dict(angle_r=2.0, scale_r=(0.95, 1.05), shift_r=4.0,
                  p_elastic=0.3, p_thick=0.2, p_noise=0.2, p_occ=0.2,
                  alpha_r=(0.92, 1.08), beta_r=(-0.04, 0.04), noise_std=0.015,
                  replicate=True),
    # ±6 模型px ≈ ±9.4 原图px，更激进的容差（可能牺牲同分布精度）
    "shift_strong": dict(angle_r=5.0, scale_r=(0.90, 1.10), shift_r=6.0,
                         p_elastic=0.5, p_thick=0.4, p_noise=0.4, p_occ=0.4,
                         alpha_r=(0.85, 1.15), beta_r=(-0.06, 0.06), noise_std=0.03,
                         replicate=True),
}


def _augment(img: np.ndarray, profile: str = "default") -> np.ndarray:
    """
    训练时数据增强：几何变换 + 弹性形变 + 亮度对比度 + 笔画粗细 + 噪声 + 局部遮挡。
    输入/输出都是 (32, 64) float32 [0, 1]。

    Args:
        profile: 增强档位（见 _AUG_PROFILES）
            - "default": 原强度（旋转±2°/缩放±5%/平移±1px/各概率0.2~0.3）
            - "strong":  更强（旋转±5°/缩放±10%/各概率0.4）
            - "shift":   原强度 + **平移±4 模型px**，修平移不变性缺失
            - "shift_strong": strong + 平移±6 模型px
    """
    if profile not in _AUG_PROFILES:
        raise ValueError(f"未知增强档位 {profile!r}，可选 {list(_AUG_PROFILES)}")
    P = _AUG_PROFILES[profile]
    h, w = img.shape

    angle_r = P["angle_r"]; scale_r = P["scale_r"]; shift_r = P["shift_r"]
    p_elastic = P["p_elastic"]; p_thick = P["p_thick"]
    p_noise = P["p_noise"]; p_occ = P["p_occ"]
    alpha_r = P["alpha_r"]; beta_r = P["beta_r"]; noise_std = P["noise_std"]

    # 1. 几何变换：旋转 + 缩放 + 平移
    angle = np.random.uniform(-angle_r, angle_r)
    scale = np.random.uniform(scale_r[0], scale_r[1])
    dx = np.random.uniform(-1.0, 1.0)
    dy = np.random.uniform(-1.0, 1.0)
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    mat[0, 2] += dx
    mat[1, 2] += dy
    aug = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # 2. 弹性形变
    if np.random.rand() < p_elastic:
        aug = _elastic_distort(aug)

    # 3. 亮度/对比度扰动
    alpha = np.random.uniform(alpha_r[0], alpha_r[1])   # 对比度
    beta = np.random.uniform(beta_r[0], beta_r[1])      # 亮度偏移
    aug = np.clip(aug * alpha + beta, 0.0, 1.0)

    # 4. 笔画粗细扰动（膨胀/腐蚀，概率 0.2）
    if np.random.rand() < p_thick:
        kernel = np.ones((2, 2), np.uint8)
        if np.random.rand() < 0.5:
            aug = cv2.dilate(aug, kernel, iterations=1)
        else:
            aug = cv2.erode(aug, kernel, iterations=1)

    # 5. 高斯噪声
    if np.random.rand() < p_noise:
        noise = np.random.normal(0.0, noise_std, aug.shape).astype(np.float32)
        aug = np.clip(aug + noise, 0.0, 1.0)

    # 6. 随机局部遮挡
    if np.random.rand() < p_occ:
        aug = _random_occlusion(aug)

    return aug.astype(np.float32)


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
