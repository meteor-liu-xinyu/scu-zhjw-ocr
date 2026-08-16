"""
PyTorch 权重 → int8 量化（BN 折叠 + per-tensor 对称量化）。

只做量化，不转 .scuocr 格式。输出 .pt 文件，包含：
  - 量化后的 int8 权重（BN 已折叠进 Conv）
  - 每个 tensor 的 scale / zero_point
  - 元数据（来源 checkpoint、fp32 准确率等）

用法：
    python quantize.py checkpoints/latest.pt -o checkpoints/latest.int8.pt
    python quantize.py checkpoints/latest.pt --eval   # 量化后自动验证精度
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Windows 控制台默认 GBK，无法打印 emoji，强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from model import CaptchaCNN, DepthwiseSeparableConv, INPUT_C, INPUT_H, INPUT_W, CAPTCHA_LEN, NUM_CLASSES, NUM_CHARS, SEBlock
from preprocess import ZhjwCaptchaDataset, collate_fn


# ── BN 折叠 ────────────────────────────────────────────────────────────

def fold_bn_into_conv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    将 BatchNorm 折叠进 Conv 权重/偏置（推理等价）。

    公式：folded_w = w * γ / √(var + ε)
          folded_b = (b - mean) * γ / √(var + ε) + β
    折叠后不再需要 BN 参数，推理更简单，量化更稳定。

    支持标准卷积（conv{i}.weight）和深度可分离卷积
    （conv{i}.depthwise.weight + conv{i}.pointwise.weight，BN 折叠进 pointwise）。
    """
    sd = dict(state_dict)
    folded: dict[str, torch.Tensor] = {}
    eps = 1e-5  # BatchNorm2d 默认 eps
    for i in range(1, 5):
        gamma = sd[f"bn{i}.weight"].float()
        beta = sd[f"bn{i}.bias"].float()
        mean = sd[f"bn{i}.running_mean"].float()
        var = sd[f"bn{i}.running_var"].float()
        scale = gamma / torch.sqrt(var + eps)

        if f"conv{i}.depthwise.weight" in sd:
            # 深度可分离卷积：BN 折叠进 pointwise（BN 在 DW→PW 之后）
            dw_w = sd[f"conv{i}.depthwise.weight"].float()
            pw_w = sd[f"conv{i}.pointwise.weight"].float()
            pw_b = sd.get(f"conv{i}.pointwise.bias", torch.zeros(pw_w.shape[0])).float()
            folded_pw_w = pw_w * scale.view(-1, 1, 1, 1)
            folded_pw_b = (pw_b - mean) * scale + beta
            folded[f"conv{i}.depthwise.weight"] = dw_w.contiguous()
            folded[f"conv{i}.pointwise.weight"] = folded_pw_w.contiguous()
            folded[f"conv{i}.pointwise.bias"] = folded_pw_b.contiguous()
        else:
            # 标准卷积
            w = sd[f"conv{i}.weight"].float()
            b = sd.get(f"conv{i}.bias", torch.zeros(w.shape[0])).float()
            folded_w = w * scale.view(-1, 1, 1, 1)
            folded_b = (b - mean) * scale + beta
            folded[f"conv{i}.weight"] = folded_w.contiguous()
            folded[f"conv{i}.bias"] = folded_b.contiguous()

    for k in ("fc1.weight", "fc1.bias", "output_layer.weight", "output_layer.bias",
              "se.fc.0.weight", "se.fc.0.bias", "se.fc.2.weight", "se.fc.2.bias"):
        folded[k] = sd[k].float().contiguous()
    return folded


# ── 对称量化 ───────────────────────────────────────────────────────────

def quantize_tensor_symmetric(t: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """对称量化：q = round(t / scale)，scale = max(|t|) / 127，zero_point 恒为 0。"""
    scale = t.abs().max().item() / 127.0
    if scale == 0:
        scale = 1.0
    q = torch.round(t / scale).clamp(-128, 127).to(torch.int8)
    return q, scale, 0


def quantize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict:
    """量化整个 state_dict，返回 {name: (int8_tensor, scale, zero_point)}。"""
    folded = fold_bn_into_conv(state_dict)
    quantized = {}
    for name, t in folded.items():
        q, scale, zp = quantize_tensor_symmetric(t)
        quantized[name] = {"q": q, "scale": scale, "zero_point": zp}
    return quantized


# ── 反量化 + 推理（验证用）────────────────────────────────────────────

class FoldedCaptchaCNN(nn.Module):
    """BN 折叠后的纯推理模型（无 BN 层，与量化格式对应）。"""

    def __init__(self, input_c: int = INPUT_C, num_chars: int = NUM_CHARS,
                 use_depthwise: bool = False):
        super().__init__()
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(input_c, 24, bias=True)
            self.conv2 = DepthwiseSeparableConv(24, 40, bias=True)
            self.conv3 = DepthwiseSeparableConv(40, 64, bias=True)
            self.conv4 = DepthwiseSeparableConv(64, 64, bias=True)
        else:
            self.conv1 = nn.Conv2d(input_c, 24, 3, padding=1, bias=True)
            self.conv2 = nn.Conv2d(24, 40, 3, padding=1, bias=True)
            self.conv3 = nn.Conv2d(40, 64, 3, padding=1, bias=True)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.se = SEBlock(64, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d((1, 4))
        self.fc1 = nn.Linear(64 * 1 * 4, 120, bias=True)
        self.output_layer = nn.Linear(120, num_chars, bias=True)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = self.se(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.output_layer(x)
        return x


def dequantize(quantized: dict) -> dict[str, torch.Tensor]:
    """反量化回 fp32 state_dict。"""
    sd = {}
    for name, item in quantized.items():
        q = item["q"].float()
        sd[name] = (q - item["zero_point"]) * item["scale"]
    return sd


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    cs = cc = ts = tc = 0
    with torch.no_grad():
        for images, labels in loader:
            B = images.size(0)
            logits = model(images).reshape(B, CAPTCHA_LEN, NUM_CLASSES)
            pred = logits.argmax(dim=-1)
            cc += (pred == labels).sum().item()
            tc += B * CAPTCHA_LEN
            cs += (pred == labels).all(dim=1).sum().item()
            ts += B
    return cc / tc * 100, cs / ts * 100


def size_bytes(sd: dict) -> int:
    """按 state_dict 统计字节数（int8=1B/元素，fp32=4B/元素）。"""
    total = 0
    for v in sd.values():
        if torch.is_tensor(v):
            total += v.numel() * (1 if v.dtype == torch.int8 else v.element_size())
    return total


# ── 主流程 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="int8 量化（BN 折叠 + 对称量化）")
    parser.add_argument("checkpoint", type=str, help="PyTorch checkpoint 路径 (*.pt)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出 .pt 文件路径")
    parser.add_argument("--eval", action="store_true",
                        help="量化后自动在测试集上验证精度")
    parser.add_argument("--data-dir", type=str, default="data", help="数据集目录")
    parser.add_argument("--seed", type=int, default=42, help="划分测试集的随机种子")
    parser.add_argument("--use-depthwise", action="store_true",
                        help="深度可分离卷积模型（conv{i}.depthwise/pointwise）")
    args = parser.parse_args()

    # 加载 checkpoint
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6 不支持 weights_only
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
        epoch = ckpt.get("epoch", "?")
        acc = ckpt.get("best_acc", "?")
        print(f"加载 checkpoint: epoch={epoch}, best_acc={acc}%")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # fp32 基线（BN 折叠后）
    folded = fold_bn_into_conv(state_dict)
    fp32_bytes = size_bytes(folded)

    # 量化
    quantized = quantize_state_dict(state_dict)
    int8_bytes = size_bytes({k: v["q"] for k, v in quantized.items()})
    print(f"\n量化完成: {len(quantized)} 个 tensor")
    print(f"  fp32 权重: {fp32_bytes/1024:.1f} KB")
    print(f"  int8 权重: {int8_bytes/1024:.1f} KB  ({int8_bytes/fp32_bytes*100:.0f}%)")

    # 保存
    output = args.output or os.path.splitext(args.checkpoint)[0] + ".int8.pt"
    torch.save({
        "format": "scuocr-int8-pt",
        "version": 2,
        "source": os.path.basename(args.checkpoint),
        "source_epoch": ckpt.get("epoch", None),
        "source_best_acc": ckpt.get("best_acc", None),
        "quantized": quantized,
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_bytes,
    }, output)
    print(f"✅ 已保存: {output} ({os.path.getsize(output)/1024:.1f} KB)")

    # 验证精度
    if args.eval:
        print("\n验证精度...")
        full = ZhjwCaptchaDataset(data_dir=args.data_dir, augment=False)
        test_size = int(len(full) * 0.1)
        val_size = int(len(full) * 0.1)
        train_size = len(full) - val_size - test_size
        _, _, test_idx = random_split(
            range(len(full)), [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed))
        test_loader = DataLoader(torch.utils.data.Subset(full, test_idx),
                                 batch_size=128, shuffle=False, num_workers=0,
                                 collate_fn=collate_fn)

        # fp32 折叠基线
        m_fp32 = FoldedCaptchaCNN(input_c=INPUT_C, use_depthwise=args.use_depthwise)
        m_fp32.load_state_dict(folded)
        ca, sa = evaluate(m_fp32, test_loader)
        print(f"  [fp32 折叠] CharAcc={ca:.2f}%  SampleAcc={sa:.2f}%")

        # int8 反量化
        m_int8 = FoldedCaptchaCNN(input_c=INPUT_C, use_depthwise=args.use_depthwise)
        m_int8.load_state_dict(dequantize(quantized))
        ca, sa = evaluate(m_int8, test_loader)
        print(f"  [int8 反量化] CharAcc={ca:.2f}%  SampleAcc={sa:.2f}%")


if __name__ == "__main__":
    main()