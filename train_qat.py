"""
QAT 训练脚本：量化感知训练（per-channel int4/int8）。

核心思路：
  - 训练时对权重做 fake-quant（STE 直通估计器），前向用量化后的权重，
    反向梯度直通回原始 fp32 权重 —— 权重学会适应量化网格。
  - per-channel 对称量化（每输出通道一个 scale）：conv 权重 per-channel，
    fc/se/output 权重 per-tensor（实验证明敏感度低）。
  - 训练完成后可导出 per-channel int4 .pt（供 export.py --mixed 转 .scuocr v3）。

用法（GPU T4 或 CPU 均可）：
    # 从现有 fp32 最佳模型微调（推荐：模型已收敛，只需适应量化网格）
    python train_qat.py --resume checkpoints/best.pt --epochs 40 --batch-size 128 --num-workers 4

    # 窄架构微调（如 62KB/42KB 先用原版配方训出 best.pt，再 QAT 微调）
    python train_qat.py --resume checkpoints/best_1.pt --widths 20,32,48,48 --fc-width 80 \
      --epochs 40 --batch-size 128 --num-workers 4 --eval-quant --export

    # 从零 QAT 训练（同时保底正常训练能力）
    python train_qat.py --epochs 120 --batch-size 128 --num-workers 4

    # 纯 CPU 环境
    python train_qat.py --resume checkpoints/best.pt --epochs 30 --batch-size 64 --num-workers 2 --device cpu
"""
import os
import sys
import argparse
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from model import (
    CaptchaCNN, INPUT_C, INPUT_H, INPUT_W, CAPTCHA_LEN, NUM_CLASSES, NUM_CHARS,
    CHARSET, SEBlock, DepthwiseSeparableConv,
)
from preprocess import ZhjwCaptchaDataset, collate_fn, _augment
from quantize import fold_bn_into_conv, FoldedCaptchaCNN, quantize_tensor_symmetric, evaluate


# ══════════════════════════════════════════════════════════════════════
#  量化工具
# ══════════════════════════════════════════════════════════════════════

class _FakeQuantSTE(torch.autograd.Function):
    """对称 fake-quant，STE 反向。

    forward: 量化 → 反量化（带量化误差的权重）
    backward: 直通（梯度原样传回）
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int, per_channel: bool):
        qmax = 127 if bits == 8 else 7
        ctx.save_for_backward(x)

        if per_channel and x.dim() >= 2:
            # 每输出通道一个 scale
            flat = x.reshape(x.shape[0], -1)
            scales = flat.abs().max(dim=1).values.clamp(min=1e-12) / qmax
            q = torch.round(flat / scales[:, None]).clamp(-qmax - 1, qmax)
            out = (q * scales[:, None]).reshape(x.shape)
        else:
            scale = x.abs().max().item() / qmax
            if scale == 0:
                scale = 1.0
            q = torch.round(x / scale).clamp(-qmax - 1, qmax)
            out = q * scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  # STE: 直通


def fake_quant(x: torch.Tensor, bits: int, per_channel: bool = True):
    return _FakeQuantSTE.apply(x, bits, per_channel)


# ══════════════════════════════════════════════════════════════════════
#  QAT 模型
# ══════════════════════════════════════════════════════════════════════

class QatCaptchaCNN(nn.Module):
    """量化感知训练版 CaptchaCNN。

    结构完全兼容 CaptchaCNN / FoldedCaptchaCNN：
      Conv3×3 ×4 + BN + ReLU + MaxPool ×4
      SE → AdaptiveAvgPool(1,4) → FC1 → Output(80)
    仅在前向时把权重 fake-quant。
    conv 权重 per-channel；fc/output/se 权重 per-tensor（实验证明敏感度低）。
    """

    def __init__(
        self,
        input_c: int = INPUT_C,
        num_chars: int = NUM_CHARS,
        dropout: float = 0.3,
        widths: tuple[int, ...] = (24, 40, 64, 64),
        fc_width: int = 120,
        qat_bits: int = 4,
        use_depthwise: bool = False,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        self.widths = widths
        self.fc_width = fc_width
        self.qat_bits = qat_bits
        self.use_depthwise = use_depthwise

        conv_fn = DepthwiseSeparableConv if use_depthwise else nn.Conv2d
        w1, w2, w3, w4 = widths

        self.conv1 = conv_fn(input_c, w1, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w1, momentum=bn_momentum)
        self.conv2 = conv_fn(w1, w2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(w2, momentum=bn_momentum)
        self.conv3 = conv_fn(w2, w3, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w3, momentum=bn_momentum)
        self.conv4 = conv_fn(w3, w4, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(w4, momentum=bn_momentum)

        self.se = SEBlock(w4, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d((1, 4))
        self.fc1 = nn.Linear(w4 * 4, fc_width, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(fc_width, num_chars, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def _conv(self, module, x):
        """对 conv 权重做 per-channel fake-quant 后卷积。"""
        w = fake_quant(module.weight, self.qat_bits, per_channel=True)
        return F.conv2d(x, w, module.bias, module.stride, module.padding,
                        module.dilation, module.groups)

    def _dw_pw(self, module, x):
        """深度可分离卷积的量化版：DW per-channel + PW per-channel。"""
        dw_w = fake_quant(module.depthwise.weight, self.qat_bits, per_channel=True)
        x = F.conv2d(x, dw_w, None, module.depthwise.stride, module.depthwise.padding,
                     module.depthwise.dilation, module.depthwise.groups)
        pw_w = fake_quant(module.pointwise.weight, self.qat_bits, per_channel=True)
        x = F.conv2d(x, pw_w, module.pointwise.bias)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv + BN + ReLU + MaxPool ×4
        for i in range(1, 5):
            conv = getattr(self, f"conv{i}")
            bn = getattr(self, f"bn{i}")
            x = self._dw_pw(conv, x) if self.use_depthwise else self._conv(conv, x)
            x = bn(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)

        x = self.se(x)          # SE 不量化（参数极少，敏感度实验为 0 损失）
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)

        # FC 层 per-tensor fake-quant
        w1 = fake_quant(self.fc1.weight, self.qat_bits, per_channel=False)
        x = torch.relu(F.linear(x, w1, self.fc1.bias))
        x = self.dropout(x)
        wo = fake_quant(self.output_layer.weight, self.qat_bits, per_channel=False)
        x = F.linear(x, wo, self.output_layer.bias)
        return x

    def state_dict_as_captchain(self) -> dict:
        """返回与 CaptchaCNN 相同的 state_dict（无量化痕迹，纯 fp32 权重）。"""
        return {k: v.detach().clone() for k, v in self.state_dict().items()}


# ══════════════════════════════════════════════════════════════════════
#  训练循环
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="QAT 量化感知训练（per-channel int4/int8）")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=60, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=128,
                   help="批次大小（GPU 上可 128~256，CPU 建议 64）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader 进程数（GPU: 4~8，CPU 建议 2）")
    p.add_argument("--device", type=str, default=None, help="cuda/cpu，默认自动")
    p.add_argument("--resume", type=str, default=None,
                   help="从已有 fp32 checkpoint 微调（推荐）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=str, default="runs/qat")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/qat")
    p.add_argument("--qat-bits", type=int, default=4, choices=[4, 8],
                   help="训练时 fake-quant 位宽（8 = 普通训练，int8 微调）")
    p.add_argument("--widths", type=str, default="24,40,64,64",
                   help="卷积通道宽（更小模型：20,32,48,48 / 16,24,40,40）")
    p.add_argument("--fc-width", type=int, default=120)
    p.add_argument("--use-depthwise", action="store_true",
                   help="深度可分离卷积（参数 -55%%）")
    p.add_argument("--aug-profile", type=str, default="default",
                   choices=["default", "strong"],
                   help="数据增强强度（default=原版弱增强，strong=更强）")
    p.add_argument("--bn-momentum", type=float, default=0.1,
                   help="BatchNorm momentum")
    p.add_argument("--early-stop", type=int, default=0,
                   help="早停耐心（连续 N 轮验证集无提升则停止，0=关闭）")
    p.add_argument("--eval-quant", action="store_true",
                   help="训练结束后用 per-channel int4 PTQ 评估（模拟部署精度）")
    p.add_argument("--export", action="store_true",
                   help="导出 per-channel int4 quantized .pt 供 export.py 转换")
    p.add_argument("--amp", action="store_true",
                   help="启用自动混合精度（GPU 训练提速 2-3 倍）")
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer,
                grad_clip=0.0, scaler=None, use_amp=False):
    model.train()
    total_loss = 0
    cs = cc = ts = tc = 0
    start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images).view(B, CAPTCHA_LEN, NUM_CLASSES)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if grad_clip > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        cc += (pred == labels).sum().item()
        tc += B * CAPTCHA_LEN
        cs += (pred == labels).all(dim=1).sum().item()
        ts += B

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader):4d} "
                  f"| Loss {loss.item():.4f} | CharAcc {cc/tc*100:.2f}%")

    avg = total_loss / len(loader)
    char_acc = cc / tc * 100
    sample_acc = cs / ts * 100
    print(f"  ── Epoch {epoch:3d} done | Loss {avg:.4f} | CharAcc {char_acc:.2f}% "
          f"| SampleAcc {sample_acc:.2f}% | {time.time()-start:.1f}s")
    writer.add_scalar("Loss/train", avg, epoch)
    writer.add_scalar("Acc/sample_train", sample_acc, epoch)
    return avg


@torch.no_grad()
def evaluate_model(model, loader, device):
    """返回 (char_acc, sample_acc)。"""
    model.eval()
    cc = tc = cs = ts = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)
        logits = model(images).view(B, CAPTCHA_LEN, NUM_CLASSES)
        pred = logits.argmax(dim=-1)
        cc += (pred == labels).sum().item()
        tc += B * CAPTCHA_LEN
        cs += (pred == labels).all(dim=1).sum().item()
        ts += B
    return cc / tc * 100, cs / ts * 100


# ── per-channel int4 量化工具（用于 --eval-quant 的部署模拟） ─────────

def quantize_per_channel_int4(folded: dict) -> dict:
    """per-channel int4 量化（每输出通道独立 scale）。

    返回 {name: {q(packed uint8), scale, zero_point, bits, shape, per_channel}}
    """
    quantized = {}
    for name, t in folded.items():
        bits = 4
        qmax = 7
        if t.dim() >= 2:
            # per-channel：沿输出通道
            flat = t.reshape(t.shape[0], -1)
            scales = flat.abs().max(dim=1).values.clamp(min=1e-12) / qmax
            q = torch.round(flat / scales[:, None]).clamp(-qmax - 1, qmax).to(torch.int8)
            # 打包 int4
            q_flat = q.reshape(-1)
            n = q_flat.numel()
            if n % 2 == 1:
                q_flat = torch.cat([q_flat, torch.zeros(1, dtype=q_flat.dtype)])
            q_u = ((q_flat + 8) & 0x0F).to(torch.uint8)
            packed = (q_u[0::2] | (q_u[1::2] << 4)).to(torch.uint8)
            quantized[name] = {
                "q": packed, "scale": scales, "zero_point": 0,
                "bits": bits, "shape": list(t.shape), "per_channel": True,
            }
        else:
            # bias: per-tensor
            scale = t.abs().max().item() / qmax
            if scale == 0:
                scale = 1.0
            q = torch.round(t / scale).clamp(-qmax - 1, qmax).to(torch.int8)
            # 打包 int4
            n = q.numel()
            if n % 2 == 1:
                q = torch.cat([q, torch.zeros(1, dtype=q.dtype)])
            q_u = ((q + 8) & 0x0F).to(torch.uint8)
            packed = (q_u[0::2] | (q_u[1::2] << 4)).to(torch.uint8)
            quantized[name] = {
                "q": packed, "scale": scale, "zero_point": 0,
                "bits": bits, "shape": list(t.shape), "per_channel": False,
            }
    return quantized


def dequantize_per_channel(quantized: dict) -> dict[str, torch.Tensor]:
    """反量化 per-channel int4 回到 fp32 state_dict。"""
    sd = {}
    for name, item in quantized.items():
        # 解包 int4
        packed = item["q"]
        lo = ((packed & 0x0F).to(torch.int8)) - 8
        hi = (((packed >> 4) & 0x0F).to(torch.int8)) - 8
        q = torch.stack([lo, hi], dim=-1).reshape(-1)
        n = torch.Size(item["shape"]).numel()
        q = q[:n].reshape(item["shape"]).float()

        if item.get("per_channel"):
            s = item["scale"]
            broadcast = s.reshape(-1, *([1] * (q.dim() - 1)))
            sd[name] = (q * broadcast).reshape(item["shape"])
        else:
            sd[name] = q * item["scale"]
    return sd


# ══════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备: {device} | QAT bits: {args.qat_bits} | 增强: {args.aug_profile}")

    use_amp = (device.type == "cuda")
    if use_amp:
        torch.backends.cudnn.benchmark = True
        print("已启用 AMP + cudnn.benchmark")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    widths = tuple(int(w) for w in args.widths.split(","))
    assert len(widths) == 4, f"需要 4 个宽度值，收到 {len(widths)}"

    # ── 数据 ──
    full = ZhjwCaptchaDataset(data_dir=args.data_dir, augment=False)
    test_size = int(len(full) * args.test_split)
    val_size = int(len(full) * args.val_split)
    train_size = len(full) - val_size - test_size
    train_idx, val_idx, test_idx = random_split(
        range(len(full)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed))

    # 训练集：数据增强（与 train.py 一致）
    class _AugDataset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base, self.indices = base, indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.base[self.indices[idx]]
            img_np = _augment(img.squeeze(0).numpy())
            return torch.from_numpy(img_np[np.newaxis].astype(np.float32)), label

    train_ds = _AugDataset(full, train_idx)
    val_ds = torch.utils.data.Subset(full, val_idx)
    test_ds = torch.utils.data.Subset(full, test_idx)
    print(f"训练: {len(train_ds)} | 验证: {len(val_ds)} | 测试: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn,
                             pin_memory=(device.type == "cuda"))

    # ── 模型 ──
    model = QatCaptchaCNN(input_c=INPUT_C, widths=widths, fc_width=args.fc_width,
                          qat_bits=args.qat_bits, use_depthwise=args.use_depthwise,
                          bn_momentum=args.bn_momentum).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,} | int8≈{n_params/1024:.0f}KB | int4≈{n_params/2/1024:.0f}KB")

    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        miss, unexp = model.load_state_dict(sd, strict=False)
        missing = [k for k in miss if k not in ("num_batches_tracked",)]
        if missing:
            print(f"警告: 未加载的键: {missing}")
        start_epoch = 0  # QAT 微调是全新训练阶段，重置 epoch 计数
        best_acc = 0.0  # QAT 微调重新跟踪 best
        print(f"已加载 {args.resume} (epoch={ckpt.get('epoch','?')}, "
              f"best_acc={ckpt.get('best_acc','?')}%)，从 QAT 微调")
        # QAT 微调用更小学习率，避免破坏已学特征
        args.lr = min(args.lr, 2e-4)
        print(f"微调模式 lr={args.lr}")

    # ── 优化器 & 调度 ──
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup)
    cos = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup), eta_min=args.lr_min)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cos], milestones=[args.warmup])

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ── 训练 ──
    print(f"开始 QAT 训练（{args.epochs} 轮）...")
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n┌─ Epoch {epoch:3d}/{args.epochs} | lr={lr:.2e}")

        train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer,
                    grad_clip=args.grad_clip, scaler=scaler, use_amp=use_amp)
        ca, sa = evaluate_model(model, val_loader, device)
        print(f"  [Val] CharAcc={ca:.2f}% SampleAcc={sa:.2f}%")
        writer.add_scalar("Acc/sample_val", sa, epoch)
        scheduler.step()

        ckpt = {"epoch": epoch, "model": model.state_dict_as_captchain(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_acc": max(best_acc, sa), "args": vars(args),
                "qat_bits": args.qat_bits}
        torch.save(ckpt, os.path.join(args.ckpt_dir, "latest.pt"))
        if sa >= best_acc:
            best_acc = sa
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"  ⭐ 新最佳验证 SampleAcc: {best_acc:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"\n⏹ 早停：连续 {args.early_stop} 轮验证集无提升，停止训练。")
            break

    # ── 最终评估 ──
    print(f"\n加载 best.pt 在测试集评估...")
    best_ckpt = torch.load(os.path.join(args.ckpt_dir, "best.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"], strict=False)
    ca, sa = evaluate_model(model, test_loader, device)
    print(f"  [Test fp32权重] CharAcc={ca:.2f}% SampleAcc={sa:.2f}%")

    # ── 可选：per-channel int4 PTQ 评估（模拟部署精度） ──
    if args.eval_quant:
        print("\n[quant] per-channel int4 PTQ 评估（模拟部署精度）...")
        sd = {k: v.cpu() for k, v in model.state_dict_as_captchain().items()}
        folded = fold_bn_into_conv(sd)
        quantized = quantize_per_channel_int4(folded)
        sd_q = dequantize_per_channel(quantized)

        m = FoldedCaptchaCNN(input_c=INPUT_C, widths=widths, fc_width=args.fc_width)
        m.load_state_dict(sd_q)
        qca, qsa = evaluate(m, test_loader)
        n_bytes = sum(v["q"].numel() for v in quantized.values())
        print(f"  [Test int4 per-channel] CharAcc={qca:.2f}% SampleAcc={qsa:.2f}% "
              f"权重≈{n_bytes/1024:.0f}KB")

        if args.export:
            out = os.path.join(args.ckpt_dir, "best.qat-int4.pt")
            torch.save({"format": "scuocr-mixed-pt", "version": 3,
                        "source": "QAT", "quantized": quantized,
                        "int8_bytes": n_bytes}, out)
            print(f"  ✅ 已导出: {out} ({os.path.getsize(out)/1024:.0f} KB)")
            print(f"     转换 .scuocr: python export.py --mixed {out} -o zhjw-model.qat.scuocr")

    writer.close()
    print(f"\n完成！最佳验证 SampleAcc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()