"""
教务处验证码 CNN 训练脚本。

用法：
    python train.py                          # 默认训练
    python train.py --epochs 50 --batch 64   # 自定义参数
    python train.py --resume checkpoints/latest.pt  # 断点续训

数据集：
    从 data/ 目录加载 IMAGES.zip + label.csv。
    下载地址：https://github.com/SunnyHaze/SCU_OAA-website-Captcha-training-set
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path

# Windows 控制台默认 GBK，无法打印 emoji，强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from model import CaptchaCNN, INPUT_C, INPUT_H, INPUT_W, CAPTCHA_LEN, NUM_CLASSES, CHARSET
from preprocess import ZhjwCaptchaDataset, collate_fn

# ── 配置 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="教务处验证码 CNN 训练")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="数据集目录（包含 IMAGES/ 和 label.csv）")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="初始学习率")
    parser.add_argument("--lr-min", type=float, default=3e-5,
                        help="最低学习率（余弦退火）")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="标签平滑系数")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪最大范数")
    parser.add_argument("--warmup", type=int, default=10,
                        help="学习率预热轮数")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="测试集比例（从训练集中划分，不参与任何训练决策）")
    parser.add_argument("--test-only", type=str, default=None,
                        help="仅运行测试评估，传入 checkpoint 路径")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader 工作进程数")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备 (cuda/cpu)，默认自动选择")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="TensorBoard 日志目录")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints",
                        help="checkpoint 保存目录")
    parser.add_argument("--widths", type=str, default="24,40,64,64",
                        help="各卷积层输出通道，逗号分隔（方案 A 简化用 24,40,48,48）")
    parser.add_argument("--fc-width", type=int, default=120,
                        help="fc1 输出宽度（方案 A 简化用 80）")
    parser.add_argument("--amp", action="store_true",
                        help="启用自动混合精度（GPU 训练提速 2-3 倍）")
    parser.add_argument("--use-depthwise", action="store_true",
                        help="使用深度可分离卷积（参数量约 -55%，50K 参数）")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 训练 ────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    grad_clip: float = 0.0,
    scaler=None,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_samples = 0
    total_samples = 0
    start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)      # (B, C, H, W)
        labels = labels.to(device)      # (B, 4)

        B = images.size(0)
        optimizer.zero_grad()

        # 自动混合精度：GPU 上 fp16 前向/反向，权重保持 fp32
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)          # (B, 80)
            logits = logits.view(B, CAPTCHA_LEN, NUM_CLASSES)  # (B, 4, 20)
            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                labels.reshape(-1),
            )

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度裁剪
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

        # 逐位准确率
        pred = logits.argmax(dim=-1)    # (B, 4)
        correct_chars += (pred == labels).sum().item()
        total_chars += B * CAPTCHA_LEN
        correct_samples += (pred == labels).all(dim=1).sum().item()
        total_samples += B

        if batch_idx % 50 == 0:
            acc = correct_chars / total_chars * 100
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader):4d} "
                  f"| Loss: {loss.item():.4f} | CharAcc: {acc:.2f}%")

    avg_loss = total_loss / len(loader)
    char_acc = correct_chars / total_chars * 100
    sample_acc = correct_samples / total_samples * 100
    elapsed = time.time() - start

    print(f"  ── Epoch {epoch:3d} done | Loss: {avg_loss:.4f} "
          f"| CharAcc: {char_acc:.2f}% | SampleAcc: {sample_acc:.2f}% "
          f"| Time: {elapsed:.1f}s")

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/char_train", char_acc, epoch)
    writer.add_scalar("Acc/sample_train", sample_acc, epoch)

    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> tuple[float, float, float]:
    return _eval_loop(model, loader, criterion, device, epoch, writer, prefix="Val")


@torch.no_grad()
def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """在测试集上做最终评估（不写 TensorBoard）。"""
    loss, char_acc, sample_acc = _eval_loop(model, loader, criterion, device, -1, None, prefix="Test")
    return loss, char_acc, sample_acc


def _eval_loop(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
    prefix: str = "",
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_samples = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        B = images.size(0)

        logits = model(images)
        logits = logits.view(B, CAPTCHA_LEN, NUM_CLASSES)

        loss = criterion(
            logits.reshape(-1, NUM_CLASSES),
            labels.reshape(-1),
        )
        total_loss += loss.item()

        pred = logits.argmax(dim=-1)
        correct_chars += (pred == labels).sum().item()
        total_chars += B * CAPTCHA_LEN
        correct_samples += (pred == labels).all(dim=1).sum().item()
        total_samples += B

    avg_loss = total_loss / len(loader)
    char_acc = correct_chars / total_chars * 100
    sample_acc = correct_samples / total_samples * 100

    label = f" [{prefix}]" if prefix else ""
    print(f"  {label} Epoch {epoch:3d} | Loss: {avg_loss:.4f} "
          f"| CharAcc: {char_acc:.2f}% | SampleAcc: {sample_acc:.2f}%")

    if writer is not None:
        writer.add_scalar(f"Loss/{prefix}", avg_loss, epoch)
        writer.add_scalar(f"Acc/char_{prefix}", char_acc, epoch)
        writer.add_scalar(f"Acc/sample_{prefix}", sample_acc, epoch)

    return avg_loss, char_acc, sample_acc


class _AugmentDataset(torch.utils.data.Dataset):
    """训练集包装器：加载图片后做数据增强。"""
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        from preprocess import _augment
        img, label = self.base[self.indices[idx]]
        img_np = img.squeeze(0).numpy()
        img_np = _augment(img_np)
        return torch.from_numpy(img_np[np.newaxis, :, :].astype(np.float32)), label


def main():
    args = parse_args()
    set_seed(args.seed)

    # ── 设备 ──
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # GPU 优化：固定输入尺寸下启用 cudnn benchmark 加速卷积
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("已启用 cudnn.benchmark")

    # 解析模型宽度（方案 A 简化：24,40,48,48 + fc 80）
    widths = tuple(int(w) for w in args.widths.split(","))
    if len(widths) != 4:
        print(f"错误: --widths 需要 4 个值，收到 {len(widths)}")
        sys.exit(1)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        print("警告: --amp 需要 GPU，当前为 CPU，已忽略")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 仅测试模式 ──
    if args.test_only:
        print(f"测试模式: 加载 {args.test_only}")
        if not os.path.isfile(args.test_only):
            print(f"错误: 找不到 checkpoint {args.test_only}")
            sys.exit(1)

        # 加载数据集（只需要测试集）
        full_dataset = ZhjwCaptchaDataset(data_dir=args.data_dir)
        test_size = int(len(full_dataset) * args.test_split)
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size - test_size
        _, _, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

        model = CaptchaCNN(input_c=INPUT_C, widths=widths, fc_width=args.fc_width,
                           use_depthwise=args.use_depthwise).to(device)
        ckpt = torch.load(args.test_only, map_location=device)
        model.load_state_dict(ckpt["model"])
        criterion = nn.CrossEntropyLoss()

        test_loss, test_char_acc, test_sample_acc = test(model, test_loader, criterion, device)
        print(f"\n[Test] Loss: {test_loss:.4f} | CharAcc: {test_char_acc:.2f}% | SampleAcc: {test_sample_acc:.2f}%")
        return

    # ── 数据集 ──
    print("加载数据集...")
    full_dataset = ZhjwCaptchaDataset(data_dir=args.data_dir, augment=False)

    # 按 seed 固定划分：训练 / 验证 / 测试
    test_size = int(len(full_dataset) * args.test_split)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size - test_size
    train_indices, val_indices, test_indices = random_split(
        range(len(full_dataset)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # 训练集：用包装器开启数据增强
    train_dataset = _AugmentDataset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── 模型 ──
    model = CaptchaCNN(input_c=INPUT_C, widths=widths, fc_width=args.fc_width,
                       use_depthwise=args.use_depthwise).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    if args.use_depthwise:
        print("使用深度可分离卷积 (Depthwise Separable)")

    # ── 损失 & 优化器 ──
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率预热 → 余弦退火
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,            # 从 lr*0.01 开始
        total_iters=args.warmup,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup),
        eta_min=args.lr_min,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup],
    )

    # ── 断点续训 ──
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"恢复训练: epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ── TensorBoard ──
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ── 训练循环 ──
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"开始训练 (共 {args.epochs} 轮)...")
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n┌─ Epoch {epoch:3d}/{args.epochs} | lr={lr:.2e}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            grad_clip=args.grad_clip,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss, char_acc, sample_acc = validate(
            model, val_loader, criterion, device, epoch, writer,
        )

        scheduler.step()

        # ── 保存 checkpoint ──
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": max(best_acc, sample_acc),
            "args": vars(args),
        }
        # 最新 checkpoint
        torch.save(ckpt, os.path.join(args.ckpt_dir, "latest.pt"))
        # 最优 checkpoint（用 >= 保证首个 epoch 也保存，避免 best.pt 缺失）
        if sample_acc >= best_acc:
            best_acc = sample_acc
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"  ⭐ 新的最佳验证集准确率: {best_acc:.2f}%")

    # ── 在测试集上做最终评估（用验证集上最好的 checkpoint） ──
    print(f"\n加载最佳 checkpoint ({os.path.join(args.ckpt_dir, 'best.pt')}) 在测试集上评估...")
    best_ckpt = torch.load(os.path.join(args.ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_loss, test_char_acc, test_sample_acc = test(model, test_loader, criterion, device)
    print(f"  [Test] Loss: {test_loss:.4f} | CharAcc: {test_char_acc:.2f}% | SampleAcc: {test_sample_acc:.2f}%")
    print(f"\n训练完成！最佳验证集样本准确率: {best_acc:.2f}% | 测试集样本准确率: {test_sample_acc:.2f}%")
    print(f"模型导出: python export.py {os.path.join(args.ckpt_dir, 'best.pt')}")
    writer.close()


if __name__ == "__main__":
    main()
