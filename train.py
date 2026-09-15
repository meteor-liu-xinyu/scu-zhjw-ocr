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
from preprocess import ZhjwCaptchaDataset, collate_fn, CROP_PRESETS

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
                        help="fc1 输出宽度（仅 --head fc 生效）")
    parser.add_argument("--head", type=str, default="fc", choices=["fc", "slot"],
                        help="分类头类型：fc=原跨槽头 fc1+output（66KB）；"
                             "slot=逐槽共享头（约 1KB，总体积 41.7KB，精度更高）")
    parser.add_argument("--init-from", type=str, default=None,
                        help="初始化权重来源 checkpoint（strict=False，只加载能对上的键）")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="冻结卷积骨干（含 SE），只训练分类头")
    parser.add_argument("--amp", action="store_true",
                        help="启用自动混合精度（GPU 训练提速 2-3 倍）")
    parser.add_argument("--separable", type=str, default="none",
                        choices=["none", "sep4", "sep34"],
                        help="空间可分离卷积方案：none=全标准3×3；"
                             "sep4=conv4 换(1×3→3×1)（实测零掉点、参数-16%%）；"
                             "sep34=conv3+conv4（实测-0.1点、参数-31%%）")
    parser.add_argument("--use-depthwise", action="store_true",
                        help="使用深度可分离卷积（参数量约 -55%%，50K 参数）")
    # ── 进阶训练技巧（均不改变部署格式，导出仍是 v2 int8） ──
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"],
                        help="优化器：adamw（默认，稳）或 sgd+momentum（CNN 常更优）")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "cosine_restarts"],
                        help="学习率调度：单程余弦（默认）或余弦重启 SGDR")
    parser.add_argument("--restart-period", type=int, default=40,
                        help="SGDR 重启周期 T_0（仅 --schedule cosine_restarts 生效）")
    parser.add_argument("--bn-momentum", type=float, default=0.1,
                        help="BatchNorm momentum（默认0.1；可试0.01更稳/0.2更快）")
    parser.add_argument("--aug-profile", type=str, default="default",
                        choices=["default", "strong", "shift", "shift_strong"],
                        help="数据增强档位：default/strong 为原行为；"
                             "shift(+平移±4模型px)/shift_strong(±6) 用于修平移不变性缺失")
    parser.add_argument("--crop", type=str, default="current",
                        choices=["current", "wide"],
                        help="裁剪窗预设。current=x40:140,y5:55（含 1.67%% 横向/15.37%% 纵向切边）；"
                             "wide=x32:150,y3:58（基本不切）。⚠ 换窗会使旧权重作废，必须重训")
    parser.add_argument("--crop-jitter", type=int, default=0,
                        help="训练集裁剪窗随机偏移上限（原图像素），模拟版式整体平移。"
                             "只作用于训练集；验证/测试集恒为 0（否则污染评估）")
    parser.add_argument("--early-stop", type=int, default=0,
                        help="早停耐心（轮）：连续 N 轮验证集无提升则停止，0=关闭")
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
    freeze_bn: bool = False,
) -> float:
    model.train()
    if freeze_bn:
        # 冻结骨干时 BN 保持 eval：不更新 running stats，骨干行为与导出时一致
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
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
    def __init__(self, base, indices, profile: str = "default"):
        self.base = base
        self.indices = indices
        self.profile = profile
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        from preprocess import _augment
        img, label = self.base[self.indices[idx]]
        img_np = img.squeeze(0).numpy()
        img_np = _augment(img_np, profile=self.profile)
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
                           use_depthwise=args.use_depthwise,
                           separable=args.separable,
                           head_type=args.head).to(device)
        ckpt = torch.load(args.test_only, map_location=device)
        model.load_state_dict(ckpt["model"])
        criterion = nn.CrossEntropyLoss()

        test_loss, test_char_acc, test_sample_acc = test(model, test_loader, criterion, device)
        print(f"\n[Test] Loss: {test_loss:.4f} | CharAcc: {test_char_acc:.2f}% | SampleAcc: {test_sample_acc:.2f}%")
        return

    # ── 数据集 ──
    print("加载数据集...")
    # 验证/测试集必须与训练集用**同一个裁剪窗**，否则评估没有意义
    full_dataset = ZhjwCaptchaDataset(data_dir=args.data_dir, augment=False,
                                      crop=args.crop)

    # 按 seed 固定划分：训练 / 验证 / 测试
    test_size = int(len(full_dataset) * args.test_split)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size - test_size
    train_indices, val_indices, test_indices = random_split(
        range(len(full_dataset)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # 训练集：数据增强。
    # crop_jitter>0 时裁剪窗抖动必须在**数据集内部**做（要拿原图），
    # 所以另建一个开了 augment 的 base；否则沿用原来的包装器路径（行为不变）。
    if args.crop_jitter > 0:
        train_base = ZhjwCaptchaDataset(data_dir=args.data_dir, augment=True,
                                        aug_profile=args.aug_profile,
                                        crop=args.crop,
                                        crop_jitter=args.crop_jitter)
        train_dataset = torch.utils.data.Subset(train_base, train_indices)
        print(f"[aug] 裁剪窗抖动 ±{args.crop_jitter}px（仅训练集）+ 增强档位 {args.aug_profile}")
    else:
        train_dataset = _AugmentDataset(full_dataset, train_indices,
                                        profile=args.aug_profile)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    if args.crop != "current":
        x1, x2, y1, y2 = CROP_PRESETS[args.crop]
        print(f"[crop] 裁剪窗 = {args.crop}  x{x1}:{x2} y{y1}:{y2}"
              f"（{x2-x1}×{y2-y1}）")
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
                       use_depthwise=args.use_depthwise,
                       separable=args.separable,
                       bn_momentum=args.bn_momentum,
                       head_type=args.head).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    head_keys = ("head.", "fc1.", "output_layer.")
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if n.startswith(head_keys))
    print(f"模型参数量: {total_params:,}  (分类头 {head_params:,} / "
          f"骨干 {total_params - head_params:,})   head_type={args.head}")

    # ── 可选：从已有 checkpoint 初始化权重（strict=False） ──
    if args.init_from:
        if not os.path.isfile(args.init_from):
            print(f"错误: 找不到初始化权重 {args.init_from}")
            sys.exit(1)
        init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        init_sd = init_ckpt["model"] if "model" in init_ckpt else init_ckpt
        missing, unexpected = model.load_state_dict(init_sd, strict=False)
        loaded = len(model.state_dict()) - len(missing)
        print(f"从 {args.init_from} 初始化: 加载 {loaded} 个张量")
        if missing:
            print(f"  未加载（新头随机初始化）: {list(missing)}")

    # ── 可选：冻结骨干，只训练分类头 ──
    if args.freeze_backbone:
        frozen = 0
        for name, p in model.named_parameters():
            if not name.startswith(head_keys):
                p.requires_grad = False
                frozen += p.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"已冻结骨干 {frozen:,} 参数，可训练 {trainable:,} 参数")

    if args.use_depthwise:
        print("使用深度可分离卷积 (Depthwise Separable)")
    if args.bn_momentum != 0.1:
        print(f"BatchNorm momentum = {args.bn_momentum}")

    # ── 损失 & 优化器 ──
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            trainable,
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
        print(f"优化器: SGD(momentum=0.9, nesterov=True)")
    else:
        optimizer = optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率预热 →（单程余弦 或 余弦重启 SGDR）
    # 注意：warmup<=0 时不能走 SequentialLR(milestones=[0])——首次 step 时
    # last_epoch 直接跳到 1，永远匹配不上 milestone 0，lr 会卡在 start_factor。
    if args.schedule == "cosine_restarts":
        restart_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, args.restart_period),
            T_mult=1,
            eta_min=args.lr_min,
        )
        if args.warmup > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=args.warmup)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, restart_scheduler],
                milestones=[args.warmup],
            )
        else:
            scheduler = restart_scheduler
        print(f"学习率调度: SGDR(T_0={args.restart_period}, eta_min={args.lr_min})")
    else:
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - max(0, args.warmup)),
            eta_min=args.lr_min,
        )
        if args.warmup > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=args.warmup)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[args.warmup],
            )
        else:
            # optim.lr_scheduler.LinearLR(total_iters=0) 会 get_lr 除零，直接改用余弦
            scheduler = cosine_scheduler
        print(f"学习率调度: 单程余弦(T_max={max(1, args.epochs - max(0, args.warmup))}, "
              f"eta_min={args.lr_min})" + ("，无预热" if args.warmup <= 0 else ""))

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
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n┌─ Epoch {epoch:3d}/{args.epochs} | lr={lr:.2e}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            grad_clip=args.grad_clip,
            scaler=scaler, use_amp=use_amp,
            freeze_bn=args.freeze_backbone,
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
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 早停：连续 N 轮验证集无提升则停止（0=关闭）
        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"\n⏹ 早停：连续 {args.early_stop} 轮验证集无提升，停止训练。")
            break

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
