"""
逐槽共享头（SlotHead）的快速训练脚本 —— 方向 1 的落地实现。

思路：卷积骨干冻结（沿用已训练好的 checkpoint），只训练约 1KB 的分类头。
因为在冻结骨干上提特征只需一次前向，训练极快（CPU 上数十秒），
而 @train.py --head slot --freeze-backbone 每轮都要重跑骨干前向（约 21s/轮）。

流程：
  1. 构造 CaptchaCNN(head_type="slot")，加载骨干权重（strict=False）
  2. 冻结骨干，用 forward hook 取 AdaptiveAvgPool 后的 (B,4,C) 特征
  3. 特征标准化 → 训练 head（AdamW + 余弦）
  4. **把标准化的均值/方差折叠进 head.fc 权重**（推理时无需额外存统计量）
  5. 保存为标准 checkpoint 格式（可直接被 export.py 使用）并评估

用法：
    python train_slot_head.py                                  # 默认：窄版骨干
    python train_slot_head.py --export -o zhjw-model.slot.scuocr
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from model import CaptchaCNN, INPUT_C, NUM_CLASSES, CAPTCHA_LEN, CHARSET
from preprocess import ZhjwCaptchaDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="逐槽共享头训练（冻结骨干）")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--init-from", type=str, default="checkpoints/best.narrow.pt",
                   help="提供骨干权重的 checkpoint")
    p.add_argument("--widths", type=str, default="20,32,48,48")
    p.add_argument("-o", "--output", type=str, default="checkpoints/best.slot.pt",
                   help="输出 checkpoint 路径")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--feat-noise", type=float, default=0.0,
                   help="特征高斯噪声标准差（正则化，0=关闭）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--export", action="store_true", help="训练后导出 int8 .scuocr")
    p.add_argument("--export-path", type=str, default="zhjw-model.slot.scuocr")
    return p.parse_args()


def extract_features(model, loader):
    """冻结骨干前向，取 pool 输出 → (N, 4, C) 与标签 (N, 4)。"""
    bufs, labs = [], []

    def hook(_m, _i, o):
        bufs.append(o.detach())

    h = model.pool.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            model(x)
            labs.append(y)
    h.remove()
    F = torch.cat(bufs, 0).squeeze(2).permute(0, 2, 1).contiguous()   # (N,4,C)
    return F, torch.cat(labs, 0)


def evaluate_head(model, X, y, device, bs=2048):
    """X: (N,4,C) 标准特征, y: (N,4) → (char_acc, sample_acc)"""
    model.eval()
    correct_c = correct_s = total = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lg = model.head(X[i:i + bs].to(device)).argmax(-1)
            yy = y[i:i + bs].to(device)
            correct_c += (lg == yy).sum().item()
            correct_s += (lg == yy).all(dim=1).sum().item()
            total += len(yy)
    return correct_c / (total * CAPTCHA_LEN) * 100, correct_s / total * 100


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    widths = tuple(int(w) for w in args.widths.split(","))
    assert len(widths) == 4

    ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    fused = any(k.startswith("fc1.") for k in sd)
    print(f"骨干来源: {args.init_from}  epoch={ckpt.get('epoch','?')} "
          f"best_acc={ckpt.get('best_acc','?')}  原头={'fc(跨槽)' if fused else 'slot'}")

    model = CaptchaCNN(input_c=INPUT_C, widths=widths, head_type="slot")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    backbone_keys = [k for k in model.state_dict() if not k.startswith("head.")]
    head_keys = [k for k in model.state_dict() if k.startswith("head.")]
    backbone_missing = [k for k in missing if not k.startswith("head.")]
    print(f"加载骨干: {len(backbone_keys) - len(backbone_missing)}/{len(backbone_keys)} 张量，"
          f"新头 {len(head_keys)} 张量随机初始化")

    for name, p in model.named_parameters():
        if not name.startswith("head."):
            p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_bone = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"可训练参数: {n_train:,}（分类头）  冻结: {n_bone:,}（骨干）")

    # ── 数据 ──
    ds = ZhjwCaptchaDataset(data_dir=args.data_dir, augment=False)
    n = len(ds)
    tv = tt = int(n * 0.1)
    tr_idx, va_idx, te_idx = random_split(
        range(n), [n - tv - tt, tv, tt],
        generator=torch.Generator().manual_seed(42))
    print(f"划分: train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)} (seed=42)")

    def loader(idx, bs=256):
        return DataLoader(torch.utils.data.Subset(ds, list(idx)), batch_size=bs,
                          shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    print("\n提取冻结特征 ...")
    t0 = time.time()
    Ftr, ytr = extract_features(model, loader(tr_idx))
    Fva, yva = extract_features(model, loader(va_idx))
    Fte, yte = extract_features(model, loader(te_idx))
    C = Ftr.shape[-1]
    print(f"特征 {tuple(Ftr.shape)} / {tuple(Fva.shape)} / {tuple(Fte.shape)}  "
          f"用时 {time.time()-t0:.1f}s")

    # ── 特征标准化（训练后折叠进 head 权重） ──
    mu = Ftr.reshape(-1, C).mean(0)
    sdv = Ftr.reshape(-1, C).std(0).clamp(min=1e-6)
    Ztr, Zva, Zte = (Ftr - mu) / sdv, (Fva - mu) / sdv, (Fte - mu) / sdv

    # ── 训练分类头 ──
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    N = len(Ztr)
    best_acc, best_state = -1.0, None
    print(f"\n训练分类头: {args.epochs} epoch, batch {args.batch_size}, lr {args.lr}")
    t0 = time.time()
    for ep in range(args.epochs):
        model.head.train()
        perm = torch.randperm(N)
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = Ztr[idx]
            if args.feat_noise > 0:
                xb = xb + torch.randn_like(xb) * args.feat_noise
            opt.zero_grad()
            loss = crit(model.head(xb).reshape(-1, NUM_CLASSES), ytr[idx].reshape(-1))
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            ca, sa = evaluate_head(model, Zva, yva, device)
            if sa > best_acc:
                best_acc = sa
                best_state = {k: v.clone() for k, v in model.head.state_dict().items()}
            print(f"  epoch {ep+1:3d}/{args.epochs}  val 单字符 {ca:.2f}%  整图 {sa:.2f}%"
                  f"   ({time.time()-t0:.1f}s)")
    model.head.load_state_dict(best_state)
    print(f"最佳验证集整图准确率: {best_acc:.2f}%")

    ca, sa = evaluate_head(model, Zte, yte, device)
    print(f"[标准化特征][Test] 单字符 {ca:.2f}%  整图 {sa:.2f}%")

    # ── 把标准化折叠进 head.fc（推理无需额外统计量） ──
    with torch.no_grad():
        W = model.head.fc.weight.data.clone()          # (20, C)
        b = model.head.fc.bias.data.clone()            # (20,)
        Wn = W / sdv[None, :]
        bn = b - Wn @ mu
        model.head.fc.weight.data.copy_(Wn)
        model.head.fc.bias.data.copy_(bn)

    # 在原始（未标准化）特征上复测，确认折叠无损
    def evaluate_raw(model, F, y):
        model.eval()
        correct_c = correct_s = total = 0
        with torch.no_grad():
            for i in range(0, len(F), 2048):
                lg = model.head(F[i:i + 2048]).argmax(-1)
                yy = y[i:i + 2048]
                correct_c += (lg == yy).sum().item()
                correct_s += (lg == yy).all(dim=1).sum().item()
                total += len(yy)
        return correct_c / (total * CAPTCHA_LEN) * 100, correct_s / total * 100

    ca2, sa2 = evaluate_raw(model, Fte, yte)
    print(f"[折叠后原始特征][Test] 单字符 {ca2:.2f}%  整图 {sa2:.2f}%  "
          f"（应与上一行一致）")

    # ── 保存 checkpoint ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "epoch": args.epochs - 1,
        "model": model.state_dict(),
        "best_acc": sa2,
        "args": vars(args),
        "note": "slot head trained on frozen backbone features; standardization folded into head.fc",
    }, args.output)
    size_b = sum(p.numel() for p in model.parameters())
    print(f"\n✅ 已保存: {args.output}")
    print(f"   参数量 {size_b:,}（骨干 {n_bone:,} + 头 {n_train:,}）")

    # ── 导出 ──
    if args.export:
        from export import export_scuocr_int8
        export_scuocr_int8(model.state_dict(), args.export_path, verbose=False)
        sz = os.path.getsize(args.export_path)
        print(f"✅ 已导出 {args.export_path}: {sz:,} B ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
