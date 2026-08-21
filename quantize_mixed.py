"""混合精度量化：逐层 int4 敏感性分析 + 生成混合精度 .pt 文件。

用法：
    python quantize_mixed.py checkpoints/best.pt -o checkpoints/best.mixed.pt --eval
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from model import INPUT_C, INPUT_H, INPUT_W, CAPTCHA_LEN, NUM_CLASSES, NUM_CHARS, SEBlock
from preprocess import ZhjwCaptchaDataset, collate_fn
from quantize import fold_bn_into_conv, FoldedCaptchaCNN, evaluate


def quantize_tensor(t: torch.Tensor, bits: int) -> tuple[torch.Tensor, float, int]:
    """对称量化，bits=8 或 4。返回 (q, scale, zero_point)。"""
    qmax = 127 if bits == 8 else 7
    scale = t.abs().max().item() / qmax
    if scale == 0:
        scale = 1.0
    q = torch.round(t / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scale, 0


def quantize_per_channel(t: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, torch.Tensor, int]:
    """per-channel 对称量化：每输出通道独立 scale（仅对 dim>=2 有效）。

    Args:
        t: 权重张量，shape (out, in, ...) 或 (out, in)
        bits: 位宽（4 或 8）
    Returns:
        (q, scales, 0): q 为 int8 张量，scales 为每输出通道的 scale 向量
    """
    qmax = 127 if bits == 8 else 7
    if t.dim() >= 2:
        flat = t.reshape(t.shape[0], -1)
        scales = flat.abs().max(dim=1).values.clamp(min=1e-12) / qmax
        q = torch.round(flat / scales[:, None]).clamp(-qmax - 1, qmax).to(torch.int8)
        q = q.reshape(t.shape)
        return q, scales, 0
    else:
        # bias 退化为 per-tensor
        return quantize_tensor(t, bits)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """把 int8 张量（值 -8..7）打包成 uint8（每 2 个值 1 字节），真正 4-bit 存储。"""
    q = q.reshape(-1)
    n = q.numel()
    if n % 2 == 1:
        q = torch.cat([q, torch.zeros(1, dtype=q.dtype)])
    q_u = ((q + 8) & 0x0F).to(torch.uint8)  # -8..7 -> 0..15
    packed = (q_u[0::2] | (q_u[1::2] << 4)).to(torch.uint8)
    return packed


def unpack_int4(packed: torch.Tensor, shape) -> torch.Tensor:
    """解包 uint8 回 int8 张量（值 -8..7）。"""
    lo = ((packed & 0x0F).to(torch.int8)) - 8
    hi = (((packed >> 4) & 0x0F).to(torch.int8)) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(-1)
    n = torch.Size(shape).numel()
    return q[:n].reshape(shape)


def quantize_mixed(state_dict: dict, plan: dict, per_channel: bool = False) -> dict:
    """按 plan 量化，plan[name] = bits。返回 {name: {q, scale, zero_point, bits, shape, [per_channel]}}。

    bits=8 时 q 为 int8 张量；bits=4 时 q 为打包后的 uint8 张量（每 2 值 1 字节）。
    per_channel=True 时 conv 权重用 per-channel 量化（每输出通道独立 scale）。
    """
    folded = fold_bn_into_conv(state_dict)
    quantized = {}
    for name, t in folded.items():
        bits = plan.get(name, 8)
        if per_channel and t.dim() >= 2 and not name.startswith("se"):
            q, scales, zp = quantize_per_channel(t, bits)
            is_pc = True
        else:
            q, scale, zp = quantize_tensor(t, bits)
            scales = scale
            is_pc = False
        if bits == 4:
            q = pack_int4(q)
        quantized[name] = {
            "q": q, "scale": scales, "zero_point": zp, "bits": bits,
            "shape": list(t.shape),
            "per_channel": is_pc,
        }
    return quantized


def dequantize(quantized: dict) -> dict[str, torch.Tensor]:
    sd = {}
    for name, item in quantized.items():
        if item["bits"] == 4:
            q = unpack_int4(item["q"], item["shape"])
        else:
            q = item["q"]
        if item.get("per_channel"):
            # per-channel 反量化
            s = item["scale"]
            if isinstance(s, torch.Tensor) and s.dim() >= 1:
                broadcast = s.reshape(-1, *([1] * (q.dim() - 1)))
                sd[name] = (q.float() * broadcast).reshape(item["shape"])
            else:
                sd[name] = (q.float() - item["zero_point"]) * float(s)
        else:
            sd[name] = (q.float() - item["zero_point"]) * float(item["scale"])
    return sd


def size_bytes(quantized: dict) -> int:
    """实际存储字节数：int8 每元素 1 字节，int4 打包后每 2 元素 1 字节。"""
    return sum(v["q"].numel() for v in quantized.values())


def main():
    parser = argparse.ArgumentParser(description="混合精度量化（int8 + int4，支持 per-channel）")
    parser.add_argument("checkpoint", type=str, help="PyTorch checkpoint 路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出 .pt 路径")
    parser.add_argument("--eval", action="store_true", help="量化后验证精度")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensitivity", action="store_true",
                        help="逐层 int4 敏感性分析（该层 int4，其余 int8）")
    parser.add_argument("--use-depthwise", action="store_true",
                        help="深度可分离卷积模型（conv{i}.depthwise/pointwise）")
    parser.add_argument("--per-channel", action="store_true",
                        help="per-channel 量化（conv 每输出通道独立 scale，精度更高）")
    parser.add_argument("--widths", type=str, default="24,40,64,64",
                        help="卷积通道宽（如 20,32,48,48 / 16,24,40,40）")
    parser.add_argument("--fc-width", type=int, default=120, help="fc1 宽度")
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6 不支持 weights_only
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    print(f"加载 checkpoint: epoch={ckpt.get('epoch','?')}, best_acc={ckpt.get('best_acc','?')}%")

    widths = tuple(int(w) for w in args.widths.split(","))
    assert len(widths) == 4

    # 测试集
    full = ZhjwCaptchaDataset(data_dir=args.data_dir, augment=False)
    test_size = int(len(full) * 0.1)
    val_size = int(len(full) * 0.1)
    train_size = len(full) - val_size - test_size
    _, _, test_idx = random_split(
        range(len(full)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed))
    test_loader = DataLoader(torch.utils.data.Subset(full, test_idx),
                             batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # fp32 基线
    folded = fold_bn_into_conv(state_dict)
    m_fp32 = FoldedCaptchaCNN(input_c=INPUT_C, widths=widths, fc_width=args.fc_width,
                              use_depthwise=args.use_depthwise)
    m_fp32.load_state_dict(folded)
    ca, sa = evaluate(m_fp32, test_loader)
    print(f"[fp32] CharAcc={ca:.2f}%  SampleAcc={sa:.2f}%")

    # 敏感性分析
    if args.sensitivity:
        print("\n逐层 int4 敏感性（该层 int4，其余 int8）:")
        for name in folded:
            plan = {k: 8 for k in folded}
            plan[name] = 4
            q = quantize_mixed(state_dict, plan, per_channel=args.per_channel)
            m = FoldedCaptchaCNN(input_c=INPUT_C, widths=widths, fc_width=args.fc_width,
                                 use_depthwise=args.use_depthwise)
            m.load_state_dict(dequantize(q))
            ca, sa = evaluate(m, test_loader)
            print(f"  int4: {name:24s} SampleAcc={sa:.2f}%")

    # 默认混合方案：Conv int8 + FC/SE int4
    plan = {k: 8 for k in folded}
    for k in folded:
        if k.startswith(("fc", "output", "se")):
            plan[k] = 4
    quantized = quantize_mixed(state_dict, plan, per_channel=args.per_channel)
    n_bytes = size_bytes(quantized)

    m_mixed = FoldedCaptchaCNN(input_c=INPUT_C, widths=widths, fc_width=args.fc_width,
                               use_depthwise=args.use_depthwise)
    m_mixed.load_state_dict(dequantize(quantized))
    ca, sa = evaluate(m_mixed, test_loader)
    pc_str = " per-channel" if args.per_channel else ""
    print(f"\n[混合: Conv int8 + FC/SE int4{pc_str}] "
          f"CharAcc={ca:.2f}%  SampleAcc={sa:.2f}%  权重={n_bytes/1024:.1f}KB")

    # 保存
    output = args.output or os.path.splitext(args.checkpoint)[0] + ".mixed.pt"
    torch.save({
        "format": "scuocr-mixed-pt",
        "version": 3,
        "source": os.path.basename(args.checkpoint),
        "source_epoch": ckpt.get("epoch", None),
        "source_best_acc": ckpt.get("best_acc", None),
        "quantized": quantized,
        "fp32_bytes": sum(v.numel() * 4 for v in folded.values()),
        "int8_bytes": n_bytes,
    }, output)
    print(f"✅ 已保存: {output} ({os.path.getsize(output)/1024:.1f} KB)")


if __name__ == "__main__":
    main()