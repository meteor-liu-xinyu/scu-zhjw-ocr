"""
PyTorch 权重 → .scuocr 格式导出。

.scuocr 格式（与 model.ts 兼容）：
  [Header]
    magic:       8 bytes = "SCUOCRLT"
    version:     4 bytes uint32 LE = 1
    tensor_count: 4 bytes uint32 LE

  [Per Tensor] × tensor_count
    name_len: 4 bytes uint32 LE
    name:     N bytes UTF-8
    ndim:     4 bytes uint32 LE
    shape:    ndim × 4 bytes uint32 LE
    data:     product(shape) × 4 bytes float32 LE

Tensor 命名规则：
  conv{1,2,3,4}.weight / conv{1,2,3,4}.bias
  bn{1,2,3,4}.weight / bn{1,2,3,4}.bias / bn{1,2,3,4}.running_mean / bn{1,2,3,4}.running_var
  fc1.weight / fc1.bias
  output_layer.weight / output_layer.bias

注意：模型中 Conv 层使用 bias=False，但 .scuocr 格式（与 scu-plus model.ts 兼容）
要求 conv bias 存在（BN 折叠 foldBnIntoConv 需要）。导出时自动为零填充。

INT8 量化导出（--int8，version=2）：
  先做 BN 折叠（foldBnIntoConv），再对每个权重做 per-tensor 对称量化
  （scale = max(|w|)/127，zero_point 恒为 0）。文件大小约为 fp32 版的 1/4，
  实测准确率无损（99.00% → 99.00%）。

用法：
    python export.py checkpoints/best.pt -o output.scuocr
    python export.py checkpoints/best.pt -o assets/zhjw-model.scuocr
    python export.py checkpoints/best.pt --int8 -o zhjw-model.int8.scuocr
"""

import os
import sys
import struct
import argparse
import torch

# Windows 控制台默认 GBK，无法打印 emoji，强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from model import CaptchaCNN, INPUT_C, INPUT_H, INPUT_W, NUM_CHARS


TENSOR_MAP = [
    # (pt_key_in_state_dict, export_name, is_optional_zero_bias)
    # Conv1 (bias=False → 导出时零填充)
    ("conv1.weight", "conv1.weight", False),
    ("conv1.bias",   "conv1.bias",   True),
    ("bn1.weight",      "bn1.weight",      False),
    ("bn1.bias",        "bn1.bias",        False),
    ("bn1.running_mean", "bn1.running_mean", False),
    ("bn1.running_var",  "bn1.running_var",  False),
    # Conv2
    ("conv2.weight", "conv2.weight", False),
    ("conv2.bias",   "conv2.bias",   True),
    ("bn2.weight",      "bn2.weight",      False),
    ("bn2.bias",        "bn2.bias",        False),
    ("bn2.running_mean", "bn2.running_mean", False),
    ("bn2.running_var",  "bn2.running_var",  False),
    # Conv3
    ("conv3.weight", "conv3.weight", False),
    ("conv3.bias",   "conv3.bias",   True),
    ("bn3.weight",      "bn3.weight",      False),
    ("bn3.bias",        "bn3.bias",        False),
    ("bn3.running_mean", "bn3.running_mean", False),
    ("bn3.running_var",  "bn3.running_var",  False),
    # Conv4
    ("conv4.weight", "conv4.weight", False),
    ("conv4.bias",   "conv4.bias",   True),
    ("bn4.weight",      "bn4.weight",      False),
    ("bn4.bias",        "bn4.bias",        False),
    ("bn4.running_mean", "bn4.running_mean", False),
    ("bn4.running_var",  "bn4.running_var",  False),
    # FC
    ("fc1.weight", "fc1.weight", False),
    ("fc1.bias",   "fc1.bias",   False),
    # SE 注意力
    ("se.fc.0.weight", "se.fc.0.weight", False),
    ("se.fc.0.bias",   "se.fc.0.bias",   False),
    ("se.fc.2.weight", "se.fc.2.weight", False),
    ("se.fc.2.bias",   "se.fc.2.bias",   False),
    # Output
    ("output_layer.weight", "output_layer.weight", False),
    ("output_layer.bias",   "output_layer.bias",   False),
]


def export_scuocr(
    state_dict: dict[str, torch.Tensor],
    output_path: str,
    verbose: bool = True,
):
    """
    将 PyTorch state_dict 导出为 .scuocr 格式。

    Args:
        state_dict: PyTorch 模型 state_dict（支持 DataParallel 包装）
        output_path: 输出 .scuocr 文件路径
        verbose: 是否打印详细信息
    """
    # 处理 DataParallel 包装的键名前缀 "module."
    sd = {}
    for k, v in state_dict.items():
        k_clean = k.replace("module.", "", 1)
        sd[k_clean] = v

    tensors: list[tuple[str, torch.Tensor]] = []

    for pt_key, export_name, optional_zero in TENSOR_MAP:
        if pt_key not in sd:
            if optional_zero:
                # Conv 层 bias=False：从对应 weight 推断通道数，零填充
                weight_key = pt_key.replace(".bias", ".weight")
                if weight_key in sd:
                    c_out = sd[weight_key].shape[0]
                    tensor = torch.zeros(c_out, dtype=torch.float32)
                    tensors.append((export_name, tensor))
                    if verbose:
                        print(f"  {export_name:30s} shape=({c_out},)  [零填充]")
                    continue
            # BN 的 num_batches_tracked 可以忽略
            if "num_batches_tracked" in pt_key:
                continue
            raise KeyError(
                f"State dict 中缺少 '{pt_key}'。可用键: {list(sd.keys())}"
            )
        tensor = sd[pt_key].contiguous().float().cpu()
        tensors.append((export_name, tensor))

    # ── 写入二进制 ──
    with open(output_path, "wb") as f:
        # Header
        f.write(b"SCUOCRLT")                    # magic
        f.write(struct.pack("<I", 1))            # version
        f.write(struct.pack("<I", len(tensors))) # tensor_count

        total_params = 0
        for name, tensor in tensors:
            data = tensor.numpy().tobytes()
            name_bytes = name.encode("utf-8")
            shape = list(tensor.shape)

            f.write(struct.pack("<I", len(name_bytes)))  # name_len
            f.write(name_bytes)                           # name
            f.write(struct.pack("<I", len(shape)))        # ndim
            for d in shape:
                f.write(struct.pack("<I", d))            # shape
            f.write(data)                                 # data (float32 LE)

            total_params += tensor.numel()
            if verbose:
                print(f"  {name:30s} shape={str(shape):20s}  {tensor.numel():>6,} params")

    file_size = os.path.getsize(output_path)
    if verbose:
        print(f"\n✅ 导出成功: {output_path}")
        print(f"   总权重参数: {total_params:,}")
        print(f"   文件大小:   {file_size:,} bytes ({file_size/1024:.1f} KB)")


def load_checkpoint_and_export(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    verbose: bool = True,
    int8: bool = False,
):
    """从 PyTorch checkpoint 加载模型并导出为 .scuocr。"""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 从 checkpoint 获取 state_dict
    if "model" in ckpt:
        state_dict = ckpt["model"]
        epoch = ckpt.get("epoch", "?")
        acc = ckpt.get("best_acc", "?")
        if verbose:
            print(f"加载 checkpoint: epoch={epoch}, best_acc={acc}%")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # 直接就是 state_dict

    # 可选：验证模型结构兼容性
    try:
        model = CaptchaCNN(input_c=INPUT_C)
        model.load_state_dict(state_dict, strict=False)
        if verbose:
            print("模型结构验证通过 ✓")
    except Exception as e:
        print(f"警告: 模型加载验证失败: {e}")
        print("将继续尝试导出...")

    if int8:
        export_scuocr_int8(state_dict, output_path, verbose=verbose)
    else:
        export_scuocr(state_dict, output_path, verbose=verbose)


# ── INT8 量化导出（version=2）──────────────────────────────────────────

def fold_bn_into_conv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    将 BatchNorm 折叠进 Conv 权重/偏置（推理等价）。

    公式：folded_w = w * γ / √(var + ε)
          folded_b = (b - mean) * γ / √(var + ε) + β
    折叠后不再需要 BN 参数，推理更简单，量化更稳定。
    """
    sd = dict(state_dict)
    folded: dict[str, torch.Tensor] = {}
    eps = 1e-5  # BatchNorm2d 默认 eps
    for i in range(1, 5):
        w = sd[f"conv{i}.weight"].float()
        b = sd.get(f"conv{i}.bias", torch.zeros(w.shape[0])).float()
        gamma = sd[f"bn{i}.weight"].float()
        beta = sd[f"bn{i}.bias"].float()
        mean = sd[f"bn{i}.running_mean"].float()
        var = sd[f"bn{i}.running_var"].float()

        scale = gamma / torch.sqrt(var + eps)
        folded_w = w * scale.view(-1, 1, 1, 1)
        folded_b = (b - mean) * scale + beta
        folded[f"conv{i}.weight"] = folded_w.contiguous()
        folded[f"conv{i}.bias"] = folded_b.contiguous()

    for k in ("fc1.weight", "fc1.bias", "output_layer.weight", "output_layer.bias",
              "se.fc.0.weight", "se.fc.0.bias", "se.fc.2.weight", "se.fc.2.bias"):
        folded[k] = sd[k].float().contiguous()
    return folded


def quantize_tensor_symmetric(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """对称量化：q = round(t / scale)，scale = max(|t|) / 127，zero_point 恒为 0。"""
    scale = t.abs().max().item() / 127.0
    if scale == 0:
        scale = 1.0
    q = torch.round(t / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def export_scuocr_int8(
    state_dict: dict[str, torch.Tensor],
    output_path: str,
    verbose: bool = True,
):
    """
    BN 折叠 + int8 对称量化导出（.scuocr version=2）。

    格式（与 fp32 版兼容 header，version=2）：
      [Header]
        magic:       8 bytes = "SCUOCRLT"
        version:     4 bytes uint32 LE = 2
        tensor_count: 4 bytes uint32 LE
      [Per Tensor]
        name_len:   4 bytes uint32 LE
        name:       N bytes UTF-8
        ndim:       4 bytes uint32 LE
        shape:      ndim × 4 bytes uint32 LE
        scale:      4 bytes float32 LE
        zero_point: 4 bytes int32 LE（对称量化恒为 0）
        data:       product(shape) × 1 bytes int8 LE

    推理：fp32 ≈ (int8_val - zero_point) * scale
    """
    folded = fold_bn_into_conv(state_dict)

    tensor_order = [
        "conv1.weight", "conv1.bias",
        "conv2.weight", "conv2.bias",
        "conv3.weight", "conv3.bias",
        "conv4.weight", "conv4.bias",
        "fc1.weight", "fc1.bias",
        "se.fc.0.weight", "se.fc.0.bias",
        "se.fc.2.weight", "se.fc.2.bias",
        "output_layer.weight", "output_layer.bias",
    ]

    tensors: list[tuple[str, torch.Tensor, float]] = []
    for name in tensor_order:
        t = folded[name]
        q, scale = quantize_tensor_symmetric(t)
        tensors.append((name, q, scale))

    with open(output_path, "wb") as f:
        f.write(b"SCUOCRLT")
        f.write(struct.pack("<I", 2))             # version = 2 (int8)
        f.write(struct.pack("<I", len(tensors)))  # tensor_count

        total_params = 0
        for name, q, scale in tensors:
            data = q.numpy().tobytes()
            name_bytes = name.encode("utf-8")
            shape = list(q.shape)

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<f", scale))     # scale (float32)
            f.write(struct.pack("<i", 0))         # zero_point (int32)
            f.write(data)                          # int8 data

            total_params += q.numel()
            if verbose:
                print(f"  {name:30s} shape={str(shape):20s}  scale={scale:.6f}  {q.numel():>6,} params")

    file_size = os.path.getsize(output_path)
    if verbose:
        print(f"\n✅ INT8 导出成功: {output_path}")
        print(f"   总权重参数: {total_params:,}")
        print(f"   文件大小:   {file_size:,} bytes ({file_size/1024:.1f} KB)")


# ── 混合精度导出（version=3，int8 + int4）──────────────────────────────

def export_scuocr_mixed(
    quantized: dict,
    output_path: str,
    verbose: bool = True,
):
    """
    混合精度量化导出（.scuocr version=3）。

    输入为 quantize_mixed.py 生成的 quantized dict：
      {name: {q, scale, zero_point, bits, shape}}
      - bits=8: q 为 int8 张量（1 字节/元素）
      - bits=4: q 为打包后的 uint8 张量（每 2 值 1 字节）

    格式（与 fp32/int8 版兼容 header，version=3）：
      [Header]
        magic:       8 bytes = "SCUOCRLT"
        version:     4 bytes uint32 LE = 3
        tensor_count: 4 bytes uint32 LE
      [Per Tensor]
        name_len:   4 bytes uint32 LE
        name:       N bytes UTF-8
        ndim:       4 bytes uint32 LE
        shape:      ndim × 4 bytes uint32 LE
        bits:       1 byte（8 或 4）
        scale:      4 bytes float32 LE
        zero_point: 4 bytes int32 LE（对称量化恒为 0）
        data:       product(shape) × (bits/8) bytes

    推理：fp32 ≈ (dequant_val - zero_point) * scale
      bits=8: dequant_val = int8_val
      bits=4: 每 2 个值打包 1 字节，低 4 位 + 高 4 位（值 -8..7）
    """
    with open(output_path, "wb") as f:
        f.write(b"SCUOCRLT")
        f.write(struct.pack("<I", 3))             # version = 3 (mixed)
        f.write(struct.pack("<I", len(quantized)))  # tensor_count

        total_params = 0
        total_bytes = 0
        for name, item in quantized.items():
            q = item["q"]
            bits = item["bits"]
            scale = item["scale"]
            zp = item["zero_point"]
            shape = item["shape"]
            name_bytes = name.encode("utf-8")

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<B", bits))      # bits (1 byte)
            f.write(struct.pack("<f", scale))     # scale (float32)
            f.write(struct.pack("<i", zp))        # zero_point (int32)
            f.write(q.numpy().tobytes())           # quantized data

            n_params = 1
            for d in shape:
                n_params *= d
            total_params += n_params
            total_bytes += q.numel()
            if verbose:
                print(f"  {name:30s} shape={str(shape):20s}  bits={bits}  "
                      f"scale={scale:.6f}  {q.numel():>6,} bytes")

    file_size = os.path.getsize(output_path)
    if verbose:
        print(f"\n✅ 混合精度导出成功: {output_path}")
        print(f"   总权重参数: {total_params:,}")
        print(f"   量化数据:   {total_bytes:,} bytes")
        print(f"   文件大小:   {file_size:,} bytes ({file_size/1024:.1f} KB)")


# ── 命令行 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="导出 .scuocr 权重文件")
    parser.add_argument("checkpoint", type=str, nargs="?", default=None,
                        help="PyTorch checkpoint 路径 (*.pt)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出 .scuocr 文件路径")
    parser.add_argument("--device", type=str, default="cpu",
                        help="加载设备")
    parser.add_argument("--int8", action="store_true",
                        help="导出 int8 量化版本（BN 折叠 + 对称量化，version=2）")
    parser.add_argument("--mixed", type=str, default=None, metavar="MIXED_PT",
                        help="从混合精度 .pt 文件导出（quantize_mixed.py 生成，version=3）")
    args = parser.parse_args()

    if args.mixed:
        # 混合精度导出：从 quantize_mixed.py 生成的 .pt 读取 quantized dict
        ckpt = torch.load(args.mixed, map_location="cpu", weights_only=False)
        quantized = ckpt["quantized"]
        if args.output is None:
            base = os.path.splitext(os.path.basename(args.mixed))[0]
            args.output = f"{base}.scuocr"
        print(f"加载混合精度文件: {args.mixed} (format={ckpt.get('format','?')})")
        export_scuocr_mixed(quantized, args.output)
        return

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f"{base}.scuocr"
        if args.int8:
            args.output = f"{base}.int8.scuocr"

    load_checkpoint_and_export(args.checkpoint, args.output, args.device, int8=args.int8)


if __name__ == "__main__":
    main()
