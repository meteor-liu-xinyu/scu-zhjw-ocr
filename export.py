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

逐槽头模型的 INT8 导出（version=4）：
  当 state_dict 含 `head.*` 键（model.py 的 head_type="slot"）时，
  版本号写为 4，分类头张量为 head.fc.weight / head.fc.bias / head.slot_bias。
  体积：骨干 40.5 KB + 头 1.0 KB ≈ 41.7 KB（对比 fc 头的 66.4 KB）。

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
    # SE 注意力
    ("se.fc.0.weight", "se.fc.0.weight", False),
    ("se.fc.0.bias",   "se.fc.0.bias",   False),
    ("se.fc.2.weight", "se.fc.2.weight", False),
    ("se.fc.2.bias",   "se.fc.2.bias",   False),
    # 分类头（两种之一，见 HEAD_TENSORS_*）
    ("fc1.weight", "fc1.weight", False),
    ("fc1.bias",   "fc1.bias",   False),
    ("output_layer.weight", "output_layer.weight", False),
    ("output_layer.bias",   "output_layer.bias",   False),
    ("head.fc.weight",   "head.fc.weight",   False),
    ("head.fc.bias",     "head.fc.bias",     False),
    ("head.slot_bias",   "head.slot_bias",   False),
]

# 分类头张量（按 head_type 二选一）
HEAD_TENSORS_FC = ("fc1.weight", "fc1.bias", "output_layer.weight", "output_layer.bias")
HEAD_TENSORS_SLOT = ("head.fc.weight", "head.fc.bias", "head.slot_bias")
HEAD_TENSORS = [h for h in HEAD_TENSORS_FC] + [h for h in HEAD_TENSORS_SLOT]

# .scuocr 版本号
#   v1 = fp32（未折叠 BN）
#   v2 = int8 + BN 折叠，原跨槽头 fc1/output_layer
#   v3 = 混合精度（int8 + int4，见 quantize_mixed.py）
#   v4 = int8 + BN 折叠，逐槽共享头 head.fc/head.slot_bias
VERSION_FP32 = 1
VERSION_INT8 = 2
VERSION_MIXED = 3
VERSION_INT8_SLOT = 4


def detect_head_type(state_dict: dict) -> str:
    """从 state_dict 键名推断分类头类型。"""
    return "slot" if any(k.startswith("head.") for k in state_dict) else "fc"


def head_tensor_order(sd: dict) -> list[str]:
    """返回该 state_dict 实际存在的分类头张量顺序。"""
    order = HEAD_TENSORS_SLOT if detect_head_type(sd) == "slot" else HEAD_TENSORS_FC
    return [k for k in order if k in sd]


def conv_out_channels(sd: dict, i: int) -> int:
    """第 i 层卷积的输出通道数。兼容三种形式：
    标准 `conv{i}.weight` / 空间可分离 `conv{i}.v.weight` / 深度可分离 `conv{i}.pointwise.weight`。
    """
    if f"conv{i}.v.weight" in sd:
        return int(sd[f"conv{i}.v.weight"].shape[0])
    if f"conv{i}.pointwise.weight" in sd:
        return int(sd[f"conv{i}.pointwise.weight"].shape[0])
    return int(sd[f"conv{i}.weight"].shape[0])


def detect_separable(sd: dict) -> str:
    """从 state_dict 判断卷积分解方案（none / sep4 / sep34 / 未知）。"""
    from model import SEPARABLE_PLANS
    sep = {i for i in range(1, 5) if f"conv{i}.h.weight" in sd}
    for name, plan in SEPARABLE_PLANS.items():
        if plan == sep:
            return name
    raise ValueError(f"无法识别的可分离方案，检测到可分离层 {sorted(sep)}")


def ordered_tensor_names(sd: dict, folded: bool) -> list[str]:
    """规范化张量写入顺序：conv1..4(+bias) → bn1..4 → se → head。

    folded=True（BN 折叠后）时没有 bn*，只剩 conv 权重/偏置。
    支持空间可分离（conv{i}.h.weight / conv{i}.v.weight / conv{i}.v.bias）
    与深度可分离（conv{i}.depthwise.weight / conv{i}.pointwise.weight/.bias）。
    """
    names: list[str] = []
    for i in range(1, 5):
        # 空间可分离：h 在前、v 在后（与推理时的执行顺序一致）
        for suf in (("h.weight", "v.weight", "v.bias")
                    if f"conv{i}.h.weight" in sd
                    else (("depthwise.weight", "pointwise.weight", "pointwise.bias")
                          if f"conv{i}.depthwise.weight" in sd
                          else ("weight", "bias"))):
            k = f"conv{i}.{suf}"
            if k in sd:
                names.append(k)
        if not folded:
            for suf in ("weight", "bias", "running_mean", "running_var"):
                k = f"bn{i}.{suf}"
                if k in sd:
                    names.append(k)
    names.extend(k for k in ("se.fc.0.weight", "se.fc.0.bias",
                             "se.fc.2.weight", "se.fc.2.bias") if k in sd)
    names.extend(head_tensor_order(sd))
    return names


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

    for export_name in ordered_tensor_names(sd, folded=False):
        if export_name in sd:
            tensor = sd[export_name].contiguous().float().cpu()
        elif export_name.endswith(".bias") and export_name.split(".")[0].startswith("conv"):
            # Conv 层 bias=False：从对应 weight 推断通道数，零填充
            weight_key = export_name.replace(".bias", ".weight")
            c_out = sd[weight_key].shape[0]
            tensor = torch.zeros(c_out, dtype=torch.float32)
            if verbose:
                print(f"  {export_name:30s} shape=({c_out},)  [零填充]")
        else:
            continue
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


def infer_arch(state_dict: dict) -> tuple[tuple[int, ...], int, str, str]:
    """从 state_dict 推断 (widths, fc_width, head_type, separable)。

    兼容三种卷积形式（标准 / 空间可分离 / 深度可分离），见 conv_out_channels()。
    """
    widths = tuple(conv_out_channels(state_dict, i) for i in range(1, 5))
    fc_width = int(state_dict["fc1.weight"].shape[0]) if "fc1.weight" in state_dict else 0
    return widths, fc_width, detect_head_type(state_dict), detect_separable(state_dict)


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

    # 可选：验证模型结构兼容性（自动推断宽度 / fc 宽度 / 头类型 / 卷积分解）
    try:
        widths, fc_width, head_type, separable = infer_arch(state_dict)
        model = CaptchaCNN(input_c=INPUT_C, widths=widths, fc_width=fc_width,
                           head_type=head_type, separable=separable)
        model.load_state_dict(state_dict)
        if verbose:
            print(f"模型结构验证通过 ✓  widths={widths} fc_width={fc_width} "
                  f"head_type={head_type} separable={separable}")
    except Exception as e:
        print(f"警告: 模型加载验证失败: {e}")
        print("将继续尝试导出...")

    if int8:
        export_scuocr_int8(state_dict, output_path, verbose=verbose)
    else:
        export_scuocr(state_dict, output_path, verbose=verbose)


# ── INT8 量化导出（version=2）──────────────────────────────────────────

def fold_bn_into_conv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """将 BatchNorm 折叠进 Conv 权重/偏置（推理等价）。

    ⚠ 本函数**委托**给 `quantize.fold_bn_into_conv`，不要在本文件重复实现。
    （原先 export.py 有一份自己的硬编码版本，只认 `conv{i}.weight`，
    遇到空间可分离 `conv{i}.h/v` 或深度可分离 `conv{i}.depthwise/pointwise`
    会 KeyError，或在下方的正则过滤里被静默丢弃 ——
    结果 int8 导出路径与量化路径行为不一致，是个真实的隐藏 bug。）
    """
    from quantize import fold_bn_into_conv as _fold
    return _fold(state_dict)


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

    tensor_order = ordered_tensor_names(folded, folded=True)

    head_type = detect_head_type(folded)
    version = VERSION_INT8_SLOT if head_type == "slot" else VERSION_INT8

    tensors: list[tuple[str, torch.Tensor, float]] = []
    for name in tensor_order:
        t = folded[name]
        q, scale = quantize_tensor_symmetric(t)
        tensors.append((name, q, scale))

    with open(output_path, "wb") as f:
        f.write(b"SCUOCRLT")
        f.write(struct.pack("<I", version))       # 2 (fc 头) 或 4 (slot 头)
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
    混合精度量化导出（.scuocr version=3，支持 per-channel）。

    输入为 quantize_mixed.py 或 train_qat.py 生成的 quantized dict：
      {name: {q, scale, zero_point, bits, shape, [per_channel]}}
      - bits=8: q 为 int8 张量（1 字节/元素）
      - bits=4: q 为打包后的 uint8 张量（每 2 值 1 字节）
      - per_channel=True: scale 为输出通道数长度的 tensor（每通道独立 scale）

    格式（bits 高位 0x80=per-channel 标记）：
      [Header]
        magic:       8 bytes = "SCUOCRLT"
        version:     4 bytes uint32 LE = 3
        tensor_count: 4 bytes uint32 LE
      [Per Tensor]
        name_len:   4 bytes uint32 LE
        name:       N bytes UTF-8
        ndim:       4 bytes uint32 LE
        shape:      ndim × 4 bytes uint32 LE
        bits:       1 byte（bit7=per-channel标记，低7位=实际位宽）
        scale:      4 bytes float32 LE（per-tensor）或 num_scales×4 bytes（per-channel）
        zero_point: 4 bytes int32 LE（对称量化恒为 0）
        data:       product(shape) × (bits/8) bytes

    格式细节：
      - per-tensor（bits & 0x80 == 0）：scale 为 1 个 float32（兼容旧版）
      - per-channel（bits & 0x80 != 0）：在 zero_point 字段后写入：
          num_scales: 4 bytes uint32 LE
          scales:     num_scales × 4 bytes float32 LE
        data 在 scales 之后
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
            shape = item["shape"]
            name_bytes = name.encode("utf-8")
            per_channel = item.get("per_channel", False)

            # bits 编码：高位 0x80 标记 per-channel
            bits_field = bits | (0x80 if per_channel else 0x00)

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<B", bits_field))  # bits (1 byte)

            if per_channel:
                # per-channel：scale 是 tensor，先写占位 float32（兼容旧版解析器），再写 num_scales + scales
                f.write(struct.pack("<f", 0.0))     # 占位 scale（旧版解析器会读到 0，但版本检测会拦截）
                f.write(struct.pack("<i", 0))       # zero_point
                scales = item["scale"]
                if isinstance(scales, torch.Tensor):
                    scales = scales.cpu().numpy()
                f.write(struct.pack("<I", len(scales)))  # num_scales
                for s in scales:
                    f.write(struct.pack("<f", float(s)))  # each scale
            else:
                # per-tensor（兼容旧版）
                scale = item["scale"]
                if isinstance(scale, torch.Tensor):
                    scale = scale.item()
                f.write(struct.pack("<f", scale))   # scale (float32)
                f.write(struct.pack("<i", item["zero_point"]))  # zero_point

            f.write(q.numpy().tobytes())           # quantized data

            n_params = 1
            for d in shape:
                n_params *= d
            total_params += n_params
            total_bytes += q.numel()
            if verbose:
                scale_str = f"per-channel({len(scales)})" if per_channel else f"{scale:.6f}"
                print(f"  {name:30s} shape={str(shape):20s}  bits={bits}  "
                      f"scale={scale_str}  {q.numel():>6,} bytes")

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
