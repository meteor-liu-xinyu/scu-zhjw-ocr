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

用法：
    python export.py checkpoints/best.pt -o output.scuocr
    python export.py checkpoints/best.pt -o assets/zhjw-model.scuocr
"""

import os
import sys
import struct
import argparse
import torch

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

    export_scuocr(state_dict, output_path, verbose=verbose)


# ── 命令行 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="导出 .scuocr 权重文件")
    parser.add_argument("checkpoint", type=str,
                        help="PyTorch checkpoint 路径 (*.pt)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出 .scuocr 文件路径")
    parser.add_argument("--device", type=str, default="cpu",
                        help="加载设备")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f"{base}.scuocr"

    load_checkpoint_and_export(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
