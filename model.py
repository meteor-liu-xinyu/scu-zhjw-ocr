"""
PyTorch 模型定义。

架构（预处理后缩放到 64×32 保持宽高比）：
  Input: 1 × 32 × 64
    → Conv3×3(1→24)  + BN + ReLU + MaxPool2×2  →  24 × 16 × 32
    → Conv3×3(24→40) + BN + ReLU + MaxPool2×2  →  40 ×  8 × 16
    → Conv3×3(40→64) + BN + ReLU + MaxPool2×2  →  64 ×  4 ×  8
    → Conv3×3(64→64) + BN + ReLU + MaxPool2×2  →  64 ×  2 ×  4
    → SE 注意力（通道重标定，+512 参数）
    → AdaptiveAvgPool(1, 4) → 保持水平位置信息
    → Flatten → 256
    → Linear(256→120) + ReLU + Dropout(0.3)
    → Linear(120→80)   ← 4 位 × 20 类

参数量约 110K（与 scu-plus 的 CaptchaModelLite 同级，可部署浏览器插件）。
"""

import torch
import torch.nn as nn


# ── 字符集 ──────────────────────────────────────────────────────────────
CHARSET = "2345678abcdefgmnpwxy"
NUM_CLASSES = len(CHARSET)  # 20
CAPTCHA_LEN = 4
NUM_CHARS = CAPTCHA_LEN * NUM_CLASSES  # 80

# ── 输入尺寸（缩放到 64×32 保持 2:1 宽高比） ──────────────────────────
INPUT_H = 32
INPUT_W = 64
INPUT_C = 1  # 单通道，归一化到 [0,1]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力：自适应校准通道权重，提升特征判别力。

    参数量极小（64 通道 reduction=16 仅 512 参数），几乎不增加模型大小。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Squeeze: 全局平均池化 → (B, C)
        w_ = self.fc(x.mean(dim=(2, 3))).view(b, c, 1, 1)
        # Excitation: 通道重标定
        return x * w_


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：DW(3×3) + PW(1×1)。

    参数量约为标准卷积的 1/8（MobileNet 同款思路）：
      标准: C_in × C_out × 3 × 3
      深度可分离: C_in × 3 × 3 (depthwise) + C_in × C_out (pointwise)
    保持通道数不变，精度损失通常很小。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SpatialSeparableConv(nn.Module):
    """空间可分离卷积：(1×3) 保持 Cin 通道，再接 (3×1) 升到 Cout。

    把 3×3 核近似为「先横后纵」两级一维核，参数量：
        标准 3×3:      Cin·Cout·9
        空间可分离:     Cin·Cin·3  +  Cin·Cout·3
    Cin=Cout=48 时 20,736 → 13,824（-33%）；Cin=32,Cout=48 时 13,824 → 7,680（-44%）。

    ⚠ 这是**空间**上的分解，与 `DepthwiseSeparableConv`（在**通道**上分解）完全不同。
    后者已被实测排除（参数量 -55% 但保不住精度）；本模块**已实测可用**：
    在 (20,32,48,48)/slot 模型上把 conv4 换成它，端到端 **99.80% → 99.80%（零掉点）**，
    参数量 -16.4%（见 tmp/sep_feasibility.py、tmp/apply_separable.py）。
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.h = nn.Conv2d(in_channels, in_channels, (1, 3),
                           padding=(0, 1), bias=False)
        self.v = nn.Conv2d(in_channels, out_channels, (3, 1),
                           padding=(1, 0), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.h(x))


# 卷积分解方案：把哪些层换成空间可分离
SEPARABLE_PLANS = {
    "none": set(),
    "sep4": {4},          # 只换 conv4 —— 实测零掉点、参数 -16%
    "sep34": {3, 4},      # conv3+conv4 —— 实测 -0.40 点、参数 -31%
}


class SlotHead(nn.Module):
    """逐槽共享分类头：对 4 个位置槽施加同一个线性映射 + 每槽独立偏置。

    参数量 = C×20 + 20 + 4×20（槽偏置）。
    C=48 时为 1,060 参数（约 1.0 KB int8），
    对比原 fc1(192→96)+output(96→80) 的 26,288 参数——跨槽混合被实测证明有害。

    Args:
        in_features: 每槽特征维（即最后一层卷积输出通道数）
        num_classes: 每槽类别数（20）
        num_slots: 字符槽数（4）
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = NUM_CLASSES,
        num_slots: int = CAPTCHA_LEN,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=True)
        # 每槽独立偏置：捕获各位置的字符先验分布差异
        self.slot_bias = nn.Parameter(torch.zeros(num_slots, num_classes))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Args: feat (B, num_slots, in_features) → (B, num_slots, num_classes)"""
        return self.fc(feat) + self.slot_bias


class CaptchaCNN(nn.Module):
    """验证码 CNN，4 层卷积 + SE 注意力 + AdaptiveAvgPool 保留位置信息。

    Args:
        widths: 各卷积层输出通道数（默认 (24, 40, 64, 64) 为原版 110K 参数；
                方案 A 简化版用 (24, 40, 48, 48) 约 69K 参数）。
        fc_width: fc1 输出宽度（默认 120；方案 A 用 80）。仅 head_type="fc" 生效。
        use_depthwise: 使用深度可分离卷积（参数量约 -55%，**已被实测排除**）。
        separable: 空间可分离方案，见 SEPARABLE_PLANS
            - "none":  全标准 3×3（默认，向后兼容）
            - "sep4":  conv4 换空间可分离（**实测零掉点、参数 -16%**，推荐）
            - "sep34": conv3+conv4 都换（实测 -0.40 点、参数 -31%）
        head_type: 分类头类型
            - "fc":   原方案 fc1(w4×4→fc_width) + ReLU + Dropout + output(fc_width→80)
            - "slot": 逐槽共享头 SlotHead（约 1KB，实测精度更高、体积小得多）
    """

    def __init__(
        self,
        input_c: int = INPUT_C,
        num_chars: int = NUM_CHARS,
        dropout: float = 0.3,
        widths: tuple[int, ...] = (24, 40, 64, 64),
        fc_width: int = 120,
        use_depthwise: bool = False,
        bn_momentum: float = 0.1,
        head_type: str = "fc",
        separable: str = "none",
    ):
        super().__init__()
        if head_type not in ("fc", "slot"):
            raise ValueError(f"head_type 必须是 'fc' 或 'slot'，收到 {head_type!r}")
        if separable not in SEPARABLE_PLANS:
            raise ValueError(f"separable 必须是 {list(SEPARABLE_PLANS)} 之一，"
                             f"收到 {separable!r}")
        if separable != "none" and use_depthwise:
            raise ValueError("separable（空间分解）与 use_depthwise（通道分解）不能同时用")
        self.input_c = input_c
        self.widths = widths
        self.fc_width = fc_width
        self.use_depthwise = use_depthwise
        self.bn_momentum = bn_momentum
        self.head_type = head_type
        self.separable = separable

        # 卷积层：标准 / 深度可分离（通道）/ 空间可分离（横纵）
        conv_fn = DepthwiseSeparableConv if use_depthwise else nn.Conv2d
        sep_layers = SEPARABLE_PLANS[separable]

        def make(i, cin, cout):
            if i in sep_layers:
                return SpatialSeparableConv(cin, cout)
            return conv_fn(cin, cout, 3, padding=1, bias=False)

        # ── 卷积层（Conv + BN + ReLU + MaxPool） ──
        w1, w2, w3, w4 = widths
        self.conv1 = make(1, input_c, w1)
        self.bn1 = nn.BatchNorm2d(w1, momentum=bn_momentum)

        self.conv2 = make(2, w1, w2)
        self.bn2 = nn.BatchNorm2d(w2, momentum=bn_momentum)

        self.conv3 = make(3, w2, w3)
        self.bn3 = nn.BatchNorm2d(w3, momentum=bn_momentum)

        self.conv4 = make(4, w3, w4)
        self.bn4 = nn.BatchNorm2d(w4, momentum=bn_momentum)

        # SE 注意力：提升通道判别力
        self.se = SEBlock(w4, reduction=16)

        # 自适应池化：保留水平位置（对应 4 个字符）
        self.pool = nn.AdaptiveAvgPool2d((1, 4))  # → w4 × 1 × 4

        # ── 分类头 ──
        if head_type == "slot":
            self.head = SlotHead(w4 * 1, NUM_CLASSES, CAPTCHA_LEN)
            self.head_out_features = w4 * 1
        else:
            self.fc1 = nn.Linear(w4 * 1 * 4, fc_width, bias=True)
            self.dropout = nn.Dropout(dropout)
            self.output_layer = nn.Linear(fc_width, num_chars, bias=True)

        # ── 权重初始化 ──
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        Args:
            x: (B, 1, 32, 64) 张量，预处理后灰度图 [0, 1]
        Returns:
            logits: (B, 80) 张量
        """
        # Conv1 → BN → ReLU → Pool  (24×16×32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)

        # Conv2 → BN → ReLU → Pool  (40×8×16)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)

        # Conv3 → BN → ReLU → Pool  (64×4×8)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)

        # Conv4 → BN → ReLU → Pool  (64×2×4)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)

        # SE 注意力：通道重标定 (64×2×4)
        x = self.se(x)

        # AdaptiveAvgPool 保留 4 个位置 → (B, C, 1, 4)
        x = self.pool(x)

        if self.head_type == "slot":
            # (B, C, 1, 4) → (B, 4, C)：每槽一个 C 维特征
            feat = x.squeeze(2).permute(0, 2, 1)
            logits = self.head(feat)              # (B, 4, 20)
            return logits.reshape(logits.size(0), -1)   # (B, 80)

        # Flatten → (B, w4*4)
        x = x.reshape(x.size(0), -1)

        # FC → Dropout → Output
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output_layer(x)

        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """
        推理：输入 (B, C, H, W) → 解码文本 + 置信度。
        Returns:
            texts: list[str]
            confidences: (B,) tensor of min char probability
        """
        logits = self.forward(x)  # (B, 80)
        B = logits.size(0)
        logits = logits.view(B, CAPTCHA_LEN, NUM_CLASSES)  # (B, 4, 20)

        probs = torch.softmax(logits, dim=-1)  # (B, 4, 20)
        best_probs, best_idx = probs.max(dim=-1)  # (B, 4), (B, 4)
        confidences = best_probs.min(dim=-1).values  # (B,)

        texts = []
        for b in range(B):
            chars = [CHARSET[idx] for idx in best_idx[b]]
            texts.append("".join(chars))

        return texts, confidences


# ── 快速测试 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CaptchaCNN(input_c=INPUT_C)
    model.eval()

    dummy = torch.randn(2, INPUT_C, INPUT_H, INPUT_W)
    out = model(dummy)
    print(f"输出形状: {out.shape}  (期望: (2, {NUM_CHARS}))")

    texts, confs = model.predict(dummy)
    print(f"示例文本: {texts}")
    print(f"置信度: {confs}")
