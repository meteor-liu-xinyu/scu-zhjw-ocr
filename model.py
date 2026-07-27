"""
PyTorch 模型定义。

架构（预处理后缩放到 64×32 保持宽高比）：
  Input: 1 × 32 × 64
    → Conv3×3(1→24)  + BN + ReLU + MaxPool2×2  →  24 × 16 × 32
    → Conv3×3(24→40) + BN + ReLU + MaxPool2×2  →  40 ×  8 × 16
    → Conv3×3(40→64) + BN + ReLU + MaxPool2×2  →  64 ×  4 ×  8
    → Conv3×3(64→64) + BN + ReLU + MaxPool2×2  →  64 ×  2 ×  4
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


class CaptchaCNN(nn.Module):
    """验证码 CNN，4 层卷积 + AdaptiveAvgPool 保留位置信息。"""

    def __init__(
        self,
        input_c: int = INPUT_C,
        num_chars: int = NUM_CHARS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_c = input_c

        # ── 卷积层（Conv + BN + ReLU + MaxPool） ──
        self.conv1 = nn.Conv2d(input_c, 24, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 40, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(40)

        self.conv3 = nn.Conv2d(40, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        # 自适应池化：保留水平位置（对应 4 个字符）
        self.pool = nn.AdaptiveAvgPool2d((1, 4))  # → 64 × 1 × 4

        # ── 全连接层 ──
        self.fc1 = nn.Linear(64 * 1 * 4, 120, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(120, num_chars, bias=True)

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

        # AdaptiveAvgPool 保留 4 个位置 → (B, 64, 1, 4)
        x = self.pool(x)

        # Flatten → (B, 256)
        x = x.view(x.size(0), -1)

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
