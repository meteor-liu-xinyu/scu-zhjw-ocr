"""
SCU 教务处验证码合成数据生成器。

所有参数由真实数据反推（见 tmp/synth_analyze.py / tmp/synth_geom.py），
不涉及任何训练。

实测参数（真实 10000 张）：
  画布       180×60 RGB，JPEG quality=88（标准 IJG 量化表，逐图完全一致）
  字符色     RGB(237, 9, 9)
  背景       水平灰度渐变，左 (198,200,197) → 右 (245,245,245)；最外圈数像素偏绿
  布局       4 字符固定节距 20px，整体居中 → 槽心 [60, 80, 100, 120]
             逐字符 x 抖动 σ≈4.1px
  字形        基线 y≈45.7。逐类墨迹高/宽/顶/底见表（数字 cap 高 ≈31.2，
             x-height≈23.7 → x-height/cap≈0.76，属高 x-height 的粗无衬线体）
  干扰线      每图 1~2 条，线宽≈3px，水平跨度中位 146px，弯曲度 σ≈6px，倾角 σ≈8.7°

用法：
    from synth_gen import ZhjwSynthGenerator
    gen = ZhjwSynthGenerator(rng=np.random.default_rng(0))
    img = gen.generate("5gn8")        # -> (60,180,3) uint8 RGB
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
CHARSET = "2345678abcdefgmnpwxy"
W, H = 180, 60
SLOT_CENTERS = (60.0, 80.0, 100.0, 120.0)
INK_RGB = np.array([237, 9, 9], np.float32)
BG_LEFT = np.array([198, 200, 197], np.float32)
BG_RIGHT = np.array([245, 245, 245], np.float32)
JPEG_QUALITY = 88
DEFAULT_FONT = r"C:\Windows\Fonts\arialbd.ttf"

# 逐类几何（tmp/cv_out/geom_table.json 由 tmp/synth_geom.py 生成）
_GEOM_FALLBACK = {
    "2": (31.8, 2.8, 46.2), "3": (31.2, 3.0, 45.7), "4": (30.0, 2.5, 44.8),
    "5": (31.7, 3.2, 45.7), "6": (31.0, 3.2, 45.6), "7": (31.5, 2.6, 45.1),
    "8": (31.0, 3.0, 45.6), "a": (23.8, 4.1, 45.9), "b": (29.9, 5.0, 45.9),
    "c": (23.3, 4.2, 45.5), "d": (29.8, 6.0, 45.8), "e": (23.0, 4.2, 45.4),
    "f": (31.3, 3.0, 45.2), "g": (30.8, 5.2, 52.1), "m": (24.2, 3.1, 45.8),
    "n": (23.7, 4.1, 45.7), "p": (29.3, 5.8, 50.9), "w": (24.0, 3.2, 45.7),
    "x": (23.7, 3.3, 45.9), "y": (30.0, 4.9, 52.0),
}


def load_geom():
    p = os.path.join(ROOT, "tmp", "cv_out", "geom_table.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            t = json.load(f)
        return {c: (t[c]["h_mean"], t[c]["h_std"], t[c]["bot_mean"])
                for c in CHARSET if c in t}
    return dict(_GEOM_FALLBACK)


class ZhjwSynthGenerator:
    """参数化生成器：所有随机量都可调，便于做保真度标定。"""

    def __init__(
        self,
        font_path: str = DEFAULT_FONT,
        render_px: int = 200,
        blur_sigma: float = 1.2,
        rot_deg: float = 4.0,
        cx_jitter: float = 4.1,
        y_jitter: float = 1.0,
        line_width: float = 2.0,
        line_color: int = 10,
        n_lines: tuple[int, int] = (1, 2),
        jpeg_quality: int = JPEG_QUALITY,
        green_frame: int = 3,
        render_scale: float = 1.0,
        geom: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.font_path = font_path
        self.blur_sigma = blur_sigma
        self.rot_deg = rot_deg
        self.cx_jitter = cx_jitter
        self.y_jitter = y_jitter
        self.line_width = line_width
        self.line_color = line_color
        self.n_lines = n_lines
        self.jpeg_quality = jpeg_quality
        self.green_frame = green_frame
        # render_scale < 1：在小画布上渲染再放大到 180×60。
        # 这是复现真实图「细笔画 + 软边缘」的关键——真实图很可能是低分辨率渲染后放大的。
        self.render_scale = render_scale
        self.geom = geom or load_geom()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._glyphs = self._prerender(render_px)

    # ── 预渲染字符（大尺寸渲染，保证抗锯齿质量） ──
    def _prerender(self, px: int) -> dict[str, np.ndarray]:
        out = {}
        font = ImageFont.truetype(self.font_path, px)
        for c in CHARSET:
            img = Image.new("L", (px * 3, px * 3), 0)
            ImageDraw.Draw(img).text((px, px), c, fill=255, font=font)
            a = np.array(img, np.uint8)
            if a.max() < 30:
                continue
            rr = np.nonzero((a > 8).sum(axis=1) > 0)[0]
            cc = np.nonzero((a > 8).sum(axis=0) > 0)[0]
            out[c] = a[rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1]
        missing = [c for c in CHARSET if c not in out]
        if missing:
            raise RuntimeError(f"字体 {self.font_path} 缺少字符: {missing}")
        return out

    # ── 背景 ──
    def _background(self) -> np.ndarray:
        ramp = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
        bg = BG_LEFT[None, None, :] * (1 - ramp) + BG_RIGHT[None, None, :] * ramp
        bg = np.repeat(bg, H, axis=0)
        # 逐图轻微亮度扰动（真实图之间也有差异）
        bg *= self.rng.normal(1.0, 0.012)
        bg = np.clip(bg, 0, 255)
        if self.green_frame > 0:
            g = self.green_frame
            bg[:g, :, 1] *= 0.72; bg[:g, :, 2] *= 0.70; bg[:g, :, 0] *= 0.62
            bg[-g:, :, 1] *= 0.86
            bg[:, :g, 1] *= 0.80
            bg[:, -g:, 1] *= 0.98
        return bg

    # ── 单个字形：缩放 + 旋转 → RGBA 图层 ──
    def _glyph_layer(self, c: str, target_h: float, angle: float):
        g = self._glyphs[c].astype(np.float32) / 255.0
        # 先旋转（在原始分辨率上转，避免缩放后插值损失）
        if abs(angle) > 1e-3:
            h, w = g.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            g = cv2.warpAffine(g, M, (nw, nh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        # 旋转后重取紧致 bbox，再缩放到目标墨迹高
        rr = np.nonzero((g > 0.03).sum(axis=1) > 0)[0]
        cc = np.nonzero((g > 0.03).sum(axis=0) > 0)[0]
        g = g[rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1]
        s = target_h / g.shape[0]
        nw = max(2, int(round(g.shape[1] * s)))
        nh = max(2, int(round(target_h)))
        g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
        if self.blur_sigma > 0:
            g = cv2.GaussianBlur(g, (0, 0), self.blur_sigma)
        return np.clip(g, 0, 1)

    # ── 干扰线 ──
    # 实测：每列游程数 ≈1（多为单条连续曲线），线宽 2.3px（p90 3.0），
    # 中线 y ≈ 29.7（σ 9.3），弯曲量 σ≈11px，78.5% 上下包络同向弯曲。
    def _draw_lines(self, canvas: np.ndarray):
        n = int(self.rng.integers(self.n_lines[0], self.n_lines[1] + 1))
        for _ in range(n):
            full = self.rng.random() < 0.72
            if full:
                x0, x1 = float(self.rng.uniform(-8, 12)), float(self.rng.uniform(W - 12, W + 8))
            else:
                x0 = float(self.rng.uniform(-5, W * 0.55))
                x1 = x0 + float(self.rng.uniform(45, 110))
            y0 = float(self.rng.normal(29.7, 11.0))
            y1 = float(self.rng.normal(29.7, 11.0))
            # 中点控制量：产生 ±11px 量级的弯曲
            yc = (y0 + y1) / 2 + float(self.rng.normal(0, 11.0))
            pts = []
            for t in np.linspace(0, 1, 80):
                x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * ((x0 + x1) / 2) + t ** 2 * x1
                y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * yc + t ** 2 * y1
                pts.append((x, y))
            pts = np.array(pts, np.float32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False,
                          (self.line_color,) * 3,
                          thickness=max(1, int(round(self.line_width))),
                          lineType=cv2.LINE_AA)

    # ── 生成 ──
    def generate(self, label: str | None = None, jpeg: bool = True) -> np.ndarray:
        if label is None:
            label = "".join(self.rng.choice(list(CHARSET), 4))
        assert len(label) == 4 and all(c in CHARSET for c in label)

        bg = self._background()
        ink = np.zeros((H, W), np.float32)
        for s, c in enumerate(label):
            hm, hs, bot = self.geom[c]
            target_h = float(np.clip(self.rng.normal(hm, hs), 12, 40))
            y_bot = float(self.rng.normal(bot, self.y_jitter))
            angle = float(self.rng.normal(0, self.rot_deg))
            g = self._glyph_layer(c, target_h, angle)
            cx = SLOT_CENTERS[s] + float(self.rng.normal(0, self.cx_jitter))
            x0 = int(round(cx - g.shape[1] / 2))
            y0 = int(round(y_bot - g.shape[0]))
            # 用 maximum 叠加（字符重叠时取覆盖更强者）
            xs0, ys0 = max(x0, 0), max(y0, 0)
            xs1, ys1 = min(x0 + g.shape[1], W), min(y0 + g.shape[0], H)
            if xs1 <= xs0 or ys1 <= ys0:
                continue
            sub = g[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
            ink[ys0:ys1, xs0:xs1] = np.maximum(ink[ys0:ys1, xs0:xs1], sub)

        canvas = bg * (1 - ink[:, :, None]) + INK_RGB[None, None, :] * ink[:, :, None]
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        self._draw_lines(canvas)

        # 低分辨率渲染再放大：复现真实图「细笔画 + 软边缘」的观感
        if self.render_scale < 0.999:
            sw = max(8, int(round(W * self.render_scale)))
            sh = max(4, int(round(H * self.render_scale)))
            small = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)
            canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

        if jpeg:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                canvas = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR),
                                      cv2.COLOR_BGR2RGB)
        return canvas


def _demo():
    import argparse
    ap = argparse.ArgumentParser(description="生成合成验证码样例")
    ap.add_argument("-n", type=int, default=36)
    ap.add_argument("--font", default=DEFAULT_FONT)
    ap.add_argument("--blur", type=float, default=0.8)
    ap.add_argument("--rot", type=float, default=3.0)
    ap.add_argument("-o", "--output", default="tmp/cv_out/synth_samples.png")
    a = ap.parse_args()
    gen = ZhjwSynthGenerator(font_path=a.font, blur_sigma=a.blur, rot_deg=a.rot,
                             rng=np.random.default_rng(0))
    tiles = []
    for i in range(a.n):
        lab = "".join(np.random.default_rng(i).choice(list(CHARSET), 4))
        img = gen.generate(lab)
        big = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        tiles.append(np.hstack([big, np.full((big.shape[0], 4, 3), 255, np.uint8)]))
    per = 4
    rows = [np.hstack(tiles[i:i + per]) for i in range(0, len(tiles), per)]
    sheet = np.vstack(rows)
    cv2.imwrite(a.output, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"saved {a.output}  ({len(tiles)} 张，2 倍放大)")
    print(f"字体 {os.path.basename(a.font)}  blur={a.blur}  rot={a.rot}")


if __name__ == "__main__":
    _demo()
