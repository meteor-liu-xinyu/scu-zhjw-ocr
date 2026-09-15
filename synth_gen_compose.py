"""
基于真实字形重组的合成数据生成器（保证字形保真）。

思路：把真实训练图里的单字符「抠」出来存成 alpha 层（含真实模糊与形变），
再随机重组到新的位置/尺寸/旋转/干扰线/JPEG 上，得到无限多的新样本。

相比字体渲染路线的优势：字形、模糊、边缘软度都完全来自真实数据。
代价：不能产生训练集里没有的字形（对 QAT/蒸馏/增强已经足够）。

关键实现：
  - alpha = clip((redness - 3) / 225, 0, 1)，保留抗锯齿的软边缘
  - 黑线覆盖处对 alpha 做 inpaint（否则字形会有洞）
  - 用与 synth_gen 相同的布局/线条/JPEG 参数

用法：
    python synth_gen_compose.py --build            # 构建字形池
    python synth_gen_compose.py --demo -n 24       # 出样例图
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CHARSET = "2345678abcdefgmnpwxy"
W, H = 180, 60
SLOT_CENTERS = (60.0, 80.0, 100.0, 120.0)
INK_RGB = np.array([237, 9, 9], np.float32)
POOL_PATH = os.path.join(ROOT, "tmp", "cv_out", "glyph_pool.pkl")

_GEOM = None


def _geom():
    global _GEOM
    if _GEOM is None:
        import json
        p = os.path.join(ROOT, "tmp", "cv_out", "geom_table.json")
        with open(p, encoding="utf-8") as f:
            t = json.load(f)
        _GEOM = {c: (t[c]["h_mean"], t[c]["h_std"], t[c]["bot_mean"]) for c in CHARSET}
    return _GEOM


def dp_split(m, n=4, wmin=9, wmax=36, lam=0.06):
    col = m.sum(axis=0).astype(np.float64)
    nz = np.nonzero(col > 0)[0]
    if len(nz) == 0:
        return None
    lo, hi = nz[0], nz[-1] + 1
    span, avg = hi - lo, (hi - lo) / n
    INF = 1e18
    dp = np.full((n + 1, hi + 2), INF)
    bk = np.zeros((n + 1, hi + 2), np.int32)
    dp[0][lo] = 0.0
    for j in range(1, n + 1):
        for x in range(lo, hi + 1):
            for w in range(wmin, wmax + 1):
                p = x - w
                if p < lo or dp[j - 1][p] >= INF:
                    continue
                c = dp[j - 1][p] + col[x] + lam * (w - avg) ** 2
                if c < dp[j][x]:
                    dp[j][x] = c
                    bk[j][x] = p
    if dp[n][hi] >= INF:
        return None
    segs, x = [], hi
    for j in range(n, 0, -1):
        p = int(bk[j][x])
        segs.append((p, x))
        x = p
    return sorted(segs)


def extract_glyphs(bgr):
    """从一张真实图里抠出 4 个字形（含真实像素色与软掩膜）；失败返回 None。

    关键：真实字符的墨迹多数是「浅红」——细笔画被 JPEG 色度下采样冲淡，
    只有笔画核心才接近纯红。因此必须保留 **原始像素色**，
    不能用「红度→alpha + 固定纯红」重建（那样墨迹强度会掉到 1/4）。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    black = ((r < 130) & (g < 130) & (b < 130)).astype(np.uint8)
    black_d = cv2.dilate(black, np.ones((3, 3), np.uint8), iterations=1)

    # 干扰线覆盖处：对 RGB 做 inpaint（细线，Telea 效果好，且能保住红色）
    if black_d.sum() > 0:
        rgb_clean = cv2.inpaint(rgb.astype(np.uint8), black_d, 3,
                                cv2.INPAINT_TELEA).astype(np.float32)
    else:
        rgb_clean = rgb

    rr_, gg_, bb_ = rgb_clean[:, :, 0], rgb_clean[:, :, 1], rgb_clean[:, :, 2]
    redness = rr_ - np.maximum(gg_, bb_)
    hard = (redness > 40.0).astype(np.uint8)
    segs = dp_split(hard)
    if segs is None:
        return None

    # ── alpha matting 求真实覆盖率 ──
    # 观测 P = B(1-t) + C·t  →  t = <B-P, B-C> / |B-C|²
    # B 为该图的逐列背景（水平渐变，逐列中位稳健），C 为纯墨色 INK_RGB。
    # 这一步必须精确：用「红度/常数」近似 alpha 会让重建笔画系统性偏淡，
    # 实测使模型整图准确率掉 8 个点。
    non_ink = (hard == 0) & (black_d == 0)
    colB = np.zeros((1, W, 3), np.float32)
    for x in range(W):
        sel = non_ink[:, x]
        colB[0, x] = rgb_clean[:, x][sel].mean(axis=0) if sel.sum() > 3 else \
            np.array([220., 220., 218.])
    B = np.repeat(colB, H, axis=0)
    C = np.array(INK_RGB, np.float32)
    dBC = B - C[None, None, :]
    t_map = ((B - rgb_clean) * dBC).sum(-1) / np.maximum((dBC * dBC).sum(-1), 1e-6)
    t_map = np.clip(t_map, 0, 1).astype(np.float32)

    out = []
    for (x1, x2) in segs:
        a0, a1 = max(0, x1 - 3), min(W, x2 + 3)
        sub_t = t_map[:, a0:a1]
        sub_hard = hard[:, a0:a1]
        if sub_hard.sum() < 20:
            out.append(None)
            continue
        rr = np.nonzero(sub_hard.sum(axis=1) > 0)[0]
        cc = np.nonzero(sub_hard.sum(axis=0) > 0)[0]
        if len(rr) == 0 or len(cc) == 0 or rr[-1] - rr[0] < 5:
            out.append(None)
            continue
        y0, y1 = max(0, rr[0] - 2), min(H, rr[-1] + 3)
        x0, x1_ = max(0, cc[0] - 2), min(a1 - a0, cc[-1] + 3)
        crop_t = sub_t[y0:y1, x0:x1_]
        if black_d[y0:y1, a0 + x0:a0 + x1_].sum() > 0.35 * crop_t.size:
            out.append(None)          # 被干扰线破坏过多，丢弃
            continue
        tr = np.nonzero((crop_t > 0.6).sum(axis=1) > 0)[0]
        tc = np.nonzero((crop_t > 0.6).sum(axis=0) > 0)[0]
        if len(tr) == 0 or len(tc) == 0:
            out.append(None)
            continue
        # 供定位使用的 bbox 必须与 tmp/synth_geom.py 统计字高/基线所用的
        # redness>40 口径一致，否则会系统性错位（实测使整图准确率塌到 39%）
        hard_crop = hard[y0:y1, a0 + x0:a0 + x1_]
        hr = np.nonzero(hard_crop.sum(axis=1) > 0)[0]
        hc = np.nonzero(hard_crop.sum(axis=0) > 0)[0]
        if len(hr) == 0 or len(hc) == 0:
            out.append(None)
            continue
        out.append({"t": crop_t.astype(np.float16),
                    "h": int(crop_t.shape[0]), "w": int(crop_t.shape[1]),
                    "ink_h": int(hr[-1] - hr[0] + 1),
                    "ink_w": int(hc[-1] - hc[0] + 1),
                    "ink_bot": int(hr[-1]),
                    "ink_left": int(hc[0]), "ink_right": int(hc[-1]),
                    "ax": int(a0 + x0), "ay": int(y0)})
    return out


def _watershed_assign(hard, segs, redness, n=None):
    """预留：像素级归属（当前未启用，见 extract_glyphs 注释）。"""
    return None


def build_pool(limit=None, out=POOL_PATH):
    rows = [r for r in csv.reader(open(os.path.join(ROOT, "data", "label.csv"),
                                       encoding="utf-8")) if len(r) >= 2]
    if limit:
        rows = rows[:limit]
    pool = {c: [] for c in CHARSET}
    n_ok = 0
    for i, (fn, lab) in enumerate(rows):
        if len(lab) != 4 or any(c not in CHARSET for c in lab):
            continue
        p = os.path.join(ROOT, "data", "IMAGES",
                         fn if os.path.splitext(fn)[1] else fn + ".jpg")
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gs = extract_glyphs(bgr)
        if gs is None:
            continue
        for s, gd in enumerate(gs):
            if gd is not None:
                gd = dict(gd)
                gd["src"] = i              # 来源图在 label.csv 中的下标（用于划分池）
                gd["src_slot"] = s
                pool[lab[s]].append(gd)
        n_ok += 1
        if (i + 1) % 2000 == 0:
            print(f"  ...{i+1}/{len(rows)}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(pool, f, protocol=4)
    tot = sum(len(v) for v in pool.values())
    print(f"字形池: {tot} 个，覆盖 {n_ok} 张图")
    print("  " + "  ".join(f"{c}:{len(pool[c])}" for c in CHARSET))
    return pool


class ComposeGenerator:
    """从字形池重组生成新验证码。"""

    def __init__(self, pool, pool_idx=None, rot_deg=1.0, cx_jitter=4.1,
                 y_jitter=1.0, h_scale=0.10, blur_sigma=0.0, line_width=2.0,
                 jpeg_quality=88, green_frame=3, rng=None,
                 replay=False, use_lines=True, use_jpeg=True, use_gradient=True,
                 use_layout=True, use_transform=True,
                 block_dx=1.2, block_dy=0.6):
        self.pool = pool
        self.pool_idx = pool_idx            # {class: [可用下标]}，用于划分 train/eval 池
        self.rot_deg = rot_deg
        self.cx_jitter = cx_jitter
        self.y_jitter = y_jitter
        self.h_scale = h_scale
        self.blur_sigma = blur_sigma
        self.line_width = line_width
        self.jpeg_quality = jpeg_quality
        self.green_frame = green_frame
        # 消融开关
        self.replay = replay                # 字形放回源图位置、不做任何变换
        self.use_lines = use_lines
        self.use_jpeg = use_jpeg
        self.use_gradient = use_gradient
        self.use_layout = use_layout        # 是否重新采样位置
        self.use_transform = use_transform  # 是否重新采样尺度/旋转
        self.block_dx = block_dx            # 块级整体水平平移 σ
        self.block_dy = block_dy            # 块级整体垂直平移 σ
        self.rng = rng if rng is not None else np.random.default_rng()

    def _background(self):
        if not self.use_gradient:
            return np.full((H, W, 3), 220.0, np.float32)
        ramp = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
        bg = (np.array([198, 200, 197], np.float32)[None, None, :] * (1 - ramp)
              + np.array([245, 245, 245], np.float32)[None, None, :] * ramp)
        bg = np.repeat(bg, H, axis=0) * self.rng.normal(1.0, 0.012)
        bg = np.clip(bg, 0, 255)
        g = self.green_frame
        if g:
            bg[:g, :, 1] *= 0.72; bg[:g, :, 2] *= 0.70; bg[:g, :, 0] *= 0.62
            bg[-g:, :, 1] *= 0.86
            bg[:, :g, 1] *= 0.80
            bg[:, -g:, 1] *= 0.98
        return bg

    def _draw_lines(self, canvas):
        n = int(self.rng.integers(1, 3))
        for _ in range(n):
            if self.rng.random() < 0.72:
                x0, x1 = float(self.rng.uniform(-8, 12)), float(self.rng.uniform(W - 12, W + 8))
            else:
                x0 = float(self.rng.uniform(-5, W * 0.55))
                x1 = x0 + float(self.rng.uniform(45, 110))
            y0 = float(self.rng.normal(29.7, 11.0))
            y1 = float(self.rng.normal(29.7, 11.0))
            yc = (y0 + y1) / 2 + float(self.rng.normal(0, 11.0))
            pts = [( (1-t)**2*x0 + 2*(1-t)*t*((x0+x1)/2) + t**2*x1,
                     (1-t)**2*y0 + 2*(1-t)*t*yc + t**2*y1 )
                   for t in np.linspace(0, 1, 80)]
            cv2.polylines(canvas, [np.round(np.array(pts, np.float32)).astype(np.int32)],
                          False, (10, 10, 10),
                          thickness=max(1, int(round(self.line_width))),
                          lineType=cv2.LINE_AA)

    def generate(self, label=None, jpeg=True, glyphs_override=None):
        if label is None:
            label = "".join(self.rng.choice(list(CHARSET), 4))
        canvas = self._background().astype(np.float32)
        # 块级刚体平移：模型对水平平移极其敏感（实测 dx=4px 掉 13 个点），
        # 因此需要单独的块级项 + 很小的逐槽残余，不能各槽独立抖动。
        dx_img = float(self.rng.normal(0, self.block_dx))
        dy_img = float(self.rng.normal(0, self.block_dy))
        for s, c in enumerate(label):
            if glyphs_override is not None:
                g = glyphs_override[s]
            else:
                idxs = self.pool_idx.get(c) if self.pool_idx else None
                if idxs is not None and len(idxs):
                    gi = int(self.rng.choice(idxs))
                    if gi >= len(self.pool[c]):
                        gi = int(self.rng.integers(len(self.pool[c])))
                else:
                    gi = int(self.rng.integers(len(self.pool[c])))
                g = self.pool[c][gi]
            g_t = g["t"].astype(np.float32)
            hm, hs, bot = _geom()[c]
            if self.use_transform:
                # 用「紧致墨迹 bbox」的高度做尺度标定：目标字高 ~ N(h_mean, h_std)
                target_h = float(np.clip(self.rng.normal(hm, hs), 12, 40))
                scl = (target_h / float(g["ink_h"])) * float(self.rng.normal(1.0, self.h_scale))
            else:
                scl = 1.0
            nw = max(2, int(round(g_t.shape[1] * scl)))
            nh = max(2, int(round(g_t.shape[0] * scl)))
            g_t = cv2.resize(g_t, (nw, nh), interpolation=cv2.INTER_AREA)
            angle = float(self.rng.normal(0, self.rot_deg)) if self.use_transform else 0.0
            if abs(angle) > 0.05:
                M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
                cos, sin = abs(M[0, 0]), abs(M[0, 1])
                mw, mh = int(nh * sin + nw * cos) + 2, int(nh * cos + nw * sin) + 2
                M[0, 2] += mw / 2 - nw / 2
                M[1, 2] += mh / 2 - nh / 2
                g_t = cv2.warpAffine(g_t, M, (mw, mh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            if self.blur_sigma > 0:
                g_t = cv2.GaussianBlur(g_t, (0, 0), self.blur_sigma)
            # 变换后重新求紧致墨迹 bbox，用它来定位（保证字高与基线不受缩放/旋转影响）
            tr = np.nonzero((g_t > 0.6).sum(axis=1) > 0)[0]
            tc = np.nonzero((g_t > 0.6).sum(axis=0) > 0)[0]
            if len(tr) == 0 or len(tc) == 0:
                continue
            ink_cx = (tc[0] + tc[-1] + 1) / 2.0
            if self.replay or not self.use_layout:
                x0, y0 = int(g["ax"]), int(g["ay"])
            else:
                cx = (SLOT_CENTERS[s] + dx_img
                      + float(self.rng.normal(0, self.cx_jitter)))
                y_bot = bot + dy_img + float(self.rng.normal(0, self.y_jitter))
                # ink_cx/底边都用「redness>40 口径」的 bbox（与 geom_table 对齐）
                hb = g["ink_bot"] * scl
                hcx = (g["ink_left"] + g["ink_right"] + 1) / 2.0 * scl
                x0 = int(round(cx - hcx))
                y0 = int(round(y_bot - hb))
            xs0, ys0 = max(x0, 0), max(y0, 0)
            xs1, ys1 = min(x0 + g_t.shape[1], W), min(y0 + g_t.shape[0], H)
            if xs1 <= xs0 or ys1 <= ys0:
                continue
            a = g_t[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0][:, :, None]
            canvas[ys0:ys1, xs0:xs1] = (canvas[ys0:ys1, xs0:xs1] * (1 - a)
                                        + INK_RGB[None, None, :] * a)

        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        if self.use_lines:
            self._draw_lines(canvas)
        if jpeg and self.use_jpeg:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                canvas = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR),
                                      cv2.COLOR_BGR2RGB)
        return canvas


def main():
    ap = argparse.ArgumentParser(description="真实字形重组合成器")
    ap.add_argument("--build", action="store_true", help="重建字形池")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("-o", "--output", default="tmp/cv_out/synth_compose.png")
    ap.add_argument("--rot", type=float, default=4.0)
    ap.add_argument("--blur", type=float, default=0.0)
    a = ap.parse_args()

    if a.build or not os.path.isfile(POOL_PATH):
        pool = build_pool(a.limit)
    else:
        with open(POOL_PATH, "rb") as f:
            pool = pickle.load(f)
        print(f"载入字形池: {sum(len(v) for v in pool.values())} 个")

    gen = ComposeGenerator(pool, rot_deg=a.rot, blur_sigma=a.blur,
                           rng=np.random.default_rng(0))
    tiles = []
    for i in range(a.n):
        lab = "".join(np.random.default_rng(i).choice(list(CHARSET), 4))
        img = gen.generate(lab)
        big = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        tiles.append(np.hstack([big, np.full((big.shape[0], 4, 3), 255, np.uint8)]))
    per = 4
    sheet = np.vstack([np.hstack(tiles[i:i + per]) for i in range(0, len(tiles), per)])
    cv2.imwrite(a.output, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"saved {a.output}")


if __name__ == "__main__":
    main()
