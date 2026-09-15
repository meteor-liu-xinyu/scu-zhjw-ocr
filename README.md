# SCU 教务处验证码识别

SCU教务处（zhjw）验证码的 CNN 识别模型。

## 定稿版本（2026-09-15）

**`out/slot_joint.sep4.qat4.int4.scuocr` —— 18.9 KB / 单字符 99.95% / 整图 99.80%**

这是当前的生产模型，配方：`(20,32,48,48)` 骨干 + **空间可分离 conv4** + 逐槽头 + **int4 权重 / int8 偏置（QAT）**。
相对最初的 107 KB 原版体积 **−82%**，相对 41.7 KB 的上一版 **−55%**，精度一字不差。

推理侧要求：

1. 预处理用默认裁剪窗 `img[5:55, 40:140]`（与训练一致）；
2. 权重格式是 `.scuocr **version=3**` + slot 头；
   解析器必须**按张量名判断头类型**（有 `head.fc.weight` 即 slot），不能按版本号；
3. int4 权重是**打包**的（每字节 2 值、低 4 位在前、带 `(q+8)&0x0F` 偏置），
   反解时记得 `-8`；偏置和 `head.slot_bias` 是 int8；
4. 参考实现在 npm 包 `@scu-plus/zhjw-captcha-ocr` 的 `src/model.ts`（**尚需适配 v3/slot/可分离层**）。

回读验收：

```bash
python -u tmp/verify_scuocr.py out/slot_joint.sep4.qat4.int4.scuocr
```

备选：`out/sep4_wide_shift.scuocr`（35.0 KB / 99.50%）抗 ±8px 版式偏移，
但推理时必须用 wide 裁剪窗 `img[3:58, 32:150]`，且**不要用它的 int4 版**（98.30%，反而更差）。

## 验证码特征

- 尺寸：180×60 RGB
- 字符数：4 位
- 字符集：`2345678abcdefgmnpwxy`（20 类，去除了 0/1/9/o/l/q/t/u/v 等易混淆字符）
- 干扰：黑色短线 + 背景渐变

## 模型版本

### Pareto 前沿（2026-09-15，全部经**落盘回读**复测，1000 张固定测试集）

| 体积 | 整图 | 单字符 | 导出文件 | 配方 |
|---:|---:|---:|---|---|
| **16.1 KB** | **99.50%** | 99.88% | `out/slot_joint.sep34.qat4.int4.scuocr` | (20,32,48,48) + 空间可分离 conv3+4 + **int4 QAT** |
| **18.9 KB** | **99.80%** | 99.95% | `out/slot_joint.sep4.qat4.int4.scuocr` | (20,32,48,48) + 空间可分离 conv4 + **int4 QAT** ⭐ **定稿** |
| 35.0 KB | 99.50% | 99.88% | `out/sep4_wide_shift.scuocr` | 同上 + **wide 裁剪窗 + 平移增强**（抗版式偏移，见下）⭐ 鲁棒性优先 |

> **41.7 KB → 18.9 KB（−55%），精度一字不差（都是 99.80%）。**
>
> 三个关键进展：
>
> 1. **int4 QAT 把 int4 的损失完全补回来了**。同一模型 PTQ int4 只有 97.40%
>    （−2.40 点），QAT（60 轮）后是 **99.80%**，与 fp32/int8 版完全相同。
>    所以「可分离 + int4 不能叠加」这个此前的负面结论**已被 QAT 化解**。
> 2. **空间可分离 conv4 是免费的**：35.0KB vs 41.7KB 稠密版，
>    1000 张测试集上 **0 修复 / 0 破坏、预测完全相同**。
> 3. 被 int4-QAT 支配而淘汰的点：41.7KB/99.80%、36.1KB/99.70%、35.0KB/99.80%、
>    29.1KB/99.70%、25.1KB/99.50%、22.2KB/99.40%、19.3KB/99.40% 等（留作对照）。

### 抗版式偏移版本（wide 裁剪窗 + 平移增强）

实测「原图整体平移 Δ」时的整图准确率（**必须用与训练一致的裁剪窗评估**，
用错窗会让准确率虚低 1.5 点以上 —— `verify_scuocr.py --crop wide` 已支持）：

| Δ原图px | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 现状（current 窗，无平移增强） | 99.80 | 99.70 | 97.90 | 85.80 | 54.60 | 23.00 | 8.90 | 3.10 |
| **wide 窗 + 平移增强** | 99.50 | 99.50 | **99.80** | **98.80** | **96.80** | **78.60** | 45.90 | 17.60 |
| 差值 | −0.30 | −0.20 | **+1.90** | **+13.0** | **+42.2** | **+55.6** | +37.0 | +14.5 |

- **Δ=8px：54.60% → 96.80%（+42.2 点）；Δ=10px：23.00% → 78.60%（+55.6 点）。**
- 容差从「±4px」提升到「**±8px 几乎无损、±10px 仍可用**」。
- 代价只需同分布精度 **−0.30 点**（99.80 → 99.50），体积不变（35.0 KB）。
- 推理时**必须用 wide 裁剪窗**（`img[3:58, 32:150]`），与训练一致；否则退回 97.8%。
- 逐槽看更明显：Δ=12px 时现状模型槽 0 只有 33.80%，新版是 **85.00%**。



### 其它保留版本

| 版本 | 架构 | 导出文件 | 大小 | 整图 | 用途 |
|------|------|----------|:----:|:----:|----------|
| 联合微调 42KB | (20,32,48,48) + 逐槽头 | `zhjw-model.slot.joint.scuocr` | 41.7 KB | 99.80% | 所有可分离/量化实验的**起点** |
| 逐槽头 42KB | 同上（冻骨干） | `zhjw-model.slot.scuocr` | 41.7 KB | 99.40% | 历史对照 |
| 窄版 66KB | (20,32,48,48)/fc96 | `zhjw-model.scuocr` | 66 KB | 99.00% | 兼容保留（旧部署） |
| 原版 107KB | (24,40,64,64)/fc120 | `zhjw-model.int8.scuocr` | 107 KB | 99.60% | 回退选项 |

> **int4 行的两个关键点**（`tmp/quant_int4_plus.py`，回读验收，非内存估算）：
>
> 1. **偏置（含 `head.slot_bias`）必须保 int8，不能跟着权重降到 int4。**
>    代价只有约 200 字节（偏置总数约 276 个），但在窄配置上值 **+2.80 点**
>    （`(20,32,40,48)` int4：96.30% → 99.10%）。旧实现
>    （`train_qat.quantize_per_channel_int4`）把偏置一起降到 int4，白丢 2.8 点。
> 2. 所有 int4 产物都是 **version=3 + slot 头**（新格式组合），
>    npm 解析器必须**按张量名判断头类型**，不能按版本号判断。



> 2026-09-15 云上 GPU 结果（本机回读复测，同一 1000 张测试集）：
>
> - **T1 逐槽头 + 骨干联合微调**（`train.py --head slot --init-from best.narrow.pt`）：
>   41.7 KB / **99.80%**（单字符 99.95%），比冻结骨干版 **+0.40 点、体积不变**。
>   配对检验：**4 张修复、0 张破坏**（纯单向收益），McNemar p=0.125。
> - **T2 骨干瘦身**（`(20,32,40,48)`，从头训）：36.1 KB / **99.70%**，
>   比 42KB 版**小 13.4%**，配对检验 p=1.00 → **精度差为纯噪声**。
>   故 36KB 版是当前 Pareto 最优点。
> - 逐槽头版比窄版小 37% 且精度高 0.4~0.8 个点：把 26 KB 的跨槽分类头
>   （`fc1(192→96)+output(96→80)`）换成 1 KB 的逐槽共享头
>   （`head.fc(48→20) + 每槽偏置`）。详见下方「分类头（`--head`）」一节。
> - 上述三者（含 36KB 版）均为 **int8 `.scuocr` version=4**，
>   浏览器 npm 包需支持 version=4 才能真正上线。


### 体积账本（压缩前先看这里）

窄版 66.4KB 的构成（`tmp/model_size_audit.py` 可复现）：

| 层 | 参数量 | 占比 |
|----|-------:|-----:|
| conv4.weight 48→48 3×3 | 20,736 | 30.8% |
| fc1.weight 192→96 | 18,432 | 27.4% |
| conv3.weight 32→48 3×3 | 13,824 | 20.5% |
| output_layer 96→80 | 7,680 | 11.4% |
| conv2.weight | 5,760 | 8.5% |
| SE（4 个张量） | 480 | 0.7% |
| conv1 + 各 bias | ≈460 | 0.7% |

换成逐槽头后（42.1KB），**骨干占 97.5%**，conv4 单独就占 49.2%：
下一步压缩只能动骨干（conv4 的空间可分离 1×3+3×1 分解可省 6,912 参数 ≈6.6KB）。

> 结论：**分类头原本占了 39%，是纯浪费**；现在只剩 2.5%。
> 剩余体积由 conv3/conv4 主导，且已无「无损」手段可省——
> 66.4KB 的数据部分整体熵为 6.79 bit/权重，理论下限 57.2KB，gzip 实测 57.6KB 已到极限。

## 模型架构

4 层 CNN + SE 注意力 + AdaptiveAvgPool 保留水平位置信息：

```
Input: 1×32×64 (灰度, [0,1] 归一化)
  → Conv3×3(1→24)  + BN + ReLU + MaxPool  →  24×16×32
  → Conv3×3(24→40) + BN + ReLU + MaxPool  →  40×8×16
  → Conv3×3(40→64) + BN + ReLU + MaxPool  →  64×4×8
  → Conv3×3(64→64) + BN + ReLU + MaxPool  →  64×2×4
  → SE 注意力（通道重标定，+580 参数）
  → AdaptiveAvgPool(1,4) → 保留4个字符位置
  → Flatten → 256
  → Linear(256→120) + ReLU + Dropout(0.3)
  → Linear(120→80)   ← 4位 × 20类
```

宽度可通过 `--widths w1,w2,w3,w4` / `--fc-width` 调整（窄版为 `20,32,48,48` + fc96）。

### 分类头（`--head`，2026-09 新增）

`--head slot` 用逐槽共享头替换原来的跨槽头：

```
AdaptiveAvgPool(1,4) → (B, C, 1, 4) → (B, 4, C)
  → 共享 Linear(C→20) + 每槽独立偏置 (4,20)   ← 1,060 参数（C=48）
```

| head | 结构 | 参数量 | int8 体积 | 整图准确率 | 总文件 |
|------|------|-------:|----------:|:----------:|-------:|
| `fc`（原） | Linear(192→96)+ReLU+Dropout+Linear(96→80) | 26,288 | 25.7 KB | 99.00% | 66.4 KB |
| **`slot`** | 共享 Linear(48→20) + 槽偏置 | **1,060** | **1.0 KB** | **99.40%** | **41.7 KB** |

> 跨槽混合（`fc`）是当前模型里最大的一笔浪费，且**精度更差**：
> 四个字符位互相独立，`fc1` 把 4 个槽拼成 192 维再混合，等于往头部注入噪声。
> 用冻结骨干特征做的对照实验（同一 1000 张测试集）：
> 跨槽 Linear(192→80) 98.90% / 共享 Linear(48→20) 99.20% /
> 分槽 Linear(48→20)+槽偏置 99.40% / 原 fc1+output 99.00%。
>
> 特征空间 kNN(k=5) 上界为 99.50% —— 该骨干的可达上限约在此，
> 想再往上需要换/加宽骨干而非改头。

## 性能

| 指标 | 原版 | 窄版 | **逐槽头版** |
|------|:----:|:----:|:-----------:|
| 测试集整图准确率 | **99.60%** | 99.00% | **99.40%** |
| 测试集单字符准确率 | **99.90%** | 99.75% | **99.85%** |
| 模型总参数量 | 110,244 | 67,372 | **42,144** |
| 导出 .scuocr 大小（int8） | 107 KB | 66 KB | **42 KB** |
| int8 量化 | 无损 | 无损 | **无损（99.40% → 99.40%）** |
| 单张推理速度（CPU） | 0.680 ms | 0.627 ms | **≈0.60 ms**（头变小，主要省 FLOPs 于 FC） |

> 注：模型训练于公开数据集（10,000 张，80% 训练 / 10% 验证 / 10% 测试）。
> 逐槽头版的精度由 `/tmp/verify_scuocr.py` **从导出文件回读**测得（不依赖内存中的模型），
> 测试集为固定划分的最后 1000 张（`seed=42`）。
> int8 量化（BN 折叠 + 对称量化）实测准确率无损。
> 速度数据来自 `tmp/bench_speed.py`（CPU 单张延迟，批量下窄版优势更大）。
>
> ⚠️ **精度差异的显著性**：测试集仅 1000 张，1 张 = 0.1%，标准误约 ±0.31%。
> 99.00% 与 99.40% 相差 4 张图，方向一致但不宜当作硬结论；
> 定标建议用 val+test 共 2000 张，或全量 10000 张交叉验证。
> 同理，评估时必须使用与训练一致的划分
> （`random_split(range(10000), [8000,1000,1000], generator=torch.Generator().manual_seed(42))`），
> 用 `np.random.RandomState(42).permutation` 得到的划分**不等价**，会污染测试集
> （实测把 99.00% 虚高到 99.68%）。

### 已探索/已排除的方案（避免重复劳动）

- ❌ **剪枝**：结构性通道剪枝直接崩（<1%）；非结构化剪枝不减少文件大小
- ❌ **深度可分离卷积**：参数量 -55%，精度保不住 99%+
- ❌ **知识蒸馏**：4 种配置全部大幅劣化（~50%），小数据集软标签噪声大
- ❌ **EMA 权重平均**：实测反而 -1.8%（97.5% vs 99.3%）
- ❌ **混合精度 int4（QAT per-channel，fc 头）**：33KB / 97.20%（原版 99.20%），掉点超 2%
  ⚠️ **此结论成立于 `fc` 头 / 原版架构，不适用于 `slot` 头**，见下方「int4 与分类头」。
- ❌ **强增强 / SGDR / 大 batch**：均不如原版弱增强 + 单程 cosine
- ❌ **熵编码 / 无损压缩**：**没有空间**。66KB 里数据 67,372 B 的整体经验熵为
  **6.79 bit/权重 → 理论下限 57,189 B**，而 gzip -9 实测 **57,635 B**、lzma -9e 57,692 B
  —— 通用压缩已经打到熵下限。且权重是独立二进制资产，npm tarball 本身走 gzip，
  实际下载体积已经是 ~58KB。**体积只能在「参数量」和「位宽」上省。**
- ❌ **PTQ int4（零训练，fc 头）**：全量化 int4 per-channel 仅 95.20%；
  只把 conv3/conv4/fc1/output 降到 int4、其余保 int8 是 36.2KB / 98.20%。
  用分位数裁剪代替 `max|w|` 只值 +0.6 点。
  ⚠️ 同上：这是 **107KB 原版（fc 头）** 的数字。**`slot` 头 PTQ int4 实测只掉 0.50 点。**

### ★ int4 与分类头：slot 头远比 fc 头抗 int4（2026-09-15 实测）

`tmp/probe_int4.py` 复现完整部署链路（fold BN → per-channel int4 → 反量化 → 评估），
同一 1000 张测试集：

| 模型 | 头 | 参数量 | int8 | **int4** | fp32 整图 | **int4 整图** | 掉点 |
|---|---|--:|--:|--:|--:|--:|--:|
| `slot_joint/best.pt` (20,32,48,48) | slot | 42,592 | 41.6K | **21.5K** | 99.80 | **99.30** | **−0.50** |
| `w_20_32_40_48/best.pt` | slot | 36,800 | 35.9K | 18.6K | 99.70 | 96.30 | −3.40 |
| `zhjw-model.int8.scuocr`（原版） | fc | 110,052 | 107.5K | 55.6K | 99.60 | 96.20 | −3.40 |

**已落盘验收**（不是内存估算）：`out/slot_joint.int4.scuocr` →
`version=3 / head=slot / 22.0 KB / 单字符 99.83% / 整图 99.30%`。

- **41.7 KB → 22.0 KB（−47%），只掉 0.50 点**，且这还是 PTQ（没做 QAT）。
- 规律：**逐槽头对 int4 的鲁棒性远好于跨槽 fc 头**（−0.50 vs −3.40）。
  推断原因是 fc 头的 `fc1(192→96)+output(96→80)` 是跨槽混合全连接，
  权重分布跨度大、量化误差会跨槽串扰；逐槽头只有 1KB 且逐槽独立。
- 但**并非越窄越抗 int4**：`(20,32,40,48)` 反而掉 3.40 点。
  通道越少、冗余越低，per-channel 尺度越不稳 → **int4 与宽度的交互需实测**，
  不能假定"窄配置 + int4"总是可叠加。

### 压缩预算（`tmp/compress_budget.py`，参数已与真实模型对拍）

三个杠杆：**widths**（通道数）× **卷积分解**（空间可分离）× **位宽**（int8/int4）。
每格 = `int8 / int4`，基准 41.7 KB / 99.80%：

| widths | 全标准 3×3 | 空间可分离 conv4 | 空间可分离 conv3+4 |
|---|---:|---:|---:|
| 20,32,48,48 | 41.5K / 20.8K | 34.7K / 17.5K | 28.7K / 14.5K |
| 20,32,40,48 | 35.9K / 18.0K | 29.3K / 14.7K | **24.8K / 12.5K** |
| 20,32,40,40 | 32.8K / 16.5K | 28.1K / 14.2K | 23.6K / 11.9K |
| 16,28,40,40 | 29.7K / 14.9K | 25.0K / 12.6K | 20.7K / 10.4K |
| 16,24,32,32 | 20.7K / 10.4K | 17.7K / 8.9K | 14.9K / **7.5K** |
| 12,20,28,28 | 15.3K / 7.7K | 13.0K / 6.6K | 10.9K / **5.5K** |

- **空间可分离卷积（1×3 → 3×1）是 3×3 核的低秩近似**，与已排除的「深度可分离」
  （在**通道**上分解）是两回事，**尚未验证** → 需训一次确认是否成立。
- 上表是**参数预算、不含精度**。已知唯一实测点：`(20,32,48,48)` + int4 = 22.0KB / 99.30%。
- ❌ **稀疏化**：非零权重占 98.4%，`|w|≤2` 仅 6.7%，无天然稀疏红利
- ❌ **朴素早停（`--early-stop`）**：`train.py`/`train_qat.py` 有该开关但**默认关闭且不应开启**。
  调度是单程 cosine（`T_max = epochs − warmup`），高 LR 阶段验证集本就高位震荡，
  早停会在退火尾段之前触发、砍掉真正涨点的阶段。GPU 上 500 轮仅 10~20 分钟，
  **跑满即可**。若确要早停，须做成 LR 保护式（仅 `lr < lr_min × k` 后开始计数）。
- ✅ **窄架构 + fc 加宽**：`(20,32,48,48)/fc96` 减 38% 体积仅掉 0.6%
- ✅ **逐槽共享头**：`--head slot` 把 25.7KB 的跨槽头换成 1.0KB 的逐槽头，
  总体积 66.4KB → **41.7KB（-37%）**，整图 99.00% → **99.40%**
- ✅ **逐槽头 + 骨干联合微调**（云 GPU，`--head slot --init-from best.narrow.pt`）：
  同体积 41.7KB 下 99.40% → **99.80%**（4 张修复 / 0 张破坏）
- ✅ **骨干瘦身到 (20,32,40,48)**（云 GPU，从头训）：
  **36.1KB / 99.70%**，比 41.7KB 版小 13.4%，精度差经 McNemar 检验为纯噪声（p=1.00）
  —— **当前 Pareto 最优点**


### 分类头的对照实验（冻结骨干特征，同一 1000 张测试集）

| 分类头 | 整图 | 单字符 | 参数量 | int8 体积 |
|--------|:----:|:------:|-------:|----------:|
| 原 `fc1(192→96)+output(96→80)` | 99.00% | 99.75% | 26,288 | 25.7 KB |
| **共享 `Linear(48→20)` + 槽偏置（`--head slot`）** | **99.40%** | **99.85%** | **1,060** | **1.0 KB** |
| 共享 `Linear(48→20)`（无槽偏置） | 99.20% | 99.80% | 980 | 1.0 KB |
| 共享 `MLP(48→64→20)` | 99.30% | 99.83% | 4,436 | 4.3 KB |
| 跨槽 `Linear(192→80)` | 98.90% | 99.70% | 15,440 | 15.1 KB |
| 原型表（每槽每类 1 个质心，非训练） | 99.10% | 99.78% | 3,840 | 3.8 KB |
| 原型表（共享，每类 4 个，非训练） | 99.20% | 99.80% | 3,840 | 3.8 KB |
| — 特征空间 kNN(k=5) 上界（参考） | 99.50% | 99.88% | — | — |

> 规律：**只要逐槽独立分类就比跨槽混合好**，且参数少一到两个数量级。
> 这与「字符位之间无依赖」的先验一致。
> 原型表版本说明分类头甚至不需要训练——CNN 骨干负责的是对齐/切分，
> 一旦特征对齐，线性映射或查表就足够（这也解释了纯传统 CV 为何只有 47%：
> 它死在切分，不是死在分类）。


## 预处理流水线

```
原图 180×60 BGR
  → 裁剪中间区域 img[5:55, 40:140] → 100×50（列 40..139、行 5..54）
  → 黑线检测 (RGB < 130) → 填固定背景色 (225,222,222)
  → 灰度化: gray = 0.299R + 0.587G + 0.114B
  → 反色: result = 1 - gray/255
  → 缩放到 64×32 (INTER_AREA)
```

训练时附加数据增强：几何变换（平移 ±1px、缩放 ±5%、旋转 ±2°）+ 弹性形变 + 亮度/对比度扰动 + 笔画粗细扰动 + 高斯噪声 + 随机局部遮挡。

### ⚠ 裁剪窗口会切掉墨迹（已量化，待修）

用 redness = `R − max(G,B) > 40` 统计全量 10000 张的墨迹范围：

| 方向 | 现行窗口 | 切到墨迹 | 溢出深度 | 建议窗口 | 修正后 |
|---|---|---:|---|---|---:|
| 横向 | x 40..139 | **167 张 (1.67%)** | 左 max 8px / 右 max 11px（p99 均为 1px） | x 32..149 | 1 张 (0.01%) |
| 纵向 | y 5..54 | **1537 张 (15.37%)** | 底部 max 3px（p90=2, p99=3） | y 3..57 | 1.20% |

- 墨迹实际包络：**x ∈ [32, 150]，y ∈ [10, 57]**（顶部从不越界，底部 `g/p/y` 下探到 57）。
- 现行窗口是历史假设「descender 到 y≈52」的产物，实测底部到 57，**1/8 的图被削掉 1~3px**。
- **两个残留错误（99.80% 模型唯一的 2 张错图）100% 都是横向切边样本**
  （`2000.jpg` 墨迹 x=[34,142]、`3417.jpg` x=[35,148]，左右都被切）；
  而基础发生率仅 1.67% → 切边把错误率放大了约 7 倍。
- 两者被**两种不同架构**（42KB 逐槽头、107KB 跨槽 fc）以**完全相同的方式**判错，
  说明信息确实不在输入里，不是模型容量或标签噪声问题。
- 修法（**不需要额外算力，但需重训**，GPU 上约 10~20 分钟）：
  1. 把 `_CROP_X1/_CROP_X2` 放宽到 32/150（保持 `_CROP_Y1/_CROP_Y2` 或一并放宽到 3/57）；
  2. 更根本的做法——**随机裁剪偏移增强**（`--aug-profile shift`），
     同时可缓解下面的「平移不变性缺失」问题；
  3. 若保持 2:1 宽高比，可直接用 `img[0:60, 30:150]`（120×60，精确 2:1，切到 1/10000）。

### ⚠ 模型几乎没有平移不变性（已知上线风险）

对整图做刚性水平平移（`dx` 单位 px）：

| dx | 0 | 2 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| 整图 | 92.50 | 87.75 | 79.50 | 63.00 | 36.75 | 15.00 |

**dx=4px 掉 13 点、dx=8px 掉 56 点** —— 这是位置记忆式模型。
含义：教务处若改版导致文字位置偏移几像素，线上精度会断崖下跌。
根因是增强里平移只有 ±1px（见 `_augment` 的 `dx/dy = uniform(-1,1)`），
而旋转/缩放/弹性形变都不改变字符的绝对位置中心。
**加固方式：加入 ±4~6px 的随机平移/裁剪偏移增强并重训。**


## 相关仓库

- **部署包**（浏览器端 CNN 推理 npm 包，内置 int8 权重）：[@scu-plus/zhjw-captcha-ocr](https://github.com/meteor-liu-xinyu/zhjw-captcha-ocr)

## 项目结构

```
zhjw-ocr/
├── model.py             # CaptchaCNN + SlotHead + SpatialSeparableConv / SEPARABLE_PLANS
├── preprocess.py        # 预处理 + ZhjwCaptchaDataset；CROP_PRESETS(current/wide) + 4 档增强
├── train.py             # 训练（--head/--separable/--crop/--crop-jitter/--aug-profile/--init-from/--resume）
├── train_slot_head.py   # 逐槽头快速训练（冻结骨干，CPU 约 2 分钟）
├── train_qat.py         # int4/int8 QAT（--head slot / --separable / --crop / --eval-quant / --export）
├── quantize.py          # BN 折叠 + FoldedCaptchaCNN（支持标准/空间可分离/深度可分离）
├── quantize_mixed.py    # 混合精度量化（int8+int4，per-channel）
├── export.py            # 导出 .scuocr（int8→v2/v4，混合精度→v3；conv_out_channels/detect_separable）
├── requirements.txt
│
├── GPU_RUN_ALL.sh       # ★ 一次性跑完所有 GPU 任务（g0 自检 → g1 QAT → g2 重训 → g3 扫描 → g4 补齐 → g5）
├── PACK_RESULTS.sh      # 打包 GPU 运行结果拷回本地（--list 先看清单）
├── RUNBOOK_COMMANDS.sh  # 本机/云端命令清单（--check/--t1/--t2/--t3/--sep/--int4plus/--pareto/--shift）
├── CLOUD_TRAINING.md    # 上云 runbook（已完成清单 + GPU 队列 + 已知坑，**建议先读**）
├── CHANGELOG.md         # 上云包变更清单
│
├── out/                 # 部署产物（.scuocr）
│   ├── slot_joint.sep4.qat4.int4.scuocr  # ★ 定稿：18.9 KB / 99.80%
│   ├── slot_joint.sep34.qat4.int4.scuocr # 16.1 KB / 99.50%
│   ├── sep4_wide_shift.scuocr            # 35.0 KB / 99.50%，抗 ±8px 偏移（需 wide 裁剪窗）
│   ├── slot_joint.sep4.scuocr            # 35.0 KB / 99.80%（int8，int4 QAT 的 fp32 源）
│   └── zhjw-model.*.scuocr               # 旧版基准（66/107 KB，兼容保留）
├── zhjw-model.slot.joint.scuocr          # 41.7 KB / 99.80%，所有实验的起点（int8）
│
├── checkpoints/         # 训练权重（**不入库**，可重训再生；交付的 .scuocr 已入库）
│   ├── slot_joint/best.pt          # 41.7KB 模型的 fp32 源（所有实验起点）
│   ├── slot_joint/best.sep4.pt     # 空间可分离 conv4（与原模型预测完全一致）
│   ├── qat_sep4/best.pt            # int4 QAT 后的 fp32 权重（定稿模型的来源）
│   ├── sep4_wide_shift/best.pt     # wide 窗 + 平移增强重训
│   └── …                           # 其余实验权重（共 17 个 best*.pt）
├── data/                # 训练数据（IMAGES 不入库，label.csv 入库）
├── gpu_logs/            # GPU 运行日志 + results.tsv + SUMMARY.txt
├── tmp/                 # 工具脚本（**已入库**）+ cv_out 中间产物（大文件忽略）
│   ├── verify_scuocr.py        # 回读验收（支持 int4 打包 / 可分离层 / --crop）
│   ├── quant_int4_plus.py      # 增强 int4：int8 偏置 + 裁剪搜索 + AdaRound + 按层位宽
│   ├── apply_separable.py      # 空间可分离替换（逐层拟合，不需要 GPU）
│   ├── sep_feasibility.py      # 可分离可行性判定
│   ├── probe_int4.py           # 零 GPU 估 int4 体积/精度
│   ├── compress_budget.py      # widths × 分解 × 位宽 预算表
│   ├── compare_models.py       # 配对对比 + McNemar
│   ├── shift_curve_orig.py     # 平移敏感性曲线（支持 --crop）
│   ├── collect_results.py      # 批量汇总成 Pareto 表
│   └── make_upload_package.py  # 打上云包（含 LF 强制校验）
├── notes/               # 项目记忆（含已排除方案与踩坑记录）
└── archive/             # 归档（gitignore）：上传包、参考项目 CaptchaOcrLite
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

从 [SCU_OAA-website-Captcha-training-set](https://github.com/SunnyHaze/SCU_OAA-website-Captcha-training-set) 下载 `IMAGES.zip` 和 `label.csv` 到 `data/` 目录。

### 训练

```bash
# 默认训练（200 epochs，原版弱增强，当前最优配置）
python train.py

# 窄版（66KB 模型，500 轮收敛到 99%）
python train.py --widths 20,32,48,48 --fc-width 96 \
  --epochs 500 --batch-size 64 --lr 3e-4 --weight-decay 5e-5

# ★ 逐槽头版（41.7KB / 99.40%）——推荐路径，CPU 上约 2 分钟
#   冻结 best.narrow.pt 的骨干，只训练约 1KB 的分类头
python train_slot_head.py --init-from checkpoints/best.narrow.pt --export

# 逐槽头 + 骨干联合微调（需 GPU 才现实；CPU 约 1.8 min/轮）
python train.py --head slot --init-from checkpoints/best.narrow.pt \
  --widths 20,32,48,48 --epochs 150 --batch-size 128 \
  --lr 3e-4 --weight-decay 5e-5 --ckpt-dir checkpoints/slot_joint

# 断点续训
python train.py --resume checkpoints/latest.pt

# 仅测试（使用 model 权重，自动识别 head 类型）
python train.py --test-only checkpoints/best.pt --head slot

# QAT int4 微调（窄版导出 33KB）
python train_qat.py --resume checkpoints/best.narrow.pt --widths 20,32,48,48 --fc-width 96 \
  --epochs 40 --batch-size 128 --lr 2e-4 --eval-quant --export
```

### 导出

```bash
# ★ 逐槽头 int8 导出（41.7KB，version=4）
python train_slot_head.py --init-from checkpoints/best.narrow.pt \
  --output checkpoints/best.slot.pt --export --export-path zhjw-model.slot.scuocr
# 或从已有 checkpoint 直接导出（头类型自动识别）
python export.py checkpoints/best.slot.pt --int8 -o zhjw-model.slot.scuocr

# 窄版 int8 导出（66KB，version=2）
python export.py checkpoints/best.narrow.pt --int8 -o zhjw-model.scuocr

# 原版 int8 导出（107KB）
python export.py checkpoints/best.pt --int8 -o zhjw-model.int8.scuocr

# fp32 导出（version=1）
python export.py checkpoints/best.narrow.pt -o zhjw-model.fp32.scuocr

# QAT int4 导出（version=3，per-channel，33KB 备选）
python export.py --mixed checkpoints/qat/best.qat-int4.pt -o zhjw-model.qat.scuocr
```

### 验收：从导出文件回读复测

导出链路的最终验收**不信任内存中的模型，只信任落盘的文件**：

```bash
python tmp/verify_scuocr.py zhjw-model.scuocr zhjw-model.slot.scuocr
```

```
zhjw-model.scuocr
   version=2  head=fc  widths=(20, 32, 48, 48)  参数量=67,372
   文件 67,966 B (66.4 KB)
   单字符 99.75%   整图 99.00%
zhjw-model.slot.scuocr
   version=4  head=slot  widths=(20, 32, 48, 48)  参数量=42,144
   文件 42,704 B (41.7 KB)
   单字符 99.85%   整图 99.40%
```

## 权重格式（`.scuocr`）

```
[Header]
  magic:        8 bytes = "SCUOCRLT"
  version:      4 bytes uint32 LE
  tensor_count: 4 bytes uint32 LE

[Per Tensor] × tensor_count
  name_len:   4 bytes uint32 LE
  name:       N bytes UTF-8
  ndim:       4 bytes uint32 LE
  shape:      ndim × 4 bytes uint32 LE
  scale:      4 bytes float32 LE        # 对称量化 scale = max|w| / qmax，zero_point 恒 0
  zero_point: 4 bytes int32 LE
  data:       product(shape) × 每元素字节数
```

| version | 含义 | 分类头张量 | 备注 |
|:-------:|------|------------|------|
| 1 | fp32，未折叠 BN | `fc1.*` / `output_layer.*` | 仅调试用 |
| 2 | int8，BN 已折叠 | `fc1.weight` `fc1.bias` `output_layer.weight` `output_layer.bias` | 当前 npm 包解析的版本 |
| 3 | 混合精度 int8+int4 | 同上 | `quantize_mixed.py` 产出 |
| **4** | **int8，BN 已折叠，逐槽头** | **`head.fc.weight` (20,C) `head.fc.bias` (20) `head.slot_bias` (4,20)** | **逐槽头版；需更新 npm 解析器** |

> v4 的推理路径：骨干不变（conv1-4 + SE + AdaptiveAvgPool(1,4)），
> 之后 `(B,C,1,4) → (B,4,C) → head.fc → + head.slot_bias → (B,4,20) → argmax`。
> 张量写入顺序为 conv1..4(weight,bias) → se → head，解析器应按 `name` 查表，
> 不要依赖顺序（导出器会把 `fc1` 排在 `se` 之后，与 v2 旧文件顺序不同但语义一致）。
> 逐槽头的特征标准化（均值/方差）已在导出前**折叠进 `head.fc` 权重**，
> 因此文件里不需要额外存放统计量。