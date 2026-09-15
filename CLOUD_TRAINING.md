# 上云训练 Runbook

本机无 CUDA（18 核 CPU），`train.py` 全反向训练约 **1.8 min/轮**，
500 轮需 15 小时，不现实。以下任务请在 GPU 机器上执行。

---

## 1. 上传清单

**代码（必须）**

```
model.py               # CaptchaCNN + SlotHead（head_type="fc"|"slot"）
preprocess.py          # 预处理 + 数据集 + 数据增强
train.py               # 训练（--head / --init-from / --freeze-backbone）
train_slot_head.py     # 逐槽头快速训练（冻结骨干，CPU 约 2 分钟）
train_qat.py           # QAT（⚠ 尚不支持 slot 头，见第 5 节）
quantize.py            # int8 量化 + FoldedCaptchaCNN（支持两种头）
quantize_mixed.py      # 混合精度量化
export.py              # 导出 .scuocr（v1/v2/v3/v4）
requirements.txt
```

**数据**

```
data/label.csv
data/IMAGES.zip          # 或已解压的 data/IMAGES/
```

**起点权重**

```
checkpoints/best.narrow.pt   # 窄版 66KB 骨干（(20,32,48,48)/fc96）—— 所有任务的起点
checkpoints/best.slot.pt     # 逐槽头模型（骨干同上 + SlotHead）
```

**验收与诊断脚本（强烈建议一起上传）**

```
tmp/verify_scuocr.py         # 从导出文件回读复测（导出链路的唯一验收手段，已支持 int4 打包）
tmp/probe_int4.py            # ★ int4 成本探针：零 GPU 估出「量化后体积/精度」
tmp/compress_budget.py       # ★ 压缩预算：widths × 卷积分解 × 位宽 的 Pareto 前沿
tmp/collect_results.py       # 批量验收汇总（T2 多候选扫描用，自动打标推荐）
tmp/compare_models.py        # 配对对比两个模型 + McNemar 检验（判断差异是否真实）
tmp/err_zoom.py              # 把出错图放大拼图，人工判断是模型错还是标签错
tmp/width_budget.py          # 参数量/体积预算表（旧版，compress_budget.py 是其超集）
tmp/head_verify_int8.py      # int8 骨干 + 各分类头精度对照
```

**合成数据脚本（已降级，暂不上传）**

用户已决定不走合成数据路线（生成质量一般、剩余工作量大）。
`synth_gen.py` / `synth_gen_compose.py` 及 `tmp/synth_*.py` 保留在仓库作记录，但
**当前优先级下不需要上传**。相关结论仍保留在 README「合成数据」与
`tmp/cv_out/` 的消融结果里，日后若要做蒸馏/int4 极限压缩再捡起来。

> 澄清：**int4 QAT 不依赖合成数据** —— 它是在已收敛模型上做量化微调，
> 10000 张真实图 + 强增强足够。合成数据当初是为「从头训小模型 / 知识蒸馏」准备的。


---

## 2. 环境

```
pip install -r requirements.txt
# 关键依赖：torch(建议 CUDA 版), opencv-python, numpy, tensorboard
```

`train_slot_head.py` 额外需要 `Pillow`（字体参数化生成器用）。

---

## 3. 评估基准（**必须严格一致，否则数字不可比**）

划分固定为：

```python
random_split(range(10000), [8000, 1000, 1000],
             generator=torch.Generator().manual_seed(42))
# 训练 8000 / 验证 1000 / 测试 1000（测试集为 randperm 的第三段）
```

**不要**用 `np.random.RandomState(42).permutation` 划分：两者不等价，
实测会把 99.00% 虚高到 99.68%。

参考成绩（1000 张测试集，**均为本机回读复测**）：

| 模型 | 单字符 | 整图 |
|------|:------:|:----:|
| `zhjw-model.scuocr`（窄版 66.4KB，fc 头） | 99.75% | 99.00% |
| `zhjw-model.slot.scuocr`（逐槽头 41.7KB，冻骨干） | 99.85% | 99.40% |
| `zhjw-model.int8.scuocr`（原版 107KB） | 99.90% | 99.60% |
| `out/w_20_32_40_48.scuocr`（瘦身 36.1KB，T2 产物） | 99.92% | **99.70%** |
| `zhjw-model.slot.joint.scuocr`（联合微调 41.7KB，T1 产物） | 99.95% | **99.80%** ← 当前最高 |

> **统计显著性**：1000 张测试集上 1 张 = 0.1%，标准误约 ±0.31%。
> 小于 0.3 点的差异不构成结论；**必须用 `tmp/compare_models.py` 做配对检验**
> （看「单向修复 vs 单向破坏」+ McNemar p 值），只比总数会被噪声误导。
>
> 配对检验实测：
> - 冻骨干 99.40% → 联合微调 99.80%：**4 张修复 / 0 张破坏**，p=0.125（方向可信，样本量不足）
> - 41.7KB/99.80% vs 36.1KB/99.70%：2 修复 / 1 破坏，**p=1.00 → 差异纯属噪声**
>   ⇒ **36.1KB 是当前 Pareto 最优点**（省 13.4% 体积，精度打平）


---

## 4. 任务队列

### T1 ★ 逐槽头 + 骨干联合训练（最高优先，本机跑不动）

目前 `best.slot.pt` 是**冻结骨干只训头**的产物（99.40%）。
联合训练让骨干适应新头，预期还能提升。

```bash
python train.py --head slot \
  --init-from checkpoints/best.narrow.pt \
  --widths 20,32,48,48 \
  --epochs 500 --batch-size 128 --lr 3e-4 --lr-min 3e-5 \
  --weight-decay 5e-5 --label-smoothing 0.1 --warmup 5 --amp \
  --num-workers 8 --ckpt-dir checkpoints/slot_joint
```

导出与验收：

```bash
python export.py checkpoints/slot_joint/best.pt --int8 -o zhjw-model.slot.joint.scuocr
python tmp/verify_scuocr.py zhjw-model.slot.joint.scuocr
```

验收标准：**整图 ≥ 99.40% 且文件 ≤ 42.7KB** 才算优于现状。

> **✅ 已完成（2026-09-15，云 GPU，跑到 epoch 224/500 时中断下载）**
> 导出 `zhjw-model.slot.joint.scuocr` → **41.7KB / 单字符 99.95% / 整图 99.80%**。
> 同体积下 +0.40 点，配对检验 4 修复 / 0 破坏。
> ⚠ 只跑到 224/500 轮，**余弦尚未退完**（LR 还没降到 lr_min），
> 继续跑满预期还能再涨 —— 这也是**不要开早停**的直接理由（见第 6 节第 6 条）。

### T2 ★ 骨干瘦身（一次提交一批候选）

`--widths` 可直接用，无需改代码。预算已算准（`tmp/width_budget.py`）：

| widths | head | 存储参数 | int8 体积 | 相对现状 | 实测 |
|--------|:----:|--------:|----------:|---------:|------|
| (20,32,48,48) | slot | 42,144 | 41.7 KB | 基准 | 99.80%（T1 联合微调）|
| (20,32,40,48) | slot | 36,376 | 36.1 KB | **-13%** | **99.70% ✅ 已验收** |
| (20,32,40,40) | slot | 33,256 | 33.1 KB | **-21%** | 待跑 |
| (20,28,40,40) | slot | 31,092 | 30.9 KB | -26% | 待跑 |
| (16,28,40,40) | slot | 30,044 | 29.9 KB | -28% | 待跑 |
| (16,24,32,32) | slot | 20,864 | 21.0 KB | -50% | 待跑 |
| (12,20,28,28) | slot | 15,368 | 15.6 KB | -63% | 待跑 |

> **实测比预算更乐观**：`(20,32,40,48)` 只掉 0.10 点（且经检验为噪声），
> 而不是预期的明显掉点。所以后面几档（33KB / 31KB / 30KB）**值得继续跑**，
> 很可能存在比 36.1KB 更靠前的 Pareto 点。建议一次提交全部剩余 5 档。
>
> ⚠ 已完成的 `w_20_32_40_48` 是**从头训**（`init_from=None`）到 epoch 247/500，
> 同样**未跑满**。两个任务都建议补跑满 500 轮。

建议一次提交全部 6 个（GPU 上每个约十几分钟）：

```bash
for w in 20,32,40,48 20,32,40,40 20,28,40,40 16,28,40,40 16,24,32,32 12,20,28,28; do
  python train.py --head slot --widths $w \
    --epochs 500 --batch-size 128 --lr 3e-4 --weight-decay 5e-5 --amp \
    --ckpt-dir checkpoints/w_$(echo $w | tr ',' '_')
done
```

然后逐个导出 + 验收（`export.py` 会自动推断 widths/head）：

```bash
mkdir -p out
for w in 20_32_40_48 20_32_40_40 20_28_40_40 16_28_40_40 16_24_32_32 12_20_28_28; do
  [ -f "checkpoints/w_$w/best.pt" ] || continue
  python export.py "checkpoints/w_$w/best.pt" --int8 -o "out/w_$w.scuocr"
done
python -u tmp/collect_results.py "out/*.scuocr" zhjw-model.slot.joint.scuocr
```

### T3 ★ int4 QAT（**当前最大的体积杠杆**，代码已就绪）

**为什么它变成第一优先**：需求已明确为「体积优先，精度次要」。int4 把权重字节
直接减半，是唯一能把模型砍到 20KB 量级的手段。

**已实测（PTQ，未做 QAT）**：`tmp/probe_int4.py` 走完整部署链路测得

| 模型 | 头 | int8 | int4 | fp32 整图 | int4 整图 | 掉点 |
|---|---|--:|--:|--:|--:|--:|
| `slot_joint/best.pt` (20,32,48,48) | slot | 41.6K | **21.5K** | 99.80 | **99.30** | **−0.50** |
| `w_20_32_40_48/best.pt` | slot | 35.9K | 18.6K | 99.70 | 96.30 | −3.40 |
| `zhjw-model.int8.scuocr`（原版） | fc | 107.5K | 55.6K | 99.60 | 96.20 | −3.40 |

已落盘验收：`out/slot_joint.int4.scuocr` → `version=3 / head=slot / 22.0KB / 99.30%`。

> **关键规律：`slot` 头比 `fc` 头抗 int4 得多**（−0.50 vs −3.40）。
> 所以之前「int4 必掉 2 点以上」的结论只对 fc 头成立，**不要再用它否决 int4 路线**。
> 但 `(20,32,40,48)` 掉 3.40 点，说明**越窄不一定越抗 int4** —— 交互需实测。

命令（`train_qat.py` 已支持 `--head slot`）：

```bash
python train_qat.py --resume checkpoints/slot_joint/best.pt \
  --head slot --widths 20,32,48,48 \
  --epochs 40 --batch-size 128 --num-workers 8 \
  --eval-quant --export

# 导出后在 .scuocr 层面验收（train_qat.py 产出的是中间格式）
python export.py --mixed checkpoints/slot_joint/best.qat-int4.pt \
  -o zhjw-model.slot.int4.scuocr
python -u tmp/verify_scuocr.py zhjw-model.slot.int4.scuocr
```

验收线：**≤22KB 且整图 ≥99.30%**（对得起 41.7KB 版掉的 0.5 点）。
QAT 预期能把 99.30% 拉回 99.5%+。

**⚠ 一个格式层面待办**：int4 走的是 `.scuocr **version=3**`，而 version=3 原先只用于
fc 头。现在出现 **v3 + slot 头**的新组合 → npm 包 `src/model.ts` 的解析器必须能
「按张量名判断头类型」，而不能按版本号判断。详见第 6 节第 9 条。

### T4 ★ 空间可分离卷积 + 裁剪修正 + 平移增强（第二杠杆，需一次重训）

三者都是**改输入分布或改结构**，会作废旧权重，所以合并成一次训练做。
预算见 `tmp/compress_budget.py`（参数已与真实模型对拍），每格 = `int8 / int4`：

| widths | 全标准 3×3 | 空间可分离 conv4 | 空间可分离 conv3+4 |
|---|---:|---:|---:|
| 20,32,48,48 | 41.5K / 20.8K | 34.7K / 17.5K | 28.7K / 14.5K |
| 20,32,40,48 | 35.9K / 18.0K | 29.3K / 14.7K | **24.8K / 12.5K** |
| 16,24,32,32 | 20.7K / 10.4K | 17.7K / 8.9K | 14.9K / **7.5K** |
| 12,20,28,28 | 15.3K / 7.7K | 13.0K / 6.6K | 10.9K / **5.5K** |

1. **空间可分离卷积**：`conv3`/`conv4` 的 3×3 拆成 (1×3)→(3×1)，
   是 3×3 核的低秩近似。conv3+conv4 一起改可省 **32%** 参数。
   ⚠ **这是未验证的杠杆** —— README 只排除了「深度可分离」（在通道上分解），
   空间可分离没试过，可能同样保不住精度，需训一次才知道。
2. **裁剪窗口修正**：`_CROP_X1/_CROP_X2` 从 40/140 放宽到 **32/150**
   （墨迹包络实测 x∈[32,150]、y∈[10,57]；现行窗口横向切 1.67%、纵向切 **15.37%**）。
3. **±4~6px 随机平移增强**（现在只有 ±1px），修复「dx=4px 掉 13 点」的不变性缺失。

> 顺序建议：**先做 T3（int4）**，因为它已实测、收益确定（−47% 体积）；
> 再把 T4 的裁剪修正/平移增强叠上去；**空间可分离放在最后单独验证**，
> 因为它可能失败，别和确定有效的改动混在一起，否则无法归因。


---

## 5. 上云前需要改的代码

### 5.1 ~~`train_qat.py` 不支持 `slot` 头~~ → **已修复（2026-09-15）**

原阻塞：`QatCaptchaCNN` 硬编码了 `fc1` / `output_layer`。改动：

- 构造函数新增 `head_type` 参数，`slot` 时建 `self.head = SlotHead(w4, NUM_CLASSES, CAPTCHA_LEN)`
- `forward` 分支：`feat = x.squeeze(2).permute(0,2,1)` → `head(feat)` → `reshape(B,-1)`；
  `fake_quant` 作用于 `head.fc.weight`（per-tensor），`slot_bias` 不量化
- 命令行新增 `--head {fc,slot}`（**必须与 `--resume` 的 checkpoint 一致**）
- 末尾 `FoldedCaptchaCNN(...)` 改为用 `export.infer_arch(sd_q)` 从 state_dict 反推
  `widths / fc_width / head_type`，避免 args 与实际模型不一致

**已端到端验证**：`out/slot_joint.int4.scuocr`（v3 / head=slot / 22.0KB / 99.30%）
可被 `tmp/verify_scuocr.py` 正确回读，与内存估算一致。

### 5.2 `tmp/verify_scuocr.py` 的 int4 解析 bug（已修复）

v3 解析器原先对 int4 段按 `n` 字节读取，但 **int4 数据是打包的**
（每字节 2 值、带 `(q+8) & 0x0F` 偏置），实际长度是 `ceil(n/2)`。
按 `n` 读会导致流错位（表现为读张量名时 `UnicodeDecodeError`）。
已新增 `_read_quant()` 统一处理 bits=4/8 两个分支。

> 这个 bug 一直没暴露，是因为 int4 精度**始终是在内存里评估的**
> （quantize → dequantize → 建模型 → 测），从没走过「写文件再读回来」。
> **教训：格式层面的改动必须做落盘回读验收，内存估算会漏掉序列化错误。**
> 同理，npm 包 `src/model.ts` 的 int4 分支也需要单独验证。

### 5.3 已修但需注意

- `train.py --warmup 0` 曾使学习率卡在目标值的 1%（`SequentialLR(milestones=[0])`
  永不切换），已修；用云端旧版代码时要留意。
- `export.py` / `quantize.py` 的 `fold_bn_into_conv` 原先硬编码分类头键名，
  换 `slot` 头会 KeyError，已改为通用透传。

---

## 6. 已知坑

1. **评估划分**必须用 `torch.Generator().manual_seed(42)`（第 3 节）。
2. **导出后必须回读验收**：`python tmp/verify_scuocr.py <文件>`。
   只信任落盘文件，不要用内存中的模型代替验收。
3. **`python -u` 或输出重定向**：Python stdout 重定向到文件时会缓冲，
   不要靠日志判断进度（读 `latest.pt` 的 `epoch` 字段更可靠）。
4. **无损压缩已到极限**：66.4KB 的数据部分熵为 6.79 bit/权重（下限 57.2KB），
   gzip -9 实测 57.6KB。体积只能在「参数量」和「位宽」上省，别再试编码层压缩。
5. **模型的平移敏感性**（重要）：实测对整图做刚性水平平移，
   dx=2px 掉 4.75 点、dx=4px 掉 13 点、dx=8px 掉 56 点。
   即当前模型几乎**没有平移不变性**，是位置记忆式的。
   根因是增强里平移只有 `uniform(-1,1)` 即 ±1px。
   → 若教务处改版导致文字位置偏移几像素，精度会断崖式下跌。见 T5。
6. **不要开早停**（`--early-stop`）：默认 0=关闭，**保持关闭**。
   调度为单程 cosine（`T_max = epochs − warmup`），高 LR 阶段验证集本就高位震荡，
   朴素早停会在退火尾段之前触发、砍掉真正涨点的阶段。
   GPU 上 500 轮仅 10~20 分钟，跑满即可。
   实测佐证：T1/T2 两个任务都在 epoch 224/247（**不到一半**）时就被下载，
   余弦尚未退完，继续跑满预期还有提升空间。若确要早停，须做成
   LR 保护式（仅 `lr < lr_min × k` 之后才开始计数）。
7. **裁剪窗口会切掉墨迹**（新发现，已量化全量 10000 张）：
   现行 `img[5:55, 40:140]` 横向切到 1.67% 的图、纵向切到 **15.37%** 的图
   （墨迹包络实测 x∈[32,150]、y∈[10,57]；底部 `g/p/y` 下探到 57，超出 y=54）。
   证据强度：**T1 模型仅有的 2 张错图 100% 都是横向切边样本**
   （`2000.jpg` x=[34,142]、`3417.jpg` x=[35,148]），而基础发生率仅 1.67%
   —— 切边把错误率放大约 7 倍。且两种不同架构（42KB 逐槽头、107KB 跨槽 fc）
   **以完全相同的方式判错**，说明信息确实不在输入里，不是容量/标签问题。
   → 修正方案见 T5。注意：改裁剪窗口会使旧权重失效，必须重训。
8. **.scuocr v4 需同步 npm 解析器**：`@scu-plus/zhjw-captcha-ocr` 的
   `src/model.ts` 目前只认 v2；不更新则 36KB / 41.7KB 版（均为 v4）无法上线。
   **这是当前唯一的上线阻塞项，且不需要 GPU。**
9. **v3 + slot 头是新的格式组合**：int4 走 `.scuocr **version=3**`，
   而 v3 原先只用于 fc 头（v4 才是 int8-slot）。所以现在存在
   **v3 也可能是 slot 头**的情况 → npm 解析器必须**按张量名判断头类型**
   （有 `head.fc.weight` → slot），**不能按版本号判断**。
   建议在 JS 侧同时接受 v3/v4 + 按名字分派，或新增 `version=5`（mixed + slot）以示区分。
10. **int4 的两条路都别丢**：`train_qat.py --export` 写出的是**中间格式**
    `checkpoints/*/best.qat-int4.pt`，必须再经
    `python export.py --mixed <pt> -o xxx.scuocr` 才能得到可部署的 `.scuocr`。
    直接用 `export.py <pt> --int8` 会得到 int8 而不是 int4。

---

## 7. 本机已完成（2026-09-15，无需 GPU）与 GPU 待办队列

需求已明确为**体积优先、精度次要**。以下全部在本机 CPU 上完成并落盘回读验收。

### 7.1 已完成：Pareto 前沿（每点都经 `tmp/verify_scuocr.py` 回读）

**云上首轮结果已合并**（结果包 `tmp/zhjw-gpu-results-20260915-0801.zip`，
停机前跑完 g1 全部 3 个任务 + g2 两个训练 + g3 第 1 个训练）：

| 体积 | 整图 | 文件 | 配方 |
|---:|---:|---|---|
| **16.1 KB** | **99.50%** | `out/slot_joint.sep34.qat4.int4.scuocr` | (20,32,48,48)+sep34+**int4 QAT**（体积冠军）|
| **18.9 KB** | **99.80%** | `out/slot_joint.sep4.qat4.int4.scuocr` | (20,32,48,48)+sep4+**int4 QAT** ⭐ 推荐主力 |
| 35.0 KB | 99.50% | `out/sep4_wide_shift.scuocr` | 同上 + wide 窗 + 平移增强 ⭐ 抗版式偏移 |

**41.7 KB → 18.9 KB（−55%），精度一字不差（都是 99.80%）。**

关键结论：

0. **g5（int4 QAT × wide+shift）已跑完，结论：体积/精度/鲁棒性三者只能取二。**
   wide 窗模型的 int4 代价是 **−0.60 点**（非 wide 的 sep4 上是 −0.00），
   分解 99.80 → −0.30（改 wide 窗）→ −1.20（int4 QAT）→ **98.30%**。
   ⇒ 抗偏移版本**要用 int8 的 35.0KB（99.50%）**，不要用它的 int4 版。
   且「改了输入分布，验收必须带同一 `--crop`」已固化进
   `GPU_RUN_ALL.sh` 的 `verify_and_record`/`qat_export`（wide 模型自动带 `--crop wide`）。
1. **int4 QAT 把 int4 的损失完全补回来了**：同一模型 PTQ int4 只有 97.40%
   （−2.40 点），60 轮 QAT 后 **99.80%**。⇒ 此前「可分离 + int4 不能叠加」
   的负面结论**已被 QAT 化解**。
   > 但 QAT 不是万灵药：稠密窄配 `w_20_32_40_48` QAT 后 19.2KB/99.30%，
   > 反不如它的 PTQ int4（19.3KB/99.40%）。宽度窄到一定程度也救不回来。
2. **偏置必须保 int8**（约 200 字节，窄配置上值 +2.80 点）。见 7.2。
3. **空间可分离 conv4 是免费的**：35.0KB vs 41.7KB 稠密版，
   1000 张测试集上 **0 修复 / 0 破坏、预测完全相同**。见 7.3。
4. **平移增强把容差从 ±4px 提到 ±8px**（正确口径实测）：

   | Δ原图px | 0 | 4 | 6 | 8 | 10 | 12 |
   |---|---:|---:|---:|---:|---:|---:|
   | 现状（current 窗） | 99.80 | 97.90 | 85.80 | **54.60** | **23.00** | 8.90 |
   | wide 窗+平移增强 | 99.50 | 99.80 | 98.80 | **96.80** | **78.60** | 45.90 |

   代价仅同分布 −0.30 点、体积不变。

### ⚠ 7.1b 评估口径：改了输入分布就必须带同一配置

`sep4_wide` / `sep4_wide_shift` 用 `--crop wide` 训练，若用默认（current）窗评估，
准确率会**虚低 1.5 点以上**（98.30% vs 真实 99.60%）—— 差点据此误判「改裁剪窗有害」。

```bash
python -u tmp/verify_scuocr.py out/sep4_wide_shift.scuocr --crop wide
python -u tmp/shift_curve_orig.py out/sep4_wide_shift.scuocr --crop wide
```

> 两个脚本都已加 `--crop {current,wide}`。**凡是改了预处理、裁剪窗、增强的模型，
> 评估时必须用同一配置。**

### 7.2 已完成：增强版 int4 量化器 `tmp/quant_int4_plus.py`

```bash
# 单模型：逐层顺序量化，含 per-tensor 裁剪系数搜索 + AdaRound + int8 偏置
python -u tmp/quant_int4_plus.py <ckpt.pt> --method clip --bias-bits 8 \
  --export out/x.int4.scuocr
# 按层分配位宽（可分离层保 int8，其余 int4）
python -u tmp/quant_int4_plus.py <ckpt> --method clip \
  --keep-int8 conv3.h,conv3.v,conv4.h,conv4.v
```

实测消融（`(20,32,48,48)`，int4）：

| 方法 | 整图 | 说明 |
|---|---:|---|
| `plain`（bias int8） | 99.40% | 基线 |
| `clip`（per-tensor 裁剪搜索） | 99.40% | 配对检验 7 修 4 破、**p=0.55 不显著** |
| `adaround` | 99.40% | **零增益**（无 α 跨过阈值） |

**两条负面结论（省掉重复踩）**：
- 「按输入能量加权的权重 MSE」**不能**当尺度搜索准则 —— 它等价假设激活是白噪声，
  会一路把裁剪系数选到 0.30，实测整图直接崩到 **10.20%**。必须用**真实层输出 MSE**。
- AdaRound 的 α **必须初始化成 round-to-nearest**（+4 该入 / −4 该舍）。
  初始化成 0 会让 `soft_round(0)=0.5`、硬阈值判假 → 全取 `floor` →
  给每个权重加 **−0.5 LSB 系统性偏置**，同样崩到 10.20%。

### 7.3 已完成：空间可分离卷积链路 `tmp/apply_separable.py`

README 原先只排除了**通道**可分离；**空间**可分离（3×3 → 1×3→3×1）是未验证的独立假设，
现已确认成立，且**不需要 GPU 就能替换**（逐层最小二乘初始化 + 拟合层输出）：

```bash
python -u tmp/apply_separable.py checkpoints/slot_joint/best.pt --plan sep4 \
  --iters 800 --export out/slot_joint.sep4.scuocr
# 可选：只微调可分离层（其余冻结），本机也能跑
python -u tmp/apply_separable.py <ckpt> --plan sep34 --joint-epochs 6
```

- ⚠ **必须按前向顺序拟合**（先 conv3 再 conv4）。反序的话替换 conv3 后 conv4 的
  输入分布已变，之前拟合好的 conv4 立刻过期，白掉 0.3 点。
- 拟合残差很小（层输出相对残差 0.74~1.46%）。
- 完整链路已打通：`model.py`（`SpatialSeparableConv` + `SEPARABLE_PLANS`）、
  `quantize.py`（BN 折叠进 v + `FoldedCaptchaCNN(separable=)`）、
  `export.py`（`conv_out_channels` / `detect_separable`；`infer_arch` 现返回 **4 元组**）、
  `train.py`/`train_qat.py` 的 `--separable`、`tmp/verify_scuocr.build_model` 自动识别。

### 7.4 ⭐ GPU 待办队列（按性价比排序，命令已就绪）

> **状态（2026-09-15 首轮后）**：
> - ✅ **G1 已完成** —— 三个 QAT 全部跑完，结果见 7.1（sep4+int4 QAT = 18.9KB/99.80%）
> - ✅ **G2 已完成** —— 两个重训都跑完，平移容差 ±4px → ±8px（见 7.1 第 4 条）
> - ⏳ **G3 只跑了 1/6**（`w20_32_40_40_sep4` 训练完但没导出）
> - ⬜ G4 未开始
> - 🆕 **G5 是当前最高优先**（下面第一条）

**G5 — int4 QAT × wide+shift（把两个成果叠起来）** ★ 最值钱，约 5 分钟

```bash
bash GPU_RUN_ALL.sh --only g5
```
等价命令（**必须带 `--crop wide`**，否则量化微调的输入分布与训练时不一致）：

```bash
python train_qat.py --resume checkpoints/sep4_wide_shift/best.pt \
  --head slot --separable sep4 --widths 20,32,48,48 --crop wide \
  --epochs 60 --batch-size 128 --num-workers 8 --amp --eval-quant --export
python export.py --mixed checkpoints/qat_wide_shift/best.qat-int4.pt \
  -o out/sep4_wide_shift.qat4.int4.scuocr
python -u tmp/verify_scuocr.py out/sep4_wide_shift.qat4.int4.scuocr --crop wide
```
目标：**~19KB 且 ≥99.5%，同时扛 ±8px 版式偏移**。
（G1 证明 QAT 能补回 int4 的 2.4 点；G2 证明容差能从 ±4px 提到 ±8px。）

**G3 — `widths × separable` 二维扫描**（剩 5/6，约 1.5 小时）

```bash
bash GPU_RUN_ALL.sh --only g3
```
> 注意 g3 已跑完的 `w20_32_40_40_sep4` 只需补导出步骤，脚本会自动跳过训练。

**G1（已完成，留作记录）**
理由：sep4 + PTQ int4 是 19.1KB 但只有 97.40%（int4 代价 −2.40）。
QAT 正是为这种情况设计的，目标是 **~19KB 且 ≥99.5%**，即用同样的体积换回 2 个点。

```bash
python train_qat.py --resume checkpoints/slot_joint/best.sep4.pt \
  --head slot --separable sep4 --widths 20,32,48,48 \
  --epochs 60 --batch-size 128 --num-workers 8 --amp \
  --eval-quant --export
python export.py --mixed checkpoints/slot_joint/best.qat-int4.pt \
  -o out/slot_joint.sep4.qat4.scuocr
python -u tmp/verify_scuocr.py out/slot_joint.sep4.qat4.scuocr
```
（本机已冒烟验证：`--separable sep4` 能正确加载 ckpt 并训练，Loss 0.71 / CharAcc 99.61%
→ int4 98.50%，即 1 轮后尚差 0.9 点，靠 60 轮把 int4 代价压回 0.3 点以内是合理目标。）

**G2 — 端到端重训：裁剪窗修正 + 平移增强**
理由：现行裁剪窗横向切 1.67%、纵向切 **15.37%** 的图；平移容忍度只有 ±4px。
两者都会白扔精度，而精度可以直接换成更小的模型。

```bash
python train.py --head slot --separable sep4 \
  --init-from checkpoints/slot_joint/best.sep4.pt \
  --widths 20,32,48,48 \
  --crop wide --aug-profile shift --crop-jitter 6 \
  --epochs 500 --batch-size 128 --lr 3e-4 --lr-min 3e-5 --amp \
  --num-workers 8 --ckpt-dir checkpoints/sep4_wide --log-dir runs/sep4_wide
```
验收：先看同分布精度有没有掉（预期小幅波动），再用 `tmp/shift_curve_orig.py`
验证 ±6px 处的容忍度是否大幅改善（当前 Δ=6px 只剩 85.80%）。

**G3 — `widths × separable` 二维扫描**
理由：两个杠杆的耦合已打通，但只在两个 checkout 上测过。一次提交整批：

```bash
for w in 20,32,40,48 20,32,40,40 20,28,40,40 16,24,32,32; do
  for plan in sep4 sep34; do
    tag=$(echo $w | tr ',' '_')_$plan
    python train.py --head slot --separable $plan --widths $w \
      --epochs 500 --batch-size 128 --lr 3e-4 --lr-min 3e-5 \
      --weight-decay 5e-5 --label-smoothing 0.1 --warmup 5 --amp \
      --num-workers 8 --ckpt-dir checkpoints/$tag --log-dir runs/$tag
  done
done
# 导出 + 汇总
for d in checkpoints/*_sep*; do
  n=$(basename $d); python export.py $d/best.pt --int8 -o out/$n.scuocr
done
python -u tmp/collect_results.py "out/*.scuocr" out/w_20_32_40_48.int4.scuocr
```

**G4 — 收尾**：T2 剩余 5 档补跑满、T1 补跑满（都只跑到约一半轮次）。

> 提醒：**不要开早停**（见第 6 节第 6 条），跑满余弦。



