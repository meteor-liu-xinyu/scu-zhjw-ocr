# 变更清单（相对 zhjw-cloud-20260915）

本包基于 `zhjw-cloud-20260915`，下列文件**必须替换**，其余未动。

## 必须更新的核心代码（6 个）

| 文件 | 改了什么 |
|---|---|
| `model.py` | 新增 `SpatialSeparableConv` + `SEPARABLE_PLANS`；`CaptchaCNN(..., separable=)` |
| `preprocess.py` | `CROP_PRESETS`(current/wide) + `set_crop/get_crop`；`crop_jitter`；`_AUG_PROFILES` 增 shift/shift_strong；修 `_augment` 漏传 profile |
| `train.py` | `--separable` / `--crop` / `--crop-jitter`；`--aug-profile` 扩到 4 档 |
| `train_qat.py` | `--head slot`（解 T3 阻塞）、`--separable`、`--crop`、`--crop-jitter`；`AugDataset` 提到模块级（修 Windows spawn 崩溃）；修 profile 漏传 |
| `quantize.py` | BN 折叠支持 `conv{i}.h/v`；`FoldedCaptchaCNN(..., separable=)` |
| `export.py` | 新增 `conv_out_channels()`/`detect_separable()`；`ordered_tensor_names` 支持 h/v；`infer_arch` 改返回 **4 元组**；**删掉重复的 `fold_bn_into_conv` 改为委托 quantize**（原来会静默丢掉 `conv{i}.h/v` 键） |

## 必须更新的工具（1 个）

| 文件 | 改了什么 |
|---|---|
| `tmp/verify_scuocr.py` | **修 int4 解析 bug**（int4 是打包的，长度 `ceil(n/2)` 而非 `n`）；`build_model` 自动识别可分离层 |

## 新增工具（9 个）

`tmp/quant_int4_plus.py`、`tmp/apply_separable.py`、`tmp/sep_feasibility.py`、
`tmp/probe_int4.py`、`tmp/compress_budget.py`、`tmp/compare_models.py`、
`tmp/err_zoom.py`、`tmp/shift_curve.py`、`tmp/shift_curve_orig.py`

## 根目录新增两个脚本

| 脚本 | 作用 |
|---|---|
| `GPU_RUN_ALL.sh` | 一次性跑完所有 GPU 任务。不用 `set -e`（失败不中断）、可按产物跳过、**被停机打断的训练会从 `latest.pt` 断点续训**（仅当 latest 的 epoch 确实超过 best 时，避免残留老档把训练带回早期）、每个产出都落盘回读验收、自动出 Pareto 表 |
| `PACK_RESULTS.sh` | 打包所有 GPU 运行结果（`out/` + `checkpoints/` + `gpu_logs/` + `runs/` + 自动生成的 `MANIFEST.txt`），用 Python `zipfile` 实现，不依赖 `zip`/`tar` 命令 |

## 新增权重（5 个）

| 文件 | 说明 |
|---|---|
| `checkpoints/slot_joint/best.pt` | **T1 结果（41.7KB/99.80%）—— 云端最重要的起点** |
| `checkpoints/slot_joint/best.sep4.pt` | 空间可分离 conv4（测试集预测与原模型完全一致） |
| `checkpoints/slot_joint/best.sep34.pt` | 空间可分离 conv3+4（−0.10 点） |
| `checkpoints/w_20_32_40_48/best.pt` | 瘦身 36.1KB/99.70% |
| `checkpoints/w_20_32_40_48/best.sep34.pt` | 窄配 + sep34 |

## 新增产物（Pareto 前沿，`out/`）

| 体积 | 整图 | 文件 |
|---:|---:|---|
| **19.3 KB** | 99.40% | `out/w_20_32_40_48.int4.scuocr` |
| 25.1 KB | 99.50% | `out/w_20_32_40_48.sep34.scuocr` |
| 29.1 KB | 99.70% | `out/slot_joint.sep34.scuocr` |
| 35.0 KB | 99.80% | `out/slot_joint.sep4.scuocr` |

## 未变化（照旧）

`quantize_mixed.py`、`train_slot_head.py`、`requirements.txt`、`synth_gen*.py`、
`notes/`、其余 `tmp/` 脚本。

## 立即可做（解压后第一条命令）

```bash
bash GPU_RUN_ALL.sh --plan      # 只看计划与预估
bash GPU_RUN_ALL.sh             # 全量跑（挂机），日志在 gpu_logs/
bash PACK_RESULTS.sh --list     # 跑完（或被中断）后，看清单
bash PACK_RESULTS.sh --no-runs  # 打包结果拷回本地
```
