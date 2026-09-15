#!/usr/bin/env bash
# =============================================================================
# zhjw-ocr 云端训练 全部命令
# 详细说明见 CLOUD_TRAINING.md；本文件只列可复制粘贴的命令。
#
# 用法：把这个文件放在项目根目录，按需逐段执行（不要无脑整跑）。
#   bash RUNBOOK_COMMANDS.sh --check     # 只跑环境与基线验收
#   bash RUNBOOK_COMMANDS.sh --t1        # T1 联合训练
#   bash RUNBOOK_COMMANDS.sh --t2        # T2 骨干瘦身扫描
# =============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
CKPT="${CKPT:-checkpoints}"
PY="${PY:-python}"

# =============================================================================
# 0. 环境
# =============================================================================
env_setup() {
  pip install -r requirements.txt
  # 若用 GPU，确认 torch 能看到卡：
  $PY - <<'EOF'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
EOF
}

# =============================================================================
# 1. 基线验收（先跑这个，确认环境和数据划分没问题）
#    期望输出：41.7KB / 单字符 99.85% / 整图 99.40%
# =============================================================================
check_baseline() {
  # 首次运行会自动把 data/IMAGES.zip 解压到 data/IMAGES/（约 10s）
  $PY -u tmp/verify_scuocr.py zhjw-model.slot.scuocr \
                              zhjw-model.scuocr \
                              zhjw-model.int8.scuocr

  # 阈值检查（脚本内自带断言，不符会报错退出）
  $PY - <<'EOF'
import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "tmp")
from verify_scuocr import parse, build_model
from quantize import evaluate
from preprocess import ZhjwCaptchaDataset, collate_fn
from torch.utils.data import DataLoader, random_split
import torch

ds = ZhjwCaptchaDataset(data_dir="data", augment=False)
n = len(ds); tv = tt = int(n*0.1)
_, _, te = random_split(range(n), [n-tv-tt, tv, tt],
                        generator=torch.Generator().manual_seed(42))
loader = DataLoader(torch.utils.data.Subset(ds, list(te)), batch_size=128,
                    shuffle=False, num_workers=0, collate_fn=collate_fn)
ver, sd = parse("zhjw-model.slot.scuocr")
m, widths, head = build_model(sd)
ca, sa = evaluate(m, loader)
kb = os.path.getsize("zhjw-model.slot.scuocr")/1024
print(f"实测 {kb:.1f}KB  单字符 {ca:.2f}%  整图 {sa:.2f}%")
assert abs(kb - 42.7) < 1.0, "体积与预期不符"
assert abs(sa - 99.40) < 0.3, "精度与预期不符 —— 检查数据划分是否用了 seed=42"
print("✓ 基线验收通过")
EOF
}

# =============================================================================
# 2. 预算表（提交 T2/T4 前先看这个，避免盲试）
# =============================================================================
budget() {
  $PY -u tmp/compress_budget.py --csv out/budget.csv   # widths × 卷积分解 × 位宽
  $PY -u tmp/width_budget.py                           # 旧版逐 widths 明细
}

# =============================================================================
# 3. T1 ★ 逐槽头 + 骨干联合训练（最高优先）
#    现状 best.slot.pt 是「冻结骨干只训头」的产物（99.40%）；
#    联合训练让骨干适应新头，预期还能提升。
#    GPU 预计 10~20 分钟（500 轮）。
# =============================================================================
t1_train() {
  $PY -u train.py --head slot \
    --init-from "$CKPT/best.narrow.pt" \
    --widths 20,32,48,48 \
    --epochs 500 --batch-size 128 \
    --lr 3e-4 --lr-min 3e-5 --weight-decay 5e-5 \
    --label-smoothing 0.1 --warmup 5 --amp \
    --num-workers 8 \
    --ckpt-dir "$CKPT/slot_joint" --log-dir runs/slot_joint
}

t1_export_check() {
  $PY export.py "$CKPT/slot_joint/best.pt" --int8 -o zhjw-model.slot.joint.scuocr
  $PY -u tmp/verify_scuocr.py zhjw-model.slot.joint.scuocr
  # 验收标准：整图 ≥ 99.40% 且文件 ≤ 42.7KB 才算优于现状
  $PY -u tmp/collect_results.py
}

# 若嫌 500 轮太久，先跑短版探路（100 轮）：
t1_quick() {
  $PY -u train.py --head slot \
    --init-from "$CKPT/best.narrow.pt" \
    --widths 20,32,48,48 --epochs 100 --batch-size 128 \
    --lr 3e-4 --weight-decay 5e-5 --amp --num-workers 8 \
    --ckpt-dir "$CKPT/slot_joint_quick"
}

# =============================================================================
# 4. T2 ★ 骨干瘦身（一次提交一批候选）
#    预算表见 tmp/width_budget.py。参考：
#      (20,32,48,48) 41.7KB  ← 现状 99.40%
#      (20,32,40,48) 36.1KB  -13%
#      (20,32,40,40) 33.1KB  -21%
#      (20,28,40,40) 30.9KB  -26%
#      (16,28,40,40) 29.9KB  -28%
#      (16,24,32,32) 21.0KB  -50%
#      (12,20,28,28) 15.6KB  -63%
# =============================================================================
t2_sweep() {
  for w in 20,32,40,48 20,32,40,40 20,28,40,40 16,28,40,40 16,24,32,32 12,20,28,28; do
    tag=$(echo "$w" | tr ',' '_')
    echo "=== widths=$w -> $CKPT/w_$tag ==="
    $PY -u train.py --head slot \
      --widths "$w" \
      --epochs 500 --batch-size 128 \
      --lr 3e-4 --lr-min 3e-5 --weight-decay 5e-5 \
      --label-smoothing 0.1 --warmup 5 --amp \
      --num-workers 8 \
      --ckpt-dir "$CKPT/w_$tag" --log-dir "runs/w_$tag"
  done
}

# 也可从现有骨干热启动，收敛更快（推荐先试这个）：
t2_sweep_warmstart() {
  for w in 20,32,40,40 16,24,32,32; do
    tag=$(echo "$w" | tr ',' '_')
    $PY -u train.py --head slot \
      --init-from "$CKPT/best.narrow.pt" \
      --widths "$w" --epochs 300 --batch-size 128 \
      --lr 3e-4 --weight-decay 5e-5 --amp --num-workers 8 \
      --ckpt-dir "$CKPT/w_$tag"
  done
}

t2_export_check() {
  mkdir -p out
  # export.py 会从 checkpoint 自动推断 widths 与 head_type，无需额外参数
  for w in 20_32_40_48 20_32_40_40 20_28_40_40 16_28_40_40 16_24_32_32 12_20_28_28; do
    [ -f "$CKPT/w_$w/best.pt" ] || continue
    $PY export.py "$CKPT/w_$w/best.pt" --int8 -o "out/w_$w.scuocr"
  done
  # 一次性汇总所有候选的体积/精度，并给出推荐
  $PY -u tmp/collect_results.py "out/*.scuocr" zhjw-model.slot.scuocr
}

# =============================================================================
# 5. T3 骨干 int4 QAT —— 当前最大的体积杠杆（权重字节减半）
#    代码已就绪（train_qat.py 已支持 --head slot，2026-09-15）
#    实测 PTQ（未 QAT）：(20,32,48,48)/slot → 22.0KB / 99.30%（掉 0.50 点）
#    QAT 的目标是把这 0.50 点补回来。
# =============================================================================
t3_qat_slot() {
  $PY -u train_qat.py \
    --resume "$CKPT/slot_joint/best.pt" \
    --head slot --widths 20,32,48,48 \
    --epochs 40 --batch-size 128 --num-workers 8 \
    --eval-quant --export
  # train_qat.py 产出的是中间格式 best.qat-int4.pt，必须再转一次才是可部署的 .scuocr
  local mixed="$CKPT/slot_joint/best.qat-int4.pt"
  [ -f "$mixed" ] || { echo "未找到 $mixed"; return 1; }
  $PY export.py --mixed "$mixed" -o zhjw-model.slot.int4.scuocr
  $PY -u tmp/verify_scuocr.py zhjw-model.slot.int4.scuocr
  echo "验收线：≤22KB 且整图 >=99.30%"
}

# 零 GPU 成本：先估「量化后体积/精度」，再决定要不要提交 QAT 任务
t3_probe() {
  $PY -u tmp/probe_int4.py "$CKPT/slot_joint/best.pt" "$CKPT/w_20_32_40_48/best.pt"
}

# =============================================================================
# 6. 辅助：体积审计 / PTQ 扫描 / 合成数据
# =============================================================================
audit() {
  $PY -u tmp/model_size_audit.py          # 逐层参数量 + 熵 + gzip/lzma
  $PY -u tmp/quant_sweep.py               # PTQ int8/int4 扫描（含分位数裁剪）
}

synth_rebuild() {
  # 字形池（74MB）未随包上传，需要时重建，约 2 分钟
  $PY -u synth_gen_compose.py --build
  $PY -u tmp/synth_geom.py                # 重新生成逐类几何表
  $PY -u synth_gen_compose.py -n 24       # 出样例图
}

synth_eval() {
  $PY -u tmp/synth_ablation.py            # 合成管线各环节消融
  $PY -u tmp/synth_eval_compose.py        # 重组合成保真度（域分类器）
}

# =============================================================================
# 7. 本机可做（无需 GPU）
# =============================================================================
# 空间可分离替换：把训练好的 conv3/conv4 换成 (1×3)→(3×1)，逐层拟合，不训练
#   sep4  = 仅 conv4（实测零掉点、参数 -16%）  ← 推荐先跑这个
#   sep34 = conv3+conv4（实测 -0.1 点、参数 -31%）
sep_apply() {
  local ck="${2:-$CKPT/slot_joint/best.pt}" plan="${3:-sep4}"
  local tag; tag=$(basename "${ck%.pt}")
  $PY -u tmp/apply_separable.py "$ck" --plan "$plan" --iters 800 \
    --export "out/$tag.$plan.scuocr"
}

# 空间可分离**可行性判定**（只看掉点与残差，不产出模型）
sep_probe() {
  $PY -u tmp/sep_feasibility.py "${2:-$CKPT/slot_joint/best.pt}" --iters 600
}

# 增强版 int4：int8 偏置 + per-tensor 裁剪系数搜索（务必带 --bias-bits 8）
#   第 3 个参数可传 --keep-int8 conv4.h,conv4.v 之类做按层位宽分配
int4_plus() {
  local ck="${2:-$CKPT/slot_joint/best.pt}"; shift 2 2>/dev/null || true
  $PY -u tmp/quant_int4_plus.py "$ck" --method clip --bias-bits 8 "$@"
}

# 全部候选汇总成 Pareto 前沿（体积/参数量/精度 + 判定）
pareto() {
  mkdir -p out
  $PY -u tmp/collect_results.py "out/*.scuocr" zhjw-model.slot.joint.scuocr
}

# 平移敏感性曲线（在原图上平移后重新裁剪，模拟版式偏移）
shift_curve() {
  $PY -u tmp/shift_curve_orig.py "${2:-zhjw-model.slot.joint.scuocr}" --max 32 --step 2
}

# =============================================================================
case "${1:-}" in
  --check)    env_setup; check_baseline; budget ;;
  --t1)       t1_train; t1_export_check ;;
  --t1-quick) t1_quick ;;
  --t2)       budget; t2_sweep; t2_export_check ;;
  --t2-warm)  t2_sweep_warmstart; t2_export_check ;;
  --t3)       t3_qat_slot ;;
  --t3-probe) t3_probe ;;
  --sep)        sep_apply "$@" ;;
  --sep-probe)  sep_probe "$@" ;;
  --int4plus)   int4_plus "$@" ;;
  --pareto)     pareto ;;
  --shift)      shift_curve "$@" ;;
  --budget)   budget ;;
  --audit)    audit ;;
  --synth)    synth_rebuild; synth_eval ;;
  *)
    echo "用法: bash RUNBOOK_COMMANDS.sh <子命令> [参数]"
    echo "本机可做（无需 GPU）:"
    echo "  --sep [ckpt] [sep4|sep34]  空间可分离替换 + 导出（sep4 零掉点、体积 -16%）"
    echo "  --sep-probe [ckpt]         空间可分离可行性判定（只看掉点）"
    echo "  --int4plus [ckpt] [...]    增强 int4（int8 偏置 + 裁剪搜索）"
    echo "  --pareto                   全部候选汇总成 Pareto 前沿"
    echo "  --shift [模型]             平移敏感性曲线"
    echo "需要 GPU:"
    echo "  --t3                        int4 QAT（可加 --separable sep4）"
    echo "  --t1 / --t2                 联合训练 / 骨干瘦身扫描"
    echo "其它:"
    echo "  --check  环境+基线验收+预算    --budget  压缩预算"
    echo "  --audit  体积审计             --synth   合成数据（已降级）"
    echo
    echo "  --check     环境 + 基线回读验收 + 压缩预算（约 1 分钟）"
    echo "  --t1        T1 逐槽头+骨干联合训练（已跑出 41.7KB/99.80%）"
    echo "  --t2        骨干瘦身扫描（(20,32,40,48) 已跑出 36.1KB/99.70%）"
    echo "  --t3        ★ int4 QAT（当前最大的体积杠杆，目标 ~22KB）"
    echo "  --t3-probe  零 GPU 估 int4 的体积/精度，先探再训"
    echo "  --budget    压缩预算（widths × 卷积分解 × 位宽）"
    echo "  --audit     体积审计 + PTQ 扫描"
    echo "  --synth     合成数据（已降级，暂不使用）"
    ;;
esac
