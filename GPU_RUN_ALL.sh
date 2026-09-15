#!/usr/bin/env bash
# =============================================================================
# GPU_RUN_ALL.sh —— 一次性跑完所有需要 GPU 的任务（挂机友好）
#
# 设计原则：
#   - **不用 set -e**：单个任务失败不能中断整晚的运行，失败会被记录并跳过。
#   - **可断点续跑**：每阶段的产物存在就跳过（--force 可强制重跑）。
#   - 每阶段独立日志到 gpu_logs/<name>.log，最后汇总到 gpu_logs/SUMMARY.txt。
#   - 每个产出的模型都走 `verify_scuocr.py` **落盘回读**验收，不信训练日志里的数字。
#
# 用法：
#   bash GPU_RUN_ALL.sh --plan          # 只看计划与预估，不执行
#   bash GPU_RUN_ALL.sh                 # 全量跑（推荐挂机）
#   bash GPU_RUN_ALL.sh --only g1       # 只跑某阶段（g0/g1/g2/g3/g4）
#   bash GPU_RUN_ALL.sh --force         # 忽略已有产物/断点，全部重跑
#   （机器意外停机后直接再跑一次即可：已完成的跳过，被中断的从 latest.pt 断点续训）
#   bash GPU_RUN_ALL.sh --workers 16    # 指定 DataLoader 进程数（默认自动探测）
#   PY=/path/to/python bash GPU_RUN_ALL.sh
#
# 预估（单卡 T4/A10 级别）：g1≈10min  g2≈40min  g3≈1.5h  g4≈1.5h  → 合计约 3.5~4h
# =============================================================================
set -u

cd "$(dirname "$0")" || exit 1

PY="${PY:-python}"
LOG_DIR="gpu_logs"
CKPT="checkpoints"
OUT="out"
mkdir -p "$LOG_DIR" "$OUT"

ONLY=""
FORCE=0
PLAN_ONLY=0
WORKERS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --plan)    PLAN_ONLY=1 ;;
    --force)   FORCE=1 ;;
    --only)    ONLY="${2:-}"; shift ;;
    --workers) WORKERS="${2:-}"; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（--help 看用法）"; exit 1 ;;
  esac
  shift
done

if [ -z "$WORKERS" ]; then
  WORKERS=$( (nproc 2>/dev/null || echo 8) )
  [ "$WORKERS" -gt 16 ] && WORKERS=16
fi

# ── 结果记录 ──
RESULTS="$LOG_DIR/results.tsv"
SUMMARY="$LOG_DIR/SUMMARY.txt"
FAILED=""

record() { printf "%s\t%s\t%s\n" "$1" "$2" "$3" >> "$RESULTS"; }

# run <阶段名> <任务名> <命令...>
# 失败不退出，只记录；日志写入 gpu_logs/<阶段名>__<任务名>.log
run() {
  local stage="$1" name="$2"; shift 2
  local log="$LOG_DIR/${stage}__${name}.log"
  echo ""
  echo "──────── [$stage] $name ────────"
  echo "  日志: $log"
  echo "  \$ $*"
  ( "$@" ) > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "  ✅ 完成"
    tail -5 "$log" | sed 's/^/     │ /'
  else
    echo "  ❌ 失败 rc=$rc"
    tail -8 "$log" | sed 's/^/     │ /'
    echo "     （完整日志：$log）"
    FAILED="${FAILED}${stage}/${name}(${rc}) "
  fi
  return $rc
}

# 产物已存在则跳过（可断点续跑）
skip_if_exists() {
  local f="$1"
  if [ "$FORCE" -eq 0 ] && [ -f "$f" ]; then
    echo "  ⏭  已有 $f，跳过（--force 可重跑）"
    return 0
  fi
  return 1
}

# 导出 + 回读验收，把结果记进 results.tsv 并**打印到控制台**
# verify_and_record <文件> [crop]
# 第 2 参是裁剪窗预设，**必须与该模型的训练窗一致**：
# 用错窗评会让准确率虚低 1.5 点以上（实测 wide 模型用 current 窗评 99.60% → 98.30%）。
verify_and_record() {
  local f="$1" crop="${2:-}"
  if [ ! -f "$f" ]; then echo "  ⚠ 缺 $f，跳过验收"; return 1; fi
  local kb line
  kb=$("$PY" -c "import os;print(f'{os.path.getsize(\"$f\")/1024:.1f}')" 2>/dev/null || echo "?")
  line=$("$PY" -u tmp/verify_scuocr.py "$f" ${crop:+"--crop"} ${crop:+"$crop"} 2>/dev/null \
         | grep -oE "单字符 [0-9.]+% +整图 [0-9.]+%" | tail -1 || true)
  if [ -n "$line" ]; then
    echo "  ✅ $(basename "$f")  ${kb} KB  $line${crop:+   [crop=$crop]}"
    record "$(basename "$f")" "$kb KB" "$line${crop:+ [$crop]}"
  else
    echo "  ⚠ $(basename "$f") 回读失败"
    record "$(basename "$f")" "$kb KB" "READBACK-FAILED"
  fi
}

# 把一个日志的末尾直接打到控制台（长训练任务里方便边挂边看）
show_log() {
  local log="$1" n="${2:-8}"
  [ -f "$log" ] && tail -"$n" "$log" | sed 's/^/     │ /'
}

# 断点续训支持：若该 ckpt 目录已有 latest.pt，输出 "--resume <path>"
# 用途：机器意外停机后重跑，从断点接着训，而不是从第 0 轮重来。
# 用法：read -r -a R <<< "$(resume_flag "$tag")"
# latest.pt 是否是**真的断点**（比 best.pt 更靠后）。
# ⚠ 只判断"文件存在"是不够的：目录里可能残留很久以前某次中断的 latest.pt
#   （本项目实测 slot_joint/latest.pt 是 epoch=8 的老残档，而 best.pt 已 epoch=224），
#   盲目续训会从严重退步的状态开始，白跑一整轮。
latest_is_ahead() {
  local dir="$CKPT/$1"
  local latest="$dir/latest.pt" best="$dir/best.pt"
  [ -f "$latest" ] || return 1
  [ -f "$best" ] || return 0          # 没有 best 就只能用 latest
  "$PY" - "$latest" "$best" >/dev/null 2>&1 <<'PYCHK'
import sys
try:
    import torch
except Exception:
    raise SystemExit(1)             # 没 torch 就保守起见不续训
def ep(p):
    try:
        return int(torch.load(p, map_location="cpu", weights_only=False).get("epoch", -1))
    except Exception:
        return -1
raise SystemExit(0 if ep(sys.argv[1]) > ep(sys.argv[2]) else 1)
PYCHK
}

resume_flag() {
  if [ "$FORCE" -eq 0 ] && latest_is_ahead "$1"; then
    printf -- "--resume %s" "$CKPT/$1/latest.pt"
  fi
}

# 选续训起点：优先本任务自己的 latest.pt（真断点续训），否则用给定的起点权重
pick_start() {
  local tag="$1" fallback="$2"
  if [ "$FORCE" -eq 0 ] && latest_is_ahead "$tag"; then
    echo "  ↻ 发现 $CKPT/$tag/latest.pt 比 best.pt 更靠后 → 从断点继续" >&2
    echo "$CKPT/$tag/latest.pt"
  else
    echo "$fallback"
  fi
}

# train_qat.py 的 --export 产出的是中间格式，必须再 export.py --mixed 转一次
# qat_export <stage> <tag> <ckptdir> [crop]   ← crop 用于回读验收
qat_export() {
  local stage="$1" tag="$2" ckptdir="$3" crop="${4:-}"
  local mixed="$ckptdir/best.qat-int4.pt"
  if [ ! -f "$mixed" ]; then echo "  ⚠ 缺 $mixed（QAT 未产出）"; return 1; fi
  run "$stage" "export_$tag" "$PY" export.py --mixed "$mixed" -o "$OUT/$tag.int4.scuocr"
  verify_and_record "$OUT/$tag.int4.scuocr" "$crop"
}

T1_COMMON=(--head slot --epochs 500 --batch-size 128 --lr 3e-4 --lr-min 3e-5
           --weight-decay 5e-5 --label-smoothing 0.1 --warmup 5 --amp)

echo "========================================================="
echo " zhjw 验证码模型 —— GPU 全量任务"
echo " python   : $("$PY" -c 'import sys;print(sys.executable)' 2>/dev/null || echo "$PY")"
echo " workers  : $WORKERS"
echo " 阶段     : ${ONLY:-g0 g1 g2 g3 g4 g5}"
echo " force    : $FORCE"
echo "========================================================="

if [ "$PLAN_ONLY" -eq 1 ]; then
cat <<'PLAN'
【g0】环境自检 + 基线回读验收（约 2 分钟）
     目的：确认云端数据划分与本地一致，否则后面所有数字都不可比。
     验收 5 个已知文件，期望 19.3KB/99.40%、25.1KB/99.50%、29.1KB/99.70%、
     35.0KB/99.80%、41.7KB/99.80%。

【g1】int4 QAT —— 当前最值钱的阶段（约 10 分钟，3 个任务）
     理由：可分离层的 int4 代价是 −2.40 点（稠密层只有 −0.40），sep4+PTQ int4
           是 19.1KB 但只有 97.40%。QAT 正是为这种情况设计的。
     g1a  sep4  → 目标 ~19KB 且 ≥99.5%
     g1b  sep34 → 目标 ~16KB 且 ≥99.3%
     g1c  稠密 (20,32,40,48) → 把体积冠军 19.3KB/99.40% 的精度提上去

【g2】裁剪窗修正 + 平移增强重训（约 40 分钟，2 个任务）
     理由：现行裁剪窗纵向切掉 15.37% 的图；平移容忍度只有 ±4px（Δ=6px 只剩 85.8%）。
           这些都是白扔的精度，而精度可以直接换成更小的模型。
     两个变体用于**归因**（分开看裁剪与增强各自的贡献）：
     g2a  --crop wide --aug-profile default
     g2b  --crop wide --aug-profile shift --crop-jitter 6
     完成后自动跑 shift_curve_orig.py 画平移容忍度曲线。

【g3】widths × separable 二维扫描（约 1.5 小时，6 个任务）
     理由：两个体积杠杆的耦合已打通，但只在 2 个 checkpoint 上测过。
     widths ∈ {(20,32,40,40), (20,28,40,40), (16,24,32,32)} × plan ∈ {sep4, sep34}

【g5】int4 QAT × wide+shift（约 5 分钟，1 个任务）★ 把两个成果叠起来
     理由：g1 已证明 QAT 能把 int4 损失补回（sep4：PTQ 97.40% → QAT 99.80%）；
           g2 已证明 wide 窗+平移增强把 ±8px 容忍度从 54.6% 提到 96.8%。
           叠起来可同时拿到「~19KB」与「扛版式偏移」。注意必须带 --crop wide。

【g4】补齐历史任务（约 1.5 小时，6 个任务）
     g4a  T1 补跑满 500 轮（之前只到 224）
     g4b  T2 剩余 5 档（20,32,40,40 / 20,28,40,40 / 16,28,40,40 / 16,24,32,32 / 12,20,28,28）

最后自动汇总：gpu_logs/results.tsv + 全量 Pareto 前沿表。
PLAN
exit 0
fi

# 取值校验：**必须在解析参数后立刻做**。
# 曾经没有这道校验，`--only g5` 传给不含 g5 的旧版脚本时会静默什么都不做，
# 还打印「✅ 没有失败的任务」—— 极具误导性。
case "$ONLY" in
  ""|g0|g1|g2|g3|g4|g5) ;;
  *) echo "❌ 未知阶段: $ONLY"
     echo "   本脚本支持的阶段: g0 g1 g2 g3 g4 g5"
     echo "   （若你确认写过 g5，说明手上的 GPU_RUN_ALL.sh 是旧版，请更新）"
     exit 1 ;;
esac

# 统计实际执行了几个阶段，用于在汇总里识别「什么都没跑」
RAN=0
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

# =============================================================================
# g0 环境自检 + 基线验收
# =============================================================================
if want g0; then
  RAN=$((RAN+1))
  echo ""
  echo "========== g0 环境自检 + 基线回读验收 =========="
  run g0 env_check "$PY" -c "import torch,sys;print('torch',torch.__version__);print('cuda',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY — 训练会很慢')"
  show_log "$LOG_DIR/g0__env_check.log" 4
  run g0 data_check "$PY" -c "import os;print('label.csv',os.path.getsize('data/label.csv'));print('IMAGES.zip',os.path.getsize('data/IMAGES.zip') if os.path.exists('data/IMAGES.zip') else 'MISSING')"
  show_log "$LOG_DIR/g0__data_check.log" 3

  echo ""
  echo "  基线回读（期望 99.40 / 99.50 / 99.70 / 99.80 / 99.80）："
  for f in out/w_20_32_40_48.int4.scuocr out/w_20_32_40_48.sep34.scuocr \
           out/slot_joint.sep34.scuocr out/slot_joint.sep4.scuocr \
           zhjw-model.slot.joint.scuocr; do
    verify_and_record "$f"
  done
  echo "  ↑ 数字对不上就是数据划分不一致 —— 先别往下跑（CLOUD_TRAINING.md 第 3 节）"
fi

# =============================================================================
# g1 int4 QAT（最值钱）
# =============================================================================
if want g1; then
  RAN=$((RAN+1))
  echo ""
  echo "========== g1 int4 QAT =========="

  # g1a sep4
  if ! skip_if_exists "$OUT/slot_joint.sep4.qat4.int4.scuocr"; then
    run g1 g1a_qat_sep4 "$PY" -u train_qat.py \
      --resume "$(pick_start qat_sep4 "$CKPT/slot_joint/best.sep4.pt")" \
      --head slot --separable sep4 --widths 20,32,48,48 \
      --epochs 60 --batch-size 128 --num-workers "$WORKERS" --amp \
      --eval-quant --export \
      --ckpt-dir "$CKPT/qat_sep4" --log-dir "runs/qat_sep4"
    qat_export g1 slot_joint.sep4.qat4 "$CKPT/qat_sep4"
  fi

  # g1b sep34
  if ! skip_if_exists "$OUT/slot_joint.sep34.qat4.int4.scuocr"; then
    run g1 g1b_qat_sep34 "$PY" -u train_qat.py \
      --resume "$(pick_start qat_sep34 "$CKPT/slot_joint/best.sep34.pt")" \
      --head slot --separable sep34 --widths 20,32,48,48 \
      --epochs 60 --batch-size 128 --num-workers "$WORKERS" --amp \
      --eval-quant --export \
      --ckpt-dir "$CKPT/qat_sep34" --log-dir "runs/qat_sep34"
    qat_export g1 slot_joint.sep34.qat4 "$CKPT/qat_sep34"
  fi

  # g1c 稠密体积冠军：把 19.3KB/99.40% 的精度提上去
  if ! skip_if_exists "$OUT/w_20_32_40_48.qat4.int4.scuocr"; then
    run g1 g1c_qat_dense "$PY" -u train_qat.py \
      --resume "$(pick_start qat_dense "$CKPT/w_20_32_40_48/best.pt")" \
      --head slot --widths 20,32,40,48 \
      --epochs 60 --batch-size 128 --num-workers "$WORKERS" --amp \
      --eval-quant --export \
      --ckpt-dir "$CKPT/qat_dense" --log-dir "runs/qat_dense"
    qat_export g1 w_20_32_40_48.qat4 "$CKPT/qat_dense"
  fi
fi

# =============================================================================
# g2 裁剪窗修正 + 平移增强重训
# =============================================================================
if want g2; then
  RAN=$((RAN+1))
  echo ""
  echo "========== g2 裁剪窗 wide + 平移增强 =========="

  # g2a 只换裁剪窗（隔离裁剪的贡献）
  if ! skip_if_exists "$OUT/sep4_wide.scuocr"; then
    run g2 g2a_wide "$PY" -u train.py "${T1_COMMON[@]}" \
      --separable sep4 --init-from "$(pick_start sep4_wide "$CKPT/slot_joint/best.sep4.pt")" \
      --widths 20,32,48,48 --crop wide --aug-profile default \
      --num-workers "$WORKERS" \
      --ckpt-dir "$CKPT/sep4_wide" --log-dir "runs/sep4_wide"
    run g2 g2a_export "$PY" export.py "$CKPT/sep4_wide/best.pt" --int8 -o "$OUT/sep4_wide.scuocr"
    verify_and_record "$OUT/sep4_wide.scuocr" wide
  fi

  # g2b 换裁剪窗 + 平移增强 + 裁剪窗抖动
  if ! skip_if_exists "$OUT/sep4_wide_shift.scuocr"; then
    run g2 g2b_wide_shift "$PY" -u train.py "${T1_COMMON[@]}" \
      --separable sep4 --init-from "$(pick_start sep4_wide_shift "$CKPT/slot_joint/best.sep4.pt")" \
      --widths 20,32,48,48 --crop wide --aug-profile shift --crop-jitter 6 \
      --num-workers "$WORKERS" \
      --ckpt-dir "$CKPT/sep4_wide_shift" --log-dir "runs/sep4_wide_shift"
    run g2 g2b_export "$PY" export.py "$CKPT/sep4_wide_shift/best.pt" --int8 -o "$OUT/sep4_wide_shift.scuocr"
    verify_and_record "$OUT/sep4_wide_shift.scuocr" wide
  fi

  # 平移容忍度曲线（对比：新手 vs 现状）
  run g2 shift_curve_new "$PY" -u tmp/shift_curve_orig.py "$OUT/sep4_wide_shift.scuocr" --max 32 --step 2
  run g2 shift_curve_base "$PY" -u tmp/shift_curve_orig.py zhjw-model.slot.joint.scuocr --max 32 --step 2
  echo "  ↑ 关键看 Δ=6px 处：现状 85.80%，目标是显著更高"
fi

# =============================================================================
# g3 widths × separable 二维扫描
# =============================================================================
if want g3; then
  RAN=$((RAN+1))
  echo ""
  echo "========== g3 widths × separable 扫描 =========="
  for w in 20,32,40,40 20,28,40,40 16,24,32,32; do
    for plan in sep4 sep34; do
      tag="w$(echo "$w" | tr ',' '_')_$plan"
      if skip_if_exists "$OUT/$tag.scuocr"; then continue; fi
      read -r -a RESUME_ARGS <<< "$(resume_flag "$tag")"
      run g3 "train_$tag" "$PY" -u train.py "${T1_COMMON[@]}" \
        --separable "$plan" --widths "$w" \
        ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
        --num-workers "$WORKERS" \
        --ckpt-dir "$CKPT/$tag" --log-dir "runs/$tag"
      run g3 "export_$tag" "$PY" export.py "$CKPT/$tag/best.pt" --int8 -o "$OUT/$tag.scuocr"
      verify_and_record "$OUT/$tag.scuocr"
      # 顺带出该配置的 int4（本机已证明可分离层抗 int4 差，这里量化后再看一次）
      run g3 "int4_$tag" "$PY" -u tmp/quant_int4_plus.py "$CKPT/$tag/best.pt" \
        --method clip --bias-bits 8 --export "$OUT/$tag.int4.scuocr"
      verify_and_record "$OUT/$tag.int4.scuocr"
    done
  done
fi

# =============================================================================
# g4 补齐历史任务（T1 跑满 + T2 剩余档）
# =============================================================================
if want g4; then
  RAN=$((RAN+1))
  echo ""
  echo "========== g4 补齐 T1/T2 =========="

  # T1 补跑满（之前只到 epoch 224/500）
  if ! skip_if_exists "$OUT/slot_joint.full.scuocr"; then
    run g4 g4a_t1_full "$PY" -u train.py "${T1_COMMON[@]}" \
      --head slot --init-from "$(pick_start slot_joint_full "$CKPT/best.narrow.pt")" \
      --widths 20,32,48,48 \
      --num-workers "$WORKERS" \
      --ckpt-dir "$CKPT/slot_joint_full" --log-dir "runs/slot_joint_full"
    run g4 g4a_export "$PY" export.py "$CKPT/slot_joint_full/best.pt" --int8 -o "$OUT/slot_joint.full.scuocr"
    verify_and_record "$OUT/slot_joint.full.scuocr"
  fi

  # T2 剩余 5 档（稠密）
  for w in 20,32,40,40 20,28,40,40 16,28,40,40 16,24,32,32 12,20,28,28; do
    tag="d$(echo "$w" | tr ',' '_')"
    if skip_if_exists "$OUT/$tag.scuocr"; then continue; fi
    read -r -a RESUME_ARGS <<< "$(resume_flag "$tag")"
    run g4 "train_$tag" "$PY" -u train.py "${T1_COMMON[@]}" \
      --head slot --widths "$w" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
      --num-workers "$WORKERS" \
      --ckpt-dir "$CKPT/$tag" --log-dir "runs/$tag"
    run g4 "export_$tag" "$PY" export.py "$CKPT/$tag/best.pt" --int8 -o "$OUT/$tag.scuocr"
    verify_and_record "$OUT/$tag.scuocr"
  done
fi

# =============================================================================
# g5 int4 QAT 打「wide 窗 + 平移增强」模型 —— 把两个成果叠起来
#    g1 已证明 QAT 能把 int4 的损失补回来（sep4：PTQ 97.40% → QAT 99.80%）；
#    g2 已证明 wide 窗 + 平移增强把 ±8px 的容忍度从 54.6% 提到 96.8%（代价 −0.3 点）。
#    两者叠起来就能同时拿到「~19KB」和「扛版式偏移」。
# =============================================================================
if want g5; then
  RAN=$((RAN+1))
  echo ""
  echo "========== g5 int4 QAT × wide+shift =========="
  if ! skip_if_exists "$OUT/sep4_wide_shift.qat4.int4.scuocr"; then
    # ⚠ 必须带 --crop wide：该模型是用 wide 窗训的，量化微调也要用同一个窗
    run g5 g5_qat_wide_shift "$PY" -u train_qat.py \
      --resume "$(pick_start qat_wide_shift "$CKPT/sep4_wide_shift/best.pt")" \
      --head slot --separable sep4 --widths 20,32,48,48 --crop wide \
      --epochs 60 --batch-size 128 --num-workers "$WORKERS" --amp \
      --eval-quant --export \
      --ckpt-dir "$CKPT/qat_wide_shift" --log-dir "runs/qat_wide_shift"
    qat_export g5 sep4_wide_shift.qat4 "$CKPT/qat_wide_shift" wide
    echo "  验收：体积应 ~19KB；准确率用 --crop wide 评（默认窗会虚低 1.5 点以上）"
    echo "        python -u tmp/verify_scuocr.py $OUT/sep4_wide_shift.qat4.int4.scuocr --crop wide"
  fi
fi

# =============================================================================
# 汇总
# =============================================================================
echo ""
echo "========== 汇总 =========="
{
  echo "运行结束：$(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
  if [ "$RAN" -eq 0 ]; then
    echo "⚠⚠ 没有任何阶段被执行！"
    echo "    常见原因：--only 的值不在 g0..g5 里，或手上的脚本是旧版（不含该阶段）。"
    echo "    本次 ONLY='${ONLY}'"
  elif [ -n "$FAILED" ]; then
    echo "❌ 失败的任务：$FAILED"
  else
    echo "✅ 没有失败的任务（已执行 $RAN 个阶段）"
  fi
  echo ""
  echo "── 本次回读验收结果（gpu_logs/results.tsv）──"
  [ -f "$RESULTS" ] && cat "$RESULTS" || echo "(空)"
  echo ""
  echo "── 全量 Pareto 前沿（含本机已有产物）──"
  "$PY" -u tmp/collect_results.py "out/*.scuocr" zhjw-model.slot.joint.scuocr 2>&1 | tail -30
} > "$SUMMARY" 2>&1
cat "$SUMMARY"

echo ""
echo "汇总已写入 $SUMMARY"
echo "把 out/*.scuocr 与 gpu_logs/ 一起拷回来即可"
