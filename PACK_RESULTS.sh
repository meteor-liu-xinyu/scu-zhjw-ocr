#!/usr/bin/env bash
# =============================================================================
# PACK_RESULTS.sh —— 把 GPU 机器上的运行结果打成一个包，拷回本地
#
# 打包内容：
#   out/*.scuocr          所有导出的模型（含各配置的 int8 / int4）
#   checkpoints/**/*.pt   所有 checkpoint（可继续断点续训，别丢）
#   gpu_logs/             每任务日志 + results.tsv + SUMMARY.txt
#   runs/                 TensorBoard 日志（--no-runs 可排除，可能较大）
#   MANIFEST.txt          自动生成：文件清单 + 各任务完成情况 + 续跑提示
#
# 不打包（本地已有，或体积过大）：data/、tmp/、__pycache__、*.pyc
#
# 用法：
#   bash PACK_RESULTS.sh              # 打包（默认含 runs/）
#   bash PACK_RESULTS.sh --no-runs    # 不含 TensorBoard 日志（推荐先看体积）
#   bash PACK_RESULTS.sh --list       # 只看清单与完成情况，不打包
#
# 打包用 Python 的 zipfile（机器上必然有 python，不依赖 zip/tar 命令）。
# =============================================================================
set -u
cd "$(dirname "$0")" || exit 1

PY="${PY:-python}"
STAMP=$(date +%Y%m%d-%H%M)
NAME="zhjw-gpu-results-$STAMP"
INCLUDE_RUNS=1
LIST_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-runs) INCLUDE_RUNS=0 ;;
    --list)    LIST_ONLY=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
  shift
done

echo "========================================================="
echo " 打包 GPU 运行结果 —— $STAMP"
echo "========================================================="
echo ""
echo "── 各目录体积 ──"
for d in out checkpoints gpu_logs runs; do
  if [ -d "$d" ]; then
    sz=$(du -sh "$d" 2>/dev/null | cut -f1 || echo "?")
    n=$(find "$d" -type f 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-14s %8s  %s 个文件\n" "$d/" "$sz" "$n"
  else
    printf "  %-14s %8s\n" "$d/" "(不存在)"
  fi
done
echo ""

# ── 完成情况：逐项核对 GPU_RUN_ALL.sh 计划里的产物 ──
EXPECTED=""
for f in out/slot_joint.sep4.qat4.scuocr out/slot_joint.sep34.qat4.scuocr \
         out/w_20_32_40_48.qat4.scuocr \
         out/sep4_wide.scuocr out/sep4_wide_shift.scuocr \
         out/slot_joint.full.scuocr; do
  EXPECTED="$EXPECTED $f"
done
for w in 20_32_40_40 20_28_40_40 16_24_32_32; do
  for p in sep4 sep34; do
    EXPECTED="$EXPECTED out/w${w}_${p}.scuocr out/w${w}_${p}.int4.scuocr"
  done
done
for w in 20_32_40_40 20_28_40_40 16_28_40_40 16_24_32_32 12_20_28_28; do
  EXPECTED="$EXPECTED out/d${w}.scuocr"
done

echo "── 计划产物完成情况 ──"
done_n=0; miss_n=0; MISSING=""
for f in $EXPECTED; do
  if [ -f "$f" ]; then
    kb=$(du -k "$f" 2>/dev/null | cut -f1)
    printf "  ✅ %-40s %6s KB\n" "$f" "$kb"
    done_n=$((done_n+1))
  else
    printf "  ⬜ %-40s (未产出)\n" "$f"
    miss_n=$((miss_n+1))
    MISSING="$MISSING $f"
  fi
done
echo ""
echo "  已产出 $done_n 项，未产出 $miss_n 项"
echo ""

echo "── checkpoint 训练进度（读 .pt 里的 epoch 字段）──"
FOUND_CKPTS=$("$PY" - <<'PYEOF' 2>/dev/null || true
import os, sys, glob
root = os.getcwd()
try:
    import torch
except Exception:
    print("  (无 torch，跳过)")
    raise SystemExit
for p in sorted(glob.glob(os.path.join(root, "checkpoints", "**", "*.pt"), recursive=True)):
    rel = os.path.relpath(p, root).replace("\\", "/")
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ep = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
        ba = ck.get("best_acc", "?") if isinstance(ck, dict) else "?"
        extra = f"epoch={ep}  best_acc={ba}"
    except Exception as e:
        extra = f"(读取失败: {type(e).__name__})"
    print(f"  {rel:44} {extra}")
PYEOF
)
if [ -n "$FOUND_CKPTS" ]; then echo "$FOUND_CKPTS"; else echo "  (无)"; fi
echo ""

if [ "$LIST_ONLY" -eq 1 ]; then
  echo "（--list 模式，未打包）"
  exit 0
fi

# ── 生成 MANIFEST ──
MANIFEST=$(mktemp)
{
  echo "zhjw 验证码模型 —— GPU 运行结果"
  echo "打包时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "主机: $(hostname 2>/dev/null || echo '?')"
  echo ""
  echo "=== 已产出（$done_n 项）==="
  for f in $EXPECTED; do [ -f "$f" ] && echo "  $f  ($(du -k "$f" 2>/dev/null | cut -f1) KB)"; done
  echo ""
  echo "=== 未产出（$miss_n 项）==="
  for f in $MISSING; do echo "  $f"; done
  echo ""
  echo "=== checkpoint 进度 ==="
  echo "$FOUND_CKPTS"
  echo ""
  echo "=== gpu_logs/SUMMARY.txt ==="
  [ -f gpu_logs/SUMMARY.txt ] && cat gpu_logs/SUMMARY.txt || echo "(无)"
  echo ""
  echo "=== results.tsv ==="
  [ -f gpu_logs/results.tsv ] && cat gpu_logs/results.tsv || echo "(无)"
  echo ""
  echo "=== 全部文件清单 ==="
  for d in out checkpoints gpu_logs runs; do
    [ -d "$d" ] || continue
    find "$d" -type f -printf "%10s  %p\n" 2>/dev/null | sort -k2
  done
} > "$MANIFEST"

# ── 用 Python 打包（不依赖 zip/tar 命令）──
ZIP="$NAME.zip"
"$PY" - "$ZIP" "$MANIFEST" "$INCLUDE_RUNS" <<'PYEOF'
import os, sys, zipfile

root = os.getcwd()
zip_path, manifest, include_runs = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
name = os.path.splitext(os.path.basename(zip_path))[0]
INCLUDE_DIRS = ["out", "checkpoints", "gpu_logs"] + (["runs"] if include_runs else [])
SKIP_EXT = {".pyc", ".pyo"}

n = 0
total = 0
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(manifest, f"{name}/MANIFEST.txt")
    n += 1
    for d in INCLUDE_DIRS:
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        for base, dirs, files in os.walk(full):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in files:
                if os.path.splitext(f)[1] in SKIP_EXT:
                    continue
                p = os.path.join(base, f)
                arc = os.path.join(name, os.path.relpath(p, root)).replace("\\", "/")
                z.write(p, arc)
                n += 1
                total += os.path.getsize(p)

size = os.path.getsize(zip_path)
print(f"\n✅ {zip_path}")
print(f"   {n} 个文件（原始 {total/1024/1024:.1f} MB）→ 压缩后 {size/1024/1024:.1f} MB")
PYEOF

rm -f "$MANIFEST"

echo ""
echo "── 下一步 ──"
echo "  1. 把这个 zip 拷回本地（scp / 网页下载均可）"
echo "  2. 若还想继续跑：解压回项目根目录后直接再跑"
echo "       bash GPU_RUN_ALL.sh"
echo "     已完成的会被自动跳过；被停机打断的训练会**从 latest.pt 断点续训**，"
echo "     不会从第 0 轮重来（前提是 checkpoints/ 原样拷回）。"
echo ""
ls -la "$ZIP" 2>/dev/null | sed 's/^/  /'
