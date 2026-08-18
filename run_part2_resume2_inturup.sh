#!/usr/bin/env bash
# ===========================================================================
#  PART 2 — RESUME #2  (เหลือแค่ 4 อย่าง)
#
#  สาเหตุที่ค้าง: ปิด VS Code ทำให้ terminal ที่เป็น parent ของ nohup หายไป
#  Python ตัวถัดไปเลย init stdout/stderr ไม่ได้ -> Errno 9 Bad file descriptor
#  ไม่ใช่บั๊กของโค้ด และงานที่รันไปแล้วไม่เสียหาย
#
#  จาก log: STEP 4b, 4c, 4d, 5 (ครบ 3 ประเทศ) และ STEP 6 เกือบครบ
#  ตัวสุดท้ายที่สำเร็จคือ outputs/v2/sweep/usa_M4_ft0.25 เมื่อ 17:31:05
#  ตัวที่ยังไม่ได้รันคือ usa_M3_ft0.25 ซึ่งเป็นคำสั่งถัดไปพอดี
#
#  วิธีรันแบบไม่ให้ค้างอีก (ใช้ tmux แทน nohup ผูกกับ VS Code):
#     tmux new -s p2
#     DEVICE=cpu bash run_part2_resume2.sh 2>&1 | tee part2c.log
#     กด Ctrl-B แล้ว D เพื่อ detach   ปิด VS Code ได้เลย
#     กลับมาดูด้วย  tmux attach -t p2
#
#  ถ้าไม่มี tmux:  brew install tmux
#  หรือใช้ setsid ตัดขาดจาก terminal:
#     DEVICE=cpu setsid nohup bash run_part2_resume2.sh > part2c.log 2>&1 < /dev/null &
# ===========================================================================
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

BRIDGE="${BRIDGE:-contextual}"
PAIRSET="outputs/pair_sets/insample"
OUT="outputs/v2/in_sample"
M4="$OUT/M4_${BRIDGE}_norm"
DEVICE="${DEVICE:-auto}"
CKPT="$M4/model_run1.pt"

[ -f "$CKPT" ] || { echo "ไม่พบ $CKPT"; exit 1; }

echo "############ RESUME #2 เริ่ม $(date) ############"

# ---------------------------------------------------------------------------
# ส่วนที่เหลือของ STEP 6 — มีตัวเดียว
# ---------------------------------------------------------------------------
if [ -f "outputs/v2/sweep/usa_M3_ft0.25/summary.json" ]; then
  echo "ข้าม usa_M3_ft0.25 (มีผลแล้ว)"
else
  echo ""
  echo "===== STEP 6 (ที่เหลือ): sweep usa M3 ratio=0.25 ====="
  python src/train_stage2_v2.py --pair_set_dir "outputs/pair_sets/fold_usa" \
    --model_type gatv2 --test_country usa \
    --fine_tune_ratio 0.25 --norm_scope all --num_runs 3 --seed 43 \
    --device "$DEVICE" --out_dir "outputs/v2/sweep/usa_M3_ft0.25"
fi

# ---------------------------------------------------------------------------
# STEP 7 — Threshold tuning
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 7: threshold tuned to recall >= 0.90 ====="
python src/train_stage2_v2.py --pair_set_dir "$PAIRSET" --model_type hybrid \
  --bridge_kind "$BRIDGE" --norm_scope all --tune_threshold --target_recall 0.90 \
  --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "$OUT/M4_recall90"

# ---------------------------------------------------------------------------
# STEP 8 — Post-hoc (เร็วมาก ไม่ต้องเทรน)
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 8: post-hoc analysis ====="
python src/posthoc_analysis.py --out_dir outputs/v2/analysis/posthoc \
  --model M4="$M4/test_probabilities.csv" \
  --model M4_mean="$OUT/M4_mean_norm/test_probabilities.csv" \
  --model M3="$OUT/M3_norm/test_probabilities.csv" \
  --model M2="$OUT/M2_norm/test_probabilities.csv" \
  --model MLP135="$OUT/mlp/test_probabilities.csv" \
  --model MLP7="$OUT/mlp_screens/test_probabilities.csv" \
  --model GCN="$OUT/gcn/test_probabilities.csv" \
  --model RF="$OUT/classical_full/RF_tuned/test_probabilities.csv" \
  --model HistGB="$OUT/classical_full/HistGB/test_probabilities.csv" \
  --compare M4:M3 --compare M4:RF --compare M4:HistGB \
  --compare M4:MLP135 --compare M3:MLP7 --compare M3:M2

# ---------------------------------------------------------------------------
# STEP 9 — Attention analysis
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 9: attention analysis ====="
python src/analyze_attention.py --pair_set_dir "$PAIRSET" --model_ckpt "$CKPT" \
  --bridge_kind "$BRIDGE" --norm_scope all \
  --pair_label_csv outputs/images/all_pairs_trainonly_labels.csv \
  --out_dir outputs/v2/analysis/attention \
  || echo "STEP 9 ล้มเหลว — ส่ง header ของ all_pairs_trainonly_labels.csv มาให้ผม"

echo ""
echo "############ เสร็จ $(date) ############"
echo ""
echo "ส่งไฟล์ให้ผม:"
echo "  zip -r v2_results.zip outputs/v2 outputs/pair_sets/*/coverage_stats.json \\"
echo "    -x '*.pt' '*.pth' '*.h5' '*.npy' '*.DS_Store'"
