#!/usr/bin/env bash
# ===========================================================================
#  PART 2 — รันหลังจากผมยืนยันผล PART 1 แล้วเท่านั้น
#
#  ก่อนรัน: แก้บรรทัด BRIDGE ข้างล่างตามที่ผมบอก
#     bash run_part2.sh
#  หรือ:
#     BRIDGE=uncond nohup bash run_part2.sh > part2.log 2>&1 &
#
#  ใช้เวลารวมประมาณ 8–9 ชั่วโมง (ปล่อยข้ามคืน)
# ===========================================================================
set -euo pipefail

# >>>>>>>>>>>>>>>>>>>>  ตั้งค่าตรงนี้  <<<<<<<<<<<<<<<<<<<<
BRIDGE="${BRIDGE:-contextual}"      # contextual | uncond | mean
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

PAIRSET="outputs/pair_sets/insample"
OUT="outputs/v2/in_sample"
M4="$OUT/M4_${BRIDGE}_norm"
DEVICE="${DEVICE:-auto}"
CKPT="$M4/model_run1.pt"
FEATS="outputs/v2/learned_node_features.pt"

if [ ! -f "$CKPT" ]; then
  echo "ไม่พบ $CKPT — รัน run_part1.sh ให้เสร็จก่อน หรือตั้ง BRIDGE ผิด"
  exit 1
fi

echo "############ PART 2 เริ่ม $(date) | bridge=$BRIDGE ############"

# ---------------------------------------------------------------------------
# STEP 4a — ดึง node features 135 มิติจากโมเดลที่เทรนแล้ว
#   --bridge_kind และ --norm_scope ต้องตรงกับตอนเทรน ไม่งั้น state_dict โหลดไม่ขึ้น
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 4a: export learned node features ====="
python src/export_node_features.py \
  --pair_set_dir "$PAIRSET" --model_ckpt "$CKPT" \
  --bridge_kind "$BRIDGE" --norm_scope all \
  --out "$FEATS"

# ---------------------------------------------------------------------------
# STEP 4b — classical baselines (LR scaled / RF default / RF tuned / XGBoost)
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 4b: classical baselines (135-dim) ====="
python src/evaluate_baseline_v2.py \
  --node_feature_pt "$FEATS" --n_runs 5 \
  --out_dir "$OUT/classical_full"

echo ""
echo "===== STEP 4b: classical baselines (7 screens) ====="
python src/evaluate_baseline_v2.py \
  --node_feature_pt "$FEATS" --n_runs 5 --screens_only \
  --out_dir "$OUT/classical_screens"

# ---------------------------------------------------------------------------
# STEP 4c — GNN baselines บน feature ชุดเดียวกัน
#   mlp คือตัวสำคัญ: ถ้ามันเสมอ M4 แปลว่ากราฟไม่ได้ช่วย
# ---------------------------------------------------------------------------
for M in gcn sage mlp; do
  echo ""
  echo "===== STEP 4c: baseline $M ====="
  python src/train_stage2_v2.py \
    --pair_set_dir "$PAIRSET" --model_type "$M" \
    --node_feature_pt "$FEATS" --norm_scope none \
    --num_runs 5 --seed 43 --device "$DEVICE" \
    --out_dir "$OUT/$M"
done

# ---------------------------------------------------------------------------
# STEP 5 — Cross-market LOCO, 4 เงื่อนไข
#   C1  zero-shot ไม่ normalise
#   C2  zero-shot normalise แค่ screens  (= ของเดิมในเปเปอร์)
#   C2b zero-shot normalise ครบ 135 มิติ (ของใหม่ ที่ควรจะเป็น)
#   C3  few-shot 15% + normalise ครบ
# ---------------------------------------------------------------------------
for C in brazil japan usa; do
  D="outputs/pair_sets/fold_$C"
  echo ""
  echo "===== STEP 5: LOCO target=$C ====="
  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type hybrid \
    --bridge_kind "$BRIDGE" --test_country "$C" --norm_scope none \
    --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "outputs/v2/loco/${C}_C1"

  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type hybrid \
    --bridge_kind "$BRIDGE" --test_country "$C" --norm_scope screens \
    --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "outputs/v2/loco/${C}_C2"

  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type hybrid \
    --bridge_kind "$BRIDGE" --test_country "$C" --norm_scope all \
    --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "outputs/v2/loco/${C}_C2b"

  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type hybrid \
    --bridge_kind "$BRIDGE" --test_country "$C" --fine_tune_ratio 0.15 \
    --norm_scope all --num_runs 5 --seed 43 --device "$DEVICE" \
    --out_dir "outputs/v2/loco/${C}_C3"
done

# ---------------------------------------------------------------------------
# STEP 6 — Few-shot sweep (ตอบ future work ของเปเปอร์เอง)
#   3 seeds ต่อจุดเพื่อประหยัดเวลา
# ---------------------------------------------------------------------------
for C in brazil japan usa; do
  for R in 0.05 0.10 0.20 0.25; do
    echo ""
    echo "===== STEP 6: sweep $C ratio=$R ====="
    python src/train_stage2_v2.py \
      --pair_set_dir "outputs/pair_sets/fold_$C" --model_type hybrid \
      --bridge_kind "$BRIDGE" --test_country "$C" --fine_tune_ratio "$R" \
      --norm_scope all --num_runs 3 --seed 43 --device "$DEVICE" \
      --out_dir "outputs/v2/sweep/${C}_ft${R}"
  done
done
# หมายเหตุ: ratio 0.15 ไม่ต้องรันซ้ำ ใช้ผลจาก STEP 5 C3 ได้เลย

# ---------------------------------------------------------------------------
# STEP 7 — Threshold tuning (แทนการใช้ 0.5 แบบไม่มีเหตุผล)
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 7: threshold tuned to recall >= 0.90 ====="
python src/train_stage2_v2.py \
  --pair_set_dir "$PAIRSET" --model_type hybrid --bridge_kind "$BRIDGE" \
  --norm_scope all --tune_threshold --target_recall 0.90 \
  --num_runs 5 --seed 43 --device "$DEVICE" \
  --out_dir "$OUT/M4_recall90"

# ---------------------------------------------------------------------------
# STEP 8 — Post-hoc (ไม่ต้องเทรนใหม่ เร็วมาก)
#   นี่คือคำตอบของ "ทำไมไม่ใช้ RF ไปเลย"
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 8: post-hoc analysis ====="
python src/posthoc_analysis.py --out_dir outputs/v2/analysis/posthoc \
  --model M4="$M4/test_probabilities.csv" \
  --model M3="$OUT/M3_norm/test_probabilities.csv" \
  --model M2="$OUT/M2_norm/test_probabilities.csv" \
  --model MLP="$OUT/mlp/test_probabilities.csv" \
  --model GCN="$OUT/gcn/test_probabilities.csv" \
  --model RF="$OUT/classical_full/RF_tuned/test_probabilities.csv" \
  --model XGB="$OUT/classical_full/XGB/test_probabilities.csv" \
  --compare M4:RF --compare M4:XGB --compare M4:M3 --compare M4:MLP

# ---------------------------------------------------------------------------
# STEP 9 — Attention analysis
#   ถ้า error เรื่องคอลัมน์ ให้ข้ามไปก่อนแล้วส่ง header ของ CSV มาให้ผม
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 9: attention analysis ====="
python src/analyze_attention.py \
  --pair_set_dir "$PAIRSET" --model_ckpt "$CKPT" \
  --bridge_kind "$BRIDGE" --norm_scope all \
  --pair_label_csv outputs/images/all_pairs_trainonly_labels.csv \
  --out_dir outputs/v2/analysis/attention \
  || echo "STEP 9 ล้มเหลว — ส่ง header ของ all_pairs_trainonly_labels.csv มาให้ผม แล้วข้ามไปก่อน"

echo ""
echo "############ PART 2 เสร็จ $(date) ############"
echo "ส่งไฟล์ให้ผม:"
echo "  zip -r v2_results.zip outputs/v2 outputs/pair_sets/*/coverage_stats.json \\"
echo "    -x '*.pt' '*.pth' '*.h5' '*.npy' '*.DS_Store'"
