#!/usr/bin/env bash
# ===========================================================================
#  PART 2 (แก้ไขหลังเห็นผล PART 1)
#
#     bash run_part2.sh
#  หรือ
#     DEVICE=cpu nohup bash run_part2.sh > part2.log 2>&1 &
#
#  เปลี่ยนจากฉบับก่อน 3 จุด:
#    A) เพิ่ม STEP 0b  ทดสอบว่ากราฟอย่างเดียวทำนาย label ได้แค่ไหน
#    B) เพิ่ม M3 (screens 7 ตัว) ลงใน LOCO และ sweep ทุกเงื่อนไข
#       จำเป็น เพราะ PART 1 บอกว่า in-sample M4 ไม่ต่างจาก M3 เลย
#       ถ้าไม่รัน M3 ข้ามตลาดด้วย จะพิสูจน์ไม่ได้ว่า visual embedding
#       มีประโยชน์ตรงไหน
#    C) ยืนยัน BRIDGE=contextual (ทั้งสามแบบไม่ต่างกันทางสถิติ p > 0.32)
#
#  เวลารวมประมาณ 10-11 ชั่วโมง
# ===========================================================================
set -euo pipefail

# macOS OpenMP guard: PyTorch, scikit-learn and XGBoost each ship their own
# libomp.dylib. Loading two of them into one process segfaults on Apple Silicon.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

BRIDGE="${BRIDGE:-contextual}"
PAIRSET="outputs/pair_sets/insample"
OUT="outputs/v2/in_sample"
M4="$OUT/M4_${BRIDGE}_norm"
DEVICE="${DEVICE:-auto}"
CKPT="$M4/model_run1.pt"
FEATS="outputs/v2/learned_node_features.pt"

[ -f "$CKPT" ] || { echo "ไม่พบ $CKPT"; exit 1; }
echo "############ PART 2 เริ่ม $(date) | bridge=$BRIDGE ############"

# ---------------------------------------------------------------------------
# STEP 0b — กราฟอย่างเดียวทำนาย label ได้แค่ไหน (ไม่ใช้ feature เลย)
#   ใช้เวลาไม่กี่นาที แต่เป็นตัวชี้ขาดว่าเปเปอร์ต้องเขียน narrative แบบไหน
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 0b: graph-only baseline (ตัวชี้ขาด) ====="
python src/label_propagation_check.py --pair_set_dir "$PAIRSET" \
  --out_dir outputs/v2/analysis/graph_leakage

# ---------------------------------------------------------------------------
# STEP 4 — features + baselines
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 4a: export learned node features ====="
python src/export_node_features.py --pair_set_dir "$PAIRSET" \
  --model_ckpt "$CKPT" --bridge_kind "$BRIDGE" --norm_scope all --out "$FEATS"

echo ""
echo "===== STEP 4b: classical baselines ====="
python src/evaluate_baseline_v2.py --node_feature_pt "$FEATS" --n_runs 5 \
  --out_dir "$OUT/classical_full"
python src/evaluate_baseline_v2.py --node_feature_pt "$FEATS" --n_runs 5 \
  --screens_only --out_dir "$OUT/classical_screens"

echo ""
echo "===== STEP 4c: GNN baselines (mlp = ตัวสำคัญ) ====="
for M in gcn sage mlp; do
  python src/train_stage2_v2.py --pair_set_dir "$PAIRSET" --model_type "$M" \
    --node_feature_pt "$FEATS" --norm_scope none --num_runs 5 --seed 43 \
    --device "$DEVICE" --out_dir "$OUT/$M"
done

echo ""
echo "===== STEP 4d: MLP บน 7 screens (ไม่มีกราฟ) ====="
python src/train_stage2_v2.py --pair_set_dir "$PAIRSET" --model_type mlp \
  --norm_scope all --num_runs 5 --seed 43 --device "$DEVICE" \
  --out_dir "$OUT/mlp_screens"

# ---------------------------------------------------------------------------
# STEP 5 — Cross-market LOCO: M4 และ M3 คู่กันทุกเงื่อนไข
# ---------------------------------------------------------------------------
for C in brazil japan usa; do
  D="outputs/pair_sets/fold_$C"
  echo ""
  echo "===== STEP 5: LOCO target=$C ====="

  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type hybrid \
    --bridge_kind "$BRIDGE" --test_country "$C" --norm_scope none \
    --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "outputs/v2/loco/${C}_M4_C1"
  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type hybrid \
    --bridge_kind "$BRIDGE" --test_country "$C" --norm_scope screens \
    --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "outputs/v2/loco/${C}_M4_C2"
  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type hybrid \
    --bridge_kind "$BRIDGE" --test_country "$C" --norm_scope all \
    --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "outputs/v2/loco/${C}_M4_C2b"
  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type hybrid \
    --bridge_kind "$BRIDGE" --test_country "$C" --fine_tune_ratio 0.15 \
    --norm_scope all --num_runs 5 --seed 43 --device "$DEVICE" \
    --out_dir "outputs/v2/loco/${C}_M4_C3"

  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type gatv2 \
    --test_country "$C" --norm_scope all \
    --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "outputs/v2/loco/${C}_M3_C2b"
  python src/train_stage2_v2.py --pair_set_dir "$D" --model_type gatv2 \
    --test_country "$C" --fine_tune_ratio 0.15 --norm_scope all \
    --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "outputs/v2/loco/${C}_M3_C3"
done

# ---------------------------------------------------------------------------
# STEP 6 — Few-shot sweep, M4 และ M3 คู่กัน (0.15 ใช้ผลจาก C3)
# ---------------------------------------------------------------------------
for C in brazil japan usa; do
  for R in 0.05 0.10 0.20 0.25; do
    echo ""
    echo "===== STEP 6: sweep $C ratio=$R ====="
    python src/train_stage2_v2.py --pair_set_dir "outputs/pair_sets/fold_$C" \
      --model_type hybrid --bridge_kind "$BRIDGE" --test_country "$C" \
      --fine_tune_ratio "$R" --norm_scope all --num_runs 3 --seed 43 \
      --device "$DEVICE" --out_dir "outputs/v2/sweep/${C}_M4_ft${R}"
    python src/train_stage2_v2.py --pair_set_dir "outputs/pair_sets/fold_$C" \
      --model_type gatv2 --test_country "$C" \
      --fine_tune_ratio "$R" --norm_scope all --num_runs 3 --seed 43 \
      --device "$DEVICE" --out_dir "outputs/v2/sweep/${C}_M3_ft${R}"
  done
done

# ---------------------------------------------------------------------------
# STEP 7 — Threshold tuning
# ---------------------------------------------------------------------------
echo ""
echo "===== STEP 7: threshold tuned to recall >= 0.90 ====="
python src/train_stage2_v2.py --pair_set_dir "$PAIRSET" --model_type hybrid \
  --bridge_kind "$BRIDGE" --norm_scope all --tune_threshold --target_recall 0.90 \
  --num_runs 5 --seed 43 --device "$DEVICE" --out_dir "$OUT/M4_recall90"

# ---------------------------------------------------------------------------
# STEP 8 — Post-hoc
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
echo "############ PART 2 เสร็จ $(date) ############"
echo "zip -r v2_results.zip outputs/v2 outputs/pair_sets/*/coverage_stats.json \\"
echo "  -x '*.pt' '*.pth' '*.h5' '*.npy' '*.DS_Store'"
