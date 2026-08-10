#!/usr/bin/env bash
# ===========================================================================
#  PART 1 — รันแล้วหยุด แล้วส่งผลให้ผมดูก่อนไป PART 2
#
#  วิธีใช้:
#     cd hybrid_gat_cnn
#     bash run_part1.sh
#
#  ถ้าอยากรันข้ามคืนแล้วเก็บ log:
#     nohup bash run_part1.sh > part1.log 2>&1 &
#
#  ทุกคำสั่งใส่ --norm_scope all ซึ่งเป็นจุดที่ขาดไปในรอบที่แล้ว
#  ทำให้ contextual bridge เทรนไม่เสถียร (seed 47 พังไป)
# ===========================================================================
set -euo pipefail

PAIRSET="outputs/pair_sets/insample"
OUT="outputs/v2/in_sample"
DEVICE="${DEVICE:-auto}"     # ถ้า MPS OOM ให้รัน:  DEVICE=cpu bash run_part1.sh

if [ ! -d "$PAIRSET" ]; then
  echo "ไม่พบ $PAIRSET — ต้องรัน STEP 0 (prepare_pair_sets.py) ก่อน"
  exit 1
fi

echo "############ PART 1 เริ่ม $(date) ############"
echo "device = $DEVICE"

# ---------------------------------------------------------------------------
# 1) Bridge ablation — 3 แบบ, 10 seeds, normalise แล้ว
#    10 seeds เพราะ 5 seeds ให้ statistical power ไม่พอ:
#    รอบที่แล้ว uncond vs meanpool ต่างกัน 0.087 แต่ p = 0.142
# ---------------------------------------------------------------------------
for B in contextual uncond mean; do
  echo ""
  echo "===== [1/3] bridge = $B (10 seeds) ====="
  python src/train_stage2_v2.py \
    --pair_set_dir "$PAIRSET" \
    --model_type hybrid --bridge_kind "$B" \
    --norm_scope all --num_runs 10 --seed 43 \
    --save_attention --device "$DEVICE" \
    --out_dir "$OUT/M4_${B}_norm"
done

# ---------------------------------------------------------------------------
# 2) M2 / M3 รันใหม่ด้วยเงื่อนไขเดียวกัน
#    ของเดิมรันแบบไม่ normalise จึงเทียบกับ M4 ไม่ได้
# ---------------------------------------------------------------------------
echo ""
echo "===== [2/3] M2 (SimpleGAT / GATv1) ====="
python src/train_stage2_v2.py \
  --pair_set_dir "$PAIRSET" --model_type simple_gat \
  --norm_scope all --num_runs 5 --seed 43 --device "$DEVICE" \
  --out_dir "$OUT/M2_norm"

echo ""
echo "===== [3/3] M3 (GATv2, screens only) ====="
python src/train_stage2_v2.py \
  --pair_set_dir "$PAIRSET" --model_type gatv2 \
  --norm_scope all --num_runs 5 --seed 43 --device "$DEVICE" \
  --out_dir "$OUT/M3_norm"

# ---------------------------------------------------------------------------
# สรุปผลย่อให้ดูทันทีว่า contextual เสถียรขึ้นหรือยัง
# ---------------------------------------------------------------------------
echo ""
echo "############ สรุป PART 1 ############"
python - <<'PY'
import json, glob, os
import pandas as pd

rows = []
for d in sorted(glob.glob('outputs/v2/in_sample/*_norm')):
    f = os.path.join(d, 'results.csv')
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    ok = df[df.best_val_f1 >= 0.80]
    rows.append({
        'model': os.path.basename(d),
        'runs': len(df),
        'converged': f"{len(ok)}/{len(df)}",
        'F1': f"{df.f1.mean():.4f}±{df.f1.std(ddof=1):.4f}",
        'F1_converged': f"{ok.f1.mean():.4f}" if len(ok) else '-',
        'PR_AUC': f"{df.pr_auc.mean():.4f}",
        'ROC_AUC': f"{df.roc_auc.mean():.4f}",
        'recall': f"{df.recall.mean():.4f}",
    })
print(pd.DataFrame(rows).to_string(index=False))
print()
print("เกณฑ์ตัดสิน:")
print("  contextual converged >= 9/10  ->  ใช้ BRIDGE=contextual ใน PART 2")
print("  contextual converged <  9/10  ->  หยุด ส่งผลให้ผมดูก่อน อย่าเพิ่งรัน PART 2")
PY

echo ""
echo "############ PART 1 เสร็จ $(date) ############"
echo "ส่งไฟล์เหล่านี้ให้ผม:"
echo "  zip -r part1.zip outputs/v2/in_sample/*_norm --include '*.csv' '*.json' '*.log'"
