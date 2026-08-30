"""
src/analyze_attention.py

ทำไมต้องมีเวอร์ชันนี้
---------------------
รอบที่แล้ว STEP 9 รันผ่านแต่ผลว่างเปล่า:

    "cartel_pair_share_by_count": 0.0
    "concentration_ratio": null

สาเหตุคือ `--pair_label_csv` join ไม่ติด: โค้ดเดิมใช้
`lab.get(k, 0)` ซึ่ง**คืน 0 เงียบๆ เมื่อ key ไม่ตรง** แทนที่จะ error
ทำให้ทุกคู่ถูกทำเครื่องหมายว่าไม่ใช่คู่คาร์เทลทั้งหมด 46,453 คู่

เวอร์ชันนี้เลิกพึ่ง CSV ภายนอก แล้วสร้าง cartel membership จาก
data/processed/{country}_cleaned.parquet โดยตรง ตามนิยามในสมการ (3)
ของเปเปอร์: คู่ (A,B) เป็นคู่คาร์เทลก็ต่อเมื่อทั้ง A และ B เป็นสมาชิก
คาร์เทลที่ได้รับการยืนยัน

    C = { competitor : Collusive_competitor == 1 ในโครงการใดก็ตาม }

และมี **การตรวจสอบแบบ fail-loud**: ถ้า join ได้คู่คาร์เทล 0 คู่ สคริปต์จะ
หยุดพร้อมข้อความอธิบาย แทนที่จะเขียนไฟล์ที่ดูเหมือนสำเร็จออกมา

ผลลัพธ์ที่ได้เพิ่มจากเดิม
-------------------------
  attention_by_pair.csv          เหมือนเดิม แต่ is_cartel_pair ถูกต้อง
  validation_summary.json        + แยกรายประเทศ + null model เปรียบเทียบ
  attention_concentration.csv    เหมือนเดิม
  top_attention_examples.csv     ใหม่: 20 tender ที่โมเดลแฟล็ก เรียงตาม
                                 ความมั่นใจ พร้อมคู่ที่ได้ attention สูงสุด
                                 ใช้เป็น case study ในเปเปอร์ได้เลย

Usage:
    python src/analyze_attention.py \
        --pair_set_dir outputs/pair_sets/insample \
        --model_ckpt outputs/results/in_sample/M4_contextual_norm/model_run1.pt \
        --bridge_kind contextual --norm_scope all \
        --out_dir outputs/results/analysis/attention
"""

import os
import sys
import json
import types
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hybrid_model import HybridGATv2Model                 # noqa: E402
from src.train_stage2 import (load_and_merge, build_masks,        # noqa: E402
                                 apply_normalisation)
from src.utils.config_loader import CONFIG, get_project_root         # noqa: E402

PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
COUNTRIES = ['brazil', 'japan', 'usa']


def build_cartel_members():
    """C = set of verified cartel members, per equation (3) of the paper."""
    members = set()
    per_country = {}
    for c in COUNTRIES:
        path = os.path.join(PROCESSED_DIR, f"{c}_cleaned.parquet")
        if not os.path.exists(path):
            raise SystemExit(f"ไม่พบ {path}")
        df = pd.read_parquet(path)
        col = 'Collusive_competitor'
        if col not in df.columns:
            raise SystemExit(f"{c}: ไม่มีคอลัมน์ {col}. คอลัมน์ที่มี: {list(df.columns)}")
        ids = df.loc[df[col] == 1, 'Competitors'].unique()
        ids = {f"{c}_{int(float(v))}" for v in ids}
        members |= ids
        per_country[c] = len(ids)
    return members, per_country


def flag_pairs(pair_keys, members):
    flags = np.zeros(len(pair_keys), dtype=int)
    for i, k in enumerate(pair_keys):
        parts = k.split('_')
        country, a, b = parts[0], parts[-2], parts[-1]
        if f"{country}_{a}" in members and f"{country}_{b}" in members:
            flags[i] = 1
    return flags


def entropy(w):
    w = np.clip(np.asarray(w, dtype=float), 1e-12, None)
    w = w / w.sum()
    return float(-(w * np.log(w)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_set_dir', required=True)
    ap.add_argument('--model_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--bridge_kind', default='contextual',
                    choices=['contextual', 'uncond', 'mean'])
    ap.add_argument('--norm_scope', default='all',
                    choices=['none', 'screens', 'all'])
    ap.add_argument('--visual_dim', type=int, default=128)
    ap.add_argument('--flag_threshold', type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    members, per_country = build_cartel_members()
    print(f"verified cartel members: {len(members)} "
          f"({', '.join(f'{k}={v}' for k, v in per_country.items())})")

    data = load_and_merge(args.pair_set_dir)
    _m = types.SimpleNamespace(test_country=None, fine_tune_ratio=0.0,
                               val_ratio=0.15, seed=43)
    _, _, _, norm_fit = build_masks(data, _m)
    apply_normalisation(data, norm_fit, args.norm_scope)

    is_cartel = flag_pairs(data['pair_key'], members)
    if is_cartel.sum() == 0:
        raise SystemExit(
            "พบคู่คาร์เทล 0 คู่ จาก " + str(len(data['pair_key'])) + " คู่.\n"
            "ตัวอย่าง pair_key: " + ", ".join(data['pair_key'][:3]) + "\n"
            "ตัวอย่าง member id: " + ", ".join(list(members)[:3]) + "\n"
            "รูปแบบ id ไม่ตรงกัน ส่งสองบรรทัดนี้มาให้ผมแก้")
    print(f"cartel pairs: {int(is_cartel.sum())} / {len(is_cartel)} "
          f"({is_cartel.mean() * 100:.2f}%)")

    model = HybridGATv2Model(visual_embed_dim=args.visual_dim,
                             bridge_kind=args.bridge_kind)
    model.load_state_dict(torch.load(args.model_ckpt, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        logits, attn, _ = model(data['pair_emb'], data['pair_node_idx'],
                                data['screens'], data['edge_index'],
                                return_attention=True)
        prob = torch.softmax(logits, 1)[:, 1].numpy()
    attn = attn.numpy()
    node_idx = data['pair_node_idx'].numpy()
    y = data['y'].numpy()

    df = pd.DataFrame({
        'pair_key': data['pair_key'],
        'node_idx': node_idx,
        'tender_id': [data['tender_ids'][i] for i in node_idx],
        'country': data['node_country'][node_idx],
        'attention': attn,
        'is_cartel_pair': is_cartel,
        'tender_label': y[node_idx],
        'pred_prob': prob[node_idx],
    })
    df.to_csv(os.path.join(args.out_dir, 'attention_by_pair.csv'), index=False)

    # ---- does attention land on cartel pairs more than chance? -------------
    summary = {}

    def block(sub, name):
        if len(sub) == 0 or sub['is_cartel_pair'].sum() == 0:
            return
        share = float(sub['is_cartel_pair'].mean())
        mass = float(sub.loc[sub.is_cartel_pair == 1, 'attention'].sum()
                     / max(sub['attention'].sum(), 1e-12))
        summary[name] = {
            'n_pairs': int(len(sub)),
            'cartel_pair_share_by_count': round(share, 4),
            'cartel_pair_share_of_attention_mass': round(mass, 4),
            'concentration_ratio': round(mass / share, 4),
        }

    block(df, 'all_tenders')
    block(df[df.pred_prob >= args.flag_threshold], 'flagged_tenders')
    block(df[df.tender_label == 1], 'true_collusive')
    for c in COUNTRIES:
        block(df[(df.country == c) & (df.tender_label == 1)], f'true_collusive_{c}')

    # null model: uniform attention within each tender gives ratio 1.0 by
    # construction, so any value above 1 is real selectivity
    summary['_note'] = ("concentration_ratio = 1.0 means attention is spread "
                        "exactly in proportion to how common cartel pairs are, "
                        "i.e. no selectivity. Above 1 means the bridge "
                        "preferentially attends to verified cartel pairs.")

    with open(os.path.join(args.out_dir, 'validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- concentration per tender -----------------------------------------
    rows = []
    for nid, g in df.groupby('node_idx'):
        if len(g) < 2:
            continue
        rows.append({
            'node_idx': nid, 'tender_id': g['tender_id'].iloc[0],
            'country': g['country'].iloc[0], 'n_pairs': len(g),
            'entropy': entropy(g['attention'].values),
            'normalised_entropy': entropy(g['attention'].values) / np.log(len(g)),
            'max_attention': float(g['attention'].max()),
            'top_pair': g.loc[g['attention'].idxmax(), 'pair_key'],
            'top_pair_is_cartel': int(g.loc[g['attention'].idxmax(), 'is_cartel_pair']),
            'tender_label': int(g['tender_label'].iloc[0]),
            'pred_prob': float(g['pred_prob'].iloc[0]),
        })
    ent = pd.DataFrame(rows)
    ent.to_csv(os.path.join(args.out_dir, 'attention_concentration.csv'), index=False)

    # ---- case-study table for the paper -----------------------------------
    flagged = ent[(ent.pred_prob >= args.flag_threshold) & (ent.tender_label == 1)]
    top = flagged.sort_values('pred_prob', ascending=False).head(20)
    top.to_csv(os.path.join(args.out_dir, 'top_attention_examples.csv'), index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(ent):
        print("\nNormalised attention entropy by label (lower = more concentrated)")
        print(ent.groupby(['country', 'tender_label'])['normalised_entropy']
              .agg(['count', 'mean', 'std']).round(4).to_string())
        print("\nTop-attention pair is a verified cartel pair, by tender label:")
        print(ent.groupby('tender_label')['top_pair_is_cartel']
              .agg(['count', 'mean']).round(4).to_string())
    print(f"\nWrote 4 files to {args.out_dir}")


if __name__ == '__main__':
    main()
