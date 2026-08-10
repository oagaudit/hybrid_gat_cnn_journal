"""
src/analyze_attention.py  --  NEW FILE

Turns the trained Bridge attention into evidence. The current paper lists the
black-box issue as a limitation; for a computational-social-science audience it
is the most interesting output the model produces, because it answers the
question an investigator would actually ask: *which bidder pair made this tender
look collusive?*

Three outputs:

  1. attention_by_pair.csv
     Per pair: attention weight, whether both firms are verified cartel members,
     the tender label, and the model's predicted probability.

  2. validation_summary.json
     Does attention concentrate on genuinely collusive pairs? Reports, over
     flagged tenders, the share of attention mass falling on verified cartel
     pairs versus the share those pairs represent by count. A ratio above 1
     means the module learned something real; a ratio near 1 means the attention
     is uninformative and the paper should say so rather than claim otherwise.

  3. attention_concentration.csv
     Entropy of the attention distribution per tender, split by predicted class.
     Concentrated attention on collusive tenders and diffuse attention on
     competitive ones is the pattern the paper's argument predicts.

Usage:
    python src/analyze_attention.py \
        --pair_set_dir outputs/pair_sets/insample \
        --model_ckpt outputs/models_stage2_v2/in_sample/hybrid/model_run1.pt \
        --out_dir outputs/analysis/attention
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hybrid_model import HybridGATv2Model      # noqa: E402
from src.train_stage2_v2 import (load_and_merge,          # noqa: E402
                                 build_masks, apply_normalisation)


def cartel_pair_flags(pair_keys, cartel_csv):
    """Mark pairs where BOTH firms are verified cartel members.

    Expects a CSV with columns country,firm_id for verified cartel members
    (equation 3 in the paper). If absent, falls back to the pair-level labels
    already stored alongside the CNN embeddings.
    """
    members = set()
    df = pd.read_csv(cartel_csv)
    for _, r in df.iterrows():
        members.add(f"{str(r['country']).lower()}_{int(r['firm_id'])}")
    flags = []
    for k in pair_keys:
        parts = k.split('_')
        c, a, b = parts[0], parts[-2], parts[-1]
        flags.append(int(f"{c}_{a}" in members and f"{c}_{b}" in members))
    return np.array(flags)


def entropy(weights):
    w = np.clip(weights, 1e-12, None)
    w = w / w.sum()
    return float(-(w * np.log(w)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_set_dir', required=True)
    ap.add_argument('--model_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--pair_label_csv', default=None,
                    help='CSV with pair_key,label from the CNN stage (fallback '
                         'ground truth for whether a pair is a cartel pair)')
    ap.add_argument('--cartel_csv', default=None,
                    help='CSV with country,firm_id of verified cartel members')
    ap.add_argument('--visual_dim', type=int, default=128)
    ap.add_argument('--bridge_kind', default='contextual',
                    choices=['contextual', 'uncond', 'mean'],
                    help='MUST match the trained checkpoint.')
    ap.add_argument('--norm_scope', default='all',
                    choices=['none', 'screens', 'all'],
                    help='MUST match training.')
    ap.add_argument('--flag_threshold', type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_and_merge(args.pair_set_dir)
    import types
    _m = types.SimpleNamespace(test_country=None, fine_tune_ratio=0.0,
                               val_ratio=0.15, seed=43)
    _, _, _, norm_fit = build_masks(data, _m)
    apply_normalisation(data, norm_fit, args.norm_scope)

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

    if args.cartel_csv:
        is_cartel_pair = cartel_pair_flags(data['pair_key'], args.cartel_csv)
    elif args.pair_label_csv:
        lab = pd.read_csv(args.pair_label_csv).set_index('pair_key')['label'].to_dict()
        is_cartel_pair = np.array([int(lab.get(k, 0)) for k in data['pair_key']])
    else:
        raise SystemExit("Provide --cartel_csv or --pair_label_csv so attention "
                         "can be validated against ground truth.")

    df = pd.DataFrame({
        'pair_key': data['pair_key'],
        'node_idx': node_idx,
        'tender_id': [data['tender_ids'][i] for i in node_idx],
        'country': data['node_country'][node_idx],
        'attention': attn,
        'is_cartel_pair': is_cartel_pair,
        'tender_label': y[node_idx],
        'pred_prob': prob[node_idx],
    })
    df.to_csv(os.path.join(args.out_dir, 'attention_by_pair.csv'), index=False)

    # ---- Does attention land on cartel pairs more than chance? ----
    summary = {}
    for scope, sub in [('all_tenders', df),
                       ('flagged_tenders', df[df.pred_prob >= args.flag_threshold]),
                       ('true_collusive', df[df.tender_label == 1])]:
        if len(sub) == 0:
            continue
        mass = sub.loc[sub.is_cartel_pair == 1, 'attention'].sum() / max(sub['attention'].sum(), 1e-12)
        share = sub['is_cartel_pair'].mean()
        summary[scope] = {
            'n_pairs': int(len(sub)),
            'cartel_pair_share_by_count': float(share),
            'cartel_pair_share_of_attention_mass': float(mass),
            'concentration_ratio': float(mass / share) if share > 0 else None,
        }
    for c in sorted(df['country'].unique()):
        sub = df[(df.country == c) & (df.tender_label == 1)]
        if len(sub) == 0 or sub['is_cartel_pair'].mean() == 0:
            continue
        mass = sub.loc[sub.is_cartel_pair == 1, 'attention'].sum() / max(sub['attention'].sum(), 1e-12)
        summary[f'true_collusive_{c}'] = {
            'n_pairs': int(len(sub)),
            'cartel_pair_share_by_count': float(sub['is_cartel_pair'].mean()),
            'cartel_pair_share_of_attention_mass': float(mass),
            'concentration_ratio': float(mass / sub['is_cartel_pair'].mean()),
        }

    with open(os.path.join(args.out_dir, 'validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Attention entropy per tender ----
    rows = []
    for nid, g in df.groupby('node_idx'):
        if len(g) < 2:
            continue
        rows.append({'node_idx': nid, 'tender_id': g['tender_id'].iloc[0],
                     'country': g['country'].iloc[0], 'n_pairs': len(g),
                     'entropy': entropy(g['attention'].values),
                     'max_attention': float(g['attention'].max()),
                     'normalised_entropy': entropy(g['attention'].values) / np.log(len(g)),
                     'tender_label': int(g['tender_label'].iloc[0]),
                     'pred_prob': float(g['pred_prob'].iloc[0])})
    ent = pd.DataFrame(rows)
    ent.to_csv(os.path.join(args.out_dir, 'attention_concentration.csv'), index=False)

    print(json.dumps(summary, indent=2))
    if len(ent):
        print("\nNormalised attention entropy by true label "
              "(lower = more concentrated):")
        print(ent.groupby('tender_label')['normalised_entropy']
              .agg(['count', 'mean', 'std']).to_string())
    print(f"\nWrote 3 files to {args.out_dir}")


if __name__ == '__main__':
    main()
