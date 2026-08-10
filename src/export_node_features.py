"""
src/export_node_features.py  --  NEW FILE

Exports the 135-dim node features the TRAINED bridge produces, plus the exact
train/val/test masks, so the classical baselines are compared on the same
features the graph model actually sees.

This matters for fairness in both directions. In the original setup the 135-dim
features fed to Random Forest came from an untrained random projection, so
neither model was being evaluated on the representation the paper describes.

Usage:
    python src/export_node_features.py \
        --pair_set_dir outputs/pair_sets/insample \
        --model_ckpt outputs/models_stage2_v2/in_sample/hybrid/model_run1.pt \
        --out outputs/analysis/learned_node_features.pt
"""

import os
import sys
import argparse
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hybrid_model import HybridGATv2Model         # noqa: E402
from src.train_stage2_v2 import (load_and_merge, build_masks,  # noqa: E402
                                 apply_normalisation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_set_dir', required=True)
    ap.add_argument('--model_ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test_country', default=None)
    ap.add_argument('--fine_tune_ratio', type=float, default=0.0)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--visual_dim', type=int, default=128)
    ap.add_argument('--bridge_kind', default='contextual',
                    choices=['contextual', 'uncond', 'mean'],
                    help='MUST match the bridge the checkpoint was trained with, '
                         'or load_state_dict will fail.')
    ap.add_argument('--norm_scope', default='all',
                    choices=['none', 'screens', 'all'],
                    help='MUST match training. Exporting features from raw '
                         'screens when the model was trained on standardised '
                         'ones silently produces garbage.')
    args = ap.parse_args()

    data = load_and_merge(args.pair_set_dir)
    mask_args = types.SimpleNamespace(
        test_country=args.test_country, fine_tune_ratio=args.fine_tune_ratio,
        val_ratio=args.val_ratio, seed=args.seed)
    train_mask, val_mask, test_mask, norm_fit = build_masks(data, mask_args)
    apply_normalisation(data, norm_fit, args.norm_scope)

    model = HybridGATv2Model(visual_embed_dim=args.visual_dim,
                             bridge_kind=args.bridge_kind)
    model.load_state_dict(torch.load(args.model_ckpt, map_location='cpu'))
    x = model.node_features(data['pair_emb'], data['pair_node_idx'], data['screens'])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'x': x, 'y': data['y'], 'train_mask': train_mask,
                'val_mask': val_mask, 'test_mask': test_mask,
                'tender_ids': data['tender_ids'],
                'node_country': list(data['node_country'])}, args.out)
    print(f"Saved {tuple(x.shape)} node features -> {args.out}")
    print(f"train={int(train_mask.sum())} val={int(val_mask.sum())} "
          f"test={int(test_mask.sum())}")


if __name__ == '__main__':
    main()
