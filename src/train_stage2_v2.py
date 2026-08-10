"""
src/train_stage2_v2.py  --  NEW FILE (replaces train_stage2.py for the journal version)

Differences from train_stage2.py, all of them things a reviewer would flag:

1. TRAINABLE BRIDGE. Consumes the ragged pair sets from prepare_pair_sets.py and
   runs the Contextual Bridge inside the model, so its attention weights are
   learned from the classification loss instead of being a fixed random
   projection.

2. NORMALISATION SCOPE. The old --global_norm z-scored only the 7 screens and
   left the 128 visual dimensions untouched, i.e. 95% of the feature vector was
   never aligned across markets. --norm_scope {screens,all,none} makes this
   explicit; 'all' also standardises the pair embeddings using source-country
   statistics only.

3. PR-AUC and probability dumps. Per-run test probabilities are written to disk
   so cost curves, threshold sweeps, bootstrap CIs and paired bootstrap tests
   can be computed afterwards without retraining.

4. THRESHOLD SELECTION. --tune_threshold picks the decision cutoff on the
   VALIDATION split (max-F1 or a recall floor) instead of hard-coding 0.5.

5. MORE MODEL TYPES: hybrid | gatv2 | simple_gat | gcn | sage | mlp,
   and --bridge_kind contextual | uncond | mean for the pooling ablation.

Examples
--------
# In-sample hybrid, 5 seeds
python src/train_stage2_v2.py --model_type hybrid --pair_set_dir outputs/pair_sets/insample \
    --out_dir outputs/models_stage2_v2/in_sample/hybrid --num_runs 5

# Pooling ablation
python src/train_stage2_v2.py --model_type hybrid --bridge_kind mean  ... --out_dir .../hybrid_meanpool
python src/train_stage2_v2.py --model_type hybrid --bridge_kind uncond ... --out_dir .../hybrid_uncond

# Graph-necessity control
python src/train_stage2_v2.py --model_type mlp ... --out_dir .../mlp

# LOCO few-shot with full normalisation
python src/train_stage2_v2.py --model_type hybrid --pair_set_dir outputs/pair_sets/fold_japan \
    --test_country japan --fine_tune_ratio 0.15 --norm_scope all \
    --out_dir outputs/models_stage2_v2/loco/japan_c3
"""

import os
import sys
import json
import random
import logging
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hybrid_model import HybridGATv2Model            # noqa: E402
from src.models.gatv2_model import GATv2Model, SimpleGAT        # noqa: E402
from src.models.gnn_baselines import create_gnn_baseline        # noqa: E402
from src.metrics_utils import full_metrics, best_threshold      # noqa: E402
from src.utils.config_loader import get_project_root            # noqa: E402

PROJECT_ROOT = get_project_root()
COUNTRIES = ['brazil', 'japan', 'usa']
GRAPH_MODELS = {'hybrid', 'gatv2', 'simple_gat', 'gcn', 'sage', 'mlp'}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(arg):
    if arg == 'cpu':
        return torch.device('cpu')
    if arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if arg == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_and_merge(pair_set_dir):
    """Merge per-country pair sets into one graph with global node indexing."""
    xs_screens, ys, edges, pair_embs, pair_idx, pair_keys = [], [], [], [], [], []
    node_country, tender_ids = [], []
    offset = 0
    for c in COUNTRIES:
        path = os.path.join(pair_set_dir, f"{c}_pair_set.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run prepare_pair_sets.py first.")
        d = torch.load(path, map_location='cpu', weights_only=False)
        n = d['screens'].size(0)

        xs_screens.append(d['screens'])
        ys.append(d['y'])
        edges.append(d['edge_index'] + offset)
        pair_embs.append(d['pair_emb'])
        pair_idx.append(d['pair_node_idx'] + offset)
        pair_keys.extend(d['pair_key'])
        node_country.extend([c] * n)
        tender_ids.extend(list(d['tender_ids']))
        offset += n

    return {
        'screens': torch.cat(xs_screens, 0),
        'y': torch.cat(ys, 0),
        'edge_index': torch.cat(edges, 1),
        'pair_emb': torch.cat(pair_embs, 0),
        'pair_node_idx': torch.cat(pair_idx, 0),
        'pair_key': pair_keys,
        'node_country': np.array(node_country),
        'tender_ids': tender_ids,
        'num_nodes': offset,
    }


def apply_normalisation(data, fit_idx, scope):
    """Z-score using statistics from `fit_idx` nodes only (no target leakage)."""
    if scope == 'none':
        return
    screens = data['screens']
    mu = screens[fit_idx].mean(0, keepdim=True)
    sd = screens[fit_idx].std(0, keepdim=True) + 1e-8
    data['screens'] = (screens - mu) / sd
    logging.info(f"Standardised screens on {len(fit_idx)} source nodes.")

    if scope == 'all' and data['pair_emb'].numel() > 0:
        fit_set = torch.zeros(data['num_nodes'], dtype=torch.bool)
        fit_set[torch.as_tensor(fit_idx, dtype=torch.long)] = True
        mask = fit_set[data['pair_node_idx']]
        if mask.sum() > 1:
            pe = data['pair_emb']
            pmu = pe[mask].mean(0, keepdim=True)
            psd = pe[mask].std(0, keepdim=True) + 1e-8
            data['pair_emb'] = (pe - pmu) / psd
            logging.info(f"Standardised pair embeddings on {int(mask.sum())} source pairs.")
        else:
            logging.warning("Too few source pairs to standardise pair embeddings; skipped.")


def build_masks(data, args):
    n = data['num_nodes']
    y = data['y'].numpy()

    if args.test_country:
        tgt = np.where(data['node_country'] == args.test_country)[0]
        src = np.where(data['node_country'] != args.test_country)[0]
        if args.fine_tune_ratio > 0:
            ft, test_idx = train_test_split(
                tgt, train_size=args.fine_tune_ratio, stratify=y[tgt],
                random_state=args.seed)
            train_pool = np.concatenate([src, ft])
        else:
            train_pool, test_idx = src, tgt
        norm_fit = src  # never fit normalisation on target data
        tr, va = train_test_split(train_pool, test_size=args.val_ratio,
                                  stratify=y[train_pool], random_state=args.seed)
    else:
        split_dir = os.path.join(PROJECT_ROOT, 'outputs/splits')
        tr, va, test_idx = [], [], []
        for c in COUNTRIES:
            with open(os.path.join(split_dir, f"{c}_split.json")) as f:
                sp = json.load(f)
            s = {'train': set(sp['train']), 'val': set(sp['val']), 'test': set(sp['test'])}
            for i, (tid, cc) in enumerate(zip(data['tender_ids'], data['node_country'])):
                if cc != c:
                    continue
                if tid in s['train']:
                    tr.append(i)
                elif tid in s['val']:
                    va.append(i)
                elif tid in s['test']:
                    test_idx.append(i)
        tr, va, test_idx = np.array(tr), np.array(va), np.array(test_idx)
        if len(tr) + len(va) + len(test_idx) != n:
            raise RuntimeError(
                f"Split mismatch: {len(tr)}+{len(va)}+{len(test_idx)} != {n} nodes. "
                "Regenerate splits and pair sets together.")
        norm_fit = tr

    def to_mask(idx):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.as_tensor(np.asarray(idx), dtype=torch.long)] = True
        return m

    return to_mask(tr), to_mask(va), to_mask(test_idx), norm_fit


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(args, screen_dim=7):
    common = dict(hidden_dim=args.hidden_dim, out_dim=2,
                  num_layers=args.num_layers, dropout=args.dropout)
    if args.model_type == 'hybrid':
        return HybridGATv2Model(
            pair_embed_dim=64, screen_dim=screen_dim,
            visual_embed_dim=args.visual_dim, bridge_kind=args.bridge_kind,
            heads=args.heads, edge_dropout=args.edge_dropout,
            normalize_visual=not args.no_visual_norm, **common)
    in_dim = screen_dim
    if args.model_type == 'gatv2':
        return GATv2Model(in_dim=in_dim, heads=args.heads,
                          edge_dropout=args.edge_dropout, **common)
    if args.model_type == 'simple_gat':
        return SimpleGAT(in_dim=in_dim, heads=args.heads, **common)
    return create_gnn_baseline(args.model_type, in_dim=in_dim, **common)


def forward(model, data, args, device, return_attention=False):
    screens = data['screens'].to(device)
    edge_index = data['edge_index'].to(device)
    if args.model_type == 'hybrid':
        return model(data['pair_emb'].to(device), data['pair_node_idx'].to(device),
                     screens, edge_index, return_attention=return_attention)
    return model(screens, edge_index)


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def run_once(data, masks, args, device, seed):
    set_seed(seed)
    train_mask, val_mask, test_mask = masks
    model = build_model(args, screen_dim=data['screens'].size(1)).to(device)

    y = data['y'].to(device)
    counts = torch.bincount(data['y'][train_mask], minlength=2).float()
    weights = (counts.sum() / (2 * counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    best_f1, best_state, patience = -1.0, None, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        logits = forward(model, data, args, device)
        loss = criterion(logits[train_mask], y[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = forward(model, data, args, device)
            prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            vloss = criterion(logits[val_mask], y[val_mask]).item()
        vm = full_metrics(data['y'][val_mask].numpy(), prob[val_mask.numpy()])
        sched.step(vloss)
        history.append({'epoch': epoch, 'train_loss': loss.item(),
                        'val_loss': vloss, 'val_f1': vm['f1'], 'val_auc': vm['roc_auc']})

        if vm['f1'] > best_f1:
            best_f1 = vm['f1']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = forward(model, data, args, device, return_attention=args.save_attention)
        if args.save_attention and args.model_type == 'hybrid':
            logits, attn, _ = out
            attn = attn.cpu().numpy()
        else:
            logits, attn = (out[0] if isinstance(out, tuple) else out), None
        prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()

    y_val = data['y'][val_mask].numpy()
    thr = 0.5
    if args.tune_threshold:
        thr = best_threshold(y_val, prob[val_mask.numpy()],
                             target_recall=args.target_recall)

    y_test = data['y'][test_mask].numpy()
    p_test = prob[test_mask.numpy()]
    metrics = full_metrics(y_test, p_test, threshold=thr)
    metrics['seed'] = seed
    metrics['best_val_f1'] = best_f1
    metrics['n_epochs'] = len(history)
    return metrics, p_test, y_test, pd.DataFrame(history), attn, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_set_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model_type', default='hybrid', choices=sorted(GRAPH_MODELS))
    ap.add_argument('--bridge_kind', default='contextual',
                    choices=['contextual', 'uncond', 'mean'])
    ap.add_argument('--test_country', default=None, choices=COUNTRIES)
    ap.add_argument('--fine_tune_ratio', type=float, default=0.0)
    ap.add_argument('--norm_scope', default='none', choices=['none', 'screens', 'all'])
    ap.add_argument('--tune_threshold', action='store_true')
    ap.add_argument('--target_recall', type=float, default=None)
    ap.add_argument('--save_attention', action='store_true')
    ap.add_argument('--num_runs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--edge_dropout', type=float, default=0.2)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--visual_dim', type=int, default=128)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--early_stop', type=int, default=15)
    ap.add_argument('--no_visual_norm', action='store_true')
    ap.add_argument('--node_feature_pt', default=None,
                    help='Exported 135-dim learned features (from export_node_features.py). '
                         'Use for gcn/sage/mlp/gatv2/simple_gat so they are compared '
                         'against M4 on identical features.')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(args.out_dir, 'train.log')),
                  logging.StreamHandler(sys.stdout)])

    device = resolve_device(args.device)
    logging.info(f"Device: {device} | config: {vars(args)}")

    data = load_and_merge(args.pair_set_dir)

    if args.node_feature_pt:
        if args.model_type == 'hybrid':
            raise SystemExit("--node_feature_pt is for the non-hybrid baselines "
                             "(gcn/sage/mlp/gatv2/simple_gat). The hybrid model "
                             "builds its own features from the trainable bridge.")
        nf = torch.load(args.node_feature_pt, map_location='cpu', weights_only=False)
        if nf['x'].size(0) != data['num_nodes']:
            raise SystemExit(f"Feature file has {nf['x'].size(0)} nodes but the "
                             f"pair sets have {data['num_nodes']}.")
        data['screens'] = nf['x'].float()
        logging.info(f"Using exported {data['screens'].size(1)}-dim learned node "
                     "features for this baseline (fair comparison against M4).")

    train_mask, val_mask, test_mask, norm_fit = build_masks(data, args)
    apply_normalisation(data, norm_fit, args.norm_scope)
    logging.info(f"nodes={data['num_nodes']} train={int(train_mask.sum())} "
                 f"val={int(val_mask.sum())} test={int(test_mask.sum())} "
                 f"pairs={data['pair_emb'].size(0)}")

    rows, probs = [], {}
    for r in range(args.num_runs):
        seed = args.seed + r
        m, p_test, y_test, hist, attn, model = run_once(
            data, (train_mask, val_mask, test_mask), args, device, seed)
        rows.append(m)
        probs[f'run{r + 1}'] = p_test
        hist.to_csv(os.path.join(args.out_dir, f'history_run{r + 1}.csv'), index=False)
        if attn is not None and r == 0:
            np.save(os.path.join(args.out_dir, 'bridge_attention_run1.npy'), attn)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_run1.pt'))
        logging.info(f"[seed {seed}] F1={m['f1']:.4f} PR-AUC={m['pr_auc']:.4f} "
                     f"ROC-AUC={m['roc_auc']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                     f"thr={m['threshold']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'results.csv'), index=False)

    prob_df = pd.DataFrame(probs)
    prob_df['y_true'] = y_test
    prob_df.to_csv(os.path.join(args.out_dir, 'test_probabilities.csv'), index=False)

    summary = {'model_type': args.model_type, 'bridge_kind': args.bridge_kind,
               'test_country': args.test_country,
               'fine_tune_ratio': args.fine_tune_ratio,
               'norm_scope': args.norm_scope, 'num_runs': args.num_runs}
    for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'fpr', 'fnr']:
        summary[f'{k}_mean'] = float(df[k].mean())
        summary[f'{k}_std'] = float(df[k].std())
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2))
    logging.info(f"Per-run test probabilities saved -> {args.out_dir}/test_probabilities.csv "
                 "(feed this to metrics_utils for cost curves and bootstrap tests)")


if __name__ == '__main__':
    main()
