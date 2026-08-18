"""
src/evaluate_baseline_v2.py  --  NEW FILE (replaces evaluate_baseline.py)

Three fixes to the classical baselines, each closing an easy line of attack:

1. SCALING. The old script fed raw, unstandardised screens to LogisticRegression.
   The paper concedes this in its Limitations ("classifiers sensitive to feature
   scale, such as Logistic Regression, may be disadvantaged"). A reviewer will
   simply reply that standardising takes one line. It is now a Pipeline with
   StandardScaler, fit on train only.

2. HONEST HYPERPARAMETERS. The paper says the baselines use "the default
   hyperparameters in scikit-learn", but the code sets RandomForest with
   max_depth=10, which is not the default (None). Either the text or the code
   had to change; this script runs BOTH the true defaults and a small
   validation-tuned grid, so the hybrid model is compared against a baseline
   that was given a fair chance rather than a handicapped one. Beating a tuned
   Random Forest is a much stronger claim than beating an arbitrary one.

3. XGBoost. Gradient boosting is the obvious missing tabular baseline; on
   135-dim features it usually outperforms Random Forest. If it beats the
   hybrid model on F1 too, better to know that now than from a reviewer.

Also writes test_probabilities.csv in the same format as train_stage2_v2.py so
posthoc_analysis.py can compare everything on identical test cases.

Features come from the TRAINED bridge (via --node_feature_pt produced by
export_node_features.py), not from the old random-projection node features.

Usage:
    python src/evaluate_baseline_v2.py \
        --node_feature_pt outputs/analysis/learned_node_features.pt \
        --out_dir outputs/models_stage2_v2/in_sample/classical
"""

import os

# ---------------------------------------------------------------------------
# macOS OpenMP guard. MUST run before torch / sklearn / xgboost are imported.
#
# On Apple Silicon this script previously segfaulted inside XGBoost
# (XGQuantileDMatrixCreateFromCallback -> __kmp_allocate_task_team). The crash
# report showed THREE separate libomp.dylib images loaded into one process:
# PyTorch ships its own, scikit-learn ships its own, and Homebrew's xgboost
# links /opt/homebrew/lib/libomp.dylib. Two OpenMP runtimes in one process is
# undefined behaviour and reliably crashes when the second one tries to spawn a
# thread team. These two variables make the situation survivable.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import ParameterGrid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics_utils import full_metrics  # noqa: E402
from src.utils.config_loader import get_project_root  # noqa: E402

PROJECT_ROOT = get_project_root()
COUNTRIES = ['brazil', 'japan', 'usa']

USE_XGB = False   # set from --use_xgboost; see note above about the segfault
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def load_features(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return (d['x'].numpy(), d['y'].numpy(),
            d['train_mask'].numpy(), d['val_mask'].numpy(), d['test_mask'].numpy())


def make_models(seed, tuned=False):
    m = {
        'LR': Pipeline([('scale', StandardScaler()),
                        ('clf', LogisticRegression(class_weight='balanced',
                                                   max_iter=2000,
                                                   random_state=seed))]),
        'RF_default': RandomForestClassifier(class_weight='balanced',
                                             random_state=seed, n_jobs=-1),
        # scikit-learn's own gradient boosting. Same model family as XGBoost,
        # comparable accuracy on tabular data, and no second OpenMP runtime, so
        # this is the default gradient-boosting baseline for the paper.
        'HistGB': HistGradientBoostingClassifier(random_state=seed,
                                                 max_iter=300, max_depth=6,
                                                 learning_rate=0.1,
                                                 class_weight='balanced'),
    }
    if USE_XGB and HAS_XGB:
        # tree_method='exact' avoids the QuantileDMatrix code path that the
        # crash report pointed at; n_jobs=1 keeps XGBoost from spawning its own
        # thread team on the conflicting runtime.
        m['XGB'] = XGBClassifier(random_state=seed, n_estimators=300,
                                 max_depth=6, learning_rate=0.1,
                                 eval_metric='logloss', n_jobs=1,
                                 tree_method='exact', device='cpu')
    return m


def tune_rf(X_tr, y_tr, X_va, y_va, seed):
    grid = ParameterGrid({'n_estimators': [200, 500],
                          'max_depth': [None, 10, 20],
                          'min_samples_leaf': [1, 3]})
    best, best_f1 = None, -1
    for p in grid:
        clf = RandomForestClassifier(class_weight='balanced', random_state=seed,
                                     n_jobs=-1, **p)
        clf.fit(X_tr, y_tr)
        f1 = full_metrics(y_va, clf.predict_proba(X_va)[:, 1])['f1']
        if f1 > best_f1:
            best, best_f1 = p, f1
    return best, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--node_feature_pt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_runs', type=int, default=5)
    ap.add_argument('--base_seed', type=int, default=43)
    ap.add_argument('--use_xgboost', action='store_true',
                    help='Also fit XGBoost. Off by default: on Apple Silicon it '
                         'loads a second OpenMP runtime alongside PyTorch and '
                         'can segfault the process. HistGB covers the same '
                         'baseline without the risk.')
    ap.add_argument('--screens_only', action='store_true',
                    help='use only the last 7 dims (fair baseline for M2/M3)')
    args = ap.parse_args()

    global USE_XGB
    USE_XGB = args.use_xgboost
    if USE_XGB and not HAS_XGB:
        print("--use_xgboost given but xgboost is not installed; skipping it.")

    os.makedirs(args.out_dir, exist_ok=True)
    X, y, tr, va, te = load_features(args.node_feature_pt)
    if args.screens_only:
        X = X[:, -7:]
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]
    print(f"features={X.shape[1]}  train={len(X_tr)} val={len(X_va)} test={len(X_te)} "
          f"positive rate (train)={y_tr.mean():.3f}")

    rf_params, rf_val_f1 = tune_rf(X_tr, y_tr, X_va, y_va, args.base_seed)
    print(f"Tuned RF on validation: {rf_params} (val F1={rf_val_f1:.4f})")

    all_rows, prob_store = [], {}
    for name in list(make_models(0).keys()) + ['RF_tuned']:
        probs, rows = [], []
        for r in range(args.n_runs):
            seed = args.base_seed + r
            if name == 'RF_tuned':
                clf = RandomForestClassifier(class_weight='balanced',
                                             random_state=seed, n_jobs=-1,
                                             **rf_params)
            else:
                clf = make_models(seed)[name]
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_te)[:, 1]
            probs.append(p)
            m = full_metrics(y_te, p)
            m.update({'model': name, 'seed': seed})
            rows.append(m)
        all_rows.extend(rows)
        prob_store[name] = np.array(probs).T

        d = pd.DataFrame({f'run{i + 1}': probs[i] for i in range(len(probs))})
        d['y_true'] = y_te
        sub = os.path.join(args.out_dir, name)
        os.makedirs(sub, exist_ok=True)
        d.to_csv(os.path.join(sub, 'test_probabilities.csv'), index=False)

        f1s = [x['f1'] for x in rows]
        pr = [x['pr_auc'] for x in rows]
        print(f"  {name:12s} F1={np.mean(f1s):.4f}±{np.std(f1s, ddof=1):.4f} "
              f"PR-AUC={np.mean(pr):.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out_dir, 'baseline_runs.csv'), index=False)
    summary = df.groupby('model')[
        ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    ].agg(['mean', 'std'])
    summary.to_csv(os.path.join(args.out_dir, 'baseline_summary.csv'))
    with open(os.path.join(args.out_dir, 'rf_tuning.json'), 'w') as f:
        json.dump({'best_params': rf_params, 'val_f1': rf_val_f1,
                   'xgboost_installed': HAS_XGB,
                   'xgboost_used': USE_XGB}, f, indent=2)
    print("\n" + summary.round(4).to_string())
    if not USE_XGB:
        print("\nNOTE: gradient boosting is covered by HistGB (scikit-learn). "
              "XGBoost is off by default because it segfaults when a second "
              "OpenMP runtime is loaded alongside PyTorch on macOS ARM. "
              "Pass --use_xgboost only if you have verified it runs.")


if __name__ == '__main__':
    main()
