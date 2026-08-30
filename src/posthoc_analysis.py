"""
src/posthoc_analysis.py

Runs on the test_probabilities.csv files that train_stage2.py writes, so none
of this requires retraining a model.

Produces the three tables that answer the "why not just use Random Forest?"
question directly:

  cost_curves.csv        expected cost per tender for each model across a range
                         of false-negative-to-false-positive cost ratios
  bootstrap_ci.csv       95% percentile CIs on F1 / PR-AUC / ROC-AUC per model,
                         resampling test cases (the interval that matters for
                         Brazil's ~15-tender test split)
  model_comparison.csv   paired bootstrap comparison between any two models on
                         the identical test set

Usage:
    python src/posthoc_analysis.py --out_dir outputs/analysis/posthoc \
        --model M4=outputs/results/in_sample/hybrid/test_probabilities.csv \
        --model M3=outputs/results/in_sample/gatv2/test_probabilities.csv \
        --model RF=outputs/results/in_sample/rf/test_probabilities.csv \
        --compare M4:RF --compare M4:M3
"""

import os
import argparse

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics_utils import (full_metrics, bootstrap_ci, cost_curve,   # noqa: E402
                               paired_bootstrap_test)


def load_probs(path):
    df = pd.read_csv(path)
    y = df['y_true'].values
    runs = [c for c in df.columns if c.startswith('run')]
    return y, df[runs].values.mean(axis=1), df[runs].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True,
                    help='NAME=path/to/test_probabilities.csv (repeatable)')
    ap.add_argument('--compare', action='append', default=[],
                    help='NAME_A:NAME_B (repeatable)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--n_boot', type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = {}
    for spec in args.model:
        name, path = spec.split('=', 1)
        models[name] = load_probs(path)

    ref_y = None
    for name, (y, _, _) in models.items():
        if ref_y is None:
            ref_y = y
        elif len(y) != len(ref_y) or not np.array_equal(y, ref_y):
            raise SystemExit(
                f"Model '{name}' has a different test set than the first model. "
                "Cost curves and paired tests require an identical test split.")

    # -- headline metrics + bootstrap CIs -------------------------------------
    rows, ci_rows = [], []
    for name, (y, p_mean, p_runs) in models.items():
        per_run = [full_metrics(y, p_runs[:, i], args.threshold)
                   for i in range(p_runs.shape[1])]
        agg = {'model': name, 'n_runs': p_runs.shape[1], 'n_test': len(y)}
        for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'fpr', 'fnr']:
            vals = [m[k] for m in per_run]
            agg[f'{k}_mean'] = float(np.nanmean(vals))
            agg[f'{k}_std'] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(agg)

        for metric in ['f1', 'pr_auc', 'roc_auc', 'recall', 'precision']:
            ci = bootstrap_ci(y, p_mean, metric, args.threshold, n_boot=args.n_boot)
            ci['model'] = name
            ci_rows.append(ci)

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'headline_metrics.csv'), index=False)
    pd.DataFrame(ci_rows)[['model', 'metric', 'point', 'lo', 'hi', 'n_valid_boot']] \
        .to_csv(os.path.join(args.out_dir, 'bootstrap_ci.csv'), index=False)

    # -- cost curves ----------------------------------------------------------
    cost_rows = []
    for name, (y, p_mean, _) in models.items():
        for c in cost_curve(y, p_mean, threshold=args.threshold):
            cost_rows.append({'model': name, **c})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(os.path.join(args.out_dir, 'cost_curves.csv'), index=False)

    # -- paired comparisons ---------------------------------------------------
    comp_rows = []
    for spec in args.compare:
        a, b = spec.split(':', 1)
        ya, pa, _ = models[a]
        _, pb, _ = models[b]
        for metric in ['f1', 'pr_auc', 'roc_auc', 'recall']:
            r = paired_bootstrap_test(ya, pa, pb, metric, args.threshold, args.n_boot)
            r.update({'model_a': a, 'model_b': b})
            comp_rows.append(r)
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(args.out_dir, 'model_comparison.csv'), index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nExpected cost per tender by FN:FP ratio")
    print(cost_df.pivot(index='fn_fp_ratio', columns='model',
                        values='expected_cost').round(4).to_string())
    if comp_rows:
        print("\nPaired bootstrap comparisons")
        print(pd.DataFrame(comp_rows)[
            ['model_a', 'model_b', 'metric', 'mean_diff', 'ci_lo', 'ci_hi',
             'p_two_sided']].round(4).to_string(index=False))
    print(f"\nWrote CSVs to {args.out_dir}")


if __name__ == '__main__':
    main()
