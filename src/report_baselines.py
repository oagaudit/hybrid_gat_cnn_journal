"""
src/report_baselines.py

Produces the two tables that Section 5.1 needs, without having to know
where the previous runs put their output. It searches for every
test_probabilities.csv under outputs/, works out which model each one
belongs to from its directory name, and then runs the headline metrics
and the paired comparisons on whichever of them it found.

The reason this exists rather than a longer posthoc_analysis.py command
line is that the output tree has changed name more than once during the
project (models_stage2_v2, v2, results), so any fixed path is wrong for
some copy of the repository.

Usage
    python src/report_baselines.py

    # look somewhere other than outputs/
    python src/report_baselines.py --root outputs/v2

    # see what was found without computing anything
    python src/report_baselines.py --list

Output
    A table of headline metrics with bootstrap intervals, and a table of
    paired comparisons between every graph model and every tabular model.
    Both are written to CSV and printed.
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics_utils import (full_metrics, bootstrap_ci,          # noqa: E402
                               paired_bootstrap_test, cost_curve)

# directory-name fragment -> label used in the paper, longest match wins
PATTERNS = [
    ('classical_full/LR', 'LR'),
    ('classical_full/RF_tuned', 'RF (tuned)'),
    ('classical_full/RF_default', 'RF (default)'),
    ('classical_full/HistGB', 'HistGB'),
    ('classical_full/XGB', 'XGB'),
    ('classical_screens/LR', 'LR (7 screens)'),
    ('classical_screens/RF_tuned', 'RF (7 screens)'),
    ('M4_contextual', 'M4'),
    ('M4_mean', 'M4 (mean pool)'),
    ('M4_uncond', 'M4 (free query)'),
    ('M4_recall90', 'M4 (recall 0.90)'),
    ('mlp_screens', 'MLP (7 screens)'),
    ('/mlp', 'MLP (135)'),
    ('/gcn', 'GCN'),
    ('/sage', 'GraphSAGE'),
    ('/M3', 'M3'),
    ('/M2', 'M2'),
]

GRAPH_MODELS = {'M4', 'M4 (mean pool)', 'M4 (free query)', 'M3', 'M2',
                'GCN', 'GraphSAGE'}
TABULAR_MODELS = {'LR', 'RF (tuned)', 'RF (default)', 'HistGB', 'XGB',
                  'MLP (135)', 'MLP (7 screens)'}


def label_for(path):
    norm = path.replace(os.sep, '/')
    best = None
    for frag, lab in PATTERNS:
        if frag in norm and (best is None or len(frag) > len(best[0])):
            best = (frag, lab)
    return best[1] if best else None


def discover(root):
    """Find in-sample probability files and label them."""
    found = {}
    for path in sorted(glob.glob(os.path.join(root, '**', 'test_probabilities.csv'),
                                 recursive=True)):
        norm = path.replace(os.sep, '/')
        # cross-market runs live under loco/ or sweep/ and are not wanted here
        if '/loco/' in norm or '/sweep/' in norm:
            continue
        lab = label_for(path)
        if lab is None:
            continue
        if lab in found:
            # Runs from an earlier, superseded configuration often survive
            # beside the final ones (for example M2 next to M2_norm). Prefer
            # the normalised variant, since that is the configuration the
            # paper reports, and say plainly which file was dropped.
            keep_new = ('_norm' in norm) and ('_norm' not in found[lab])
            kept, dropped = ((path, found[lab]) if keep_new
                             else (found[lab], path))
            print(f"  note: two directories map to {lab}\n"
                  f"        using   {kept}\n"
                  f"        ignored {dropped}")
            found[lab] = kept
            continue
        found[lab] = path
    return found


def load(path):
    df = pd.read_csv(path)
    runs = [c for c in df.columns if c.startswith('run')]
    return df['y_true'].values, df[runs].values.mean(axis=1), df[runs].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs')
    ap.add_argument('--out_dir', default=None,
                    help='default: <root>/analysis/baseline_report')
    ap.add_argument('--n_boot', type=int, default=3000)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--list', action='store_true',
                    help='show what was found and stop')
    args = ap.parse_args()

    found = discover(args.root)
    if not found:
        # The results often live in a backup copy of the project rather than
        # in the working tree, so look around before giving up.
        print(f"nothing under {args.root}; searching nearby directories ...")
        hits = []
        for base in ('.', '..'):
            hits += glob.glob(os.path.join(base, '**', 'test_probabilities.csv'),
                              recursive=True)
        roots = sorted({h.split('outputs')[0] + 'outputs'
                        for h in hits if 'outputs' in h})
        msg = [f"no test_probabilities.csv found under {args.root}."]
        if roots:
            msg.append("\nFound results here instead. Re-run with one of:")
            for r in roots:
                msg.append(f'    python src/report_baselines.py --root "{r}"')
        else:
            msg.append("\nNothing found nearby either. Locate them with:")
            msg.append("    find ~ -name test_probabilities.csv 2>/dev/null")
        raise SystemExit("\n".join(msg))

    print(f"found {len(found)} in-sample models under {args.root}/\n")
    for lab in sorted(found):
        print(f"  {lab:<18} {found[lab]}")
    if args.list:
        return

    out_dir = args.out_dir or os.path.join(args.root, 'analysis',
                                           'baseline_report')
    os.makedirs(out_dir, exist_ok=True)

    data = {lab: load(p) for lab, p in found.items()}
    ref = next(iter(data.values()))[0]
    usable = {}
    for lab, (y, pm, pr) in data.items():
        if len(y) != len(ref) or not np.array_equal(y, ref):
            print(f"\n  skipping {lab}: different test set, so it cannot be "
                  "compared on paired resamples")
            continue
        usable[lab] = (y, pm, pr)

    # ---- headline metrics ------------------------------------------------
    rows = []
    for lab, (y, pm, pr) in usable.items():
        per_run = [full_metrics(y, pr[:, i], args.threshold)
                   for i in range(pr.shape[1])]
        r = {'model': lab, 'n_runs': pr.shape[1], 'n_test': len(y)}
        # Two estimators are reported deliberately. The pooled figure comes
        # from averaging the per-run probabilities and is what the bootstrap
        # interval is computed around, so point and interval always refer to
        # the same quantity. The per-run mean is kept alongside it because a
        # gap between the two indicates that the runs disagree, which is
        # itself worth seeing.
        for k in ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            r[f'{k}_runmean'] = float(np.nanmean([m[k] for m in per_run]))
            r[f'{k}_runsd'] = (float(np.nanstd([m[k] for m in per_run], ddof=1))
                               if pr.shape[1] > 1 else 0.0)
        pooled = full_metrics(y, pm, args.threshold)
        for k in ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            r[k] = pooled[k]
        for k in ['f1', 'pr_auc']:
            ci = bootstrap_ci(y, pm, k, args.threshold, n_boot=args.n_boot)
            r[f'{k}_lo'], r[f'{k}_hi'] = ci['lo'], ci['hi']
        rows.append(r)
    head = pd.DataFrame(rows).sort_values('pr_auc', ascending=False)
    head.to_csv(os.path.join(out_dir, 'headline_metrics.csv'), index=False)

    print("\n" + "=" * 78)
    print("HEADLINE METRICS   (brackets are 95% bootstrap intervals)")
    print("=" * 78)
    print("point estimates are computed on the run-averaged probabilities; "
          "the\nlast column shows the spread of F1 across individual runs\n")
    print(f"{'model':<18}{'F1':>22}{'PR-AUC':>22}{'recall':>9}{'F1 per run':>16}")
    for _, r in head.iterrows():
        f1 = f"{r['f1']:.3f} [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}]"
        pr = f"{r['pr_auc']:.3f} [{r['pr_auc_lo']:.3f}, {r['pr_auc_hi']:.3f}]"
        rm = f"{r['f1_runmean']:.3f}±{r['f1_runsd']:.3f}"
        print(f"{r['model']:<18}{f1:>22}{pr:>22}{r['recall']:>9.3f}{rm:>16}")

    # ---- every graph model against every tabular model --------------------
    comp = []
    for g in [m for m in usable if m in GRAPH_MODELS]:
        for t in [m for m in usable if m in TABULAR_MODELS]:
            for metric in ['f1', 'pr_auc', 'recall']:
                res = paired_bootstrap_test(usable[g][0], usable[g][1],
                                            usable[t][1], metric,
                                            args.threshold, args.n_boot)
                res.update({'model_a': g, 'model_b': t})
                comp.append(res)
    if comp:
        cdf = pd.DataFrame(comp)[['model_a', 'model_b', 'metric', 'mean_diff',
                                  'ci_lo', 'ci_hi', 'p_two_sided']]
        cdf.to_csv(os.path.join(out_dir, 'graph_vs_tabular.csv'), index=False)
        print("\n" + "=" * 78)
        print("GRAPH MODELS AGAINST TABULAR MODELS")
        print("=" * 78)
        print(f"{'A':<13}{'B':<17}{'metric':<9}{'diff':>9}{'95% CI':>22}"
              f"{'p':>9}")
        for _, r in cdf.iterrows():
            ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
            p = '<0.001' if r['p_two_sided'] < 0.001 else f"{r['p_two_sided']:.3f}"
            print(f"{r['model_a']:<13}{r['model_b']:<17}{r['metric']:<9}"
                  f"{r['mean_diff']:>+9.3f}{ci:>22}{p:>9}")

    # ---- cost curves -----------------------------------------------------
    cost = []
    for lab, (y, pm, _) in usable.items():
        for c in cost_curve(y, pm, threshold=args.threshold):
            cost.append({'model': lab, **c})
    pd.DataFrame(cost).to_csv(os.path.join(out_dir, 'cost_curves.csv'),
                              index=False)

    print(f"\nwrote three CSVs to {out_dir}/")
    print("Use headline_metrics.csv for Table 2 and graph_vs_tabular.csv "
          "for Table 3.")


if __name__ == '__main__':
    main()
