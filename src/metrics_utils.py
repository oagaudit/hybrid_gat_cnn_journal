"""
src/metrics_utils.py  --  NEW FILE

Metrics the current pipeline does not compute but the journal version needs.

Why each one matters for this specific paper:

1. average_precision (PR-AUC). With 11-33% positives, ROC-AUC is optimistic
   and the paper leans on it heavily to argue M4 beats Random Forest. PR-AUC is
   the standard companion metric for imbalanced screening problems and will be
   asked for.

2. Bootstrap confidence intervals. Brazil has 101 tenders, so its test split
   holds roughly 15 tenders. A point estimate of F1 on 15 cases is close to
   meaningless without an interval, and reporting one pre-empts the criticism.

3. Expected-cost curves. The current F1 comparison against Random Forest is a
   statistical tie (p = 0.7362). The paper's actual argument is that recall is
   worth more than precision in cartel screening. That argument should be made
   with numbers: at a false-negative-to-false-positive cost ratio of k, which
   model has lower expected cost?

4. Threshold sweep. Both models are evaluated at the default 0.5 cutoff, which
   nothing justifies. A screening tool would be tuned on validation data to a
   target recall or a target review budget.
"""

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)

RNG = np.random.default_rng(20260805)


def full_metrics(y_true, y_prob, threshold=0.5):
    """All metrics at a given threshold, including PR-AUC."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    out = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': np.nan if single_class else roc_auc_score(y_true, y_prob),
        'pr_auc': np.nan if single_class else average_precision_score(y_true, y_prob),
        'positive_rate': float(y_true.mean()),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                'fpr': fp / max(fp + tn, 1), 'fnr': fn / max(fn + tp, 1)})
    return out


def bootstrap_ci(y_true, y_prob, metric='f1', threshold=0.5, n_boot=2000,
                 alpha=0.05):
    """Percentile bootstrap CI over test cases (not over training seeds).

    Resamples the *test set*, so it captures how much the estimate depends on
    which tenders happened to land in the test split. This is the interval that
    matters for Brazil.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = full_metrics(yt, yp, threshold)
        stats.append(m[metric])
    if not stats:
        return {'metric': metric, 'point': np.nan, 'lo': np.nan, 'hi': np.nan,
                'n_valid_boot': 0}
    stats = np.asarray(stats)
    point = full_metrics(y_true, y_prob, threshold)[metric]
    return {'metric': metric, 'point': point,
            'lo': float(np.percentile(stats, 100 * alpha / 2)),
            'hi': float(np.percentile(stats, 100 * (1 - alpha / 2))),
            'n_valid_boot': int(len(stats))}


def expected_cost(y_true, y_prob, cost_fn=10.0, cost_fp=1.0, threshold=0.5):
    """Expected cost per tender at a given FN:FP cost ratio."""
    m = full_metrics(y_true, y_prob, threshold)
    n = len(y_true)
    return (m['fn'] * cost_fn + m['fp'] * cost_fp) / max(n, 1)


def cost_curve(y_true, y_prob, ratios=(1, 2, 5, 10, 20, 50), threshold=0.5):
    """Expected cost across a range of FN:FP ratios.

    Use this to replace the bare statement that F1 is statistically tied with
    Random Forest. Report the ratio at which the hybrid model overtakes it.
    """
    return [{'fn_fp_ratio': r,
             'expected_cost': expected_cost(y_true, y_prob, cost_fn=float(r),
                                            cost_fp=1.0, threshold=threshold)}
            for r in ratios]


def best_threshold(y_true, y_prob, objective='f1', target_recall=None):
    """Pick a threshold on validation data instead of defaulting to 0.5.

    objective='f1'      -> threshold maximising F1
    target_recall=0.9   -> cheapest threshold meeting a recall floor, which is
                           how a competition authority would actually set it
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]
    if len(thr) == 0:
        return 0.5

    if target_recall is not None:
        ok = np.where(rec >= target_recall)[0]
        if len(ok) == 0:
            return float(thr[int(np.argmax(rec))])
        return float(thr[ok[int(np.argmax(prec[ok]))]])

    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def paired_bootstrap_test(y_true, prob_a, prob_b, metric='f1', threshold=0.5,
                          n_boot=2000):
    """Compare two models on the same test set without assuming normality.

    The current paper runs a paired t-test on n=5 seeds, which tests whether
    training variance differs, not whether the models differ on the data. This
    resamples test cases and reports how often model A beats model B, which is
    the comparison a reviewer will find more convincing at these sample sizes.
    """
    y_true = np.asarray(y_true)
    pa, pb = np.asarray(prob_a), np.asarray(prob_b)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        ma = full_metrics(yt, pa[idx], threshold)[metric]
        mb = full_metrics(yt, pb[idx], threshold)[metric]
        diffs.append(ma - mb)
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return {'mean_diff': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'p_two_sided': np.nan}
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {'metric': metric, 'mean_diff': float(diffs.mean()),
            'ci_lo': float(np.percentile(diffs, 2.5)),
            'ci_hi': float(np.percentile(diffs, 97.5)),
            'p_two_sided': float(min(p, 1.0)),
            'n_valid_boot': int(diffs.size)}
