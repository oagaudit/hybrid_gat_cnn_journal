"""
src/report_stage1.py

Collects the pair-level results of the Stage 1 encoder into one table.

Why this exists. The manuscript argues that the visual representation
carries usable information at the level of firm pairs but adds nothing
once aggregated to the tender level. The second half of that claim is
supported by Table 3; the first half was never reported, because
train_cnn.py writes its per-run metrics to a CSV that nothing downstream
reads. Without it a reviewer can offer a simpler explanation for the
whole paper -- that the encoder simply does not work -- and the argument
about bidder-pair coverage falls with it.

Nothing is recomputed here. train_cnn.py already evaluates the held-out
pair split and writes
    <output_dir>/<split>_cnn_stage1_trainonly_results.csv
with one row per seed. This script finds those files wherever they are,
aggregates them, and prints a table ready to paste into the manuscript.

Usage
    python src/report_stage1.py
    python src/report_stage1.py --root outputs --out_dir outputs/results/analysis/stage1
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd

METRICS = [('test_acc', 'Accuracy'), ('test_prec', 'Precision'),
           ('test_rec', 'Recall'), ('test_f1', 'F1'), ('test_auc', 'ROC-AUC')]


def find_files(root):
    pats = ['**/*cnn_stage1*results.csv', '**/metrics_summary.csv']
    hits = []
    for p in pats:
        hits += glob.glob(os.path.join(root, p), recursive=True)
    return sorted(set(hits))


def label_of(path):
    """Name the fold from the file name: pooled in-sample, or a LOCO fold."""
    base = os.path.basename(path)
    stem = base.replace('_cnn_stage1_trainonly_results.csv', '')
    if stem in ('metrics_summary.csv', 'metrics_summary'):
        parent = os.path.basename(os.path.dirname(os.path.dirname(path)))
        stem = parent or 'unknown'
    mapping = {'insample': 'Pooled (within-market)',
               'pooled': 'Pooled (within-market)',
               'fold_brazil': 'LOCO fold: Brazil held out',
               'fold_japan': 'LOCO fold: Japan held out',
               'fold_usa': 'LOCO fold: United States held out'}
    return mapping.get(stem, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs')
    ap.add_argument('--out_dir', default='outputs/results/analysis/stage1')
    args = ap.parse_args()

    files = [f for f in find_files(args.root) if 'cnn_stage1' in f]
    if not files:
        raise SystemExit(
            f"no Stage 1 result files under {args.root}.\n"
            "Look for them with:\n"
            "    find . -name '*cnn_stage1*results.csv'\n"
            "If none exist, the encoder was trained before this file was "
            "written; re-run src/train_cnn.py to produce it.")

    print(f"found {len(files)} Stage 1 result file(s)\n")
    rows = []
    for f in files:
        df = pd.read_csv(f)
        have = [m for m, _ in METRICS if m in df.columns]
        if not have:
            print(f"  skipping {f}: no test metrics in it")
            continue
        r = {'fold': label_of(f), 'runs': len(df), 'file': f}
        for m, _ in METRICS:
            if m in df.columns:
                r[m] = float(df[m].mean())
                r[m + '_sd'] = (float(df[m].std(ddof=1)) if len(df) > 1 else 0.0)
        rows.append(r)
        print(f"  {r['fold']:<34} {len(df)} runs   {f}")

    if not rows:
        raise SystemExit("found the files but none contained test metrics")

    out = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'stage1_pair_level.csv')
    out.to_csv(path, index=False)

    print("\n" + "=" * 74)
    print("STAGE 1, PAIR-LEVEL CLASSIFICATION   (mean over seeds, SD in brackets)")
    print("=" * 74)
    hdr = f"{'fold':<34}{'runs':>5}"
    for _, name in METRICS:
        hdr += f"{name:>14}"
    print(hdr)
    for _, r in out.iterrows():
        line = f"{r['fold']:<34}{int(r['runs']):>5}"
        for m, _ in METRICS:
            if m in r and not pd.isna(r[m]):
                line += f"{r[m]:>8.3f} ({r[m + '_sd']:.3f})".rjust(14)
            else:
                line += f"{'--':>14}"
        print(line)

    print(f"\nwrote {path}")
    print("\nPut the pooled row into a short subsection in Section 5, and point "
          "the two forward references at it: the sentence in Section 4.2 about "
          "where the evidence for the representation lies, and the sentence in "
          "Section 6.5 about information at the pair level. Both currently "
          "point at sections that report tender-level results.")


if __name__ == '__main__':
    main()
