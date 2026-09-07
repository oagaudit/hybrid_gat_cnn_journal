"""
src/make_figures.py

สร้าง Figure ทั้งหมดสำหรับเวอร์ชันวารสาร ตามสเปกของ Springer:
  ความกว้าง 174 mm (คอลัมน์เดียว) หรือ 84 mm (สองคอลัมน์)
  ฟอนต์ Helvetica/Arial, line art 1200 dpi, halftone 300 dpi

Usage:
    python src/make_figures.py --results_root outputs/results \
        --attention_dir outputs/results/analysis/attention \
        --out_dir outputs/results/figures
"""

import os
import json
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1 / 25.4
W_FULL, W_HALF = 174 * MM, 84 * MM
DPI_LINE = 1200

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'savefig.bbox': 'tight',
})

C_M4, C_M3, C_GREY = '#1b4f72', '#c0392b', '#7f8c8d'
COUNTRIES = ['brazil', 'japan', 'usa']
LABELS = {'brazil': 'Brazil', 'japan': 'Japan', 'usa': 'United States'}
COVERAGE = {'brazil': 15.29, 'japan': 50.26, 'usa': 94.35}


def load_sweep(root):
    rows = []
    for f in glob.glob(os.path.join(root, 'sweep', '*', 'summary.json')):
        d = json.load(open(f))
        c, m, r = os.path.basename(os.path.dirname(f)).split('_')
        rows.append({'country': c, 'model': m, 'ratio': float(r[2:]),
                     'f1': d['f1_mean'], 'f1_sd': d['f1_std'],
                     'pr': d['pr_auc_mean']})
    for f in glob.glob(os.path.join(root, 'loco', '*_C3', 'summary.json')):
        d = json.load(open(f))
        c, m, _ = os.path.basename(os.path.dirname(f)).split('_')
        rows.append({'country': c, 'model': m, 'ratio': 0.15,
                     'f1': d['f1_mean'], 'f1_sd': d['f1_std'],
                     'pr': d['pr_auc_mean']})
    for f in glob.glob(os.path.join(root, 'loco', '*_C2b', 'summary.json')):
        d = json.load(open(f))
        c, m, _ = os.path.basename(os.path.dirname(f)).split('_')
        rows.append({'country': c, 'model': m, 'ratio': 0.0,
                     'f1': d['f1_mean'], 'f1_sd': d['f1_std'],
                     'pr': d['pr_auc_mean']})
    return pd.DataFrame(rows).sort_values(['country', 'model', 'ratio'])


def load_structural(root):
    """The featureless baseline under the same few-shot protocol, if run."""
    f = os.path.join(root, 'analysis', 'graph_leakage',
                     'structural_baseline_fewshot.csv')
    return pd.read_csv(f) if os.path.exists(f) else None


def fig_sweep(df, out, struct=None):
    """Fig. A — few-shot label budget. The headline policy figure.

    The third line is the point of the figure rather than a detail. Without
    it a reader cannot tell how much of each curve is learning and how much
    is label propagation through a graph that, once the target market is
    partly labelled, can reach almost every test tender.
    """
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, W_FULL * 0.34), sharey=True)
    for ax, c in zip(axes, COUNTRIES):
        if struct is not None:
            b = struct[struct.country == c].sort_values('label_budget')
            if len(b):
                ax.plot(b.label_budget * 100, b.f1_best, marker='^',
                        color='0.35', ls='--', ms=3.5, lw=1.0,
                        label='structural baseline (no features)')
        for m, col, mk in [('M4', C_M4, 'o'), ('M3', C_M3, 's')]:
            s = df[(df.country == c) & (df.model == m)]
            ax.errorbar(s.ratio * 100, s.f1, yerr=s.f1_sd, marker=mk,
                        color=col, capsize=2, markersize=3.5,
                        label='M4 (visual + screens)' if m == 'M4'
                        else 'M3 (screens only)')
        ax.axvline(5, color=C_GREY, ls=':', lw=0.8)
        ax.set_title(f"{LABELS[c]}\n(pair coverage {COVERAGE[c]:.0f}%)", fontsize=8)
        ax.set_xlabel('')
        ax.set_ylim(0, 1.02)
        ax.grid(axis='y', alpha=0.25, lw=0.4)
    axes[0].set_ylabel('F1-score')
    axes[0].legend(frameon=False, loc='lower right')
    for ax in axes:
        ax.set_xlabel('Labelled target-market\ntenders (%)')
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(os.path.join(out, 'fig_fewshot_sweep.pdf'), dpi=DPI_LINE)
    fig.savefig(os.path.join(out, 'fig_fewshot_sweep.png'), dpi=600)
    plt.close(fig)


def fig_coverage(df, out):
    """Fig. B — the conditional finding: coverage governs whether visual helps."""
    adv = []
    for c in COUNTRIES:
        for ratio, tag in [(0.0, 'Zero-shot'), (0.15, 'Few-shot 15%')]:
            a = df[(df.country == c) & (df.model == 'M4') & (df.ratio == ratio)]
            b = df[(df.country == c) & (df.model == 'M3') & (df.ratio == ratio)]
            if len(a) and len(b):
                adv.append({'country': c, 'cond': tag,
                            'cov': COVERAGE[c],
                            'delta': a.f1.iloc[0] - b.f1.iloc[0]})
    a = pd.DataFrame(adv)

    fig, ax = plt.subplots(figsize=(W_HALF, W_HALF * 0.85))
    for tag, col, mk in [('Zero-shot', C_GREY, 's'), ('Few-shot 15%', C_M4, 'o')]:
        s = a[a.cond == tag].sort_values('cov')
        ax.plot(s['cov'].values, s['delta'].values, marker=mk, color=col,
                ls='--', ms=5, label=tag)
    for _, r in a[a.cond == 'Few-shot 15%'].iterrows():
        ax.annotate(LABELS[r['country']], (float(r['cov']), float(r['delta'])),
                    textcoords='offset points', xytext=(4, 5), fontsize=7)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('Bidder-pair coverage (%)')
    ax.set_ylabel('F1 advantage of M4 over M3')
    ax.legend(frameon=False, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(os.path.join(out, 'fig_coverage_advantage.pdf'), dpi=DPI_LINE)
    fig.savefig(os.path.join(out, 'fig_coverage_advantage.png'), dpi=600)
    plt.close(fig)


def fig_cost(root, out):
    """Fig. C — expected cost as the false-negative penalty rises."""
    f = os.path.join(root, 'analysis', 'posthoc', 'cost_curves.csv')
    if not os.path.exists(f):
        print('skip cost curve, file missing')
        return
    d = pd.read_csv(f)
    keep = {'M4': C_M4, 'M3': C_M3, 'RF': '#e67e22', 'HistGB': '#8e44ad',
            'MLP135': C_GREY}
    fig, ax = plt.subplots(figsize=(W_HALF, W_HALF * 0.85))
    for m, col in keep.items():
        s = d[d.model == m].sort_values('fn_fp_ratio')
        if len(s):
            ax.plot(s.fn_fp_ratio, s.expected_cost, marker='o', ms=3,
                    color=col, label=m)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cost of a false negative relative to a false positive')
    ax.set_ylabel('Expected cost per tender')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.4, which='both')
    fig.savefig(os.path.join(out, 'fig_cost_curves.pdf'), dpi=DPI_LINE)
    fig.savefig(os.path.join(out, 'fig_cost_curves.png'), dpi=600)
    plt.close(fig)


def fig_attention(att_dir, out):
    """Fig. D — what the bridge attends to, on the informative subset."""
    f = os.path.join(att_dir, 'attention_by_pair.csv')
    if not os.path.exists(f):
        print('skip attention figure, file missing')
        return
    d = pd.read_csv(f)
    g = d.groupby('node_idx')['is_cartel_pair'].agg(['size', 'sum'])
    mixed = set(g[(g['sum'] > 0) & (g['sum'] < g['size'])].index)
    m = d[d.node_idx.isin(mixed)]

    rows = []
    for nid, gg in m.groupby('node_idx'):
        top = gg.loc[gg.attention.idxmax()]
        rows.append({'country': gg.country.iloc[0],
                     'observed': int(top.is_cartel_pair),
                     'chance': gg.is_cartel_pair.mean()})
    r = pd.DataFrame(rows)
    agg = r.groupby('country').agg(observed=('observed', 'mean'),
                                   chance=('chance', 'mean')).reindex(COUNTRIES)

    fig, ax = plt.subplots(figsize=(W_HALF, W_HALF * 0.75))
    x = np.arange(len(agg))
    ax.bar(x - 0.19, agg.chance, 0.36, color=C_GREY, label='Expected by chance')
    ax.bar(x + 0.19, agg.observed, 0.36, color=C_M4, label='Observed')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in agg.index])
    ax.set_ylabel('Top-attention pair is a cartel pair')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    fig.savefig(os.path.join(out, 'fig_attention_selectivity.pdf'), dpi=DPI_LINE)
    fig.savefig(os.path.join(out, 'fig_attention_selectivity.png'), dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', default='outputs/results')
    ap.add_argument('--attention_dir', default='outputs/results/analysis/attention')
    ap.add_argument('--out_dir', default='outputs/results/figures')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_sweep(args.results_root)
    df.to_csv(os.path.join(args.out_dir, 'sweep_table.csv'), index=False)

    struct = load_structural(args.results_root)
    if struct is None:
        print('note: structural_baseline_fewshot.csv not found, so the sweep '
              'figure omits the featureless baseline. Produce it with\n'
              '      python src/structural_baseline.py --fewshot')
    fig_sweep(df, args.out_dir, struct)
    fig_coverage(df, args.out_dir)
    fig_cost(args.results_root, args.out_dir)
    fig_attention(args.attention_dir, args.out_dir)
    print(f'wrote figures to {args.out_dir}')


if __name__ == '__main__':
    main()
