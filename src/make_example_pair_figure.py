"""
src/make_example_pair_figure.py

Renders the worked example that Section 4.2 refers to: one competitive and
one collusive bidder pair, drawn from the same market and constructed
exactly as image_generator.py constructs the images the encoder is
trained on.

Two things about this script matter for the paper to be accurate.

1. Contextual points follow the training-time definition, which is
   narrower than "every other pair in the tender". For a focal pair
   (A, B), image_generator.py plots A against every other bidder it met
   in those tenders, and B against every other bidder it met, and nothing
   else. Pairs among third parties are not drawn. The contextual layer is
   therefore about how the two focal firms behave against outsiders, not
   about the tender at large. Reproducing that here is the difference
   between a figure that illustrates the model's input and a figure that
   merely resembles it.

2. The example is chosen by an explicit criterion rather than by eye. A
   figure showing one hand-picked pair invites the reader to wonder how
   many pairs were inspected first. We therefore score every qualifying
   pair for the rotation pattern and report the score, so the selection is
   reproducible and can be stated in the caption.

   rotation score = mean |b_A - b_B| x (1 - |mean sign(b_A - b_B)|)

   The first factor is large when the two firms bid far apart; the second
   is large when neither firm is consistently the higher bidder, that is,
   when the roles alternate. Their product is high only for a pair that
   repeatedly bids far apart while swapping who is low, which is the
   signature of bid rotation.

Usage
    python src/make_example_pair_figure.py --country brazil \
        --out outputs/results/figures/fig_example_pairs.pdf

    # reproduce a specific figure
    python src/make_example_pair_figure.py --country brazil \
        --competitive 50 76 --collusive 69 76 \
        --out outputs/results/figures/fig_example_pairs.pdf

The script prints the identifiers, the number of joint tenders and the
rotation score for both panels. Put those numbers in the caption.
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import CONFIG, get_project_root  # noqa: E402

PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])

MM = 1 / 25.4
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'savefig.bbox': 'tight',
})


# ---------------------------------------------------------------------------
def load_market(country):
    path = os.path.join(PROCESSED_DIR, f"{country}_cleaned.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run src/data_preprocessing.py first.")
    df = pd.read_parquet(path)
    if 'Bid_norm' not in df.columns:
        # same min-max normalisation as Equation (1)
        g = df.groupby('Tender')['Bid_value']
        lo, hi = g.transform('min'), g.transform('max')
        span = (hi - lo).replace(0, np.nan)
        df['Bid_norm'] = ((df['Bid_value'] - lo) / span).fillna(0.5)
    df['firm'] = df['Competitors'].astype(float).astype(int)
    return df


def build_pairs(df):
    """Replicate the pair and context construction of image_generator.py."""
    pair_points = defaultdict(list)     # (a,b) -> [(norm_a, norm_b), ...]
    pair_context = defaultdict(list)    # (a,b) -> [(x, y), ...]

    for _tid, g in df.groupby('Tender'):
        bidders = list(zip(g['firm'].values, g['Bid_norm'].values))
        if len(bidders) < 2:
            continue
        # every ordered-by-id pair present in this tender
        this = []
        for i in range(len(bidders)):
            for j in range(i + 1, len(bidders)):
                (ia, na), (ib, nb) = bidders[i], bidders[j]
                key = (ia, ib) if ia <= ib else (ib, ia)
                pa, pb = (na, nb) if ia <= ib else (nb, na)
                pair_points[key].append((pa, pb))
                this.append((key, pa, pb))
        # for each focal pair, context is each focal firm against its other
        # opponents in this tender, exactly as in image_generator.py
        seen = defaultdict(list)
        for key, pa, pb in this:
            a, b = key
            seen[a].append((b, pa, pb))
            seen[b].append((a, pb, pa))
        for key, _pa, _pb in this:
            a, b = key
            ctx = []
            for other, self_norm, other_norm in seen[a]:
                if other != b:
                    ctx.append((self_norm, other_norm))
            for other, self_norm, other_norm in seen[b]:
                if other != a:
                    ctx.append((self_norm, other_norm))
            pair_context[key].extend(sorted(set(ctx)))
    return pair_points, pair_context


def rotation_score(points):
    """High when the pair bids far apart and alternates who bids low."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return 0.0
    d = p[:, 0] - p[:, 1]
    separation = float(np.mean(np.abs(d)))
    signs = np.sign(d)
    signs = signs[signs != 0]
    balance = 1.0 - abs(float(np.mean(signs))) if len(signs) else 0.0
    return separation * balance


def boundary_stats(points, tol=0.15):
    """Share of joint tenders at opposite extremes, and share both low.

    Within-tender min-max normalisation forces one bidder to 0 and one to 1
    in every tender, so some mass on the boundary is mechanical. These two
    shares are reported together because their difference is what carries
    meaning: a rotating pair should be at opposite extremes often and low
    together almost never.
    """
    p = np.asarray(points, dtype=float)
    if len(p) == 0:
        return 0.0, 0.0
    lo, hi = p.min(axis=1), p.max(axis=1)
    opposite = float(np.mean((lo <= tol) & (hi >= 1 - tol)))
    both_low = float(np.mean((p[:, 0] <= tol) & (p[:, 1] <= tol)))
    return opposite, both_low


def random_pair_baseline(df, key, tol=0.15):
    """Opposite-extremes share expected of a pair drawn at random.

    For each tender in which the focal pair met, we ask what fraction of all
    bidder pairs present would satisfy the opposite-extremes condition, and
    average that over those tenders. With a tight tolerance this tends to
    1 / C(n, 2), because a tender with n bidders has exactly one minimum and
    one maximum; the tolerance makes the exact value slightly larger.
    """
    a, b = key
    shares = []
    for _tid, g in df.groupby('Tender'):
        firms = g['firm'].values
        if a not in firms or b not in firms:
            continue
        v = g['Bid_norm'].values
        n = len(v)
        if n < 2:
            continue
        hits = 0
        for i in range(n):
            for j in range(i + 1, n):
                lo, hi = min(v[i], v[j]), max(v[i], v[j])
                if lo <= tol and hi >= 1 - tol:
                    hits += 1
        shares.append(hits / (n * (n - 1) / 2))
    return float(np.mean(shares)) if shares else 0.0


def cartel_members(df):
    return set(df.loc[df['Collusive_competitor'] == 1, 'firm'].unique())


# ---------------------------------------------------------------------------
def panel_raw(ax, focal, context, marker=10, ctx_marker=4, ctx_alpha=0.3):
    """The encoder input: white focal points and grey context on black."""
    ax.set_facecolor('black')
    if len(context):
        c = np.asarray(context)
        ax.scatter(c[:, 0], c[:, 1], s=ctx_marker, c='gray',
                   alpha=ctx_alpha, edgecolors='none')
    f = np.asarray(focal)
    ax.scatter(f[:, 0], f[:, 1], s=marker, c='white', alpha=0.9,
               edgecolors='none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel_axes(ax, focal, context, xlabel):
    """The same data with axes, for a human reader."""
    if len(context):
        c = np.asarray(context)
        ax.scatter(c[:, 0], c[:, 1], s=5, c='0.70', edgecolors='none', zorder=1)
    f = np.asarray(focal)
    ax.scatter(f[:, 0], f[:, 1], s=20, c='#111111', edgecolors='none', zorder=3)
    ax.plot([0, 1], [0, 1], lw=0.5, ls=':', c='0.55', zorder=2)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--country', default='brazil')
    ap.add_argument('--competitive', nargs=2, type=int, default=None)
    ap.add_argument('--collusive', nargs=2, type=int, default=None)
    ap.add_argument('--min_interactions', type=int,
                    default=CONFIG.get('image_gen', {}).get('min_interactions', 3))
    ap.add_argument('--min_for_example', type=int, default=5,
                    help='a panel with only three or four points cannot show a '
                         'pattern; require at least this many joint tenders')
    ap.add_argument('--tol', type=float, default=0.15,
                    help='how close to the tender minimum or maximum counts '
                         'as an extreme')
    ap.add_argument('--suggest', action='store_true',
                    help='rank the qualifying pairs and exit without drawing')
    ap.add_argument('--out',
                    default='outputs/results/figures/fig_example_pairs.pdf')
    args = ap.parse_args()

    df = load_market(args.country)
    points, context = build_pairs(df)
    members = cartel_members(df)

    elig = {k: v for k, v in points.items()
            if len(v) >= max(args.min_interactions, args.min_for_example)}
    if not elig:
        raise SystemExit(
            f"no pair in {args.country} has at least {args.min_for_example} "
            "joint tenders; lower --min_for_example or try another market")

    scored = {k: rotation_score(v) for k, v in elig.items()}

    def is_col(k):
        return k[0] in members and k[1] in members

    if args.collusive:
        coll = tuple(sorted(args.collusive))
    else:
        cands = [k for k in elig if is_col(k)]
        if not cands:
            raise SystemExit("no eligible collusive pair in this market")
        coll = max(cands, key=lambda k: scored[k])

    if args.competitive:
        comp = tuple(sorted(args.competitive))
    else:
        n_target = len(points[coll])
        cands = [k for k in elig if not is_col(k)]
        if not cands:
            raise SystemExit("no eligible competitive pair in this market")
        # closest in sample size, then least rotational, so the contrast is
        # about the pattern rather than about how much data each panel has
        comp = min(cands, key=lambda k: (abs(len(points[k]) - n_target), scored[k]))

    stats = {}
    for lab, k in (('competitive', comp), ('collusive', coll)):
        if k not in points:
            raise SystemExit(f"pair {k} does not co-participate in any tender")
        opp, low = boundary_stats(points[k], args.tol)
        base = random_pair_baseline(df, k, args.tol)
        stats[lab] = dict(pair=k, n=len(points[k]), rot=rotation_score(points[k]),
                          opposite=opp, both_low=low, baseline=base,
                          ctx=len(context[k]))

    print()
    hdr = (f"{'panel':<12}{'pair':>10}{'tenders':>9}{'rot':>7}"
           f"{'opposite':>10}{'random':>8}{'both low':>10}")
    print(hdr)
    print('-' * len(hdr))
    for lab in ('competitive', 'collusive'):
        d = stats[lab]
        print(f"{lab:<12}{str(tuple(int(x) for x in d['pair'])):>10}"
              f"{d['n']:>9}{d['rot']:>7.2f}{d['opposite']:>10.2f}"
              f"{d['baseline']:>8.2f}{d['both_low']:>10.2f}")
    print("\nopposite = share of joint tenders with one firm within "
          f"{args.tol} of the tender minimum and the partner within "
          f"{args.tol} of the maximum")
    print("random   = the same share for a pair drawn at random from the "
          "bidders present in those tenders")
    print("both low = share of joint tenders in which both bid within "
          f"{args.tol} of the minimum")
    print("\nPut the opposite and both-low shares into the caption and into "
          "Section 4.2. Pass the pairs back with --competitive and "
          "--collusive to reproduce this figure.")

    if args.suggest:
        print("\nTop qualifying pairs by opposite-extremes share:")
        rows = []
        for k, v in elig.items():
            opp, low = boundary_stats(v, args.tol)
            rows.append((opp, low, rotation_score(v), len(v), k, is_col(k)))
        rows.sort(reverse=True)
        print(f"{'pair':>10}{'class':>13}{'tenders':>9}{'opposite':>10}"
              f"{'both low':>10}{'rot':>7}")
        for opp, low, rot, n, k, col in rows[:15]:
            print(f"{str(tuple(int(x) for x in k)):>10}"
                  f"{'collusive' if col else 'competitive':>13}"
                  f"{n:>9}{opp:>10.2f}{low:>10.2f}{rot:>7.2f}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(174 * MM, 176 * MM),
                             constrained_layout=True)
    for col, (k, lab) in enumerate(((comp, 'Competitive pair'),
                                    (coll, 'Collusive pair'))):
        a, b = int(k[0]), int(k[1])
        d = stats['competitive' if col == 0 else 'collusive']
        panel_raw(axes[0, col], points[k], context[k])
        axes[0, col].set_title(f"{lab}  ({a}, {b})", pad=5)
        panel_axes(axes[1, col], points[k], context[k],
                   f"normalised bid of firm {a}\n"
                   f"{d['n']} joint tenders\n"
                   f"opposite extremes {d['opposite']:.2f}"
                   f"  (random {d['baseline']:.2f})")
    axes[0, 0].set_ylabel('encoder input')
    axes[1, 0].set_ylabel('normalised bid of the partner firm')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=1200, bbox_inches=None)
    fig.savefig(os.path.splitext(args.out)[0] + '.png', dpi=600,
                bbox_inches=None)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
