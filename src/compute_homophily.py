"""
src/compute_homophily.py

ทำไมต้องมี
----------
Zhu et al. (2020) นิยาม *edge homophily ratio* ไว้ว่า

    h = |{(u,v) in E : y_u = y_v}| / |E|

คือสัดส่วนของ edge ที่เชื่อมโหนดซึ่งมี label เดียวกัน และแสดงว่า GNN
มาตรฐานทำงานได้ดีเมื่อ h สูง แต่เมื่อ h ต่ำ (heterophily) กลับแพ้ MLP ที่
ไม่ใช้กราฟเลย

สองข้อนี้ทำให้ h เป็นตัวเลขที่อธิบายผลของเราได้ตรงที่สุด:
  - MLP ของเราได้ F1 = 0.606 ขณะที่โมเดลกราฟได้ราว 0.90
    ตามกรอบของ Zhu et al. นี่คือลายเซ็นของกราฟที่มี homophily สูง
  - neighbour-voting baseline ที่ไม่ใช้ feature เลยได้ PR-AUC 0.900
    ซึ่งเป็นสิ่งที่ต้องเกิดขึ้นเมื่อ h สูง

สคริปต์นี้คำนวณ h ต่อประเทศและรวม เพื่อให้เปเปอร์รายงานตัวเลขนี้แทนที่จะ
พูดลอยๆ ว่า "โครงสร้างกราฟมี signal สูง" และเพื่อเทียบกับ null model
(สัดส่วนที่คาดหวังถ้า label สุ่มบนโครงสร้างเดิม) ซึ่งเป็นเกณฑ์ว่า h ที่ได้
สูงกว่าโดยบังเอิญหรือไม่

Usage:
    python src/compute_homophily.py --pair_set_dir outputs/pair_sets/insample \
        --out_dir outputs/results/analysis/homophily
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_stage2 import load_and_merge   # noqa: E402

COUNTRIES = ['brazil', 'japan', 'usa']
RNG = np.random.default_rng(20260819)


def edge_homophily(edge_index, y, mask=None):
    """h = share of edges whose endpoints carry the same label."""
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    if mask is not None:
        keep = mask[src] & mask[dst]
        src, dst = src[keep], dst[keep]
    if len(src) == 0:
        return np.nan, 0
    same = (y[src] == y[dst])
    return float(same.mean()), int(len(src))


def null_homophily(edge_index, y, mask=None, n_perm=200):
    """Expected h if labels were permuted over the same structure.

    The permutation must be carried out WITHIN the node set being measured.
    Permuting globally and then restricting to one market's edges compares
    that market's graph against the pooled class balance rather than its own,
    which makes markets with an above-average positive rate look heterophilic
    when they are not. This is the exact bug that produced the first version
    of these numbers.
    """
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    if mask is not None:
        keep = mask[src] & mask[dst]
        src, dst = src[keep], dst[keep]
        nodes = np.where(mask)[0]
        # relabel so that permutation happens over this market's nodes only
        pos = np.full(len(y), -1, dtype=np.int64)
        pos[nodes] = np.arange(len(nodes))
        src, dst = pos[src], pos[dst]
        y_local = y[nodes]
    else:
        y_local = y
    if len(src) == 0:
        return np.nan
    vals = []
    for _ in range(n_perm):
        yp = RNG.permutation(y_local)
        vals.append((yp[src] == yp[dst]).mean())
    return float(np.mean(vals))


def analytic_null(y, mask=None):
    """Closed form of the permutation expectation, as a cross-check.

    E[h] = [k(k-1) + (n-k)(n-k-1)] / [n(n-1)] for n nodes of which k are
    positive. Reported alongside the simulated value so that a discrepancy
    between them signals an implementation error rather than a finding.
    """
    yy = y if mask is None else y[mask]
    n = len(yy)
    k = int(yy.sum())
    if n < 2:
        return np.nan
    return (k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_set_dir', required=True)
    ap.add_argument('--out_dir', default='outputs/results/analysis/homophily')
    ap.add_argument('--n_perm', type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_and_merge(args.pair_set_dir)
    y = data['y'].numpy()
    country = np.asarray(data['node_country'])
    ei = data['edge_index']

    rows = []
    h_all, n_all = edge_homophily(ei, y)
    nl = null_homophily(ei, y, None, args.n_perm)
    rows.append({'scope': 'pooled', 'edge_homophily': round(h_all, 4),
                 'null_homophily': round(nl, 4),
                 'null_analytic': round(analytic_null(y), 4),
                 'excess': round(h_all - nl, 4),
                 'n_edges': n_all,
                 'positive_rate': round(float(y.mean()), 4)})

    for c in COUNTRIES:
        m = (country == c)
        h, n = edge_homophily(ei, y, m)
        nl = null_homophily(ei, y, m, args.n_perm)
        rows.append({'scope': c, 'edge_homophily': round(h, 4),
                     'null_homophily': round(nl, 4),
                     'null_analytic': round(analytic_null(y, m), 4),
                     'excess': round(h - nl, 4),
                     'n_edges': n,
                     'positive_rate': round(float(y[m].mean()), 4)})

    with open(os.path.join(args.out_dir, 'edge_homophily.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    w = max(len(r['scope']) for r in rows)
    print(f"{'scope':<{w}}  {'h':>8}  {'null':>8}  {'null(a)':>8}  "
          f"{'excess':>8}  {'edges':>11}  {'pos rate':>9}")
    for r in rows:
        print(f"{r['scope']:<{w}}  {r['edge_homophily']:>8.4f}  "
              f"{r['null_homophily']:>8.4f}  {r['null_analytic']:>8.4f}  "
              f"{r['excess']:>+8.4f}  {r['n_edges']:>11,}  "
              f"{r['positive_rate']:>9.4f}")
    if any(abs(r['null_homophily'] - r['null_analytic']) > 0.01 for r in rows):
        print("\nWARNING: simulated and analytic nulls disagree by more than "
              "0.01. Check that the permutation is confined to the masked "
              "node set.")
    print(f"\nwrote {args.out_dir}/edge_homophily.json")
    print("\nInterpretation: the excess column is the comparable quantity.")
    print("Raw h is not comparable across markets because the null depends on")
    print("each market's own class balance. Excess above zero means labels")
    print("cluster along edges, which is the condition under which message")
    print("passing succeeds without informative node features.")


if __name__ == '__main__':
    main()
