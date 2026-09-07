"""
src/structural_baseline.py

ทำไมต้องมีสคริปต์นี้
--------------------
PART 1 แสดงว่า GATv2 บน statistical screens 7 ตัว (M3) ได้ F1 = 0.8896
แต่ EDA ในเปเปอร์เอง (Section IV.A) บอกว่า *"The absolute correlation between
any screen and the collusion label was below 0.07"*

feature ที่สหสัมพันธ์กับ label ต่ำกว่า 0.07 ไม่ควรให้ F1 ระดับ 0.89 ได้
ดังนั้นข้อมูลต้องมาจากที่อื่น และที่เดียวที่เหลือคือ **โครงสร้างกราฟ**

ปัญหาคือวิธีสร้างกราฟกับวิธีนิยาม label ในเปเปอร์เชื่อมกันโดยตรง:
  - edge: เชื่อมสอง tender ถ้ามี bidder ร่วมกันอย่างน้อย 1 ราย (Section III.H)
  - label: y_v = 1 ถ้ามีสมาชิกคาร์เทลที่ผ่านการยืนยันเข้าร่วม tender นั้น

ถ้าบริษัท X เป็นสมาชิกคาร์เทล ทุก tender ที่ X เข้าร่วมจะถูก label = 1 ทั้งหมด
และทุก tender เหล่านั้นก็เชื่อมถึงกันหมดด้วย edge เพราะแชร์ bidder X
GNN 2 ชั้นจึงกระจาย label ไปตามกลุ่มได้เกือบสมบูรณ์แบบ **โดยไม่ต้องดู feature เลย**

สคริปต์นี้วัดว่าปรากฏการณ์นั้นแรงแค่ไหน โดยทำนายจากกราฟกับ label ของชุด train
เพียงอย่างเดียว ไม่ใช้ feature ใดๆ ทั้งสิ้น

การตีความผล
-----------
  F1 >= 0.80  โครงสร้างกราฟอธิบายผลได้เกือบทั้งหมด ต้องเขียนเรื่องนี้ในเปเปอร์
              อย่างตรงไปตรงมา และ contribution ต้องย้ายไปอยู่ที่ cross-market
  F1 ~ 0.5-0.8 กราฟช่วยมาก แต่ยังมีที่ว่างให้ feature
  F1 < 0.5    ไม่มีปัญหา ผลของ M3/M4 มาจาก feature จริง

Usage:
    python src/structural_baseline.py --pair_set_dir outputs/pair_sets/insample
"""

import os
import sys
import types
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_stage2 import load_and_merge, build_masks   # noqa: E402
from src.metrics_utils import full_metrics, best_threshold     # noqa: E402


def neighbour_vote(edge_index, num_nodes, known_mask, known_y, hops=1):
    """Predicted score = fraction of known neighbours (within `hops`) labelled 1."""
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[s].append(d)

    known = known_mask.numpy()
    y = known_y.numpy()
    scores = np.full(num_nodes, np.nan)

    for v in range(num_nodes):
        seen, frontier = {v}, [v]
        for _ in range(hops):
            nxt = []
            for u in frontier:
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w)
                        nxt.append(w)
            frontier = nxt
        neigh = [w for w in seen if w != v and known[w]]
        scores[v] = float(np.mean(y[neigh])) if neigh else np.nan
    return scores


def sweep(args):
    """Run the baseline under the few-shot protocol at several label budgets.

    The within-market version of this baseline answers whether topology alone
    predicts the label. The cross-market version answers a second question
    that matters just as much: in a few-shot run, the target market's labels
    are visible during training, so the same propagation is available again.
    Reporting the baseline at each budget alongside the trained models shows
    whether the transfer curve reflects learning or a second round of label
    propagation. A baseline that stays flat while the models climb settles the
    question; one that climbs with them changes the conclusion.
    """
    rows = []
    for country in ['brazil', 'japan', 'usa']:
        pair_dir = args.pair_set_dir_template.format(country=country)
        if not os.path.isdir(pair_dir):
            print(f"  {pair_dir} not found, skipping {country}")
            continue
        data = load_and_merge(pair_dir)
        for ratio in args.ratios:
            m = types.SimpleNamespace(test_country=country,
                                      fine_tune_ratio=ratio,
                                      val_ratio=args.val_ratio, seed=args.seed)
            train_mask, val_mask, test_mask, _ = build_masks(data, m)
            known = train_mask | val_mask
            n = data['num_nodes']
            y = data['y']
            te = test_mask.numpy()

            # only target-market labels can propagate to target-market tests,
            # because the graph has no edge between markets
            tgt = np.asarray(data['node_country']) == country
            n_known_target = int((known.numpy() & tgt).sum())

            s = neighbour_vote(data['edge_index'], n, known, y, hops=1)
            isolated = int(np.isnan(s[te]).sum())
            s = np.nan_to_num(s, nan=float(y[known].float().mean()))
            yt, st = y[test_mask].numpy(), s[te]
            if len(np.unique(yt)) < 2:
                print(f"  {country} at {ratio}: test set has one class, skipped")
                continue
            thr = best_threshold(yt, st)
            m05 = full_metrics(yt, st, 0.5)
            mb = full_metrics(yt, st, thr)
            rows.append({'country': country, 'label_budget': ratio,
                         'n_test': int(test_mask.sum()),
                         'labelled_target_tenders': n_known_target,
                         'test_no_labelled_neighbour': isolated,
                         'base_rate': float(yt.mean()),
                         'pr_auc': m05['pr_auc'], 'roc_auc': m05['roc_auc'],
                         'f1_best': mb['f1']})
            print(f"  {country:7s} budget {ratio:.2f}  "
                  f"labelled target {n_known_target:5d}  "
                  f"PR-AUC {m05['pr_auc']:.4f}  F1best {mb['f1']:.4f}")

    if not rows:
        raise SystemExit("no folds could be evaluated; check --pair_set_dir_template")
    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'structural_baseline_fewshot.csv')
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nCompare these against the trained models at the same budgets. "
          "A flat, low curve here means the transfer results are not "
          "propagation; a curve that tracks the models means they partly are.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_set_dir', default='outputs/pair_sets/insample')
    ap.add_argument('--out_dir', default='outputs/results/analysis/graph_leakage')
    ap.add_argument('--fewshot', action='store_true',
                    help='run under the leave-one-country-out few-shot '
                         'protocol at several label budgets instead of '
                         'within-market')
    ap.add_argument('--pair_set_dir_template',
                    default='outputs/pair_sets/fold_{country}')
    ap.add_argument('--ratios', type=float, nargs='+',
                    default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=43)
    args = ap.parse_args()

    if args.fewshot:
        print("Structural baseline under the few-shot transfer protocol\n")
        sweep(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_and_merge(args.pair_set_dir)
    m = types.SimpleNamespace(test_country=None, fine_tune_ratio=0.0,
                              val_ratio=0.15, seed=43)
    train_mask, val_mask, test_mask, _ = build_masks(data, m)

    known = train_mask | val_mask
    n = data['num_nodes']
    y = data['y']

    print(f"nodes={n} edges={data['edge_index'].size(1)} "
          f"train+val={int(known.sum())} test={int(test_mask.sum())}")
    print(f"base rate (test) = {y[test_mask].float().mean():.4f}\n")

    lines = []
    countries = data['node_country']
    te = test_mask.numpy()

    for hops in (1, 2):
        s = neighbour_vote(data['edge_index'], n, known, y, hops=hops)
        isolated = int(np.isnan(s[te]).sum())
        s = np.nan_to_num(s, nan=float(y[known].float().mean()))
        yt = y[test_mask].numpy()
        st = s[te]

        # threshold 0.5 is meaningless for a raw vote fraction, so also report
        # the best achievable F1 on this score, which is the fair comparison
        # against a trained classifier
        thr = best_threshold(yt, st)
        m05 = full_metrics(yt, st, 0.5)
        mbest = full_metrics(yt, st, thr)

        line = (f"{hops}-hop neighbour vote  (NO features at all)\n"
                f"    PR-AUC  = {m05['pr_auc']:.4f}   <-- threshold-free, use this\n"
                f"    ROC-AUC = {m05['roc_auc']:.4f}   <-- threshold-free, use this\n"
                f"    F1 @0.50      = {m05['f1']:.4f} "
                f"(P={m05['precision']:.4f} R={m05['recall']:.4f})\n"
                f"    F1 @best={thr:.3f} = {mbest['f1']:.4f} "
                f"(P={mbest['precision']:.4f} R={mbest['recall']:.4f})\n"
                f"    test nodes with no known neighbour: {isolated}\n")

        if hops == 1:
            line += "    per country:\n"
            for c in ['brazil', 'japan', 'usa']:
                sel = te & (countries == c)
                if sel.sum() == 0 or len(np.unique(y.numpy()[sel])) < 2:
                    continue
                mc = full_metrics(y.numpy()[sel], s[sel],
                                  best_threshold(y.numpy()[sel], s[sel]))
                line += (f"      {c:7s} n={int(sel.sum()):5d} "
                         f"base={y.numpy()[sel].mean():.3f} "
                         f"PR-AUC={mc['pr_auc']:.4f} F1best={mc['f1']:.4f}\n")
        print(line)
        lines.append(line)

    with open(os.path.join(args.out_dir, 'graph_only_baseline.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print("เทียบกับ PART 1:  M3 (GATv2, 7 screens) F1 = 0.8896  |  "
          "M4 (135 dims) F1 = 0.8841")
    print("ถ้าตัวเลขข้างบนใกล้เคียง แปลว่าโครงสร้างกราฟทำงานเกือบทั้งหมด")


if __name__ == '__main__':
    main()
