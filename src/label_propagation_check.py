"""
src/label_propagation_check.py  --  NEW FILE, รันก่อนอย่างอื่น

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
    python src/label_propagation_check.py --pair_set_dir outputs/pair_sets/insample
"""

import os
import sys
import types
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_stage2_v2 import load_and_merge, build_masks   # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_set_dir', required=True)
    ap.add_argument('--out_dir', default='outputs/v2/analysis/graph_leakage')
    args = ap.parse_args()

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
