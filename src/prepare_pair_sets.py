"""
src/prepare_pair_sets.py  --  NEW FILE (replaces prepare_node_features.py)

Instead of collapsing each tender's bidder-pair embeddings into a 135-dim vector
offline with an untrained bridge, this script stores the *raw set* of pair
embeddings per tender in a ragged (flat + index) layout. The bridge then runs
inside the Stage-2 model and is trained by the classification loss.

Output per country: outputs/pair_sets/{scope}/{country}_pair_set.pt containing
    pair_emb      (P, 64)  float32  all pair embeddings in the country, stacked
    pair_node_idx (P,)     long     tender-node index each pair belongs to
    pair_key      list[str]         "{country}_{idA}_{idB}", aligned with pair_emb
    screens       (N, 7)   float32  statistical screens per tender
    y             (N,)     long     tender-level label
    edge_index    (2, E)   long     copied from prepare_graph_data.py output
    tender_ids    list               node ordering, copied from the graph file
    n_pairs       (N,)     long     number of observed pairs per tender

Note on coverage: a pair only has an embedding if the two firms co-participated
in at least `min_interactions` tenders (3 by default), so some tenders have zero
observed pairs. Those nodes get an exact zero visual vector. The script reports
the coverage rate per country because it is a material caveat for the paper:
in the current data 29.7% of Brazilian tenders have no observed pair at all.

Usage:
    python src/prepare_pair_sets.py \
        --embedding_csv outputs/embeddings/insample/all_pair_embeddings_with_labels.csv \
        --output_dir outputs/pair_sets/insample
"""

import os
import sys
import json
import argparse
import logging
import itertools

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import CONFIG, get_project_root  # noqa: E402

PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
GRAPH_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['graph_dir'])
LOG_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['log_dir'])
SCREEN_COLS = ['CV', 'SPD', 'DIFFP', 'RD', 'SKEW', 'KURT', 'KSTEST']
COUNTRIES = ['brazil', 'japan', 'usa']

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, 'prepare_pair_sets.log')),
              logging.StreamHandler(sys.stdout)],
)


def load_pair_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    emb_cols = [f'emb_{i}' for i in range(64)]
    missing = [c for c in emb_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Embedding CSV is missing columns: {missing[:5]} ...")
    mat = df[emb_cols].to_numpy(dtype=np.float32)
    return {k: mat[i] for i, k in enumerate(df['pair_key'].astype(str))}


def pair_keys_for_tender(country, bidders):
    keys = []
    for i, j in itertools.combinations(range(len(bidders)), 2):
        a, b = int(float(bidders[i])), int(float(bidders[j]))
        lo, hi = (a, b) if a <= b else (b, a)
        keys.append(f"{country}_{lo}_{hi}")
    return keys


def load_graph_node_order(country):
    graph_path = os.path.join(GRAPH_DIR, f"{country}_graph.pt")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Run prepare_graph_data.py first: {graph_path} not found")
    g = torch.load(graph_path, map_location='cpu', weights_only=False)
    edge_index = g.edge_index if hasattr(g, 'edge_index') else g['edge_index']
    tender_ids = getattr(g, 'tender_ids', None) if hasattr(g, 'edge_index') else g.get('tender_ids')
    if tender_ids is None:
        raise RuntimeError(f"{country}_graph.pt has no tender_ids; node order cannot be aligned.")
    return edge_index, list(tender_ids)


def process_country(country, emb_map, output_dir):
    logging.info(f"--- {country.upper()} ---")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{country}_cleaned.parquet"))
    edge_index, tender_ids = load_graph_node_order(country)

    screen_cols = [c for c in SCREEN_COLS if c in df.columns]
    if len(screen_cols) != len(SCREEN_COLS):
        logging.warning(f"{country}: missing screens {set(SCREEN_COLS) - set(screen_cols)}")

    by_tender = {tid: g for tid, g in df.groupby('Tender')}
    node_index = {tid: i for i, tid in enumerate(tender_ids)}

    n_nodes = len(tender_ids)
    screens = np.zeros((n_nodes, len(SCREEN_COLS)), dtype=np.float32)
    labels = np.zeros(n_nodes, dtype=np.int64)
    n_pairs = np.zeros(n_nodes, dtype=np.int64)

    pair_emb_list, pair_node_idx, pair_key_list = [], [], []
    total_slots = covered_slots = empty_tenders = 0

    for tid in tender_ids:
        idx = node_index[tid]
        g = by_tender.get(tid)
        if g is None:
            logging.warning(f"{country}: tender {tid} in graph but not in cleaned data")
            continue

        vals = g.iloc[0][screen_cols].to_numpy(dtype=np.float32)
        screens[idx, :len(screen_cols)] = vals
        labels[idx] = int(g['Collusive_competitor'].max())

        keys = pair_keys_for_tender(country, g['Competitors'].values)
        total_slots += len(keys)
        hits = 0
        for k in keys:
            if k in emb_map:
                pair_emb_list.append(emb_map[k])
                pair_node_idx.append(idx)
                pair_key_list.append(k)
                hits += 1
        covered_slots += hits
        n_pairs[idx] = hits
        if hits == 0:
            empty_tenders += 1

    pair_emb = (np.stack(pair_emb_list).astype(np.float32)
                if pair_emb_list else np.zeros((0, 64), dtype=np.float32))

    cov = covered_slots / total_slots * 100 if total_slots else 0.0
    logging.info(f"  nodes={n_nodes}  pair-slots={total_slots}  "
                 f"covered={covered_slots} ({cov:.1f}%)  "
                 f"tenders with zero observed pairs={empty_tenders} "
                 f"({empty_tenders / max(n_nodes, 1) * 100:.1f}%)")

    os.makedirs(output_dir, exist_ok=True)
    payload = {
        'pair_emb': torch.from_numpy(pair_emb),
        'pair_node_idx': torch.tensor(pair_node_idx, dtype=torch.long),
        'pair_key': pair_key_list,
        'screens': torch.from_numpy(screens),
        'y': torch.from_numpy(labels),
        'edge_index': edge_index,
        'tender_ids': tender_ids,
        'n_pairs': torch.from_numpy(n_pairs),
        'country': country,
    }
    out_path = os.path.join(output_dir, f"{country}_pair_set.pt")
    torch.save(payload, out_path)
    logging.info(f"  saved -> {out_path}")

    return {'country': country, 'nodes': int(n_nodes), 'pair_slots': int(total_slots),
            'covered': int(covered_slots), 'coverage_pct': round(cov, 2),
            'empty_tenders': int(empty_tenders),
            'empty_pct': round(empty_tenders / max(n_nodes, 1) * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedding_csv', required=True)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    logging.info("Loading pair embeddings ...")
    emb_map = load_pair_embeddings(args.embedding_csv)
    logging.info(f"Loaded {len(emb_map)} pair embeddings from {args.embedding_csv}")

    stats = [process_country(c, emb_map, args.output_dir) for c in COUNTRIES]

    stats_path = os.path.join(args.output_dir, 'coverage_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"\nCoverage table (report this in the paper) -> {stats_path}")
    logging.info("\n" + pd.DataFrame(stats).to_string(index=False))


if __name__ == '__main__':
    main()
