"""
src/scripts/prepare_graph_data.py
Build tender-level graph and node features for Stage 2 (GAT)

Outputs:
- graph_data_{country}.pt: PyTorch Geometric Data object
  (contains x: node features, edge_index, edge_weight, y: labels)

NOTE: edge_weight is computed but NOT used in GATv2 model (unweighted edges are used)
as described in Section 4.4.2 of the thesis.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import CONFIG, get_project_root

# ============================================
# Setup PROJECT_ROOT and paths from config
# ============================================
PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
OUTPUT_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['graph_dir'])
LOG_DIR = os.path.join(PROJECT_ROOT, CONFIG['data'].get('log_dir', 'outputs/logs'))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================
# Setup logging (console + file)
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'prepare_graph_data.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# Parameters
TEMPORAL_SIGMA = 30  # days for temporal decay


def compute_screens(tender_df):
    """
    Compute the 7 statistical screens for a tender.
    
    Args:
        tender_df: DataFrame containing bids for a single tender
        
    Returns:
        dict: Dictionary of screen values or None if insufficient bids
    """
    bids = tender_df['Bid_norm'].values
    if len(bids) < 2:
        return None
    
    bids_sorted = np.sort(bids)
    n = len(bids_sorted)
    min_bid = bids_sorted[0]
    max_bid = bids_sorted[-1]
    mean_bid = np.mean(bids_sorted)
    std_bid = np.std(bids_sorted, ddof=1) if n > 1 else 0
    
    # CV (Coefficient of Variation)
    cv = std_bid / mean_bid if mean_bid > 0 else 0
    
    # SPD (Spread)
    spd = (max_bid - min_bid) / min_bid if min_bid > 0 else 0
    
    # DIFFP (Difference percent between two lowest bids)
    if n >= 2:
        diffp = (bids_sorted[1] - bids_sorted[0]) / bids_sorted[0] if bids_sorted[0] > 0 else 0
    else:
        diffp = 0
    
    # RD (Relative distance)
    losing_bids = bids_sorted[1:] if n > 1 else []
    std_losing = np.std(losing_bids, ddof=1) if len(losing_bids) > 1 else 1
    rd = (bids_sorted[1] - bids_sorted[0]) / std_losing if n >= 2 and std_losing > 0 else 0
    
    # SKEW (Skewness)
    if n >= 3 and std_bid > 0:
        skew = (n / ((n-1)*(n-2))) * np.sum(((bids_sorted - mean_bid) / std_bid) ** 3)
    else:
        skew = 0
    
    # KURT (Excess Kurtosis) - requires n >= 4
    if n >= 4 and std_bid > 0:
        kurt = (n*(n+1) / ((n-1)*(n-2)*(n-3))) * np.sum(((bids_sorted - mean_bid) / std_bid) ** 4) - (3*(n-1)**2 / ((n-2)*(n-3)))
    else:
        kurt = 0
    
    # KS test statistic (Kolmogorov-Smirnov against uniform distribution)
    if n >= 2:
        ks_stat = max(
            max([(i+1)/n - bids_sorted[i] for i in range(n)]),
            max([bids_sorted[i] - i/n for i in range(n)])
        )
    else:
        ks_stat = 0
    
    return {
        'CV': cv,
        'SPD': spd,
        'DIFFP': diffp,
        'RD': rd,
        'SKEW': skew,
        'KURT': kurt,
        'KS': ks_stat
    }


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets of bidders."""
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def temporal_decay(date1, date2, sigma=TEMPORAL_SIGMA):
    """Compute temporal decay factor: exp(-(Δt)^2 / σ^2)."""
    delta = abs((date1 - date2).days)
    return np.exp(- (delta ** 2) / (sigma ** 2))


def process_country(country_name):
    """
    Process a single country: compute screens, build graph, and save.
    
    Args:
        country_name: Name of the country (brazil, japan, usa)
    """
    logging.info(f"\n📂 Processing {country_name.upper()}...")
    
    # Load cleaned data
    cleaned_path = os.path.join(PROCESSED_DIR, f"{country_name}_cleaned.parquet")
    if not os.path.exists(cleaned_path):
        logging.error(f"Cleaned file not found: {cleaned_path}")
        return None
    
    df = pd.read_parquet(cleaned_path)
    
    # Check for Date column for temporal decay
    if 'Date' not in df.columns:
        logging.warning(f"{country_name} has no Date column. Temporal decay will be skipped.")
        use_temporal = False
    else:
        use_temporal = True
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
    
    # Group by tender
    tender_groups = df.groupby('Tender')
    
    node_features = []      # screens (7-dim)
    node_labels = []        # tender-level labels
    tender_ids = []         # tender IDs
    tender_dates = []       # dates for temporal decay
    bidder_sets = []        # sets of bidders per tender for Jaccard similarity
    
    for tender_id, tender_df in tqdm(tender_groups, desc=f"Computing screens for {country_name}"):
        screens = compute_screens(tender_df)
        if screens is None:
            continue
        
        # Label: 1 if any bidder is collusive
        label = int(tender_df['Collusive_competitor'].max())
        
        node_features.append([screens['CV'], screens['SPD'], screens['DIFFP'],
                              screens['RD'], screens['SKEW'], screens['KURT'], screens['KS']])
        node_labels.append(label)
        tender_ids.append(tender_id)
        
        if use_temporal:
            tender_date = tender_df['Date'].iloc[0]
            tender_dates.append(tender_date)
        else:
            tender_dates.append(None)
        
        bidders = set(tender_df['Competitors'].astype(str).values)
        bidder_sets.append(bidders)
    
    n_nodes = len(node_features)
    logging.info(f"  Total tenders after filtering: {n_nodes}")
    logging.info(f"  Label distribution: 0={node_labels.count(0)}, 1={node_labels.count(1)}")
    
    # Build edge index and edge weights
    edge_index = []
    edge_weight = []
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            jacc = jaccard_similarity(bidder_sets[i], bidder_sets[j])
            if jacc <= 0:
                continue
            
            if use_temporal and tender_dates[i] is not None and tender_dates[j] is not None:
                t_decay = temporal_decay(tender_dates[i], tender_dates[j], sigma=TEMPORAL_SIGMA)
            else:
                t_decay = 1.0
            
            weight = jacc * t_decay
            if weight > 0:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_weight.append(weight)
                edge_weight.append(weight)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float) if edge_weight else torch.empty(0)
    
    logging.info(f"  Edges (undirected): {edge_index.size(1)//2}")
    
    # Normalize node features (screens)
    scaler = StandardScaler()
    node_features_np = np.array(node_features)
    node_features_scaled = scaler.fit_transform(node_features_np)
    x = torch.tensor(node_features_scaled, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)
    
    # Create and save graph
    try:
        from torch_geometric.data import Data
        graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        graph_data.tender_ids = tender_ids
        graph_data.country = country_name
        out_path = os.path.join(OUTPUT_DIR, f"{country_name}_graph.pt")
        torch.save(graph_data, out_path)
        logging.info(f"  Saved graph to {out_path}")
    except ImportError:
        graph_dict = {
            'x': x,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'y': y,
            'tender_ids': tender_ids,
            'country': country_name
        }
        out_path = os.path.join(OUTPUT_DIR, f"{country_name}_graph.pt")
        torch.save(graph_dict, out_path)
        logging.info(f"  Saved graph dict to {out_path}")
    
    return graph_data


def main():
    logging.info("=" * 60)
    logging.info("🔧 Preparing Graph Data for Stage 2 (GAT)")
    logging.info("=" * 60)
    
    for country_config in CONFIG['countries']:
        country_name = country_config['name']
        if country_name not in ['brazil', 'japan', 'usa']:
            logging.info(f"Skipping {country_name} (not in selected list)")
            continue
        process_country(country_name)
    
    logging.info("\n Graph data preparation completed.")


if __name__ == "__main__":
    main()