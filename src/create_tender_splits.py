"""
src/create_tender_splits.py
Create tender splits (train/val/test) before create image to prevent Data Leakage
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config_loader import CONFIG, get_project_root

PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
OUTPUT_DIR = os.path.join(PROJECT_ROOT, CONFIG['data'].get('splits_dir', 'outputs/splits'))
LOG_DIR = os.path.join(PROJECT_ROOT, CONFIG['data'].get('log_dir', 'outputs/logs'))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step1_splits.log')),
        logging.StreamHandler(sys.stdout) #print on terminal
    ]
)

COUNTRIES = [c['name'] for c in CONFIG['countries']]

try:
    SPLIT_RATIO = {
        'train': CONFIG['split_in_sample']['train_ratio'],
        'val': CONFIG['split_in_sample']['val_ratio'],
        'test': CONFIG['split_in_sample']['test_ratio']
    }
except KeyError:
    logging.warning("Use default split ratio (70/15/15)")
    SPLIT_RATIO = {'train': 0.70, 'val': 0.15, 'test': 0.15}

try:
    RANDOM_SEED = CONFIG['split']['random_seed']
except KeyError:
    logging.warning("Use default random_seed=42")
    RANDOM_SEED = 42

def create_tender_splits(country_name):
    logging.info(f"Processing {country_name.upper()}...")
    
    cleaned_path = os.path.join(PROCESSED_DIR, f"{country_name}_cleaned.parquet")
    if not os.path.exists(cleaned_path):
        logging.error(f"File not found: {cleaned_path}")
        return None
    
    df = pd.read_parquet(cleaned_path)
    tenders = df['Tender'].unique()
    logging.info(f"Total tenders: {len(tenders)}")
    
    tender_labels = df.groupby('Tender')['Collusive_competitor'].max().to_dict()
    labels = [tender_labels[t] for t in tenders]
    
    train_val, test = train_test_split(
        tenders, 
        test_size=SPLIT_RATIO['test'],
        stratify=labels,
        random_state=RANDOM_SEED
    )
    
    train_val_labels = [tender_labels[t] for t in train_val]
    val_ratio_adjusted = SPLIT_RATIO['val'] / (1 - SPLIT_RATIO['test'])
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=train_val_labels,
        random_state=RANDOM_SEED
    )
    
    splits = {
        'train': train.tolist(),
        'val': val.tolist(),
        'test': test.tolist(),
        'stats': {
            'total': len(tenders),
            'train': len(train),
            'val': len(val),
            'test': len(test)
        }
    }
    
    out_path = os.path.join(OUTPUT_DIR, f"{country_name}_split.json")
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    logging.info(f"Saved: train={len(train)}, val={len(val)}, test={len(test)}")
    return splits

def main():
    logging.info("=" * 60)
    logging.info("Creating Tender Splits (Before Image Generation)")
    logging.info("=" * 60)
    for country in COUNTRIES:
        create_tender_splits(country)
    logging.info(f"Done! Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()