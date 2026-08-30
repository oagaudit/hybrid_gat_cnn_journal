"""
src/data_preprocessing.py
load data, clean (min_bids=2), Normalization, and save processed files
"""

import sys
import pandas as pd
import numpy as np
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import CONFIG, get_project_root

PROJECT_ROOT = get_project_root()
RAW_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['raw_dir'])
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
os.makedirs(PROCESSED_DIR, exist_ok=True)

REQUIRED_COLUMNS = ['Tender', 'Bid_value', 'Competitors', 'Winner', 
                    'Collusive_competitor', 'Number_bids', 'Date']
SCREENS = ['CV', 'SPD', 'DIFFP', 'RD', 'KURT', 'SKEW', 'KSTEST']

def load_raw_data(country_config):
    """load raw CSV and rename columns"""
    file_path = os.path.join(RAW_DIR, country_config['file'])
    df = pd.read_csv(file_path)
    
    # 1. Delete space  
    df.columns = df.columns.str.strip()
    
    # 2. Rename column  
    rename_map = {}
    for col in df.columns:
        if col.lower() == 'tender': rename_map[col] = 'Tender'
        if col.lower() == 'bid_value': rename_map[col] = 'Bid_value'
        if col.lower() == 'competitors': rename_map[col] = 'Competitors'
        if col.lower() == 'winner': rename_map[col] = 'Winner'
        if col.lower() == 'collusive_competitor': rename_map[col] = 'Collusive_competitor'
        if col.lower() == 'number_bids': rename_map[col] = 'Number_bids'
        if col.lower() == 'date': rename_map[col] = 'Date'
    
    df = df.rename(columns=rename_map)
    df['country'] = country_config['name']
    return df

def filter_min_bids(df, min_bids=2):
    """Delete row with Number_bids < min_bids"""
    before = len(df)
    df = df[df['Number_bids'] >= min_bids].copy()
    after = len(df)
    print(f"  Removed {before - after} rows (Number_bids < {min_bids})")
    return df

def normalize_bid_values(df):
    """Min-Max normalization with tender (use transform to speed up and prevent miss columns)"""
    # Calculate Vectorization  
    min_bids = df.groupby('Tender')['Bid_value'].transform('min')
    max_bids = df.groupby('Tender')['Bid_value'].transform('max')
    range_bids = max_bids - min_bids
    
    # If different period gap assign value Normalization else assign 0.5
    df['Bid_norm'] = np.where(range_bids > 0, (df['Bid_value'] - min_bids) / range_bids, 0.5)
    return df

def validate_screens(df, country_name):
    """Check correct value"""
    missing = df[SCREENS].isnull().sum().sum()
    if missing > 0:
        print(f"  Warning: {missing} missing values in screens for {country_name}")
    small_tenders = df[df['Number_bids'] < 4]
    if len(small_tenders) > 0:
        kurt_zero = (small_tenders['KURT'] == 0).sum()
        skew_zero = (small_tenders['SKEW'] == 0).sum()
        print(f"  Note: {kurt_zero}/{len(small_tenders)} tenders with n<4 have KURT=0")
    return df

def save_processed_data(df, country_name):
    """Save data to format Parquet"""
    output_path = os.path.join(PROCESSED_DIR, f"{country_name}_cleaned.parquet")
    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    return output_path

def get_tender_summary(df):
    """Create summary tender level for graph"""
    # Check columns first
    expected_base_cols = ['Number_bids', 'Date', 'country']
    base_cols = [col for col in expected_base_cols if col in df.columns]
    
    # 1. get basic data
    tender_df = df.groupby('Tender')[base_cols].first().reset_index()
    
    # 2. get Label
    label_df = df.groupby('Tender')['Collusive_competitor'].max().reset_index()
    
    # 3. collect tender name 
    bidder_lists = df.groupby('Tender')['Competitors'].apply(lambda x: list(x.astype(str))).reset_index()
    bidder_lists.columns = ['Tender', 'Bidders_list']
    
    # concat data
    tender_df = tender_df.merge(label_df, on='Tender').merge(bidder_lists, on='Tender')
    return tender_df

def main():
    print("=" * 60)
    print("Phase 1: Data Preprocessing (min_bids=2)")
    print("=" * 60)
    
    all_dfs = []
    for country_config in CONFIG['countries']:
        print(f"\nProcessing {country_config['name'].upper()}...")
        df = load_raw_data(country_config)
        print(f"  Loaded {len(df)} rows")
        df = filter_min_bids(df, min_bids=CONFIG['preprocessing']['min_bids'])
        df = normalize_bid_values(df)
        df = validate_screens(df, country_config['name'])
        save_processed_data(df, country_config['name'])
        
        tender_df = get_tender_summary(df)
        tender_path = os.path.join(PROCESSED_DIR, f"{country_config['name']}_tenders.parquet")
        tender_df.to_parquet(tender_path, index=False)
        print(f"  Saved tender summary: {len(tender_df)} tenders")
        all_dfs.append(df)
    
    if len(all_dfs) > 1:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_path = os.path.join(PROCESSED_DIR, "all_countries_cleaned.parquet")
        merged_df.to_parquet(merged_path, index=False)
        print(f"\nMerged dataset saved: {len(merged_df)} bids from {merged_df['country'].nunique()} countries")
    
    print("\n" + "=" * 60)
    print("Data preprocessing completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()