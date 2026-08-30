"""
src/scripts/image_generator.py
Create Bid Rotation images with optional:
- Filtering sparse pairs (min_interactions)
- Contextual points (other firm pairs in same tender) in gray
- Filter by allowed_tenders (to prevent data leakage)

=====================================================================
FIXED VERSION — generates TWO separate image sets per country:

  1. "{country}_trainonly_pair_images.h5" / "..._trainonly_pair_labels.csv"
     -> built ONLY from train-split tenders (splits['train'])
     -> use this to TRAIN the CNN (train_cnn.py) so that no bid from
        val/test tenders ever influences the CNN weights.

  2. "{country}_full_pair_images.h5" / "..._full_pair_labels.csv"
     -> built from ALL tenders (train+val+test)
     -> use this ONLY for inference / embedding extraction
        (extract_embeddings.py) to build node features for Stage 2.
        This is safe because it is a forward pass only, the CNN
        weights are frozen and were never fitted on this data.

Previously, the leakage-prevention filter existed in the code but was
NEVER actually applied at the call site in main() — the line
`allowed_tenders = splits['train']` was commented out and replaced
with `splits['train'] + splits['val'] + splits['test']`, which is
equivalent to using ALL tenders (i.e. no filtering at all). This
version fixes that by explicitly generating both sets with the
correct tender lists.
=====================================================================
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils.config_loader import CONFIG, get_project_root

PROJECT_ROOT = get_project_root()


PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
IMAGES_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['images_dir'])
SPLIT_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['splits_dir'])
LOG_DIR = os.path.join(PROJECT_ROOT, CONFIG['data'].get('log_dir', 'outputs/logs'))

# สร้างโฟลเดอร์ที่จำเป็น
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================
# ตั้งค่า logging (แสดงบนจอ + ไฟล์)
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'step2_images.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# ============================================
# Load parameter from config.yaml
# ============================================
IMAGE_SIZE = CONFIG['preprocessing'].get('image_size', 96)
DPI = CONFIG['image_gen'].get('dpi', 100)
MARKER_SIZE = CONFIG['image_gen'].get('marker_size', 8)
MIN_INTERACTIONS = CONFIG['image_gen'].get('min_interactions', 3)
ADD_CONTEXT = CONFIG['image_gen'].get('add_context', True)
CONTEXT_ALPHA = CONFIG['image_gen'].get('context_alpha', 0.2)
default_ctx_marker = max(1, MARKER_SIZE // 2)
CONTEXT_MARKER_SIZE = CONFIG['image_gen'].get('context_marker_size', default_ctx_marker)
USE_HDF5 = CONFIG['preprocessing'].get('use_hdf5', True)

logging.info("=" * 60)
logging.info(" Image Generator Started (leakage-safe, dual-set version)")
logging.info("=" * 60)
logging.info(f"Settings:")
logging.info(f"  IMAGE_SIZE: {IMAGE_SIZE}")
logging.info(f"  DPI: {DPI}")
logging.info(f"  MARKER_SIZE: {MARKER_SIZE}")
logging.info(f"  MIN_INTERACTIONS: {MIN_INTERACTIONS}")
logging.info(f"  ADD_CONTEXT: {ADD_CONTEXT}")
logging.info(f"  USE_HDF5: {USE_HDF5}")
logging.info("=" * 60)


# ============================================
# Create image function
# ============================================
def generate_bid_rotation_image(bid_pairs, context_pairs=None, image_size=96, dpi=100, marker_size=8):
    """
    Create grayscale image of bid rotation.
    - bid_pairs: list of (x,y) for the main firm pair
    - context_pairs: optional list of (x,y) for additional context (other pairs)
    """
    if not bid_pairs and not context_pairs:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)

    # Plot main pairs (white, opaque)
    x = [p[0] for p in bid_pairs]
    y = [p[1] for p in bid_pairs]
    ax.scatter(x, y, s=marker_size, c='white', alpha=0.9, edgecolors='none')

    # Plot context pairs (gray, transparent) if provided
    if context_pairs:
        ctx_x = [p[0] for p in context_pairs]
        ctx_y = [p[1] for p in context_pairs]
        ax.scatter(ctx_x, ctx_y, s=CONTEXT_MARKER_SIZE, c='gray', alpha=CONTEXT_ALPHA, edgecolors='none')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()
    image_rgba = np.asarray(fig.canvas.buffer_rgba())
    image = image_rgba[..., :3]
    plt.cla()
    plt.clf()
    plt.close(fig)

    image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    from PIL import Image
    img_pil = Image.fromarray(image_gray.astype('uint8'))
    img_pil = img_pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return np.array(img_pil)


def get_firm_cartel_status(df):
    return df.groupby('Competitors')['Collusive_competitor'].max().to_dict()


def generate_pair_images_for_country(df, output_name, allowed_tenders=None, use_hdf5=True):
    """
    Generate pair images for a country (or a country + subset tag, e.g. "brazil_trainonly").

    Args:
        df: DataFrame with bid data (already filtered to the target country)
        output_name: base name used for output files, e.g. "brazil_trainonly" or "brazil_full".
                     This is NOT necessarily the raw country name — it also encodes which
                     tender subset was used, so trainonly and full outputs never collide.
        allowed_tenders: List of tender IDs to include (if None, use all rows in df)
        use_hdf5: Save as HDF5 if True

    Returns:
        index_df: DataFrame with columns [pair_key, label, country] for this subset.
                  'country' is the ORIGINAL country name (without the _trainonly/_full suffix)
                  so downstream scripts that group by country still work correctly.
    """
    logging.info(f"\n Processing {output_name.upper()}...")

    # --- FILTER BY ALLOWED TENDERS (to prevent data leakage) ---
    if allowed_tenders is not None:
        allowed_set = set(allowed_tenders)
        before = len(df)
        df = df[df['Tender'].isin(allowed_set)]
        after = len(df)
        logging.info(f"  Filtered to {after} bids (from {before}) using allowed_tenders "
                     f"({len(allowed_set)} tenders allowed)")
    # --- END OF FILTER ---

    logging.info(f"  Loaded {len(df)} bids, {df['Tender'].nunique()} tenders")

    if len(df) == 0:
        logging.warning(f"  No bids remain for {output_name} after filtering. Skipping.")
        return pd.DataFrame(columns=['pair_key', 'label', 'country'])

    # original country name is recovered from the 'country' column already present in df,
    # NOT from output_name (output_name may carry a _trainonly/_full suffix)
    original_country_name = df['country'].iloc[0] if 'country' in df.columns else output_name

    firm_is_cartel = get_firm_cartel_status(df)
    pair_data = defaultdict(list)
    pair_bidders = {}
    tender_pairs = defaultdict(list)

    for tender_id, tender_df in df.groupby('Tender'):
        bidders = tender_df[['Competitors', 'Bid_norm']].values
        if len(bidders) < 2:
            continue

        tender_pairs_this = []
        for i in range(len(bidders)):
            for j in range(i + 1, len(bidders)):
                bidder_a, norm_a = bidders[i]
                bidder_b, norm_b = bidders[j]
                id_a = int(float(bidder_a))
                id_b = int(float(bidder_b))
                if id_a <= id_b:
                    pair_key = (str(id_a), str(id_b))
                    pair_data[pair_key].append((norm_a, norm_b))
                    tender_pairs_this.append((pair_key, norm_a, norm_b))
                else:
                    pair_key = (str(id_b), str(id_a))
                    pair_data[pair_key].append((norm_b, norm_a))
                    tender_pairs_this.append((pair_key, norm_b, norm_a))
                if pair_key not in pair_bidders:
                    pair_bidders[pair_key] = pair_key
        tender_pairs[tender_id] = tender_pairs_this

    logging.info(f"  Found {len(pair_data)} unique bidder pairs")

    # Label each pair
    pair_labels = {}
    for pair_key in pair_bidders:
        str_id_a, str_id_b = pair_key
        id_a_int = int(str_id_a)
        id_b_int = int(str_id_b)
        is_collusive = 1 if (firm_is_cartel.get(id_a_int, 0) == 1 and firm_is_cartel.get(id_b_int, 0) == 1) else 0
        pair_labels[pair_key] = is_collusive

    # Build context
    pair_context = defaultdict(list)
    if ADD_CONTEXT:
        for tender_id, pairs_in_tender in tender_pairs.items():
            firm_norms = defaultdict(list)
            for (pk, na, nb) in pairs_in_tender:
                id_a, id_b = pk
                firm_norms[id_a].append((id_b, na, nb))
                firm_norms[id_b].append((id_a, nb, na))
            for (pk, na, nb) in pairs_in_tender:
                id_a, id_b = pk
                ctx = []
                for (other_id, norm_self, norm_other) in firm_norms[id_a]:
                    if other_id != id_b:
                        ctx.append((norm_self, norm_other))
                for (other_id, norm_self, norm_other) in firm_norms[id_b]:
                    if other_id != id_a:
                        ctx.append((norm_self, norm_other))
                ctx = list(set(ctx))
                pair_context[pk].extend(ctx)
        logging.info("  Context pairs generated for each main pair")

    # Filter pairs with too few interactions
    filtered_pair_data = {}
    filtered_pair_labels = {}
    filtered_pair_context = {}
    skipped_count = 0
    for pair_key, bid_pairs in pair_data.items():
        if len(bid_pairs) >= MIN_INTERACTIONS:
            filtered_pair_data[pair_key] = bid_pairs
            filtered_pair_labels[pair_key] = pair_labels[pair_key]
            if ADD_CONTEXT:
                filtered_pair_context[pair_key] = pair_context.get(pair_key, [])
        else:
            skipped_count += 1
    logging.info(f"  Kept {len(filtered_pair_data)} pairs (>={MIN_INTERACTIONS} interactions), "
                 f"skipped {skipped_count} sparse pairs")

    # Generate images
    images_dict = {}
    pair_keys_str = []
    labels_list = []

    for pair_key, bid_pairs in tqdm(filtered_pair_data.items(), desc=f"  Generating images ({output_name})"):
        context = filtered_pair_context.get(pair_key, []) if ADD_CONTEXT else None
        image = generate_bid_rotation_image(
            bid_pairs,
            context_pairs=context,
            image_size=IMAGE_SIZE,
            dpi=DPI,
            marker_size=MARKER_SIZE
        )
        pair_key_str = f"{pair_key[0]}_{pair_key[1]}"
        images_dict[pair_key_str] = image
        pair_keys_str.append(pair_key_str)
        labels_list.append(filtered_pair_labels[pair_key])

    if use_hdf5:
        h5_path = os.path.join(IMAGES_DIR, f"{output_name}_pair_images.h5")
        with h5py.File(h5_path, 'w') as hf:
            for pk, img in images_dict.items():
                hf.create_dataset(pk, data=img, compression='gzip', compression_opts=4)
        logging.info(f"  Saved {len(images_dict)} pair images to {h5_path}")

    index_df = pd.DataFrame({
        'pair_key': pair_keys_str,
        'label': labels_list,
        'country': original_country_name  # keep the REAL country name, not output_name
    })
    index_path = os.path.join(IMAGES_DIR, f"{output_name}_pair_labels.csv")
    index_df.to_csv(index_path, index=False)

    logging.info(f"  Label distribution: 0={labels_list.count(0)}, 1={labels_list.count(1)}")
    return index_df


def load_split(country_name):
    """Load the tender split json for a country. Returns None if not found."""
    split_path = os.path.join(SPLIT_DIR, f"{country_name}_split.json")
    if not os.path.exists(split_path):
        logging.warning(f" Split file not found for {country_name}: {split_path}")
        return None
    with open(split_path, 'r') as f:
        splits = json.load(f)
    return splits


# ============================================
# main
# ============================================
def main():
    logging.info("=" * 60)
    logging.info("Phase 2: Image Generation (Pair-level, Cross-Tender, Leakage-Safe)")
    logging.info("=" * 60)

    trainonly_labels = []
    full_labels = []

    for country_config in CONFIG['countries']:
        country_name = country_config['name']

        splits = load_split(country_name)
        if splits is None:
            logging.error(f"  Skipping {country_name}: no split file. "
                           f"Run create_tender_splits.py first.")
            continue

        cleaned_path = os.path.join(PROCESSED_DIR, f"{country_name}_cleaned.parquet")
        if not os.path.exists(cleaned_path):
            logging.error(f"Error: {cleaned_path} not found. Skip.")
            continue

        df_raw = pd.read_parquet(cleaned_path)

        # =========================================================
        # SET 1: TRAIN-ONLY — used to train the CNN (train_cnn.py)
        # =========================================================
        train_tenders = splits['train']
        logging.info(f"[TRAINONLY] {country_name}: using {len(train_tenders)} train tenders "
                     f"(out of {splits['stats']['total']} total)")
        df_trainonly = generate_pair_images_for_country(
            df_raw.copy(),
            output_name=f"{country_name}_trainonly",
            allowed_tenders=train_tenders,
            use_hdf5=USE_HDF5
        )
        if len(df_trainonly) > 0:
            trainonly_labels.append(df_trainonly)

        # =========================================================
        # SET 2: FULL — used ONLY for inference / embedding extraction
        # (extract_embeddings.py). Safe because CNN weights are frozen
        # and were fitted using the TRAIN-ONLY set above.
        # =========================================================
        all_tenders = splits['train'] + splits['val'] + splits['test']
        logging.info(f"[FULL] {country_name}: using {len(all_tenders)} tenders "
                     f"(train+val+test) for embedding extraction only")
        df_full = generate_pair_images_for_country(
            df_raw.copy(),
            output_name=f"{country_name}_full",
            allowed_tenders=all_tenders,
            use_hdf5=USE_HDF5
        )
        if len(df_full) > 0:
            full_labels.append(df_full)

    # ---- Combine TRAIN-ONLY across countries (for pooled CNN training) ----
    if trainonly_labels:
        combined_trainonly = pd.concat(trainonly_labels, ignore_index=True)
        combined_trainonly_path = os.path.join(IMAGES_DIR, "all_pairs_trainonly_labels.csv")
        combined_trainonly.to_csv(combined_trainonly_path, index=False)
        logging.info(f"\n[TRAINONLY] Total pairs across all countries: {len(combined_trainonly)}")
        logging.info(f"[TRAINONLY] Label distribution: "
                     f"0={len(combined_trainonly[combined_trainonly['label'] == 0])}, "
                     f"1={len(combined_trainonly[combined_trainonly['label'] == 1])}")
    else:
        logging.error("No TRAINONLY data was generated for any country.")

    # ---- Combine FULL across countries (for pooled embedding extraction) ----
    if full_labels:
        combined_full = pd.concat(full_labels, ignore_index=True)
        combined_full_path = os.path.join(IMAGES_DIR, "all_pairs_full_labels.csv")
        combined_full.to_csv(combined_full_path, index=False)
        logging.info(f"\n[FULL] Total pairs across all countries: {len(combined_full)}")
        logging.info(f"[FULL] Label distribution: "
                     f"0={len(combined_full[combined_full['label'] == 0])}, "
                     f"1={len(combined_full[combined_full['label'] == 1])}")
    else:
        logging.error("No FULL data was generated for any country.")

    logging.info("\n" + "=" * 60)
    logging.info(" Image generation completed!")
    logging.info(" -> Use *_trainonly_pair_images.h5 / all_pairs_trainonly_labels.csv "
                 "to TRAIN the CNN (train_cnn.py)")
    logging.info(" -> Use *_full_pair_images.h5 / all_pairs_full_labels.csv "
                 "ONLY to extract embeddings for Stage 2 (extract_embeddings.py)")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
