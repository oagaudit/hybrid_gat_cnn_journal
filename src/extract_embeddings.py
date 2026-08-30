"""
src/scripts/extract_embeddings.py
Extract 64-dim embeddings from a pre-trained (frozen) CNN for all bid rotation pairs.

=====================================================================
FIXED VERSION — this script ONLY reads images from the
"*_full_pair_images.h5" / "*_full_pair_labels.csv" files produced by
the fixed image_generator.py (built from ALL tenders: train+val+test).

This is safe (does NOT leak data) because:
  - The CNN weights used here must come from a checkpoint trained by
    train_cnn.py on "*_trainonly_*" data only (never fitted on val/test
    tenders, and for LOCO folds, never fitted on the target country at all).
  - This script performs a forward pass only (model.eval(), no gradient
    update, no backprop) — it does not change the CNN in any way.

extract_embeddings.py must NEVER be pointed at "*_trainonly_*" files for
this purpose, since those exclude val/test tenders and Stage 2 needs
node features for every tender in the graph.
=====================================================================

Supports:
- Single country (--country brazil)
- All countries found in the images_dir (default), each read from its
  own per-country "_full" file (no merged 'all' h5 required)
- Automatically reads model_type from checkpoint
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_model import create_cnn_model

REQUIRED_SUFFIX = "full"


class PairDataset(Dataset):
    def __init__(self, h5_path, pair_keys):
        self.h5_path = h5_path
        self.pair_keys = pair_keys
        self.h5 = None

    def __len__(self):
        return len(self.pair_keys)

    def __getitem__(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, 'r')
        key = self.pair_keys[idx]
        img = self.h5[key][:]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        return img, key

    def __del__(self):
        if self.h5 is not None:
            self.h5.close()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        try:
            torch.randn(1).to("mps")
            return torch.device("mps")
        except:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def assert_full(*paths):
    """Safeguard: refuse to proceed if any resolved path does not carry the
    'full' tag. Extracting embeddings from a trainonly file would silently
    produce node features for only a subset of tenders, breaking Stage 2
    graph construction (and defeats the purpose of this script)."""
    for p in paths:
        base = os.path.basename(p).lower()
        if REQUIRED_SUFFIX not in base or "trainonly" in base:
            print(f"REFUSING TO RUN: '{p}' does not look like a FULL file "
                  f"(expects '{REQUIRED_SUFFIX}' in filename, and must NOT be "
                  f"a 'trainonly' file). Stage 2 needs embeddings for every "
                  f"tender (train+val+test), not just the training subset.")
            sys.exit(1)


def get_country_full_paths(images_dir, country):
    h5_path = os.path.join(images_dir, f'{country}_full_pair_images.h5')
    csv_path = os.path.join(images_dir, f'{country}_full_pair_labels.csv')
    if not os.path.exists(h5_path) or not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Full-set files not found for '{country}':\n"
            f"  {h5_path}\n  {csv_path}\n"
            f"Run image_generator.py first (it creates the *_full_* files "
            f"from all tenders: train+val+test)."
        )
    return h5_path, csv_path


def load_model_for_country(args, device, country=None):
    """
    Load the CNN checkpoint to use for a given country.
    If args.model_file_map provides a per-country override (used for LOCO
    fold extraction, where each target country must be scored with a CNN
    that never saw that country during training), use that checkpoint.
    Otherwise fall back to args.model_path (single shared checkpoint).
    """
    model_path = args.model_path
    if args.model_file_map is not None and country is not None:
        override = args.model_file_map.get(country)
        if override:
            model_path = os.path.join(args.model_dir, override)

    checkpoint = torch.load(model_path, map_location=device)

    model_type = None
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        if isinstance(ckpt_args, dict):
            model_type = ckpt_args.get('model_type')
        elif hasattr(ckpt_args, 'model_type'):
            model_type = ckpt_args.model_type
    if not model_type and args.model_type:
        model_type = args.model_type
    if not model_type:
        model_type = 'default'

    print(f"  Using model_type: {model_type} (checkpoint: {model_path})")

    model = create_cnn_model(model_type, embedding_dim=64, dropout_rate=0.5)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        val_f1 = checkpoint.get('val_f1', 'N/A')
    else:
        model.load_state_dict(checkpoint)
        val_f1 = 'N/A'

    model = model.to(device)
    model.eval()  # inference only — weights are frozen, never updated here
    print(f"  Loaded model from {model_path} (best val F1: {val_f1})")
    return model


def extract_for_country(country, args, device):
    """Extract embeddings for a single country, reading from its FULL
    (train+val+test) image/label files."""
    print(f"\n--- Processing {country} (full set) ---")

    try:
        h5_path, csv_path = get_country_full_paths(args.images_dir, country)
    except FileNotFoundError as e:
        print(str(e))
        return

    assert_full(h5_path, csv_path)

    df = pd.read_csv(csv_path)
    pair_keys = df['pair_key'].tolist()
    print(f"Total pairs: {len(pair_keys)}")

    model = load_model_for_country(args, device, country=country)

    dataset = PairDataset(h5_path, pair_keys)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    embeddings_list = []
    keys_list = []
    with torch.no_grad():
        for images, keys in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device)
            emb = model.extract_embedding(images)
            embeddings_list.append(emb.cpu().numpy())
            keys_list.extend(keys)

    embeddings = np.vstack(embeddings_list)

    out_npy = os.path.join(args.output_dir, f'{country}_pair_embeddings.npy')
    np.save(out_npy, embeddings)

    out_df = df.copy()
    # NOTE: image_generator.py writes pair_key WITHOUT a country prefix
    # (e.g. "123_456"), keeping the country separately in its own 'country'
    # column. But prepare_pair_sets.py builds its lookup keys in the
    # form "{country_name}_{id_a}_{id_b}" (WITH prefix) when aggregating
    # pair embeddings per tender. If we don't add the prefix here, every
    # lookup in prepare_pair_sets.py silently misses and falls back to
    # a zero vector for every pair, effectively zeroing out the 128-dim
    # visual embedding portion of every tender's node feature.
    out_df['pair_key'] = f"{country}_" + out_df['pair_key'].astype(str)
    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    out_df = pd.concat([out_df.reset_index(drop=True), emb_df], axis=1)
    out_csv = os.path.join(args.output_dir, f'{country}_pair_embeddings_with_labels.csv')
    out_df.to_csv(out_csv, index=False)

    print(f"Saved {len(pair_keys)} embeddings to {out_npy} and {out_csv}")
    return out_df


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from a frozen, trained CNN (Stage 1 -> Stage 2 bridge)')
    parser.add_argument('--country', type=str, default=None,
                        help='Single country name to process. If omitted, processes every country '
                             'found in all_pairs_full_labels.csv (or --countries_list if given).')
    parser.add_argument('--countries_list', type=str, nargs='+', default=None,
                        help='Explicit list of countries to process (overrides auto-detection).')
    parser.add_argument('--images_dir', type=str, default='./outputs/images',
                        help='Directory with the *_full_pair_images.h5 / *_full_pair_labels.csv files')
    parser.add_argument('--model_dir', type=str, default='./outputs/models',
                        help='Directory containing model checkpoint(s)')
    parser.add_argument('--model_file', type=str, default='best_cnn.pth',
                        help='Default model weights file name (used unless overridden per-country)')
    parser.add_argument('--model_file_map', type=str, default=None,
                        help='Optional JSON string mapping country -> checkpoint filename, for LOCO fold '
                             'extraction, e.g. \'{"brazil": "best_cnn_fold_excl_brazil.pth"}\'. '
                             'The checkpoint for a target country must have been trained WITHOUT that '
                             'country in its training data.')
    parser.add_argument('--output_dir', type=str, default='./outputs/embeddings',
                        help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Use 0 to avoid pickle issues on macOS')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Override model type (only if not in checkpoint)')
    args = parser.parse_args()

    args.model_path = os.path.join(args.model_dir, args.model_file)
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    if args.model_file_map is not None:
        import json
        args.model_file_map = json.loads(args.model_file_map)

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    if args.countries_list:
        countries = args.countries_list
    elif args.country:
        countries = [args.country]
    else:
        combined_labels_path = os.path.join(args.images_dir, 'all_pairs_full_labels.csv')
        if not os.path.exists(combined_labels_path):
            print(f"{combined_labels_path} not found. Please specify --country or --countries_list, "
                  f"or run image_generator.py first.")
            sys.exit(1)
        df_all = pd.read_csv(combined_labels_path)
        countries = sorted(df_all['country'].unique().tolist())
        print(f"Processing all countries found in all_pairs_full_labels.csv: {countries}")

    all_out = []
    for country in countries:
        out_df = extract_for_country(country, args, device)
        if out_df is not None:
            all_out.append(out_df)

    if all_out:
        combined = pd.concat(all_out, ignore_index=True)
        combined_csv = os.path.join(args.output_dir, 'all_pair_embeddings_with_labels.csv')
        combined.to_csv(combined_csv, index=False)
        print(f"\nSaved combined embeddings for {len(countries)} countries to {combined_csv}")

    print("\nExtraction complete. Ready for Stage 2 (GNN) node-feature preparation.")


if __name__ == '__main__':
    main()
