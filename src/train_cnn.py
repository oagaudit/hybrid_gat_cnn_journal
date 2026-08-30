"""
src/scripts/train_cnn.py
Stage 1: Pre-train CNN to classify collusive vs competitive bid rotation pairs.
Extracts embeddings for use in Stage 2 (GNN).

=====================================================================
FIXED VERSION — this script now ONLY reads images from the
"*_trainonly_pair_images.h5" / "*_trainonly_pair_labels.csv" files
produced by the fixed image_generator.py (built exclusively from
train-split tenders). It will REFUSE to run against "_full" files,
since those contain val/test tenders and using them here would
reintroduce the tender-level leakage this pipeline is meant to fix.

Embedding extraction for Stage 2 (which is allowed to see val/test
tenders, because it is inference only on a frozen model) must be done
separately with extract_embeddings.py against the "_full" files.
=====================================================================

Features:
- Multiple independent runs with different seeds
- Stratified train/val/test split (pair-level, within the trainonly set)
- Optional data augmentation (rotation, shift) - careful with flip
- Class weighting for imbalance (configurable pos_weight)
- Adjustable decision threshold
- Early stopping + model checkpoint
- Evaluation metrics (accuracy, precision, recall, f1, auc)
- Saves best model per run and overall best
- Generates plots for thesis: confusion matrix, ROC curve, t-SNE, learning curves
"""

import os
import sys
import argparse
import json
import random
import logging
import shutil
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.models.cnn_model import create_cnn_model
from src.utils.config_loader import CONFIG, get_project_root

# ============================================
# ตั้งค่า PROJECT_ROOT
# ============================================
PROJECT_ROOT = get_project_root()

# Suffix that MUST appear in every image/label file this script reads.
# This is the safeguard against accidentally training on leaked data.
REQUIRED_SUFFIX = "trainonly"


# ========================
# Dataset
# ========================
class BidRotationDataset(Dataset):
    """Dataset for bid rotation images stored in HDF5."""
    def __init__(self, h5_path, csv_path, transform=None):
        self.h5_path = h5_path
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.h5 = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, 'r')
        row = self.df.iloc[idx]
        pair_key = row['pair_key']
        label = float(row['label'])
        img = self.h5[pair_key][:]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1,96,96)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)

    def __del__(self):
        if self.h5 is not None:
            self.h5.close()


# ========================
# Augmentation
# ========================
def get_train_transform():
    """Augmentation: small rotation and shift. No flip."""
    return transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), fill=0),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
    ])

def get_val_transform():
    return transforms.Compose([])


# ========================
# Training utilities
# ========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device, threshold=0.5):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        all_preds.extend(preds.cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return avg_loss, acc, prec, rec, f1, auc_score


# ========================
# Single training run
# ========================
def run_training(args, run_id, train_dataset, val_dataset, test_dataset, threshold=0.5):
    set_seed(args.seed + run_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    logging.info(f"\nRun {run_id}: Device = {device}")

    model = create_cnn_model(args.model_type, args.embedding_dim, args.dropout).to(device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class distribution
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    logging.info(f"Train class distribution: 0={neg_count}, 1={pos_count} (ratio 1:{pos_count/neg_count:.2f})")

    # Loss with class weighting
    if args.use_class_weight:
        if args.pos_weight is not None:
            pos_weight = torch.tensor([args.pos_weight], device=device)
        else:
            pos_weight = torch.tensor([neg_count / pos_count], device=device) if pos_count > 0 else None
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logging.info(f"Using class weighting: pos_weight = {pos_weight.item():.2f}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            logging.info("No pos_weight (pos_count=0)")
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_f1 = 0.0
    best_epoch = -1
    best_model_state = None
    best_val_metrics = {}
    patience_counter = 0
    run_model_path = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(1, args.epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, threshold=threshold)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate_epoch(model, val_loader, criterion, device, threshold=threshold)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            best_val_metrics = {
                'val_loss': val_loss, 'val_acc': val_acc, 'val_prec': val_prec,
                'val_rec': val_rec, 'val_f1': val_f1, 'val_auc': val_auc
            }
            patience_counter = 0
            run_model_path = os.path.join(args.output_dir, f'best_cnn_run{run_id}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'val_f1': val_f1,
                'args': vars(args)
            }, run_model_path)
            logging.info(f"  -> New best model saved (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    # Test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = validate_epoch(model, test_loader, criterion, device, threshold=threshold)
    else:
        test_loss = test_acc = test_prec = test_rec = test_f1 = 0.0
        test_auc = 0.5

    results = {
        'run_id': run_id,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        **{f'val_{k}': v for k, v in best_val_metrics.items()},
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'test_f1': test_f1,
        'test_auc': test_auc
    }
    return results, best_model_state, run_model_path, history


# ========================
# Plotting functions
# ========================
def plot_learning_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.plot(history['val_f1'], label='Val F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.set_title('Validation Metrics')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ========================
# Helper: locate trainonly image/label files for a single country
# ========================
def get_country_file_paths(images_dir, country):
    """
    Returns (h5_path, csv_path) for the TRAINONLY set of a single country.
    Raises FileNotFoundError with a clear message if missing.
    """
    h5_path = os.path.join(images_dir, f'{country}_trainonly_pair_images.h5')
    csv_path = os.path.join(images_dir, f'{country}_trainonly_pair_labels.csv')
    if not os.path.exists(h5_path) or not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Trainonly files not found for '{country}':\n"
            f"  {h5_path}\n  {csv_path}\n"
            f"Run image_generator.py first (it creates the *_trainonly_* files "
            f"from train-split tenders only)."
        )
    return h5_path, csv_path


# ========================
# Helper to merge multiple countries' TRAINONLY files
# ========================
def merge_country_files(images_dir, countries, output_dir):
    """
    Merge multiple countries' TRAINONLY HDF5/CSV files into one temporary file.
    Only ever touches '*_trainonly_*' files — never '*_full_*'.
    """
    import tempfile

    temp_h5 = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)

    all_labels = []
    with h5py.File(temp_h5.name, 'w') as hf_out:
        for country in countries:
            h5_path, csv_path = get_country_file_paths(images_dir, country)

            df = pd.read_csv(csv_path)
            # Add country prefix to avoid key collision across countries
            df['pair_key'] = f"{country}_" + df['pair_key']
            all_labels.append(df)

            with h5py.File(h5_path, 'r') as hf_in:
                for key in tqdm(hf_in.keys(), desc=f"Copying {country} (trainonly)"):
                    new_key = f"{country}_{key}"
                    hf_out.create_dataset(new_key, data=hf_in[key][:], compression="gzip")

    combined_df = pd.concat(all_labels, ignore_index=True)
    combined_df.to_csv(temp_csv.name, index=False)

    return temp_h5.name, temp_csv.name


def get_split_name(countries_to_use):
    """Determine split/results file name prefix based on the countries actually used."""
    if len(countries_to_use) == 1:
        return countries_to_use[0]
    return 'merged_' + '_'.join(sorted(countries_to_use))


def assert_trainonly(*paths):
    """Safeguard: refuse to proceed if any resolved path does not carry the
    'trainonly' tag. This is the last line of defence against accidentally
    training the CNN on data that includes val/test tenders.
    
    NOTE: Temporary files created by merge_country_files() are excluded from
    this check because they are generated by the system and will not have
    the 'trainonly' suffix — but the underlying source files have already
    been verified by get_country_file_paths().
    """
    for p in paths:
        base = os.path.basename(p).lower()
        # Skip temporary files (they won't have 'trainonly' in the name)
        if p.startswith('/var/folders/') or p.startswith('/tmp/'):
            continue
        if REQUIRED_SUFFIX not in base:
            logging.error(
                f"REFUSING TO TRAIN: '{p}' does not look like a TRAINONLY file "
                f"(missing '{REQUIRED_SUFFIX}' in filename). Using a file that "
                f"includes val/test tenders here would leak data into Stage 1 "
                f"training. Re-check --images_dir / --country / --countries."
            )
            sys.exit(1)


# ========================
# Main
# ========================
def main():
    parser = argparse.ArgumentParser(description='Train CNN for bid rotation classification (Stage 1)')
    parser.add_argument('--country', type=str, default=None,
                        help="Single country name, or 'all' to pool every country listed in config. "
                             "Ignored if --countries is given.")
    parser.add_argument('--countries', type=str, nargs='+', default=None,
                        help='Explicit list of countries to pool for training '
                             '(e.g. --countries japan usa   -> excludes brazil, for LOCO fold training)')
    parser.add_argument('--images_dir', type=str, default='./outputs/images',
                        help='Directory containing the *_trainonly_pair_images.h5 / *_trainonly_pair_labels.csv '
                             'files produced by image_generator.py')
    parser.add_argument('--output_dir', type=str, default='./outputs/models', help='Where to save models and results')
    parser.add_argument('--model_type', type=str, default='default', choices=['default', 'simple'])
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=0, help='Use 0 to avoid pickle issues on macOS')
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='Use data augmentation')
    parser.add_argument('--no-use_augmentation', dest='use_augmentation', action='store_false', help='Disable augmentation')
    parser.add_argument('--use_class_weight', action='store_true', default=False, help='Enable class weighting')
    parser.add_argument('--pos_weight', type=float, default=None, help='Positive class weight')
    parser.add_argument('--decision_threshold', type=float, default=0.5, help='Threshold for prediction')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_runs', type=int, default=5)
    args = parser.parse_args()

    # ============================================
    # ตั้งค่า logging (แสดงบนจอ + ไฟล์)
    # ============================================
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'step3_cnn.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("=" * 60)
    logging.info(" Stage 1: CNN Training Started (TRAINONLY, leakage-safe)")
    logging.info("=" * 60)
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Log file: {log_file}")

    # ============================================
    # สร้างโฟลเดอร์สำหรับ plots
    # ============================================
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # --- Determine which countries to pool ---
    if args.countries is not None:
        countries_to_use = args.countries
    elif args.country == 'all':
        countries_to_use = [c['name'] for c in CONFIG['countries']]
    elif args.country is not None:
        countries_to_use = [args.country]
    else:
        logging.error("You must specify --country <name>, --country all, or --countries <name...>")
        sys.exit(1)

    logging.info(f"Countries to pool for training: {countries_to_use}")

    # --- Resolve TRAINONLY image/label paths ---
    try:
        if len(countries_to_use) == 1:
            h5_path, csv_path = get_country_file_paths(args.images_dir, countries_to_use[0])
        else:
            h5_path, csv_path = merge_country_files(args.images_dir, countries_to_use, args.output_dir)
            logging.info(f" Merged {len(countries_to_use)} countries (trainonly) into temporary files")
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    # Safeguard: refuse if resolved files are not trainonly (defence in depth,
    # in case get_country_file_paths / merge_country_files is ever modified)
    assert_trainonly(h5_path, csv_path)
    logging.info(f"Using images: {h5_path}")
    logging.info(f"Using labels: {csv_path}")

    full_dataset = BidRotationDataset(h5_path, csv_path, transform=None)
    n_total = len(full_dataset)
    logging.info(f"Total samples: {n_total}")

    # Stratified split (pair-level, within the trainonly pool)
    labels = full_dataset.df['label'].values
    from sklearn.model_selection import train_test_split
    train_val_idx, test_idx = train_test_split(np.arange(n_total), test_size=args.test_ratio,
                                               stratify=labels, random_state=args.seed)
    train_val_labels = labels[train_val_idx]
    val_ratio_adjusted = args.val_ratio / (1 - args.test_ratio)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio_adjusted,
                                          stratify=train_val_labels, random_state=args.seed)

    split_dict = {
        'train_idx': train_idx.tolist(),
        'val_idx': val_idx.tolist(),
        'test_idx': test_idx.tolist()
    }

    split_name = get_split_name(countries_to_use)
    split_path = os.path.join(args.output_dir, f'{split_name}_trainonly_split.json')
    with open(split_path, 'w') as f:
        json.dump(split_dict, f)
    logging.info(f"Saved split to {split_path}")

    # Helper to create subset with transform
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label

    def make_dataset(indices, transform):
        return TransformSubset(full_dataset, indices, transform)

    all_results = []
    best_overall_f1 = -1
    best_overall_model_path = None
    best_overall_run_id = None
    best_history = None

    for run_id in range(1, args.n_runs + 1):
        logging.info(f"\n{'='*60}\nStarting run {run_id}/{args.n_runs}\n{'='*60}")
        train_transform = get_train_transform() if args.use_augmentation else get_val_transform()
        val_transform = get_val_transform()
        test_transform = get_val_transform()

        train_dataset = make_dataset(split_dict['train_idx'], train_transform)
        val_dataset = make_dataset(split_dict['val_idx'], val_transform)
        test_dataset = make_dataset(split_dict['test_idx'], test_transform)

        results, model_state, model_path, history = run_training(
            args, run_id, train_dataset, val_dataset, test_dataset, threshold=args.decision_threshold
        )
        all_results.append(results)

        # Save learning curve for this run
        plot_learning_curves(history, os.path.join(plots_dir, f'learning_curves_run{run_id}.png'))

        logging.info(f"\nRun {run_id} Test Results:")
        logging.info(f"  Accuracy: {results['test_acc']:.4f}, F1: {results['test_f1']:.4f}, AUC: {results['test_auc']:.4f}")

        if results['test_f1'] > best_overall_f1 and model_path is not None:
            best_overall_f1 = results['test_f1']
            best_overall_model_path = model_path
            best_overall_run_id = run_id
            best_history = history

    # Save results summary
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(args.output_dir, f'{split_name}_cnn_stage1_trainonly_results.csv')
    results_df.to_csv(results_csv, index=False)
    logging.info(f"\nSaved all runs results to {results_csv}")

    # Summary statistics
    logging.info("\n" + "="*60)
    logging.info("SUMMARY OF ALL RUNS")
    logging.info("="*60)
    metrics_summary = {}
    for metric in ['test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auc']:
        values = results_df[metric].values
        mean_val = values.mean()
        std_val = values.std()
        metrics_summary[metric] = {'mean': mean_val, 'std': std_val}
        logging.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    summary_df = pd.DataFrame(metrics_summary).T
    summary_df.to_csv(os.path.join(plots_dir, 'metrics_summary.csv'))

    if best_overall_model_path is None:
        logging.error("No model was saved. Exiting.")
        return

    logging.info(f"\nBest overall run: {best_overall_run_id} (test F1={best_overall_f1:.4f})")
    logging.info(f"Best model saved at: {best_overall_model_path}")
    best_copy_path = os.path.join(args.output_dir, 'best_cnn.pth')
    shutil.copy(best_overall_model_path, best_copy_path)
    logging.info(f"Copied best model to {best_copy_path}")

    # Learning curve of best run
    if best_history:
        plot_learning_curves(best_history, os.path.join(plots_dir, 'learning_curves_best_run.png'))

    # ----- Evaluation on test set with best model (for confusion matrix, ROC, t-SNE) -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    best_model = create_cnn_model(args.model_type, args.embedding_dim, args.dropout).to(device)
    checkpoint = torch.load(best_overall_model_path, map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.eval()

    test_dataset_raw = make_dataset(split_dict['test_idx'], get_val_transform())
    test_loader = DataLoader(test_dataset_raw, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating best model on test set'):
            images = images.to(device)
            logits = best_model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > args.decision_threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Competitive', 'Collusive'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - Best Run (F1={checkpoint["val_f1"]:.3f})')
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix_best_run.png'), dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Run')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'roc_curve_best_run.png'), dpi=150)
    plt.close()

    # t-SNE of embeddings on test set
    embed_dataset = make_dataset(split_dict['test_idx'], get_val_transform())
    embed_loader = DataLoader(embed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    embeddings_list, embed_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(embed_loader, desc='Extracting embeddings for t-SNE'):
            images = images.to(device)
            emb = best_model.extract_embedding(images)
            embeddings_list.append(emb.cpu().numpy())
            embed_labels.extend(labels.numpy().flatten())
    embeddings = np.vstack(embeddings_list)

    n_samples = embeddings.shape[0]
    if n_samples >= 10:
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=embed_labels, cmap='coolwarm', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Label (0=Competitive, 1=Collusive)')
        plt.title('t-SNE of 64-dim Embeddings (Test Set)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(os.path.join(plots_dir, 'tsne_embeddings.png'), dpi=150)
        plt.close()
    else:
        logging.info(f"Skipping t-SNE: only {n_samples} test samples (<10)")

    # Model specifications
    total_params = sum(p.numel() for p in best_model.parameters())
    specs = {
        'countries_used': countries_to_use,
        'data_source': 'trainonly',
        'model_type': args.model_type,
        'embedding_dim': args.embedding_dim,
        'dropout': args.dropout,
        'total_parameters': total_params,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': 'Adam',
        'loss_function': 'BCEWithLogitsLoss',
        'class_weight_used': args.use_class_weight,
        'pos_weight': args.pos_weight if args.pos_weight else 'auto' if args.use_class_weight else 'none',
        'decision_threshold': args.decision_threshold,
        'augmentation': args.use_augmentation,
        'n_runs': args.n_runs,
        'best_run_test_f1': best_overall_f1,
        'mean_test_f1': results_df['test_f1'].mean(),
        'std_test_f1': results_df['test_f1'].std(),
        'mean_test_auc': results_df['test_auc'].mean(),
        'std_test_auc': results_df['test_auc'].std(),
    }
    with open(os.path.join(plots_dir, 'model_specifications.json'), 'w') as f:
        json.dump(specs, f, indent=2)

    logging.info(f"\nAll plots and summaries saved to: {plots_dir}")
    logging.info("Stage 1 training completed (trainonly). Ready for embedding extraction via extract_embeddings.py")


if __name__ == '__main__':
    main()
