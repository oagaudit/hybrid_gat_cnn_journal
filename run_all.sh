#!/usr/bin/env bash
# ===========================================================================
#  run_all.sh  --  full pipeline, Stage 1 through analysis
#
#  Replaces the four ad-hoc scripts used while the experiments were being
#  developed (run_part1.sh, run_part2.sh and the two resume scripts). Each
#  step is idempotent: it is skipped when its output already exists, so the
#  script can be re-run after an interruption without redoing finished work.
#
#  Usage
#      bash run_all.sh                 # everything
#      bash run_all.sh stage1          # bid-rotation images + CNN encoder
#      bash run_all.sh stage2          # bridge + GATv2, within market
#      bash run_all.sh crossmarket     # leave-one-country-out + label sweep
#      bash run_all.sh analysis        # baselines, post-hoc, figures
#
#  Environment
#      DEVICE=cpu bash run_all.sh      # force CPU (default: auto)
#      SEED=43                         # base seed (default: 43)
#      FORCE=1                         # ignore existing outputs and redo
#
#  Total runtime is roughly 12 hours on an Apple M3 with 16 GB of memory.
#  Run it under tmux rather than nohup: closing the terminal that owns a
#  nohup job takes its file descriptors with it, and the next Python process
#  then dies with "Fatal Python error: init_sys_streams".
#      tmux new -s run
#      DEVICE=cpu bash run_all.sh 2>&1 | tee run.log
#      Ctrl-B then D to detach
# ===========================================================================
set -euo pipefail

# macOS OpenMP guard. PyTorch, scikit-learn and XGBoost each ship their own
# libomp; loading two of them into one process segfaults on Apple Silicon.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

DEVICE="${DEVICE:-auto}"
SEED="${SEED:-43}"
FORCE="${FORCE:-0}"
BRIDGE="${BRIDGE:-contextual}"
WHAT="${1:-all}"

RES=outputs/results
PAIRSET=outputs/pair_sets
INSAMPLE="$RES/in_sample"
COUNTRIES=(brazil japan usa)

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
have() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
skip() { echo "    skip (exists): $1"; }

# ---------------------------------------------------------------------------
stage1() {
  say "Stage 1a  clean the raw procurement records"
  if have data/processed/usa_cleaned.parquet; then skip data/processed
  else python src/data_preprocessing.py; fi

  say "Stage 1b  fix the tender-level splits before any image is generated"
  if have outputs/splits/usa_split.json; then skip outputs/splits
  else python src/create_tender_splits.py --seed "$SEED"; fi

  say "Stage 1c  render bid-rotation images"
  if have outputs/images/usa_pair_images.h5; then skip outputs/images
  else python src/image_generator.py; fi

  say "Stage 1d  train the CNN encoder (5 seeds)"
  if have outputs/models/pooled_m1/best_cnn_run1.pth; then skip outputs/models
  else python src/train_cnn.py --num_runs 5 --seed "$SEED" --device "$DEVICE"; fi

  say "Stage 1e  extract and cache the 64-dim pair embeddings"
  if have outputs/embeddings/insample/all_pair_embeddings_with_labels.csv
  then skip outputs/embeddings
  else python src/extract_embeddings.py; fi
}

# ---------------------------------------------------------------------------
stage2() {
  say "Stage 2a  build the tender graphs"
  if have outputs/graph_data/usa_graph.pt; then skip outputs/graph_data
  else python src/prepare_graph_data.py; fi

  say "Stage 2b  assemble ragged pair sets (reports bidder-pair coverage)"
  for scope in insample fold_brazil fold_japan fold_usa; do
    if have "$PAIRSET/$scope/usa_pair_set.pt"; then skip "$PAIRSET/$scope"; continue; fi
    python src/prepare_pair_sets.py \
      --embedding_csv "outputs/embeddings/$scope/all_pair_embeddings_with_labels.csv" \
      --output_dir "$PAIRSET/$scope"
  done

  say "Stage 2c  structural baseline (no node features at all)"
  if have "$RES/analysis/graph_leakage/graph_only_baseline.txt"
  then skip "$RES/analysis/graph_leakage"
  else python src/structural_baseline.py --pair_set_dir "$PAIRSET/insample" \
         --out_dir "$RES/analysis/graph_leakage"; fi

  say "Stage 2d  edge homophily of the co-bidding graphs"
  if have "$RES/analysis/homophily/edge_homophily.json"
  then skip "$RES/analysis/homophily"
  else python src/compute_homophily.py --pair_set_dir "$PAIRSET/insample" \
         --out_dir "$RES/analysis/homophily"; fi

  say "Stage 2e  bridge ablation, 10 seeds each"
  for b in contextual uncond mean; do
    out="$INSAMPLE/M4_${b}"
    if have "$out/summary.json"; then skip "$out"; continue; fi
    python src/train_stage2.py --pair_set_dir "$PAIRSET/insample" \
      --model_type hybrid --bridge_kind "$b" --norm_scope all \
      --num_runs 10 --seed "$SEED" --save_attention --device "$DEVICE" \
      --out_dir "$out"
  done

  say "Stage 2f  graph ablation, M2 (GATv1) and M3 (GATv2)"
  for pair in "simple_gat M2" "gatv2 M3"; do
    set -- $pair
    if have "$INSAMPLE/$2/summary.json"; then skip "$INSAMPLE/$2"; continue; fi
    python src/train_stage2.py --pair_set_dir "$PAIRSET/insample" \
      --model_type "$1" --norm_scope all --num_runs 5 --seed "$SEED" \
      --device "$DEVICE" --out_dir "$INSAMPLE/$2"
  done

  say "Stage 2g  decision threshold tuned for recall >= 0.90"
  if have "$INSAMPLE/M4_recall90/summary.json"; then skip "$INSAMPLE/M4_recall90"
  else python src/train_stage2.py --pair_set_dir "$PAIRSET/insample" \
         --model_type hybrid --bridge_kind "$BRIDGE" --norm_scope all \
         --tune_threshold --target_recall 0.90 --num_runs 5 --seed "$SEED" \
         --device "$DEVICE" --out_dir "$INSAMPLE/M4_recall90"; fi
}

# ---------------------------------------------------------------------------
crossmarket() {
  say "Cross-market  leave-one-country-out, four conditions, M4 and M3"
  for c in "${COUNTRIES[@]}"; do
    d="$PAIRSET/fold_$c"
    for cond in "none C1" "screens C2" "all C2b"; do
      set -- $cond
      out="$RES/loco/${c}_M4_$2"
      if have "$out/summary.json"; then skip "$out"; continue; fi
      python src/train_stage2.py --pair_set_dir "$d" --model_type hybrid \
        --bridge_kind "$BRIDGE" --test_country "$c" --norm_scope "$1" \
        --num_runs 5 --seed "$SEED" --device "$DEVICE" --out_dir "$out"
    done
    for m in "hybrid M4" "gatv2 M3"; do
      set -- $m
      out="$RES/loco/${c}_$2_C3"
      if have "$out/summary.json"; then skip "$out"; continue; fi
      extra=""; [ "$1" = "hybrid" ] && extra="--bridge_kind $BRIDGE"
      python src/train_stage2.py --pair_set_dir "$d" --model_type "$1" $extra \
        --test_country "$c" --fine_tune_ratio 0.15 --norm_scope all \
        --num_runs 5 --seed "$SEED" --device "$DEVICE" --out_dir "$out"
    done
    out="$RES/loco/${c}_M3_C2b"
    if have "$out/summary.json"; then skip "$out"
    else python src/train_stage2.py --pair_set_dir "$d" --model_type gatv2 \
           --test_country "$c" --norm_scope all --num_runs 5 --seed "$SEED" \
           --device "$DEVICE" --out_dir "$out"; fi
  done

  say "Cross-market  target-market label budget sweep (0.15 comes from C3)"
  for c in "${COUNTRIES[@]}"; do
    for r in 0.05 0.10 0.20 0.25; do
      for m in "hybrid M4" "gatv2 M3"; do
        set -- $m
        out="$RES/sweep/${c}_$2_ft${r}"
        if have "$out/summary.json"; then skip "$out"; continue; fi
        extra=""; [ "$1" = "hybrid" ] && extra="--bridge_kind $BRIDGE"
        python src/train_stage2.py --pair_set_dir "$PAIRSET/fold_$c" \
          --model_type "$1" $extra --test_country "$c" --fine_tune_ratio "$r" \
          --norm_scope all --num_runs 3 --seed "$SEED" --device "$DEVICE" \
          --out_dir "$out"
      done
    done
  done
}

# ---------------------------------------------------------------------------
analysis() {
  M4="$INSAMPLE/M4_${BRIDGE}"
  FEATS="$RES/learned_node_features.pt"

  say "Analysis  export the 135-dim features the trained bridge produces"
  if have "$FEATS"; then skip "$FEATS"
  else python src/export_node_features.py --pair_set_dir "$PAIRSET/insample" \
         --model_ckpt "$M4/model_run1.pt" --bridge_kind "$BRIDGE" \
         --norm_scope all --out "$FEATS"; fi

  say "Analysis  classical baselines on the same features"
  if have "$INSAMPLE/classical_full/baseline_summary.csv"
  then skip "$INSAMPLE/classical_full"
  else python src/evaluate_baseline.py --node_feature_pt "$FEATS" --n_runs 5 \
         --out_dir "$INSAMPLE/classical_full"; fi
  if have "$INSAMPLE/classical_screens/baseline_summary.csv"
  then skip "$INSAMPLE/classical_screens"
  else python src/evaluate_baseline.py --node_feature_pt "$FEATS" --n_runs 5 \
         --screens_only --out_dir "$INSAMPLE/classical_screens"; fi

  say "Analysis  GNN baselines, and the MLP that removes the graph entirely"
  for m in gcn sage mlp; do
    if have "$INSAMPLE/$m/summary.json"; then skip "$INSAMPLE/$m"; continue; fi
    python src/train_stage2.py --pair_set_dir "$PAIRSET/insample" \
      --model_type "$m" --node_feature_pt "$FEATS" --norm_scope none \
      --num_runs 5 --seed "$SEED" --device "$DEVICE" --out_dir "$INSAMPLE/$m"
  done
  if have "$INSAMPLE/mlp_screens/summary.json"; then skip "$INSAMPLE/mlp_screens"
  else python src/train_stage2.py --pair_set_dir "$PAIRSET/insample" \
         --model_type mlp --norm_scope all --num_runs 5 --seed "$SEED" \
         --device "$DEVICE" --out_dir "$INSAMPLE/mlp_screens"; fi

  say "Analysis  cost curves, bootstrap intervals, paired comparisons"
  python src/posthoc_analysis.py --out_dir "$RES/analysis/posthoc" \
    --model M4="$M4/test_probabilities.csv" \
    --model M4_mean="$INSAMPLE/M4_mean/test_probabilities.csv" \
    --model M3="$INSAMPLE/M3/test_probabilities.csv" \
    --model M2="$INSAMPLE/M2/test_probabilities.csv" \
    --model MLP135="$INSAMPLE/mlp/test_probabilities.csv" \
    --model MLP7="$INSAMPLE/mlp_screens/test_probabilities.csv" \
    --model GCN="$INSAMPLE/gcn/test_probabilities.csv" \
    --model RF="$INSAMPLE/classical_full/RF_tuned/test_probabilities.csv" \
    --model HistGB="$INSAMPLE/classical_full/HistGB/test_probabilities.csv" \
    --compare M4:M3 --compare M4:RF --compare M4:HistGB \
    --compare M4:MLP135 --compare M3:MLP7 --compare M3:M2

  say "Analysis  what the bridge attends to"
  python src/analyze_attention.py --pair_set_dir "$PAIRSET/insample" \
    --model_ckpt "$M4/model_run1.pt" --bridge_kind "$BRIDGE" \
    --norm_scope all --out_dir "$RES/analysis/attention"

  say "Analysis  figures for the manuscript"
  python src/make_figures.py --results_root "$RES" \
    --attention_dir "$RES/analysis/attention" --out_dir "$RES/figures"
}

# ---------------------------------------------------------------------------
case "$WHAT" in
  stage1)      stage1 ;;
  stage2)      stage2 ;;
  crossmarket) crossmarket ;;
  analysis)    analysis ;;
  all)         stage1; stage2; crossmarket; analysis ;;
  *) echo "usage: bash run_all.sh [all|stage1|stage2|crossmarket|analysis]"; exit 1 ;;
esac

say "done  $(date)"
