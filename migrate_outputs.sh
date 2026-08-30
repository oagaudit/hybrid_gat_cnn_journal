#!/usr/bin/env bash
# ===========================================================================
#  migrate_outputs.sh
#
#  Copies the results from the old project into the layout that the cleaned
#  pipeline expects, so that run_all.sh recognises finished work and skips
#  it instead of recomputing 12 hours of experiments.
#
#  The old tree grew during development and uses names that no longer match:
#      outputs/v2/in_sample/M4_contextual_norm  ->  outputs/results/in_sample/M4_contextual
#      outputs/v2/loco, sweep, analysis         ->  outputs/results/...
#  The "_norm" suffix existed only while both a normalised and an
#  unnormalised run were being kept side by side. The cleaned pipeline always
#  normalises, so the suffix is dropped.
#
#  Usage, run from inside the new repository:
#      bash migrate_outputs.sh "../hybrid_gat_cnn_journal_bk copy"
#
#  It copies rather than moves, so the old project is left untouched.
#  Nothing it writes is tracked by git: outputs/ and data/ are ignored.
# ===========================================================================
set -euo pipefail

OLD="${1:-}"
if [ -z "$OLD" ] || [ ! -d "$OLD" ]; then
  echo "usage: bash migrate_outputs.sh <path to the old project>"
  echo
  echo "example:"
  echo '  bash migrate_outputs.sh "../hybrid_gat_cnn_journal_bk copy"'
  exit 1
fi

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
copy() {  # copy $1 to $2 only if the source exists
  if [ -e "$OLD/$1" ]; then
    mkdir -p "$(dirname "$2")"
    cp -R "$OLD/$1" "$2"
    printf '    %-52s -> %s\n' "$1" "$2"
  else
    printf '    %-52s   (absent, skipped)\n' "$1"
  fi
}

say "Reading from: $OLD"

# --- Stage 1 products, names unchanged ---------------------------------
say "Stage 1 products"
for d in data/processed outputs/splits outputs/images outputs/models \
         outputs/embeddings outputs/graph_data outputs/pair_sets; do
  copy "$d" "$d"
done

# --- Stage 2 results, renamed ------------------------------------------
say "Within-market runs"
for pair in "M4_contextual_norm:M4_contextual" \
            "M4_uncond_norm:M4_uncond" \
            "M4_mean_norm:M4_mean" \
            "M2_norm:M2" "M3_norm:M3" \
            "M4_recall90:M4_recall90" \
            "gcn:gcn" "sage:sage" "mlp:mlp" "mlp_screens:mlp_screens" \
            "classical_full:classical_full" \
            "classical_screens:classical_screens"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  copy "outputs/v2/in_sample/$src" "outputs/results/in_sample/$dst"
done

say "Cross-market runs and analyses"
for d in loco sweep analysis figures; do
  copy "outputs/v2/$d" "outputs/results/$d"
done
copy "outputs/v2/learned_node_features.pt" \
     "outputs/results/learned_node_features.pt"

# --- what the pipeline will look for -----------------------------------
say "Checking what run_all.sh will now find"
check() {
  if [ -e "$1" ]; then printf '    found   %s\n' "$1"
  else                 printf '    MISSING %s\n' "$1"; fi
}
check data/processed/usa_cleaned.parquet
check outputs/splits/usa_split.json
check outputs/images/usa_pair_images.h5
check outputs/embeddings/insample/all_pair_embeddings_with_labels.csv
check outputs/pair_sets/insample/usa_pair_set.pt
check outputs/results/in_sample/M4_contextual/summary.json
check outputs/results/in_sample/M3/summary.json
check outputs/results/learned_node_features.pt

cat <<'MSG'

Anything marked MISSING will simply be recomputed by run_all.sh; the
script is idempotent and only redoes steps whose output is absent.

Next:
    bash run_all.sh analysis        # ~20 minutes, exercises most of the code
    bash run_all.sh                 # everything, skipping what is present
MSG
