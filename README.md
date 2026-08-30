# When does visual information improve cross-market bid-rigging detection?

Code for the manuscript

> **When does visual information improve cross-market bid-rigging detection?
> Evidence from three procurement markets**
> Mati Nakphon, University of Europe for Applied Sciences, Potsdam
> `mati.nakphon@ue-germany.de`

A convolutional encoder over bid-rotation images is joined to a graph
attention network over a tender-level co-bidding graph by a trainable
attention module that uses statistical screens as its query. Cleaning
leaves 3,414 tenders across Brazil, Japan and the United States; 92 US
tenders share no bidder with any other and cannot be reached by message
passing, so the analysed set is 3,322.

## What the experiments found

The paper separates three sources of predictive signal and asks how much
each adds once the other two are present:

- **statistical** — seven screens over one tender's bid distribution
- **visual** — CNN embeddings of how a firm pair has bid over repeated meetings
- **structural** — which tenders share a bidder, ignoring bid values

**Within a market the visual signal adds nothing measurable.** A GATv2 on
the seven screens matches the full 135-dimensional model
(ΔF1 = −0.001, 95% CI [−0.046, 0.047]). A neighbour-voting baseline using
*no node features at all* reaches PR-AUC 0.900, so the structural signal
already carries most of the within-market task.
→ `src/structural_baseline.py`, `src/compute_homophily.py`

**Across markets the pattern reverses, tracking bidder-pair coverage.**
The visual advantage runs from −0.167 F1 at 15% coverage (Brazil) to
+0.122 at 94% (United States). Coverage is computable from procurement
records before any training.
→ `src/prepare_pair_sets.py`

**Dynamic attention matters.** GATv2 beats GATv1 by ΔF1 = +0.131
(p = 0.001) and converges in 10 of 10 runs against 2 of 5. It also beats
GCN and GraphSAGE on PR-AUC, so the gain comes from how neighbours are
weighted, not from message passing as such.

**Five per cent labelled data recovers most of the transferable
performance**, with the curve flat beyond 10%.

## Layout

```
config/config.yaml       paths and shared hyperparameters
run_all.sh               the whole pipeline, resumable
src/
  data_preprocessing.py       clean records, compute screens
  create_tender_splits.py     fix splits BEFORE any image exists
  image_generator.py          96x96 bid-rotation images per firm pair
  train_cnn.py                Stage 1 encoder; frozen afterwards
  extract_embeddings.py       cache the 64-dim pair embeddings
  prepare_graph_data.py       edge if two tenders share a bidder
  prepare_pair_sets.py        ragged pair sets + coverage report
  train_stage2.py             bridge + GATv2 and every ablation arm
  export_node_features.py     the 135-dim features the bridge produces
  evaluate_baseline.py        LR, random forest, gradient boosting
  structural_baseline.py      prediction from graph structure alone
  compute_homophily.py        edge homophily against a permutation null
  posthoc_analysis.py         cost curves, intervals, comparisons
  report_baselines.py         the two results tables, paths auto-detected
  analyze_attention.py        which pair the model drew on
  metrics_utils.py            PR-AUC, bootstrap, expected cost, thresholds
  make_figures.py             manuscript figures
  make_example_pair_figure.py the worked example of Figure 2
  models/                     cnn, bridge, hybrid, gatv2, gnn baselines
```

Everything under `outputs/` is generated; only the small result tables
cited in the paper are tracked.

## Install

```bash
git clone https://github.com/oagaudit/hybrid_gat_cnn_journal.git
cd hybrid_gat_cnn_journal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13, PyTorch 2.11, PyTorch Geometric 2.7. Roughly 12 hours on an
Apple M3 with 16 GB.

## Data

From the supplementary material of García Rodríguez et al. (2022), **not
redistributed here**. Download from
<https://doi.org/10.1016/j.autcon.2021.104047> and place in `data/raw/`:

```
DB_Collusion_America_processed.csv
DB_Collusion_Brazil_processed.csv
DB_Collusion_Japan_processed.csv
```

Italy is excluded (no date column) and Switzerland (no bidder identifier,
which the graph requires).

## Run

```bash
tmux new -s run
DEVICE=cpu bash run_all.sh 2>&1 | tee run.log     # Ctrl-B then D to detach
```

Or one phase at a time: `stage1`, `stage2`, `crossmarket`, `analysis`.
Each step is skipped when its output exists, so an interrupted run can be
restarted; `FORCE=1` redoes everything.

Use `tmux` rather than `nohup`: closing the terminal that owns a `nohup`
job takes its file descriptors with it, and the next Python process dies
with `Fatal Python error: init_sys_streams`.

## Reproducibility notes

**Splits precede images.** Tender-level splits are written before any
image is rendered. Each leave-one-country-out fold trains its own encoder
on source-market images only. In few-shot runs only the Stage 2
classifier is fine-tuned.

**The bridge is trained**, jointly with the graph network, so its
attention weights come from the classification loss. Tenders with no
qualifying pair get an exact zero vector, not a learned bias.

**Normalisation covers the whole vector.** `--norm_scope all`
standardises the 128 visual dimensions as well as the seven screens,
using source-market statistics only. `--norm_scope screens` reproduces
the narrower variant reported as a separate condition.

**Nulls are computed within the market being measured.**
`compute_homophily.py` prints the closed-form expectation beside the
simulated one; a gap above 0.01 signals an implementation error.

**Intervals, not point estimates.** Comparisons use a percentile
bootstrap over test cases with 3,000 resamples. Brazil's within-market
test split holds about fifteen tenders.

## Scope

The framework detects *potential* bid rigging from observable bid data.
It does not establish legal evidence of collusion and does not determine
liability. Output is intended to prioritise tenders for human review.

## Citation

```bibtex
@article{nakphon2026,
  author  = {Nakphon, Mati},
  title   = {When does visual information improve cross-market bid-rigging
             detection? Evidence from three procurement markets},
  journal = {Journal of Computational Social Science},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code under the MIT License. The procurement data belongs to its original
authors and is subject to their terms.
