# CAS747_finalproject
CAS747 finalproject Based on paper **Graph Neural Networks for Link Prediction with Subgraph Sketching** url:https://arxiv.org/abs/2209.15486

The project focuses on reproducing and evaluating three link prediction pipelines:

- **GCN baseline**
- **ELPH**
- **BUDDY**

The codebase includes:
- formal training through `train_runner.py`
- checkpoint-based evaluation through `runner.py`
- analysis notebooks for comparing accuracy and runtime

# Project Stucture
```text
Yucheng_Yao/
├── data/
│   ├── Cora/
│   ├── PubMed/
│   ├── ogbl_collab/
│   └── urls.txt
├── notebooks/
│   ├── cfg_tuning.ipynb
│   ├── data_analysis.ipynb
│   └── sanity_checks.ipynb
├── results/
│   ├── models/
│   │   └── sanity_checking/
│   ├── plots/
│   ├── tables/
│   └── raw_json/
├── src/
│   ├── data_processing/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── buddy.py
│   │   ├── elph.py
│   │   └── train.py
│   └── utils/
│       ├── buddy_helpers.py
│       ├── features.py
│       ├── helpers.py
│       ├── plot_tools.py
│       ├── sketches.py
│       ├── table_tools.py
│       └── timer.py
├── cfgs.py
├── train_runner.py
├── runner.py
├── README.md
└── requirements.txt
```

# Quick Start

## Installation
Create and activate a Python environment, then install dependencies:

`pip install -r requirements.txt`

## Data Preparation
Datasets are expected under:

`data/`

This project uses:
- Cora
- PubMed
- ogbl-collab

If a dataset is missing, PyG / OGB may download it automatically depending on the loader.

## Configurations
Formal experiment configurations are stored in: `cfgs.py`
which currently includes:
- CORA_BASELINE
- CORA_ELPH_PRIMARY
- CORA_ELPH_BACKUP
- CORA_BUDDY_PRIMARY
- CORA_BUDDY_BACKUP
- PUBMED_BASELINE
- PUBMED_ELPH_PRIMARY
- PUBMED_ELPH_BACKUP
- PUBMED_BUDDY_PRIMARY
- PUBMED_BUDDY_BACKUP
- COLLAB_BASELINE
- COLLAB_BUDDY_PRIMARY

## Evaluate a trained checkpoint with `runner.py`
`runner.py` is used to load existing `.pt` checkpoints and evaluate them again without retraining.

Run from the project root directory:
`python runner.py --mode eval --cfg-name {CFG_NAME} --cfg-module {CFG_FILE} --checkpoint results/models/{CHECKPONT_FILE} --seed {SEEDS NUMBER}`

`{CFG_NAME}` is all configs that included in `cfgs.py`; {CFG_FILE} is normally `cfgs.py`, or user can create their own one; `{CHECKPONT_FILE}` is checkponit files with `.pt` in location `results/models/...`; `{SEEDS NUMBER}` are just numbers for seeds

**Example: evaluate one checkpoint**

`python runner.py --mode eval --cfg-name CORA_BUDDY_PRIMARY --cfg-module cfgs --checkpoint results/models/CORA_BUDDY_PRIMARY_seed1.pt --seed 1`

**Example: compare multiple checkpoints**

`python runner.py --mode compare --cfg-module cfgs --seed 1 --cfg-names CORA_BASELINE CORA_ELPH_PRIMARY CORA_BUDDY_PRIMARY --checkpoints results/models/CORA_BASELINE_seed1.pt results/models/CORA_ELPH_PRIMARY_seed1.pt results/models/CORA_BUDDY_PRIMARY_seed1.pt`

## Train more models with `train_runner.py`
`train_runner.py` is used for training.

Run from the project root directory:
`python train_runner.py --cfg-name {CFG_NAME} --cfg-module {CFG_FILE} --seeds {SEEDS NUMBER}`

`{CFG_NAME}` is all configs that included in `cfgs.py`; {CFG_FILE} is normally `cfgs.py`, or user can create their own one; `{SEEDS NUMBER}` are just numbers for seeds

 **Example: train target configuration**

 `python train_runner.py --cfg-name CORA_BUDDY_PRIMARY --cfg-module cfgs --seeds 1 2 3`

## Outputs
All outputs are stored under: `results/`

### `results/models/`
Saved checkpoints as:

`{CFG_NAME}_seed{SEED}.pt`

**Example:**

`CORA_BUDDY_PRIMARY_seed1.pt`

### results/tables/
Saved final summary tables:
- final_accuracy_summary.csv
- final_runtime_summary.csv

### results/plots/
Saved plots generated during notebook-based analysis.

### results/raw_json/
Saved raw JSON records saved during training.

# Notes on Models
## GCN baseline

A simple GCN-based link prediction baseline implemented with:

GCN encoder
MLP link predictor
## ELPH

The report and outputs refer to the structural model as ELPH.

Internally, the implementation uses an edge-aware variant based on ELPHEdgeAware, but all external reporting and evaluation use the unified model label ELPH.

## BUDDY

BUDDY uses:

propagated node features
sketch-based structural features
preprocessing cache construction before training/evaluation

Because of this, BUDDY runtime includes an explicit preprocessing stage.

# Practical Implementation Note: log1p

The final implementation keeps a `log1p` transformation on structural features.

This is not presented as a redefinition of the original paper’s method. Instead, it is used as a practical stabilization step in implementation. Since sketch-based structural features may vary substantially in scale, `log1p` helps compress large values and makes the resulting inputs more numerically well-behaved.

For this reason, the final reported experiments retain the `log1p` version of the implementation.