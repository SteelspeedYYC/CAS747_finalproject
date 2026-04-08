# cfgs.py
# Centralized experiment configurations for CAS747 final project.

from __future__ import annotations

# ------------------------------------------------------------
# Cora final candidates
# ------------------------------------------------------------

CORA_ELPH_PRIMARY = {
    "dataset": "Cora",
    "model_name": "ELPHEdgeAware_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "hidden_channels": 32,
    "predictor_hidden_channels": 128,
    "message_hidden_channels": 64,
    "update_hidden_channels": 64,
    "dropout": 0.20,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
    "use_log_features": True,
}

CORA_ELPH_BACKUP = {
    "dataset": "Cora",
    "model_name": "ELPHEdgeAware_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "hidden_channels": 64,
    "predictor_hidden_channels": 64,
    "message_hidden_channels": 64,
    "update_hidden_channels": 64,
    "dropout": 0.20,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
    "use_log_features": True,
}

CORA_BUDDY_PRIMARY = {
    "dataset": "Cora",
    "model_name": "BUDDY_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "feature_propagation": "mean",
    "predictor_hidden_channels": 32,
    "dropout": 0.00,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
    "structural_use_log": True,
}

CORA_BUDDY_BACKUP = {
    "dataset": "Cora",
    "model_name": "BUDDY_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "feature_propagation": "mean",
    "predictor_hidden_channels": 32,
    "dropout": 0.10,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
    "structural_use_log": True,
}

# ------------------------------------------------------------
# Pubmed final candidates
# ------------------------------------------------------------

PUBMED_ELPH_PRIMARY = {
    "dataset": "Pubmed",
    "model_name": "ELPHEdgeAware_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "hidden_channels": 64,
    "predictor_hidden_channels": 128,
    "message_hidden_channels": 64,
    "update_hidden_channels": 64,
    "dropout": 0.20,
    "lr": 0.005,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
    "use_log_features": True,
}

PUBMED_ELPH_BACKUP = {
    "dataset": "Pubmed",
    "model_name": "ELPHEdgeAware_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "hidden_channels": 32,
    "predictor_hidden_channels": 64,
    "message_hidden_channels": 64,
    "update_hidden_channels": 64,
    "dropout": 0.20,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
    "use_log_features": True,
}

PUBMED_BUDDY_PRIMARY = {
    "dataset": "Pubmed",
    "model_name": "BUDDY_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "feature_propagation": "mean",
    "predictor_hidden_channels": 32,
    "dropout": 0.10,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
    "structural_use_log": True,
}

PUBMED_BUDDY_BACKUP = {
    "dataset": "Pubmed",
    "model_name": "BUDDY_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "feature_propagation": "mean",
    "predictor_hidden_channels": 32,
    "dropout": 0.00,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
    "structural_use_log": True,
}

# ------------------------------------------------------------
# Collab final candidates
# ------------------------------------------------------------

COLLAB_BUDDY_PRIMARY = {
    "dataset": "Collab",
    "model_name": "BUDDY_log",
    "num_hops": 2,
    "minhash_num_perm": 128,
    "hll_p": 8,
    "feature_propagation": "mean",
    "predictor_hidden_channels": 64,
    "dropout": 0.10,
    "lr": 0.005,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 50,
    "structural_use_log": True,
}

# ------------------------------------------------------------
# Baseline GCN final candidates
# ------------------------------------------------------------

CORA_BASELINE = {
    "dataset": "Cora",
    "model_name": "GCN_baseline",
    "hidden_channels": 64,
    "dropout": 0.20,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
}

PUBMED_BASELINE = {
    "dataset": "Pubmed",
    "model_name": "GCN_baseline",
    "hidden_channels": 64,
    "dropout": 0.20,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 100,
}

COLLAB_BASELINE = {
    "dataset": "Collab",
    "model_name": "GCN_baseline",
    "hidden_channels": 64,
    "dropout": 0.20,
    "lr": 0.010,
    "weight_decay": 1e-4,
    "epochs": 25,
    "patience": 8,
    "monitor": "val_hits@K",
    "monitor_hits_k": 50,
}