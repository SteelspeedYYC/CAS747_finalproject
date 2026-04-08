# train_runner.py
# Formal training entry for final experiments

from __future__ import annotations

import argparse
import copy
import sys
import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import optim

from src.utils.helpers import set_seed, get_device, models_dir
from src.utils.table_tools import (
    record_experiment_result,
    record_runtime_result,
    export_all_current_summaries,
)
from src.data_processing.load_data import get_data_object
from src.data_processing.preprocess import prepare_link_prediction_data
from src.utils.buddy_helpers import build_buddy_cache
from src.utils.timer import (
    time_buddy_preprocessing,
    time_training_epoch,
    time_inference_full_split,
)
from src.models.baselines import GCNBaseline
from src.models.elph import ELPHEdgeAware
from src.models.buddy import BUDDY
from src.models.train import (
    fit,
    fit_buddy,
    train_one_epoch,
    train_one_epoch_buddy,
)
from src.evaluation.evaluate import evaluate_split, evaluate_split_buddy


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Formal training runner for CAS747 final project")

    parser.add_argument(
        "--cfg-name",
        type=str,
        required=True,
        help="Configuration name, such as CORA_ELPH_PRIMARY",
    )
    parser.add_argument(
        "--cfg-module",
        type=str,
        default="cfg",
        help="Python module containing configuration dictionaries",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Random seeds to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose training progress",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip exporting final summary CSV files",
    )

    return parser.parse_args()


def load_cfg_object(cfg_module_name: str, cfg_name: str) -> dict[str, Any]:
    """
    Load one configuration dictionary from a cfg module

    Args:
        cfg_module_name: Module name containing configs
        cfg_name: Configuration dictionary name
    Returns:
        Configuration dictionary
    """
    cfg_module = importlib.import_module(cfg_module_name)

    if not hasattr(cfg_module, cfg_name):
        raise ValueError(f"Config '{cfg_name}' not found in module '{cfg_module_name}'.")

    cfg = getattr(cfg_module, cfg_name)

    if not isinstance(cfg, dict):
        raise TypeError(f"Config '{cfg_name}' must be a dict.")

    return copy.deepcopy(cfg)


def resolve_device(device_arg: str) -> torch.device:
    """
    Resolve target device from command-line argument

    Args:
        device_arg: Requested device string
    Returns:
        torch.device object
    """
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    return get_device()


# Just in case
def normalize_model_label(model_name: str) -> str:
    """
    Normalize internal model names to report-facing names

    Args:
        model_name: Raw internal model name
    Returns:
        Normalized model label
    """
    if model_name in {"ELPH", "ELPHEdgeAware", "ELPHEdge", "ELPHEdgeAware_log"}:
        return "ELPH"

    if model_name in {"GCN_baseline", "GCNBaseline"}:
        return "GCN_baseline"

    if model_name in {"BUDDY", "BUDDY_log"}:
        return "BUDDY"

    return model_name


# Tag added to cfgs!!!
# ALSO NEED TO GO FIX cfg_tuning
def prepare_data(cfg: dict[str, Any]) -> tuple[Any, Any, Any]:
    """
    Load dataset and build train/val/test splits

    Args:
        cfg: Configuration dictionary
    Returns:
        train_data, val_data, test_data
    """
    dataset_name = cfg["dataset"]
    project_root = Path(__file__).resolve().parent
    data = get_data_object(dataset_name, root=project_root / "data")
    train_data, val_data, test_data = prepare_link_prediction_data(dataset_name, data)
    return train_data, val_data, test_data


def build_model_from_cfg(cfg: dict[str, Any], train_data: Any, device: torch.device) -> tuple[nn.Module, str]:
    """
    Build a model from configuration

    Args:
        cfg: Configuration dictionary
        train_data: Training split data
        device: Target device
    Returns:
        model, normalized display model name
    """
    model_name = cfg["model_name"]
    display_model_name = normalize_model_label(model_name)

    if display_model_name == "GCN_baseline":
        model = GCNBaseline(
            in_channels=train_data.x.size(1),
            hidden_channels=cfg["hidden_channels"],
            emb_channels=cfg.get("emb_channels", cfg["hidden_channels"]),
            predictor_hidden_channels=cfg.get("predictor_hidden_channels", cfg["hidden_channels"]),
            dropout=cfg.get("dropout", 0.0),
        ).to(device)
        return model, display_model_name

    if display_model_name == "ELPH":
        model = ELPHEdgeAware(
            in_channels=train_data.x.size(1),
            hidden_channels=cfg["hidden_channels"],
            predictor_hidden_channels=cfg.get("predictor_hidden_channels", cfg["hidden_channels"]),
            num_hops=cfg.get("num_hops", 2),
            minhash_num_perm=cfg.get("minhash_num_perm", 128),
            hll_p=cfg.get("hll_p", 8),
            message_hidden_channels=cfg.get("message_hidden_channels", cfg["hidden_channels"]),
            update_hidden_channels=cfg.get("update_hidden_channels", cfg["hidden_channels"]),
            dropout=cfg.get("dropout", 0.0),
            use_log_features=cfg.get("use_log_features", True),
        ).to(device)
        return model, display_model_name

    if display_model_name == "BUDDY":
        if "predictor_hidden_channels" not in cfg:
            raise ValueError("BUDDY config must contain 'predictor_hidden_channels'.")

        model = BUDDY(
            node_feature_dim=train_data.x.size(1),
            num_hops=cfg.get("num_hops", 2),
            predictor_hidden_channels=cfg["predictor_hidden_channels"],
            dropout=cfg.get("dropout", 0.0),
            structural_use_log=cfg.get("structural_use_log", True),
        ).to(device)
        return model, display_model_name

    raise ValueError(f"Unsupported model_name: {model_name}")


def build_optimizer_from_cfg(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer from configuration

    Args:
        model: PyTorch model
        cfg: Configuration dictionary
    Returns:
        Optimizer
    """
    return optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
    )


def build_buddy_cache_if_needed(
    cfg: dict[str, Any],
    train_data: Any,
    device: torch.device,
    display_model_name: str,
) -> tuple[dict[str, Any] | None, float]:
    """
    Build and time BUDDY cache if the current model is BUDDY

    Args:
        cfg: Configuration dictionary
        train_data: Training split data
        device: Target device
        display_model_name: Report-facing model name
    Returns:
        buddy_cache_or_none, preprocessing_time_seconds
    """
    if display_model_name != "BUDDY":
        return None, 0.0

    buddy_cache, preprocess_sec = time_buddy_preprocessing(
        build_buddy_cache_fn=build_buddy_cache,
        x=train_data.x.to(device),
        edge_index=train_data.edge_index.to(device),
        num_hops=cfg.get("num_hops", 2),
        minhash_num_perm=cfg.get("minhash_num_perm", 128),
        hll_p=cfg.get("hll_p", 8),
        feature_propagation=cfg.get("feature_propagation", "mean"),
        cache_device=device,
        timer_device=device,
    )
    return buddy_cache, preprocess_sec


def evaluate_all_splits(
    model: nn.Module,
    display_model_name: str,
    train_data: Any,
    val_data: Any,
    test_data: Any,
    device: torch.device,
    monitor_hits_k: int,
    buddy_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Evaluate train, validation, and test splits

    Args:
        model: Trained model
        display_model_name: Report-facing model name
        train_data: Training split
        val_data: Validation split
        test_data: Test split
        device: Evaluation device
        monitor_hits_k: Hits@K value
        buddy_cache: Optional BUDDY cache
    Returns:
        train_metrics, val_metrics, test_metrics
    """
    hits_ks = [monitor_hits_k]

    if display_model_name == "BUDDY":
        if buddy_cache is None:
            raise ValueError("buddy_cache must not be None when timing BUDDY.")
        train_metrics = evaluate_split_buddy(
            model=model,
            data=train_data,
            buddy_cache=buddy_cache,
            device=device,
            hits_ks=hits_ks,
        )
        val_metrics = evaluate_split_buddy(
            model=model,
            data=val_data,
            buddy_cache=buddy_cache,
            device=device,
            hits_ks=hits_ks,
        )
        test_metrics = evaluate_split_buddy(
            model=model,
            data=test_data,
            buddy_cache=buddy_cache,
            device=device,
            hits_ks=hits_ks,
        )
        return train_metrics, val_metrics, test_metrics

    train_metrics = evaluate_split(
        model=model,
        data=train_data,
        device=device,
        hits_ks=hits_ks,
    )
    val_metrics = evaluate_split(
        model=model,
        data=val_data,
        device=device,
        hits_ks=hits_ks,
    )
    test_metrics = evaluate_split(
        model=model,
        data=test_data,
        device=device,
        hits_ks=hits_ks,
    )
    return train_metrics, val_metrics, test_metrics


def measure_runtime(
    model: nn.Module,
    display_model_name: str,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    test_data: Any,
    device: torch.device,
    monitor_hits_k: int,
    buddy_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Measure runtime for one epoch of training and full-split inference

    Args:
        model: Trained model
        display_model_name: Report-facing model name
        optimizer: Optimizer
        train_data: Training split
        test_data: Test split
        device: Target device
        monitor_hits_k: Hits@K value
        buddy_cache: Optional BUDDY cache
    Returns:
        Runtime dictionary
    """
    criterion = nn.BCEWithLogitsLoss()
    hits_ks = [monitor_hits_k]

    if display_model_name == "BUDDY":
        _, train_sec = time_training_epoch(
            train_one_epoch_fn=train_one_epoch_buddy,
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            sync_device=device,
            criterion=criterion,
            buddy_cache=buddy_cache,
        )

        _, inference_sec = time_inference_full_split(
            evaluate_split_fn=evaluate_split_buddy,
            model=model,
            data=test_data,
            sync_device=device,
            criterion=criterion,
            hits_ks=hits_ks,
            buddy_cache=buddy_cache,
        )

        return {
            "train_sec": train_sec,
            "inference_sec": inference_sec,
        }

    _, train_sec = time_training_epoch(
        train_one_epoch_fn=train_one_epoch,
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        sync_device=device,
        criterion=criterion,
    )

    _, inference_sec = time_inference_full_split(
        evaluate_split_fn=evaluate_split,
        model=model,
        data=test_data,
        sync_device=device,
        criterion=criterion,
        hits_ks=hits_ks,
    )

    return {
        "train_sec": train_sec,
        "inference_sec": inference_sec,
    }


def run_one_seed(
    cfg_name: str,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    verbose: bool,
) -> None:
    """
    Run one configuration under one random seed

    Args:
        cfg_name: Configuration name
        cfg: Configuration dictionary
        seed: Random seed
        device: Target device
        verbose: Whether to show training progress
    """
    set_seed(seed)

    dataset_name = cfg["dataset"]
    train_data, val_data, test_data = prepare_data(cfg)

    model, display_model_name = build_model_from_cfg(cfg, train_data, device)
    optimizer = build_optimizer_from_cfg(model, cfg)

    checkpoint_path = models_dir() / f"{cfg_name}_seed{seed}.pt"

    buddy_cache, preprocess_sec = build_buddy_cache_if_needed(
        cfg=cfg,
        train_data=train_data,
        device=device,
        display_model_name=display_model_name,
    )

    criterion = nn.BCEWithLogitsLoss()
    monitor = cfg.get("monitor", "val_loss")
    monitor_hits_k = cfg.get("monitor_hits_k", 100)

    if display_model_name == "BUDDY":
        if buddy_cache is None:
            raise ValueError("buddy_cache must not be None when timing BUDDY.")
        history = fit_buddy(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            val_data=val_data,
            buddy_cache=buddy_cache,
            device=device,
            epochs=cfg["epochs"],
            criterion=criterion,
            verbose=verbose,
            patience=cfg.get("patience"),
            checkpoint_path=checkpoint_path,
            restore_best_model=True,
            monitor=monitor,
            monitor_hits_k=monitor_hits_k,
        )
    else:
        history = fit(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            val_data=val_data,
            device=device,
            epochs=cfg["epochs"],
            criterion=criterion,
            verbose=verbose,
            patience=cfg.get("patience"),
            checkpoint_path=checkpoint_path,
            restore_best_model=True,
            monitor=monitor,
            monitor_hits_k=monitor_hits_k,
        )

    # 公式化coding
    train_metrics, val_metrics, test_metrics = evaluate_all_splits(
        model=model,
        display_model_name=display_model_name,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        device=device,
        monitor_hits_k=monitor_hits_k,
        buddy_cache=buddy_cache,
    )

    metric_name = f"hits@{monitor_hits_k}" if monitor == "val_hits@K" else monitor.replace("val_", "")

    record_experiment_result({
        "dataset": dataset_name,
        "model": display_model_name,
        "cfg_name": cfg_name,
        "seed": seed,
        "metric_name": metric_name,
        "monitor": monitor,
        "monitor_hits_k": monitor_hits_k,
        "best_epoch": history.get("best_epoch"),
        "epochs_trained": history.get("epochs_ran"),
        "train_loss": history["train_loss"][-1] if len(history.get("train_loss", [])) > 0 else None,
        "val_loss": val_metrics.get("loss"),
        "test_loss": test_metrics.get("loss"),
        "val_auc": val_metrics.get("auc"),
        "test_auc": test_metrics.get("auc"),
        "val_ap": val_metrics.get("ap"),
        "test_ap": test_metrics.get("ap"),
        "val_hits@K": val_metrics.get(f"hits@{monitor_hits_k}"),
        "test_hits@K": test_metrics.get(f"hits@{monitor_hits_k}"),
    })

    runtime_metrics = measure_runtime(
        model=model,
        display_model_name=display_model_name,
        optimizer=optimizer,
        train_data=train_data,
        test_data=test_data,
        device=device,
        monitor_hits_k=monitor_hits_k,
        buddy_cache=buddy_cache,
    )

    record_runtime_result({
        "dataset": dataset_name,
        "model": display_model_name,
        "cfg_name": cfg_name,
        "seed": seed,
        "preprocess_sec": preprocess_sec,
        "train_sec": runtime_metrics["train_sec"],
        "inference_sec": runtime_metrics["inference_sec"],
    })

    print(
        f"[Done] cfg={cfg_name} seed={seed} "
        f"test_auc={test_metrics.get('auc', None)} "
        f"test_ap={test_metrics.get('ap', None)} "
        f"test_hits@{monitor_hits_k}={test_metrics.get(f'hits@{monitor_hits_k}', None)}"
    )


def main() -> None:
    """
    Main entry point
    """
    args = parse_args()
    cfg = load_cfg_object(args.cfg_module, args.cfg_name)
    device = resolve_device(args.device)
    verbose = not args.quiet

    print(f"Running cfg={args.cfg_name}")
    print(f"Dataset={cfg['dataset']}, model={normalize_model_label(cfg['model_name'])}, device={device}")
    print(f"Seeds={args.seeds}")

    for seed in args.seeds:
        run_one_seed(
            cfg_name=args.cfg_name,
            cfg=cfg,
            seed=seed,
            device=device,
            verbose=verbose,
        )

    if not args.skip_summary:
        acc_path, rt_path = export_all_current_summaries()
        print(f"Saved accuracy summary to: {acc_path}")
        print(f"Saved runtime summary to: {rt_path}")


if __name__ == "__main__":
    main()