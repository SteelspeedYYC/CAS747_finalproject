# runner.py
# Load trained checkpoints for evaluation, comparison, and spot checks

from __future__ import annotations

import argparse
import copy
import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import optim

from src.utils.helpers import get_device, load_checkpoint, set_seed
from src.data_processing.load_data import get_data_object
from src.data_processing.preprocess import prepare_link_prediction_data
from src.utils.buddy_helpers import build_buddy_cache
from src.models.baselines import GCNBaseline
from src.models.elph import ELPHEdgeAware
from src.models.buddy import BUDDY
from src.evaluation.evaluate import evaluate_split, evaluate_split_buddy


project_root = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Checkpoint evaluation runner for CAS747 final project")

    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["eval", "compare"],
        help="Runner mode",
    )
    parser.add_argument(
        "--cfg-name",
        type=str,
        help="Configuration name for eval mode",
    )
    parser.add_argument(
        "--cfg-names",
        type=str,
        nargs="+",
        help="Configuration names for compare mode",
    )
    parser.add_argument(
        "--cfg-module",
        type=str,
        default="cfgs",
        help="Python module containing configuration dictionaries",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path for eval mode",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="Checkpoint paths for compare mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed used for data split reproduction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device",
    )

    return parser.parse_args()


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


def prepare_data(cfg: dict[str, Any]) -> tuple[Any, Any, Any]:
    """
    Load dataset and build train/val/test splits

    Args:
        cfg: Configuration dictionary
    Returns:
        train_data, val_data, test_data
    """
    dataset_name = cfg["dataset"]
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


def build_buddy_cache_if_needed(
    cfg: dict[str, Any],
    train_data: Any,
    device: torch.device,
    display_model_name: str,
) -> dict[str, Any] | None:
    """
    Build BUDDY cache if the current model is BUDDY

    Args:
        cfg: Configuration dictionary
        train_data: Training split data
        device: Target device
        display_model_name: Report-facing model name
    Returns:
        BUDDY cache or None
    """
    if display_model_name != "BUDDY":
        return None

    buddy_cache = build_buddy_cache(
        x=train_data.x.to(device),
        edge_index=train_data.edge_index.to(device),
        num_hops=cfg.get("num_hops", 2),
        minhash_num_perm=cfg.get("minhash_num_perm", 128),
        hll_p=cfg.get("hll_p", 8),
        feature_propagation=cfg.get("feature_propagation", "mean"),
        cache_device=device,
    )
    return buddy_cache


def evaluate_checkpoint(
    cfg_name: str,
    checkpoint_path: str | Path,
    cfg_module: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load one checkpoint and evaluate it on validation and test splits

    Args:
        cfg_name: Configuration name
        checkpoint_path: Model checkpoint path
        cfg_module: Config module name
        seed: Seed used to reproduce data split
        device: Evaluation device
    Returns:
        Evaluation result dictionary
    """
    set_seed(seed)

    cfg = load_cfg_object(cfg_module, cfg_name)
    train_data, val_data, test_data = prepare_data(cfg)

    model, display_model_name = build_model_from_cfg(cfg, train_data, device)
    load_checkpoint(model=model, path=checkpoint_path, optimizer=None, map_location=device)

    monitor_hits_k = cfg.get("monitor_hits_k", 100)
    hits_ks = [monitor_hits_k]

    buddy_cache = build_buddy_cache_if_needed(
        cfg=cfg,
        train_data=train_data,
        device=device,
        display_model_name=display_model_name,
    )

    if display_model_name == "BUDDY":
        if buddy_cache is None:
            raise ValueError("buddy_cache must not be None when evaluating BUDDY.")

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
    else:
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

    return {
        "cfg_name": cfg_name,
        "dataset": cfg["dataset"],
        "model": display_model_name,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "monitor_hits_k": monitor_hits_k,
        "val_auc": val_metrics.get("auc"),
        "val_ap": val_metrics.get("ap"),
        "val_hits@K": val_metrics.get(f"hits@{monitor_hits_k}"),
        "test_auc": test_metrics.get("auc"),
        "test_ap": test_metrics.get("ap"),
        "test_hits@K": test_metrics.get(f"hits@{monitor_hits_k}"),
    }


def print_eval_result(result: dict[str, Any]) -> None:
    """
    Print one evaluation result in a readable format

    Args:
        result: Evaluation result dictionary
    """
    k = result["monitor_hits_k"]

    print("-" * 60)
    print(f"cfg_name   : {result['cfg_name']}")
    print(f"dataset    : {result['dataset']}")
    print(f"model      : {result['model']}")
    print(f"checkpoint : {result['checkpoint']}")
    print(f"seed       : {result['seed']}")
    print()
    print(f"val_auc    : {result['val_auc']:.6f}")
    print(f"val_ap     : {result['val_ap']:.6f}")
    print(f"val_hits@{k}: {result['val_hits@K']:.6f}")
    print(f"test_auc   : {result['test_auc']:.6f}")
    print(f"test_ap    : {result['test_ap']:.6f}")
    print(f"test_hits@{k}: {result['test_hits@K']:.6f}")
    print("-" * 60)


def print_compare_results(results: list[dict[str, Any]]) -> None:
    """
    Print multiple evaluation results as a simple comparison table

    Args:
        results: List of evaluation result dictionaries
    """
    if len(results) == 0:
        return

    k = results[0]["monitor_hits_k"]

    print("=" * 100)
    print(
        f"{'cfg_name':<24}"
        f"{'model':<14}"
        f"{'val_auc':<12}"
        f"{'val_ap':<12}"
        f"{f'val_hits@{k}':<14}"
        f"{'test_auc':<12}"
        f"{'test_ap':<12}"
        f"{f'test_hits@{k}':<14}"
    )
    print("=" * 100)

    for result in results:
        print(
            f"{result['cfg_name']:<24}"
            f"{result['model']:<14}"
            f"{result['val_auc']:<12.6f}"
            f"{result['val_ap']:<12.6f}"
            f"{result['val_hits@K']:<14.6f}"
            f"{result['test_auc']:<12.6f}"
            f"{result['test_ap']:<12.6f}"
            f"{result['test_hits@K']:<14.6f}"
        )

    print("=" * 100)


def main() -> None:
    """
    Main entry point
    """
    args = parse_args()
    device = resolve_device(args.device)

    if args.mode == "eval":
        if args.cfg_name is None or args.checkpoint is None:
            raise ValueError("eval mode requires --cfg-name and --checkpoint")

        result = evaluate_checkpoint(
            cfg_name=args.cfg_name,
            checkpoint_path=args.checkpoint,
            cfg_module=args.cfg_module,
            seed=args.seed,
            device=device,
        )
        print_eval_result(result)
        return

    if args.mode == "compare":
        if args.cfg_names is None or args.checkpoints is None:
            raise ValueError("compare mode requires --cfg-names and --checkpoints")

        if len(args.cfg_names) != len(args.checkpoints):
            raise ValueError("--cfg-names and --checkpoints must have the same length")

        results = []
        for cfg_name, checkpoint in zip(args.cfg_names, args.checkpoints):
            result = evaluate_checkpoint(
                cfg_name=cfg_name,
                checkpoint_path=checkpoint,
                cfg_module=args.cfg_module,
                seed=args.seed,
                device=device,
            )
            results.append(result)

        print_compare_results(results)
        return


if __name__ == "__main__":
    main()