# table_helpers.py
# Utilities for saving experiment results and building summary tables

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any
import csv
import json
import math

from .helpers import ensure_parent_dir, results_dir, tables_dir


# FIXED!
def sanitize_name(text: str) -> str:
    """
    Convert a string into a filesystem-friendly token

    Args:
        text: Input string
    Returns:
        Sanitized string safe for filenames
    """
    return (
        str(text)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("@", "at")
        .replace(":", "_")
    )


def build_result_stem(cfg_name: str, seed: int | None = None, suffix: str | None = None) -> str:
    """
    Build a consistent filename stem from cfg_name, seed, and optional suffix

    Args:
        cfg_name: Configuration name
        seed: Random seed
        suffix: Optional suffix such as 'runtime'
    Returns:
        Filename stem string
    """
    parts = [sanitize_name(cfg_name)]

    if seed is not None:
        parts.append(f"seed{seed}")

    if suffix is not None:
        parts.append(sanitize_name(suffix))

    return "_".join(parts)


def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Safely convert a value to float

    Args:
        value: Input value
        default: Value to return if conversion fails
    Returns:
        Float value or default
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return default

    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value

    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def safe_mean(values: list[Any]) -> float | None:
    """
    Compute the mean of valid numeric values

    Args:
        values: Input list
    Returns:
        Mean value or None if no valid numbers exist
    """
    clean_values = [safe_float(v) for v in values]
    clean_values = [v for v in clean_values if v is not None]

    if len(clean_values) == 0:
        return None

    return mean(clean_values)


def safe_std(values: list[Any]) -> float | None:
    """
    Compute the sample standard deviation of valid numeric values

    Args:
        values: Input list
    Returns:
        Standard deviation, 0.0 for one value, or None if empty
    """
    clean_values = [safe_float(v) for v in values]
    clean_values = [v for v in clean_values if v is not None]

    if len(clean_values) == 0:
        return None

    if len(clean_values) == 1:
        return 0.0

    return stdev(clean_values)


def format_mean_std(mean_value: float | None, std_value: float | None, decimals: int = 2) -> str:
    """
    Format mean and std as a display string

    Args:
        mean_value: Mean value
        std_value: Standard deviation value
        decimals: Number of decimal places
    Returns:
        Formatted string such as '88.00±0.44'
    """
    if mean_value is None:
        return ""

    if std_value is None:
        return f"{mean_value:.{decimals}f}"

    return f"{mean_value:.{decimals}f}±{std_value:.{decimals}f}"


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """
    Save a dictionary to a JSON file

    Args:
        data: Dictionary to save
        path: Target JSON path
        indent: JSON indentation size
    Returns:
        Path to saved JSON file
    """
    path = Path(path)
    ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

    return path


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON file into a dictionary

    Args:
        path: JSON file path
    Returns:
        Loaded dictionary
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_rows_to_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> Path:
    """
    Save a list of dictionaries to a CSV file

    Args:
        rows: List of row dictionaries
        path: Target CSV path
        fieldnames: Optional explicit field order
    Returns:
        Path to saved CSV file
    """
    path = Path(path)
    ensure_parent_dir(path)

    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return path

    if fieldnames is None:
        fieldnames = []
        seen = set()

        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return path


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a raw result row into a consistent schema

    Args:
        row: Raw result dictionary
    Returns:
        Normalized result dictionary
    """
    normalized = dict(row)

    numeric_keys = [
        "seed",
        "monitor_hits_k",
        "best_epoch",
        "epochs_trained",
        "train_loss",
        "val_loss",
        "test_loss",
        "val_auc",
        "test_auc",
        "val_ap",
        "test_ap",
        "val_hits@K",
        "test_hits@K",
        "preprocess_sec",
        "train_sec",
        "inference_sec",
    ]

    for key in numeric_keys:
        if key in normalized:
            normalized[key] = safe_float(normalized[key])

    for int_key in ["seed", "monitor_hits_k", "best_epoch", "epochs_trained"]:
        if normalized.get(int_key) is not None:
            normalized[int_key] = int(normalized[int_key])

    return normalized


def result_json_path(cfg_name: str, seed: int) -> Path:
    """
    Build the JSON path for one experiment result

    Args:
        cfg_name: Configuration name
        seed: Random seed
    Returns:
        Path to result JSON file
    """
    stem = build_result_stem(cfg_name=cfg_name, seed=seed)
    return results_dir() / f"{stem}.json"


def runtime_json_path(cfg_name: str, seed: int) -> Path:
    """
    Build the JSON path for one runtime result

    Args:
        cfg_name: Configuration name
        seed: Random seed
    Returns:
        Path to runtime JSON file
    """
    stem = build_result_stem(cfg_name=cfg_name, seed=seed, suffix="runtime")
    return results_dir() / f"{stem}.json"


def record_experiment_result(row: dict[str, Any], save_csv: bool = False) -> Path:
    """
    Save one experiment result row as JSON and optionally CSV

    Args:
        row: Result dictionary
        save_csv: Whether to also save a one-row CSV copy
    Returns:
        Path to saved JSON file
    """
    row = normalize_result_row(row)

    cfg_name = row.get("cfg_name")
    seed = row.get("seed")

    if cfg_name is None:
        raise ValueError("record_experiment_result requires 'cfg_name'.")
    if seed is None:
        raise ValueError("record_experiment_result requires 'seed'.")

    json_path = result_json_path(cfg_name=cfg_name, seed=seed)
    save_json(row, json_path)

    if save_csv:
        csv_path = tables_dir() / f"{build_result_stem(cfg_name, seed)}.csv"
        save_rows_to_csv([row], csv_path)

    return json_path


def record_runtime_result(row: dict[str, Any], save_csv: bool = False) -> Path:
    """
    Save one runtime result row as JSON and optionally CSV

    Args
        row: Runtime result dictionary
        save_csv: Whether to also save a one-row CSV copy
    Returns:
        Path to saved JSON file
    """
    row = normalize_result_row(row)

    cfg_name = row.get("cfg_name")
    seed = row.get("seed")

    if cfg_name is None:
        raise ValueError("record_runtime_result requires 'cfg_name'.")
    if seed is None:
        raise ValueError("record_runtime_result requires 'seed'.")

    json_path = runtime_json_path(cfg_name=cfg_name, seed=seed)
    save_json(row, json_path)

    if save_csv:
        csv_path = tables_dir() / f"{build_result_stem(cfg_name, seed, 'runtime')}.csv"
        save_rows_to_csv([row], csv_path)

    return json_path


def aggregate_seed_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    value_keys: list[str],
    decimals: int = 2,
) -> list[dict[str, Any]]:
    """
    Aggregate multiple seed rows by group keys

    Args:
        rows: Input result rows
        group_keys: Keys used to group rows
        value_keys: Numeric keys to aggregate
        decimals: Decimal places for display formatting
    Returns:
        List of aggregated summary rows
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        row = normalize_result_row(row)
        group_key = tuple(row.get(key) for key in group_keys)
        grouped[group_key].append(row)

    summary_rows: list[dict[str, Any]] = []

    for group_key, group_rows in grouped.items():
        summary_row: dict[str, Any] = {}

        for idx, key in enumerate(group_keys):
            summary_row[key] = group_key[idx]

        seeds: list[int] = [
            int(row["seed"])
            for row in group_rows
            if row.get("seed") is not None
        ]
        summary_row["n_seeds"] = len(seeds)
        summary_row["seeds"] = ",".join(str(seed) for seed in sorted(seeds))

        for value_key in value_keys:
            values = [row.get(value_key) for row in group_rows]
            value_mean = safe_mean(values)
            value_std = safe_std(values)

            summary_row[f"{value_key}_mean"] = value_mean
            summary_row[f"{value_key}_std"] = value_std
            summary_row[f"{value_key}_display"] = format_mean_std(value_mean, value_std, decimals=decimals)

        summary_rows.append(summary_row)

    return summary_rows


def build_accuracy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build the multi-seed accuracy summary table

    Args:
        rows: Per-seed experiment result rows
    Returns:
        Aggregated accuracy summary rows
    """
    return aggregate_seed_rows(
        rows=rows,
        group_keys=["dataset", "model", "cfg_name", "metric_name", "monitor", "monitor_hits_k"],
        value_keys=[
            "val_loss",
            "test_loss",
            "val_auc",
            "test_auc",
            "val_ap",
            "test_ap",
            "val_hits@K",
            "test_hits@K",
        ],
        decimals=2,
    )


def build_runtime_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build the multi-seed runtime summary table

    Args:
        rows: Per-seed runtime rows
    Returns:
        Aggregated runtime summary rows
    """
    return aggregate_seed_rows(
        rows=rows,
        group_keys=["dataset", "model", "cfg_name"],
        value_keys=[
            "preprocess_sec",
            "train_sec",
            "inference_sec",
        ],
        decimals=3,
    )


def export_accuracy_summary(rows: list[dict[str, Any]], filename: str = "final_accuracy_summary.csv") -> Path:
    """
    Export the aggregated accuracy summary CSV

    Args:
        rows: Per-seed experiment result rows
        filename: Output CSV filename
    Returns:
        Path to saved CSV file
    """
    summary_rows = build_accuracy_summary(rows)
    return save_rows_to_csv(summary_rows, tables_dir() / filename)


def export_runtime_summary(rows: list[dict[str, Any]], filename: str = "final_runtime_summary.csv") -> Path:
    """
    Export the aggregated runtime summary CSV

    Args:
        rows: Per-seed runtime rows
        filename: Output CSV filename
    Returns:
        Path to saved CSV file
    """
    summary_rows = build_runtime_summary(rows)
    return save_rows_to_csv(summary_rows, tables_dir() / filename)


def load_result_rows(include_runtime: bool = False) -> list[dict[str, Any]]:
    """
    Load saved JSON result rows from results directory

    Args:
        include_runtime: Whether to include runtime JSON files
    Returns:
        List of loaded result dictionaries
    """
    json_paths = sorted(results_dir().glob("*.json"))

    if not include_runtime:
        json_paths = [path for path in json_paths if not path.stem.endswith("_runtime")]

    return [normalize_result_row(load_json(path)) for path in json_paths]


def load_runtime_rows() -> list[dict[str, Any]]:
    """
    Load saved runtime JSON rows from results directory

    Returns:
        List of loaded runtime dictionaries
    """
    json_paths = sorted(results_dir().glob("*_runtime.json"))
    return [normalize_result_row(load_json(path)) for path in json_paths]


def export_all_current_summaries() -> tuple[Path, Path]:
    """
    Export both accuracy and runtime summary CSV files from current JSON results

    Returns:
        Tuple of (accuracy_csv_path, runtime_csv_path)
    """
    accuracy_rows = load_result_rows(include_runtime=False)
    runtime_rows = load_runtime_rows()

    accuracy_path = export_accuracy_summary(accuracy_rows)
    runtime_path = export_runtime_summary(runtime_rows)

    return accuracy_path, runtime_path