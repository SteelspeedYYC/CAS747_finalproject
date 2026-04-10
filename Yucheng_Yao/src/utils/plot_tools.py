# plot_tools.py
# This file provides plotting helper functions.

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .helpers import ensure_dir, plots_dir


def ensure_plot_dir() -> Path:
    """
    Ensure the plot output directory exists

    Returns:
        Path to results/plots directory
    """
    return ensure_dir(plots_dir())


def load_summary_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a summary CSV file as a pandas DataFrame

    Args:
        path: CSV file path
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(path)


def filter_by_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Filter rows by dataset name

    Args:
        df: Input DataFrame
        dataset: Dataset name
    Returns:
        Filtered DataFrame
    """
    return df[df["dataset"] == dataset].copy()


def filter_by_models(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """
    Filter rows by model names

    Args:
        df: Input DataFrame
        models: Optional list of model names
    Returns:
        Filtered DataFrame
    """
    if models is None:
        return df.copy()

    return df[df["model"].isin(models)].copy()


def sort_by_model_order(df: pd.DataFrame, model_order: list[str] | None = None) -> pd.DataFrame:
    """
    Sort rows by a user-defined model order

    Args:
        df: Input DataFrame
        model_order: Optional ordered list of model names
    Returns:
        Sorted DataFrame
    """
    if model_order is None:
        return df.copy()

    df = df.copy()
    order_map = {model: idx for idx, model in enumerate(model_order)}
    df["_model_order"] = df["model"].map(order_map)
    df = df.sort_values("_model_order").drop(columns="_model_order")
    return df


def save_current_figure(save_path: str | Path, dpi: int = 200, tight: bool = True) -> Path:
    """
    Save the current matplotlib figure

    Args:
        save_path: Output image path
        dpi: Output figure dpi
        tight: Whether to use tight bounding box
    Returns:
        Path to saved figure
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if tight:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.savefig(save_path, dpi=dpi)

    return save_path


def plot_accuracy_bar(
    accuracy_df: pd.DataFrame,
    dataset: str,
    metric_col: str = "test_hits@K_mean",
    error_col: str | None = "test_hits@K_std",
    model_order: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    rotate_xticks: bool = False,
) -> None:
    """
    Plot a bar chart for model accuracy comparison on one dataset

    Args:
        accuracy_df: Accuracy summary DataFrame
        dataset: Dataset name
        metric_col: Column used for bar heights
        error_col: Optional column used for error bars
        model_order: Optional display order of models
        save_path: Optional output image path
        title: Optional plot title
        ylabel: Optional y-axis label
        rotate_xticks: Whether to rotate x tick labels
    """
    df = filter_by_dataset(accuracy_df, dataset)
    df = sort_by_model_order(df, model_order)

    x = df["model"].tolist()
    y = df[metric_col].tolist()

    yerr = None
    if error_col is not None and error_col in df.columns:
        yerr = df[error_col].tolist()

    plt.figure(figsize=(7, 5))
    plt.bar(x, y, yerr=yerr, capsize=4)

    if title is None:
        title = f"{dataset}: accuracy comparison"
    if ylabel is None:
        ylabel = metric_col

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)

    if rotate_xticks:
        plt.xticks(rotation=20)

    plt.grid(axis="y", alpha=0.3)

    if save_path is not None:
        save_current_figure(save_path)

    plt.show()


# NOT REALLY USED, table is enough
def plot_runtime_bar(
    runtime_df: pd.DataFrame,
    dataset: str,
    metric_col: str,
    error_col: str | None = None,
    model_order: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    rotate_xticks: bool = False,
) -> None:
    """
    Plot a bar chart for one runtime metric on one dataset

    Args:
        runtime_df: Runtime summary DataFrame
        dataset: Dataset name
        metric_col: Runtime metric column, such as preprocess_sec_mean
        error_col: Optional error column, such as preprocess_sec_std
        model_order: Optional display order of models
        save_path: Optional output image path
        title: Optional plot title
        ylabel: Optional y-axis label
        rotate_xticks: Whether to rotate x tick labels
    """
    df = filter_by_dataset(runtime_df, dataset)
    df = sort_by_model_order(df, model_order)

    x = df["model"].tolist()
    y = df[metric_col].tolist()

    yerr = None
    if error_col is not None and error_col in df.columns:
        yerr = df[error_col].tolist()

    plt.figure(figsize=(7, 5))
    plt.bar(x, y, yerr=yerr, capsize=4)

    if title is None:
        title = f"{dataset}: {metric_col}"
    if ylabel is None:
        ylabel = metric_col

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)

    if rotate_xticks:
        plt.xticks(rotation=20)

    plt.grid(axis="y", alpha=0.3)

    if save_path is not None:
        save_current_figure(save_path)

    plt.show()


def plot_tradeoff_scatter(
    merged_df: pd.DataFrame,
    dataset: str,
    x_col: str = "inference_sec_mean",
    y_col: str = "test_hits@K_mean",
    label_col: str = "model",
    model_order: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annotate_points: bool = True,
) -> None:
    """
    Plot an accuracy-runtime tradeoff scatter plot on one dataset

    Args:
        merged_df: DataFrame containing both accuracy and runtime summary fields
        dataset: Dataset name
        x_col: Column for x-axis
        y_col: Column for y-axis
        label_col: Column used for point labels
        model_order: Optional display order of models
        save_path: Optional output image path
        title: Optional plot title
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        annotate_points: Whether to annotate each point
    """
    df = filter_by_dataset(merged_df, dataset)
    df = sort_by_model_order(df, model_order)

    plt.figure(figsize=(7, 5))
    plt.scatter(df[x_col], df[y_col])

    if annotate_points:
        for _, row in df.iterrows():
            plt.annotate(
                row[label_col],
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
            )

    if title is None:
        title = f"{dataset}: accuracy-runtime tradeoff"
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)

    if save_path is not None:
        save_current_figure(save_path)

    plt.show()


def plot_stability_errorbar(
    accuracy_df: pd.DataFrame,
    dataset: str,
    mean_col: str = "test_hits@K_mean",
    std_col: str = "test_hits@K_std",
    model_order: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    rotate_xticks: bool = False,
) -> None:
    """
    Plot model mean performance with standard deviation error bars

    Args:
        accuracy_df: Accuracy summary DataFrame
        dataset: Dataset name
        mean_col: Mean metric column
        std_col: Standard deviation column
        model_order: Optional display order of models
        save_path: Optional output image path
        title: Optional plot title
        ylabel: Optional y-axis label
        rotate_xticks: Whether to rotate x tick labels
    """
    df = filter_by_dataset(accuracy_df, dataset)
    df = sort_by_model_order(df, model_order)

    x = df["model"].tolist()
    y = df[mean_col].tolist()
    yerr = df[std_col].tolist()

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)

    if title is None:
        title = f"{dataset}: stability comparison"
    if ylabel is None:
        ylabel = mean_col

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)

    if rotate_xticks:
        plt.xticks(rotation=20)

    plt.grid(axis="y", alpha=0.3)

    if save_path is not None:
        save_current_figure(save_path)

    plt.show()


def merge_accuracy_runtime(
    accuracy_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge accuracy and runtime summary tables

    Args:
        accuracy_df: Accuracy summary DataFrame
        runtime_df: Runtime summary DataFrame
    Returns:
        Merged DataFrame
    """
    merge_keys = ["dataset", "model", "cfg_name"]

    merged_df = pd.merge(
        accuracy_df,
        runtime_df,
        on=merge_keys,
        how="inner",
        suffixes=("_acc", "_rt"),
    )

    return merged_df