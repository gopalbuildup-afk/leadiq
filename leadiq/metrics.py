from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def safe_pr_auc(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(y_true, probability))


def safe_roc_auc(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, probability))


def top_k_indices(probability: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = len(probability)
    k = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(-np.asarray(probability), kind="mergesort")
    return order[:k]


def lift_at(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    fraction: float,
) -> float:
    y = np.asarray(y_true).astype(int)
    overall = y.mean()
    if overall == 0:
        return float("nan")
    idx = top_k_indices(probability, fraction)
    return float(y[idx].mean() / overall)


def recall_at(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    fraction: float,
) -> float:
    y = np.asarray(y_true).astype(int)
    total_positive = y.sum()
    if total_positive == 0:
        return float("nan")
    idx = top_k_indices(probability, fraction)
    return float(y[idx].sum() / total_positive)


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.asarray(probability).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(p, edges[1:-1], right=False)

    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        accuracy = y[mask].mean()
        confidence = p[mask].mean()
        ece += (mask.sum() / len(y)) * abs(accuracy - confidence)
    return float(ece)


def reliability_table(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y_true).astype(float)
    p = np.asarray(probability).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(p, edges[1:-1], right=False)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            rows.append(
                {
                    "bin": b + 1,
                    "lower": edges[b],
                    "upper": edges[b + 1],
                    "count": 0,
                    "mean_predicted": np.nan,
                    "observed_rate": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin": b + 1,
                "lower": edges[b],
                "upper": edges[b + 1],
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    include_calibration: bool = True,
) -> dict[str, float]:
    metrics = {
        "pr_auc": safe_pr_auc(y_true, probability),
        "roc_auc": safe_roc_auc(y_true, probability),
        "lift_at_10pct": lift_at(y_true, probability, 0.10),
        "lift_at_20pct": lift_at(y_true, probability, 0.20),
        "recall_at_10pct": recall_at(y_true, probability, 0.10),
        "recall_at_20pct": recall_at(y_true, probability, 0.20),
    }
    if include_calibration:
        metrics["brier_score"] = float(
            brier_score_loss(y_true, probability)
        )
        metrics["ece_10bin"] = expected_calibration_error(
            y_true,
            probability,
            n_bins=10,
        )
    return metrics


def plot_reliability_diagram(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    output_path: str | Path,
) -> None:
    table = reliability_table(
        y_true,
        probability,
        n_bins=10,
    )
    valid = table[table["count"] > 0]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(
        valid["mean_predicted"],
        valid["observed_rate"],
        marker="o",
        label="Model",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed admission rate")
    ax.set_title("10-bin Reliability Diagram")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
