"""
Evaluation metrics for both NLP (classification) and IS2 (regression) tasks.
Saves results to model_outputs/ as JSON.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    save_path: str | None = "model_outputs/nlp_metrics.json",
) -> dict:
    """
    Compute binary classification metrics.

    Args:
        y_true:    Ground-truth labels (0/1).
        y_pred:    Predicted labels (0/1).
        y_proba:   Predicted probabilities, shape (N,) or (N, 2). Used for AUC.
        save_path: If provided, save results as JSON to this path.

    Returns:
        dict of metrics.
    """
    y_true = np.ravel(np.asarray(y_true))
    y_pred = np.ravel(np.asarray(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])

    metrics = {
        "task": "classification",
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    if y_proba is not None:
        y_score = np.asarray(y_proba)
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass

    _save(metrics, save_path)
    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = "model_outputs/is2_metrics.json",
) -> dict:
    """
    Compute regression metrics (MAE, MSE, RMSE, R²).

    Args:
        y_true:    Ground-truth values.
        y_pred:    Predicted values.
        save_path: If provided, save results as JSON to this path.

    Returns:
        dict of metrics.
    """
    y_true = np.ravel(np.asarray(y_true))
    y_pred = np.ravel(np.asarray(y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        "task": "regression",
        "n_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    _save(metrics, save_path)
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"\n=== {metrics.get('task', 'Metrics').upper()} ===")
    for k, v in metrics.items():
        if k != "task":
            print(f"  {k}: {v}")


def _save(data: dict, path: str | None) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved -> {path}")
