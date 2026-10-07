"""Model evaluation and comparison utilities.

Functions:
    evaluate_model, compare_models, generate_confusion_matrix,
    compute_pr_auc, find_optimal_threshold, precision_at_k
Plus plotting helpers that save PNG reports.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray,
                   k: int = 10) -> float:
    """Compute Precision@K (fraction of top-K scores that are positive).

    Args:
        y_true: True binary labels.
        y_scores: Predicted scores / probabilities.
        k: Cut-off rank.

    Returns:
        Precision@K float in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    top_k = np.argsort(y_scores)[::-1][:k]
    return float(np.mean(y_true[top_k]))


def evaluate_model(model, X_test, y_test,
                   threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate a classifier on the test set.

    Args:
        model: Fitted classifier with predict_proba.
        X_test: Test features.
        y_test: True labels.
        threshold: Decision threshold on positive-class probability.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, pr_auc,
        precision_at_10 and confusion-matrix counts (tn, fp, fn, tp).
    """
    try:
        y_test = np.asarray(y_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:  # fallback (e.g. decision_function)
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max()
                                                 - scores.min() + 1e-9)
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred,
                                         labels=[0, 1]).ravel()
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": compute_pr_auc(y_test, y_proba),
            "precision_at_10": precision_at_k(y_test, y_proba, k=10),
            "threshold": threshold,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        logger.info("Evaluation (thr=%.2f): %s", threshold,
                    {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in metrics.items()})
        return metrics
    except Exception as exc:  # noqa: BLE001
        logger.error("evaluate_model failed: %s", exc)
        raise


def compare_models(models_list: List[Tuple[str, object]], X_test, y_test,
                   threshold: float = 0.5) -> pd.DataFrame:
    """Evaluate several models and return a side-by-side comparison table.

    Args:
        models_list: List of (name, fitted model) tuples.
        X_test: Test features.
        y_test: True labels.
        threshold: Decision threshold applied to all models.

    Returns:
        DataFrame indexed by model name with metric columns.
    """
    rows = []
    for name, model in models_list:
        try:
            m = evaluate_model(model, X_test, y_test, threshold)
            m["model"] = name
            rows.append(m)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to evaluate %s: %s", name, exc)
    df = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
    if not df.empty:
        logger.info("\n%s", df.round(4).to_string())
    return df


def generate_confusion_matrix(y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict[str, int]:
    """Compute confusion-matrix counts.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dict with tn, fp, fn, tp.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                      labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute area under the precision-recall curve.

    Args:
        y_true: True labels.
        y_scores: Predicted probabilities.

    Returns:
        PR-AUC float (average precision).
    """
    try:
        return float(average_precision_score(y_true, y_scores))
    except Exception as exc:  # noqa: BLE001
        logger.error("compute_pr_auc failed: %s", exc)
        return 0.0


def find_optimal_threshold(y_val: np.ndarray, y_proba: np.ndarray,
                           target_recall: float = 0.90,
                           min_precision: float = 0.85) -> float:
    """Find threshold maximising recall subject to precision >= min.

    Uses sklearn's precision_recall_curve. Falls back to the threshold with
    the best F1 if no threshold satisfies the precision constraint.

    Args:
        y_val: Validation labels.
        y_proba: Validation probabilities.
        target_recall: Desired recall level (informational).
        min_precision: Minimum acceptable precision.

    Returns:
        Optimal threshold float.
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_val,
                                                               y_proba)
        # precision_recall_curve returns len(thresholds) = len(p)-1
        best_thr, best_rec = 0.5, -1.0
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
            if p >= min_precision and r > best_rec:
                best_rec, best_thr = r, float(t)
        if best_rec < 0:  # no threshold met constraint -> max F1
            f1s = [2 * p * r / (p + r + 1e-9)
                   for p, r in zip(precision[:-1], recall[:-1])]
            best_thr = float(thresholds[int(np.argmax(f1s))])
            logger.warning(
                "No threshold achieved precision>=%.2f; using max-F1 "
                "threshold=%.3f", min_precision, best_thr)
        else:
            logger.info("Optimal threshold=%.3f (recall=%.3f, "
                        "target=%.2f)", best_thr, best_rec, target_recall)
        return best_thr
    except Exception as exc:  # noqa: BLE001
        logger.error("find_optimal_threshold failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Plotting (PNG reports)
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: str = "reports/confusion_matrix.png",
                          title: str = "Confusion Matrix") -> str:
    """Plot and save a confusion-matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Output PNG path.
        title: Plot title.

    Returns:
        The save path.
    """
    _ensure_dir(save_path)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Dissatisfied", "Satisfied"],
                yticklabels=["Dissatisfied", "Satisfied"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)
    return save_path


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   save_path: str = "reports/roc_curve.png",
                   label: str = "model") -> str:
    """Plot and save ROC curve.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        save_path: Output PNG path.
        label: Legend label.

    Returns:
        The save path.
    """
    _ensure_dir(save_path)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)
    return save_path


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray,
                  save_path: str = "reports/pr_curve.png",
                  label: str = "model") -> str:
    """Plot and save precision-recall curve.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        save_path: Output PNG path.
        label: Legend label.

    Returns:
        The save path.
    """
    _ensure_dir(save_path)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{label} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)
    return save_path


def plot_feature_importance(importances: np.ndarray,
                            feature_names: List[str], top_n: int = 10,
                            save_path: Optional[str] = None) -> "plt.Figure":
    """Plot top-N feature importances (returns figure, optionally saves).

    Args:
        importances: Importance array aligned with feature_names.
        feature_names: Feature name list.
        top_n: Number of top features to show.
        save_path: Optional PNG path.

    Returns:
        Matplotlib figure.
    """
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx][::-1],
            importances[idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved %s", save_path)
    return fig
