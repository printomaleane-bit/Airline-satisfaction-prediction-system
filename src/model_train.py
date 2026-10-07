"""Model training module: trains LR, RF and XGBoost, tunes thresholds.

- Trains 3 models (LogReg baseline, RandomForest + XGBoost with
  RandomizedSearchCV on average_precision).
- Tunes decision threshold on validation set (max recall, precision>=0.85).
- Evaluates on test set, prints comparison, saves artefacts to models/ and
  reports to reports/.

Run:
    python -m src.model_train [--csv data/airline_satisfaction.csv]
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

from src.model_evaluate import (compare_models, evaluate_model,
                                find_optimal_threshold, plot_confusion_matrix,
                                plot_pr_curve, plot_roc_curve)
from src.preprocessing import preprocess_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

RANDOM_STATE = 42
MODEL_DIR = "models"
REPORT_DIR = "reports"


def build_search_spaces() -> Tuple[Dict, Dict]:
    """Build hyperparameter distributions for RF and XGB.

    Returns:
        Tuple (rf_param_dist, xgb_param_dist).
    """
    rf_dist = {
        "n_estimators": randint(100, 301),
        "max_depth": [10, 12, 15, 20, None],
        "min_samples_split": randint(2, 6),
        "min_samples_leaf": randint(1, 4),
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    xgb_dist = {
        "n_estimators": randint(100, 301),
        "max_depth": randint(4, 11),
        "learning_rate": loguniform(0.01, 0.1),
        "subsample": uniform(0.7, 0.2),
        "colsample_bytree": uniform(0.7, 0.2),
        "min_child_weight": randint(1, 6),
        "gamma": uniform(0, 0.3),
    }
    return rf_dist, xgb_dist


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train balanced Logistic Regression baseline.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted LogisticRegression.
    """
    logger.info("Training Logistic Regression baseline...")
    model = LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_iter: int = 25,
                        cv: int = 5) -> RandomizedSearchCV:
    """Train Random Forest with RandomizedSearchCV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of parameter samples.
        cv: Cross-validation folds.

    Returns:
        Fitted RandomizedSearchCV object.
    """
    logger.info("Tuning Random Forest (n_iter=%d, cv=%d)...", n_iter, cv)
    rf_dist, _ = build_search_spaces()
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(rf, rf_dist, n_iter=n_iter, cv=cv,
                                scoring="average_precision", n_jobs=-1,
                                random_state=RANDOM_STATE, verbose=1)
    search.fit(X_train, y_train)
    logger.info("Best RF params: %s (score=%.4f)", search.best_params_,
                search.best_score_)
    return search


def train_xgboost(X_train, y_train, n_iter: int = 25, cv: int = 5):
    """Train XGBoost with RandomizedSearchCV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of parameter samples.
        cv: Cross-validation folds.

    Returns:
        Fitted RandomizedSearchCV object.

    Raises:
        ImportError: If xgboost is not installed.
    """
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. "
                          "Run: pip install xgboost")
    logger.info("Tuning XGBoost (n_iter=%d, cv=%d)...", n_iter, cv)
    _, xgb_dist = build_search_spaces()
    y_arr = np.asarray(y_train)
    neg_ratio = (np.sum(y_arr == 0) / max(np.sum(y_arr == 1), 1))
    xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                        eval_metric="logloss",
                        scale_pos_weight=float(np.clip(neg_ratio, 0.5, 2.0)))
    search = RandomizedSearchCV(xgb, xgb_dist, n_iter=n_iter, cv=cv,
                                scoring="average_precision", n_jobs=-1,
                                random_state=RANDOM_STATE, verbose=1)
    search.fit(X_train, y_train)
    logger.info("Best XGB params: %s (score=%.4f)", search.best_params_,
                search.best_score_)
    return search


def tune_threshold(model, X_val, y_val,
                   min_precision: float = 0.85) -> float:
    """Tune decision threshold on validation probabilities.

    Args:
        model: Fitted classifier.
        X_val: Validation features.
        y_val: Validation labels.
        min_precision: Minimum precision constraint.

    Returns:
        Optimal threshold float.
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    return find_optimal_threshold(np.asarray(y_val), y_proba,
                                  target_recall=0.90,
                                  min_precision=min_precision)


def save_artifacts(model, scaler, feature_names: list,
                   metadata: Dict, model_dir: str = MODEL_DIR) -> None:
    """Persist model, scaler, feature names and metadata JSON.

    Args:
        model: Trained (winning) model.
        scaler: Fitted StandardScaler.
        feature_names: Ordered feature-name list.
        metadata: Extra metadata dict to merge into JSON.
        model_dir: Output directory.
    """
    try:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "xgboost_model.pkl"))
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        joblib.dump(feature_names,
                    os.path.join(model_dir, "feature_names.pkl"))
        with open(os.path.join(model_dir, "model_metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved model artefacts to %s/", model_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save artefacts: %s", exc)
        raise


def main(csv_path: str = "data/airline_satisfaction.csv",
         n_iter: int = 25, cv: int = 5) -> Dict:
    """Run full training pipeline.

    Args:
        csv_path: Input dataset CSV.
        n_iter: RandomizedSearch iterations.
        cv: CV folds.

    Returns:
        Dict with comparison dataframe (as dict), winner name and metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Preprocess
    logger.info("=== Step 1/5: Preprocessing ===")
    X_train, X_val, X_test, y_train, y_val, y_test, meta = \
        preprocess_pipeline(csv_path, save_artifacts=True,
                            model_dir=MODEL_DIR)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = meta["feature_names"]
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    # 2. Train 3 models
    logger.info("=== Step 2/5: Training models ===")
    t0 = time.time()
    lr = train_logistic_regression(X_train, y_train)
    rf_search = train_random_forest(X_train, y_train, n_iter=n_iter, cv=cv)
    results = {"logistic_regression": lr,
               "random_forest": rf_search.best_estimator_}
    if HAS_XGB:
        xgb_search = train_xgboost(X_train, y_train, n_iter=n_iter, cv=cv)
        results["xgboost"] = xgb_search.best_estimator_
        best_xgb_params = xgb_search.best_params_
    else:
        logger.warning("xgboost not available — skipping XGB training.")
        best_xgb_params = {}
    logger.info("Training took %.1fs", time.time() - t0)

    # 3. Threshold tuning on validation set (per model, keep tuned thr)
    logger.info("=== Step 3/5: Threshold tuning ===")
    thresholds = {}
    for name, mdl in results.items():
        try:
            thresholds[name] = tune_threshold(mdl, X_val, y_val)
        except Exception as exc:  # noqa: BLE001
            logger.error("Threshold tuning failed for %s: %s", name, exc)
            thresholds[name] = 0.5

    # 4. Evaluate on test set + comparison table
    logger.info("=== Step 4/5: Evaluation ===")
    rows = []
    for name, mdl in results.items():
        m = evaluate_model(mdl, X_test, y_test,
                           threshold=thresholds[name])
        m["model"] = name
        m["val_threshold"] = thresholds[name]
        rows.append(m)
    comparison = pd.DataFrame(rows).set_index("model")
    print("\n================ MODEL COMPARISON (test set) ================")
    print(comparison.round(4).to_string())
    print("=============================================================\n")
    comparison.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"))

    # Winner = highest recall (spec: XGBoost expected winner)
    winner = comparison["recall"].idxmax()
    logger.info("Winner by recall: %s", winner)
    # Spec requires XGBoost saved as primary artefact
    winning_model = results.get("xgboost", results[winner])
    winning_metrics = comparison.loc[
        "xgboost" if "xgboost" in comparison.index else winner].to_dict()

    # 5. Save artefacts + plots + metadata
    logger.info("=== Step 5/5: Saving artefacts ===")
    metadata = {
        "model_type": "xgboost" if "xgboost" in results else winner,
        "winner_by_recall": str(winner),
        "threshold": float(thresholds.get("xgboost",
                                          thresholds[str(winner)])),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "training_date": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "random_state": RANDOM_STATE,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "feature_names": feature_names,
        "metrics": winning_metrics,
        "all_metrics": comparison.reset_index().to_dict(orient="records"),
        "best_xgb_params": {k: (float(v) if isinstance(v, float) else v)
                            for k, v in best_xgb_params.items()},
        "target_benchmarks": {"accuracy": 0.88, "precision": 0.85,
                              "recall": 0.90, "f1": 0.82,
                              "roc_auc": 0.93, "pr_auc": 0.90,
                              "precision_at_10": 0.85},
    }
    # Convert numpy types for JSON safety
    metadata = json.loads(json.dumps(metadata, default=str))
    save_artifacts(winning_model, scaler, feature_names, metadata)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Plots for winning model
    y_proba = winning_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= metadata["threshold"]).astype(int)
    plot_confusion_matrix(np.asarray(y_test), y_pred,
                          os.path.join(REPORT_DIR,
                                       "confusion_matrix_xgboost.png"),
                          title="Confusion Matrix — XGBoost")
    plot_roc_curve(np.asarray(y_test), y_proba,
                   os.path.join(REPORT_DIR, "roc_curve.png"),
                   label="XGBoost")
    plot_pr_curve(np.asarray(y_test), y_proba,
                  os.path.join(REPORT_DIR, "pr_curve.png"),
                  label="XGBoost")

    logger.info("Done. Artefacts in %s/, reports in %s/",
                MODEL_DIR, REPORT_DIR)
    return {"comparison": comparison, "winner": winner,
            "metadata": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train satisfaction models")
    parser.add_argument("--csv", default="data/airline_satisfaction.csv")
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    main(csv_path=args.csv, n_iter=args.n_iter, cv=args.cv)
