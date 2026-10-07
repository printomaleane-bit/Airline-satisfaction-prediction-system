"""Smoke tests for preprocessing and evaluation utilities."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))


def test_engineer_features_creates_nine():
    """Engineered dataframe must contain the 9 new features."""
    from src.preprocessing import engineer_features, generate_synthetic_data
    df = generate_synthetic_data(n_rows=200, seed=0)
    eng = engineer_features(df)
    expected = ["delay_ratio", "service_quality_score",
                "customer_value_index", "delay_interaction",
                "delay_distance_ratio", "service_delay_gap",
                "passenger_age_category", "class_ordinal",
                "loyalty_indicator"]
    for col in expected:
        assert col in eng.columns, f"missing {col}"


def test_preprocess_pipeline_shapes():
    """Pipeline returns 60/20/20 chronological splits."""
    from src.preprocessing import preprocess_pipeline
    X_tr, X_v, X_te, y_tr, y_v, y_te, meta = preprocess_pipeline()
    assert len(X_tr) > len(X_v) and abs(len(X_v) - len(X_te)) <= 2
    assert X_tr.shape[1] == X_v.shape[1] == X_te.shape[1]
    assert meta["n_features"] == X_tr.shape[1]


def test_evaluate_model_metrics_range():
    """evaluate_model returns metrics in [0, 1] on synthetic data."""
    from sklearn.linear_model import LogisticRegression
    from src.model_evaluate import evaluate_model, find_optimal_threshold
    from src.preprocessing import preprocess_pipeline
    X_tr, X_v, X_te, y_tr, y_v, y_te, _ = preprocess_pipeline()
    clf = LogisticRegression(max_iter=200).fit(X_tr[:1000], y_tr[:1000])
    proba = clf.predict_proba(X_v[:500])[:, 1]
    thr = find_optimal_threshold(np.asarray(y_v[:500]), proba)
    assert 0.0 <= thr <= 1.0
    m = evaluate_model(clf, X_te[:500], y_te[:500], threshold=thr)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc",
              "pr_auc", "precision_at_10"):
        assert 0.0 <= m[k] <= 1.0, k


def test_api_prepare_and_heuristic():
    """API helpers work without trained artefacts."""
    import pandas as pd
    from api.main import _heuristic_proba, _prepare  # noqa
    raw = pd.DataFrame([{
        "age": 30, "gender": "Male", "customer_type": "New",
        "travel_class": "Economy", "travel_type": "Personal travel",
        "airline": "Airline_A", "aircraft_type": "B737",
        "departure_delay": 10, "arrival_delay": 5,
        "flight_distance": 1000, "miles_flown": 5000,
        "seat_comfort": 4, "food_drink": 4, "onboard_service": 4,
        "wifi": 3, "baggage": 4, "cleanliness": 4}])
    X = _prepare(raw)
    assert len(X) == 1 and X.notnull().all().all()
    p = _heuristic_proba(raw)
    assert 0.0 <= float(p[0]) <= 1.0
