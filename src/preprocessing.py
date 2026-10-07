"""Data preprocessing module for Airline Customer Satisfaction Prediction.

Handles loading, missing-value treatment, feature engineering (9 features),
encoding, chronological splitting and scaling.

Typical usage:
    from src.preprocessing import preprocess_pipeline
    X_train, X_val, X_test, y_train, y_val, y_test, meta = preprocess_pipeline()
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "satisfaction_label"
RANDOM_STATE = 42

# Columns expected in the raw CSV (flexible — missing ones are synthesised
# with sensible defaults so the pipeline still runs on partial data).
NUMERIC_COLS = [
    "age", "departure_delay", "arrival_delay", "flight_distance",
    "miles_flown", "seat_comfort", "food_drink", "onboard_service",
    "wifi", "baggage", "cleanliness",
]
ONEHOT_COLS = ["airline", "aircraft_type", "travel_type"]
OTHER_CAT_COLS = ["gender", "customer_type", "travel_class"]

COLUMN_ALIASES = {
    # common variants in public airline-satisfaction datasets -> canonical
    "Age": "age",
    "Gender": "gender",
    "Customer Type": "customer_type",
    "Type of Travel": "travel_type",
    "Class": "travel_class",
    "Flight Distance": "flight_distance",
    "Seat comfort": "seat_comfort",
    "Food and drink": "food_drink",
    "Inflight wifi service": "wifi",
    "Inflight entertainment": "onboard_service",
    "On-board service": "onboard_service",
    "Ease of Online booking": "booking_ease",
    "On-board service ": "onboard_service",
    "Leg room service": "leg_room",
    "Baggage handling": "baggage",
    "Checkin service": "checkin",
    "Cleanliness": "cleanliness",
    "Departure Delay in Minutes": "departure_delay",
    "Arrival Delay in Minutes": "arrival_delay",
    "satisfaction": "satisfaction_label",
    "satisfaction_v2": "satisfaction_label",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known alias columns to canonical snake_case names.

    Args:
        df: Raw dataframe.

    Returns:
        Dataframe with renamed columns.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns.astype(str)]
    rename = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
        logger.info("Renamed columns: %s", rename)
    # lower-case + underscores for anything else
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Two raw columns can map to the same target (e.g. 'Inflight
    # entertainment' and 'On-board service' -> 'onboard_service').
    # Average duplicates instead of silently keeping both.
    if len(set(df.columns)) != len(df.columns):
        out = pd.DataFrame(index=df.index)
        for col in dict.fromkeys(df.columns):  # unique, order-preserving
            block = df.loc[:, df.columns == col]
            if block.shape[1] == 1:
                out[col] = block.iloc[:, 0]
            else:
                num = block.apply(pd.to_numeric, errors="coerce")
                if num.notna().any().any():
                    out[col] = num.mean(axis=1)
                else:
                    out[col] = block.iloc[:, 0]
        df = out
        logger.info("Averaged duplicate columns after renaming.")
    # Drop ID / index-like columns that carry no signal (would leak order)
    for id_col in ["id", "unnamed:_0", "unnamed:_00", "unnamed__0",
                   "unnamed:_0.1", "index"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
            logger.info("Dropped ID column '%s'.", id_col)
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load airline satisfaction dataset from CSV.

    Falls back to generating a synthetic dataset (for demo / CI) when the
    file does not exist.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Raw dataframe.

    Raises:
        ValueError: If the file exists but cannot be parsed.
    """
    if not os.path.exists(csv_path):
        logger.warning("Dataset %s not found — generating synthetic data.",
                       csv_path)
        return generate_synthetic_data(n_rows=5000, seed=RANDOM_STATE)
    try:
        df = pd.read_csv(csv_path)
        logger.info("Loaded dataset %s with shape %s", csv_path, df.shape)
        return normalise_columns(df)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load dataset: %s", exc)
        raise ValueError(f"Could not load dataset {csv_path}: {exc}") from exc


def generate_synthetic_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic airline-satisfaction-like dataset.

    Args:
        n_rows: Number of rows to generate.
        seed: Random seed.

    Returns:
        Synthetic dataframe with the expected schema.
    """
    rng = np.random.default_rng(seed)
    airlines = ["Airline_A", "Airline_B", "Airline_C"]
    aircraft = ["B737", "A320", "B777"]
    travel_types = ["Business travel", "Personal travel"]
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n_rows),
        "gender": rng.choice(["Male", "Female"], n_rows),
        "customer_type": rng.choice(["New", "Returning",
                                     "Loyal Customer",
                                     "disloyal Customer"], n_rows),
        "travel_class": rng.choice(["Economy", "Business", "First"], n_rows,
                                   p=[0.6, 0.3, 0.1]),
        "travel_type": rng.choice(travel_types, n_rows),
        "airline": rng.choice(airlines, n_rows),
        "aircraft_type": rng.choice(aircraft, n_rows),
        "departure_delay": np.clip(rng.normal(15, 40, n_rows), -30, 300),
        "arrival_delay": np.clip(rng.normal(12, 40, n_rows), -30, 300),
        "flight_distance": rng.integers(100, 4000, n_rows),
        "miles_flown": rng.integers(0, 300000, n_rows),
        "seat_comfort": rng.integers(1, 6, n_rows),
        "food_drink": rng.integers(1, 6, n_rows),
        "onboard_service": rng.integers(1, 6, n_rows),
        "wifi": rng.integers(1, 6, n_rows),
        "baggage": rng.integers(1, 6, n_rows),
        "cleanliness": rng.integers(1, 6, n_rows),
        "humidity": rng.uniform(20, 90, n_rows),
        "rainfall": rng.exponential(2.0, n_rows),
        "wind_speed": rng.uniform(0, 40, n_rows),
        "destination": rng.choice(["JFK", "LAX", "ORD", "DFW"], n_rows),
    })
    # Latent satisfaction score -> binary label (~52/48-ish balance)
    service = df[["seat_comfort", "food_drink", "onboard_service",
                  "wifi"]].mean(axis=1)
    delay_pen = (df["departure_delay"].clip(lower=0)
                 + df["arrival_delay"].clip(lower=0)) / 120.0
    logit = (service - 3.0) * 1.6 - delay_pen * 1.2 \
        + (df["travel_class"] == "First") * 0.6 \
        + (df["travel_class"] == "Business") * 0.3 \
        + rng.normal(0, 0.8, n_rows)
    prob = 1 / (1 + np.exp(-logit))
    df["satisfaction_label"] = (rng.random(n_rows) < prob).astype(int)
    # Inject ~2% missing values to exercise missing-value handling
    for col in ["arrival_delay", "miles_flown", "wifi"]:
        mask = rng.random(n_rows) < 0.02
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values per column.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with columns [missing_count, missing_pct] sorted desc.
    """
    missing = df.isnull().sum()
    report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": (missing / max(len(df), 1) * 100).round(2),
    })
    report = report[report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False)
    if not report.empty:
        logger.info("Missing values:\n%s", report.to_string())
    else:
        logger.info("No missing values detected.")
    return report


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values: median for numerics, mode for categoricals.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with no missing values.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            logger.info("Filled NaN in numeric '%s' with median=%s", col,
                        fill)
        else:
            fill = df[col].mode().iloc[0] if not df[col].mode().empty \
                else "Unknown"
            df[col] = df[col].fillna(fill)
            logger.info("Filled NaN in categorical '%s' with mode=%s", col,
                        fill)
    return df


# ---------------------------------------------------------------------------
# Feature engineering (9 features)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 9 engineered features.

    Args:
        df: Dataframe with base columns.

    Returns:
        Dataframe with 9 additional columns.
    """
    df = df.copy()

    # Ensure required base columns exist (defaults if absent)
    defaults = {"departure_delay": 0.0, "arrival_delay": 0.0,
                "flight_distance": 1000.0, "miles_flown": 0.0,
                "seat_comfort": 3, "food_drink": 3, "onboard_service": 3,
                "wifi": 3, "age": 35, "travel_class": "Economy",
                "customer_type": "New"}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
            logger.warning("Column '%s' missing — filled with default %s",
                           col, val)

    dep = pd.to_numeric(df["departure_delay"], errors="coerce").fillna(0)
    arr = pd.to_numeric(df["arrival_delay"], errors="coerce").fillna(0)
    dist = pd.to_numeric(df["flight_distance"],
                         errors="coerce").fillna(1000).clip(lower=1)
    miles = pd.to_numeric(df["miles_flown"], errors="coerce").fillna(0)
    age = pd.to_numeric(df["age"], errors="coerce").fillna(35)

    service_cols = [c for c in ["seat_comfort", "food_drink",
                                "onboard_service", "wifi"] if c in df.columns]
    service = df[service_cols].mean(axis=1) if service_cols \
        else pd.Series(3.0, index=df.index)

    is_business = df["travel_class"].astype(str).str.lower().eq("business")
    cust = df["customer_type"].astype(str).str.lower()
    cust_mult = np.where(cust.str.contains("loyal|return"), 1.5, 1.0)

    # 1. delay_ratio
    df["delay_ratio"] = (dep + arr) / (dist + 1)
    # 2. service_quality_score
    df["service_quality_score"] = service
    # 3. customer_value_index
    df["customer_value_index"] = (miles / 1000.0) * cust_mult
    # 4. delay_interaction
    df["delay_interaction"] = dep * np.where(is_business, 2, 1)
    # 5. delay_distance_ratio
    df["delay_distance_ratio"] = df["delay_ratio"] / np.sqrt(dist)
    # 6. service_delay_gap
    df["service_delay_gap"] = df["service_quality_score"] - \
        (df["delay_ratio"] * 50)
    # 7. passenger_age_category (ordinal 0-3)
    df["passenger_age_category"] = pd.cut(
        age, bins=[0, 25, 40, 60, 120],
        labels=[0, 1, 2, 3]).astype(int)
    # 8. class_ordinal (handles Economy/Business/First and Eco/Eco Plus)
    class_map = {"economy": 1, "eco": 1, "eco_plus": 1, "ecoplus": 1,
                 "eco plus": 1, "business": 2, "first": 3}
    df["class_ordinal"] = df["travel_class"].astype(str).str.lower() \
        .str.replace(" ", "_").map(class_map).fillna(1).astype(int)
    # 9. loyalty_indicator
    df["loyalty_indicator"] = (miles > 100000).astype(int)

    logger.info("Engineered 9 features.")
    return df


# ---------------------------------------------------------------------------
# Encoding + split + scale
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame, target_col: str = TARGET_COL
                    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """One-hot encode categoricals and separate target.

    - One-hot: airline, aircraft_type, travel_type (drop_first=True)
    - Binary/label: gender, customer_type mapped to 0/1-ish ints
    - travel_class kept via class_ordinal (original dropped to avoid dup)

    Args:
        df: Engineered dataframe.
        target_col: Name of target column.

    Returns:
        Tuple (X, y, feature_names). y is None if target absent.
    """
    df = df.copy()
    y = None
    # Normalise target: accept 'satisfied'/'dissatisfied' strings too
    if target_col in df.columns:
        y = df[target_col]
        if y.dtype == object:
            y = y.astype(str).str.lower().map(
                {"satisfied": 1, "satisfaction": 1, "1": 1, "true": 1,
                 "dissatisfied": 0, "neutral or dissatisfied": 0,
                 "0": 0, "false": 0}).fillna(0).astype(int)
        else:
            y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        df = df.drop(columns=[target_col])

    # Map simple binary categoricals
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.lower().map(
            {"male": 1, "m": 1, "female": 0, "f": 0}).fillna(0).astype(int)
    if "customer_type" in df.columns:
        df["customer_type"] = df["customer_type"].astype(str).str.lower().map(
            {"loyal customer": 1, "returning": 1, "loyal": 1,
             "disloyal customer": 0, "new": 0}).fillna(0).astype(int)
    # Drop raw travel_class (class_ordinal carries the signal)
    if "travel_class" in df.columns:
        df = df.drop(columns=["travel_class"])
    # Drop free-text / high-cardinality leftovers that are not one-hot cols
    for col in ["destination", "booking_ease", "leg_room", "checkin",
                "humidity", "rainfall", "wind_speed"]:
        # keep humidity/rainfall/wind_speed if present (numeric signal)
        pass

    onehot = [c for c in ONEHOT_COLS if c in df.columns]
    if onehot:
        df = pd.get_dummies(df, columns=onehot, drop_first=True,
                            dtype=float)
    # Any remaining object columns -> one-hot (safe fallback)
    leftover = df.select_dtypes(include=["object"]).columns.tolist()
    if leftover:
        logger.warning("One-hot encoding leftover columns: %s", leftover)
        df = pd.get_dummies(df, columns=leftover, drop_first=True,
                            dtype=float)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df, y, df.columns.tolist()


def chronological_split(
    X: pd.DataFrame, y: Optional[pd.Series],
    train_ratio: float = 0.6, val_ratio: float = 0.2,
) -> Tuple:
    """Chronological (order-preserving) train/val/test split.

    Args:
        X: Feature matrix (row order = time order).
        y: Target series or None.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation (rest goes to test).

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, X_val, X_test = (X.iloc[:n_train], X.iloc[n_train:n_train + n_val],
                              X.iloc[n_train + n_val:])
    if y is None:
        return X_train, X_val, X_test, None, None, None
    return (X_train, X_val, X_test, y.iloc[:n_train],
            y.iloc[n_train:n_train + n_val], y.iloc[n_train + n_val:])


def preprocess_pipeline(
    csv_path: str = "data/airline_satisfaction.csv",
    train_ratio: float = 0.6, val_ratio: float = 0.2,
    save_artifacts: bool = False,
    model_dir: str = "models",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           Optional[pd.Series], Optional[pd.Series],
           Optional[pd.Series], Dict]:
    """End-to-end preprocessing pipeline.

    Args:
        csv_path: Input CSV path.
        train_ratio: Train fraction.
        val_ratio: Validation fraction.
        save_artifacts: If True, persist scaler + feature names.
        model_dir: Directory for saved artifacts.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, metadata dict).
    """
    try:
        df = load_dataset(csv_path)
        missing_report = report_missing(df)
        df = handle_missing(df)
        df = engineer_features(df)
        X, y, feature_names = encode_features(df)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            chronological_split(X, y, train_ratio, val_ratio)

        # Scale numeric features (fit on train only)
        scaler = StandardScaler()
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        # Align columns defensively
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        metadata = {
            "n_rows": len(df),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "numeric_columns": num_cols,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "class_balance": (y.value_counts(normalize=True).to_dict()
                              if y is not None else {}),
            "missing_report": missing_report.to_dict()
            if not missing_report.empty else {},
        }
        logger.info("Preprocessing done: train=%s val=%s test=%s feats=%d",
                    len(X_train), len(X_val), len(X_test),
                    len(feature_names))

        if save_artifacts:
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
            joblib.dump(feature_names,
                        os.path.join(model_dir, "feature_names.pkl"))
            logger.info("Saved scaler + feature names to %s", model_dir)

        return X_train, X_val, X_test, y_train, y_val, y_test, metadata
    except Exception as exc:  # noqa: BLE001
        logger.error("Preprocessing pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    X_tr, X_v, X_te, y_tr, y_v, y_te, meta = preprocess_pipeline(
        save_artifacts=True)
    print(f"Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")
    print(f"Features ({len(meta['feature_names'])}): "
          f"{meta['feature_names'][:10]}...")
    print(f"Class balance: {meta['class_balance']}")
