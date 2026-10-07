"""Utility functions for the Airline Satisfaction project.

Provides data validation, feature encoding helpers, prediction logging,
model serialisation helpers and configuration management.
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration values. Empty dict if file missing.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", config_path)
        return config
    except FileNotFoundError:
        logger.warning("Config file %s not found, using defaults.", config_path)
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load config %s: %s", config_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "age", "gender", "customer_type", "travel_class",
    "departure_delay", "arrival_delay", "flight_distance", "miles_flown",
]

SERVICE_COLS = [
    "seat_comfort", "food_drink", "onboard_service", "wifi", "baggage",
    "cleanliness",
]


def validate_dataframe(df, required: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate that a dataframe contains required columns and sane values.

    Args:
        df: Input pandas DataFrame.
        required: List of required column names. Defaults to REQUIRED_COLUMNS.

    Returns:
        Dict with keys ``valid`` (bool), ``missing_columns`` (list) and
        ``n_rows`` (int).
    """
    required = required or REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    result = {"valid": len(missing) == 0, "missing_columns": missing,
              "n_rows": len(df)}
    if missing:
        logger.warning("Missing columns: %s", missing)
    return result


def validate_passenger_input(data: Dict[str, Any]) -> List[str]:
    """Validate a single passenger input dict.

    Args:
        data: Passenger feature dictionary.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: List[str] = []
    try:
        age = data.get("age")
        if age is None or not (0 < age < 120):
            errors.append("age must be between 1 and 119")
        for col in ("departure_delay", "arrival_delay"):
            val = data.get(col)
            if val is None or not (-60 <= val <= 600):
                errors.append(f"{col} must be between -60 and 600")
        fd = data.get("flight_distance", 0)
        if fd is None or fd <= 0:
            errors.append("flight_distance must be positive")
        for col in SERVICE_COLS:
            val = data.get(col, 3)
            if val is not None and not (1 <= val <= 5):
                errors.append(f"{col} rating must be 1-5")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"validation error: {exc}")
    return errors


# ---------------------------------------------------------------------------
# Feature encoding / decoding
# ---------------------------------------------------------------------------

CLASS_ORDINAL = {"Economy": 1, "Business": 2, "First": 3,
                 "economy": 1, "business": 2, "first": 3,
                 "Eco": 1, "ECO": 1, "eco": 1, "Eco Plus": 1,
                 "eco plus": 1, "eco_plus": 1, "BUSINESS": 2, "FIRST": 3}

AGE_BINS = [0, 25, 40, 60, 120]
AGE_LABELS = ["Young", "Adult", "Middle-aged", "Senior"]


def encode_class_ordinal(value: str) -> int:
    """Encode travel class to ordinal integer.

    Args:
        value: Class string (Economy/Business/First).

    Returns:
        Ordinal int (Economy=1, Business=2, First=3, unknown=1).
    """
    if isinstance(value, (int, float)):
        return int(value)
    return CLASS_ORDINAL.get(str(value).strip(), 1)


def decode_prediction(label: int) -> str:
    """Decode binary label to human readable string.

    Args:
        label: 1 for satisfied, 0 for dissatisfied.

    Returns:
        "SATISFIED" or "DISSATISFIED".
    """
    return "SATISFIED" if int(label) == 1 else "DISSATISFIED"


def recommend_action(prediction: int, confidence: float) -> str:
    """Recommend an operational action from prediction + confidence.

    Args:
        prediction: 1 = satisfied, 0 = dissatisfied.
        confidence: Model confidence in [0, 1].

    Returns:
        One of "NO ACTION", "MONITOR", "INTERVENE".
    """
    if prediction == 1 and confidence >= 0.75:
        return "NO ACTION"
    if prediction == 1:
        return "MONITOR"
    if confidence >= 0.65:
        return "INTERVENE"
    return "MONITOR"


def suggested_interventions(row: Dict[str, Any]) -> List[str]:
    """Generate suggested interventions for a dissatisfied passenger.

    Args:
        row: Passenger feature dict.

    Returns:
        List of human-readable intervention strings.
    """
    actions: List[str] = []
    try:
        if row.get("departure_delay", 0) > 30 or row.get("arrival_delay", 0) > 30:
            actions.append("Proactive delay apology + rebooking / lounge pass")
        service_cols = ["seat_comfort", "food_drink", "onboard_service",
                        "wifi", "baggage", "cleanliness"]
        low = [c for c in service_cols if row.get(c, 5) <= 2]
        if low:
            actions.append(f"Service recovery for: {', '.join(low)} "
                           "(crew check-in, complimentary offer)")
        if str(row.get("travel_class", "")).lower() == "business":
            actions.append("Priority handling: high-value Business passenger")
        if row.get("miles_flown", 0) > 100000:
            actions.append("Loyalty save: bonus miles / tier-care outreach")
        if not actions:
            actions.append("General service check: crew visit + feedback form")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build interventions: %s", exc)
    return actions


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------

def log_prediction(log_path: str, record: Dict[str, Any]) -> None:
    """Append a prediction record to a CSV audit log.

    Args:
        log_path: Path to predictions.csv log file.
        record: Dict to append (must be flat / JSON-serialisable values).
    """
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        row = dict(record)
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to log prediction to %s: %s", log_path, exc)


# ---------------------------------------------------------------------------
# Model serialisation helpers
# ---------------------------------------------------------------------------

def save_artifact(obj: Any, path: str) -> None:
    """Save a Python object with joblib.

    Args:
        obj: Object to save.
        path: Destination path.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(obj, path)
        logger.info("Saved artifact to %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save artifact %s: %s", path, exc)
        raise


def load_artifact(path: str) -> Any:
    """Load a joblib artifact.

    Args:
        path: Path to the .pkl file.

    Returns:
        Deserialised object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    try:
        obj = joblib.load(path)
        logger.info("Loaded artifact from %s", path)
        return obj
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load artifact %s: %s", path, exc)
        raise


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dict as pretty-printed JSON.

    Args:
        data: Dictionary to save.
        path: Destination path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON to %s", path)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
