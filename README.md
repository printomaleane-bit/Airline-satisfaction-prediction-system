# ✈️ Airline Customer Satisfaction Prediction

Predict airline passenger satisfaction (**Satisfied vs Dissatisfied**) in real-time with XGBoost, served via a Streamlit dashboard and a FastAPI REST API.

- **Target:** Recall ≥ 90%, F1 ≥ 82%, Precision@10 ≥ 85%, latency ≤ 100ms
- **Models:** Logistic Regression (baseline) · Random Forest · XGBoost (champion)
- **Dataset:** 129,487 rows × 22 features (`data/airline_satisfaction.csv`)

## Project structure

```
├── api/main.py            # FastAPI REST API (/predict, /predict-batch, /health, /model-info, /kpis)
├── app/streamlit_app.py   # Streamlit dashboard (5 tabs)
├── src/
│   ├── preprocessing.py   # loading, missing values, 9 engineered features, encoding, split, scaling
│   ├── model_train.py     # trains 3 models, threshold tuning, evaluation, saves artefacts
│   ├── model_evaluate.py  # metrics, comparison, ROC/PR/confusion-matrix plots
│   └── utils.py           # validation, encoding, logging, serialisation, config
├── config.yaml            # dataset / split / model / KPI configuration
├── data/                  # place airline_satisfaction.csv here
├── models/                # xgboost_model.pkl, scaler.pkl, feature_names.pkl, model_metadata.json
├── reports/               # model_comparison.csv, metrics.json, *.png
└── logs/predictions.csv   # prediction audit log (auto-created)
```

## Quickstart

```powershell
# 1. Create env + install
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Add dataset (or skip — a synthetic demo dataset is auto-generated)
# data/airline_satisfaction.csv  (must contain satisfaction_label target)

# 3. Train (fast demo: --n-iter 5 --cv 2; full: defaults 25/5)
python -m src.model_train --csv data/airline_satisfaction.csv --n-iter 10 --cv 3

# 4a. Streamlit dashboard
streamlit run app/streamlit_app.py

# 4b. FastAPI
uvicorn api.main:app --reload --port 8000
# Swagger UI -> http://localhost:8000/docs
```

## API examples

```powershell
# Health
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 34, "gender": "Female", "customer_type": "Returning",
  "travel_class": "Business", "departure_delay": 45, "arrival_delay": 40,
  "flight_distance": 1200, "miles_flown": 85000, "seat_comfort": 2,
  "food_drink": 2, "service": 2, "wifi": 1, "baggage": 3, "cleanliness": 3
}'
```

## The 9 engineered features

| # | Feature | Formula |
|---|---------|---------|
| 1 | `delay_ratio` | (DepDelay + ArrDelay) / (Distance + 1) |
| 2 | `service_quality_score` | mean(Seat, Food, Service, WiFi) |
| 3 | `customer_value_index` | (Miles/1000) × 1.5 if loyal/returning else 1.0 |
| 4 | `delay_interaction` | DepDelay × 2 if Business else ×1 |
| 5 | `delay_distance_ratio` | delay_ratio / √Distance |
| 6 | `service_delay_gap` | service_score − delay_ratio × 50 |
| 7 | `passenger_age_category` | bins [0-25, 25-40, 40-60, 60+ ] → 0-3 |
| 8 | `class_ordinal` | Economy=1, Business=2, First=3 |
| 9 | `loyalty_indicator` | 1 if Miles > 100k else 0 |

## Deployment

- **Streamlit Cloud:** set main file to `app/streamlit_app.py`, add `requirements.txt`.
- **Render/Railway (API):** start command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
- Commit `models/*.pkl` + `models/model_metadata.json` or retrain in CI.

## Tests

```powershell
pytest tests/ -v
```
