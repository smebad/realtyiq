import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH    = Path("models/xgboost_price_model.pkl")
METADATA_PATH = Path("models/model_metadata.json")
ENCODER_PATH  = Path("models/target_encoder.pkl")

# Module-level cache — model loads once, stays in memory
_model    = None
_metadata = None
_encoder  = None

# Function to load model, encoder, and metadata into module-level cache
def _load_artifacts() -> None:
    
    global _model, _metadata, _encoder

    if _model is not None:
        return  # already loaded

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run: python -m src.ml.train"
        )

    logger.info("Loading model from disk ...")

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(METADATA_PATH, "r") as f:
        _metadata = json.load(f)

    with open(ENCODER_PATH, "rb") as f:
        _encoder = pickle.load(f)

    logger.info(
        f"Model loaded. Version: {_metadata['model_version']} | "
        f"R²: {_metadata['metrics']['r2']}"
    )

# Function to predict price for a single property
def predict_price(features: dict) -> dict:

    _load_artifacts()

    feature_cols = _metadata["feature_cols"]

    # Build a single-row DataFrame in the exact column order the model expects
    row = {col: features.get(col, 0) for col in feature_cols}
    X   = pd.DataFrame([row])[feature_cols]

    # Predict in log space, convert back to USD
    log_pred        = _model.predict(X)[0]
    predicted_price = float(np.expm1(log_pred))

    # Simple confidence band: ±10% (replace with quantile regression in v2)
    low  = round(predicted_price * 0.90, 2)
    high = round(predicted_price * 1.10, 2)

    return {
        "predicted_price":  round(predicted_price, 2),
        "confidence_range": {"low": low, "high": high},
        "model_version":    _metadata["model_version"],
        "r2_score":         _metadata["metrics"]["r2"],
    }

# Function to return model metadata (used by API endpoint)
def get_model_metadata() -> dict:

    _load_artifacts()
    return _metadata

# Function to predict prices for multiple properties at once
def batch_predict(features_list: list[dict]) -> list[dict]:
    
    return [predict_price(f) for f in features_list]