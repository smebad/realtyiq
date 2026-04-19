import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor

from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.data.features import engineer_features, get_feature_columns
from src.ml.evaluate import evaluate_model, plot_feature_importance, plot_shap_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_PATH   = Path("data/raw/AmesHousing.csv")
MODEL_DIR       = Path("models")
MODEL_PATH      = MODEL_DIR / "xgboost_price_model.pkl"
METADATA_PATH   = MODEL_DIR / "model_metadata.json"
ENCODER_PATH    = MODEL_DIR / "target_encoder.pkl"

# Config
TARGET          = "SalePrice"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
HIGH_CARD_COLS  = ["Neighborhood", "House Style", "MS SubClass"]

XGBOOST_PARAMS  = {
    "n_estimators":       1000,
    "learning_rate":      0.05,
    "max_depth":          6,
    "min_child_weight":   1,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,   # L1 regularisation
    "reg_lambda":         1.0,   # L2 regularisation
    "random_state":       RANDOM_STATE,
    "n_jobs":             -1,
    "early_stopping_rounds": 50,
}

# End to end training pipeline
def run_training_pipeline() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Load and process data
    logger.info("Loading and processing data ...")
    df = load_raw_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    # 2. Target encode high cardinality categoricals
    logger.info("Target-encoding high-cardinality columns ...")
    df, encoder = _target_encode(df, HIGH_CARD_COLS, TARGET)

    # 3. Log-transform target
    # Log transform makes price distribution more normal
    # models trained on log(price) almost always outperform raw price
    df["log_price"] = np.log1p(df[TARGET])

    # 4. Train / test split
    feature_cols = get_feature_columns(df, target=TARGET)
    feature_cols = [c for c in feature_cols if c != "log_price"]

    X = df[feature_cols]
    y = df["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # 5. Cross validation
    logger.info("Running 5-fold cross validation ...")
    cv_model = XGBRegressor(**{
        k: v for k, v in XGBOOST_PARAMS.items()
        if k != "early_stopping_rounds"
    }, n_estimators=300)

    cv_scores = cross_val_score(
        cv_model, X_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2",
        n_jobs=-1,
    )
    logger.info(
        f"CV R² scores: {cv_scores.round(4)} | "
        f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )

    # 6. Final model training with early stopping
    logger.info("Training final model with early stopping ...")
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )
    logger.info(f"Best iteration: {model.best_iteration}")

    # 7. Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_cols)
    metrics["cv_r2_mean"]  = round(float(cv_scores.mean()), 4)
    metrics["cv_r2_std"]   = round(float(cv_scores.std()), 4)
    metrics["best_iteration"] = int(model.best_iteration)
    metrics["train_rows"]  = len(X_train)
    metrics["test_rows"]   = len(X_test)
    metrics["n_features"]  = len(feature_cols)

    logger.info("\n" + "─" * 40)
    logger.info("FINAL METRICS")
    logger.info("─" * 40)
    for k, v in metrics.items():
        logger.info(f"  {k:<25} {v}")

    # 8. Generate explainability plots
    logger.info("Generating SHAP and feature importance plots ...")
    plot_feature_importance(model, feature_cols)
    plot_shap_values(model, X_test, feature_cols)

    # 9. Save model artifacts
    _save_artifacts(model, encoder, feature_cols, metrics)

    logger.info(f"Model saved to {MODEL_PATH}")
    return metrics

# Function for target encoding high-cardinality categorical features
def _target_encode(
    df: pd.DataFrame,
    cols: list[str],
    target: str,
) -> tuple[pd.DataFrame, TargetEncoder]:
    
    available = [c for c in cols if c in df.columns]
    if not available:
        return df, None

    encoder = TargetEncoder(smooth="auto", random_state=42)
    df[available] = encoder.fit_transform(df[available], df[target])

    return df, encoder

# Function for saving model artifacts
def _save_artifacts(
    model: XGBRegressor,
    encoder,
    feature_cols: list[str],
    metrics: dict,
) -> None:

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save encoder
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    # Save metadata: version, features, metrics
    metadata = {
        "model_version":  "v1.0",
        "model_type":     "XGBRegressor",
        "target":         "log(SalePrice)",
        "feature_cols":   feature_cols,
        "metrics":        metrics,
        "xgboost_params": {
            k: v for k, v in XGBOOST_PARAMS.items()
            if k != "early_stopping_rounds"
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    run_training_pipeline()