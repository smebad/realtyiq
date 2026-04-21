import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squarred_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLOTS_DIR = Path("models/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Function to calculate all regression metrics
def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],    
) -> dict:
  
  y_pred_log = model.predict(X_test)

  # Convert back from log space to USD
  y_pred = np.expm1(y_pred_log)
  y_true = np.expm1(y_test)

  rmse = float(np.sqrt(mean_squarred_error(y_true, y_pred)))
  mae = float(mean_absolute_error(y_true, y_pred))
  r2 = float(r2_score(y_true, y_pred))
  mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

  metrics = {
    "rmse": round(rmse, 2),
    "mae": round(mae, 2),
    "r2": round(r2, 2),
    "mape": round(mape, 2),
  }

  # Also save a predicted vs actual plot
  _plot_predicted_vs_actual(y_true, y_pred)

  return metrics

# Function to plot predicted vs actual values
def _plot_predicted_vs_actual(
    y_true: pd.Series, 
    y_pred: np.ndarray,
) -> None:
  
  # Scatter plot of predicted vs actual prices
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.scatter(y_true, y_pred, alpha=0.4, color="#2563eb", s=15)

  # Perfect prediction line
  min_val = min(y_true.min(), y_pred.min())
  max_val = max(y_true.max(), y_pred.max())
  ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)

  ax.set_xlabel("Actual Price (USD)", fontsize=12)
  ax.set_ylabel("Predicted Price (USD)", fontsize=12)
  ax.set_title("Actual vs Predicted Sale Price", fontsize=14, fontweight="bold")

  r2 = float(r2_score(y_true, y_pred))
  ax.annotate(
    f"R² = {r2:.4f}",
    xy=(0.05, 0.92), xycoords="axes fraction",
    fontsize=12, color="darkred",
  )

  plt.tight_layout()
  plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
  plt.close()
  logger.info("Saved actual_vs_predicted.png")

def plot_feature_importance(
    model: XGBRegressor,
    feature_cols: list[str],
    top_n: int = 20,
) -> None:
  importance =model.feature_importances_
  feat_df = (
    pd.DataFrame({"feature": feature_cols, "importance": importance})
    .sort_values("importance", ascending=False)
    .tail(top_n)
  )

  fig, ax = plt.subplots(figsize=(9, 7))
  bars = ax.barh(feat_df["feature"], feat_df["importance"], color="#2563eb")
  ax.set_xlabel("Importance (Gain)", fontsize=12)
  ax.set_title(f"Top {top_n} Feature Importances (XGBoost Gain)", fontsize=13, fontweight="bold")

  # Add value labels on bars
  for bar, val in zip(bars, feat_df["importance"]):
      ax.text(
          bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
          f"{val:.3f}", va="center", fontsize=8
      )

  plt.tight_layout()
  plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
  plt.close()
  logger.info("Saved feature_importance.png")

# Function for SHAP (SHapley Additive exPlanations) summary plot
def plot_shap_values(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    max_display: int = 20,
) -> None:
    
    logger.info("Computing SHAP values (this may take ~30 seconds) ...")

    explainer   = shap.TreeExplainer(model)
    # Use a sample of 500 for speed
    X_sample    = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Summary / Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_cols,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Summary Plot — Feature Impact on Price Prediction",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_summary.png")

    # Bar plot of mean absolute SHAP values
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_cols,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title("Mean |SHAP Value| per Feature", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_bar.png")
