import logging

import numpy as np
import pandas as pd

# Configure module level logger
logger = logging.getlogger(__name__)

# Columns to drop: too many missing values or not relevant for ML pipeline
COLUMNS_TO_DROP = [
  "Order",
  "PID",
  "Misc Feature",
  "Misc Val",
  "Pool QC",
  "Fence",
  "Alley",
]

# Numeric columns where NaN means "not present" and should be filled with 0
NUMERIC_FILL_ZERO = [
  "Total Bsmt SF",
  "Garage Cars",
  "Garage Area",
  "Fireplaces",
  "Mas Vnr Area",
  "BsmtFin SF 1",
  "BsmtFin SF 2",
  "Bsmt Unf SF",
  "Bsmt Full Bath",
  "Bsmt Half Bath",
]

# Categorical columns where NaN means "not present" and should be filled with "None"
CATEGORICAL_FILL_NONE = [
  "Garage Type",
  "Garage Finish",
  "Garage Qual",
  "Garage Cond",
  "Bsmt Qual",
  "Bsmt Cond",
  "Bsmt Exposure",
  "BsmtFin Type 1",
  "BsmtFin Type 2",
  "Mas Vnr Type",
  "Fireplace Qu",
]

# Main function to clean the dataframe
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  logger.info("Starting data cleaning pipeline ...")

  df = df.copy()
  df = _drop_useless_columns(df)
  df = _fix_dtypes(df)
  df = _fill_missing_values(df)
  df = _remove_outliers(df)

  logger.info(f"Data cleaning completed. Final shape: {df.shape}")
  return df

# Helper function to drop columns that are low value or leaky
def _drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
  cols_present = [col for col in COLUMNS_TO_DROP if c in df.columns]
  df = df.drop(columns=cols_present)
  logger.info(f"Dropped {len(cols_present)} columns: {cols_present}")
  return df

# Helper function to fix columns that were loaded with wrong data types
def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
  # MS SubClass is a categorical variable encoded as integers, convert to string
  if "MS SubClass" in df.columns:
    df["MS SubClass"] = df["MS SubClass"].astype(str)
  
  # Convert year columns to integers
  for col in ["Year Built", "Year Remod/Add", "Yr Sold"]:
    if col in df.columns:
      df[col] = df[col].astype(int)

  return df

# Helper function to fill missing values with justifiable strategies
def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
  # Fill "not present" numeric columns with 0
  for col in NUMERIC_FILL_ZERO:
    if col in df.columns:
      df[col] = df[col].fillna(0)
  
  # Fill "not present" categorical columns with "None"
  for col in CATEGORICAL_FILL_NONE:
    if col in df.columns:
      df[col] = df[col].fillna("None")

  # Fill remaining numeric nulls with median
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  for col in numeric_cols:
    if df[col].isnull().sum() > 0:
      median_val = df[col].median()
      df[col] = df[col].fillna(median_val)
      logger.info(f"Filled numeric null in '{col}' with median={median_val:.2f}")

  # Fill remaining categorical nulls with mode
  categorical_cols = df.select_dtypes(include=["object"]).columns
  for col in categorical_cols:
    if df[col].isnull().sum() > 0: 
      mode_val = df[col].mode()[0]
      df[col] = df[col].fillna(mode_val)
      logger.info(f"Filled categorical null in '{col}' with mode='{mode_val}'")

  return df

# Helper function to remove outliers in SalePrice and Gr Liv Area
def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
  original_len = len(df)

  # Remove properties > 4000 sqft
  if "Gr Liv Area" in df.columns:
    df = df[df["Gr Liv Area"] <= 4000]

  # Remove sale prices below $10k
  if "SalePrice" in df.columns:
    df = df[df["SalePrice"] <= 10_000]

  removed = original_len - len(df)
  logger.info(f"Removed {removed} outlier rows. Remaining: {len(df):,}")

  return df