from pathlib import Path
import pandas as pd
import pytest

from src.data.loader import load_raw_data, get_data_summary
from src.data.cleaner import clean_data
from src.data.features import engineer_features, get_feature_columns

RAW_PATH = Path("data/raw/AmesHousing.csv")

# Test data loading
@pytest.mark.skipif(not RAW_PATH.exists(), reason="Raw data not downloaded yet.")
def test_load_raw_data():
  df = load_raw_data(RAW_PATH)
  assert len(df) > 2000, "Expected atleast 2000 rows"
  assert "SalePrice" in df.columns

# Test clean data
@pytest.mark.skipif(not RAW_PATH.exists(), reason="Raw data not downloaded yet.")
def test_clean_data():
  df = load_raw_data(RAW_PATH)
  cleaned = clean_data(df)
  # No null should remain
  assert cleaned.isnull().sum().sum() == 0, "Nulls remain after cleaning"
  # Outliers removed
  assert cleaned["Gr Liv Area"].max() <= 4000

# Test feature engineering
@pytest.mark.skipif(not RAW_PATH.exists(), reason="Raw data not downloaded yet.")
def test_feature_engineering():
  df = load_raw_data(RAW_PATH)
  cleaned = clean_data(df)
  featured = engineer_features(cleaned)
  # New columns were created
  assert "house_age" in featured.columns
  assert "total_bathrooms" in featured.columns
  assert "qual_x_area" in featured.columns
  # No new nulls introduced
  assert featured.isnull().sum().sum() == 0

# Test get feature columns
@pytest.mark.skipif(not RAW_PATH.exists(), reason="Raw data not downloaded yet")
def test_get_feature_columns():
    df = load_raw_data(RAW_PATH)
    cleaned = clean_data(df)
    featured = engineer_features(cleaned)
    features = get_feature_columns(featured)
    assert "SalePrice" not in features
    assert len(features) > 10