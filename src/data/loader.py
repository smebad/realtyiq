import logging
from pathlib import Path

import pandas as pd

# Configure module level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Columns for ML pipeline
REQUIRED_COLUMNS = [
  "SalePrice",
  "Gr Liv Area",
  "Overall Qual",
  "Year Built",
  "Total Bsmt SF",
  "Garage Cars",
  "Full Bath",
  "Bedroom AbvGr",
  "Lot Area",
  "Neighborhood",
  "House Style",
  "Heating QC",
  "Central Air",
  "Fireplaces",
]

# Function to load raw data from a CSV file
def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    filepath = Path(file_path)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}. "
            "Download the dataset and place it in the data/raw/ directory."
        )

    logger.info(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    _validate_columns(df)

    return df

# Helper function to validate the presence of required columns
def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing required comlumns: {missing}\n"
            "Make sure you have the correct dataset and that it is properly formatted. "
        )
    
    logger.info("Column validation passed. All required columns are present.")

# Function to summarize the dataframe
def get_data_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict(),
    }