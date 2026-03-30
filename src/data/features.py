import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Feature engineering functions
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")

    df = df.copy()
    df = _create_age_features(df)
    df = _create_area_features(df)
    df = _create_quality_features(df)
    df = _create_interaction_features(df)
    df = _encode_categoricals(df)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")

    return df

# Function for age based features
def _create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    
    if "Yr Sold" in df.columns and "Year Built" in df.columns:
        df["house_age"] = df["Yr Sold"] - df["Year Built"]
        
        # Clip negatives (data entry errors)
        df["house_age"] = df["house_age"].clip(lower=0)

    if "Yr Sold" in df.columns and "Year Remod/Add" in df.columns:
        df["years_since_remodel"] = df["Yr Sold"] - df["Year Remod/Add"]
        df["years_since_remodel"] = df["years_since_remodel"].clip(lower=0)
        df["was_remodeled"] = (
            df["Year Remod/Add"] != df["Year Built"]
        ).astype(int)
        
    return df

# Function for area based features
def _create_area_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Total finished area above and below ground
    bsmt_cols = ["BsmtFin SF 1", "BsmtFin SF 2"]
    available = [c for c in bsmt_cols if c in df.columns]
    if available and "Gr Liv Area" in df.columns:
        df["total_finished_area"] = df["Gr Liv Area"] + df[available].sum(axis=1)

    # Total bathrooms (full bath = 1, half bath = 0.5)
    bath_cols = {
        "Full Bath": 1.0,
        "Half Bath": 0.5,
        "Bsmt Full Bath": 1.0,
        "Bsmt Half Bath": 0.5,
    }
    bath_total = sum(
        df[col] * weight
        for col, weight in bath_cols.items()
        if col in df.columns
    )
    df["total_bathrooms"] = bath_total

    # Total porch area
    porch_cols = [
        "Open Porch SF", "Enclosed Porch",
        "3Ssn Porch", "Screen Porch"
    ]
    available_porch = [c for c in porch_cols if c in df.columns]
    if available_porch:
        df["total_porch_area"] = df[available_porch].sum(axis=1)

    # Has garage flag
    if "Garage Cars" in df.columns:
        df["has_garage"] = (df["Garage Cars"] > 0).astype(int)

    return df

# Function for quality based features
def _create_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Map string quality ratings to ordered numbers
    quality_map = {
        "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0
    }

    for col in ["Heating QC", "Fireplace Qu", "Garage Qual", "Bsmt Qual"]:
        if col in df.columns:
            new_col = col.lower().replace(" ", "_") + "_score"
            df[new_col] = df[col].map(quality_map).fillna(0).astype(int)

    # Overall quality squared captures non-linear premium at high quality
    if "Overall Qual" in df.columns:
        df["overall_qual_squared"] = df["Overall Qual"] ** 2

    return df

# Function for interaction features
def _create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Quality × Area interaction: a big, high-quality house commands a premium
    if "Overall Qual" in df.columns and "Gr Liv Area" in df.columns:
        df["qual_x_area"] = df["Overall Qual"] * df["Gr Liv Area"]

    # Neighborhood × Quality mean encode (target encode) - will be done in train.py
    # Just flagging here that it's done later..
    return df

# Function for encoding categorical variables
def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:

    # Binary encoding
    if "Central Air" in df.columns:
        df["central_air"] = (df["Central Air"] == "Y").astype(int)
        df = df.drop(columns=["Central Air"])

    # Label encode remaining object columns
    # (XGBoost with enable_categorical handles this natively, but Im encoding explicitly for scikit-learn compatibility)
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target if somehow present in object cols
    object_cols = [c for c in object_cols if c != "SalePrice"]

    le = LabelEncoder()
    for col in object_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    logger.info(f"Label-encoded {len(object_cols)} categorical columns.")
    return df

# Function to get feature columns (excluding target)
def get_feature_columns(df: pd.DataFrame, target: str = "SalePrice") -> list[str]:
    return [col for col in df.columns if col != target]