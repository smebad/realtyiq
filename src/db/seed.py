import logging
import sys
from pathlib import Path

import pandas as pd

from src import db
from src.db.database import SessionLocal, create_all_tables
from src.db.crud import create_listing, count_listings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/ames_featured.csv")

# Map CSV column names to model field names
COLUMN_MAP = {
    "Neighborhood":        "neighborhood",
    "House Style":         "house_style",
    "MS Zoning":           "ms_zoning",
    "Lot Area":            "lot_area",
    "Gr Liv Area":         "gr_liv_area",
    "Total Bsmt SF":       "total_bsmt_sf",
    "total_finished_area": "total_finished_area",
    "Bedroom AbvGr":       "bedroom_abvgr",
    "Full Bath":           "full_bath",
    "Half Bath":           "half_bath",
    "total_bathrooms":     "total_bathrooms",
    "Overall Qual":        "overall_qual",
    "Overall Cond":        "overall_cond",
    "Heating QC":          "heating_qc",
    "central_air":         "central_air",
    "Year Built":          "year_built",
    "Year Remod/Add":      "year_remod",
    "house_age":           "house_age",
    "was_remodeled":       "was_remodeled",
    "Fireplaces":          "fireplaces",
    "Garage Cars":         "garage_cars",
    "has_garage":          "has_garage",
    "total_porch_area":    "total_porch_area",
    "SalePrice":           "sale_price",
}

# Seed the database from the processed CSV
def seed_database(limit: int | None = None):
  if not PROCESSED_PATH.exists():
    logger.error(
      f"Processed data not found at {PROCESSED_PATH}. "
      "Please run the data pipeline first."
    )
    sys.exit(1)

  logger.info("Creating database tables...")
  create_all_tables()

  db = SessionLocal()

  try:
    existing = count_listings(db)
    if existing > 0:
      logger.info(f"Database already has {existing} listings. Skipping seed.")
      return
    
    logger.info(f"Reading {PROCESSED_PATH}...")
    df = pd.read_csv(PROCESSED_PATH)

    if limit:
      df = df.head(limit)

    # Keep only columns that are mapped
    available_cols = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df_subset = df[list(available_cols.keys())].rename(columns=available_cols)

    # Convert bool columns
    for bool_col in ["central_air", "was_remodeled", "has_garage"]:
      if bool_col in df_subset.columns:
        df_subset[bool_col] = df_subset[bool_col].astype(bool)

    records = df_subset.to_dict(orient="records")

    logger.info(f"Inserting {len(records):,} listings ...")
    for i, record in enumerate(records):
      # Clean NaN values to None for SQLAlchemy
      clean = {
         k: (None if (isinstance(v, float) and v != v) else v)
         for k, v in record.items()
      }
      create_listing(db, clean)
      
      if (i + 1) % 500 == 0:
          logger.info(f"  Inserted {i + 1:,} / {len(records):,} ...")

    logger.info(f"Seeding complete. Total listings: {count_listings(db)}")

  finally:
    db.close()

if __name__ == "__main__":
  seed_database()
    