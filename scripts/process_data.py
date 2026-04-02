import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.data.features import engineer_features

df = load_raw_data("data/raw/AmesHousing.csv")
df = clean_data(df)
df = engineer_features(df)

Path("data/processed").mkdir(exist_ok=True)
df.to_csv("data/processed/ames_featured.csv", index=False)
print(f"Saved {len(df)} rows to data/processed/ames_featured.csv")
print(f"Columns: {list(df.columns)}")