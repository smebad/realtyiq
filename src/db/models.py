from datetime import datetime

from sqlalchemy import (
  Boolean,
  DateTime,
  Float,
  Integer,
  String,
  Text,
  func,
)

from sqlalchemy.orm import Mapped, mapped_column
from src.db.database import Base

# Core property listing table
class Listings(Base):
  __tablename__ = "listings"

  # Primary key
  id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

  # Location
  neighborhood: Mapped[str] = mapped_column(String(100), index=True)
  house_style: Mapped[str] = mapped_column(String(50))
  ms_zoning: Mapped[str | None] = mapped_column(String(50), nullable=True)

  # Size
  lot_area: Mapped[float] = mapped_column(Float)
  gr_liv_area: Mapped[float] = mapped_column(Float) # above ground sqft
  total_bsmt_sf: Mapped[float] = mapped_column(Float, default=0)
  total_finished_area: Mapped[float] = mapped_column(Float)

# Rooms
  bedroom_abvgr: Mapped[int] = mapped_column(Integer)
  full_bath: Mapped[int] = mapped_column(Integer)
  half_bath: Mapped[int] = mapped_column(Integer, default=0)
  total_bathrooms: Mapped[float] = mapped_column(Float)

# Quality and Condition
  overall_qual: Mapped[int] = mapped_column(Integer)      # 1–10 scale
  overall_cond: Mapped[int] = mapped_column(Integer)
  heating_qc: Mapped[str] = mapped_column(String(10))
  central_air: Mapped[bool] = mapped_column(Boolean, default=True)

# Age
  year_built: Mapped[int] = mapped_column(Integer)
  year_remod: Mapped[int] = mapped_column(Integer)
  house_age: Mapped[int] = mapped_column(Integer)
  was_remodeled: Mapped[bool] = mapped_column(Boolean, default=False)

# Extras
  fireplaces: Mapped[int] = mapped_column(Integer, default=0)
  garage_cars: Mapped[int] = mapped_column(Integer, default=0)
  has_garage: Mapped[bool] = mapped_column(Boolean, default=False)
  total_porch_area: Mapped[float] = mapped_column(Float, default=0)

# Price
  sale_price: Mapped[float | None] = mapped_column(Float, nullable=True)
  predicted_price: Mapped[float | None] = mapped_column(Float, nullable=True)

# AI generated content
  description: Mapped[str | None] = mapped_column(Text, nullable=True)

# Search
# Stores the FAISS vector index ID for fast retrieval
embedding_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

# Metadata
is_active: Mapped[bool] = mapped_column(Boolean, default=True)
created_at: Mapped[datetime] = mapped_column(
    DateTime, server_default=func.now()
)
updated_at: Mapped[datetime] = mapped_column(
    DateTime, server_default=func.now(), onupdate=func.now()
)

# Utility methods
def __repr__(self) -> str:
    return (
        f"<Listing id={self.id} "
        f"neighborhood={self.neighborhood!r} "
        f"price={self.sale_price}>"
    )

# Method to convert listing to dictionary for API responses
def to_dict(self) -> dict:
    return {
        "id": self.id,
        "neighborhood": self.neighborhood,
        "house_style": self.house_style,
        "gr_liv_area": self.gr_liv_area,
        "bedroom_abvgr": self.bedroom_abvgr,
        "total_bathrooms": self.total_bathrooms,
        "overall_qual": self.overall_qual,
        "year_built": self.year_built,
        "house_age": self.house_age,
        "fireplaces": self.fireplaces,
        "has_garage": self.has_garage,
        "central_air": self.central_air,
        "sale_price": self.sale_price,
        "predicted_price": self.predicted_price,
        "description": self.description,
    }

# Additional tables for logging and monitoring
class PredictionLog(Base):

  __tablename__ = "prediction_logs"

  id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
  listing_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
  input_features: Mapped[str] = mapped_column(Text)   # JSON string
  predicted_price: Mapped[float] = mapped_column(Float)
  actual_price: Mapped[float | None] = mapped_column(Float, nullable=True)
  model_version: Mapped[str] = mapped_column(String(50), default="v1.0")
  created_at: Mapped[datetime] = mapped_column(
      DateTime, server_default=func.now()
  )

# Logs user interactions with the AI assistant
class ChatLog(Base):

  __tablename__ = "chat_logs"

  id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
  user_query: Mapped[str] = mapped_column(Text)
  retrieved_listing_ids: Mapped[str] = mapped_column(Text)  # JSON list
  llm_response: Mapped[str] = mapped_column(Text)
  created_at: Mapped[datetime] = mapped_column(
      DateTime, server_default=func.now()
  )