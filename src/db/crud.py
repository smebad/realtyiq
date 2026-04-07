import json
import logging
from typing import Optional

from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from src.db.models import ChatLog, Listing, PredictionLog

logger = logging.getLogger(__name__)

# LISTINGS CRUD

# Function to insert a new listing row and return created object
def create_listing(db: Session, listing_data: dict) -> Listing:
  listing = Listing(**listing_data)
  db.add(listing)
  db.commit()
  db.refresh(listing)
  return listing

# Function to fetch a single listing by primary key. Returns None if not found.
def get_listing(db: Session, listing_id: int) -> Optional[Listing]:
  return db.query(Listing).filter(Listing.id == listing_id).first()

# Function to fetch multiple listings with optional filters and pagination.
def get_listings(
    db: Session,
    skip: int = 0,
    limit: int = 20,
    neighborhood: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_bedrooms: Optional[int] = None,
    min_area: Optional[float] = None,
) -> list[Listing]:
    
    query = db.query(Listing).filter(Listing.is_active == True)

    if neighborhood:
        query = query.filter(Listing.neighborhood == neighborhood)
    if min_price:
        query = query.filter(Listing.sale_price >= min_price)
    if max_price:
        query = query.filter(Listing.sale_price <= max_price)
    if min_bedrooms:
        query = query.filter(Listing.bedroom_abvgr >= min_bedrooms)
    if min_area:
        query = query.filter(Listing.gr_liv_area >= min_area)

    return query.order_by(desc(Listing.created_at)).offset(skip).limit(limit).all()

# Function to fetch multiple listings by a list of IDs. Used by RAG context builder.
def get_listings_by_ids(db: Session, ids: list[int]) -> list[Listing]:
    return db.query(Listing).filter(Listing.id.in_(ids)).all()

# Function to update fields on an existing listing.
def update_listing(
    db: Session, listing_id: int, update_data: dict
) -> Optional[Listing]:
    
    listing = get_listing(db, listing_id)
    if not listing:
        return None
    for key, value in update_data.items():
        setattr(listing, key, value)
    db.commit()
    db.refresh(listing)
    return listing

# Function to soft-delete a listing by setting is_active=False. Returns success flag.
def delete_listing(db: Session, listing_id: int) -> bool:

    listing = get_listing(db, listing_id)
    if not listing:
        return False
    listing.is_active = False
    db.commit()
    return True

# Additional utility functions for analytics and dropdown options
def count_listings(db: Session) -> int:
    return db.query(func.count(Listing.id)).scalar()

# Function to get unique neighborhoods for filter dropdowns
def get_neighborhoods(db: Session) -> list[str]:

    results = db.query(Listing.neighborhood).distinct().all()
    return sorted([r[0] for r in results])

# Function to get price stats for dashboard summary cards
def get_price_stats(db: Session) -> dict:

    result = db.query(
        func.min(Listing.sale_price).label("min_price"),
        func.max(Listing.sale_price).label("max_price"),
        func.avg(Listing.sale_price).label("avg_price"),
        func.count(Listing.id).label("total_listings"),
    ).filter(Listing.sale_price.isnot(None)).first()

    return {
        "min_price": round(result.min_price or 0, 2),
        "max_price": round(result.max_price or 0, 2),
        "avg_price": round(result.avg_price or 0, 2),
        "total_listings": result.total_listings or 0,
    }

# PREDICTION LOGS

# Function to log a prediction made by the AI model, including input features and actual price if available.
def log_prediction(
    db: Session,
    input_features: dict,
    predicted_price: float,
    listing_id: int | None = None,
    actual_price: float | None = None,
) -> PredictionLog:
    log = PredictionLog(
        listing_id=listing_id,
        input_features=json.dumps(input_features),
        predicted_price=predicted_price,
        actual_price=actual_price,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

# Function to retrieve recent prediction logs for monitoring and evaluation purposes.
def get_recent_predictions(db: Session, limit: int = 50) -> list[PredictionLog]:
    return (
        db.query(PredictionLog)
        .order_by(desc(PredictionLog.created_at))
        .limit(limit)
        .all()
    )

# CHAT LOGS

# Function to log a conversation turn with the AI assistant, including user query, retrieved listing IDs, and LLM response.
def log_chat(
    db: Session,
    user_query: str,
    retrieved_listing_ids: list[int],
    llm_response: str,
) -> ChatLog:
    chat = ChatLog(
        user_query=user_query,
        retrieved_listing_ids=json.dumps(retrieved_listing_ids),
        llm_response=llm_response,
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat