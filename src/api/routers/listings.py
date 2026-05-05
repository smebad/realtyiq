import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.schemas import (
    ListingCreate,
    ListingListResponse,
    ListingResponse,
    ListingUpdate,
    StatsResponse,
)
from src.db.crud import (
    count_listings,
    create_listing,
    delete_listing,
    get_listing,
    get_listings,
    get_neighborhoods,
    get_price_stats,
    update_listing,
)
from src.db.database import get_db
from src.ml.predict import predict_price

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/listings", tags=["Listings"])

# Compute fields that are derived from user inputs, such as total bathrooms, house age, etc.
def _compute_derived_fields(data: dict) -> dict:
    
    import datetime
    current_year = datetime.datetime.now().year

    data["house_age"]           = current_year - data.get("year_built", current_year)
    data["was_remodeled"]       = False
    data["year_remod"]          = data.get("year_built", current_year)
    data["total_bathrooms"]     = data.get("full_bath", 0) + data.get("half_bath", 0) * 0.5
    data["has_garage"]          = data.get("garage_cars", 0) > 0
    data["total_finished_area"] = data.get("gr_liv_area", 0) + data.get("total_bsmt_sf", 0)
    data["total_porch_area"]    = 0.0
    data["ms_zoning"]           = "RL"
    data["heating_qc"]          = "3"   # encoded TA = typical/average

    return data

# Aggregate statistics for the dashboard, including price distribution and neighborhood breakdown
@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    
    stats         = get_price_stats(db)
    neighborhoods = get_neighborhoods(db)
    return StatsResponse(neighborhoods=neighborhoods, **stats)

# List listings with optional filters
@router.get("", response_model=ListingListResponse)
def list_listings(
    neighborhood:  Optional[str]   = Query(None),
    min_price:     Optional[float] = Query(None, ge=0),
    max_price:     Optional[float] = Query(None, ge=0),
    min_bedrooms:  Optional[int]   = Query(None, ge=0),
    min_area:      Optional[float] = Query(None, ge=0),
    page:          int             = Query(1, ge=1),
    per_page:      int             = Query(20, ge=1, le=100),
    db:            Session         = Depends(get_db),
):
    skip     = (page - 1) * per_page
    listings = get_listings(
        db,
        skip=skip,
        limit=per_page,
        neighborhood=neighborhood,
        min_price=min_price,
        max_price=max_price,
        min_bedrooms=min_bedrooms,
        min_area=min_area,
    )
    total = count_listings(db)

    return ListingListResponse(
        total=total,
        page=page,
        per_page=per_page,
        listings=[ListingResponse.model_validate(l) for l in listings],
    )

# Fetch a single listing by ID
@router.get("/{listing_id}", response_model=ListingResponse)
def get_single_listing(listing_id: int, db: Session = Depends(get_db)):

    listing = get_listing(db, listing_id)
    if not listing:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")
    return ListingResponse.model_validate(listing)

# Create a new property listing. Automatically computes derived fields and runs price prediction if price not provided.
@router.post("", response_model=ListingResponse, status_code=201)
def create_new_listing(body: ListingCreate, db: Session = Depends(get_db)):

    data = body.model_dump()
    data = _compute_derived_fields(data)

    # Auto predict price if not provided
    if not data.get("sale_price"):
        try:
            prediction    = predict_price(data)
            data["predicted_price"] = prediction["predicted_price"]
        except Exception as e:
            logger.warning(f"Price prediction failed for new listing: {e}")

    listing = create_listing(db, data)
    return ListingResponse.model_validate(listing)

# Partially update an existing listing.
@router.patch("/{listing_id}", response_model=ListingResponse)
def update_existing_listing(
    listing_id: int,
    body: ListingUpdate,
    db: Session = Depends(get_db),
):

    update_data = body.model_dump(exclude_none=True)
    listing     = update_listing(db, listing_id, update_data)
    if not listing:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")
    return ListingResponse.model_validate(listing)

# Soft-delete an existing listing
@router.delete("/{listing_id}", status_code=204)
def delete_existing_listing(listing_id: int, db: Session = Depends(get_db)):
    
    success = delete_listing(db, listing_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")