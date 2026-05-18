from src.search.retriever import search as vector_search, get_index_stats

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.schemas import (
    ListingListResponse,
    ListingResponse,
    SemanticSearchRequest,
)
from src.db.crud import count_listings, get_listings
from src.db.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])

# Filter-based property search endpoint 
@router.get("", response_model=ListingListResponse)
def filtered_search(
    neighborhood:  Optional[str]   = Query(None),
    min_price:     Optional[float] = Query(None),
    max_price:     Optional[float] = Query(None),
    min_bedrooms:  Optional[int]   = Query(None),
    min_area:      Optional[float] = Query(None),
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

# Semantic search endpoint for natural language queries over listings
@router.post("/semantic")
def semantic_search(
    body: SemanticSearchRequest,
    db: Session = Depends(get_db),
):
    try:
        results = vector_search(query=body.query, db=db, top_k=body.top_k)
        return {
            "query":        body.query,
            "total_results": len(results),
            "results":      results,
        }
    except FileNotFoundError:
        return {
            "query":   body.query,
            "results": [],
            "message": "Search index not built yet. Run: python -m src.search.indexer",
        }

# Return FAISS index statistics
@router.get("/index-stats")
def index_stats():
    try:
        return get_index_stats()
    except FileNotFoundError:
        return {"message": "Index not built yet. Run: python -m src.search.indexer"}