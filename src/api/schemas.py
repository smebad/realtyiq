from typing import Optional
from pydantic import BaseModel, Field, field_validator


# LISTINGS
# Shared fields for create/update + separate response model with DB-generated fields
class ListingBase(BaseModel):
    neighborhood:        str   = Field(..., example="CollgCr")
    house_style:         str   = Field(..., example="1Story")
    lot_area:            float = Field(..., gt=0, example=8450)
    gr_liv_area:         float = Field(..., gt=0, example=1710)
    total_bsmt_sf:       float = Field(0, ge=0, example=856)
    bedroom_abvgr:       int   = Field(..., ge=0, le=20, example=3)
    full_bath:           int   = Field(..., ge=0, le=10, example=2)
    half_bath:           int   = Field(0, ge=0, le=10, example=1)
    overall_qual:        int   = Field(..., ge=1, le=10, example=7)
    overall_cond:        int   = Field(..., ge=1, le=10, example=5)
    year_built:          int   = Field(..., ge=1800, le=2030, example=2003)
    fireplaces:          int   = Field(0, ge=0, example=1)
    garage_cars:         int   = Field(0, ge=0, example=2)
    central_air:         bool  = Field(True, example=True)
    sale_price:          Optional[float] = Field(None, gt=0, example=208500)

# Request body for creating a new listing
class ListingCreate(ListingBase):
    pass

# Request body for updating a listing
class ListingUpdate(BaseModel):
    neighborhood:   Optional[str]   = None
    house_style:    Optional[str]   = None
    gr_liv_area:    Optional[float] = None
    bedroom_abvgr:  Optional[int]   = None
    full_bath:      Optional[int]   = None
    overall_qual:   Optional[int]   = None
    sale_price:     Optional[float] = None
    description:    Optional[str]   = None

# Response body for listing endpoints, includes DB-generated fields and optional predicted price/description
class ListingResponse(ListingBase):
    id:               int
    total_bathrooms:  float
    total_finished_area: float
    house_age:        int
    has_garage:       bool
    was_remodeled:    bool
    total_porch_area: float
    predicted_price:  Optional[float] = None
    description:      Optional[str]   = None

    model_config = {"from_attributes": True}  # allows ORM to Pydantic conversion

# Paginated list of listings
class ListingListResponse(BaseModel):
    total:    int
    page:     int
    per_page: int
    listings: list[ListingResponse]

# PREDICTION
# Input features for prediction and response format with confidence range and model info
class PredictRequest(BaseModel):
    gr_liv_area:      float = Field(..., gt=0, example=1500,
                                   description="Above-ground living area in sqft")
    overall_qual:     int   = Field(..., ge=1, le=10, example=7,
                                   description="Overall quality score 1-10")
    year_built:       int   = Field(..., ge=1800, le=2030, example=2000)
    total_bsmt_sf:    float = Field(0, ge=0, example=800)
    garage_cars:      int   = Field(0, ge=0, le=6, example=2)
    full_bath:        int   = Field(1, ge=0, le=6, example=2)
    half_bath:        int   = Field(0, ge=0, le=4, example=0)
    bedroom_abvgr:    int   = Field(3, ge=0, le=15, example=3)
    fireplaces:       int   = Field(0, ge=0, le=5, example=1)
    lot_area:         float = Field(8000, gt=0, example=8000)
    central_air:      bool  = Field(True, example=True)
    neighborhood:     str   = Field("CollgCr", example="CollgCr")

    @field_validator("gr_liv_area", "lot_area")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Area must be positive")
        return v

# Price prediction result with confidence range and model info for transparency and trustworthiness
class PredictResponse(BaseModel):
    predicted_price:  float
    confidence_range: dict        # {"low": float, "high": float}
    model_version:    str
    r2_score:         float
    input_summary:    dict        # echo back key inputs for transparency

# SEARCH
# Filters for property search and natural language semantic search query with pagination and top-k results
class SearchRequest(BaseModel):
    neighborhood:   Optional[str]   = None
    min_price:      Optional[float] = Field(None, ge=0)
    max_price:      Optional[float] = Field(None, ge=0)
    min_bedrooms:   Optional[int]   = Field(None, ge=0)
    min_area:       Optional[float] = Field(None, ge=0)
    page:           int             = Field(1, ge=1)
    per_page:       int             = Field(20, ge=1, le=100)

# Natural language semantic search query with top-k results for relevance ranking
class SemanticSearchRequest(BaseModel):
    query:   str = Field(..., min_length=3, example="3 bedroom house with garage near good schools")
    top_k:   int = Field(5, ge=1, le=20)

# AI ASSISTANT
# User message to the AI assistant and response format with source listing IDs and count for transparency
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2,
                        example="Which neighborhoods have the best price per sqft?")

# AI assistant response with source listings and count for transparency and trustworthiness
class ChatResponse(BaseModel):
    answer:              str
    retrieved_listing_ids: list[int]
    sources_used:        int

# GENERAL
# Health check response with model and DB status for monitoring and reliability
class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    db_connected:  bool
    total_listings: int

# Statistics response with aggregated data for insights and market trends
class StatsResponse(BaseModel):
    total_listings: int
    avg_price:      float
    min_price:      float
    max_price:      float
    neighborhoods:  list[str]