import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictRequest, PredictResponse
from src.ml.predict import get_model_metadata, predict_price

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Helper function to map API request fields to model feature names and compute engineered features
def _build_feature_dict(body: PredictRequest) -> dict:

    import datetime
    current_year = datetime.datetime.now().year
    house_age    = current_year - body.year_built

    return {
        # Raw features
        "Gr Liv Area":          body.gr_liv_area,
        "Overall Qual":         body.overall_qual,
        "Year Built":           body.year_built,
        "Total Bsmt SF":        body.total_bsmt_sf,
        "Garage Cars":          body.garage_cars,
        "Full Bath":            body.full_bath,
        "Half Bath":            body.half_bath,
        "Bedroom AbvGr":        body.bedroom_abvgr,
        "Fireplaces":           body.fireplaces,
        "Lot Area":             body.lot_area,
        "central_air":          int(body.central_air),

        # Engineered features
        "house_age":            house_age,
        "years_since_remodel":  house_age,
        "was_remodeled":        0,
        "total_bathrooms":      body.full_bath + body.half_bath * 0.5,
        "total_finished_area":  body.gr_liv_area + body.total_bsmt_sf,
        "has_garage":           int(body.garage_cars > 0),
        "total_porch_area":     0.0,
        "overall_qual_squared": body.overall_qual ** 2,
        "qual_x_area":          body.overall_qual * body.gr_liv_area,

        # Defaults for features not in the public API
        "total_porch_area":     0.0,
        "heating_qc_score":     3,
        "bsmt_qual_score":      3,
        "garage_qual_score":    3,
        "fireplace_qu_score":   0,
    }

# Function to predict the sale price of a property based on its features, with error handling for model loading and prediction issues
@router.post("", response_model=PredictResponse)
def predict_property_price(body: PredictRequest):

    try:
        features = _build_feature_dict(body)
        result   = predict_price(features)

        return PredictResponse(
            predicted_price  = result["predicted_price"],
            confidence_range = result["confidence_range"],
            model_version    = result["model_version"],
            r2_score         = result["r2_score"],
            input_summary    = {
                "gr_liv_area":  body.gr_liv_area,
                "overall_qual": body.overall_qual,
                "year_built":   body.year_built,
                "bedrooms":     body.bedroom_abvgr,
                "bathrooms":    body.full_bath + body.half_bath * 0.5,
            },
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run: python -m src.ml.train",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Route to get model metadata
@router.get("/model-info")
def get_model_info():

    try:
        return get_model_metadata()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run training first.",
        )