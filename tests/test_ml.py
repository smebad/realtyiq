from pathlib import Path
import pytest

MODEL_EXISTS = Path("models/xgboost_price_model.pkl").exists()


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
def test_predict_returns_valid_price():
    from src.ml.predict import predict_price

    result = predict_price({
        "Gr Liv Area":         1500,
        "Overall Qual":        7,
        "house_age":           30,
        "total_bathrooms":     2.0,
        "Bedroom AbvGr":       3,
        "Garage Cars":         2,
        "has_garage":          1,
        "total_finished_area": 1500,
        "qual_x_area":         10500,
        "overall_qual_squared": 49,
        "Fireplaces":          1,
        "central_air":         1,
    })

    assert "predicted_price" in result
    assert 50_000 < result["predicted_price"] < 1_000_000, \
        f"Prediction out of range: {result['predicted_price']}"
    assert result["confidence_range"]["low"] < result["predicted_price"]
    assert result["confidence_range"]["high"] > result["predicted_price"]


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
def test_model_metadata_has_required_keys():
    from src.ml.predict import get_model_metadata

    meta = get_model_metadata()
    for key in ["model_version", "feature_cols", "metrics"]:
        assert key in meta

    assert meta["metrics"]["r2"] > 0.80, \
        f"R² too low: {meta['metrics']['r2']} — retrain the model"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
def test_batch_predict():
    from src.ml.predict import batch_predict

    sample = [
        {"Gr Liv Area": 1200, "Overall Qual": 6, "house_age": 40,
         "total_bathrooms": 1.5, "Bedroom AbvGr": 2, "Garage Cars": 1,
         "has_garage": 1, "total_finished_area": 1200, "qual_x_area": 7200,
         "overall_qual_squared": 36, "Fireplaces": 0, "central_air": 1},
        {"Gr Liv Area": 2500, "Overall Qual": 9, "house_age": 5,
         "total_bathrooms": 3.5, "Bedroom AbvGr": 4, "Garage Cars": 3,
         "has_garage": 1, "total_finished_area": 3200, "qual_x_area": 22500,
         "overall_qual_squared": 81, "Fireplaces": 2, "central_air": 1},
    ]

    results = batch_predict(sample)
    assert len(results) == 2
    # Bigger, higher quality house should cost more
    assert results[1]["predicted_price"] > results[0]["predicted_price"]