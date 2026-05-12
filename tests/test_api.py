import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.db.database import create_all_tables

client = TestClient(app)

# Create all tables before each test
def setup_module():
    create_all_tables()

# Test cases for API endpoints
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "model_loaded" in data
    assert "db_connected" in data

# Test root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()

# Test listing retrieval
def test_list_listings():
    response = client.get("/listings?per_page=5")
    assert response.status_code == 200
    data = response.json()
    assert "listings" in data
    assert "total" in data

# Test single listing retrieval (may return 404 if ID doesn't exist)
def test_get_single_listing():
    response = client.get("/listings/1")
    assert response.status_code in [200, 404]

# Test prediction endpoint with valid input
def test_predict_endpoint():
    payload = {
        "gr_liv_area":   1500,
        "overall_qual":  7,
        "year_built":    2000,
        "total_bsmt_sf": 800,
        "garage_cars":   2,
        "full_bath":     2,
        "half_bath":     0,
        "bedroom_abvgr": 3,
        "fireplaces":    1,
        "lot_area":      8000,
        "central_air":   True,
        "neighborhood":  "CollgCr",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert 50_000 < data["predicted_price"] < 1_000_000

# Test prediction endpoint with invalid input
def test_predict_invalid_input():
    """Negative area should be rejected."""
    response = client.post("/predict", json={"gr_liv_area": -100, "overall_qual": 5,
                                              "year_built": 2000})
    assert response.status_code == 422   # Unprocessable Entity

# Test search endpoint with filters
def test_filtered_search():
    response = client.get("/search?min_bedrooms=3&max_price=300000")
    assert response.status_code == 200

# Test listing stats endpoint
def test_listing_stats():
    response = client.get("/listings/stats")
    assert response.status_code == 200
    data = response.json()
    assert "avg_price" in data
    assert "neighborhoods" in data

# Test chat endpoint (AI assistant)
def test_chat_endpoint():
    response = client.post("/assistant/chat",
                           json={"message": "What are the cheapest neighborhoods?"})
    assert response.status_code == 200
    assert "answer" in response.json()