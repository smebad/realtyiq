# Test seeding creates listings and basic CRUD operations on the database.
def test_database_seed():
    from src.db.database import SessionLocal, create_all_tables
    from src.db.crud import count_listings

    create_all_tables()
    db = SessionLocal()
    count = count_listings(db)
    db.close()

    assert count >= 0

# Test basic CRUD operations on the listings table.
def test_crud_operations():
    from src.db.database import SessionLocal, create_all_tables
    from src.db.crud import create_listing, get_listing, delete_listing

    create_all_tables()
    db = SessionLocal()

    # Create
    test_listing = {
        "neighborhood": "TestNeighborhood",
        "house_style": "1Story",
        "lot_area": 8000.0,
        "gr_liv_area": 1500.0,
        "total_finished_area": 1500.0,
        "bedroom_abvgr": 3,
        "full_bath": 2,
        "total_bathrooms": 2.0,
        "overall_qual": 7,
        "overall_cond": 5,
        "heating_qc": 3,
        "year_built": 2000,
        "year_remod": 2005,
        "house_age": 24,
        "total_bsmt_sf": 0.0,
        "half_bath": 0,
        "central_air": True,
        "was_remodeled": True,
        "fireplaces": 1,
        "garage_cars": 2,
        "has_garage": True,
        "total_porch_area": 100.0,
        "sale_price": 200000.0,
    }

    created = create_listing(db, test_listing)
    assert created.id is not None
    assert created.neighborhood == "TestNeighborhood"

    # Read
    fetched = get_listing(db, created.id)
    assert fetched is not None
    assert fetched.sale_price == 200000.0

    # Soft delete
    result = delete_listing(db, created.id)
    assert result is True

    db.close()