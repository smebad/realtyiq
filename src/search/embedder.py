import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Embedding model details
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module level cache — model loads once
_embedder = None

# Function to load the sentence transformer model (cached after first call)
def get_embedder() -> SentenceTransformer:

    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {MODEL_NAME} ...")
        _embedder = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embedder

# Function to convert a listing's fields into a natural language description for embedding
def listing_to_text(listing: dict) -> str:

    # Pull fields with safe defaults
    neighborhood = listing.get("neighborhood", "unknown area")
    style        = listing.get("house_style", "house")
    bedrooms     = listing.get("bedroom_abvgr", 0)
    bathrooms    = listing.get("total_bathrooms", 1)
    area         = listing.get("gr_liv_area", 0)
    quality      = listing.get("overall_qual", 5)
    year_built   = listing.get("year_built", 2000)
    house_age    = listing.get("house_age", 20)
    fireplaces   = listing.get("fireplaces", 0)
    garage_cars  = listing.get("garage_cars", 0)
    central_air  = listing.get("central_air", True)
    price        = listing.get("sale_price") or listing.get("predicted_price", 0)
    bsmt_sf      = listing.get("total_bsmt_sf", 0)

    # Quality label mapping
    quality_labels = {
        1: "very poor", 2: "poor", 3: "fair", 4: "below average",
        5: "average", 6: "above average", 7: "good",
        8: "very good", 9: "excellent", 10: "outstanding"
    }
    
    try:
        quality_label = quality_labels.get(int(float(quality)), "average")
    except (ValueError, TypeError):
        quality_label = "average"

    # Build natural language text
    parts = [
        f"{bedrooms} bedroom {style.lower()} in {neighborhood}.",
        f"Property has {bathrooms:.1f} bathrooms and {area:.0f} square feet of living space.",
    ]

    if bsmt_sf > 0:
        parts.append(f"Includes {bsmt_sf:.0f} square feet of basement.")

    parts.append(f"Overall quality is {quality_label}.")
    parts.append(
        f"Built in {year_built}, "
        f"{'recently built' if house_age < 10 else f'approximately {house_age} years old'}."
    )

    if fireplaces > 0:
        parts.append(f"Has {fireplaces} fireplace{'s' if fireplaces > 1 else ''}.")

    if garage_cars > 0:
        parts.append(f"Garage fits {garage_cars} car{'s' if garage_cars > 1 else ''}.")
    else:
        parts.append("No garage.")

    parts.append("Has central air conditioning." if central_air else "No central air conditioning.")

    if price and price > 0:
        parts.append(f"Priced at ${price:,.0f}.")

    return " ".join(parts)

# Function to encode a list of text strings into embedding vectors using the model
def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:

    model = get_embedder()

    logger.info(f"Encoding {len(texts):,} texts in batches of {batch_size} ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize — enables cosine similarity via dot product
        convert_to_numpy=True,
    )
    logger.info(f"Encoding complete. Shape: {embeddings.shape}")
    return embeddings

# Function to encode a single search query into an embedding vector
def encode_query(query: str) -> np.ndarray:

    model = get_embedder()
    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector