import logging
from pathlib import Path

import faiss
import numpy as np
from sqlalchemy.orm import Session

from src.db.database import SessionLocal
from src.db.models import Listing
from src.search.embedder import encode_texts, listing_to_text

logger = logging.getLogger(__name__)

ROOT            = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR  = ROOT / "data" / "embeddings"
INDEX_PATH      = EMBEDDINGS_DIR / "faiss_index.bin"
IDS_PATH        = EMBEDDINGS_DIR / "listing_ids.npy"

# Main function to build the FAISS index from database listings
def build_index(batch_size: int = 64) -> None:

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all active listings from DB
    logger.info("Loading listings from database ...")
    db = SessionLocal()

    try:
        listings = db.query(Listing).all()
        
    finally:
        db.close()

    if not listings:
        raise RuntimeError("No listings in database. Run the seeder first.")

    logger.info(f"Loaded {len(listings):,} listings.")

    # 2. Convert listings to text
    logger.info("Converting listings to text descriptions ...")
    texts = [listing_to_text(l.to_dict()) for l in listings]
    listing_ids = np.array([l.id for l in listings], dtype=np.int64)

    # Peek at first description so we can verify quality
    logger.info(f"Sample text:\n  {texts[0]}")

    # 3. Generate embeddings 
    embeddings = encode_texts(texts, batch_size=batch_size)

    # 4. Build FAISS index
    logger.info("Building FAISS index ...")
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    logger.info(f"Index contains {index.ntotal:,} vectors of dimension {dimension}.")

    # 5. Save index and ID mapping 
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(IDS_PATH), listing_ids)

    logger.info(f"Saved FAISS index to {INDEX_PATH}")
    logger.info(f"Saved listing ID map to {IDS_PATH}")
    logger.info("Index build complete.")


if __name__ == "__main__":
    build_index()