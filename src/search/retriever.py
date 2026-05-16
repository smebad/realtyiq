import logging
from pathlib import Path

import faiss
import numpy as np
from sqlalchemy.orm import Session

from src.search.embedder import encode_query

logger = logging.getLogger(__name__)

ROOT           = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
INDEX_PATH     = EMBEDDINGS_DIR / "faiss_index.bin"
IDS_PATH       = EMBEDDINGS_DIR / "listing_ids.npy"

# Module level cache
_index       = None
_listing_ids = None

# Load the FAISS index and listing ID map into memory (cached after first call)
def _load_index() -> None:

    global _index, _listing_ids

    if _index is not None:
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Run: python -m src.search.indexer"
        )

    logger.info("Loading FAISS index ...")
    _index       = faiss.read_index(str(INDEX_PATH))
    _listing_ids = np.load(str(IDS_PATH))
    logger.info(f"Index loaded. {_index.ntotal:,} vectors.")

# Function to perform a search for listings matching a natural language query
def search(
    query: str,
    db: Session,
    top_k: int = 5,
) -> list[dict]:

    _load_index()

    # 1. Encode the query
    query_vector = encode_query(query).astype(np.float32)

    # 2. Search FAISS
    # Returns scores and positions in the index
    scores, positions = _index.search(query_vector, top_k)

    scores    = scores[0]     # shape (top_k,)
    positions = positions[0]  # shape (top_k,)

    # 3. Map positions -> listing DB ids
    result_ids = [
        int(_listing_ids[pos])
        for pos in positions
        if pos != -1   # FAISS returns -1 for empty slots
    ]

    # 4. Fetch listings from DB
    from src.db.crud import get_listings_by_ids
    listings = get_listings_by_ids(db, result_ids)

    # Re-order to match FAISS score order
    listing_map = {l.id: l for l in listings}
    ordered     = [listing_map[rid] for rid in result_ids if rid in listing_map]

    # 5. Attach similarity scores and convert to dicts
    results = []
    for listing, score in zip(ordered, scores):
        d              = listing.to_dict()
        d["similarity_score"] = round(float(score), 4)
        results.append(d)

    logger.info(f"Query: {query!r} → {len(results)} results")
    return results

# Utility function to get stats about the loaded index (for debugging)
def get_index_stats() -> dict:

    _load_index()
    return {
        "total_vectors": int(_index.ntotal),
        "dimensions":    int(_index.d),
        "index_path":    str(INDEX_PATH),
    }