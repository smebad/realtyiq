import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.schemas import ChatRequest, ChatResponse
from src.db.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assistant", tags=["AI Assistant"])

# Function to handle chat queries to the AI assistant about properties
@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, db: Session = Depends(get_db)):
    
    logger.info(f"Chat query: {body.message!r}")

    return ChatResponse(
        answer="AI assistant will be fully wired in Phase 7. "
               "It will retrieve relevant listings and answer using a local LLM.",
        retrieved_listing_ids=[],
        sources_used=0,
    )