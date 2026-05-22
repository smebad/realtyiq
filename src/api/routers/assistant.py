import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.schemas import ChatRequest, ChatResponse
from src.db.database import get_db
from src.rag.assistant import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assistant", tags=["AI Assistant"])

# AI assistant powered by RAG: retrieves relevant listings and answers user questions
@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, db: Session = Depends(get_db)):
    
    logger.info(f"Assistant query: {body.message!r}")

    result = answer_question(
        question=body.message,
        db=db,
        top_k=5,
    )

    return ChatResponse(
        answer                 = result["answer"],
        retrieved_listing_ids  = result["retrieved_listing_ids"],
        sources_used           = result["sources_used"],
    )