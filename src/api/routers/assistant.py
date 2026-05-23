import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.schemas import ChatRequest, ChatResponse
from src.db.database import get_db
from src.rag.assistant import answer_question

from src.db.crud import log_chat

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

    # Log to database for monitoring
    try:
        log_chat(
            db=db,
            user_query=body.message,
            retrieved_listing_ids=result["retrieved_listing_ids"],
            llm_response=result["answer"],
        )
    except Exception as e:
        logger.warning(f"Failed to log chat: {e}")

    return ChatResponse(
        answer                 = result["answer"],
        retrieved_listing_ids  = result["retrieved_listing_ids"],
        sources_used           = result["sources_used"],
    )