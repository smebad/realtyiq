import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.rag.context_builder import (
    build_flan_prompt,
    build_prompt,
    format_listings_as_context,
)
from src.search.retriever import search as vector_search

load_dotenv()

logger = logging.getLogger(__name__)

HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_URL  = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Local model cache
_local_model     = None
_local_tokenizer = None

# Main function to answer user questions using RAG pipeline
def answer_question(
    question: str,
    db: Session,
    top_k: int = 5,
    use_local: bool = False,
) -> dict:
    
    # 1. Retrieve relevant listings from vector search
    logger.info(f"Retrieving top {top_k} listings for: {question!r}")
    try:
        listings = vector_search(query=question, db=db, top_k=top_k)
    except FileNotFoundError:
        return {
            "answer": (
                "The search index is not built yet. "
                "Please run: python -m src.search.indexer"
            ),
            "retrieved_listing_ids": [],
            "sources_used": 0,
        }

    retrieved_ids = [l["id"] for l in listings]
    logger.info(f"Retrieved listing IDs: {retrieved_ids}")

    # 2. Build context for LLM
    context = format_listings_as_context(listings)

    # 3. Generate answer using either HuggingFace API or local model
    if use_local or not HF_TOKEN:
        logger.info("Using local flan-t5 model ...")
        answer = _answer_with_local_model(question, context)
    else:
        logger.info("Using HuggingFace Inference API ...")
        answer = _answer_with_hf_api(question, context)

    return {
        "answer":                answer,
        "retrieved_listing_ids": retrieved_ids,
        "sources_used":          len(listings),
    }

# Helper function to call HuggingFace Inference API
def _answer_with_hf_api(question: str, context: str) -> str:

    prompt  = build_prompt(question, context)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":  300,
            "temperature":     0.3,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        # HF API returns a list of generated texts
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        elif isinstance(result, dict) and "error" in result:
            logger.warning(f"HF API error: {result['error']}")
            return _answer_with_local_model(question, context)
        else:
            return str(result)

    except requests.exceptions.Timeout:
        logger.warning("HF API timed out — falling back to local model")
        return _answer_with_local_model(question, context)

    except requests.exceptions.RequestException as e:
        logger.warning(f"HF API request failed: {e} — falling back to local model")
        return _answer_with_local_model(question, context)

# Helper function to call local flan-t5-base model via transformers
def _answer_with_local_model(question: str, context: str) -> str:

    global _local_model, _local_tokenizer

    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        if _local_model is None:
            logger.info("Loading flan-t5-base locally (first run downloads ~250MB) ...")
            model_name       = "google/flan-t5-base"
            _local_tokenizer = T5Tokenizer.from_pretrained(model_name)
            _local_model     = T5ForConditionalGeneration.from_pretrained(model_name)
            logger.info("Local model loaded.")

        prompt = build_flan_prompt(question, context)

        # Prompt to fit model's max input length
        inputs = _local_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True,
        )

        answer = _local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    except Exception as e:
        logger.error(f"Local model failed: {e}")
        return (
            "I was unable to generate an answer at this time. "
            f"Here are the top matching listings for your query: "
            f"{', '.join(str(i) for i in [l for l in context[:200]])}..."
        )