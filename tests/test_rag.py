import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.db.database import SessionLocal
from src.rag.assistant import answer_question

db = SessionLocal()

questions = [
    "Which neighborhoods have the most affordable homes?",
    "Show me large homes with good quality scores",
    "What is the best value home with a garage and fireplace?",
]

for q in questions:
    print("=" * 60)
    print(f"Q: {q}")

    result = answer_question(q, db, top_k=5)

    print(f"\nA: {result['answer']}")
    print(f"\nSources: {result['retrieved_listing_ids']}")

db.close()