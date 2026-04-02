import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

# Sql database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./realtyiq.db")

# SQLite specific argument to allow usage of the same connection across multiple threads
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

# Database session
engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
class Base(DeclarativeBase):
    pass

# FastAPI dependency to get a database session
def get_db():
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to create all tables in the database
def create_all_tables():
    
    from src.db import models
    Base.metadata.create_all(bind=engine)