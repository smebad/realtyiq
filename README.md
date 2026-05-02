# 🏠 RealtyIQ — Marketplace Intelligence Platform

> An end-to-end AI platform for property price prediction, semantic search, and conversational listing intelligence.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 🚧 Project Status
This project is being built incrementally and documented phase by phase.

## 🎯 What This Project Does
- **Price Prediction** — XGBoost regression model with SHAP explainability
- **Semantic Search** — Embedding-based property search using Sentence Transformers + FAISS
- **AI Assistant** — RAG-powered chatbot grounded in real listing data (Hugging Face LLM)
- **REST API** — FastAPI backend with full CRUD and inference endpoints
- **Dashboard** — Streamlit UI for predictions, search, and chat

## 🛠️ Tech Stack
| Layer | Tools |
|---|---|
| ML | XGBoost, Scikit-learn, SHAP |
| Search | Sentence Transformers, FAISS |
| LLM / RAG | Hugging Face Transformers |
| Backend | FastAPI, SQLAlchemy, SQLite |
| Frontend | Streamlit, Plotly |
| DevOps | Docker, GitHub Actions |

## 📁 Project Structure
*(Updated as each phase is completed)*
- `src/data/loader.py` — Raw data loading and schema validation
- `src/data/cleaner.py` — Null handling, type fixes, outlier removal
- `src/data/features.py` — Feature engineering (age, area, quality, interactions)
- `src/db/models.py` — SQLAlchemy ORM: Listing, PredictionLog, ChatLog tables
- `src/db/crud.py` — All database operations (Create, Read, Update, Delete)
- `src/db/seed.py` — Seeds 2,925 real property listings into SQLite
- `notebooks/01_eda.ipynb` — Exploratory data analysis with Plotly charts

## 🚀 How to Run
*(Coming soon — updated after deployment phase)*

## 📊 ML Metrics

| Metric | Value |
|--------|-------|
| R² Score | 0.94 |
| CV R² (5-fold) | 0.906 ± 0.008 |
| RMSE | ~$20,061 |
| MAE | ~$12,610 |
| MAPE | 6.94% |

> 📓 See full training walkthrough with SHAP plots:
> [`notebooks/03_model_training.ipynb`](notebooks/03_model_training.ipynb)