import sys
from pathlib import Path

# Imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Config
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="RealtyIQ",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8fafc; color: #1e293b; }

    /* Force all text visible */
    .stMarkdown, .stText, p, h1, h2, h3, h4, label, div {
        color: #1e293b !important;
    }

    /* Except sidebar which stays white text */
    [data-testid="stSidebar"] * { color: white !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2563eb 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Listing cards */
    .listing-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .listing-card h4 { color: #1e3a5f; margin: 0 0 8px 0; }
    .listing-card .price {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2563eb;
    }
    .listing-card .badge {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px;
    }

    /* Chat bubbles */
    .chat-user {
        background: #2563eb;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bot {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        color: #1e293b;
    }
    .source-pill {
        display: inline-block;
        background: #f0fdf4;
        color: #16a34a;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
    }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: white;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box .price-label { font-size: 1rem; opacity: 0.85; }
    .prediction-box .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 8px 0;
    }
    .prediction-box .range {
        font-size: 0.9rem;
        opacity: 0.75;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 20px;
        padding-bottom: 8px;
        border-bottom: 3px solid #2563eb;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# HELPERS

# GET request to the API
def api_get(endpoint: str, params: dict = None) -> dict | None:

    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Run: uvicorn src.api.main:app --reload --port 8000")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error calling {endpoint}: {type(e).__name__}: {e}")
        return None

# POST request to the API
def api_post(endpoint: str, payload: dict) -> dict | None:

    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

# Format price
def format_price(price: float | None) -> str:
    if not price:
        return "N/A"
    return f"${price:,.0f}"

# Render a listing card
def render_listing_card(listing: dict, show_score: bool = False) -> None:

    price = listing.get("sale_price") or listing.get("predicted_price")
    score = listing.get("similarity_score")

    score_html = ""
    if show_score and score:
        pct = int(score * 100)
        score_html = f'<span class="badge">Match: {pct}%</span>'

    st.markdown(f"""
    <div class="listing-card">
        <h4>🏠 {listing.get('neighborhood', 'Unknown')} — {listing.get('house_style', '')}</h4>
        <div class="price">{format_price(price)}</div>
        <div style="margin: 10px 0; color: #64748b; font-size: 0.9rem;">
            🛏 {listing.get('bedroom_abvgr', 0)} bed &nbsp;|&nbsp;
            🚿 {listing.get('total_bathrooms', 0):.1f} bath &nbsp;|&nbsp;
            📐 {listing.get('gr_liv_area', 0):,.0f} sqft &nbsp;|&nbsp;
            📅 Built {listing.get('year_built', 'N/A')}
        </div>
        <div>
            <span class="badge">⭐ Quality {listing.get('overall_qual', 0)}/10</span>
            {'<span class="badge">🚗 Garage</span>' if listing.get('has_garage') else ''}
            {'<span class="badge">🔥 Fireplace</span>' if listing.get('fireplaces', 0) > 0 else ''}
            {'<span class="badge">❄️ AC</span>' if listing.get('central_air') else ''}
            {score_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# Sidebar Navigation and Footer

with st.sidebar:
    st.markdown("## 🏠 RealtyIQ")
    st.markdown("*AI-Powered Real Estate Intelligence*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Search Listings", "💰 Price Predictor", "🤖 AI Assistant"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API health indicator
    health = api_get("/health")
    if health:
        status_color = "🟢" if health.get("status") == "ok" else "🟡"
        st.markdown(f"{status_color} **API Status:** {health.get('status', 'unknown').upper()}")
        st.markdown(f"🏠 **Listings:** {health.get('total_listings', 0):,}")
        st.markdown(f"🤖 **Model:** {'Loaded' if health.get('model_loaded') else 'Not loaded'}")
    else:
        st.markdown("🔴 **API:** Offline")

    st.markdown("---")
    st.markdown("Built with XGBoost + FAISS + RAG")
    st.markdown("*Portfolio Project*")


# PAGE 1 — DASHBOARD

if page == "📊 Dashboard":
    st.markdown('<div class="section-header">📊 Market Dashboard</div>',
                unsafe_allow_html=True)

    # KPI Cards
    stats = api_get("/listings/stats")

    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Listings", f"{stats['total_listings']:,}")
        with col2:
            st.metric("Average Price", format_price(stats['avg_price']))
        with col3:
            st.metric("Min Price", format_price(stats['min_price']))
        with col4:
            st.metric("Max Price", format_price(stats['max_price']))

    st.markdown("---")

    # Charts Row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 💰 Price Distribution")
        data = api_get("/listings", params={"per_page": 500})
        if data and data.get("listings"):
            df = pd.DataFrame(data["listings"])
            df = df[df["sale_price"].notna()]
            fig = px.histogram(
                df, x="sale_price", nbins=50,
                labels={"sale_price": "Sale Price (USD)"},
                color_discrete_sequence=["#2563eb"],
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                xaxis=dict(tickformat="$,.0f"),
                bargap=0.05,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🏘️ Median Price by Neighborhood")
        if data and data.get("listings"):
            df = pd.DataFrame(data["listings"])
            df = df[df["sale_price"].notna()]
            neighborhood_df = (
                df.groupby("neighborhood")["sale_price"]
                .median()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            fig2 = px.bar(
                neighborhood_df,
                x="sale_price", y="neighborhood",
                orientation="h",
                labels={"sale_price": "Median Price", "neighborhood": ""},
                color="sale_price",
                color_continuous_scale="Blues",
            )
            fig2.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                coloraxis_showscale=False,
                xaxis=dict(tickformat="$,.0f"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Charts Row 2
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.markdown("#### ⭐ Price by Quality Score")
        if data and data.get("listings"):
            df = pd.DataFrame(data["listings"])
            df = df[df["sale_price"].notna()]
            fig3 = px.box(
                df, x="overall_qual", y="sale_price",
                labels={
                    "overall_qual": "Quality Score (1-10)",
                    "sale_price": "Sale Price (USD)"
                },
                color="overall_qual",
                color_discrete_sequence=px.colors.sequential.Blues,
            )
            fig3.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                yaxis=dict(tickformat="$,.0f"),
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col_right2:
        st.markdown("#### 📐 Area vs Price")
        if data and data.get("listings"):
            df = pd.DataFrame(data["listings"])
            df = df[df["sale_price"].notna()]
            fig4 = px.scatter(
                df.sample(min(300, len(df))),
                x="gr_liv_area", y="sale_price",
                color="overall_qual",
                labels={
                    "gr_liv_area": "Living Area (sqft)",
                    "sale_price": "Sale Price (USD)",
                    "overall_qual": "Quality"
                },
                opacity=0.6,
                color_continuous_scale="RdYlGn",
            )
            fig4.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(tickformat="$,.0f"),
            )
            st.plotly_chart(fig4, use_container_width=True)

    # Model Performance
    st.markdown("---")
    st.markdown("#### 🤖 ML Model Performance")
    model_info = api_get("/predict/model-info")
    if model_info and "metrics" in model_info:
        m = model_info["metrics"]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            st.metric("R² Score", f"{m.get('r2', 0):.3f}")
        with mc2:
            st.metric("RMSE", format_price(m.get('rmse', 0)))
        with mc3:
            st.metric("MAE", format_price(m.get('mae', 0)))
        with mc4:
            st.metric("MAPE", f"{m.get('mape', 0):.1f}%")
        with mc5:
            st.metric("CV R²", f"{m.get('cv_r2_mean', 0):.3f}")


# PAGE 2 — SEARCH LISTINGS

elif page == "🔍 Search Listings":
    st.markdown('<div class="section-header">🔍 Search Listings</div>',
                unsafe_allow_html=True)

    # Search Mode Toggle
    search_mode = st.radio(
        "Search mode",
        ["🔧 Filter Search", "🧠 Semantic Search"],
        horizontal=True,
    )

    if search_mode == "🔧 Filter Search":
        st.markdown("##### Filter by property features")

        # Get neighborhoods for dropdown
        stats = api_get("/listings/stats")
        neighborhoods = ["All"] + (stats.get("neighborhoods", []) if stats else [])

        col1, col2, col3 = st.columns(3)
        with col1:
            neighborhood = st.selectbox("Neighborhood", neighborhoods)
            min_bedrooms = st.number_input("Min Bedrooms", 0, 10, 0)
        with col2:
            min_price = st.number_input("Min Price ($)", 0, 1_000_000, 0, step=10_000)
            max_price = st.number_input("Max Price ($)", 0, 1_000_000, 500_000, step=10_000)
        with col3:
            min_area = st.number_input("Min Area (sqft)", 0, 5000, 0, step=100)
            per_page = st.slider("Results per page", 5, 50, 10)

        if st.button("🔍 Search", type="primary", use_container_width=True):
            params = {"per_page": per_page}
            if neighborhood != "All":
                params["neighborhood"] = neighborhood
            if min_price > 0:
                params["min_price"] = min_price
            if max_price > 0:
                params["max_price"] = max_price
            if min_bedrooms > 0:
                params["min_bedrooms"] = min_bedrooms
            if min_area > 0:
                params["min_area"] = min_area

            with st.spinner("Searching..."):
                results = api_get("/search", params=params)

            if results:
                st.markdown(f"**{results['total']:,} listings found** "
                            f"(showing {len(results['listings'])})")
                for listing in results["listings"]:
                    render_listing_card(listing)

    else:  # Semantic Search
        st.markdown("##### Describe what you are looking for in plain English")

        query = st.text_input(
            "Search query",
            placeholder="e.g. spacious 4 bedroom with garage and fireplace in a good neighborhood",
            label_visibility="collapsed",
        )
        top_k = st.slider("Number of results", 3, 15, 5)

        if st.button("🧠 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Finding similar listings..."):
                    results = api_post(
                        "/search/semantic",
                        {"query": query, "top_k": top_k}
                    )

                if results and results.get("results"):
                    st.markdown(
                        f"**{len(results['results'])} matches** for: *\"{query}\"*"
                    )
                    for listing in results["results"]:
                        render_listing_card(listing, show_score=True)
                elif results:
                    st.info("No results found. Try a different query.")
            else:
                st.warning("Please enter a search query.")


# PAGE 3 — PRICE PREDICTOR

elif page == "💰 Price Predictor":
    st.markdown('<div class="section-header">💰 AI Price Predictor</div>',
                unsafe_allow_html=True)
    st.markdown("Enter property details to get an instant ML-powered price estimate.")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("##### Property Details")

        with st.form("predict_form"):
            gr_liv_area  = st.number_input("Living Area (sqft)", 500, 5000, 1500, 50)
            overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
            year_built   = st.number_input("Year Built", 1900, 2024, 2000)

            col_a, col_b = st.columns(2)
            with col_a:
                full_bath    = st.number_input("Full Bathrooms", 0, 6, 2)
                bedroom      = st.number_input("Bedrooms", 0, 10, 3)
                garage_cars  = st.number_input("Garage Capacity (cars)", 0, 4, 2)
            with col_b:
                half_bath    = st.number_input("Half Bathrooms", 0, 4, 0)
                fireplaces   = st.number_input("Fireplaces", 0, 4, 0)
                total_bsmt   = st.number_input("Basement Area (sqft)", 0, 3000, 0, 50)

            lot_area     = st.number_input("Lot Area (sqft)", 1000, 50000, 8000, 500)
            central_air  = st.checkbox("Central Air Conditioning", value=True)
            neighborhood = st.text_input("Neighborhood", "CollgCr")

            submitted = st.form_submit_button(
                "🔮 Predict Price", type="primary", use_container_width=True
            )

    with col_result:
        st.markdown("##### Prediction Result")

        if submitted:
            payload = {
                "gr_liv_area":   gr_liv_area,
                "overall_qual":  overall_qual,
                "year_built":    year_built,
                "total_bsmt_sf": total_bsmt,
                "garage_cars":   garage_cars,
                "full_bath":     full_bath,
                "half_bath":     half_bath,
                "bedroom_abvgr": bedroom,
                "fireplaces":    fireplaces,
                "lot_area":      lot_area,
                "central_air":   central_air,
                "neighborhood":  neighborhood,
            }

            with st.spinner("Running ML model..."):
                result = api_post("/predict", payload)

            if result:
                low  = result["confidence_range"]["low"]
                high = result["confidence_range"]["high"]

                st.markdown(f"""
                <div class="prediction-box">
                    <div class="price-label">Estimated Market Value</div>
                    <div class="price-value">{format_price(result['predicted_price'])}</div>
                    <div class="range">Confidence range: {format_price(low)} — {format_price(high)}</div>
                </div>
                """, unsafe_allow_html=True)

                # Input summary
                st.markdown("##### What you entered")
                summary = result.get("input_summary", {})
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.metric("Living Area", f"{summary.get('gr_liv_area', gr_liv_area):,.0f} sqft")
                    st.metric("Quality Score", f"{summary.get('overall_qual', overall_qual)}/10")
                with s_col2:
                    st.metric("Bedrooms", summary.get('bedrooms', bedroom))
                    st.metric("Bathrooms", summary.get('bathrooms', full_bath))

                # Model info
                st.markdown(
                    f"*Model: {result.get('model_version', 'v1.0')} | "
                    f"R² = {result.get('r2_score', 0):.3f}*"
                )

        else:
            st.markdown("""
            <div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
                <div style="font-size: 3rem;">🏠</div>
                <div style="font-size: 1rem; margin-top: 12px;">
                    Fill in the property details and click<br>
                    <strong>Predict Price</strong> to get an estimate.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Show SHAP plot
        shap_path = ROOT / "models" / "plots" / "shap_summary.png"
        if shap_path.exists():
            st.markdown("---")
            st.markdown("##### What drives price predictions?")
            st.image(str(shap_path), caption="SHAP Feature Importance", use_column_width=True)


# PAGE 4 — AI ASSISTANT

elif page == "🤖 AI Assistant":
    st.markdown('<div class="section-header">🤖 RealtyIQ AI Assistant</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Ask any question about the properties. "
        "The assistant retrieves relevant listings and answers from real data."
    )

    # Example questions
    st.markdown("##### Try asking:")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    example_questions = [
        "Which neighborhoods have the most affordable homes?",
        "Find large homes with high quality scores",
        "What homes have garages and fireplaces?",
    ]

    for col, question in zip([ex_col1, ex_col2, ex_col3], example_questions):
        with col:
            if st.button(f"💬 {question}", use_container_width=True):
                st.session_state.setdefault("messages", [])
                st.session_state["messages"].append({
                    "role": "user", "content": question
                })
                with st.spinner("Thinking..."):
                    response = api_post("/assistant/chat", {"message": question})
                if response:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response.get("answer", ""),
                        "sources": response.get("retrieved_listing_ids", []),
                    })

    st.markdown("---")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                sources = msg.get("sources", [])
                source_pills = "".join(
                    f'<span class="source-pill">Listing #{s}</span>'
                    for s in sources[:5]
                )
                st.markdown(
                    f'<div class="chat-bot">🤖 {msg["content"]}'
                    f'<div style="margin-top:8px">{source_pills}</div></div>',
                    unsafe_allow_html=True,
                )

    # Input
    st.markdown("---")
    input_col, btn_col = st.columns([5, 1])

    with input_col:
        user_input = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the best value home under $200,000?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with btn_col:
        send = st.button("Send →", type="primary", use_container_width=True)

    if send and user_input.strip():
        st.session_state["messages"].append({
            "role": "user", "content": user_input
        })

        with st.spinner("Searching listings and generating answer..."):
            response = api_post("/assistant/chat", {"message": user_input})

        if response:
            st.session_state["messages"].append({
                "role":    "assistant",
                "content": response.get("answer", "No answer generated."),
                "sources": response.get("retrieved_listing_ids", []),
            })
        st.rerun()

    if st.session_state.get("messages") and st.button("🗑️ Clear chat"):
        st.session_state["messages"] = []
        st.rerun()