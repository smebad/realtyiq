import logging

logger = logging.getLogger(__name__)

# Convert a list of listing dicts into a structured text block for the LLM
def format_listings_as_context(listings: list[dict]) -> str:
    
    if not listings:
        return "No relevant listings found in the database."

    lines = ["Here are the relevant property listings from our database:\n"]

    for i, listing in enumerate(listings, 1):
        price = listing.get("sale_price") or listing.get("predicted_price")
        price_str = f"${price:,.0f}" if price else "Price not available"

        garage = (
            f"{listing.get('garage_cars', 0)}-car garage"
            if listing.get("has_garage")
            else "no garage"
        )

        fireplace_str = (
            f"{listing.get('fireplaces', 0)} fireplace(s)"
            if listing.get("fireplaces", 0) > 0
            else "no fireplace"
        )

        air = "central AC" if listing.get("central_air") else "no central AC"

        block = f"""
Listing {i} (ID: {listing.get('id')}):
  - Location: {listing.get('neighborhood', 'Unknown')} neighborhood
  - Style: {listing.get('house_style', 'Unknown')}
  - Size: {listing.get('gr_liv_area', 0):.0f} sqft above ground
  - Bedrooms: {listing.get('bedroom_abvgr', 0)}
  - Bathrooms: {listing.get('total_bathrooms', 0):.1f}
  - Quality score: {listing.get('overall_qual', 0)}/10
  - Year built: {listing.get('year_built', 'Unknown')}
  - Features: {garage}, {fireplace_str}, {air}
  - Price: {price_str}
""".strip()

        lines.append(block)

    return "\n\n".join(lines)

# Build the prompt for the LLM, combining instructions, context, and user question
def build_prompt(user_question: str, context: str) -> str:

    system_instruction = """You are RealtyIQ Assistant, a helpful real estate expert.
Answer the user's question using ONLY the property listings provided below.
Be specific and mention actual listing IDs, prices, neighborhoods, and features.
If the listings do not contain enough information to answer, say so honestly.
Keep your answer concise and helpful (3-5 sentences max).
Do not make up any properties or prices that are not in the listings."""

    prompt = f"""<s>[INST] {system_instruction}

{context}

User question: {user_question} [/INST]"""

    return prompt

# A simpler prompt format for smaller models like flan-t5 that may not handle complex instructions as well
def build_flan_prompt(user_question: str, context: str) -> str:

    return f"""Answer this real estate question using only the listings below.

{context}

Question: {user_question}
Answer:"""