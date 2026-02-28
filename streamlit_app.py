import streamlit as st
import numpy as np
import pandas as pd
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Product Recommender",
    page_icon="🛒",
    layout="centered",
)

# ── Load model assets (cached – runs only once per session) ───────────────────
@st.cache_resource
def load_assets():
    """Load the cosine-similarity matrix and products dataframe once."""
    base = os.path.dirname(os.path.abspath(__file__))
    matrix_path   = os.path.join(base, "model", "model.npy")
    products_path = os.path.join(base, "model", "products.csv")

    try:
        cosine_sim = np.load(matrix_path)
        products   = pd.read_csv(products_path)
        indices    = pd.Series(
            products.index, index=products["product_name"]
        ).drop_duplicates()
        return cosine_sim, products, indices
    except FileNotFoundError:
        return None, None, None

cosine_sim, products, indices = load_assets()

# ── Recommendation logic (same as original Flask version) ─────────────────────
def get_recommendations(product_name, num_recommendations=5):
    if product_name not in indices:
        return pd.DataFrame()

    idx        = indices[product_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : num_recommendations + 1]

    product_indices = [i[0] for i in sim_scores]
    return products.iloc[product_indices]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛒 Amazon Product Recommender")
st.markdown("Select a product below and get the **top 5 similar products** instantly.")

if products is None:
    st.error(
        "⚠️ Model assets not found. "
        "Please make sure `model/model.npy` and `model/products.csv` exist."
    )
    st.stop()

# Searchable dropdown — all known product names
product_list     = sorted(products["product_name"].dropna().unique().tolist())
selected_product = st.selectbox(
    "🔍 Search or select a product:",
    options=[""] + product_list,
    index=0,
)

if st.button("Get Recommendations", use_container_width=True):
    if not selected_product:
        st.warning("Please select a product first.")
    else:
        with st.spinner("Finding similar products…"):
            recs = get_recommendations(selected_product)

        if recs.empty:
            st.error(f'Product "{selected_product}" not found in the dataset.')
        else:
            st.success(f'Top 5 recommendations for **"{selected_product}"**:')
            st.markdown("---")

            for i, (_, row) in enumerate(recs.iterrows(), start=1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {row['product_name']}**")
                    st.caption(f"📁 {row.get('category', 'N/A')}")
                with col2:
                    st.metric("Rating", f"⭐ {row.get('rating', 'N/A')}")
                    st.markdown(f"**₹ {row.get('actual_price', 'N/A')}**")
                st.markdown("---")
