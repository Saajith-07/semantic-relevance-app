# app.py
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

# Load model once
MODEL_NAME = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
def load_model():
    return SentenceTransformer(MODEL_NAME)   # don’t push to device (GPU not supported)

embedder_model = load_model()

st.set_page_config(page_title="Relevance Checker", layout="centered")

st.title("🔍 Relevance Checker")
st.write("Enter a **topic** and some **text/code/content**, and see how relevant they are!")

# --- Input fields ---
topic = st.text_area("📝 Enter Topic:", placeholder="e.g. Prime number")
content = st.text_area("📄 Enter Content/Text:", placeholder="Paste code or text here...")

if st.button("Check Relevance"):
    if topic.strip() and content.strip():
        # Encode both
        topic_vec = embedder_model.encode(topic, convert_to_tensor=True, normalize_embeddings=True, device=device)
        content_vec = embedder_model.encode(content, convert_to_tensor=True, normalize_embeddings=True, device=device)

        # Cosine similarity
        similarity = util.cos_sim(topic_vec, content_vec).cpu().item()
        relevance_pct = round(((similarity + 1) / 2) * 100, 2)

        # Display result
        if relevance_pct >= 90:
            color = "✅ **Highly Relevant**"
        elif relevance_pct >= 50:
            color = "⚠️ **Somewhat Relevant**"
        else:
            color = "❌ **Not Relevant**"

        st.subheader(f"Relevance Score: {relevance_pct}%")
        st.markdown(color)
    else:
        st.warning("Please fill in both fields before checking.")
