import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# ----------------------------------------------------
# Load and cache model (only once, avoids reloads)
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, good for semantic similarity

model = load_model()

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.title("Semantic Text Similarity App")
st.write("This app uses a SentenceTransformer model to compute embeddings and similarity.")

# Text inputs
text1 = st.text_area("Enter first text:")
text2 = st.text_area("Enter second text:")

if st.button("Compute Similarity"):
    if text1.strip() and text2.strip():
        # Get embeddings
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)

        # Cosine similarity
        similarity = util.pytorch_cos_sim(emb1, emb2).item()

        st.success(f"Relavance Check: *{similarity:.4f}*")
    else:
        st.warning("Please enter text in both boxes.")
