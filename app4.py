# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load the model (you can use other pretrained models as well)
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Semantic Text Similarity App")
st.write("This app uses a SentenceTransformer model to compute embeddings and similarity between two texts.")

# User inputs
text1 = st.text_input("Enter first text:")
text2 = st.text_area("Enter second text:")

if st.button("Compute Similarity"):
    if text1 and text2:
        # Compute embeddings
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)

        # Compute cosine similarity
        similarity = util.cos_sim(embedding1, embedding2).item()

        # Convert to percentage
        similarity_percentage = similarity * 100

        # Show result
        st.success(f"Relevance Check: {similarity_percentage:.2f}%")
    else:
        st.warning("Please enter text in both fields.")
