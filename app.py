import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re

# ----------------------------
# Preprocessing helpers
# ----------------------------
def remove_emojis(txt: str) -> str:
    emoji_re = re.compile(
        "["
        u"\U0001F600-\U0001F64F"   # emoticons
        u"\U0001F300-\U0001F5FF"   # symbols & pictographs
        u"\U0001F680-\U0001F6FF"   # transport
        u"\U0001F1E0-\U0001F1FF"   # flags
        u"\U00002700-\U000027BF"   # dingbats
        u"\U000024C2-\U0001F251"   # enclosed chars
        "]+", flags=re.UNICODE
    )
    return emoji_re.sub("", str(txt))

def preprocess(txt: str) -> str:
    txt = remove_emojis(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# ----------------------------
# Model loading
# ----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(MODEL_NAME, device=device)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔎 Semantic Relevance Checker")
st.write("Check how relevant text/content is to a given topic using Sentence Transformers.")

# Input method
mode = st.radio("Choose input type:", ["Single Topic (vs many contents)", "Row-by-Row (each topic with its content)"])

uploaded_file = st.file_uploader("Upload CSV file (must have 'topic' and 'content' columns)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "topic" not in df.columns or "content" not in df.columns:
        st.error("CSV must contain 'topic' and 'content' columns.")
    else:
        if mode == "Single Topic (vs many contents)":
            # assume one topic in CSV
            main_topic = df["topic"].iloc[0]
            contents = df["content"].astype(str).tolist()

            topic_vec = embedder.encode(preprocess(main_topic), convert_to_tensor=True, normalize_embeddings=True)
            content_vecs = embedder.encode([preprocess(c) for c in contents], convert_to_tensor=True, normalize_embeddings=True)

            sims = util.cos_sim(topic_vec, content_vecs).cpu().numpy().flatten()
            scores = [round(((s + 1) / 2) * 100, 2) for s in sims]

            df_out = pd.DataFrame({"topic": main_topic, "content": contents, "relevance_pct": scores})

        else:  # row-by-row
            topics = [preprocess(t) for t in df["topic"].astype(str).tolist()]
            contents = [preprocess(c) for c in df["content"].astype(str).tolist()]

            topic_vecs = embedder.encode(topics, convert_to_tensor=True, normalize_embeddings=True)
            content_vecs = embedder.encode(contents, convert_to_tensor=True, normalize_embeddings=True)

            sims = util.cos_sim(topic_vecs, content_vecs).cpu().numpy()
            diag_vals = sims.diagonal()
            scores = [round(((s + 1) / 2) * 100, 2) for s in diag_vals]

            df_out = df.copy()
            df_out["relevance_pct"] = scores

        # Display results
        st.subheader("📊 Relevance Results")
        st.dataframe(df_out.sort_values("relevance_pct", ascending=False))

        # Download option
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", csv, "relevance_results.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to start.")
