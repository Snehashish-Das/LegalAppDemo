import streamlit as st
import fitz
import re
import unicodedata
import numpy as np
import faiss
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Legal RAG System",
    layout="wide"
)

st.title("⚖️ Legal Document RAG System")


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.header("RAG Configuration")

# Chunk Size
chunk_size = st.sidebar.selectbox(
    "Select Chunk Size",
    [100, 150, 200, 250, 300],
    index=2
)

# Overlap
overlap = st.sidebar.selectbox(
    "Select Overlap",
    [20, 30, 50, 60, 100],
    index=3
)

# Top K Retrieval
top_k = st.sidebar.slider(
    "Top K Chunks",
    min_value=1,
    max_value=10,
    value=5
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = SentenceTransformer('all-MiniLM-L6-v2')

    return model


model = load_model()


# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

def extract_text_from_pdf(uploaded_file):

    text = ""

    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    for page in pdf_document:

        text += page.get_text()

    return text


# ==========================================
# LEGAL PREPROCESSING
# ==========================================

def preprocess_legal_text(text):

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Remove parsed page markers
    text = re.sub(
        r'<PARSED TEXT FOR PAGE:.*?>',
        ' ',
        text
    )

    # Remove Indian Kanoon footer/header
    text = re.sub(
        r'Indian Kanoon - http\S+',
        ' ',
        text
    )

    # Remove page numbers
    text = re.sub(
        r'\b\d+\b\s*$',
        ' ',
        text,
        flags=re.MULTILINE
    )

    # Remove excessive line breaks
    text = re.sub(r'\n+', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove weird unicode characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Keep important punctuations
    text = re.sub(
        r'[^a-zA-Z0-9\s\.,:\-\(\)/&]',
        ' ',
        text
    )

    # Fix spacing before periods
    text = re.sub(r'\s+\.', '.', text)

    # Final cleanup
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==========================================
# CHUNKING FUNCTION
# ==========================================

def create_chunks(text, chunk_size=200, overlap=60):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Legal PDF",
    type=["pdf"]
)


# ==========================================
# MAIN PIPELINE
# ==========================================

if uploaded_file is not None:

    st.success("PDF Uploaded Successfully!")

    # --------------------------------------
    # Extract Text
    # --------------------------------------

    with st.spinner("Extracting text from PDF..."):

        extraction_start = time.time()

        raw_text = extract_text_from_pdf(uploaded_file)

        extraction_end = time.time()

        extraction_time = extraction_end - extraction_start

    # --------------------------------------
    # Preprocess
    # --------------------------------------

    with st.spinner("Preprocessing legal document..."):

        preprocess_start = time.time()

        cleaned_text = preprocess_legal_text(raw_text)

        preprocess_end = time.time()

        preprocess_time = preprocess_end - preprocess_start

    # --------------------------------------
    # Chunking
    # --------------------------------------

    with st.spinner("Creating chunks..."):

        chunk_start = time.time()

        chunks = create_chunks(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        chunk_end = time.time()

        chunk_time = chunk_end - chunk_start

    # --------------------------------------
    # Display Configuration
    # --------------------------------------

    st.info(
        f"""
        Chunk Size: {chunk_size}
        
        Overlap: {overlap}
        
        Top K Chunks: {top_k}
        """
    )

    # --------------------------------------
    # Metrics
    # --------------------------------------

    avg_tokens = np.mean(
        [len(chunk.split()) for chunk in chunks]
    )

    st.subheader("📊 Processing Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Chunks",
        len(chunks)
    )

    col2.metric(
        "Average Tokens",
        round(avg_tokens, 2)
    )

    col3.metric(
        "Chunking Time (sec)",
        round(chunk_time, 2)
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "Extraction Time (sec)",
        round(extraction_time, 2)
    )

    col5.metric(
        "Preprocessing Time (sec)",
        round(preprocess_time, 2)
    )

    # --------------------------------------
    # Generate Embeddings
    # --------------------------------------

    with st.spinner("Generating embeddings..."):

        embedding_start = time.time()

        embeddings = model.encode(
            chunks,
            show_progress_bar=True
        )

        embedding_end = time.time()

        embedding_time = embedding_end - embedding_start

    st.metric(
        "Embedding Time (sec)",
        round(embedding_time, 2)
    )

    # --------------------------------------
    # Create FAISS Index
    # --------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    st.success("✅ RAG Pipeline Ready!")

    # ======================================
    # QUERY INPUT
    # ======================================

    query = st.text_input(
        "Enter your legal query:"
    )

    # ======================================
    # RETRIEVAL
    # ======================================

    if query:

        # ----------------------------------
        # Query Embedding
        # ----------------------------------

        query_embedding = model.encode([query])

        # ----------------------------------
        # Retrieval
        # ----------------------------------

        retrieval_start = time.time()

        distances, indices = index.search(
            np.array(query_embedding),
            top_k
        )

        retrieval_end = time.time()

        retrieval_time = (
            retrieval_end - retrieval_start
        )

        st.metric(
            "Retrieval Time (sec)",
            round(retrieval_time, 8)
        )

        st.subheader(
            f"📄 Top {top_k} Relevant Chunks"
        )

        # ----------------------------------
        # Similarity Scores
        # ----------------------------------

        retrieved_embeddings = embeddings[indices[0]]

        similarities = cosine_similarity(
            query_embedding,
            retrieved_embeddings
        )[0]

        avg_similarity = np.mean(similarities)

        st.metric(
            "Average Similarity Score",
            round(avg_similarity, 4)
        )

        # ----------------------------------
        # Display Results
        # ----------------------------------

        for i, idx in enumerate(indices[0]):

            st.markdown(f"## Chunk {i+1}")

            st.write(
                f"### Similarity Score: "
                f"{similarities[i]:.4f}"
            )

            st.write(chunks[idx])

            st.divider()