import streamlit as st
from chromadb import Client
from chromadb.config import Settings
from mistralai import MistralClient  # Import MistralAI client
import os
import pdfplumber  # Add pdfplumber for PDF text extraction
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
st.set_page_config(page_title="Embeddings Indexer", layout="wide")

# --- API Key Input ---
api_key = st.sidebar.text_input("Enter your MistralAI API Key:", type="password")

# --- Initialize MistralAI and ChromaDB ---
@st.cache_resource
def initialize_indexer(api_key):
    mistral_client = MistralClient(api_key=api_key)
    chroma_client = Client(Settings())
    try:
        collection = chroma_client.get_collection(name="Students1")
    except Exception as e:
        collection = chroma_client.create_collection(name="Students1")

    return mistral_client, collection

if api_key:
    mistral_client, chroma_client = initialize_indexer(api_key)

    st.title("📚 Upload & Index Documents with MistralAI")

    # --- Document Upload Section ---
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload text or PDF files", type=['txt', 'pdf'], accept_multiple_files=True)

    if uploaded_files:
        st.success("Documents uploaded successfully!")

        # --- Indexing Section ---
        st.header("2. Indexing Documents")

        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    content = ""
                    for page in pdf.pages:
                        content += page.extract_text()  # Extract text using pdfplumber
            else:
                content = uploaded_file.read().decode("utf-8")

            doc_id = uploaded_file.name
            documents.append({"id": doc_id, "content": content})

        # Index documents in ChromaDB
        with st.spinner("Indexing documents..."):
            for doc in documents:
                embedding_response = mistral_client.embeddings.create(
                    input=doc['content'],
                    model="text-embedding-mistral"  # Use the appropriate MistralAI model
                )
                embedding = embedding_response.data[0].embedding  

                chroma_client.add(ids=doc['id'], embeddings=embedding, metadatas={})

        st.success("Documents indexed successfully!")

        # --- Save Index ---
        st.header("3. Save Index")
        if st.button("Save Index"):
            chroma_client.save("index_data")
            st.success("Index saved successfully!")

else:
    st.warning("Please enter your MistralAI API Key to proceed.")
