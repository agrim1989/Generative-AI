import streamlit as st
from chromadb import Client
from chromadb.config import Settings
from google import genai
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# --- Configuration ---
st.set_page_config(page_title="Embeddings Indexer", layout="wide")

# --- API Key Input ---
api_key = st.sidebar.text_input("Enter your Google GenAI API Key:", type="password")

# --- Initialize GenAI and ChromaDB ---
@st.cache_resource
def initialize_indexer(api_key):
    gen_ai = genai.Client(api_key=api_key)
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "langchain-test-index"  # change if desired
    index = pc.Index(index_name)
    return gen_ai, index

if api_key:
    gen_ai, index = initialize_indexer(api_key)

    st.title("📚 Retrive response from Gen AI")

    query = st.text_input("Enter your query:")
    if query:
        response = index.query(
            query=query, limit=3, top_k=3
        )
        response = response["matches"]
        print(response)
        for match in response:
            st.write(f"Title: {match['metadata']['title']}")
            st.write(f"Content: {match['metadata']['content']}")
            st.write(f"Source: {match['metadata']['source']}")
            st.write("\n")
else:
    st.warning("Please enter your Google GenAI API Key to proceed.")
