import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator

# Check if MPS (Apple GPU) is available
device = 0 if torch.backends.mps.is_available() else -1
if device == 0:
    st.write("✅ MPS (GPU) is enabled!")
else:
    st.write("⚠️ Running on CPU. Consider enabling GPU for better performance.")

# Initialize summarization pipeline with caching to avoid re-downloading
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

summarizer = load_summarizer()

# Function to create Chroma index from uploaded PDF
@st.cache_resource
def create_chroma_index_from_pdf(pdf_file):
    with open("uploaded_file.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    loader = PyPDFLoader("uploaded_file.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

# Function to search Chroma index
def search_chroma_index(vectorstore, query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    return results

# Function to summarize results
def summarize_results(results):
    combined_text = "\n\n".join([result.page_content for result in results])
    summary = summarizer(combined_text, max_length=75, min_length=20, do_sample=False)
    return summary[0]["summary_text"]

# Streamlit UI
st.title("📄 File-Based Conversational Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    vectorstore = create_chroma_index_from_pdf(uploaded_file)
    st.success("PDF indexed successfully!")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input for conversation
    user_query = st.chat_input("Ask me anything about the uploaded PDF...")

    if user_query:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Retrieve relevant context from Chroma index
        search_results = search_chroma_index(vectorstore, user_query)

        # Summarize response
        bot_response = summarize_results(search_results)

        # Display chatbot response
        with st.chat_message("assistant"):
            st.write(bot_response)

        # Save response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
