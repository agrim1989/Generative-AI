from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings


def main():
    loader = DirectoryLoader("data/",
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                chunk_overlap=30)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

if __name__ == "__main__":
    main()