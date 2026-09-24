from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_embedding(model: str):
    return OllamaEmbeddings(model=model)

def store_vector_db(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)