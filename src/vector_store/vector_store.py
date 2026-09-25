from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_embedding(model: str):
    return OllamaEmbeddings(model=model)

def store_vector(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

def save_vector_db(vector_store, path: str):
    vector_store.save_local(path)

def load_vector_db(embeddings, path: str):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)