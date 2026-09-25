import os
import shutil
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ingestion.document_loader import load_documents
from prompts.templates import contextPrompt
from splitter.text_splitter import split_documents
from vector_store.vector_store import (
    load_embedding,
    store_vector,
    save_vector_db,
    load_vector_db,
)

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 100))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

app = FastAPI(title="RAG Application using LangChain")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    if not files or all(f.filename == "" for f in files):
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="You can upload a maximum of 5 files at a time."
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    saved_files = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{file.filename}'. Only .pdf files are supported."
            )
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = load_embedding(EMBEDDING_MODEL)
    vector_store = store_vector(chunks, embeddings)
    save_vector_db(vector_store, INDEX_DIR)

    return {
        "message": "Documents uploaded, ingested, and FAISS index persisted successfully.",
        "uploaded_files": saved_files,
        "total_chunks": len(chunks)
    }

@app.get("/generate")
def generate_response(question: str):
    embeddings = load_embedding(EMBEDDING_MODEL)
    vector_store = load_vector_db(embeddings, INDEX_DIR)

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="Vector store not found. Please upload documents via POST /ingest first."
        )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | contextPrompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(question)

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)