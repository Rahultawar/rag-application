# RAG Application using LangChain & FastAPI

A clean, modular Retrieval-Augmented Generation (RAG) application built with **LangChain**, **FastAPI**, **FAISS**, **Ollama**, and **Google Gemini**.

This application allows users to upload PDF documents, automatically splits and embeds their content into a local vector database, and answers user questions strictly based on the provided documents.

---

## 📌 Features

- **Multi-Document Ingestion**: Upload up to 5 PDF files at a time via `POST /ingest`.
- **Memory-Efficient Processing**: Uses `lazy_load()` to stream documents into memory during ingestion.
- **Configurable Text Chunking**: Splits documents using `RecursiveCharacterTextSplitter` with customizable chunk size and overlap.
- **Local Embeddings**: Computes dense vector embeddings locally using **Ollama** (defaults to `nomic-embed-text`).
- **Persistent Vector Index**: Saves vector embeddings to a local FAISS index (`faiss_index/`) on disk to avoid re-indexing on every query.
- **LangChain Expression Language (LCEL)**: Uses modern LCEL chains (`retriever | prompt | llm | parser`) for predictable execution flow.
- **Context-Grounded Responses**: Prompt template designed to answer questions strictly from retrieved context and reject out-of-context questions.
- **Interactive API Documentation**: Built-in Swagger UI provided by FastAPI for testing endpoints.

---

## 🏗️ Architecture & Workflow

```text
[User / Client]
       │
       ├── (1) Upload PDFs ──────► POST /ingest
       │                                │
       │                                ├── Save to 'data/' folder
       │                                ├── Lazy load documents
       │                                ├── Split into text chunks
       │                                ├── Generate embeddings (Ollama)
       │                                └── Persist FAISS index to disk ('faiss_index/')
       │
       └── (2) Ask Question ────► GET /generate?question=...
                                        │
                                        ├── Load persisted FAISS index
                                        ├── Retrieve top-K relevant chunks
                                        ├── Format context & question into prompt
                                        ├── Query Google Gemini LLM (via LCEL chain)
                                        └── Return generated answer
```

---

## 📁 Project Structure

```text
rag-application/
│
├── data/                       # Directory where uploaded PDF documents are stored
├── faiss_index/                # Persisted FAISS vector index files (generated after ingestion)
│
├── src/
│   ├── ingestion/              # Document loading logic
│   │   ├── __init__.py
│   │   └── document_loader.py  # Loads PDFs with DirectoryLoader & lazy_load
│   │
│   ├── splitter/               # Text chunking logic
│   │   ├── __init__.py
│   │   └── text_splitter.py    # RecursiveCharacterTextSplitter wrapper
│   │
│   ├── vector_store/           # Embedding & Vector DB logic
│   │   ├── __init__.py
│   │   └── vector_store.py     # Ollama embeddings, FAISS indexing, save & load
│   │
│   ├── prompts/                # Prompt templates
│   │   ├── __init__.py
│   │   └── templates.py        # Context prompt template
│   │
│   ├── retriever/              # Retriever helpers
│   │   ├── __init__.py
│   │   └── retriever.py
│   │
│   ├── __init__.py
│   └── main.py                 # FastAPI application and API routes (/ingest, /generate)
│
├── .env.example                # Example environment configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Prerequisites

Before running the application, make sure you have the following installed:

1. **Python 3.10+**
2. **Google Gemini API Key**: [Get an API key from Google AI Studio](https://aistudio.google.com/).
3. **Ollama**: [Download and install Ollama](https://ollama.com/) locally to run embedding models.
   - Pull the default embedding model:
     ```bash
     ollama pull nomic-embed-text
     ```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Rahultawar/rag-application.git
cd rag-application
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` and fill in your configuration:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=gemini-2.5-flash
CHUNK_SIZE=100
CHUNK_OVERLAP=20
```

---

## 🏃 Running the Application

Make sure **Ollama** is running in the background:
```bash
ollama serve
```

Start the FastAPI application:
```bash
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```
*Or directly with Python:*
```bash
python src/main.py
```

Once running, access the interactive Swagger API documentation at:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 📖 API Usage Guide

### 1. Ingest Documents (`POST /ingest`)
Upload up to 5 `.pdf` files to store them and generate the vector index.

- **URL**: `/ingest`
- **Method**: `POST`
- **Payload**: `multipart/form-data` with `files` field.
- **Example using cURL:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/ingest" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "files=@sample_document.pdf"
  ```
- **Response:**
  ```json
  {
    "message": "Documents uploaded, ingested, and FAISS index persisted successfully.",
    "uploaded_files": ["sample_document.pdf"],
    "total_chunks": 42
  }
  ```

### 2. Query Documents (`GET /generate`)
Ask questions based on the ingested documents.

- **URL**: `/generate`
- **Method**: `GET`
- **Query Parameter**: `question` (string)
- **Example using cURL:**
  ```bash
  curl -X GET "http://127.0.0.1:8000/generate?question=What%20are%20the%20candidate%20qualifications%3F"
  ```
- **Response:**
  ```json
  {
    "response": "Based on the provided documents, the candidate holds a Bachelor's degree in Computer Science..."
  }
  ```

> **Note:** If no documents have been ingested yet, `/generate` returns an HTTP 400 error asking you to run `/ingest` first.

---

## 🔧 Environment Configuration Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(Required)* | Google Gemini API key |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model name used for computing embeddings |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model name used for generation |
| `CHUNK_SIZE` | `100` | Target character size for each document chunk |
| `CHUNK_OVERLAP` | `20` | Number of overlapping characters between chunks |
