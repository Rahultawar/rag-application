from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ingestion.document_loader import load_documents
from prompts.templates import contextPrompt
from splitter.text_splitter import split_documents
from vector_store.vector_store import load_embedding, store_vector_db

load_dotenv()

app = FastAPI(title="RAG Application using LangChain")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.get("/generate")
def generate_response(question: str):
    docs = load_documents("data")
    chunks = split_documents(docs, 100, 20)
    embeddings = load_embedding("nomic-embed-text")
    vector_store = store_vector_db(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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