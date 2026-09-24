def retrieve(retriever, question: str):
        return retriever.invoke(question)