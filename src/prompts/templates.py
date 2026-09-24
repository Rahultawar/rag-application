from langchain_core.prompts import PromptTemplate

contextPrompt = PromptTemplate(
    template="""
    Answer the question based on the context.
    Context: {context}
    Question: {question}
    """,
    input_variables = ['context', 'question']
)
