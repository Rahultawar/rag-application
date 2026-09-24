from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_documents(data_path: str):
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()
