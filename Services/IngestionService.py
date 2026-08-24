from langchain_community.document_loaders import PyPDFLoader,CSVLoader,TextLoader,DirectoryLoader
from Utils.Config import get_settings
from Utils.logger import get_logger
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = get_logger(__name__)
setting = get_settings()

def load_Documents(file_path:str,doc_type:str)->List[Document]:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        loader = CSVLoader(file_path)

    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = doc_type
        doc.metadata["file_path"] = file_path
    logger.info(f"Loaded {len(docs)} documents,from {file_path}")
    return docs
def diretory_loader(folder_path:str)->List[Document]:
    loader = DirectoryLoader(folder_path)
    docs = loader.load()
    logger.info(f"Documents loaded from {len(docs)}")

    return docs


def split_Documents(docs:List[Document])->List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=setting.CHUNK_SIZE,
        chunk_overlap=setting.CHUNK_OVERLAP,
        separators=["\n\n","\n","."," "]
    )
    chunk = splitter.split_documents(docs)
    logger.info(f"Split Document into  {len(chunk)} chunks")
    return chunk

