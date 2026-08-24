from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from Services.vectorstore_service import get_retriever
from Services.llm_factory import get_llm
from Utils.logger import get_logger
from typing import List
from langchain_core.documents import Document

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert AI assistant for building energy management.
Answer questions using ONLY the provided context from ASHRAE standards and BMS manuals.
If the answer is not in the context, say: "This information is not available in the loaded documents."

Context:
{context}
"""

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')} | Type: {d.metadata.get('doc_type', 'N/A')}]\n{d.page_content}"
        for d in docs
    )

def build_rag_chain():
    retriever = get_retriever(search_type="mmr")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def ask_question(question: str) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(question)
    chain = build_rag_chain()
    answer = chain.invoke(question)

    sources = [
        {
            "content": d.page_content[:300],
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
        }
        for d in docs
    ]
    return {"answer": answer, "sources": sources}