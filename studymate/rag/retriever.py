"""
Reusable local-document retrieval logic. Import from app.py.
"""
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 4

_embeddings = None
_vectordb = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def local_index_exists() -> bool:
    return PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())


def _get_vectordb():
    global _vectordb
    if _vectordb is None:
        _vectordb = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=_get_embeddings())
    return _vectordb


def retrieve_local_context(query: str, top_k: int = TOP_K):
    """
    Returns (context_string, list_of_source_dicts) from local documents.
    Returns ("", []) if no local index has been built yet.
    """
    if not local_index_exists():
        return "", []

    retriever = _get_vectordb().as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(query)

    context_parts = []
    sources = []
    for i, doc in enumerate(results):
        context_parts.append(f"[Local Source {i+1}]: {doc.page_content}")
        source_name = doc.metadata.get("source", "local document")
        sources.append({"title": Path(source_name).name, "url": None})

    context = "\n\n".join(context_parts)
    return context, sources
