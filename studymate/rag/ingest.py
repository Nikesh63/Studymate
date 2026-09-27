"""
Ingests documents from data/documents/ into a local Chroma vector store.
Supports .txt, .md, and .pdf files. Run this once, and re-run any time you
add/change documents.

Run: python rag/ingest.py
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = Path(__file__).parent.parent / "data" / "documents"
PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_documents():
    docs = []
    for path in DOCS_DIR.glob("**/*"):
        if path.suffix.lower() in (".txt", ".md"):
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
    return docs


def main():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        print(f"No documents found in {DOCS_DIR}. Add .txt/.md/.pdf files there first.")
        return
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s).")
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    print(f"Embedding with {EMBEDDING_MODEL} (runs on CPU, this may take a bit)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Building Chroma index at {PERSIST_DIR}...")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(PERSIST_DIR))
    print("Done. Index is ready for retriever.py / app.py.")


if __name__ == "__main__":
    main()
