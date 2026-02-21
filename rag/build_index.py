from rag.document_loader import load_documents
from rag.chunker import chunk_document
from rag.embedder import Embedder
from rag.vector_store import VectorStore

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")

def build_index():

    docs = load_documents(DOC_DIR)
    embedder = Embedder()

    all_chunks = []
    metadata = []

    for doc in docs:
        chunks = chunk_document(doc["content"])
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                "source": doc["filename"],
                "text": chunk
            })

    embeddings = embedder.embed(all_chunks)
    store = VectorStore(len(embeddings[0]))
    store.add(embeddings, metadata)

    return store, embedder

if __name__ == "__main__":
    build_index()