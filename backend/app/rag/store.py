"""Collegamenti condivisi verso Ollama (embedding) e Qdrant (vector DB)."""

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ..config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """Trasforma un testo in un vettore di numeri, usando il modello di embedding in Ollama."""
    return OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)

@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Vista LangChain sulla collection già popolata da ingest.py."""
    return QdrantVectorStore(
        client=get_client(),
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )