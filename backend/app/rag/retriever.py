"""Ricerca nella knowledge base.
"""

from langchain_core.documents import Document
from qdrant_client import models

from ..config import settings
from .store import get_vector_store


def search_knowledge(
    query: str,
    destination: str | None = None,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """Restituisce i k pezzi più simili alla query, con il punteggio (più alto = più simile).

    Se `destination` è indicata (es. "Parigi") la ricerca si limita ai documenti di quella città.
    """
    store = get_vector_store()
    k = k or settings.rag_top_k

    only_destination = None
    if destination:
        only_destination = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.destination",  # langchain-qdrant salva i metadati sotto "metadata"
                    match=models.MatchValue(value=destination.strip().capitalize()),
                )
            ]
        )

    if only_destination is not None:
        results = store.similarity_search_with_score(query, k=k, filter=only_destination)
    else:
        results = store.similarity_search_with_score(query, k=k)
    return results