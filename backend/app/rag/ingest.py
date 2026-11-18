"""Indicizza la knowledge base in Qdrant.

Uso:  python -m app.rag.ingest [--force]

Idempotente: se la knowledge base (e i parametri di chunking/embedding) non sono
cambiati dall'ultima indicizzazione, esce subito senza toccare Qdrant.
Con --force reindicizza comunque.
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams

from ..config import settings
from .store import get_client, get_embeddings, get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kb-ingest")

# File e sezioni che NON sono conoscenza: istruzioni per il modello, metadati dei documenti.
EXCLUDED_FILES = {"README.md"}
EXCLUDED_SECTIONS = {"Come l'agente deve rispondere", "Fonti consultate e limiti"}
PER_ITEM_SECTIONS = {"Attrazioni e musei"}


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Separa l'intestazione YAML tra '---' dal corpo del documento."""
    if text.startswith("---"):
        _, header, body = text.split("---", 2)
        return yaml.safe_load(header) or {}, body
    return {}, text

def split_section(sezione: str, text: str, by_size: RecursiveCharacterTextSplitter) -> list[str]:
    """Divide una sezione in chunk: per voce di elenco se indipendenti, altrimenti per lunghezza."""
    if sezione in PER_ITEM_SECTIONS:
        items = [line.strip() for line in text.split("\n") if line.strip().startswith("- ")]
        if items:
            return items
    return by_size.split_text(text)

def load_chunks(kb_dir: Path) -> list[Document]:
    """Legge i .md e li divide in pezzi (chunk), ognuno con i suoi metadati."""
    by_header = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titolo"), ("##", "sezione")],
        strip_headers=True,
    )
    by_size = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)

    chunks: list[Document] = []
    for path in sorted(kb_dir.rglob("*.md")):
        if path.name in EXCLUDED_FILES:
            continue
        # utf-8-sig e replace: tolgono BOM e a-capo Windows (\r\n) dei file creati su Windows
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        meta, body = split_frontmatter(text)

        for section in by_header.split_text(body):
            titolo = section.metadata.get("titolo", path.stem)
            sezione = section.metadata.get("sezione")
            # niente sezione = testo introduttivo prima del primo "##": lo saltiamo
            if not sezione or sezione in EXCLUDED_SECTIONS:
                continue

            for piece in split_section(sezione, section.page_content, by_size):
                metadata = {
                    "source": str(path.relative_to(kb_dir)).replace("\\", "/"),
                    "doc_type": meta.get("doc_type"),
                    "destination": meta.get("destination"),
                    "titolo": titolo,
                    "sezione": sezione,
                }
                chunks.append(
                    Document(
                        # Il prefisso dice DI COSA parla il pezzo: senza, "Quartieri" di Roma
                        # e di Parigi avrebbero lo stesso significato per il modello di embedding.
                        page_content=f"{titolo} — {sezione}\n{piece}",
                        metadata={k: v for k, v in metadata.items() if v is not None},
                    )
                )
    return chunks


def wait_for_qdrant(timeout: int = 60) -> None:
    """Qdrant può impiegare qualche secondo ad avviarsi: riprova finché risponde."""
    client = get_client()
    deadline = time.time() + timeout
    while True:
        try:
            client.get_collections()
            return
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(2)


def compute_kb_hash(chunks: list[Document]) -> str:
    """Impronta dei chunk prodotti + modello di embedding.

    Si calcola sui chunk (non sui file) così cambia anche quando si modificano i
    parametri di chunking o le sezioni escluse, non solo il testo della KB.
    """
    payload = json.dumps(
        [settings.ollama_embed_model, [(c.page_content, c.metadata) for c in chunks]],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_up_to_date(client, name: str, kb_hash: str, expected_count: int) -> bool:
    """True se la collection esiste, è completa e contiene l'indicizzazione di questa stessa KB."""
    if not client.collection_exists(name):
        return False
    # Il conteggio protegge dalle indicizzazioni interrotte a metà.
    if client.count(name, exact=True).count != expected_count:
        return False
    points, _ = client.scroll(name, limit=1, with_payload=True)
    if not points:
        return False
    # langchain-qdrant salva i metadati sotto la chiave "metadata"
    stored = (points[0].payload or {}).get("metadata", {}).get("kb_hash")
    return stored == kb_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Indicizza la knowledge base in Qdrant.")
    parser.add_argument("--force", action="store_true", help="reindicizza anche se la KB non è cambiata")
    args = parser.parse_args()

    chunks = load_chunks(Path(settings.kb_dir))
    if not chunks:
        raise SystemExit(f"Nessun documento trovato in {settings.kb_dir}")
    kb_hash = compute_kb_hash(chunks)
    for chunk in chunks:
        chunk.metadata["kb_hash"] = kb_hash
    logger.info("Creati %d chunk dai documenti in %s", len(chunks), settings.kb_dir)

    wait_for_qdrant()
    client = get_client()
    name = settings.qdrant_collection

    if not args.force and is_up_to_date(client, name, kb_hash, len(chunks)):
        logger.info("Knowledge base invariata (hash %s): indicizzazione saltata", kb_hash[:12])
        return

    embeddings = get_embeddings()

    # La dimensione del vettore dipende dal modello di embedding: la ricaviamo da un test.
    dim = len(embeddings.embed_query("test"))
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    get_vector_store().add_documents(chunks)  # calcola gli embedding e salva in Qdrant
    logger.info("Indicizzati %d chunk nella collection '%s' (vettori da %d dimensioni, hash %s)", len(chunks), name, dim, kb_hash[:12])


if __name__ == "__main__":
    main()