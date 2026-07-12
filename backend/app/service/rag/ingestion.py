"""Ingestion de documents dans les tables vectorielles RAG.

Connector "documents" minimal : découpe le texte (chunking), calcule les embeddings Ollama
et insère les chunks dans la table pgvector correspondante. Upsert idempotent par ``docId``.

Bibliothèques haut niveau (pas de boilerplate maison) :
- ``langchain-text-splitters`` : découpe récursive robuste (paragraphes -> phrases -> mots).
- ``pypdf`` : extraction de texte des PDF.
"""

import hashlib
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text
from sqlmodel import Session, SQLModel

from app.models import VectorCode, VectorDocs, VectorJira, VectorPdf
from app.service.rag import embeddings
from app.service.rag.sources import SOURCES

# Découpe : ~1000 caractères avec 200 de recouvrement (valeurs usuelles RAG).
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# data_type -> modèle de table.
_MODEL_BY_TYPE: dict[str, type[SQLModel]] = {
    "CONFLUENCE": VectorDocs,
    "JIRA": VectorJira,
    "PDF": VectorPdf,
    "GITHUB": VectorCode,
}
# data_type -> modèle d'embedding Ollama (déduit de la config des sources).
_EMBEDDING_MODEL_BY_TYPE: dict[str, str] = {s.data_type: s.embedding_model for s in SOURCES}


def extract_text_from_file(filename: str, data: bytes) -> str:
    """Extrait le texte d'un fichier uploadé (.pdf via pypdf, sinon décodage texte)."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


class DocumentIngestionService:
    """Insère des documents dans une table vectorielle (par défaut ``vector_docs``)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def ingest_text(
        self,
        *,
        content: str,
        title: str,
        url: str | None = None,
        source: str = "CONFLUENCE",
        doc_id: str | None = None,
        extra_metadata: dict | None = None,
    ) -> int:
        """Découpe, embed et insère un document. Retourne le nombre de chunks insérés.

        ``doc_id`` sert à l'upsert idempotent : les chunks existants du même document
        sont supprimés avant réinsertion (équivalent de ``deleteByMetadata`` côté Java).
        """
        source = source.upper()
        model = _MODEL_BY_TYPE.get(source)
        if model is None:
            raise ValueError(f"Source inconnue : {source} (attendu : {list(_MODEL_BY_TYPE)})")
        embedding_model = _EMBEDDING_MODEL_BY_TYPE[source]

        effective_doc_id = doc_id or hashlib.sha256((url or title).encode()).hexdigest()[:32]
        self._delete_existing(model, effective_doc_id)

        chunks = _splitter.split_text(content)
        base_metadata = {
            "source": source,
            "type": source,
            "title": title,
            "docId": effective_doc_id,
            **({"url": url} if url else {}),
            **(extra_metadata or {}),
        }

        for index, chunk in enumerate(chunks):
            vector = embeddings.embed(chunk, embedding_model)
            row = model(
                content=chunk,
                doc_metadata={**base_metadata, "chunk": index},
                embedding=vector,
            )
            self._session.add(row)

        self._session.commit()
        return len(chunks)

    def ingest_file(
        self,
        *,
        filename: str,
        data: bytes,
        title: str | None = None,
        url: str | None = None,
        source: str = "CONFLUENCE",
        doc_id: str | None = None,
    ) -> int:
        """Ingestion d'un fichier uploadé (texte extrait puis délégué à ``ingest_text``)."""
        content = extract_text_from_file(filename, data)
        return self.ingest_text(
            content=content,
            title=title or filename,
            url=url,
            source=source,
            doc_id=doc_id,
        )

    def _delete_existing(self, model: type[SQLModel], doc_id: str) -> None:
        table = model.__tablename__  # type: ignore[attr-defined]
        self._session.execute(
            text(f"DELETE FROM {table} WHERE metadata->>'docId' = :doc_id"),  # noqa: S608 (table interne)
            {"doc_id": doc_id},
        )
