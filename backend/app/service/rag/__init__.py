"""Pipeline RAG "merged" — portage de l'endpoint ``/api/chat/merge`` du backend Spring Boot.

Recherche hybride pgvector + BM25 (ParadeDB), fusion RRF pondérée, reranking cross-encoder,
puis génération en streaming via Mistral. Voir ``service.RagService``.
"""

from app.service.rag.service import RagService

__all__ = ["RagService"]
