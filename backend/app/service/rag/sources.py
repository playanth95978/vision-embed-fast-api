"""Configuration des sources RAG et constantes du pipeline.

Réplique le mapping ``DataType`` -> ``*VectorStoreService`` du backend Java, ainsi que les
constantes de ``ChatService`` (RRF, seuils, tailles de troncature).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RagSource:
    """Une source RAG = une table pgvector + son modèle d'embedding Ollama + son poids RRF."""

    data_type: str          # nom logique (posé en metadata "type", sert au poids RRF)
    table: str              # table pgvector (schéma Spring AI : id, content, metadata, embedding)
    embedding_model: str    # modèle Ollama
    dimensions: int
    weight: float           # DataType.getWeight() -> pondération du RRF


# Ordre et valeurs alignés sur DataType.java + les *VectorStoreService (backend Spring Boot).
SOURCES: list[RagSource] = [
    RagSource("CONFLUENCE", "vector_docs", "nomic-embed-text", 768, 1.0),
    RagSource("JIRA", "vector_jira", "nomic-embed-text", 768, 0.8),
    RagSource("GITHUB", "vector_code", "mxbai-embed-large", 1024, 1.2),
    RagSource("PDF", "vector_pdf", "nomic-embed-text", 768, 0.9),
]

# Tables autorisées (garde-fou anti-injection, comme ALLOWED_TABLES côté Java).
ALLOWED_TABLES: frozenset[str] = frozenset(s.table for s in SOURCES)
WEIGHT_BY_TYPE: dict[str, float] = {s.data_type: s.weight for s in SOURCES}

# Constantes du pipeline (ChatService / AbstractVectorStoreService).
RRF_K = 60
PER_SOURCE_LIMIT = 30       # AbstractVectorStoreService.getLimit()
RRF_TOP_K = 20              # ChatService.retrieveContextWithSources
RERANK_TOP_K = 5
FALLBACK_RERANK_TOP_K = 3
SIMILARITY_THRESHOLD = 0.3  # SearchRequest.similarityThreshold
NO_CONTEXT = "NO_CONTEXT"
