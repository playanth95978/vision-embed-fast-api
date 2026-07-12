"""Reranker cross-encoder (BAAI/bge-reranker-v2-m3).

Le backend Java exécute ce modèle en ONNX (DJL + onnxruntime, ~100 lignes de plomberie).
En Python il se charge directement via FlagEmbedding. On suit le même pattern Singleton
paresseux que ``app.service.image_embedding.ImageEmbedding``.
"""

from typing import Any

from app.core.config import settings
from app.service.rag.schemas import Doc


class Reranker:
    """Singleton paresseux autour du cross-encoder de reranking."""

    _model: Any = None

    @classmethod
    def _get_model(cls) -> Any:
        if cls._model is None:
            # Import différé : le chargement du modèle est coûteux (~2 Go au 1er appel).
            from FlagEmbedding import FlagReranker

            cls._model = FlagReranker(settings.RERANKER_MODEL, use_fp16=True)
        return cls._model

    @classmethod
    def rerank(cls, query: str, docs: list[Doc], top_k: int) -> list[Doc]:
        """Rerank puis troncature à ``top_k`` (équiv. RerankerService.rerank).

        Le score est stocké dans ``metadata['rerankScore']`` comme côté Java.
        """
        if not docs:
            return []
        model = cls._get_model()
        pairs = [[query, d.text] for d in docs]
        scores = model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]

        ranked = sorted(zip(docs, scores, strict=True), key=lambda p: p[1], reverse=True)
        result: list[Doc] = []
        for doc, score in ranked[:top_k]:
            metadata = {**doc.metadata, "rerankScore": float(score)}
            result.append(Doc(id=doc.id, text=doc.text, metadata=metadata, score=doc.score))
        return result
