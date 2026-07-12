"""Reranker cross-encoder (BAAI/bge-reranker-v2-m3).

Le backend Java exécute ce modèle en ONNX (DJL + onnxruntime, ~100 lignes de plomberie).
En Python il se charge via le ``CrossEncoder`` de sentence-transformers (compatible
transformers 5.x, contrairement à FlagEmbedding). On suit le même pattern Singleton
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
            from sentence_transformers import CrossEncoder

            cls._model = CrossEncoder(settings.RERANKER_MODEL)
        return cls._model

    @classmethod
    def rerank(cls, query: str, docs: list[Doc], top_k: int) -> list[Doc]:
        """Rerank puis troncature à ``top_k`` (équiv. RerankerService.rerank).

        Le score (sigmoïde -> 0..1) est stocké dans ``metadata['rerankScore']`` comme côté Java.
        """
        if not docs:
            return []
        import torch

        model = cls._get_model()
        pairs = [(query, d.text) for d in docs]
        # bge-reranker : sortie logit unique -> sigmoïde pour un score de pertinence 0..1.
        scores = model.predict(pairs, activation_fn=torch.nn.Sigmoid())

        ranked = sorted(zip(docs, scores, strict=True), key=lambda p: p[1], reverse=True)
        result: list[Doc] = []
        for doc, score in ranked[:top_k]:
            metadata = {**doc.metadata, "rerankScore": float(score)}
            result.append(Doc(id=doc.id, text=doc.text, metadata=metadata, score=doc.score))
        return result
