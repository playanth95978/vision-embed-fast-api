"""Reranker cross-encoder ONNX — portage de ``OnnxRerankerService`` (backend Java).

Le modèle et ses réglages viennent de job-search-ai, où ils ont été choisis après mesure :
``bge-reranker-base`` quantifié INT8 (278 M, multilingue), tronqué à 128 tokens, toutes les
inférences sur tous les cœurs. Voir ``RerankerModelComparisonBenchmark`` et
``RerankerMaxLengthBenchmark`` côté Java, et ``scripts/bench_reranker.py`` ici.

Le modèle (~280 Mo) n'est pas versionné : il est téléchargé au premier chargement s'il est
absent, comme le fait ``RerankerModelProvisioner`` côté Java.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.service.rag.schemas import Doc

logger = logging.getLogger(__name__)

# Racine du backend (…/backend), pour résoudre les chemins relatifs de la configuration.
_BASE_DIR = Path(__file__).resolve().parents[3]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _BASE_DIR / candidate


def _ensure_present(path: Path, url: str, label: str) -> None:
    """Télécharge le fichier s'il manque (équivalent de ``RerankerModelProvisioner``)."""
    if path.exists() and path.stat().st_size > 0:
        return
    import httpx

    logger.info("%s absent (%s) — téléchargement depuis %s", label, path, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Écriture dans un fichier temporaire puis renommage : un téléchargement interrompu ne
    # laisse pas un modèle tronqué qui échouerait de façon illisible au chargement suivant.
    temporary = path.with_suffix(path.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
    temporary.replace(path)
    logger.info("%s installé (%.0f Mo)", label, path.stat().st_size / 1e6)


class Reranker:
    """Singleton paresseux autour du cross-encoder ONNX."""

    _session: Any = None
    _tokenizer: Any = None
    _lock = threading.Lock()

    @classmethod
    def _load(cls) -> tuple[Any, Any]:
        if cls._session is not None:
            return cls._session, cls._tokenizer

        # Double vérification : plusieurs requêtes concurrentes ne doivent pas ouvrir deux
        # sessions ONNX (plusieurs centaines de Mo chacune).
        with cls._lock:
            if cls._session is not None:
                return cls._session, cls._tokenizer

            # Imports différés : le chargement coûte quelques secondes et n'a pas à peser
            # sur le démarrage de l'API.
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = _resolve(settings.RERANKER_ONNX_PATH)
            tokenizer_path = _resolve(settings.RERANKER_TOKENIZER_PATH)
            _ensure_present(model_path, settings.RERANKER_MODEL_URL, "modèle de reranking")
            _ensure_present(tokenizer_path, settings.RERANKER_TOKENIZER_URL, "tokenizer de reranking")

            options = ort.SessionOptions()
            threads = settings.RERANKER_INTRA_OP_THREADS or (os.cpu_count() or 1)
            options.intra_op_num_threads = threads
            options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            # Padding/troncature fixes : le modèle Java alimente des tenseurs de taille
            # constante, on reproduit exactement la même forme d'entrée.
            tokenizer.enable_truncation(max_length=settings.RERANKER_MAX_LENGTH)
            tokenizer.enable_padding(length=settings.RERANKER_MAX_LENGTH)

            cls._session, cls._tokenizer = session, tokenizer
            logger.info(
                "Reranker ONNX chargé (%s, maxLength=%d, intraOpThreads=%d)",
                model_path.name,
                settings.RERANKER_MAX_LENGTH,
                threads,
            )
            return session, tokenizer

    @classmethod
    def score(cls, query: str, texts: list[str], max_length: int | None = None) -> list[float]:
        """Scores bruts (logits) du cross-encoder pour chaque paire (query, texte)."""
        if not texts:
            return []
        session, tokenizer = cls._load()

        if max_length is not None and max_length != settings.RERANKER_MAX_LENGTH:
            tokenizer.enable_truncation(max_length=max_length)
            tokenizer.enable_padding(length=max_length)

        encodings = tokenizer.encode_batch([(query, text) for text in texts])
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        # Sortie (batch, 1) : un logit de pertinence par paire.
        return [float(row[0]) for row in outputs[0]]

    @classmethod
    def rerank(cls, query: str, docs: list[Doc], top_k: int) -> list[Doc]:
        """Rerank puis troncature à ``top_k`` (équiv. ``RerankerService.rerank``).

        Le score est ramené en 0..1 par sigmoïde et stocké dans ``metadata['rerankScore']``,
        comme côté Java.
        """
        if not docs:
            return []
        if not query or not query.strip():
            return docs[:top_k]

        logits = cls.score(query, [d.text for d in docs])
        # Sigmoïde : les logits bruts n'ont pas d'échelle exploitable côté UI.
        scores = 1.0 / (1.0 + np.exp(-np.asarray(logits)))

        ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        result: list[Doc] = []
        for doc, score in ranked[:top_k]:
            metadata = {**doc.metadata, "rerankScore": float(score)}
            result.append(Doc(id=doc.id, text=doc.text, metadata=metadata, score=doc.score))
        return result
