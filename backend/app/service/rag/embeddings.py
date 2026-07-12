"""Embeddings via Ollama.

Réutilise le client ``ollama`` déjà présent dans le projet (cf. ``app.chat``), au lieu
d'introduire un client HTTP dédié. Appels synchrones et bloquants : à exécuter dans un
threadpool depuis le code async (``anyio.to_thread.run_sync``).
"""

from ollama import Client

from app.core.config import settings

_client = Client(host=settings.OLLAMA_BASE_URL)


def embed(text: str, model: str) -> list[float]:
    """Génère l'embedding d'un texte avec le modèle Ollama donné.

    Équivalent de ``OllamaEmbeddingModel`` côté Spring AI.
    """
    response = _client.embeddings(model=model, prompt=text)
    return list(response["embedding"])
