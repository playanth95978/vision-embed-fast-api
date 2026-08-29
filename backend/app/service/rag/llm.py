"""LLM du pipeline RAG : réécriture de requête (bloquant) + génération streaming (async).

Deux fournisseurs, sélectionnés par ``settings.RAG_LLM_PROVIDER`` :
- ``"ollama"`` (défaut) : 100% local, modèle ``settings.RAG_LLM_MODEL`` (ex. ``mistral:latest``).
- ``"mistral"`` : API Mistral cloud (nécessite ``MISTRAL_API_KEY``).

Les appels de réécriture sont bloquants (à exécuter en threadpool) ; le streaming est async.
"""

from collections.abc import AsyncIterator

from ollama import AsyncClient, Client

from app.core.config import settings
from app.service.rag.prompts import build_rewrite_prompt

_ollama = Client(host=settings.OLLAMA_BASE_URL)
_mistral = None  # client Mistral construit paresseusement (évite d'exiger une clé si inutilisé)


def _get_mistral():
    global _mistral
    if _mistral is None:
        from mistralai.client import Mistral

        _mistral = Mistral(api_key=settings.MISTRAL_API_KEY)
    return _mistral


# -- Réécriture de requête (bloquant) ----------------------------------------


def rewrite_query(query: str, data_type: str | None) -> str:
    """Réécriture de requête (QueryRewriterService.rewrite), fallback sur l'original si échec.

    Court-circuits fidèles au Java : requête vide -> inchangée ; < 10 caractères -> inchangée.
    """
    if not query or not query.strip():
        return query
    trimmed = query.strip()
    if len(trimmed) < 10:
        return trimmed
    prompt = build_rewrite_prompt(trimmed, data_type)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": trimmed},
    ]
    try:
        if settings.RAG_LLM_PROVIDER == "ollama":
            resp = _ollama.chat(model=settings.RAG_LLM_MODEL, messages=messages, stream=False)
            rewritten = (resp["message"]["content"] or "").strip()
        else:
            resp = _get_mistral().chat.complete(model=settings.MISTRAL_MODEL, messages=messages)
            rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or trimmed
    except Exception:
        return trimmed


# -- Génération en streaming (async) -----------------------------------------


async def stream(system_prompt: str, user_query: str) -> AsyncIterator[str]:
    """Stream des tokens de la réponse (équiv. chatClient.prompt().stream().content())."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    if settings.RAG_LLM_PROVIDER == "ollama":
        aclient = AsyncClient(host=settings.OLLAMA_BASE_URL)
        async for part in await aclient.chat(model=settings.RAG_LLM_MODEL, messages=messages, stream=True):
            delta = part["message"]["content"]
            if delta:
                yield delta
    else:
        response = await _get_mistral().chat.stream_async(model=settings.MISTRAL_MODEL, messages=messages)
        async for chunk in response:
            delta = chunk.data.choices[0].delta.content
            if delta:
                yield delta
