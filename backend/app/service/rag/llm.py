"""Client Mistral : réécriture de requête (bloquant) + génération en streaming (async)."""

from collections.abc import AsyncIterator

from mistralai import Mistral

from app.core.config import settings
from app.service.rag.prompts import build_rewrite_prompt

_client = Mistral(api_key=settings.MISTRAL_API_KEY)


def rewrite_query(query: str, data_type: str | None) -> str:
    """Réécriture de requête (QueryRewriterService.rewrite), fallback sur l'original si échec.

    Court-circuits fidèles au Java : requête vide -> inchangée ; < 10 caractères -> inchangée.
    Appel bloquant (à exécuter en threadpool depuis du code async).
    """
    if not query or not query.strip():
        return query
    trimmed = query.strip()
    if len(trimmed) < 10:
        return trimmed
    prompt = build_rewrite_prompt(trimmed, data_type)
    try:
        resp = _client.chat.complete(
            model=settings.MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": trimmed},
            ],
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or trimmed
    except Exception:
        return trimmed


async def stream(system_prompt: str, user_query: str) -> AsyncIterator[str]:
    """Stream des tokens de la réponse (équiv. chatClient.prompt().stream().content())."""
    response = await _client.chat.stream_async(
        model=settings.MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )
    async for chunk in response:
        delta = chunk.data.choices[0].delta.content
        if delta:
            yield delta
