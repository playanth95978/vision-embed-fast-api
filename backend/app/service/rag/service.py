"""Orchestration du pipeline RAG "merged" — portage de ``ChatService.chatMerged``.

Séquence : rewrite -> recherche hybride -> RRF -> rerank -> contexte (+ fallback) ->
system prompt -> streaming LLM, encadré par les events SSE start / token* / sources / end.

La partie retrieval est synchrone (Session SQLModel + modèle de reranking bloquants) et
s'exécute dans un threadpool ; la génération LLM est asynchrone (streaming non bloquant).
"""

import uuid
from collections.abc import AsyncIterator

import anyio
from sqlmodel import Session

from app.models import ChatEvent, SourceInfo
from app.service.rag import llm
from app.service.rag.prompts import build_system_prompt
from app.service.rag.reranker import Reranker
from app.service.rag.retrieval import (
    Retriever,
    extract_sources,
    format_document_with_metadata,
    reciprocal_rank_fusion,
)
from app.service.rag.sources import (
    FALLBACK_RERANK_TOP_K,
    NO_CONTEXT,
    PER_SOURCE_LIMIT,
    RERANK_TOP_K,
    RRF_TOP_K,
    SOURCES,
)

# Regroupe les deltas Mistral avant émission (nod à bufferTimeout(20, 100ms) WebFlux).
_TOKEN_BUFFER_SIZE = 5


class RagService:
    def __init__(self, session: Session) -> None:
        self._retriever = Retriever(session)

    # -- Retrieval (synchrone, exécuté en threadpool) ------------------------

    def _retrieve_context_with_sources(self, query: str) -> tuple[str, list[SourceInfo]]:
        rewritten = llm.rewrite_query(query, None)

        rankings = self._retriever.hybrid_search(rewritten)
        rrf_docs = reciprocal_rank_fusion(rankings, RRF_TOP_K)
        reranked = Reranker.rerank(query, rrf_docs, RERANK_TOP_K)

        sources = extract_sources(reranked)
        context = (
            "\n\n".join(format_document_with_metadata(d) for d in reranked[:RERANK_TOP_K])
            if reranked
            else NO_CONTEXT
        )

        merged = f"--- SOURCE: MERGED ---\n{context}\n\n" if context != NO_CONTEXT else ""
        final_context = merged if merged else NO_CONTEXT
        final_context = self._retrieve_fallback_context(final_context, rewritten, merged, sources)
        return final_context, sources

    def _retrieve_fallback_context(
        self, final_context: str, rewritten_query: str, merged: str, sources: list[SourceInfo]
    ) -> str:
        """Fallback multi-sources si aucun contexte fusionné (ChatService.retrieveFallbackContext)."""
        if final_context != NO_CONTEXT:
            return final_context
        acc = merged
        for src in SOURCES:
            docs = self._retriever.fetch_documents(src, rewritten_query, PER_SOURCE_LIMIT)
            if not docs:
                continue
            reranked = Reranker.rerank(rewritten_query, docs, FALLBACK_RERANK_TOP_K)
            if not reranked:
                continue
            ctx = "\n\n".join(format_document_with_metadata(d) for d in reranked)
            acc += f"--- SOURCE: {src.data_type} ---\n{ctx}\n\n"
            sources.extend(extract_sources(reranked))
        return acc if acc else NO_CONTEXT

    # -- Orchestration (async) -----------------------------------------------

    async def chat_merged(self, query: str, conversation_id: str | None) -> AsyncIterator[ChatEvent]:
        effective_id = conversation_id or str(uuid.uuid4())

        # Retrieval bloquant -> threadpool pour ne pas geler l'event loop.
        final_context, sources = await anyio.to_thread.run_sync(
            self._retrieve_context_with_sources, query
        )
        system_prompt = build_system_prompt(final_context)
        sources_json = SourceInfo.dump_list(sources)

        yield ChatEvent(id=effective_id, type="start")

        buffer: list[str] = []
        async for token in llm.stream(system_prompt, query):
            buffer.append(token)
            if len(buffer) >= _TOKEN_BUFFER_SIZE:
                yield ChatEvent(id=effective_id, type="token", content="".join(buffer))
                buffer.clear()
        if buffer:
            yield ChatEvent(id=effective_id, type="token", content="".join(buffer))

        yield ChatEvent(id=effective_id, type="sources", content=sources_json)
        yield ChatEvent(id=effective_id, type="end")
