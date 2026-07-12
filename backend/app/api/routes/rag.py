"""Endpoint RAG "merged" — équivalent de ``ChatController.mergeVectorChat`` (Spring Boot).

Flux Server-Sent Events dont chaque ``data`` est un ``ChatEvent`` JSON (start / token* /
sources / end), consommable tel quel par le front.
"""

from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import SessionDep
from app.service.rag import RagService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/merge")
async def merge_vector_chat(
    session: SessionDep,
    message: str,
    conversationId: str | None = None,
) -> StreamingResponse:
    """Répond à ``message`` via le pipeline RAG hybride, en streaming SSE."""
    service = RagService(session)

    async def event_generator() -> AsyncIterable[str]:
        async for event in service.chat_merged(message, conversationId):
            # data = JSON du ChatEvent, comme le sérialise Spring WebFlux.
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
