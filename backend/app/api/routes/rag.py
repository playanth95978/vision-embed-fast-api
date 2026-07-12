"""Endpoint RAG "merged" — équivalent de ``ChatController.mergeVectorChat`` (Spring Boot).

Flux Server-Sent Events dont chaque ``data`` est un ``ChatEvent`` JSON (start / token* /
sources / end), consommable tel quel par le front.
"""

from collections.abc import AsyncIterable

import anyio
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, SessionDep
from app.models import IngestResponse, IngestTextRequest
from app.service.rag import RagService
from app.service.rag.ingestion import DocumentIngestionService

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


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    session: SessionDep,
    current_user: CurrentUser,
    body: IngestTextRequest,
) -> IngestResponse:
    """Ingeste un document texte (chunking + embeddings) dans la table vectorielle."""
    service = DocumentIngestionService(session)
    chunks = await anyio.to_thread.run_sync(
        lambda: service.ingest_text(
            content=body.content,
            title=body.title,
            url=body.url,
            source=body.source,
            doc_id=body.docId,
        )
    )
    return IngestResponse(chunks=chunks, source=body.source.upper())


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    session: SessionDep,
    current_user: CurrentUser,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    url: str | None = Form(None),
    source: str = Form("CONFLUENCE"),
) -> IngestResponse:
    """Ingeste un fichier uploadé (.pdf, .txt, .md…) dans la table vectorielle."""
    data = await file.read()
    filename = file.filename or "document"
    service = DocumentIngestionService(session)
    chunks = await anyio.to_thread.run_sync(
        lambda: service.ingest_file(
            filename=filename,
            data=data,
            title=title,
            url=url,
            source=source,
        )
    )
    return IngestResponse(chunks=chunks, source=source.upper())
