from collections.abc import AsyncIterable

import anyio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.chat import get_chat_response, get_chat_response_no_stream, get_translate_chat_query

router = APIRouter(prefix="/chat", tags=["chat"])
prompt = "You are a professional assistant. Your responses should be concise and helpful. {{query}} You have to translate the user's question in english."



@router.post("/stream")
async def sse_items(q: str | None = None) -> StreamingResponse:
    """
    Stream chat responses as SSE using standard FastAPI StreamingResponse.
    """

    async def event_generator() -> AsyncIterable[str]:
        # get_chat_response is a synchronous iterator (Ollama library default)
        # We run it in a loop and yield chunks. 
        # Using anyio.to_thread.run_sync for non-blocking execution of sync iterator
        messages = [{"role": "user", "content": q}] if q else []
        response_iterator = await anyio.to_thread.run_sync(get_chat_response, messages)

        for chunk in response_iterator:
            # Yielding the content in SSE format
            # chunk is expected to be a ChatResponse or dict
            content = ""
            if isinstance(chunk, dict):
                content = chunk.get("message", {}).get("content", "")
            else:
                # If it's the Ollama object
                content = getattr(chunk.message, 'content', '')

            if content:
                yield f"data: {content}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/translate")
async def translate_chat_query(q: str | None = None) -> dict[str, str]:
    return get_translate_chat_query(q)
# Code below omitted 👇
