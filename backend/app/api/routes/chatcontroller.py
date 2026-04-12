from collections.abc import AsyncIterable

import anyio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.chat import get_chat_response, get_chat_response_no_stream, get_translate_chat_query

router = APIRouter(prefix="/chat", tags=["chat"])
prompt = "You are a professional assistant. Your responses should be concise and helpful. {{query}} You have to translate the user's question in english."



from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

@router.post("/stream")
async def sse_items(request: ChatRequest) -> StreamingResponse:
    """
    Stream chat responses as SSE using standard FastAPI StreamingResponse.
    """

    async def event_generator() -> AsyncIterable[str]:
        # convert to list of dicts for ollama
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
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
