from collections.abc import AsyncIterable
import anyio
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat import get_chat_response

router = APIRouter(prefix="/items", tags=["chat"])


class Item(BaseModel):
    name: str
    description: str | None


items = [
    Item(name="Plumbus", description="A multi-purpose household device."),
    Item(name="Portal Gun", description="A portal opening device."),
    Item(name="Meeseeks Box", description="A box that summons a Meeseeks."),
]


@router.post("/stream")
async def sse_items(msg: list[dict[str, str]]) -> StreamingResponse:
    """
    Stream chat responses as SSE using standard FastAPI StreamingResponse.
    """
    async def event_generator() -> AsyncIterable[str]:
        # get_chat_response is a synchronous iterator (Ollama library default)
        # We run it in a loop and yield chunks. 
        # Using anyio.to_thread.run_sync for non-blocking execution of sync iterator
        response_iterator = await anyio.to_thread.run_sync(get_chat_response, msg)
        
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

# Code below omitted 👇
