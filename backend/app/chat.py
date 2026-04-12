from typing import Iterator, Union

import anyio
from ollama import chat, ChatResponse

from app.api.routes.chatcontroller import prompt

AI_MODEL = 'qwen2.5:7b'


def get_chat_response(messages: list[dict[str, str]]) -> Union[ChatResponse, Iterator[ChatResponse]]:
    """Retrieve chat response from Ollama API.

    Args:
        messages (list[dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.

    Returns:
        Union[ChatResponse, Iterator[ChatResponse]]: Chat response or iterator of chat responses.
    """
    return chat(model=AI_MODEL, messages=messages, stream=True)

def get_chat_response_no_stream(messages: list[dict[str, str]]) -> ChatResponse:
    """Retrieve chat response from Ollama API.

    Args:
        messages (list[dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.

    Returns:
        Union[ChatResponse, Iterator[ChatResponse]]: Chat response or iterator of chat responses.
    """
    return chat(model=AI_MODEL, messages=messages, stream=False)


async def get_translate_chat_query(q: str | None = None) -> dict[str, str]:
    """
    Get a direct chat response for translation query.
    """
    if not q:
        return {"response": ""}

    full_prompt = prompt.replace('{{query}}', q)
    messages = [{"role": "user", "content": full_prompt}]

    # Using anyio.to_thread.run_sync for non-blocking execution of sync Ollama call
    response = await anyio.to_thread.run_sync(get_chat_response_no_stream, messages)

    # Extract content from ChatResponse
    content = ""
    if isinstance(response, dict):
        content = response.get("message", {}).get("content", "")
    else:
        # If it's the Ollama object
        content = getattr(response.message, 'content', '')

    return {"response": content}