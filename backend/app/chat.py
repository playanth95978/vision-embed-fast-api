from typing import Iterator, Union

from ollama import chat, ChatResponse

AI_MODEL = 'qwen2.5:7b'


def get_chat_response(messages: list[dict[str, str]]) -> Union[ChatResponse, Iterator[ChatResponse]]:
    """Retrieve chat response from Ollama API.

    Args:
        messages (list[dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.

    Returns:
        Union[ChatResponse, Iterator[ChatResponse]]: Chat response or iterator of chat responses.
    """
    return chat(model=AI_MODEL, messages=messages, stream=True)