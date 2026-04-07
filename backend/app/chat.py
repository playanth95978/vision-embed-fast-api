from typing import Iterator, Union

from ollama import chat, ChatResponse

AI_MODEL = 'gemma3'

stream = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)


def get_chat_response(messages: list[dict[str, str]]) -> Union[ChatResponse, Iterator[ChatResponse]]:
    """Retrieve chat response from Ollama API.

    Args:
        messages (list[dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.

    Returns:
        Union[ChatResponse, Iterator[ChatResponse]]: Chat response or iterator of chat responses.
    """
    return chat(model='gemma3', messages=messages, stream=True)