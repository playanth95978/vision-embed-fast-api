"""Structures internes du pipeline RAG (non exposées par l'API)."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Doc:
    """Équivalent léger de ``org.springframework.ai.document.Document``."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
