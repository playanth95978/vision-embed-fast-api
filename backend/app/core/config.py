import secrets
import warnings
from typing import Annotated, Any, Literal

from pydantic import (
    AnyUrl,
    BeforeValidator,
    EmailStr,
    HttpUrl,
    PostgresDsn,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",") if i.strip()]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Use top level .env file (one level above ./backend/)
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore",
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    FRONTEND_HOST: str = "http://localhost:5173"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]

    PROJECT_NAME: str
    SENTRY_DSN: HttpUrl | None = None
    POSTGRES_SERVER: str
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""
    # Désactivé par défaut : l'écho SQL + le logging DEBUG ralentissent fortement
    # les migrations Alembic et le démarrage. À n'activer que pour du debug local.
    SQLALCHEMY_ECHO: bool = False

    # --- RAG (portage de l'endpoint "merged" Spring Boot) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Fournisseur LLM pour la génération : "ollama" (local, défaut) ou "mistral" (cloud).
    RAG_LLM_PROVIDER: Literal["ollama", "mistral"] = "ollama"
    # Modèle Ollama utilisé quand RAG_LLM_PROVIDER == "ollama".
    RAG_LLM_MODEL: str = "mistral:latest"
    # Mistral cloud (utilisé seulement si RAG_LLM_PROVIDER == "mistral") — clé via .env, jamais en dur.
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-medium-latest"
    # --- Reranking (réglages repris du backend Java, où ils ont été mesurés) ---
    # bge-reranker-base INT8 (278 M, multilingue) : 2,9x plus rapide que bge-reranker-v2-m3
    # pour la meilleure fidélité de classement des candidats évalués (RerankerModelComparisonBenchmark).
    # Modèle et tokenizer sont SOLIDAIRES : en changer un sans l'autre produit des scores absurdes
    # sans la moindre erreur au démarrage.
    RERANKER_ONNX_PATH: str = "models/rerankers/bge-base/model.onnx"
    RERANKER_TOKENIZER_PATH: str = "models/rerankers/bge-base/tokenizer.json"
    RERANKER_MODEL_URL: str = (
        "https://huggingface.co/onnx-community/bge-reranker-base-ONNX/resolve/main/onnx/model_int8.onnx"
    )
    RERANKER_TOKENIZER_URL: str = (
        "https://huggingface.co/onnx-community/bge-reranker-base-ONNX/resolve/main/tokenizer.json"
    )
    # Troncature en tokens. Mesuré sur bge-reranker-base : 128 au lieu de 256 rapporte ~1,8x
    # pour une corrélation de rangs de ~0,83. Ne pas descendre à 64, le top-1 s'effondre.
    RERANKER_MAX_LENGTH: int = 128
    # Threads ONNX par inférence ; 0 = tous les cœurs. Mesuré : brider dégrade débit ET latence.
    RERANKER_INTRA_OP_THREADS: int = 0
    # Ancien modèle torch (sentence-transformers), conservé pour le banc comparatif.
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+psycopg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_SERVER,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )

    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    SMTP_PORT: int = 587
    SMTP_HOST: str | None = None
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    EMAILS_FROM_EMAIL: EmailStr | None = None
    EMAILS_FROM_NAME: str | None = None

    @model_validator(mode="after")
    def _set_default_emails_from(self) -> Self:
        if not self.EMAILS_FROM_NAME:
            self.EMAILS_FROM_NAME = self.PROJECT_NAME
        return self

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48

    @computed_field  # type: ignore[prop-decorator]
    @property
    def emails_enabled(self) -> bool:
        return bool(self.SMTP_HOST and self.EMAILS_FROM_EMAIL)

    EMAIL_TEST_USER: EmailStr = "test@example.com"
    FIRST_SUPERUSER: EmailStr
    FIRST_SUPERUSER_PASSWORD: str

    def _check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == "changethis":
            message = (
                f'The value of {var_name} is "changethis", '
                "for security, please change it, at least for deployments."
            )
            if self.ENVIRONMENT == "local":
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        self._check_default_secret("SECRET_KEY", self.SECRET_KEY)
        self._check_default_secret("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD)
        self._check_default_secret(
            "FIRST_SUPERUSER_PASSWORD", self.FIRST_SUPERUSER_PASSWORD
        )

        return self


settings = Settings()  # type: ignore
