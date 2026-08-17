from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_asyncpg_url(url: str) -> str:
    """Force asyncpg dialect and map libpq SSL query params for asyncpg."""
    if url.startswith("postgresql+asyncpg://"):
        scheme_fixed = url
    elif url.startswith("postgresql+psycopg2://"):
        scheme_fixed = "postgresql+asyncpg://" + url.removeprefix(
            "postgresql+psycopg2://"
        )
    elif url.startswith("postgresql+psycopg://"):
        scheme_fixed = "postgresql+asyncpg://" + url.removeprefix(
            "postgresql+psycopg://"
        )
    elif url.startswith("postgres://"):
        scheme_fixed = "postgresql+asyncpg://" + url.removeprefix("postgres://")
    elif url.startswith("postgresql://"):
        scheme_fixed = "postgresql+asyncpg://" + url.removeprefix(
            "postgresql://"
        )
    else:
        scheme_fixed = url

    parsed = urlparse(scheme_fixed)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)
    if sslmode in {"require", "verify-ca", "verify-full", "prefer"}:
        query.setdefault("ssl", "require")
    return urlunparse(parsed._replace(query=urlencode(query)))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    database_url_direct: str | None = None

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    # Some gateways hang/break on dimensions= — set false to omit the param.
    embedding_send_dimensions: bool = True
    chat_model: str = "gpt-4o-mini"
    openai_timeout_sec: float = 30.0
    openai_max_retries: int = 1

    max_upload_size_mb: int = 25
    ocr_languages: str = "rus+eng"
    pdf_ocr_min_chars: int = 20
    libreoffice_path: str = "soffice"
    libreoffice_timeout_sec: int = 60
    tesseract_cmd: str | None = None

    chunk_size: int = 1500
    chunk_overlap: int = 200

    app_env: str = "dev"
    log_level: str = "INFO"

    def sqlalchemy_url(self) -> str:
        """Pooled URL for the application."""
        return normalize_asyncpg_url(self.database_url)

    def alembic_url(self) -> str:
        """Prefer direct (non-pooler) URL for migrations."""
        return self.database_url_direct or self.database_url

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
