from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_asyncpg_url(url: str) -> str:
    """Map libpq sslmode/channel_binding query params to asyncpg-friendly ones."""
    if "+asyncpg" not in url:
        return url
    parsed = urlparse(url)
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

    app_env: str = "dev"
    log_level: str = "INFO"

    def sqlalchemy_url(self) -> str:
        """Pooled URL for the application."""
        return normalize_asyncpg_url(self.database_url)

    def alembic_url(self) -> str:
        """Prefer direct (non-pooler) URL for migrations."""
        return self.database_url_direct or self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
