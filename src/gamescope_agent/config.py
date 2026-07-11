"""Environment-based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    project_root: Path
    data_dir: Path
    chroma_dir: Path
    collection_name: str
    openai_api_key: str | None
    tavily_api_key: str | None
    openai_model: str
    embedding_model: str
    openai_base_url: str | None

    @classmethod
    def load(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env")

        def resolve_path(name: str, default: str) -> Path:
            raw = Path(os.getenv(name, default))
            return raw if raw.is_absolute() else root / raw

        def optional_env(name: str) -> str | None:
            """Return a stripped optional value and fully unset blank variables.

            Some SDKs distinguish between an absent environment variable and an
            environment variable containing an empty string. Removing blank
            variables lets those SDKs apply their documented defaults.
            """

            raw = os.getenv(name)
            if raw is None:
                return None

            value = raw.strip()
            if value:
                return value

            os.environ.pop(name, None)
            return None

        base_url = optional_env("OPENAI_BASE_URL")
        if base_url:
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise RuntimeError(
                    "OPENAI_BASE_URL must be a complete HTTP(S) URL, for example "
                    "'https://api.openai.com/v1', or it must be omitted entirely."
                )

        return cls(
            project_root=root,
            data_dir=resolve_path("GAMESCOPE_DATA_DIR", "data/games"),
            chroma_dir=resolve_path("GAMESCOPE_CHROMA_DIR", ".chroma"),
            collection_name=os.getenv("GAMESCOPE_COLLECTION", "gamescope"),
            openai_api_key=optional_env("OPENAI_API_KEY"),
            tavily_api_key=optional_env("TAVILY_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()
            or "gpt-5-mini",
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ).strip()
            or "text-embedding-3-small",
            openai_base_url=base_url,
        )

    def require_openai(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Copy .env.example to .env "
                "and add a key."
            )
        return self.openai_api_key
