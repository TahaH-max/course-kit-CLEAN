"""Settings read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    db_path: str = field(default_factory=lambda: _env("COURSEKIT_DB", str(ROOT / "coursekit.db")))
    sample_dir: Path = ROOT / "data" / "sample_syllabi"
    # Pin "today" for demos and tests (YYYY-MM-DD); blank means the real date.
    today_override: str = field(default_factory=lambda: _env("COURSEKIT_TODAY"))

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto").lower())
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_chat_model: str = field(default_factory=lambda: _env("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    azure_endpoint: str = field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT"))
    azure_api_key: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY"))
    azure_api_version: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    azure_chat_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT"))
    azure_embedding_deployment: str = field(
        default_factory=lambda: _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))

    def resolved_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.azure_endpoint and self.azure_api_key and self.azure_chat_deployment:
            return "azure"
        if self.openai_api_key:
            return "openai"
        return "offline"


def get_settings() -> Settings:
    return Settings()
