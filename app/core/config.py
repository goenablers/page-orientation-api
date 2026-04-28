import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Application settings (from environment)."""

    app_env: str = "development"
    port: int = 5000

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def is_staging(self) -> bool:
        return self.app_env.lower() == "staging"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        port=int(os.getenv("PORT", "5000")),
    )
