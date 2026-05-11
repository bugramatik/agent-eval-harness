"""Settings read from the environment, with defaults that work locally."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the harness.

    Every value can be overridden with an environment variable of the same
    name in upper case, or through a local .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://harness:harness@localhost:5432/harness"
    openai_api_key: str = ""
    agent_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    max_steps: int = 8
    request_timeout: float = 60.0
    scenario_dir: str = "scenarios"

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, built once."""
    return Settings()
