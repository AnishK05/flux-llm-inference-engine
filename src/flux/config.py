from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVED_MODEL_ID = "flux-qwen-0.5b"
HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL_ALIASES = frozenset({SERVED_MODEL_ID, HF_MODEL_ID})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLUX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = HF_MODEL_ID
    dtype: str = "fp32"
    device: str = "cpu"
    max_batch_size: int = 8
    max_waiting: int = 256
    block_size: int = 16
    num_kv_blocks: str = "auto"
    max_seq_len: int = 1024
    scheduler: str = "fcfs"
    intra_threads: str = "auto"
    redis_url: str = "redis://localhost:6379/0"
    enable_redis: bool = False
    rate_limit_per_min: int = 120
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    load_model: bool = True
    max_new_tokens_cap: int = 64
    # naive | cached | queued | continuous. Product default is the Phase 5 loop.
    serve_engine: str = "continuous"

    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
