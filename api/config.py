from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    MODEL_WEIGHTS_PATH: str = ""
    EMBEDDING_DIM: int = 256
    SIMILARITY_THRESHOLD: float = 0.85

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
