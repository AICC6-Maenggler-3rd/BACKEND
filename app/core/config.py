from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml" / "artifacts"
POI_MODEL_PATH = MODEL_DIR / "nextpoi_sasrec.pt"
PLACES_PATH = MODEL_DIR / "places_with_ids.csv"
class Settings(BaseSettings):
    MONGO_URI: str
    MONGO_DB_NAME: str
    postgresql_url: str
    openai_api_key: str
    claude_api_key: str
    langchain_tracing_v2: bool
    langchain_endpoint: str
    langchain_api_key: str
    langchain_project: str
    google_client_id: str
    google_client_secret: str
    google_redirect_uri: str
    naver_client_id: str
    naver_client_secret: str
    naver_redirect_uri: str
    kakao_client_id: str
    kakao_client_secret: str
    kakao_redirect_uri: str
    ncp_client_id:str
    ncp_client_secret:str
    tmap_api_key:str
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()