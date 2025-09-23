from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    MONGO_DB_NAME: str
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
    class Config:
        env_file = ".env"

settings = Settings()