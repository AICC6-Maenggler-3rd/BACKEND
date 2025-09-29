from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

DATABASE_URL = settings.postgresql_url

# 비동기 엔진 생성
engine = create_async_engine(DATABASE_URL, echo=True)

# 세션 생성
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# ORM Base 클래스
Base = declarative_base()


async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session