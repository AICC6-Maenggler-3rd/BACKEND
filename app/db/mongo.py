from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings

client: AsyncIOMotorClient = None
db: AsyncIOMotorDatabase = None

async def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    print("✅ MongoDB connected")

    # 세션 TTL 인덱스 생성 (expires_at 기준으로 자동 삭제)
    await db.sessions.create_index("expires_at", expireAfterSeconds=0)
    print("✅ MongoDB connected")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("✅ MongoDB connection closed")