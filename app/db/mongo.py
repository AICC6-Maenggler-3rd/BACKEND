from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings

client: AsyncIOMotorClient = None
db: AsyncIOMotorDatabase = None

def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    print("✅ MongoDB connected")

def close_mongo_connection():
    global client
    if client:
        client.close()
        print("✅ MongoDB connection closed")