
from app.schemas.activity_log_schema import ActivityLogBase

async def create_activity_log(log: ActivityLogBase):
    from app.db.mongo import db as mongo_db
    if mongo_db is None:
        raise RuntimeError("MongoDB not connected")

    result = await mongo_db["user_activity_logs"].insert_one(log.dict())
    return str(result.inserted_id)