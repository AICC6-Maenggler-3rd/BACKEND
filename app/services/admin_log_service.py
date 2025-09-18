
from app.schemas.admin_log_schema import AdminLogBase

async def create_admin_log(log: AdminLogBase):
    from app.db.mongo import db
    if db is None:
        raise RuntimeError("MongoDB not connected")

    result = await db["admin_logs"].insert_one(log.dict())
    return str(result.inserted_id)