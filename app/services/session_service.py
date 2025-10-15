import uuid
from datetime import datetime, timedelta

SESSION_EXPIRE_SECONDS = 1800

async def create_session(user_id: int):
    from app.db.mongo import db
    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(seconds=SESSION_EXPIRE_SECONDS)

    await db.sessions.insert_one({
        "_id": session_id,
        "user_id": user_id,
        "expires_at": expires_at
    })

    return session_id

async def get_session(session_id: str):
    from app.db.mongo import db
    session = await db.sessions.find_one({"_id": session_id})
    if not session:
        return None
    return session

async def delete_session(session_id: str):
    from app.db.mongo import db
    await db.sessions.delete_one({"_id": session_id})

async def refresh_session(session_id: str) -> int:
    from app.db.mongo import db
    new_expiry_time = datetime.utcnow() + timedelta(seconds=SESSION_EXPIRE_SECONDS)
    await db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"expires_at": new_expiry_time}}
    )
    return SESSION_EXPIRE_SECONDS

async def delete_user_sessions(user_id: int):
    from app.db.mongo import db as mongo
    await mongo.sessions.delete_many({"user_id": user_id})