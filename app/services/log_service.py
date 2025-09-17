from datetime import datetime

async def create_log(user_id: str, action: str, ip: str, user_agent: str, status_code: int = 200, extra: dict = None):
    # 내부에서 import
    from app.db.mongo import db as mongo_db
    if mongo_db is None:
        print("⚠️ MongoDB not initialized, skipping log")
        return None

    log = {
        "user_id": user_id,
        "action": action,
        "ip": ip,
        "user_agent": user_agent,
        "status_code": status_code,
        "timestamp": datetime.utcnow(),
        "extra": extra or {}
    }
    result = await mongo_db["user_logs"].insert_one(log)
    return result.inserted_id