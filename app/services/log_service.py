from app.schemas.log_schema import UserLogSchema

async def create_log(
    user_id: str,
    action: str,
    ip: str,
    user_agent: str,
    status_code: int = 200,
    extra: dict | None = None
):
    from app.db.mongo import db
    if db is None:
        print("⚠️ MongoDB not initialized")
        return None

    # Pydantic Schema로 데이터 검증
    log = UserLogSchema(
        user_id=user_id,
        action=action,
        ip=ip,
        user_agent=user_agent,
        status_code=status_code,
        extra=extra
    )

    # DB에 저장
    result = await db["user_logs"].insert_one(log.dict())
    return result.inserted_id