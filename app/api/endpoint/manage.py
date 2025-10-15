from fastapi import APIRouter, Depends
from app.repositories import userdb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
router = APIRouter()

@router.get("/users/count")
async def get_users_count(db: AsyncSession = Depends(get_db)):
    users = await userdb.get_users(db)
    return {"count": len(users)}