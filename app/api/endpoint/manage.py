from fastapi import APIRouter, Depends
from app.repositories import userdb
from app.db.postgresql import get_db,place_db
from sqlalchemy.ext.asyncio import AsyncSession
router = APIRouter()

@router.get("/dashboard")
async def get_users_count(db: AsyncSession = Depends(get_db)):
    users = await userdb.get_users(db)
    users_count = len(users)
    places = await place_db.get_places(db)
    places_count = len(places)
    return {"users_count": users_count, "places_count": places_count}