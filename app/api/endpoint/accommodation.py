from fastapi import APIRouter, Response, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from app.auth.dependencies import get_current_user
from app.repositories import accommodationdb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
router = APIRouter()

@router.get("/list")
async def get_accommodation_list(page: int = 1, limit: int = 30, lat: float = -1, lng: float = -1, radius: float = -1, query: str = "", db: AsyncSession = Depends(get_db)):
    try:
        accommodation_list = await accommodationdb.get_accommodation_list(db, page, limit, lat, lng, radius, query)
        return accommodation_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{accommodation_id}")
async def get_accommodation(accommodation_id: int, db: AsyncSession = Depends(get_db)):
    try:
        accommodation = await accommodationdb.get_accommodation(db, accommodation_id)
        return accommodation
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))