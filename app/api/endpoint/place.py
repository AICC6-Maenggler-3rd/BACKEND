from fastapi import APIRouter, Response, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from app.auth.dependencies import get_current_user
from app.repositories import placedb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
router = APIRouter()

@router.get("/list")
async def get_place_list(page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)):
    try:
        place_list = await placedb.get_place_list(db, page, limit)
        return place_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{place_id}")
async def get_place(place_id: int, db: AsyncSession = Depends(get_db)):
    try:
        place = await placedb.get_place(db, place_id)
        return place
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))