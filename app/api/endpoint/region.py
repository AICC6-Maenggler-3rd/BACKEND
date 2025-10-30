from fastapi import APIRouter, HTTPException, Depends
from app.repositories import regiondb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
router = APIRouter()

@router.get("/list")
async def get_all_region(db: AsyncSession = Depends(get_db)):
    print("==================================================")
    try:
        region_list = await regiondb.get_all_region(db)
        return region_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))