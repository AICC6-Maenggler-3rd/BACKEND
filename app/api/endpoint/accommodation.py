from fastapi import APIRouter, Response, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from app.auth.dependencies import get_current_user
from app.repositories import accommodationdb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional

# Pydantic 모델들
class AccommodationCreate(BaseModel):
    name: str
    address: str
    address_la: float
    address_lo: float
    type: str
    phone: str
    image_url: Optional[str] = None

class AccommodationUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    address_la: Optional[float] = None
    address_lo: Optional[float] = None
    type: Optional[str] = None
    phone: Optional[str] = None
    image_url: Optional[str] = None

router = APIRouter()

@router.get("/search")
async def search_accommodation(query: str, page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)):
    try:
        accommodation_list = await accommodationdb.search_accommodation(db, query, page, limit)
        return accommodation_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def get_accommodation_list(page: int = 1, limit: int = 30, lat: float = -1, lng: float = -1, radius: float = -1, db: AsyncSession = Depends(get_db)):
    try:
        accommodation_list = await accommodationdb.get_accommodation_list(db, page, limit, lat, lng, radius)
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

@router.post("/")
async def create_accommodation(accommodation_data: AccommodationCreate, db: AsyncSession = Depends(get_db)):
    try:
        accommodation = await accommodationdb.create_accommodation(db, accommodation_data)
        return {"message": "숙소가 성공적으로 생성되었습니다.", "accommodation": accommodation}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{accommodation_id}")
async def update_accommodation(accommodation_id: int, accommodation_data: AccommodationUpdate, db: AsyncSession = Depends(get_db)):
    try:
        print(f"숙소 수정 요청 - ID: {accommodation_id}, 데이터: {accommodation_data}")
        accommodation = await accommodationdb.update_accommodation(db, accommodation_id, accommodation_data)
        if not accommodation:
            raise HTTPException(status_code=404, detail="숙소를 찾을 수 없습니다.")
        print(f"숙소 수정 성공 - ID: {accommodation_id}")
        return {"message": "숙소가 성공적으로 수정되었습니다.", "accommodation": accommodation}
    except HTTPException:
        raise
    except Exception as e:
        print(f"숙소 수정 에러 - ID: {accommodation_id}, 에러: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{accommodation_id}")
async def delete_accommodation(accommodation_id: int, db: AsyncSession = Depends(get_db)):
    try:
        success = await accommodationdb.delete_accommodation(db, accommodation_id)
        if not success:
            raise HTTPException(status_code=404, detail="숙소를 찾을 수 없습니다.")
        return {"message": "숙소가 성공적으로 삭제되었습니다."}
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))