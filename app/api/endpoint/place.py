from fastapi import APIRouter, Response, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from app.auth.dependencies import get_current_user
from app.repositories import placedb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.place_schema import PlaceListResponse
from app.services.place_service import search_place_rag
router = APIRouter()

@router.get("/search")
async def search_place(query: str, page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)) -> PlaceListResponse:
    try:
        place_list = await placedb.search_place(db, query, page, limit)
        return place_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search_rag")
async def search_place_by_rag(query: str, count: int = 30, db: AsyncSession = Depends(get_db)) -> PlaceListResponse:
    try:
        place_list = await search_place_rag(db, query, count)
        return place_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/address")
async def get_place_list_by_address(address: str, page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)) -> PlaceListResponse:
    try:
        print(address, page, limit)
        place_list = await placedb.get_place_list_by_address(db,address, page, limit)
        return place_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/radius")
async def get_place_list_by_radius(lat: float, lng: float, radius: float, page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)) -> PlaceListResponse:
    try:
        place_list = await placedb.get_place_list_by_address_and_radius(db, lat, lng, radius, page, limit)
        return place_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def get_place_list(page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)) -> PlaceListResponse:
    try:
        place_list = await placedb.get_place_list(db, page, limit)
        return place_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/category")
async def get_place_list_by_category(category_id: int, page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)) -> PlaceListResponse:
    """카테고리별 장소 목록 조회"""
    try:
        place_list = await placedb.get_place_list_by_category(db, category_id, page, limit)
        return place_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{place_id}")
async def get_place(place_id: int, db: AsyncSession = Depends(get_db)):
    try:
        place = await placedb.get_place(db, place_id)
        return place
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insta-nicknames")
async def get_insta_nicknames(db: AsyncSession = Depends(get_db)):
    """인스타그램 닉네임 목록 조회"""
    try:
        nicknames = await placedb.get_insta_nicknames(db)
        return {"insta_nicknames": nicknames}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/by-instagram/{insta_nickname}")
async def get_places_by_instagram(insta_nickname: str, page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)) -> PlaceListResponse:
    """특정 인스타그램 닉네임으로 장소 조회"""
    try:
        place_list = await placedb.get_places_by_insta_nickname(db, insta_nickname, page, limit)
        return place_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))