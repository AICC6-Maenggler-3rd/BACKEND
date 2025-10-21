from app.services import itinerary_service
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, Request
from app.schemas.postgre_schema import ItineraryCreate, ItineraryGenerate, ItineraryResponse, ItineraryListResponse, ItinerarySchema, ItineraryCreateRequest
from sqlalchemy import func, select
from app.models.postgre_model import Itinerary
from app.auth.dependencies import get_current_user

router = APIRouter()

@router.post("/generate")
async def generate_itinerary(req: Request, db: AsyncSession = Depends(get_db)) -> ItineraryResponse:
    body = await req.json()
    print(body)
    generate_itinerary_request = ItineraryGenerate(**body)
    try:
        itinerary = await itinerary_service.generate_itinerary(db, generate_itinerary_request)
        return itinerary
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/{itinerary_id}")
async def get_itinerary(itinerary_id: int, db: AsyncSession = Depends(get_db)) -> ItineraryResponse:
    try:
        print("🟢 [DEBUG] get_itinerary called with id:", itinerary_id)
        itinerary = await itinerary_service.get_itinerary(db, itinerary_id)
        return itinerary
    except Exception as e:
        # print(e)
        print("❌ [ERROR] get_itinerary:", e)
        raise HTTPException(status_code=400, detail=f"Itinerary {itinerary_id} not found")

@router.get("/user/{user_id}")
async def get_user_itineraries(
    user_id: int,
    page: int = 1,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
) -> ItineraryListResponse:
    """
    특정 사용자의 일정 리스트를 조회
    """
    try:
        result = await itinerary_service.get_user_itineraries(db, user_id, page, limit)
        return ItineraryListResponse(**result)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/my/list")
async def get_my_itineraries(
    page: int = 1,
    limit: int = 10,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ItineraryListResponse:
    """
    현재 로그인한 사용자의 일정 리스트를 조회
    """
    try:
        result = await itinerary_service.get_user_itineraries(db, current_user, page, limit)
        return ItineraryListResponse(**result)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/createItinerary")
async def create_itinerary_with_name(req: Request, db: AsyncSession = Depends(get_db)) -> ItineraryCreateRequest:
    body = await req.json()
    create_itinerary_request = ItineraryCreateRequest(**body)
    print("[DEBUG] CREATE ITINERARY REQUEST WITH NAME: ")
    try:
        itinerary = await itinerary_service.create_itinerary_with_name(db, create_itinerary_request)
        return itinerary
    except Exception as e:
        print("[ERROR] create_itinerary_with_name : ", e)
        raise HTTPException(status_code=500, detail=str(e))