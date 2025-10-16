from app.services import itinerary_service
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, Request
from app.schemas.postgre_schema import ItineraryCreate, ItineraryGenerate, ItineraryResponse


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
    itinerary = await itinerary_service.get_itinerary(db, itinerary_id)
    return itinerary

@router.post("/create")
async def create_itinerary(req: Request, db: AsyncSession = Depends(get_db)) -> ItineraryResponse:
    body = await req.json()
    create_itinerary_request = ItineraryCreate(**body)
    itinerary = await itinerary_service.create_itinerary(db, create_itinerary_request)
    return itinerary
