from app.rag.itinerary import search_places
from app.core.config import RAG_DIR
from app.schemas.place_schema import PlaceListResponse
from fastapi import HTTPException
from app.schemas.place_schema import PlaceSchema
from app.repositories import placedb
from sqlalchemy.ext.asyncio import AsyncSession
async def search_place_rag(db: AsyncSession, query: str, count: int) -> PlaceListResponse:
    try:
        ids = await search_places(RAG_DIR, query, count)
        places = await placedb.get_places_by_ids(db, [int(id) for id in ids])

        result = PlaceListResponse(
            places=places,
            total_pages=1,
            total_count=len(places)
        )
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))