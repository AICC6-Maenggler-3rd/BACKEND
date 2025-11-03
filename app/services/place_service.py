from app.rag.itinerary import search_places
from app.core.config import RAG_DIR
from app.schemas.place_schema import PlaceListResponse
from fastapi import HTTPException
from app.schemas.place_schema import PlaceSchema
from app.repositories import placedb
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.regiondb import get_region_by_name
import re
import traceback
async def search_place_rag(db: AsyncSession, query: str, count: int) -> PlaceListResponse:
    try:
        region_name = get_region_name(query)
        print(f"[RAG Search] Query: {query}, Count: {count}, RAG_DIR: {RAG_DIR}")
        if region_name:
            print("region_name", region_name)
            location = await get_region_by_name(db, region_name)
            if location:
                print("location", location)
                ids = await search_places(str(RAG_DIR), query, count, location.address_la, location.address_lo)
            else:
                ids = await search_places(str(RAG_DIR), query, count)
        else:
            ids = await search_places(str(RAG_DIR), query, count)
        places = await placedb.get_places_by_ids(db, [int(id) for id in ids])

        result = PlaceListResponse(
            places=places,
            total_pages=1,
            total_count=len(places)
        )
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[RAG Search Error]: {e}")
        print(f"[Traceback]: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))


def get_region_name(query: str) -> str:
    major = ["서울", "인천", "경기", "강원", "충청", "전라", "경상", "제주", "부산", "대전", "대구", "울산", "세종", "광주"]
    for m in major:
        if m in query:
            return m
    regex = r"(\w+구|\w+시|\w+도|\w+특별시)"
    match = re.search(regex, query)
    return match.group(1) if match else None