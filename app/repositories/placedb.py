from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func
# from app.models.user import User 
from app.models.postgre_model import Place
from datetime import datetime, timezone
import math
from pydantic import BaseModel
from typing import List
from app.schemas.place_schema import PlaceSchema, PlaceListResponse




async def search_place(db: AsyncSession, query: str, page: int, limit: int) -> PlaceListResponse:
    search_query = select(Place).where(
        Place.name.ilike(f"%{query}%")
    ).order_by(Place.place_id.asc())
    return await paginate(db, search_query, page, limit)    

async def get_place(db: AsyncSession, place_id: int) -> Place:
    result = await db.execute(
        select(Place).where(
            Place.place_id == place_id
        )
    )
    return result.scalar_one_or_none()

async def get_random_place(db: AsyncSession) -> Place:
    result = await db.execute(
        select(Place).order_by(func.random()).limit(1)
    )
    return result.scalars().first()

async def get_place_list(db: AsyncSession, page: int, limit: int) -> PlaceListResponse:
    """
    장소 목록 조회
    params:
        page: int
        limit: int
    returns:
        data: list[Place]
        total_pages: int
    """
    result = await paginate(db, select(Place).order_by(Place.place_id.asc()), page, limit)

    return result

async def get_place_list_by_address(db: AsyncSession, address: str, page: int, limit: int) -> PlaceListResponse:
    search_query = get_place_list_by_address_query(address)
    return await paginate(db, search_query, page, limit)


async def get_place_list_by_address_and_radius(db: AsyncSession, lat: float, lng: float, radius: float, page: int, limit: int) -> PlaceListResponse:
    search_query = get_places_within_radius_query(lat, lng, radius)
    return await paginate(db, search_query, page, limit)

async def get_places(db: AsyncSession) -> list[Place]:
    """모든 장소 조회"""
    result = await db.execute(select(Place))
    return result.scalars().all()

async def get_insta_nicknames(db: AsyncSession) -> list[str]:
    """인스타그램 닉네임 목록 조회 (중복 제거)"""
    result = await db.execute(
        select(Place.insta_nickname)
        .where(Place.insta_nickname.isnot(None))
        .distinct()
    )
    return [nickname for nickname in result.scalars().all() if nickname]

async def get_places_by_insta_nickname(db: AsyncSession, insta_nickname: str, page: int, limit: int) -> PlaceListResponse:
    """특정 인스타그램 닉네임으로 장소 조회"""
    search_query = select(Place).where(
        Place.insta_nickname == insta_nickname
    ).order_by(Place.place_id.asc())
    return await paginate(db, search_query, page, limit)

async def get_places_by_category(db: AsyncSession, category_id: int) -> list[Place]:
    """특정 카테고리의 장소 조회 (many-to-many 관계)"""
    from app.models.postgre_model import place_category_table
    
    result = await db.execute(
        select(Place)
        .join(place_category_table, Place.place_id == place_category_table.c.place_id)
        .where(place_category_table.c.category_id == category_id)
    )
    return result.scalars().all()


async def paginate(db: AsyncSession, query, page: int, limit: int):
    offset = (page - 1) * limit
    result = await db.execute(query.offset(offset).limit(limit))
    total_count = await db.execute(select(func.count()).select_from(query))
    page_count = (total_count.scalar_one() + limit - 1) // limit
    return PlaceListResponse(
        places=[PlaceSchema.model_validate(place, from_attributes=True) for place in result.scalars().all()],
        total_pages=page_count
    )
def get_place_list_by_address_query(address: str):
    return select(Place).where(
            Place.address.ilike(f"%{address}%")
        ).order_by(Place.place_id.asc())


def get_places_within_radius_query(lat0: float, lng0: float, radius_m: float):
    """
    특정 좌표(lat0, lng0)로부터 radius_m 미터 이내의 Place 조회
    """
    # 먼저 bounding box로 대략 필터링 (성능 최적화)
    delta_lat = radius_m / 111320  # 1도 위도 ≈ 111.32km
    delta_lng = radius_m / (111320 * math.cos(math.radians(lat0)))

    lat_min = lat0 - delta_lat
    lat_max = lat0 + delta_lat
    lng_min = lng0 - delta_lng
    lng_max = lng0 + delta_lng

    lat_rad = func.radians(Place.address_la)
    lng_rad = func.radians(Place.address_lo)
    lat0_rad = math.radians(lat0)
    lng0_rad = math.radians(lng0)

    # Haversine 거리 계산
    R = 6371000
    distance_expr = 2 * R * func.asin(
        func.sqrt(
            func.pow(func.sin((lat_rad - lat0_rad) / 2), 2) +
            func.cos(lat0_rad) * func.cos(lat_rad) *
            func.pow(func.sin((lng_rad - lng0_rad) / 2), 2)
        )
    )
    
    stmt = (
        select(Place)
        .where(
            Place.address_la.between(lat_min, lat_max),
            Place.address_lo.between(lng_min, lng_max),
            distance_expr <= radius_m
        )
    )

    return stmt