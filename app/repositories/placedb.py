from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func
# from app.models.user import User 
from app.models.postgre_model import Place
from datetime import datetime, timezone
import math

async def search_place(db: AsyncSession, query: str, page: int, limit: int):
    offset = (page - 1) * limit
    result = await db.execute(
        select(Place).where(
            Place.name.ilike(f"%{query}%")
        ).order_by(Place.place_id.asc()).offset(offset).limit(limit)
    )
    total_count = await db.execute(
        select(func.count()).select_from(Place).where(
            Place.name.ilike(f"%{query}%")
        )
    )
    page_count = (total_count.scalar_one() + limit - 1) // limit
    return {
        "data": result.scalars().all(),
        "total_pages": page_count
    }

async def get_place(db: AsyncSession, place_id: int) -> Place:
    result = await db.execute(
        select(Place).where(
            Place.place_id == place_id
        )
    )
    return result.scalar_one_or_none()

async def get_place_list(db: AsyncSession, page: int, limit: int, lat: float, lng: float, radius:float):
    """
    장소 목록 조회
    params:
        page: int
        limit: int
    returns:
        data: list[Place]
        total_pages: int
    """
    offset = (page - 1) * limit
    if lat != -1 and lng != -1 and radius != -1:
        result = await db.execute(
            get_places_within_radius_query(lat, lng, radius).offset(offset).limit(limit)
        )
    else:
        result = await db.execute(
            select(Place).order_by(Place.place_id.asc()).offset(offset).limit(limit)
        )
    total_count = await db.execute(
        select(func.count()).select_from(Place)
    )
    page_count = (total_count.scalar_one() + limit - 1) // limit

    return {
        "data": result.scalars().all(),
        "total_pages": page_count
    }

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