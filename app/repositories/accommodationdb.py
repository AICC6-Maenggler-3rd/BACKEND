from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func , text
from app.models.postgre_model import Accommodation
from datetime import datetime, timezone
import math

async def search_accommodation(db: AsyncSession, query: str, page: int, limit: int):
    offset = (page - 1) * limit
    sql = text("""
                    SELECT
                      accommodation_id, name, address, address_la, address_lo,
                      type, phone, image_url, created_at, deleted_at, updated_at
                    FROM accommodation
                    WHERE name ILIKE '%' || :query || '%'
                    ORDER BY accommodation_id ASC
                    OFFSET :offset
                    LIMIT :limit
                """)

    result = await db.execute(sql, {"query": query, "offset": offset, "limit": limit}).mappings().all()
    total_count = await db.execute(
        select(func.count()).select_from(Accommodation).where(
            Accommodation.name.ilike(f"%{query}%")
        )
    )
    page_count = (total_count.scalar_one() + limit - 1) // limit
    return {
        "data": result.scalars().all(),
        "total_pages": page_count
    }

async def get_accommodation(db: AsyncSession, accommodation_id: int) -> Accommodation:
    result = await db.execute(
        select(Accommodation).where(
            Accommodation.accommodation_id == accommodation_id
        )
    )
    return result.scalar_one_or_none()

async def get_accommodation_list(db: AsyncSession, page: int, limit: int, lat: float, lng: float, radius:float):
    """
    숙소 목록 조회
    params:
        page: int
        limit: int
    returns:
        data: list[Accommodation]
        total_pages: int
    """
    offset = (page - 1) * limit
    if lat != -1 and lng != -1 and radius != -1:
        result = await db.execute(
            get_accommodations_within_radius_query(lat, lng, radius).offset(offset).limit(limit)
        )
    else:
        result = await db.execute(
            select(Accommodation).order_by(Accommodation.accommodation_id.asc()).offset(offset).limit(limit)
        )
    total_count = await db.execute(
        select(func.count()).select_from(Accommodation)
    )
    page_count = (total_count.scalar_one() + limit - 1) // limit

    return {
        "data": result.scalars().all(),
        "total_pages": page_count
    }

def get_accommodations_within_radius_query(lat0: float, lng0: float, radius_m: float):
    """
    특정 좌표(lat0, lng0)로부터 radius_m 미터 이내의 Accommodation 조회
    """
    # 먼저 bounding box로 대략 필터링 (성능 최적화)
    delta_lat = radius_m / 111320  # 1도 위도 ≈ 111.32km
    delta_lng = radius_m / (111320 * math.cos(math.radians(lat0)))

    lat_min = lat0 - delta_lat
    lat_max = lat0 + delta_lat
    lng_min = lng0 - delta_lng
    lng_max = lng0 + delta_lng

    lat_rad = func.radians(Accommodation.address_la)
    lng_rad = func.radians(Accommodation.address_lo)
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
        select(Accommodation)
        .where(
            Accommodation.address_la.between(lat_min, lat_max),
            Accommodation.address_lo.between(lng_min, lng_max),
            distance_expr <= radius_m
        )
    )

    return stmt