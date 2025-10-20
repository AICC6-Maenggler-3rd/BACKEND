from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func, text, delete
from app.models.postgre_model import Accommodation
from datetime import datetime, timezone
import math
from typing import Optional
from pydantic import BaseModel

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
    
    # 결과를 딕셔너리 리스트로 변환
    data = [dict(row) for row in result]
    
    return {
        "data": data,
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

async def create_accommodation(db: AsyncSession, accommodation_data: BaseModel) -> Accommodation:
    """
    새로운 숙소 생성
    """
    accommodation = Accommodation(
        name=accommodation_data.name,
        address=accommodation_data.address,
        address_la=accommodation_data.address_la,
        address_lo=accommodation_data.address_lo,
        type=accommodation_data.type,
        phone=accommodation_data.phone,
        image_url=accommodation_data.image_url,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    
    db.add(accommodation)
    await db.commit()
    await db.refresh(accommodation)
    return accommodation

async def update_accommodation(db: AsyncSession, accommodation_id: int, accommodation_data: BaseModel) -> Optional[Accommodation]:
    """
    숙소 정보 수정
    """
    print(f"update_accommodation 호출 - ID: {accommodation_id}")
    
    # 먼저 숙소가 존재하는지 확인
    result = await db.execute(
        select(Accommodation).where(Accommodation.accommodation_id == accommodation_id)
    )
    accommodation = result.scalar_one_or_none()
    
    if not accommodation:
        print(f"숙소를 찾을 수 없음 - ID: {accommodation_id}")
        return None
    
    # 업데이트할 필드만 수정
    update_data = accommodation_data.model_dump(exclude_unset=True)
    print(f"업데이트 데이터: {update_data}")
    
    if update_data:
        update_data['updated_at'] = datetime.now(timezone.utc).replace(tzinfo=None)
        
        await db.execute(
            update(Accommodation)
            .where(Accommodation.accommodation_id == accommodation_id)
            .values(**update_data)
        )
        await db.commit()
        print(f"데이터베이스 업데이트 완료 - ID: {accommodation_id}")
        
        # 업데이트된 숙소 정보 조회
        result = await db.execute(
            select(Accommodation).where(Accommodation.accommodation_id == accommodation_id)
        )
        updated_accommodation = result.scalar_one_or_none()
        print(f"업데이트된 숙소 정보: {updated_accommodation}")
        return updated_accommodation
    
    print(f"업데이트할 데이터가 없음 - ID: {accommodation_id}")
    return accommodation

async def delete_accommodation(db: AsyncSession, accommodation_id: int) -> bool:
    """
    숙소 삭제 (soft delete)
    """
    # 먼저 숙소가 존재하는지 확인
    result = await db.execute(
        select(Accommodation).where(Accommodation.accommodation_id == accommodation_id)
    )
    accommodation = result.scalar_one_or_none()
    
    if not accommodation:
        return False
    
    # soft delete (deleted_at 필드에 현재 시간 설정)
    await db.execute(
        update(Accommodation)
        .where(Accommodation.accommodation_id == accommodation_id)
        .values(
            deleted_at=datetime.now(timezone.utc).replace(tzinfo=None),
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
        )
    )
    await db.commit()
    
    return True