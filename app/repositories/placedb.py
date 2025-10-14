from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func
# from app.models.user import User 
from app.models.postgre_model import Place
from datetime import datetime, timezone

async def get_place(db: AsyncSession, place_id: int) -> Place:
    result = await db.execute(
        select(Place).where(
            Place.place_id == place_id
        )
    )
    return result.scalar_one_or_none()

async def get_place_list(db: AsyncSession, page: int, limit: int):
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

