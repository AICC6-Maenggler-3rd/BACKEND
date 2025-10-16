from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func
# from app.models.user import User 
from app.models.postgre_model import Category
from datetime import datetime, timezone
import math
from pydantic import BaseModel
from typing import List
from app.schemas.category_schema import CategorySchema, CategoryListResponse

async def get_categories(db: AsyncSession) -> list[Category]:
    """모든 장소 조회"""
    result = await db.execute(select(Category))
    return result.scalars().all()