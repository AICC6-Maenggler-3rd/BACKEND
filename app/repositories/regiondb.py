from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func , text
from app.models.postgre_model import Region
from datetime import datetime, timezone
import math

async def get_region_by_name(db: AsyncSession, name: str):
    sql = text("""
                    SELECT
                      address_la, address_lo
                    FROM region
                    WHERE name ILIKE '%' || :name || '%'
                    LIMIT 1
                """)

    result = await db.execute(sql, {"name": name})
    row = result.mappings().one_or_none()
    return row

async def get_all_region(db: AsyncSession):
    sql = text("""
                    SELECT
                      name, address_la, address_lo
                    FROM region
                """)

    result = await db.execute(sql)
    return result.mappings().all()