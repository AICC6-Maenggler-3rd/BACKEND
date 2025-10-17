from fastapi import APIRouter, HTTPException, Depends
from app.repositories import regiondb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
router = APIRouter()

# @router.get("/{name}")
# async def get_region(name: str, db: AsyncSession = Depends(get_db)):
#     try:
#         region = await regiondb.get_region_by_name(db, name)
#         if region is None:
#             raise HTTPException(status_code=404, detail=f"Region '{name}' not found")
#         return {"address_la": region.address_la, "address_lo": region.address_lo}
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def get_all_region(db: AsyncSession = Depends(get_db)):
    print("==================================================")
    try:
        region_list = await regiondb.get_all_region(db)
        return region_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))