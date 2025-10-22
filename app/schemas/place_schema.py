from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel


class PlaceBase(BaseModel):
    name: str
    address: str
    address_la: float
    address_lo: float
    type: str
    count: int = 0
    website: Optional[str] = None
    image_url: Optional[str] = None
    insta_nickname: Optional[str] = None
    open_hour: Optional[str] = None
    close_hour: Optional[str] = None
    description: Optional[str] = None


class PlaceSchema(PlaceBase):
    place_id: int
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

    model_config = {
        "from_attributes": True  # ✅ ORM 객체 -> Pydantic 변환 허용
    }


class PlaceListResponse(BaseModel):
    places: list[PlaceSchema]
    total_pages: int