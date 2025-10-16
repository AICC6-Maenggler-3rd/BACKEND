from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class CategoryBase(BaseModel):
    name: str
    status: str = "active"  # 활성화, 비활성화


class CategorySchema(CategoryBase):
    category_id: int
    created_at: datetime
    updated_at: datetime
   

    model_config = {
        "from_attributes": True  # ✅ ORM 객체 -> Pydantic 변환 허용
    }


class CategoryListResponse(BaseModel):
    categories: list[CategorySchema]
    total_pages: int
