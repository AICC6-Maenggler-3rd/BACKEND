from __future__ import annotations
from gettext import install
import nntplib
from typing import Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, ConfigDict, model_validator

# ---------- Reference DTOs (순환 방지용 요약) ----------
class UserRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_id: int
    email: Optional[str] = None
    name: Optional[str] = None

class PlaceRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    place_id: int
    name: Optional[str] = None

class AccommodationRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    accommodation_id: int
    name: Optional[str] = None

class CategoryRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    category_id: int
    name: Optional[str] = None

# ---------- AppUser ----------
class AppUserSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_id: int
    email: str
    name: str
    status: str
    role: str
    provider: Optional[str] = None
    provider_user_id: str
    created_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    # relations
    places_registered: Optional[List[PlaceRef]] = None

# ---------- Category ----------
class CategorySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    category_id: int
    name: str
    status: str
    created_at: datetime
    updated_at: datetime
    # relations
    places: List[PlaceRef] = []

# ---------- Place ----------
class PlaceSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    place_id: int
    name: str
    address: str
    address_la: float
    address_lo: float
    type: str
    count: int
    website: Optional[str] = None
    image_url: Optional[str] = None
    insta_nickname: Optional[str] = None
    open_hour: Optional[str] = None
    close_hour: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime
    deleted_at: Optional[datetime] = None
    updated_at: datetime
    # relations
    # registrars: List[UserRef] = []
    # categories: List[CategoryRef] = []

# ---------- Accommodation ----------
class AccommodationSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    accommodation_id: int
    name: str
    address: str
    address_la: float
    address_lo: float
    type: str
    phone: Optional[str] = None
    image_url: Optional[str] = None
    created_at: datetime
    deleted_at: Optional[datetime] = None
    updated_at: datetime

# ---------- ItineraryItem ----------
class ItineraryItemSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    item_id: int
    itinerary_id: int
    place_id: Optional[int] = None
    accommodation_id: Optional[int] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    is_required: bool
    created_at: datetime
    deleted_at: Optional[datetime] = None
    # relations
    place: Optional[PlaceRef] = None
    accommodation: Optional[AccommodationRef] = None


# ---------- Itinerary ----------
class ItinerarySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    itinerary_id: int
    user_id: Optional[int] = None
    relation: Optional[str] = None
    name: Optional[str] = None
    start_at: datetime
    end_at: datetime
    location: Optional[str] = None
    theme: Optional[str] = None
    created_at: datetime
    deleted_at: Optional[datetime] = None
    updated_at: datetime
    # relations - 리스트 조회에서는 관계 속성 제외
    # user: Optional[UserRef] = None
    # items: Optional[List[ItineraryItemSchema]] = None

# ---------- Association rows (필요 시) ----------
class PlaceRegistrarLink(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    place_id: int
    user_id: int
    created_at: datetime

class PlaceCategoryLink(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    category_id: int
    place_id: int

# ---------- 사용 예 ----------
# dto = ItinerarySchema.model_validate(sa_itinerary, from_attributes=True)
# place_dto = PlaceSchema.model_validate(sa_place, from_attributes=True)
# user_dto  = AppUserSchema.model_validate(sa_user, from_attributes=True)

class ItineraryPlaceItem(ItineraryItemSchema):
    info: Optional[PlaceSchema] = None

class ItineraryAccommodationItem(ItineraryItemSchema):
    info: AccommodationSchema

class ItineraryItemResponse(BaseModel):
    item_type: str
    # data는 Place 또는 Accommodations
    data: Union[ItineraryPlaceItem, ItineraryAccommodationItem]

class ItineraryResponse(BaseModel):
    location: Optional[str] = None
    theme: Optional[str] = None
    start_at: datetime
    end_at: datetime
    relation: Optional[str] = None
    user_id: Optional[int] = None
    items: List[ItineraryItemResponse]
    name: Optional[str] = None

# ──────────── 일정 아이템 생성 모델 ────────────
class ItineraryItemCreate(BaseModel):
    # item_type: str
    place_id: Optional[int] = None
    accommodation_id: Optional[int] = None
    start_time: datetime
    end_time: Optional[datetime]
    is_required: bool

    @model_validator(mode="after")
    def normalize_zero_ids(self):
        # 0을 None으로 정규화
        if self.place_id == 0:
            self.place_id = None
        if self.accommodation_id == 0:
            self.accommodation_id = None
        # 최소 하나는 존재해야 함
        if self.place_id is None and self.accommodation_id is None:
            raise ValueError("Either place_id or accommodation_id must be provided")
        return self

    # @model_validator(mode="after")
    # def validate_item(self):
    #     if self.item_type == "place" and not self.place_id:
    #         raise ValueError("place_id is required when item_type is 'place'")
    #     if self.item_type == "accommodation" and not self.accommodation_id:
    #         raise ValueError("accommodation_id is required when item_type is 'accommodation'")
    #     return self

class ItineraryCreate(BaseModel):
    location: str
    theme: Optional[str] = None
    start_at: datetime
    end_at: datetime
    relation: Optional[str] = None
    user_id: Optional[int] = None
    items: List[ItineraryItemCreate]

# 
class ItineraryCreateRequest(BaseModel):
    """
    Args : 일정 name 포함 생성 요청
    """
    user_id: Optional[int] = None
    relation: Optional[str] = None
    start_at: datetime
    end_at: datetime
    location: str
    theme: Optional[str] = None
    name: str
    items: List[ItineraryItemCreate]

class ItineraryGenerate(BaseModel):
    base_itinerary: ItineraryCreate
    model_name: str

# ----------- 일정 리스트 응답 -----------
class ItineraryListResponse(BaseModel):
    itineraries: List[ItinerarySchema]
    total_count: int
    page: int
    limit: int
    total_pages: int

# ----------- 일정 상세 조회 -----------
class ItineraryDetailMetadata(BaseModel):
    itinerary_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    total_items: int
    active_items: int
    deleted_items: int
    total_places: int
    total_accommodations: int
    required_items: int
    optional_items: int

class ItineraryDetailResponse(BaseModel):
    itinerary: ItineraryResponse
    metadata: ItineraryDetailMetadata

class ItineraryItemWithStatus(BaseModel):
    item_id: int
    itinerary_id: int
    place_id: int
    accommodation_id: Optional[int] =  None
    start_time: datetime
    end_time: Optional[datetime] = None
    is_required: bool
    created_at: datetime
    deleted_at: Optional[datetime] = None
    is_deleted: bool
    item_type: str
    info: Union[PlaceSchema, AccommodationSchema]