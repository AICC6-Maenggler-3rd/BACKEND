from app.schemas.postgre_schema import ItineraryItemSchema, ItinerarySchema, PlaceSchema, AccommodationSchema, ItineraryResponse, ItineraryItemResponse, ItineraryPlaceItem, ItineraryAccommodationItem, ItineraryCreate, ItineraryGenerate, ItineraryCreateRequest
from app.models.postgre_model import Itinerary, ItineraryItem, Place, Accommodation
# from app.schemas.place_schema import PlaceSchema
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from time import sleep
from pydantic import BaseModel, model_validator
from typing import Optional, List, Union
from datetime import datetime, timezone, timedelta
from app.repositories.placedb import get_place
from app.services.generate_itinerary_service import none_generate_itinerary, random_generate_itinerary, nextpoi_generate_itinerary
from sqlalchemy import func


def add_korean_timezone(dt: datetime) -> datetime:
    """UTC 시간에 9시간을 더해서 한국 시간으로 변환"""
    if dt is None:
        return None
    return dt + timedelta(hours=9)


async def generate_itinerary(db: AsyncSession, generate_itinerary_request: ItineraryGenerate) -> ItineraryResponse:
    # 들어온 정보를 이용해서 db에서 장소와 숙소 정보를 조회하여  ItineraryResponse 생성
    # db에 itinerary를 추가하지는 않음
    if generate_itinerary_request.model_name == "none":
        itinerary = await none_generate_itinerary(db, generate_itinerary_request)
    elif generate_itinerary_request.model_name == "random":
        itinerary = await random_generate_itinerary(db, generate_itinerary_request)
    elif generate_itinerary_request.model_name == "gru4rec":
        itinerary = await nextpoi_generate_itinerary(db, generate_itinerary_request)
    else:
        raise ValueError(f"Invalid model name: {generate_itinerary_request.model_name}")
    return itinerary

async def get_itinerary_items(db: AsyncSession, user_id: int) -> List[ItineraryItem]:
    result = await db.execute(
        select(ItineraryItem)
        .join(Itinerary, ItineraryItem.itinerary_id == Itinerary.itinerary_id)
        .filter(Itinerary.user_id == user_id)
    )
    return result.scalars().all()

async def get_itinerary_places(db: AsyncSession, itinerary_items: List[ItineraryItem]) -> List[Place]:
    result = await db.execute(
        select(Place)
        .join(ItineraryItem, Place.place_id == ItineraryItem.place_id)
        .filter(ItineraryItem.itinerary_id.in_(itinerary_items))
    )
    return result.scalars().all()

async def get_itinerary_accommodations(db: AsyncSession, itinerary_items: List[ItineraryItem]) -> List[Accommodation]:
    result = await db.execute(
        select(Accommodation)
        .join(ItineraryItem, Accommodation.accommodation_id == ItineraryItem.accommodation_id)
        .filter(ItineraryItem.itinerary_id.in_(itinerary_items))
    )
    return result.scalars().all()

async def get_itinerary_response(db: AsyncSession, itinerary_id: int) -> ItineraryResponse:
    """
    itinerary_id로 일정과 관련 데이터를 조회하고
    ItineraryResponse 형태로 반환 (async 버전)
    """

    # itinerary + item + place/accommodation 미리 로드
    result = await db.execute(
        select(Itinerary)
        .options(
            joinedload(Itinerary.items)
            .joinedload(ItineraryItem.place)
            .joinedload(Place.registrars),
            joinedload(Itinerary.items)
            .joinedload(ItineraryItem.place)
            .joinedload(Place.categories),
            joinedload(Itinerary.items)
            .joinedload(ItineraryItem.accommodation),
        )
        .filter(Itinerary.itinerary_id == itinerary_id)
    )
    itinerary = result.scalars().first()

    if not itinerary:
        raise ValueError(f"Itinerary {itinerary_id} not found")

    items_response: List[ItineraryItemResponse] = []

    for item in itinerary.items:
        # 숙소 항목
        if item.accommodation_id:
            acc_schema = AccommodationSchema.model_validate(item.accommodation)
            item_data = ItineraryAccommodationItem(
                item_id=item.item_id,
                itinerary_id=item.itinerary_id,
                place_id=item.place_id,
                accommodation_id=item.accommodation_id,
                start_time=add_korean_timezone(item.start_time),
                end_time=add_korean_timezone(item.end_time),
                is_required=item.is_required,
                created_at=item.created_at,
                info=acc_schema,
            )
            items_response.append(
                ItineraryItemResponse(item_type="accommodation", data=item_data)
            )

        # 장소 항목
        else:
            if item.place is None:
                print(f"[WARNING] Place data is None for item {item.item_id}")
                continue
            place_schema = PlaceSchema.model_validate(item.place)
            item_data = ItineraryPlaceItem(
                item_id=item.item_id,
                itinerary_id=item.itinerary_id,
                place_id=item.place_id,
                accommodation_id=None,
                start_time=add_korean_timezone(item.start_time),
                end_time=add_korean_timezone(item.end_time),
                is_required=item.is_required,
                created_at=item.created_at,
                info=place_schema,
            )
            items_response.append(
                ItineraryItemResponse(item_type="place", data=item_data)
            )

    # ItineraryResponse 반환
    return ItineraryResponse(
        location=itinerary.location,
        theme=itinerary.theme,
        start_at=add_korean_timezone(itinerary.start_at),
        end_at=add_korean_timezone(itinerary.end_at),
        relation=itinerary.relation,
        user_id=itinerary.user_id,
        items=items_response,
        name=itinerary.name
    )


async def create_itinerary(db:AsyncSession, itinerary_data:ItineraryCreate):
    itinerary = Itinerary(
        user_id=itinerary_data.user_id,
        relation=itinerary_data.relation,
        location=itinerary_data.location,
        theme=itinerary_data.theme,
        start_at=itinerary_data.start_at,
        end_at=itinerary_data.end_at,
        name=itinerary_data.name
    )
    db.add(itinerary)
    await db.flush()  # itinerary_id 확보

    for item_data in itinerary_data.items:
        item = ItineraryItem(
            itinerary_id=itinerary.itinerary_id,
            place_id=item_data.place_id,
            accommodation_id=item_data.accommodation_id,
            start_time=item_data.start_time,
            end_time=item_data.end_time,
            is_required=item_data.is_required,
        )
        db.add(item)

    await db.commit()
    await db.refresh(itinerary)
    
    return await get_itinerary_response(db, itinerary.itinerary_id)

async def create_itinerary_with_name(db:AsyncSession, itinerary_data:ItineraryCreateRequest):

    print("[DEBUG] ITINERARY DATA WITH NAME: ")

    def to_naive_utc(dt: datetime | None) -> datetime | None:
        if dt is None:
            return None
        if dt.tzinfo is None:
            # Already naive; treat as UTC-naive
            return dt
        # Convert to UTC then drop tzinfo to store into TIMESTAMP WITHOUT TIME ZONE
        return dt.astimezone(timezone.utc).replace(tzinfo=None)

    itinerary = Itinerary(
        user_id=itinerary_data.user_id,
        relation=itinerary_data.relation,
        location=itinerary_data.location,
        theme=itinerary_data.theme,
        start_at=to_naive_utc(itinerary_data.start_at),
        end_at=to_naive_utc(itinerary_data.end_at),
        name=itinerary_data.name,
    )
    db.add(itinerary)
    await db.flush()  # itinerary_id 확보
    
    print("[DEBUG] ITINERARY DATA : ", itinerary.itinerary_id)

    for item_data in itinerary_data.items:
        item = ItineraryItem(
            itinerary_id=itinerary.itinerary_id,
            place_id=item_data.place_id if item_data.place_id != 0 else None,
            accommodation_id=item_data.accommodation_id if item_data.accommodation_id != 0 else None,
            start_time=to_naive_utc(item_data.start_time),
            end_time=to_naive_utc(item_data.end_time),
            is_required=item_data.is_required,
        )
        db.add(item)

    await db.commit()
    await db.refresh(itinerary)
    
    # 생성된 일정 데이터를 반환
    return itinerary_data

async def get_itinerary(db: AsyncSession, itinerary_id: int) -> ItineraryResponse:
    return await get_itinerary_response(db, itinerary_id)

async def get_user_itineraries(db: AsyncSession, user_id: int, page: int = 1, limit: int = 10) -> dict:
    # 전체 일정 개수 조회
    count_result = await db.execute(
        select(func.count(Itinerary.itinerary_id))
        .filter(Itinerary.user_id == user_id)
        .filter(Itinerary.deleted_at.is_(None))
    )
    total_count = count_result.scalar()

    offset = (page - 1) * limit

    # 일정 리스트 조회
    result = await db.execute(
        select(Itinerary)
        .filter(Itinerary.user_id == user_id)
        .filter(Itinerary.deleted_at.is_(None))
        .order_by(Itinerary.start_at.desc())
        .offset(offset)
        .limit(limit)
    )
    itineraries = result.scalars().all()

    # ItinerarySchema로 변환
    itinerary_list = []
    for itinerary in itineraries:
        itinerary_schema = ItinerarySchema.model_validate(itinerary)
        itinerary_list.append(itinerary_schema)
    return {
        "itineraries": itinerary_list,
        "total_count": total_count,
        "page": page,
        "limit": limit,
        "total_pages": (total_count + limit -1) //limit
    }

async def get_itinerary_detail_with_metadata(db: AsyncSession, itinerary_id: int) -> dict:
    try:
        # 일정 기본 정보 조회
        result = await db.execute(
            select(Itinerary)
            .options(
                joinedload(Itinerary.items)
                .joinedload(ItineraryItem.place)
                .joinedload(Place.registrars),
                joinedload(Itinerary.items)
                .joinedload(ItineraryItem.place)
                .joinedload(Place.categories),
                joinedload(Itinerary.items)
                .joinedload(ItineraryItem.accommodation),
            )
            .filter(Itinerary.itinerary_id == itinerary_id)
            .filter(Itinerary.deleted_at.is_(None))
        )
        itinerary = result.scalars().first()

        if not itinerary:
            raise ValueError(f"Itinerary {itinerary_id} not found")

        # 삭제되지 않은 아이템들만 필터링
        active_items = [item for item in itinerary.items if item.deleted_at is None]
        deleted_items = [item for item in itinerary.items if item.deleted_at is not None]
        
        # 시간순으로 정렬
        sorted_items = sorted(active_items, key=lambda x: x.start_time)
        
        items_response: List[ItineraryItemResponse] = []

        for item in sorted_items:
            # 숙소 항목 처리
            if item.accommodation_id:
                acc_schema = AccommodationSchema.model_validate(item.accommodation)
                item_data = ItineraryAccommodationItem(
                    item_id=item.item_id,
                    itinerary_id=item.itinerary_id,
                    place_id=item.place_id,
                    accommodation_id=item.accommodation_id,
                    start_time=add_korean_timezone(item.start_time),
                    end_time=add_korean_timezone(item.end_time),
                    is_required=item.is_required,
                    created_at=item.created_at,
                    info=acc_schema,
                )
                items_response.append(
                    ItineraryItemResponse(item_type="accommodation", data=item_data)
                )
            # 장소 항목 처리
            else:
                if item.place is None:
                    print(f"[WARNING] Place data is None for item {item.item_id}")
                    # place_id가 NULL인 경우 기본 정보만으로 아이템 생성
                    item_data = ItineraryPlaceItem(
                        item_id=item.item_id,
                        itinerary_id=item.itinerary_id,
                        place_id=item.place_id,
                        accommodation_id=None,
                        start_time=add_korean_timezone(item.start_time),
                        end_time=add_korean_timezone(item.end_time),
                        is_required=item.is_required,
                        created_at=item.created_at,
                        info=None,  # place 정보가 없으므로 None
                    )
                    items_response.append(
                        ItineraryItemResponse(item_type="place", data=item_data)
                    )
                else:
                    place_schema = PlaceSchema.model_validate(item.place)
                    item_data = ItineraryPlaceItem(
                        item_id=item.item_id,
                        itinerary_id=item.itinerary_id,
                        place_id=item.place_id,
                        accommodation_id=None,
                        start_time=add_korean_timezone(item.start_time),
                        end_time=add_korean_timezone(item.end_time),
                        is_required=item.is_required,
                        created_at=item.created_at,
                        info=place_schema,
                    )
                    items_response.append(
                        ItineraryItemResponse(item_type="place", data=item_data)
                    )

        # 통계 정보 계산
        total_places = sum(1 for item in active_items if not item.accommodation_id)
        total_accommodations = sum(1 for item in active_items if item.accommodation_id)
        required_items = sum(1 for item in active_items if item.is_required)

        print(f"[DEBUG] 총 아이템 수: {len(active_items)}")
        print(f"[DEBUG] 응답할 아이템 수: {len(items_response)}")
        print(f"[DEBUG] items_response: {items_response}")

        return {
            "itinerary": ItineraryResponse(
                location=itinerary.location,
                theme=itinerary.theme,
                start_at=add_korean_timezone(itinerary.start_at),
                end_at=add_korean_timezone(itinerary.end_at),
                relation=itinerary.relation,
                user_id=itinerary.user_id,
                items=items_response,
                name=itinerary.name
            ),
            "metadata": {
                "itinerary_id": itinerary.itinerary_id,
                "created_at": itinerary.created_at,
                "updated_at": itinerary.updated_at,
                "deleted_at": itinerary.deleted_at,
                "total_items": len(itinerary.items),
                "active_items": len(active_items),
                "deleted_items": len(deleted_items),
                "total_places": total_places,
                "total_accommodations": total_accommodations,
                "required_items": required_items,
                "optional_items": len(active_items) - required_items
            }
        }

    except Exception as e:
        print(f"[ERROR] get_itinerary_detail_with_metadata: {e}")
        raise e

async def delete_itinerary(db: AsyncSession, itinerary_id: int, user_id: int) -> bool:
    try:
        print(f"[DEBUG] delete_itinerary called with id: {itinerary_id}, user_id: {user_id}")
        
        # 일정 존재 여부 및 소유자 확인
        result = await db.execute(
            select(Itinerary)
            .options(joinedload(Itinerary.items))
            .filter(Itinerary.itinerary_id == itinerary_id)
            .filter(Itinerary.deleted_at.is_(None))  # 삭제되지 않은 일정만
        )
        itinerary = result.scalars().first()
        
        if not itinerary:
            raise ValueError(f"Itinerary {itinerary_id} not found")
        
        # 사용자 권한 확인
        if itinerary.user_id != user_id:
            raise ValueError("You don't have permission to delete this itinerary")
        
        # soft delete 실행
        itinerary.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # 관련 아이템들도 soft delete
        for item in itinerary.items:
            if item.deleted_at is None: 
                item.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
        
        await db.commit()
        
        print(f"[SUCCESS] Itinerary {itinerary_id} deleted successfully")
        return True
        
    except ValueError as e:
        print(f"[ERROR] delete_itinerary validation error: {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] delete_itinerary: {e}")
        await db.rollback()
        raise e