from app.schemas.postgre_schema import ItineraryItemSchema, ItinerarySchema, PlaceSchema, AccommodationSchema
from app.models.postgre_model import Itinerary, ItineraryItem, Place, Accommodation
from app.services.itinerary_service import ItineraryGenerate, ItineraryResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import POI_MODEL_PATH, PLACES_PATH
from datetime import datetime, timedelta
from app.repositories.placedb import get_place, get_random_place
from app.repositories.regiondb import get_region_by_name
from app.services.itinerary_service import ItineraryPlaceItem, ItineraryItemResponse
from app.ml.next_poi_sas_rec import recommend_next
import math
from typing import List


async def get_generate_model_list()-> List[str]:
  return ["none", "random", "nextpoi"]

async def none_generate_itinerary(db: AsyncSession, generate_itinerary_request: ItineraryGenerate) -> ItineraryResponse:
    # 아무것도 하지 않고 그냥 일정 생성
    itinerary = ItineraryResponse(
        location=generate_itinerary_request.base_itinerary.location,
        theme=generate_itinerary_request.base_itinerary.theme,
        start_at=generate_itinerary_request.base_itinerary.start_at,
        end_at=generate_itinerary_request.base_itinerary.end_at,
        relation=generate_itinerary_request.base_itinerary.relation,
        user_id=generate_itinerary_request.base_itinerary.user_id,
        items=[],
    )

    for item in generate_itinerary_request.base_itinerary.items:
        if item.place_id:
            print(item.place_id)
            place = await get_place(db, item.place_id)
            place_schema = PlaceSchema.model_validate(place)
            place_data = ItineraryPlaceItem(
                item_id=-1,
                itinerary_id=-1,
                place_id=item.place_id,
                accommodation_id=None,
                start_time=item.start_time,
                end_time=item.end_time,
                is_required=item.is_required,
                created_at=datetime.now(),
                info=place_schema,
            )
            itinerary.items.append(ItineraryItemResponse(item_type="place", data=place_data))
    return itinerary

async def random_generate_itinerary(db: AsyncSession, generate_itinerary_request: ItineraryGenerate) -> ItineraryResponse:
  # 기존에 있는 장소는 유지하고, 하루에 장소가 3개가 되도록 랜덤으로 장소를 추가
  itinerary = await none_generate_itinerary(db, generate_itinerary_request)

  duration = calculate_duration(generate_itinerary_request.base_itinerary.start_at, generate_itinerary_request.base_itinerary.end_at)
  print("duration", duration)
  for day in range(duration):
    # 해당 날짜의 장소 목록
    place_ids = [x.data.place_id for x in itinerary.items if x.item_type == "place" and calculate_day_index(generate_itinerary_request.base_itinerary.start_at, x.data.start_time) == day]
    print("place_ids", place_ids)
    place_count = len(place_ids)
    if place_count >= 3:
      continue
    # 해달 날짜의 장소 갯수가 3개가 되도록 중복되지 않게 랜덤하게 추가
    for i in range(3 - place_count):
      # 해달날짜에 해당하는 place_id 목록
      is_added = False
      for k in range(10):
        print("get random place")
        place = await get_random_place(db)
        print("place", place.place_id)
        if place.place_id not in place_ids:
          is_added = True
          break
      if not is_added:
        print("failed to add place")
        break
      
      itinerary.items.append(ItineraryItemResponse(item_type="place", data=ItineraryPlaceItem(
        item_id=-1,
        itinerary_id=-1,
        place_id=place.place_id,
        accommodation_id=None,
        start_time=generate_itinerary_request.base_itinerary.start_at + timedelta(days=day) + timedelta(hours=9),
        end_time=generate_itinerary_request.base_itinerary.start_at + timedelta(days=day) + timedelta(hours=18),
        is_required=True,
        created_at=datetime.now(),
        info=PlaceSchema.model_validate(place),
      )))
  
  return itinerary

# 기간 계산, 시작일과 종료일 포함, 자정을 기준으로 계산
def calculate_duration(start_at: datetime, end_at: datetime) -> int:
  return math.ceil((end_at - start_at).total_seconds() / (24 * 60 * 60)) + 1

# 시작일을 기준으로 몇 번째 날짜인지 계산, 시작일은 0일로 계산
def calculate_day_index(start_at: datetime, date: datetime) -> int:
  return math.floor((date - start_at).total_seconds() / (24 * 60 * 60))

async def nextpoi_generate_itinerary(db: AsyncSession, generate_itinerary_request: ItineraryGenerate) -> ItineraryResponse:
  # 기존에 있는 장소는 유지하고, 하루에 장소가 3개가 되도록 랜덤으로 장소를 추가
  itinerary = await none_generate_itinerary(db, generate_itinerary_request)
  visit_place_ids = [x.data.place_id for x in itinerary.items if x.item_type == "place"]
  CKPT = POI_MODEL_PATH
  PLACES = PLACES_PATH
  duration = calculate_duration(generate_itinerary_request.base_itinerary.start_at, generate_itinerary_request.base_itinerary.end_at)
  for day in range(duration):
    # 해당 날짜의 장소 목록
    place_ids = [x.data.place_id for x in itinerary.items if x.item_type == "place" and calculate_day_index(generate_itinerary_request.base_itinerary.start_at, x.data.start_time) == day]
    print("place_ids", place_ids)
    place_count = len(place_ids)
    if place_count >= 5:
      continue
    region = await get_region_by_name(db, generate_itinerary_request.base_itinerary.location)
    if region is None:
      print("region not found")
      continue
    companions = generate_itinerary_request.base_itinerary.relation
    themes = generate_itinerary_request.base_itinerary.theme
    if themes is None:
      themes = ["여행"]
    try:
      recs = await recommend_next(ckpt_path=CKPT, places_path=PLACES, prefix_place_ids=place_ids, region_lat=region.address_la, region_lng=region.address_lo, companions=companions, themes=themes, already_visited=visit_place_ids)
    except Exception as e:
      print("error", e)
      continue
    if len(recs) == 0:
      continue
    print("recs", recs)
    for rec in recs[:5-place_count]:
      place = await get_place(db, rec["place_id"])
      place_schema = PlaceSchema.model_validate(place)
      itinerary.items.append(ItineraryItemResponse(item_type="place", data=ItineraryPlaceItem(
        item_id=-1,
        itinerary_id=-1,
        place_id=rec["place_id"],
        accommodation_id=None,
        start_time=generate_itinerary_request.base_itinerary.start_at + timedelta(days=day) + timedelta(hours=9),
        end_time=generate_itinerary_request.base_itinerary.start_at + timedelta(days=day) + timedelta(hours=18),
        is_required=True,
        created_at=datetime.now(),
        info=place_schema,
      )))
      visit_place_ids.append(rec["place_id"])

  return itinerary