from app.schemas.postgre_schema import ItineraryItemSchema, ItinerarySchema, PlaceSchema, AccommodationSchema
from app.models.postgre_model import Itinerary, ItineraryItem, Place, Accommodation
from app.services.itinerary_service import ItineraryGenerate, ItineraryResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import POI_MODEL_PATH, PLACES_PATH
from datetime import datetime, timedelta
from app.repositories.placedb import get_place, get_random_place
from app.repositories.regiondb import get_region_by_name
from app.services.itinerary_service import ItineraryPlaceItem, ItineraryItemResponse
from app.ml.next_poi_gru4rec import get_next_poi_list
from app.contentmodel.content_based_recommendation_service import content_based_service
import math
from typing import List


async def get_generate_model_list()-> List[str]:
  return ["none", "random", "gru4rec", "content_based"]

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
      last_place_id = place_ids[-1]
      last_place = await get_place(db, last_place_id)
      # recs = await recommend_next(ckpt_path=CKPT, places_path=PLACES, prefix_place_ids=place_ids, region_lat=last_place.address_la, region_lng=last_place.address_lo, companions=companions, themes=themes, already_visited=visit_place_ids)
      recs = await get_next_poi_list(model=CKPT, places=PLACES, start_lat=last_place.address_la, start_lng=last_place.address_lo, companion=companions, cats=themes, required=place_ids, length=4, radius_km=5.0)
      print("recs", recs)

      for place_id in recs[len(place_ids):]:
        place = await get_place(db, place_id)
        place_schema = PlaceSchema.model_validate(place)
        itinerary.items.append(ItineraryItemResponse(item_type="place", data=ItineraryPlaceItem(
          item_id=-1,
          itinerary_id=-1,
          place_id=place_id,
          accommodation_id=None,
          start_time=generate_itinerary_request.base_itinerary.start_at + timedelta(days=day) + timedelta(hours=9),
          end_time=generate_itinerary_request.base_itinerary.start_at + timedelta(days=day) + timedelta(hours=18),
          is_required=True,
          created_at=datetime.now(),
          info=place_schema,
        )))
        visit_place_ids.append(place_id)
    except Exception as e:
      print("error", e)
      continue
  return itinerary

async def content_based_generate_itinerary(db: AsyncSession, generate_itinerary_request: ItineraryGenerate) -> ItineraryResponse:
  """콘텐츠 기반 추천을 사용한 일정 생성"""
  # 기존에 있는 장소는 유지
  itinerary = await none_generate_itinerary(db, generate_itinerary_request)
  
  # 이미 방문한 장소 ID 목록
  visit_place_ids = [x.data.place_id for x in itinerary.items if x.item_type == "place"]
  
  duration = calculate_duration(generate_itinerary_request.base_itinerary.start_at, generate_itinerary_request.base_itinerary.end_at)
  
  for day in range(duration):
    # 해당 날짜의 장소 목록
    place_ids = [x.data.place_id for x in itinerary.items if x.item_type == "place" and calculate_day_index(generate_itinerary_request.base_itinerary.start_at, x.data.start_time) == day]
    place_count = len(place_ids)
    
    # 하루에 최대 5개 장소까지
    if place_count >= 5:
      continue
    
    # 콘텐츠 기반 추천으로 장소 추천
    try:
      recommendations = await content_based_service.recommend_places(
        db=db,
        theme=generate_itinerary_request.base_itinerary.theme or "여행",
        relation=generate_itinerary_request.base_itinerary.relation or "혼자",
        location=generate_itinerary_request.base_itinerary.location,
        num_recommendations=5 - place_count,
        exclude_place_ids=visit_place_ids
      )
      
      print(f"[DEBUG] Day {day+1}: Requested {5 - place_count} recommendations, got {len(recommendations)}")
      
      # 추천된 장소들을 일정에 추가
      for rec in recommendations:
        place_id = rec['place_id']
        place = await get_place(db, place_id)
        
        if place:
          place_schema = PlaceSchema.model_validate(place)
          itinerary.items.append(ItineraryItemResponse(
            item_type="place", 
            data=ItineraryPlaceItem(
              item_id=-1,
              itinerary_id=-1,
              place_id=place_id,
              accommodation_id=None,
              start_time=generate_itinerary_request.base_itinerary.start_at + timedelta(days=day) + timedelta(hours=9),
              end_time=generate_itinerary_request.base_itinerary.start_at + timedelta(days=day) + timedelta(hours=18),
              is_required=True,
              created_at=datetime.now(),
              info=place_schema,
            )
          ))
          visit_place_ids.append(place_id)
      
    except Exception as e:
      print(f"Error in content-based recommendation for day {day}: {e}")
      continue
  
  return itinerary