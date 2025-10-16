from fastapi import APIRouter

from app.api.endpoint import example, auth, map, place , accommodation, manage, itinerary , region


api_router = APIRouter()

api_router.include_router(
  example.router, 
  prefix="/example",
  tags=["example"]
)

api_router.include_router(
  auth.router,
  prefix='/auth',
  tags=["Auth"]
)

api_router.include_router(
  place.router,
  prefix='/place',
  tags=["place"]
)

api_router.include_router(
  map.router,
  prefix='/map',
  tags=["map"]
)

api_router.include_router(

  itinerary.router,
  prefix='/itinerary',
  tags=["itinerary"]
)
api_router.include_router(
  accommodation.router,
  prefix='/accommodation',
  tags=["accommodation"]
)

api_router.include_router(
  manage.router,
  prefix='/manage',
  tags=["manage"]
)

api_router.include_router(
  region.router,
  prefix='/region',
  tags=["region"]
)