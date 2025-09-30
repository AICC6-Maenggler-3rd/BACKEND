from fastapi import APIRouter
from app.api.endpoint import example, auth, map

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
  map.router,
  prefix='/map',
  tags=["map"]
)