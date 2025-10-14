from pydantic import BaseModel, Field


class MapPoint(BaseModel):
  lng: float
  lat: float

class RoutePathSchema(BaseModel):
  start_point: MapPoint
  end_point: MapPoint
  path: list[MapPoint]
  distance: float
  duration: float
  transport: str
