from fastapi import APIRouter, Depends, HTTPException, Request,Query
from pydantic import BaseModel
import httpx
from app.core.config import settings
from app.schemas.route_path_schema import MapPoint, RoutePathSchema
from app.services.map_path_service import find_paths
router = APIRouter()

NAVER_DIRECTIONS_URL = "https://maps.apigw.ntruss.com/map-direction/v1/driving"

@router.get("/directions")
async def get_directions(
    start: str = Query(..., description="출발지 좌표: lng,lat"),
    goal: str = Query(..., description="목적지 좌표: lng,lat"),
    waypoints: str | None = Query(None, description="경유지: lng,lat|lng,lat"),
):
    NCP_CLIENT_ID = settings.ncp_client_id
    NCP_CLIENT_SECRET = settings.ncp_client_secret
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NCP_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NCP_CLIENT_SECRET,
    }

    params = {
        "start": start,
        "goal": goal,
    }
    if waypoints:
        params["waypoints"] = waypoints

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(NAVER_DIRECTIONS_URL, headers=headers, params=params)
        except httpx.RequestError as exc:
            # 요청 자체 실패 (네트워크 등)
            raise HTTPException(status_code=500, detail=f"Request error: {exc}") from exc

        if resp.status_code == 401:
            # 인증 실패
            raise HTTPException(status_code=401, detail=f"Unauthorized: {resp.text}")
        elif resp.status_code != 200:
            # 기타 오류
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Error from Naver API: {resp.text}"
            )

        # 성공
        data = resp.json()
        return data
    

class RouteRequest(BaseModel):
    startX: float
    startY: float
    endX: float
    endY: float
    viaPoints: list[dict] = []  # [{"x":126.985,"y":37.567}]
    transport: str = "CAR"  # "CAR" or "PEDESTRIAN"

    def getPassList(self) -> str:
        if self.viaPoints == None or len(self.viaPoints) == 0:
            return ""
        points = []
        for point in self.viaPoints:
            points.append(str(point['x'])+","+str(point['y']))
        return "_".join(points)



@router.post("/route")
async def get_route(req: RouteRequest):
    url = "https://apis.openapi.sk.com/tmap/routes?version=1&format=json&callback=result&appKey=" + settings.tmap_api_key
    pedestrian_url = "https://apis.openapi.sk.com/tmap/routes/pedestrian?version=1&format=json&callback=result&appKey=" + settings.tmap_api_key
    search_option = 0

    # TMAP 교통수단에 따른 옵션
        
    payload = {
        "startX": req.startX,
        "startY": req.startY,
        "endX": req.endX,
        "endY": req.endY,
        "reqCoordType": "WGS84GEO",
        "resCoordType": "WGS84GEO",
        "searchOption": 0,
        "startName":"출발",
        "endName":"도착"
    }
    if not (req.viaPoints == None or len(req.viaPoints) == 0):
        payload["passList"] = req.getPassList()
    # if req.transport.upper() == "PEDESTRIAN":
    #     search_option = 8  # 보행자 경로 옵션
    # else:
    #     search_option = 0  # 자동차 추천

    if req.transport.upper() == "PEDESTRIAN":
        url = pedestrian_url

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        data = response.json()
        

    # 경로 좌표 추출
    route_coords = []
    for feature in data.get("features", []):
        print(feature)
        if feature["geometry"]["type"] == "LineString" and (feature['properties']==None or not feature['properties']['description'] == '경유지와 연결된 가상의 라인입니다'):
            for coord in feature["geometry"]["coordinates"] :
                route_coords.append({"lng": coord[0], "lat": coord[1]})

    def remove_duplicates(coords):
        cleaned = []
        prev = None
        for c in coords:
            if prev != c:  # 이전 좌표와 같으면 skip
                cleaned.append(c)
            prev = c
        return cleaned


    print("route_coords")
    return {"route": remove_duplicates(route_coords)}


class PathRequest(BaseModel):
    waypoints: list[MapPoint]
    transport: str = "CAR"

class PathResponse(BaseModel):
    start_point: MapPoint
    end_point: MapPoint
    path: list[MapPoint]
    distance: float
    duration: float
    transport: str

@router.post("/path")
async def get_path(req: PathRequest) -> PathResponse:
    return await find_paths(req.waypoints, req.transport)
