from app.schemas.route_path_schema import MapPoint, RoutePathSchema
from app.core.config import settings
import httpx

async def find_paths(waypoints: list[MapPoint], transport: str = "CAR") -> RoutePathSchema:
    path = []
    routes = []
    distance = 0
    duration = 0
    for i in range(len(waypoints)-1):
        start_point = waypoints[i]
        end_point = waypoints[i+1]
        route = await find_path(start_point, end_point, transport)
        routes.append(route)

        path.extend(route.path)
        distance += route.distance
        duration += route.duration


    result = RoutePathSchema(
        start_point=waypoints[0],
        end_point=waypoints[-1],
        path=path,
        distance=distance,
        duration=duration,
        transport=transport
    )
    return result

async def find_path(start_point: MapPoint, end_point: MapPoint, transport: str = "CAR") -> RoutePathSchema:
    url = "https://apis.openapi.sk.com/tmap/routes?version=1&format=json&callback=result&appKey=" + settings.tmap_api_key
    pedestrian_url = "https://apis.openapi.sk.com/tmap/routes/pedestrian?version=1&format=json&callback=result&appKey=" + settings.tmap_api_key
    search_option = 0
    from app.db.mongo import db as mongo_db

    find_path_result = await mongo_db["route_paths"].find_one({"start_point": start_point.dict(), "end_point": end_point.dict(), "transport": transport})
    if find_path_result:
        return RoutePathSchema(**find_path_result)
    else:
        find_path_result = await mongo_db["route_paths"].find_one({"start_point": end_point.dict(), "end_point": start_point.dict(), "transport": transport})
        if find_path_result:
            find_path_result["path"].reverse()
            return RoutePathSchema(**find_path_result)

    payload = {
        "startX": start_point.lng,
        "startY": start_point.lat,
        "endX": end_point.lng,
        "endY": end_point.lat,
        "reqCoordType": "WGS84GEO",
        "resCoordType": "WGS84GEO",
        "searchOption": 0,
        "startName":"출발",
        "endName":"도착"
    }
    if transport == "PEDESTRIAN":
        url = pedestrian_url

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        data = response.json()

    # 경로 좌표 추출
    route_coords = []
    for feature in data.get("features", []):
        if feature["geometry"]["type"] == "LineString" and (feature['properties']==None or not feature['properties']['description'] == '경유지와 연결된 가상의 라인입니다'):
            for coord in feature["geometry"]["coordinates"] :
                route_coords.append({"lng": coord[0], "lat": coord[1]})

    result = RoutePathSchema(
        start_point=start_point,
        end_point=end_point,
        path=route_coords,
        distance=data["features"][0]["properties"]["totalDistance"],
        duration=data["features"][0]["properties"]["totalTime"],
        transport=transport
    )


    await mongo_db["route_paths"].insert_one(result.dict())

    return result

