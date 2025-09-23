from fastapi import APIRouter, Depends, HTTPException, Request,Query
import httpx
from app.core.config import settings

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