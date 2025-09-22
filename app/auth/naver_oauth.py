import httpx
import os
from app.core.config import settings
from fastapi import HTTPException
import secrets
import urllib.parse

NAVER_CLIENT_ID = settings.naver_client_id
NAVER_CLIENT_SECRET = settings.naver_client_secret
NAVER_REDIRECT_URI = settings.naver_redirect_uri

def get_naver_login_url():
    state = secrets.token_urlsafe(16)  # CSRF 방지용 state
    base_url = "https://nid.naver.com/oauth2.0/authorize"
    params = {
        "response_type": "code",
        "client_id": NAVER_CLIENT_ID,
        "redirect_uri": NAVER_REDIRECT_URI,
        "state": state,
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

async def get_naver_tokens(code: str, state: str):
    url = "https://nid.naver.com/oauth2.0/token"
    params = {
        "grant_type": "authorization_code",
        "client_id": NAVER_CLIENT_ID,
        "client_secret": NAVER_CLIENT_SECRET,
        "code": code,
        "state": state
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()
        if "access_token" not in data:
            raise HTTPException(status_code=400, detail=f"네이버 토큰 발급 실패: {data}")
        return data

async def get_naver_userinfo(access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://openapi.naver.com/v1/nid/me", headers=headers)
        data = resp.json()
        if data.get("resultcode") != "00":
            raise HTTPException(status_code=400, detail=f"네이버 사용자 정보 조회 실패: {data}")
        return data["response"]