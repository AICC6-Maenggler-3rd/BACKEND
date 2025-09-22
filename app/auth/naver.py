from urllib.parse import urlencode
from app.auth.base import OAuthProvider
from app.core.config import settings
from typing import Dict
import httpx

class NaverOAuth(OAuthProvider):
    CLIENT_ID = settings.naver_client_id
    CLIENT_SECRET = settings.naver_client_secret
    REDIRECT_URI = settings.naver_redirect_uri
    AUTH_URL = "https://nid.naver.com/oauth2.0/authorize"
    TOKEN_URL = "https://nid.naver.com/oauth2.0/token"
    USERINFO_URL = "https://openapi.naver.com/v1/nid/me"

    def get_login_url(self) -> str:
        params = {
            "response_type": "code",
            "client_id": self.CLIENT_ID,
            "redirect_uri": self.REDIRECT_URI,
            "state": "RANDOM_STATE_STRING"
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    async def get_tokens(self, code: str) -> Dict:
        params = {
            "grant_type": "authorization_code",
            "client_id": self.CLIENT_ID,
            "client_secret": self.CLIENT_SECRET,
            "code": code,
            "state": "RANDOM_STATE_STRING"
        }
        async with httpx.AsyncClient() as client:
            r = await client.get(self.TOKEN_URL, params=params)
            r.raise_for_status()
            return r.json()

    async def fetch_user_info(self, tokens: Dict) -> Dict:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        async with httpx.AsyncClient() as client:
            r = await client.get(self.USERINFO_URL, headers=headers)
            r.raise_for_status()
            return r.json().get("response", {})