from urllib.parse import urlencode
from app.auth.base import OAuthProvider
from app.core.config import settings
from typing import Dict
import httpx

class KakaoOAuth(OAuthProvider):
    CLIENT_ID = settings.kakao_client_id
    CLIENT_SECRET = settings.kakao_client_secret
    REDIRECT_URI = settings.kakao_redirect_uri
    AUTH_URL = "https://kauth.kakao.com/oauth/authorize"
    TOKEN_URL = "https://kauth.kakao.com/oauth/token"
    USERINFO_URL = "https://kapi.kakao.com/v2/user/me"

    def get_login_url(self) -> str:
        params = {
            "client_id": self.CLIENT_ID,
            "redirect_uri": self.REDIRECT_URI,
            "response_type": "code"
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    async def get_tokens(self, code: str) -> Dict:
        data = {
            "grant_type": "authorization_code",
            "client_id": self.CLIENT_ID,
            "client_secret": self.CLIENT_SECRET,
            "redirect_uri": self.REDIRECT_URI,
            "code": code
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(self.TOKEN_URL, data=data)
            r.raise_for_status()
            return r.json()

    async def fetch_user_info(self, tokens: Dict) -> Dict:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        async with httpx.AsyncClient() as client:
            r = await client.get(self.USERINFO_URL, headers=headers)
            r.raise_for_status()
            return r.json()