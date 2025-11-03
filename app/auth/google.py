from urllib.parse import urlencode
from app.auth.base import OAuthProvider
from app.core.config import settings
from typing import Dict
import httpx

class GoogleOAuth(OAuthProvider):
    CLIENT_ID =  settings.google_client_id
    CLIENT_SECRET = settings.google_client_secret
    REDIRECT_URI = settings.google_redirect_uri
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"

    def get_login_url(self) -> str:
        params = {
            "client_id": self.CLIENT_ID,
            "redirect_uri": self.REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline"
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    async def get_tokens(self, code: str) -> Dict:
      async with httpx.AsyncClient(timeout=10.0, read=20.0) as client:
          resp = await client.post(self.TOKEN_URL, data={
              "code": code,
              "client_id": self.CLIENT_ID,
              "client_secret": self.CLIENT_SECRET,
              "redirect_uri": self.REDIRECT_URI,
              "grant_type": "authorization_code",
          })
          resp.raise_for_status()
          return resp.json()

    async def fetch_user_info(self, tokens: Dict) -> Dict:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        async with httpx.AsyncClient(timeout=10.0, read=20.0) as client:
            r = await client.get(self.USERINFO_URL, headers=headers)
            r.raise_for_status()
            return r.json()