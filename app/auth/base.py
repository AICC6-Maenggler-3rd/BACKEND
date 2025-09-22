from abc import ABC, abstractmethod
from typing import Dict

class OAuthProvider(ABC):
    @abstractmethod
    def get_login_url(self) -> str:
        """소셜 로그인 URL 생성"""

    @abstractmethod
    async def get_tokens(self, code: str) -> Dict:
        """인증 코드로 액세스 토큰 및 리프레시 토큰 발급"""

    @abstractmethod
    async def fetch_user_info(self, tokens: Dict) -> Dict:
        """토큰으로 사용자 정보 조회"""