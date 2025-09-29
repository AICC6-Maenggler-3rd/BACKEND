from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from starlette.background import BackgroundTasks
from app.services.log_service import create_log
from app.auth.dependencies import get_current_user
class UserLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # BackgroundTasks 초기화
        background = BackgroundTasks()

        user_id = -1 # "anonymous"
        try:
            # get_current_user를 통해 세션 확인 + 갱신
            user_id = await get_current_user(request, response)
        except Exception:
            # 로그인 안되어 있으면 anonymous 처리
            pass
        action = f"{request.method} {request.url.path}"
        ip = request.client.host
        user_agent = request.headers.get("user-agent", "unknown")

        background.add_task(
            create_log,
            user_id=user_id,
            action=action,
            ip=ip,
            user_agent=user_agent,
            status_code=response.status_code
        )

        # 원래 response에 background_tasks 추가
        response.background = background
        return response