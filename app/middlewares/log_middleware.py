from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from starlette.background import BackgroundTasks
from app.services.log_service import create_log

class UserLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # BackgroundTasks를 사용해 비동기 로그 저장
        background = BackgroundTasks()
        user_id = request.headers.get("X-User-Id", "anonymous")
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