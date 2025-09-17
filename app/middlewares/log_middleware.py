from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from app.services.log_service import create_log
from fastapi import BackgroundTasks

class UserLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 요청 처리
        response = await call_next(request)

        # BackgroundTasks로 비동기 기록
        background_tasks = BackgroundTasks()
        user_id = request.headers.get("X-User-Id", "anonymous")  # JWT 등에서 추출 가능
        background_tasks.add_task(
            create_log,
            user_id=user_id,
            action=f"{request.method} {request.url.path}",
            ip=request.client.host,
            user_agent=request.headers.get("user-agent"),
            status_code=response.status_code
        )

        # response에 background_tasks 추가
        response.background = background_tasks
        return response