from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from starlette.background import BackgroundTasks
from app.services.log_service import create_log
from app.services.session_service import get_session

class UserLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # BackgroundTasks를 사용해 비동기 로그 저장
        background = BackgroundTasks()
        
        # 세션에서 user_id 가져오기
        session_id = request.cookies.get("session_id")
        user_id = None
        if session_id:
            session = await get_session(session_id)
            if session:
                user_id = session.get("user_id")
        
        # user_id가 있으면 로그 기록, 없으면 기록하지 않음
        if user_id is not None:
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