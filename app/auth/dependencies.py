from fastapi import Request, HTTPException, status, Response
from app.services.session_service import get_session, refresh_session

async def get_current_user(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")

    session = await refresh_user_session(request, response)
    # print("session : ",session)
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    return session

async def refresh_user_session(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return None  # 세션 없음

    session = await get_session(session_id)
    if not session:
        return None  # 만료된 세션

    # DB에서 세션 만료 시간 갱신
    new_expiry = await refresh_session(session_id)  # refresh_session 구현 필요

    # 쿠키 만료 시간 갱신
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=new_expiry,  # 새 만료 시간 (초)
        samesite="lax",
        secure=False
    )
    return session["user_id"]