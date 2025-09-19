from fastapi import APIRouter, Response, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from app.auth.google_oauth import get_google_login_url, get_google_tokens, get_google_userinfo
from app.services.session_service import create_session, delete_session, get_session
from app.auth.dependencies import get_current_user
from bson.objectid import ObjectId
from app.services.activity_log_service import create_activity_log
from app.schemas.activity_log_schema import ActivityLogBase

router = APIRouter()

@router.get("/google/login")
async def google_login():
    url = get_google_login_url()
    return {"auth_url": url}

@router.get("/google/callback")
async def google_callback(request: Request, response: Response, code: str):

    # 쿠키에서 세션 가져오기
    existing_session_id = request.cookies.get("session_id")
    if existing_session_id:
        try:
            session = await get_current_user(request, response)
            if session:
                # 유효한 세션이면 바로 프론트로 리다이렉트
                return RedirectResponse(url="http://localhost:5180/userinfo")
        except HTTPException:
            # 세션 만료면 무시하고 새 로그인 진행
            pass

    from app.db.mongo import db

    try:
        tokens = await get_google_tokens(code)
        userinfo = await get_google_userinfo(tokens["access_token"])
    except Exception as e:
        raise HTTPException(status_code=400, detail="구글 로그인 실패")

    # MongoDB 사용자 저장 (없으면 생성)
    user = await db.users.find_one({"email": userinfo["email"]})
    if not user:
        new_user = {
            "email": userinfo["email"],
            "name": userinfo.get("name"),
            "picture": userinfo.get("picture"),
            "provider": "google",
        }
        result = await db.users.insert_one(new_user)
        user_id = str(result.inserted_id)
    else:
        user_id = str(user["_id"])

    # 세션 발급
    session_id = await create_session(user_id)

    log = ActivityLogBase(
        user_id=user_id,
        action="login",
        metadata={}
    )
    await create_activity_log(log)

    redirect = RedirectResponse(url="http://localhost:5180/userinfo")
    redirect.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=1800,
        samesite="lax",
        secure=False
    )
    return redirect


@router.get("/user")
async def get_auth_user(user=Depends(get_current_user)):
    from app.db.mongo import db
    user_info = await db.users.find_one({"_id": ObjectId(user)})
    print(user_info)
    user_data = {
        "id": str(user_info["_id"]),
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
        "provider": user_info.get("provider")
    }
    return {"user": user_data}

@router.post("/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id:
        
        user_id = await get_current_user(request, response)
        # DB에서 세션 삭제
        await delete_session(session_id)

        # 클라이언트 쿠키 삭제
        response.delete_cookie(
            "session_id",
            path="/",
            httponly=True,
            samesite="lax"
        )

        log = ActivityLogBase(
            user_id=user_id,
            action="logout",
            metadata={} 
        )
        await create_activity_log(log)

    return {"message": "Logged out successfully"}