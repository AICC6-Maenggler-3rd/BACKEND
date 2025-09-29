from fastapi import APIRouter, Response, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from app.services.session_service import create_session, delete_session, get_session
from app.auth.dependencies import get_current_user
from bson.objectid import ObjectId
from app.services.activity_log_service import create_activity_log
from app.schemas.activity_log_schema import ActivityLogBase
from app.auth.google import GoogleOAuth
from app.auth.naver import NaverOAuth
from app.auth.kakao import KakaoOAuth
from app.repositories import userdb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
router = APIRouter()

PROVIDERS = {
    "google": GoogleOAuth(),
    "naver": NaverOAuth(),
    "kakao": KakaoOAuth(),
}

@router.get("/{provider}/login")
async def social_login(provider: str):
    if provider not in PROVIDERS:
        return {"error": "Unsupported provider"}
    url = PROVIDERS[provider].get_login_url()
    return {"auth_url": url}


@router.get("/{provider}/callback")
async def social_callback(provider: str, request: Request, response: Response, code: str , db: AsyncSession = Depends(get_db)):
    if provider not in PROVIDERS:
        return {"error": "Unsupported provider"}
    
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

    oauth = PROVIDERS[provider]
    tokens = await oauth.get_tokens(code)
    userinfo = await oauth.fetch_user_info(tokens)

    from app.db.mongo import db as mongo

    # provider_user_id 추출
    if provider == "google":
        provider_user_id = userinfo.get("id")
        email = userinfo.get("email")
        name = userinfo.get("name")
        # picture = userinfo.get("picture")
    elif provider == "naver":
        provider_user_id = userinfo["response"].get("id")
        email = userinfo.get("email")
        name = userinfo.get("name")
        # picture = userinfo.get("profile_image")
    elif provider == "kakao":
        provider_user_id = str(userinfo.get("id"))
        kakao_account = userinfo.get("kakao_account", {})
        email = kakao_account.get("email")
        profile = kakao_account.get("profile", {})
        name = profile.get("nickname")
        # picture = profile.get("profile_image_url")

    print(userinfo)
    _user = await userdb.get_user_by_provider(db, provider, provider_user_id)

    if not _user:
        _user = await userdb.create_user(
            db=db,
            email=email,
            name=name,
            provider=provider,
            provider_user_id=provider_user_id,
            role="user"
        )
    user_id = _user.user_id
    await userdb.update_last_login(db, _user.user_id)

    session_id = await create_session(user_id)

    log = ActivityLogBase(
        user_id=user_id,
        action="login",
        metadata={}
    )
    await create_activity_log(log)

    redirect = RedirectResponse(url="http://localhost:5180/userinfo")
    redirect.set_cookie("session_id", session_id, httponly=True, max_age=1800, samesite="lax", secure=False)
    return redirect


@router.get("/user")
async def get_auth_user(user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # user_info = await db.users.find_one({"_id": ObjectId(user)})
    user_info = await userdb.get_user(db,user)
    user_data = {
        "id": str(user_info.user_id),
        "email": user_info.email,
        "name": user_info.name,
        "provider": user_info.provider
    }
    return {"user": user_data}

@router.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    users = await userdb.get_users(db)
    return {"users": users}

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