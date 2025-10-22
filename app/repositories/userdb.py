from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
# from app.models.user import User 
from app.models.postgre_model import AppUser
from datetime import datetime, timezone, timedelta

async def get_user(db: AsyncSession, user_id:int) -> AppUser:
    result = await db.execute(
        select(AppUser).where(
            AppUser.user_id == user_id 
        )
    )
    return result.scalar_one_or_none()

async def get_users(db: AsyncSession) -> list[AppUser]:
    result = await db.execute(
        select(AppUser)
    )
    return result.scalars().all()

# ✅ provider + provider_user_id로 사용자 조회
async def get_user_by_provider(db: AsyncSession, provider: str, provider_user_id: str) -> AppUser:
    result = await db.execute(
        select(AppUser).where(
            AppUser.provider == provider,
            AppUser.provider_user_id == provider_user_id
        )
    )
    return result.scalar_one_or_none()


# ✅ email로 사용자 조회
async def get_user_by_email(db: AsyncSession, email: str):
    result = await db.execute(
        select(AppUser).where(AppUser.email == email)
    )
    return result.scalar_one_or_none()


# ✅ 새로운 사용자 생성
async def create_user(db: AsyncSession, email: str, name: str,
                      provider: str, provider_user_id: str, role: str = "user") -> AppUser:
    new_user = AppUser(
        email=email,
        name=name,
        role=role,
        provider=provider,
        provider_user_id=provider_user_id,
        # status, created_at, last_login_at 은 DB default 값 적용됨
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user


# ✅ 로그인 시 마지막 로그인 시간 업데이트
async def update_last_login(db: AsyncSession, user_id: int):
    # 한국 시간 (UTC+9)으로 설정
    kst = timezone(timedelta(hours=9))
    now_kst_naive = datetime.now(kst).replace(tzinfo=None)
    await db.execute(
        update(AppUser)
        .where(AppUser.user_id == user_id)
        .values(last_login_at=now_kst_naive) 
    )
    await db.commit()

# ✅ 사용자 삭세 및 상태 업데이트
async def soft_delete_user(db: AsyncSession, user_id: int) -> AppUser | None:
    # 한국 시간 (UTC+9)으로 설정
    kst = timezone(timedelta(hours=9))
    now_kst_naive = datetime.now(kst).replace(tzinfo=None)
    stmt = (
        update(AppUser)
        .where(AppUser.user_id == user_id)
        .values(
            deleted_at=now_kst_naive,
            status="deactive",
        )
        .returning(AppUser)
    )
    result = await db.execute(stmt)

    await db.commit()

    updated_user = result.scalar_one_or_none()

    return updated_user
