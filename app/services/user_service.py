from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories import userdb
from app.services.activity_log_service import create_activity_log
from app.schemas.activity_log_schema import ActivityLogBase
from app.services.session_service import delete_user_sessions
from datetime import datetime, timezone

class UserService:
  @staticmethod
  async def soft_delete_user(db: AsyncSession, user_id: int) -> dict:
    # 1. 사용자 존재 여부 확인
    user = await userdb.get_user(db, user_id)
    if not user:
      return{"success": False, "message": "사용자를 찾을 수 없습니다."}

    # 2. 이미 삭제된 사용자인지 확인
    if getattr (user, "deleted_at", None) is not None:
      return{"success": False, "message": "이미 삭제된 사용자입니다."}

    # 3. soft delete 실행
    # success=await userdb.soft_delete_user(db, user_id)
    updated_user = await userdb.soft_delete_user(db, user_id)
    if updated_user is None:
      return{"success": False, "message": "사용자 삭제에 실패했습니다."}

    # 4. 세션 삭제
    await delete_user_sessions(user_id)

    # 5. 활동 로그 기록
    log = ActivityLogBase(
      user_id=user_id,
      action="user_deleted",
      metadata={"deletion_type": "soft_delete"}
    )
    await create_activity_log(log)

    # 6. 응답
    return {
      "success": True,
      "message": "탈퇴가 완료되었습니다.",
      "status": "deactive",
      "deactivated_at": updated_user.deleted_at
    }

  @staticmethod
  async def reactivate_user(db: AsyncSession, user_id: int) -> dict:
    user = await userdb.get_user(db, user_id)

    if not user:
      return {"success": False, "message": "사용자를 찾을 수 없습니다."}

    if user.status == "active":
      return {"success": True, "message": "이미 활성화된 사용자입니다."}

    # 1. 상태 변경
    user.status = "active"

    # 2. deleted_at null로 초기화
    user.deleted_at = None

    try:
      db.add(user)
      await db.commit()
      await db.refresh(user)
      
      return {"success": True, "message": "회원 상태가 active로 변경되었습니다.", "user": user}

    except Exception as e:
      await db.rollback()
      return {"success": False, "message": f"사용자 재활성화에 실패했습니다: {e}" }

